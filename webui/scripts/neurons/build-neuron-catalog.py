"""Build the Neuron Lens explorer's static data from a neuron-lens catalog (Qwen3-8B or Gemma 4 12B).

    uv run --no-project --with pyarrow --with numpy --with tokenizers python -I \
        scripts/neurons/build-neuron-catalog.py <catalog dir> <tokenizer.json> <out dir> [translations.jsonl,...]

The catalog dir is neuron-lens's ``data/catalog/<model>``, whose name picks the model's profile below;
the files go to ``<out dir>/<model>/``. It holds ``neurons/<table>/L<layer>.parquet`` (a row per pole,
the + poles first) and ``heads/<table>.parquet`` (a row per head) for its tables, ``acts``,
``drivers``, ``reads``, ``legible`` and ``labels``; Gemma 4 adds ``neurons/{image,audio}/``,
``neurons/{image,audio}_optimised.parquet``, ``heads/media.parquet`` and ``media/*_docs.parquet``.
Every neuron and head is written, with everything the page would otherwise compute baked in: decoded
tokens, the hub filter, merged variants, strength percentiles, which reads to trust at each depth
(the labeller's plan, so the page shows the evidence Haiku saw), the tokens two lenses (or a lens and
a measurement) agree on, the translations from ``translate.py`` (every run's ``translations.jsonl``,
comma-separated: a term means the same whichever model's run translated it), and for Gemma 4 each
pole's image and sound evidence.

Output, all JSON:

- ``index.json``: the model, layer metadata and the curated "easiest" neurons, chosen by legibility
  and label confidence;
- ``neurons/L<layer>/index.json``: every neuron's titles, score and flags;
  ``neurons/L<layer>/c<chunk>.json``: every neuron's detail, ``CHUNK`` to a file;
- ``scores.json``: every neuron's readability as a byte, base64 per layer (the All view's texture);
- ``heads/index.json``: every head's labels, roles and pattern; ``heads/L<layer>.json``: details;
- ``search.json``: word -> curated neuron poles and heads; ``terms.json``: token -> English;
  ``vocab.json``: every token decoded, which the details name by id;
- Gemma 4: ``media/<modality>/d<block>.json``: each image's or sound's source, labels, caption (and
  its translation), link, and where ``media.py`` packed it when this site can show it, ``DOC_BLOCK``
  documents to a file, which the poles' top documents name by number. Run ``media.py`` first.

``NEURON_LAYERS=8,21`` builds only those layers (for development).
"""
import ast
import base64
import collections
import json
import multiprocessing
import os
import re
import sys
import unicodedata

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Per model: its geometry; neuron-lens's 10_labels.py reading plan by depth (the last layer of each
# step, then the reads shown, primary first); the page's depth bands (last layer, key); the layer from
# which a write's shape is read (§5); and which head-test column to use. With 16 heads a layer no
# Gemma 4 head passes a false-discovery test (findings §13), so its heads use the uncorrected p.
PROFILES = {
    "qwen3-8b": {
        "name": "Qwen3-8B", "layers": 36, "mlp": 12288, "heads": 32,
        "in": [(4, ["read_S", "read_S_z"]), (7, ["read_S_z", "read_S"]), (8, ["read_S", "read_S_z", "read_R", "read_R_z"]),
               (12, ["read_R", "read_R_z"]), (20, ["read_R_z", "read_R"]), (35, ["read_R", "read_R_z", "read_S_z"])],
        "out": [(4, ["write_R_z", "write_S_z"]), (20, ["write_R", "write_R_z", "write_S_z"]), (35, ["write_R", "write_R_z"])],
        "ov": [(8, ["ov_S", "ov_R"]), (35, ["ov_R"])],
        "ov_S_trusted": 20,
        "bands": [(0, "embed"), (7, "early"), (8, "crossover"), (12, "concept"), (20, "dark"), (31, "predict"), (35, "suppress")],
        "late": 21, "head_test": "q", "global": [], "media": [],
    },
    "gemma-4-12b": {
        "name": "Gemma 4 12B", "layers": 48, "mlp": 15360, "heads": 16,
        "in": [(4, ["read_S", "read_S_z"]), (13, ["read_S_z", "read_S"]), (32, ["read_S_z", "read_R_z"]), (47, ["read_S_z", "read_R"])],
        "out": [(4, ["write_R_z", "write_S_z"]), (13, ["write_S", "write_R", "write_R_z"]), (32, ["write_R", "write_R_z"]),
                (47, ["write_R", "write_R_z"])],
        "ov": [(13, ["ov_S", "ov_R"]), (47, ["ov_R"])],
        "ov_S_trusted": 13,
        "bands": [(4, "token"), (13, "early"), (32, "middle"), (47, "late")],
        "late": 33, "head_test": "p", "global": list(range(5, 48, 6)), "media": ["image", "audio"],
    },
}

KEEP_PER_LAYER = 40
CHUNK = 256
DOC_BLOCK = 2048
HUB = 0.03  # a raw read's token in this share of the layer's top 64s is a hub (neuron-lens §6)
Q = 0.05  # a legibility test passes at this q-value
CROSS = 0.95  # a cross-modal rank past this passes (findings §15)
NEURON_READS = ["read_S", "read_S_z", "read_R", "read_R_z", "write_S", "write_S_z", "write_R", "write_R_z"]
HEAD_READS = ["ov_R", "ov_R_z", "ov_R_neg", "ov_R_neg_z", "ov_S", "ov_S_z", "ov_S_neg", "ov_S_neg_z",
              "qk_S", "qk_S_z", "qk_S_neg", "qk_S_neg_z", "qk_self_S", "qk_self_S_z"]
NEURON_SIDES = ("read_S", "read_R", "write_S", "write_R")
# Head roles, at the catalog's own thresholds (09_legible.py's pattern, checks/head_lens_sides.py), and
# Gemma 4's in images (checks/head_media.py, findings §16).
(ROLE_INDUCTION, ROLE_DUPLICATE, ROLE_PREVIOUS, ROLE_SELF, ROLE_SINK, ROLE_NEXT, ROLE_COPY, ROLE_DARK,
 ROLE_CARRIES, ROLE_NAMES) = (1 << k for k in range(10))
# Neuron flags.
(FLAG_DETECTOR, FLAG_PREDICTS, FLAG_SUPPRESSES, FLAG_PLUS_WRITES, FLAG_MINUS_WRITES, FLAG_DARK,
 FLAG_IMAGE, FLAG_AUDIO, FLAG_CROSSMODAL) = (1 << k for k in range(9))
STOP = set("""a an the and or of to in on for with by as at from into its it this that these those is are be
words word tokens token terms term pieces piece fragments fragment related like such mixed e.g eg i.e
weak clear unclear no not theme themes output input mostly mainly slight slightly some other various
fires writes promotes pushes suppresses attends copies when after before about""".split())

catalog, tokjson, out_root = sys.argv[1:4]
translation_paths = [x for x in (sys.argv[4] if len(sys.argv) > 4 else "").split(",") if x]
MODEL = os.path.basename(os.path.normpath(catalog))
P = PROFILES[MODEL]
LAYERS, MLP, HEADS = P["layers"], P["mlp"], P["heads"]
out = f"{out_root}/{MODEL}"
ONLY = sorted({int(x) for x in os.environ.get("NEURON_LAYERS", "").split(",") if x}) or list(range(LAYERS))
LABELLED = os.path.exists(f"{catalog}/heads/labels.parquet")

tok = Tokenizer.from_file(tokjson)
VOCAB = tok.get_vocab_size()


def decode(i):
    s = tok.decode([i])
    if "�" in s or s == "":
        return "\u0001" + (tok.id_to_token(i) or f"<{i}>")  # byte piece or special token: shown raw and marked
    return s


TOKS = [decode(i) for i in range(VOCAB)]


def stem(t):
    """Variant key: case, leading punctuation and common English suffixes removed (release, Released, -releases)."""
    if t.startswith("\u0001"):
        return t
    k = t.strip().lower().lstrip("_-.").lstrip("/([")
    for suf in ("ing", "ed", "es", "s", "ly"):
        if k.endswith(suf) and len(k) - len(suf) >= 4:
            k = k[: -len(suf)]
            break
    return k[:-1] if k.endswith("e") and len(k) > 4 else k


def junk_text(s):
    s = s.strip()
    return not s or all(unicodedata.category(c)[0] in "PSZC" for c in s)


STEM = [stem(t) for t in TOKS]
JUNK = np.array([t.startswith("\u0001") or junk_text(t) for t in TOKS])

# ---------- translations (translate.py): vocabulary terms, whole contexts and media captions
TERMS, CONTEXT_EN, CAPTION_EN = {}, {}, {}
for translations_path in translation_paths:
    if not os.path.exists(translations_path):
        continue
    for line in open(translations_path, encoding="utf-8"):
        r = json.loads(line)
        # Drop answers that add nothing: English after all, the source echoed back, or (for a term) a
        # note such as "(fragment)" in place of a meaning.
        if r.get("lang") == "en" or r["en"].casefold() == r["s"].strip().casefold():
            continue
        if r["k"] == "term":
            if not r["en"].startswith("("):
                TERMS[r["s"].strip()] = [r["en"], r.get("ro", ""), r.get("lang", "")]
        elif r["k"] == "caption":
            CAPTION_EN[r["s"]] = [r["en"], r.get("lang", "")]
        elif r.get("ok"):
            CONTEXT_EN[r["s"]] = [r["en"], r.get("lang", "")]


# ---------- shared helpers

def by_depth(steps, L):
    return next(v for last, v in steps if L <= last)


def neuron_plan(L):
    """(input reads, output reads) shown at this depth, primary first: 10_labels.py's plan."""
    return by_depth(P["in"], L), by_depth(P["out"], L)


def head_header(L):
    """A head file's reads: the plan's OV reads, then QK, then the other OV family (weak past the
    model's early layers)."""
    ov = by_depth(P["ov"], L)
    primary = {ov[0], "qk_S"}
    trusted = (set(ov) | {f"{c}_neg" for c in ov} | {"qk_S_neg", "qk_self_S"}
               | ({"ov_S", "ov_S_neg"} if L <= P["ov_S_trusted"] else set()))
    fams = [f for f in ("ov_R", "ov_R_neg", "ov_S", "ov_S_neg") if f.split("_neg")[0] in ov]
    fams += ["qk_S", "qk_S_neg", "qk_self_S"] + [f for f in ("ov_R", "ov_R_neg", "ov_S", "ov_S_neg") if f not in fams]
    return read_header([c for f in fams for c in (f, f"{f}_z") if c in HEAD_READS], primary, trusted, by_family=True)


def band(L):
    return by_depth(P["bands"], L)


def mat(table, name, dtype=np.float32):
    """A fixed-size list column as an [n, k] array."""
    col = table.column(name).combine_chunks()
    return col.flatten().to_numpy(zero_copy_only=False).astype(dtype).reshape(len(col), -1)


def col(table, name):
    return table.column(name).to_numpy(zero_copy_only=False)


def merged(ids, skip=frozenset(), limit=24):
    """Distinct tokens in rank order, variants folded into their first occurrence: [(token id, count, key)]."""
    order, count, first = [], collections.Counter(), {}
    for i in ids:
        k = STEM[i] or TOKS[i]
        if k in skip:
            continue
        if k not in count:
            if len(order) >= limit:
                break
            first[k] = i
            order.append(k)
        count[k] += 1
    return [(first[k], count[k], k) for k in order]


def encoded(toks, shared=frozenset(), measured=frozenset()):
    """A token list for the page, by vocab.json id: a bare id, or [id, variants, flags] (1: lenses agree, 2: measured)."""
    out = []
    for t, n, k in toks:
        flags = (1 if k in shared else 0) | (2 if k in measured else 0)
        out.append(int(t) if n == 1 and not flags else [int(t), n, flags])
    return out


def drivers_of(ids, vals, n=12):
    return [[int(t), float(f"{float(v):.3g}")] for t, v in zip(ids[:n], vals[:n])]


LIMIT = {"primary": 24, "trusted": 12, "weak": 8}


PEAK = re.compile(r"^('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")×(\d+)$", re.S)


def parse_counts(s, limit=8):
    """"'tok'×3 | 'tok'×2" -> [[tok, 3], ...] (tokens are Python reprs)."""
    out = []
    for part in (s or "").split(" | "):
        m = PEAK.match(part.strip(" "))
        if m:
            try:
                out.append([ast.literal_eval(m.group(1)), int(m.group(2))])
            except (ValueError, SyntaxError):
                continue
        if len(out) >= limit:
            break
    return out


def contexts_of(s):
    """Top contexts as [text] or [text, english, lang]; the text keeps the ⟦ ⟧ and ⟪ ⟫ markers."""
    out = []
    for c in (s or "").split(" ‖ "):
        if not c:
            continue
        en = CONTEXT_EN.get(c)
        out.append([c, *en] if en else [c])
    return out


def shorten(text, n=84):
    text = " ".join((text or "").split())
    if len(text) <= n:
        return text
    return text[:n].rsplit(" ", 1)[0].rstrip(",;:—-(") + "…"


def label_words(text):
    return [w for w in re.findall(r"[^\W\d_][\w'-]*", (text or "").lower()) if len(w) > 2 and w not in STOP]


def hubs_of(ids):
    """Raw reads' hub tokens: in HUB or more of the rows' top lists."""
    return np.bincount(ids.ravel(), minlength=VOCAB) / len(ids) >= HUB


def read_header(order, primary, trusted, by_family=False):
    """The reads every unit of a file lists, in order: [{col, trust, twin}]. A twin (the raw or z read
    of a family shown above it) lists only the tokens its family hasn't. Neurons trust single
    columns; heads trust a family (a read and its z twin) at once."""
    seen, out = set(), []
    for c in order:
        fam = c.removesuffix("_z")
        key = fam if by_family else c
        trust = "primary" if key in primary else "trusted" if key in trusted else "weak"
        out.append({"col": c, "trust": trust, "twin": fam in seen})
        seen.add(fam)
    return out


def search_weights(header, reads, peaks, label_texts):
    """Word -> weight for one pole or head: label words, trusted read tokens, peaks, and translations."""
    weights, display = collections.Counter(), {}

    def put(k, w, shown):
        if k and weights[k] < w:
            weights[k] = w
            display.setdefault(k, shown)

    for h, (_, tokens) in zip(header, reads):
        if h["trust"] not in ("primary", "trusted"):
            continue
        for rank, t in enumerate(tokens[:24]):
            text, flags = (TOKS[t], 0) if isinstance(t, int) else (TOKS[t[0]], t[2])
            if text.startswith("\u0001") or junk_text(text):
                continue
            w = (3 if rank < 8 else 2 if rank < 24 else 1) + (2 if flags & 1 else 0) + (2 if flags & 2 else 0)
            put(stem(text), w, text.strip())
            en = TERMS.get(text.strip())
            if en:
                for word in label_words(en[0]):
                    put(stem(word), max(1, w - 2), word)
    for t, _ in peaks:
        if not junk_text(t):
            put(stem(t), 4, t.strip())
    for text in label_texts:
        for word in label_words(text):
            put(stem(word), 5, word)
    return weights, display


def table_or_none(path, columns=None):
    return pq.read_table(path, columns=columns) if os.path.exists(path) else None


# ---------- Gemma 4's images and sounds

MEDIA_COLS = ["label", "label_z", "labels_str", "sig", "stable", "auroc_heldout", "top_docs", "top_vals",
              "text_fire_rank", "text_read_rank", "text_write_rank", "write_R_ids", "write_R_z_ids"]


def num(v, digits=3):
    v = float(v)
    return None if v != v else float(f"{v:.{digits}g}")


def optimised(modality):
    """(layer, neuron, pole) -> the optimised read (15_media_optimised.py) and its held-out gain:
    an image patch as a PNG data URL, or a 40 ms frame as 640 int16 samples, base64, peak-normalised."""
    t = table_or_none(f"{catalog}/neurons/{modality}_optimised.parquet")
    if t is None:
        return {}
    out = {}
    for r in t.to_pylist():
        o = {"gain": num(r["gain"]), "lens": num(r["gain_lens"]), "base": num(r["act_base"])}
        if modality == "image":
            o["png"] = "data:image/png;base64," + base64.b64encode(r["opt_png"]["bytes"]).decode()
        else:
            wave = np.asarray(r["opt_wave"], np.float32)
            wave = wave / max(float(np.abs(wave).max()), 1e-6)
            o["wave"] = base64.b64encode(np.round(wave * 32767).astype("<i2").tobytes()).decode()
        out[r["layer"], r["neuron"], r["pole"]] = o
    return out


def media_layer(L, modality, opt):
    """Per row of the layer: the pole's media evidence (findings §14, §15), or None."""
    t = table_or_none(f"{catalog}/neurons/{modality}/L{L:02d}.parquet", ["neuron", "pole", *MEDIA_COLS])
    if t is None:
        return None
    rows = t.to_pylist()
    w_ids, wz_ids = mat(t, "write_R_ids", np.int32), mat(t, "write_R_z_ids", np.int32)
    hub = hubs_of(w_ids)
    out = []
    for i, r in enumerate(rows):
        ranks = [num(r[f"text_{k}_rank"]) for k in ("fire", "read", "write")]
        raw = merged([int(x) for x in w_ids[i] if not hub[x]], limit=16)
        # label_z is signed like the activation; the page shows it toward this pole.
        m = {"label": r["label"], "z": num(r["label_z"] * r["pole"]), "auroc": num(r["auroc_heldout"]),
             "reliable": bool(r["sig"] and r["stable"]),
             "labels": [x.strip() for x in (r["labels_str"] or "").split(" | ") if x.strip()][:6],
             "top": [[int(d), num(v)] for d, v in zip(r["top_docs"][:8], r["top_vals"][:8])],
             "write": encoded(raw), "writeZ": encoded(merged([int(x) for x in wz_ids[i]], skip=frozenset(k for _, _, k in raw), limit=12)),
             "ranks": ranks if any(x is not None for x in ranks) else None}
        o = opt.get((L, r["neuron"], r["pole"]))
        if o:
            m["opt"] = o
        out.append(m)
    return out


def crossmodal(m):
    return bool(m and m["reliable"] and m["ranks"] and all(x is not None and x >= CROSS for x in m["ranks"]))


# ---------- neurons, one layer per process

OPTIMISED = {m: optimised(m) for m in P["media"]}


def neuron_layer(L):
    ins, outs = neuron_plan(L)
    trusted = ins + outs
    path = lambda t: f"{catalog}/neurons/{t}/L{L:02d}.parquet"
    if not os.path.exists(path("reads")):
        print(f"L{L:02d}: no reads, skipped", flush=True)
        return None
    cols = []
    for c in NEURON_READS:
        cols += [f"{c}_ids", f"{c}_val"]
    reads = pq.read_table(path("reads"), columns=["neuron", "pole", "vocab_skew", "vocab_kurt", *cols])
    acts = pq.read_table(path("acts"), columns=["neuron", "pole", "act_top", "act_max", "rate", "rate_dclm", "rate_zh",
                                                 "rate_code", "rate_chat", "pos_share", "peak_tokens", "contexts"])
    drivers = pq.read_table(path("drivers"), columns=["neuron", "pole", "drivers_ids", "drivers_val"])
    legible = pq.read_table(path("legible"))
    labels = table_or_none(path("labels"))
    n = reads.num_rows
    neuron, pole = col(reads, "neuron").astype(int), col(reads, "pole").astype(int)
    for t in (acts, drivers, legible, labels):  # the tables mirror one another, row for row
        assert t is None or ((col(t, "neuron") == neuron).all() and (col(t, "pole") == pole).all()), f"L{L}: tables out of order"
    media = {m: media_layer(L, m, OPTIMISED[m]) for m in P["media"]}

    ids = {c: mat(reads, f"{c}_ids", np.int32) for c in NEURON_READS}
    vals = {c: mat(reads, f"{c}_val") for c in NEURON_READS}
    strength = {c: np.sort(vals[c][:, :8].mean(1)) for c in NEURON_READS}
    pct = {c: np.searchsorted(strength[c], vals[c][:, :8].mean(1)) * 100 // n for c in NEURON_READS}
    hubs = {c: (np.zeros(VOCAB, bool) if c.endswith("_z") else hubs_of(ids[c])) for c in NEURON_READS}
    kurt = col(reads, "vocab_kurt")
    kurt_pct = np.searchsorted(np.sort(kurt), kurt) * 100 // n
    skew = col(reads, "vocab_skew")

    silent, dark = col(legible, "silent").astype(bool), col(legible, "dark").astype(bool)
    cur_mode = np.nan_to_num(col(legible, "cur_mode").astype(float))
    next_mode = np.nan_to_num(col(legible, "next_mode").astype(float))
    q = {s: np.nan_to_num(col(legible, f"{s}_q").astype(float), nan=1.0) for s in NEURON_SIDES}
    passed = sum((q[s] <= Q).astype(int) for s in NEURON_SIDES)
    detector = (cur_mode >= 1 / 3) | (next_mode >= 1 / 3)
    if labels is not None:
        fires, writes, note = (labels.column(c).to_pylist() for c in ("fires", "writes", "note"))
        fc = np.nan_to_num(col(labels, "fires_conf").astype(float))
        wc = np.nan_to_num(col(labels, "writes_conf").astype(float))
    else:
        fires, writes, note = [""] * n, [""] * n, [""] * n
        fc, wc = np.zeros(n), np.zeros(n)
    # A reliable image or sound label is a read that holds too, weighted by its held-out AUROC.
    media_read = np.zeros(n)
    for rows in media.values():
        if rows:
            media_read = np.maximum(media_read, [float(m["auroc"] or 0) if m["reliable"] else 0.0 for m in rows])

    # Easiest to read: tests that hold at the pole's own top firings, then Haiku's confidence.
    pole_score = np.where(silent, -1.0, 1.2 * np.minimum(passed, 3) + 3.0 * np.maximum(fc, wc) + 1.0 * np.minimum(fc, wc)
                          + 0.8 * detector + 1.0 * media_read)
    plus, minus = pole_score[:MLP], pole_score[MLP:]
    neuron_score = np.maximum(plus, minus) + 0.25 * np.minimum(plus, minus)

    act_rows = acts.select(["act_top", "act_max", "rate", "rate_dclm", "rate_zh", "rate_code", "rate_chat", "pos_share",
                            "peak_tokens", "contexts"]).to_pylist()
    drv_ids, drv_val = mat(drivers, "drivers_ids", np.int32), mat(drivers, "drivers_val")

    def measured_stems(i, peaks):
        s = {STEM[t] for t in drv_ids[i][:20] if not JUNK[t]} | {stem(t) for t, _ in peaks}
        s.discard("")
        return s

    header = read_header(trusted + [c for c in NEURON_READS if c not in trusted], {ins[0], outs[0]}, set(trusted))

    def pole_detail(i):
        a = act_rows[i]
        peaks = parse_counts(a["peak_tokens"], 6)
        meas = measured_stems(i, peaks)
        rd, fam_keys, twins = [], collections.defaultdict(set), {}
        for h in header:
            c, fam = h["col"], h["col"].removesuffix("_z")
            toks = merged([int(t) for t in ids[c][i] if not hubs[c][t]], skip=frozenset(twins.get(fam, ())), limit=LIMIT[h["trust"]])
            twins.setdefault(fam, set()).update(k for _, _, k in toks)
            if h["trust"] != "weak":
                fam_keys[fam] |= {k for _, _, k in toks[:24] if k and not k.startswith("\u0001")}
            rd.append((int(pct[c][i]), toks))
        fams = list(fam_keys.values())
        shared = set().union(*[x & y for j, x in enumerate(fams) for y in fams[j + 1:]]) if len(fams) > 1 else set()
        every = set().union(*fams) if fams else set()
        p = {"pole": int(pole[i]), "reads": [[v, encoded(toks, shared, meas)] for v, toks in rd],
             "shared": sorted(shared)[:12], "measured": sorted(meas & every)[:12],
             "label": {"fires": fires[i], "fc": round(float(fc[i]), 2), "writes": writes[i], "wc": round(float(wc[i]), 2), "note": note[i]},
             "act": {"top": round(a["act_top"], 3), "max": round(a["act_max"], 3), "rate": float(f"{a['rate']:.3g}"),
                     "slices": [float(f"{a[k]:.3g}") for k in ("rate_dclm", "rate_zh", "rate_code", "rate_chat")],
                     "pos": round(a["pos_share"], 3)},
             "peaks": peaks, "contexts": contexts_of(a["contexts"]),
             "drivers": drivers_of(drv_ids[i], drv_val[i]),
             "legible": {s: round(float(q[s][i]), 3) for s in NEURON_SIDES},
             "silent": bool(silent[i]), "dark": bool(dark[i]),
             "detector": "token" if cur_mode[i] >= 1 / 3 else "next" if next_mode[i] >= 1 / 3 else None,
             "score": round(float(max(pole_score[i], 0)), 2)}
        if L >= P["late"]:
            p["shape"] = {"kurtPct": int(kurt_pct[i]), "skew": round(float(skew[i]), 3),
                          "kind": ("predicts" if skew[i] > 0 else "suppresses") if kurt_pct[i] >= 90 else None}
        if media:
            p["media"] = {m: rows[i] for m, rows in media.items() if rows}
        return p

    def title(i):
        """The pole's clearer label, shortened; True when it is the writes label. Unlabelled, its reliable
        media label or its peak tokens stand in."""
        if labels is None:
            for m, rows in media.items():
                if rows and rows[i]["reliable"]:
                    return f"{m}: {rows[i]['label']}", False, 0.0
            return ", ".join(t.strip() for t, _ in parse_counts(act_rows[i]["peak_tokens"], 4)), False, 0.0
        return (shorten(writes[i]), True, wc[i]) if wc[i] > fc[i] else (shorten(fires[i]), False, fc[i])

    def media_flags(poles):
        flags = 0
        for p in poles:
            got = p.get("media") or {}
            flags |= ((FLAG_IMAGE if (got.get("image") or {}).get("reliable") else 0)
                      | (FLAG_AUDIO if (got.get("audio") or {}).get("reliable") else 0)
                      | (FLAG_CROSSMODAL if any(crossmodal(x) for x in got.values()) else 0))
        return flags

    d_out = f"{out}/neurons/L{L:02d}"
    os.makedirs(d_out, exist_ok=True)
    rows, details = [], []
    for nrn in range(MLP):
        i_plus, i_minus = nrn, MLP + nrn
        poles = [pole_detail(i_plus), pole_detail(i_minus)]
        details.append({"n": nrn, "poles": poles})
        (tp, wp, cp), (tm, wm, cm) = title(i_plus), title(i_minus)
        flags = ((FLAG_DETECTOR if any(p["detector"] for p in poles) else 0)
                 | (FLAG_PREDICTS if any((p.get("shape") or {}).get("kind") == "predicts" for p in poles) else 0)
                 | (FLAG_SUPPRESSES if any((p.get("shape") or {}).get("kind") == "suppresses" for p in poles) else 0)
                 | (FLAG_PLUS_WRITES if wp else 0) | (FLAG_MINUS_WRITES if wm else 0)
                 | (FLAG_DARK if all(p["dark"] or p["silent"] for p in poles) else 0)
                 | media_flags(poles))
        rows.append([nrn, tp, tm, round(float(max(neuron_score[nrn], 0)), 1), flags, int(round(cp * 100)), int(round(cm * 100))])
        if len(details) == CHUNK or nrn == MLP - 1:
            json.dump({"layer": L, "reads": header, "neurons": details}, open(f"{d_out}/c{nrn // CHUNK:03d}.json", "w"), ensure_ascii=False, separators=(",", ":"))
            details = []
    json.dump({"layer": L, "neurons": rows}, open(f"{d_out}/index.json", "w"), ensure_ascii=False, separators=(",", ":"))

    chosen = sorted(range(MLP), key=lambda k: -neuron_score[k])[:KEEP_PER_LAYER]
    curated = []
    for nrn in chosen:
        searches = []
        for side, i in enumerate((nrn, MLP + nrn)):
            p = pole_detail(i)
            media_labels = [m["label"] for m in (p.get("media") or {}).values() if m["reliable"]]
            searches.append(search_weights(header, p["reads"], p["peaks"], [fires[i], writes[i], *media_labels]))
        curated.append({"row": [L, *rows[nrn]], "search": searches})
    meta = {"layer": L, "band": band(L), "count": len(chosen),
            "stats": {"silent": round(float(silent.mean()), 4), "detector": round(float((detector & ~silent).mean()), 4),
                      "legible": round(float(((passed > 0) & ~silent).mean()), 4), "dark": round(float(dark.mean()), 4)}}
    for m, rows_m in media.items():
        if rows_m:
            meta["stats"][m] = round(float(np.mean([x["reliable"] for x in rows_m])), 4)
    if L in P["global"]:
        meta["global"] = True
    print(f"L{L:02d}: {MLP} neurons, curated {len(chosen)}, silent {meta['stats']['silent']:.2f} dark {meta['stats']['dark']:.2f}", flush=True)
    return {"meta": meta, "curated": curated, "scores": neuron_score.astype(np.float32)}


# ---------- heads

def heads():
    names = ["acts", "drivers", "reads", "legible"] + (["labels"] if LABELLED else [])
    t = {name: pq.read_table(f"{catalog}/heads/{name}.parquet") for name in names}
    media_t = table_or_none(f"{catalog}/heads/media.parquet")
    layer, head = col(t["reads"], "layer").astype(int), col(t["reads"], "head").astype(int)
    for x in [*t.values(), *([media_t] if media_t is not None else [])]:
        assert (col(x, "layer") == layer).all() and (col(x, "head") == head).all(), "head tables out of order"
    n = len(layer)
    ids = {c: mat(t["reads"], f"{c}_ids", np.int32) for c in HEAD_READS}
    vals = {c: mat(t["reads"], f"{c}_val") for c in HEAD_READS}
    pct = {c: np.searchsorted(np.sort(vals[c][:, :8].mean(1)), vals[c][:, :8].mean(1)) * 100 // n for c in HEAD_READS}
    hubs = {c: (np.zeros(VOCAB, bool) if c.endswith("_z") else hubs_of(ids[c])) for c in HEAD_READS}
    a, d, r, g = (t[k].to_pylist() for k in ("acts", "drivers", "reads", "legible"))
    lab = t["labels"].to_pylist() if LABELLED else [None] * n
    med = media_t.to_pylist() if media_t is not None else [None] * n
    out_rz = mat(t["acts"], "out_R_z_ids", np.int32)
    drv = {k: (mat(t["drivers"], f"{k}_ids", np.int32), mat(t["drivers"], f"{k}_val")) for k in ("drivers", "keys")}
    test = P["head_test"]

    def q(row, k):
        v = row[k]
        return 1.0 if v is None or v != v else float(v)

    def head_media(M):
        """In image and audio contexts (checks/head_media.py): reading the media against an even spread,
        attention within it and to grid neighbours, the content-carrier z, and the label-writer score."""
        if M is None:
            return None
        return {m: {"read": num(M[f"{m}_read_ratio"]), "within": num(M[f"{m}_within"]), "nbr": num(M[f"{m}_nbr"]),
                    "content": num(M[f"{m}_content_z"]), "write": num(M[f"{m}_write_mid"]), "writeQ": num(M[f"{m}_write_mid_q"])}
                for m in P["media"]}

    rows, details, search = [], collections.defaultdict(list), []
    for i in range(n):
        L, h = int(layer[i]), int(head[i])
        A, G, R, B, M = a[i], g[i], r[i], lab[i], med[i]
        header = head_header(L)
        rd, twins = [], {}
        for hd in header:
            c, fam = hd["col"], hd["col"].removesuffix("_z")
            toks = merged([int(x) for x in ids[c][i] if not hubs[c][x]], skip=frozenset(twins.get(fam, ())), limit=LIMIT[hd["trust"]])
            twins.setdefault(fam, set()).update(k for _, _, k in toks)
            rd.append([int(pct[c][i]), encoded(toks)])
        hm = head_media(M)
        img = (hm or {}).get("image") or {}
        roles = ((ROLE_INDUCTION if A["induction"] > 0.3 else 0) | (ROLE_DUPLICATE if A["duplicate"] > 0.3 else 0)
                 | (ROLE_PREVIOUS if A["prev"] > 0.5 else 0) | (ROLE_SELF if A["self"] > 0.5 else 0)
                 | (ROLE_SINK if A["sink"] > 0.9 else 0) | (ROLE_NEXT if q(G, f"ov_R_{test}") <= Q else 0)
                 | (ROLE_COPY if q(G, f"ov_copy_{test}") <= Q else 0) | (ROLE_DARK if G["dark"] else 0)
                 | (ROLE_CARRIES if (img.get("content") or 0) > 3 else 0)
                 | (ROLE_NAMES if img.get("writeQ") is not None and img["writeQ"] <= Q else 0))
        B = B or {"fires": "", "fires_conf": 0, "attends": "", "attends_conf": 0, "writes": "", "writes_conf": 0, "note": ""}
        conf = [float(B["fires_conf"] or 0), float(B["attends_conf"] or 0), float(B["writes_conf"] or 0)]
        modes = [float(G["cur_mode"] or 0), float(G["src_mode"] or 0)]
        score = (2.0 * (q(G, f"ov_R_{test}") <= Q) + 1.5 * (q(G, f"ov_copy_{test}") <= Q) + 1.0 * bool(G["pattern"])
                 + 0.8 * (max(modes) >= 1 / 3) + 3.0 * max(conf) + 1.0 * sum(conf) / 3)
        pattern = [round(A[k], 3) for k in ("sink", "prev", "self", "induction", "duplicate")]
        peaks = parse_counts(A["peak_tokens"], 6)
        detail = {
            "h": h, "kv": int(A["kv_head"]),
            "label": {"fires": B["fires"], "fc": round(conf[0], 2), "attends": B["attends"], "ac": round(conf[1], 2),
                      "writes": B["writes"], "wc": round(conf[2], 2), "note": B["note"]},
            "pattern": pattern, "roles": roles,
            "act": {"top": round(A["out_top"], 2), "max": round(A["out_max"], 2), "rate": float(f"{A['rate']:.3g}"),
                    "slices": [float(f"{A[k]:.3g}") for k in ("rate_dclm", "rate_zh", "rate_code", "rate_chat")]},
            "copy": {"ovR": round(R["ov_R_copy"], 3), "ovS": round(R["ov_S_copy"], 3), "self": round(R["qk_self"], 3),
                     "qkOv": round(R["qk_ov_rho"], 3), "qkCtx": round(R["qk_ctx_rho"], 3)},
            "reads": rd,
            "outRz": encoded(merged([int(x) for x in out_rz[i]], limit=16)),
            "drivers": drivers_of(drv["drivers"][0][i], drv["drivers"][1][i]),
            "keys": drivers_of(drv["keys"][0][i], drv["keys"][1][i]),
            "peaks": peaks, "sources": parse_counts(A["src_tokens"], 6), "contexts": contexts_of(A["contexts"]),
            "legible": {k: round(q(G, f"{k}_{test}"), 3) for k in ("ov_R", "ov_copy", "ov_S", "qk_S")},
            "modes": [round(m, 3) for m in modes], "dark": bool(G["dark"]), "score": round(score, 2)}
        if hm:
            detail["media"] = hm
        details[L].append(detail)
        rows.append([L, h, int(A["kv_head"]), shorten(B["fires"], 72), shorten(B["attends"], 72), shorten(B["writes"], 72),
                     *[int(round(c * 100)) for c in conf], roles, round(score, 1), pattern])
        search.append(search_weights(header, rd, peaks, [B["fires"], B["attends"], B["writes"]]))
    os.makedirs(f"{out}/heads", exist_ok=True)
    for L, hs in details.items():
        json.dump({"layer": L, "reads": head_header(L), "heads": hs}, open(f"{out}/heads/L{L:02d}.json", "w"), ensure_ascii=False, separators=(",", ":"))
    json.dump({"heads": rows}, open(f"{out}/heads/index.json", "w"), ensure_ascii=False, separators=(",", ":"))
    print(f"heads: {n}", flush=True)
    return rows, search


# ---------- Gemma 4's media documents

def media_docs():
    """Each image's or sound's source, labels (without the source prefix), caption and link, in blocks of
    DOC_BLOCK: {s, l, c, u}, with en: [English, language] for a translated caption and h: [pack, offset,
    length] for one media.py packed. AudioSet links go to the clip's place in its video."""
    for m in P["media"]:
        t = table_or_none(f"{catalog}/media/{m}_docs.parquet")
        if t is None:
            continue
        hosted_path = f"{out}/media/{m}/hosted.json"
        hosted = {int(k): v for k, v in json.load(open(hosted_path)).items()} if os.path.exists(hosted_path) else {}
        rows = sorted(t.to_pylist(), key=lambda r: r["doc"])
        os.makedirs(f"{out}/media/{m}", exist_ok=True)
        for b in range(0, len(rows), DOC_BLOCK):
            block = []
            for r in rows[b:b + DOC_BLOCK]:
                labels = [x.split(":", 1)[-1].strip() for x in (r["labels"] or "").split(" | ") if x.strip()]
                link = r["link"] if isinstance(r["link"], str) else ""
                ref = json.loads(r["ref"]) if r["ref"] else {}
                if r["source"] == "audioset" and ref.get("video_id"):
                    start = ref.get("start_seconds") or ref.get("start") or 0
                    link = f"https://www.youtube.com/watch?v={ref['video_id']}" + (f"&t={int(float(start))}s" if start else "")
                caption = r["caption"] if isinstance(r["caption"], str) else ""
                row = {"s": r["source"], "l": labels[:4], "c": shorten(caption, 160), "u": link}
                if caption in CAPTION_EN:
                    row["en"] = [shorten(CAPTION_EN[caption][0], 160), CAPTION_EN[caption][1]]
                if r["doc"] in hosted:
                    row["h"] = hosted[r["doc"]]
                block.append(row)
            json.dump({"first": rows[b]["doc"], "docs": block}, open(f"{out}/media/{m}/d{b // DOC_BLOCK:03d}.json", "w"),
                      ensure_ascii=False, separators=(",", ":"))
        print(f"media docs, {m}: {len(rows)}, shown {len(hosted)}", flush=True)


def main():
    os.makedirs(out, exist_ok=True)
    head_rows, head_search = heads()
    media_docs()
    with multiprocessing.get_context("fork").Pool(min(6, len(ONLY))) as pool:
        layers = dict(zip(ONLY, pool.map(neuron_layer, ONLY)))

    metas, curated, scores = [], [], []
    for L in range(LAYERS):
        res = layers.get(L)
        metas.append(res["meta"] if res else {"layer": L, "band": band(L), "missing": True, **({"global": True} if L in P["global"] else {})})
        if res:
            curated += res["curated"]
    top = max(float(res["scores"].max()) for res in layers.values() if res)
    for L in range(LAYERS):
        res = layers.get(L)
        values = np.zeros(MLP, np.uint8) if not res else np.clip(np.round(255 * (np.maximum(res["scores"], 0) / top) ** 0.8), 0, 255).astype(np.uint8)
        scores.append(base64.b64encode(values.tobytes()).decode())
    json.dump({"layers": scores}, open(f"{out}/scores.json", "w"), separators=(",", ":"))

    # Search: word -> [ref, side, weight]; side 0/1 is a curated neuron's + or − pole (ref indexes
    # index.json's neurons), side 2 a head (ref = layer * HEADS + head).
    forms, display = collections.defaultdict(list), {}
    for ref, c in enumerate(curated):
        for side, (weights, shown) in enumerate(c["search"]):
            for k, w in weights.most_common(40):
                forms[k].append([ref, side, w])
                display.setdefault(k, shown[k])
    for row, (weights, shown) in zip(head_rows, head_search):
        for k, w in weights.most_common(40):
            forms[k].append([row[0] * HEADS + row[1], 2, w])
            display.setdefault(k, shown[k])
    json.dump({"forms": forms, "display": {k: display[k] for k in forms}}, open(f"{out}/search.json", "w"), ensure_ascii=False, separators=(",", ":"))
    json.dump(TERMS, open(f"{out}/terms.json", "w"), ensure_ascii=False, separators=(",", ":"))
    json.dump(TOKS, open(f"{out}/vocab.json", "w"), ensure_ascii=False, separators=(",", ":"))
    json.dump({"model": P["name"], "id": MODEL, "layers": LAYERS, "neuronsPerLayer": MLP, "headsPerLayer": HEADS, "chunk": CHUNK,
               "docBlock": DOC_BLOCK, "labelled": LABELLED, "media": P["media"],
               "layerMeta": metas, "neurons": [c["row"] for c in curated],
               "columns": ["layer", "neuron", "titlePlus", "titleMinus", "score", "flags", "confPlus", "confMinus"]},
              open(f"{out}/index.json", "w"), ensure_ascii=False, separators=(",", ":"))
    print(f"curated {len(curated)}, search forms {len(forms)}, terms {len(TERMS)}, contexts translated {len(CONTEXT_EN)}")


if __name__ == "__main__":
    main()
