"""Machine-translate the Neuron Lens explorer's non-English text with Claude Haiku 5.5 (Batch API).

    uv run --no-project --with anthropic --with pyarrow --with torch --with tokenizers python -I \
        scripts/neurons/translate.py <stage> --catalog <catalog dir> --tokenizer <tokenizer.json> --work <dir>

Three kinds of item, collected from a neuron-lens catalog (Qwen3-8B or Gemma 4 12B):

- ``term``: every kept vocabulary token that is (or may be) a word from another language, and every
  non-Latin-script run quoted in a Haiku label. Translated to a short English meaning, a
  romanization and a language code.
- ``context``: every top-firing snippet (neurons' and heads' ``acts.contexts``) with non-ASCII
  letters or several non-English function words. Translated whole, keeping the ``⟦ ⟧`` and
  ``⟪ ⟫`` markers on the matching English words.
- ``caption``: Gemma 4's image and sound captions and transcripts in other languages (FLEURS, MLS).

``--parts`` picks what ``collect`` gathers (``vocab,contexts,captions,labels``), so a model's labels
can be translated in their own run once they exist; ``--known`` names other runs' ``items.jsonl``,
whose items are skipped. Each run keeps its state in its own ``--work`` folder, by default
``webui/.neuron-lens/<model>``.

Stages (state under ``--work``; nothing is sent to the API without an explicit stage):

- ``collect``: ``items.jsonl``, the sorted unique items, and the request plan;
- ``pilot``: one batch of ``--n`` random requests; prints sample translations, the parse rate and
  the full run's projected cost into ``state.json``;
- ``submit``: every remaining request, in batches under the API's limits; refuses unless a pilot
  projected under ``--cap`` dollars;
- ``status``, ``results`` (collect finished batches into ``translations.jsonl``), ``retry`` (failed
  requests, once);
- ``fill``: after ``results``, one batch of what the first pass skipped but can't be English (items
  with non-Latin letters and no translation, plus the failed requests' items), in smaller requests
  that ask for every item. Haiku at low effort sometimes answers a long list only in part.

The builder (``build-neuron-catalog.py``) reads every run's ``translations.jsonl``. The API key comes from
``ANTHROPIC_API_KEY`` or ``~/.config/anthropic/api_key``.
"""
import argparse
import collections
import json
import os
import random
import re
import sys
import time
import unicodedata
from pathlib import Path

MODEL = "claude-haiku-5-5"
PRICE = {"input": 0.05, "output": 0.25, "cache_read": 0.005, "cache_write": 0.0625}  # $/MTok, Haiku 5.5 batch prices
PER_REQUEST = {"term": 120, "context": 24, "caption": 24}
FILL_PER_REQUEST = {"term": 40, "context": 8, "caption": 8}
MAX_TOKENS = {"term": 6000, "context": 6000, "caption": 6000}
BATCH_REQUESTS = 50_000
MIN_COUNT = 5  # the catalog's kept tokens: seen 5+ times in the activation corpora

TERM_SYSTEM = """You translate pieces of text into English for an interpretability explorer. Each \
item is a token from a language model's tokenizer or a short term quoted in a description of a neuron. A \
leading space marks the start of a word, and tokens are often pieces of longer words.

Return an entry only for items written in a language other than English. Skip English words and \
English word pieces, loanwords used in English, code identifiers, file extensions, abbreviations, \
numbers, symbols, emoji, and names written the same way in English. A Latin-script item that could \
plausibly be English is English.

For each entry:
- i: the item's number.
- lang: its ISO 639-1 code (zh, ja, ko, ru, de, fr, es, ...). Chinese characters without kana are zh.
- en: its English meaning, as short as possible: one to four words, lowercase except proper nouns. \
For a piece of a longer word, give the meaning it most likely carries; if it carries none on its \
own, skip it.
- ro: a romanization for non-Latin scripts (Hanyu Pinyin with tone marks for Chinese, Hepburn for \
Japanese, Revised Romanization for Korean, a standard transliteration otherwise), or "" for \
Latin-script items."""

CONTEXT_SYSTEM = """You translate short snippets of text into English for an interpretability \
explorer. Each snippet is a window around one token where a neuron or an attention head fired: the \
token is marked ⟦like this⟧, and for an attention head the earlier position it read from is marked \
⟪like this⟫. ⏎ stands for a line break. Snippets start and end mid-sentence.

Return an entry only for snippets containing text in a language other than English. Skip snippets \
that are already English, and code whose only non-English parts are identifiers.

For each entry:
- i: the snippet's number.
- lang: the ISO 639-1 code of its main non-English language.
- en: a faithful English translation of the whole snippet, itself a fragment where the snippet is \
one. Keep English words, code, numbers, names and ⏎ as they are. Keep each marker exactly once, \
around the English word or words carrying the marked token's meaning; if the marked token is a \
grammatical piece with no English counterpart, mark the word it attaches to. Add no explanations."""


CAPTION_SYSTEM = """You translate captions and transcripts of images and sounds into English for an \
interpretability explorer. Transcripts of speech may lack punctuation and capitals.
Return an entry only for items in a language other than English. Skip items that are already English.
For each entry:
- i: the item's number.
- lang: the ISO 639-1 code of its language.
- en: a faithful English translation of the whole item. Keep names and numbers as they are. Add no \
explanations."""
SYSTEM = {"term": TERM_SYSTEM, "context": CONTEXT_SYSTEM, "caption": CAPTION_SYSTEM}


def schema(kind: str) -> dict:
    fields = {"i": {"type": "integer"}, "lang": {"type": "string"}, "en": {"type": "string"}}
    if kind == "term":
        fields["ro"] = {"type": "string"}
    entry = {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}
    return {"type": "object", "properties": {"items": {"type": "array", "items": entry}},
            "required": ["items"], "additionalProperties": False}


# ---------- what needs translating

def latin(c: str) -> bool:
    return "LATIN" in unicodedata.name(c, "")


def foreign_letter(c: str) -> bool:
    """A letter (or combining mark) of a script other than Latin."""
    return unicodedata.category(c)[0] in "LM" and ord(c) > 0x7F and not latin(c)


FOREIGN_RUN = re.compile(r"[^\W\d_]+", re.UNICODE)

# Function words common in other Latin-script languages and rare in English, to catch accentless text.
FOREIGN_STOP = set("""de la el los las del que por para con una uno sus pero como mas este esta
est les des une dans pour avec sur pas qui sont aux ces cette nous vous ils elle
und der die das ist nicht mit dem den ein eine auf auch sich wird sind zum zur oder aber wie
il di che per non sono della delle degli anche come gli nel nella
het een van niet zijn wordt ook voor met
och att det som inte jag har med
yang dan ini itu dengan untuk tidak dari dalam akan pada adalah
ng mga sa ang na""".split())
EN_STOP = set("""the and of to in is that for it with as was on be by this are not or from at
which an have has but they you we he she his her their its were been will would can there""".split())


def foreign_term(s: str) -> bool:
    return any(foreign_letter(c) for c in s)


def maybe_foreign_word(s: str) -> bool:
    """A Latin-script token worth screening: accented letters, or a plain word of 3+ letters."""
    t = s.strip()
    if any(unicodedata.category(c)[0] == "L" and ord(c) > 0x7F for c in t):
        return True
    return bool(re.fullmatch(r"[A-Za-z]{3,}", t))


def foreign_context(s: str) -> bool:
    if any(unicodedata.category(c)[0] == "L" and ord(c) > 0x7F for c in s):
        return True
    words = re.findall(r"[a-z]+", s.lower())
    f = sum(w in FOREIGN_STOP for w in words)
    return f >= 2 and f > sum(w in EN_STOP for w in words)


def label_runs(text: str) -> set[str]:
    """Runs of non-Latin letters in a label, such as the Chinese examples Haiku quotes."""
    out = set()
    for m in FOREIGN_RUN.finditer(text or ""):
        run = m.group(0)
        if foreign_term(run):
            out.add(run)
    return out


def collect(args) -> None:
    import pyarrow.parquet as pq
    import torch
    from tokenizers import Tokenizer

    cat = Path(args.catalog)
    parts = set(args.parts.split(","))
    known = set()
    for path in filter(None, args.known.split(",")):
        known |= {(r["k"], r["s"]) for r in map(json.loads, open(path, encoding="utf-8"))}
    terms, contexts, captions = set(), set(), set()
    if "vocab" in parts:
        tok = Tokenizer.from_file(args.tokenizer)
        counts = torch.load(args.token_counts, weights_only=False)["counts"]
        for i in range(min(tok.get_vocab_size(), len(counts))):
            if counts[i] < MIN_COUNT:
                continue
            s = tok.decode([i])
            if "�" in s or not s.strip():
                continue
            if foreign_term(s) or maybe_foreign_word(s):
                terms.add(s.strip())
        print("vocabulary terms", len(terms), flush=True)

    def add_contexts(col):
        for s in col:
            for c in (s or "").split(" ‖ "):
                if c and foreign_context(c):
                    contexts.add(c)

    def add_label_runs(path, columns):
        if path.exists():
            lab = pq.read_table(path, columns=columns)
            for c in lab.column_names:
                for text in lab.column(c).to_pylist():
                    terms.update(label_runs(text))

    for path in sorted((cat / "neurons" / "acts").glob("L*.parquet")):
        if "contexts" in parts:
            add_contexts(pq.read_table(path, columns=["contexts"]).column(0).to_pylist())
        if "labels" in parts:
            add_label_runs(cat / "neurons" / "labels" / path.name, ["fires", "writes", "note"])
        print(f"{path.stem}: terms {len(terms)} contexts {len(contexts)}", flush=True)
    if "contexts" in parts:
        add_contexts(pq.read_table(cat / "heads" / "acts.parquet", columns=["contexts"]).column(0).to_pylist())
    if "labels" in parts:
        add_label_runs(cat / "heads" / "labels.parquet", ["fires", "attends", "writes", "note"])
    if "captions" in parts:
        for path in sorted((cat / "media").glob("*_docs.parquet")):
            for c in pq.read_table(path, columns=["caption"]).column(0).to_pylist():
                if c and foreign_context(c):
                    captions.add(c)
        print("captions", len(captions), flush=True)

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    kinds = [(k, sorted(x for x in items if (k, x) not in known)) for k, items in (("term", terms), ("context", contexts), ("caption", captions))]
    with open(work / "items.jsonl", "w", encoding="utf-8") as f:
        for kind, items in kinds:
            for s in items:
                f.write(json.dumps({"k": kind, "s": s}, ensure_ascii=False) + "\n")
    print("items: " + ", ".join(f"{len(x)} {k}s" for k, x in kinds)
          + f" -> {sum(-(-len(x) // PER_REQUEST[k]) for k, x in kinds)} requests")


# ---------- requests

def plan(work: Path) -> dict[str, list[str]]:
    """custom_id -> sources, in the stable order of items.jsonl."""
    by_kind = collections.defaultdict(list)
    for line in open(work / "items.jsonl", encoding="utf-8"):
        r = json.loads(line)
        by_kind[r["k"]].append(r["s"])
    out = {}
    for kind, items in by_kind.items():
        n = PER_REQUEST[kind]
        for j in range(0, len(items), n):
            out[f"{kind}-{j // n:05d}"] = items[j:j + n]
    return out


FILL_NOTE = "Every item below contains text in another language: return an entry for each one you can translate.\n\n"


def request(cid: str, items: list[str]) -> dict:
    kind = cid.split("-")[0]
    listing = "\n".join(json.dumps({"i": i, "t": s}, ensure_ascii=False) for i, s in enumerate(items))
    note = FILL_NOTE if "-fill" in cid else ""
    return {"custom_id": cid, "params": {
        "model": MODEL, "max_tokens": MAX_TOKENS[kind],
        "thinking": {"type": "disabled"},  # allowed at effort high or below
        "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": schema(kind)}},
        "system": [{"type": "text", "text": SYSTEM[kind], "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": f"{note}Items, one JSON object a line:\n{listing}"}],
    }}


def client():
    import anthropic
    key = os.environ.get("ANTHROPIC_API_KEY")
    path = Path.home() / ".config" / "anthropic" / "api_key"
    if not key and path.exists():
        key = path.read_text().strip()
    if not key:
        raise SystemExit("no API key: set ANTHROPIC_API_KEY or put it in ~/.config/anthropic/api_key")
    return anthropic.Anthropic(api_key=key)


def load_state(work: Path) -> dict:
    p = work / "state.json"
    return json.loads(p.read_text()) if p.exists() else {"batches": {}, "done": [], "failed": []}


def save_state(work: Path, s: dict) -> None:
    (work / "state.json").write_text(json.dumps(s, indent=1))


def cost(usage: dict) -> float:
    return (usage.get("input_tokens", 0) * PRICE["input"] + usage.get("output_tokens", 0) * PRICE["output"]
            + usage.get("cache_read_input_tokens", 0) * PRICE["cache_read"]
            + usage.get("cache_creation_input_tokens", 0) * PRICE["cache_write"]) / 1e6


def usage_of(message) -> dict:
    u = message.usage
    return {k: getattr(u, k, 0) or 0 for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")}


def parse(cid: str, items: list[str], message) -> list[dict] | None:
    """The request's translations as cache rows, or None when the response can't be used."""
    kind = cid.split("-")[0]
    text = next((b.text for b in message.content if b.type == "text"), "")
    try:
        entries = json.loads(text)["items"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    rows = []
    for e in entries:
        i = e.get("i")
        if not isinstance(i, int) or not 0 <= i < len(items) or not e.get("en", "").strip():
            continue
        row = {"k": kind, "s": items[i], "lang": e.get("lang", ""), "en": e["en"].strip()}
        if kind == "term":
            row["ro"] = e.get("ro", "").strip()
        else:
            # A translation must keep each marker of its source exactly once to be shown in place.
            row["ok"] = all(row["en"].count(m) == items[i].count(m) for m in "⟦⟧⟪⟫")
        rows.append(row)
    return rows


def wait(c, batch_id: str) -> None:
    while c.messages.batches.retrieve(batch_id).processing_status != "ended":
        time.sleep(20)


def pilot(args) -> None:
    work = Path(args.work)
    c, allc = client(), plan(work)
    rng = random.Random(0)
    by_kind = collections.defaultdict(list)
    for cid in allc:
        by_kind[cid.split("-")[0]].append(cid)
    pick = [cid for kind, ids in by_kind.items() for cid in rng.sample(ids, min(args.n, len(ids)))]
    b = c.messages.batches.create(requests=[request(cid, allc[cid]) for cid in pick])
    print("pilot batch", b.id, len(pick), "requests", flush=True)
    wait(c, b.id)
    use = collections.defaultdict(lambda: collections.Counter())
    ok, rows = collections.Counter(), []
    for r in c.messages.batches.results(b.id):
        kind = r.custom_id.split("-")[0]
        if r.result.type != "succeeded":
            continue
        use[kind].update(usage_of(r.result.message))
        got = parse(r.custom_id, allc[r.custom_id], r.result.message)
        if got is not None:
            ok[kind] += 1
            rows += got
    picked = collections.Counter(cid.split("-")[0] for cid in pick)
    projected = sum(cost(use[k]) / picked[k] * len(by_kind[k]) for k in use)
    s = load_state(work)
    s["pilot"] = {"batch": b.id, "requests": len(pick), "parsed": dict(ok), "usage": {k: dict(v) for k, v in use.items()},
                  "spent": sum(cost(v) for v in use.values()), "projected": round(projected, 2)}
    save_state(work, s)
    with open(work / "pilot.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(s["pilot"], indent=1))
    for row in rng.sample(rows, min(40, len(rows))):
        print(json.dumps(row, ensure_ascii=False))


def submit(args) -> None:
    work = Path(args.work)
    c, allc, s = client(), plan(work), load_state(work)
    p = s.get("pilot")
    if not p or p["projected"] > args.cap:
        raise SystemExit(f"refusing: run a pilot that projects under ${args.cap}")
    sent = {x for b in s["batches"].values() for x in b["ids"]} | set(s["done"])
    todo = [cid for cid in allc if cid not in sent]
    for j in range(0, len(todo), BATCH_REQUESTS):
        ids = todo[j:j + BATCH_REQUESTS]
        b = c.messages.batches.create(requests=[request(cid, allc[cid]) for cid in ids])
        s["batches"][b.id] = {"ids": ids, "collected": False}
        save_state(work, s)
        print("submitted", b.id, len(ids), flush=True)


def status(args) -> None:
    work = Path(args.work)
    c, s = client(), load_state(work)
    for i, info in s["batches"].items():
        b = c.messages.batches.retrieve(i)
        print(i, b.processing_status, dict(b.request_counts), "collected" if info["collected"] else "")


def results(args) -> None:
    work = Path(args.work)
    c, s = client(), load_state(work)
    allc = plan(work) | s.get("fill", {})
    done, failed = set(s["done"]), set(s["failed"])
    spent = collections.Counter(s.get("usage", {}))
    with open(work / "translations.jsonl", "a", encoding="utf-8") as f:
        for i, info in s["batches"].items():
            if info["collected"] or c.messages.batches.retrieve(i).processing_status != "ended":
                continue
            for r in c.messages.batches.results(i):
                rows = parse(r.custom_id, allc[r.custom_id], r.result.message) if r.result.type == "succeeded" else None
                if r.result.type == "succeeded":
                    spent.update(usage_of(r.result.message))
                if rows is None:
                    failed.add(r.custom_id)
                    continue
                failed.discard(r.custom_id)
                done.add(r.custom_id)
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            info["collected"] = True
            print("collected", i, flush=True)
    s["done"], s["failed"], s["usage"] = sorted(done), sorted(failed), dict(spent)
    save_state(work, s)
    print(f"done {len(done)} / {len(allc)} requests, failed {len(failed)}, spent ${cost(spent):.2f}")


def retry(args) -> None:
    work = Path(args.work)
    c, s = client(), load_state(work)
    allc = plan(work) | s.get("fill", {})
    if not s["failed"]:
        print("nothing to retry")
        return
    b = c.messages.batches.create(requests=[request(cid, allc[cid]) for cid in s["failed"]])
    s["batches"][b.id] = {"ids": s["failed"], "collected": False, "retry": True}
    save_state(work, s)
    print("retry batch", b.id, len(s["failed"]))


def fill(args) -> None:
    work = Path(args.work)
    c, allc, s = client(), plan(work), load_state(work)
    have = {(r["k"], r["s"]) for r in map(json.loads, open(work / "translations.jsonl", encoding="utf-8"))}
    todo = {"term": {}, "context": {}, "caption": {}}  # dicts as ordered sets
    for cid in s["failed"]:
        todo[cid.split("-")[0]].update(dict.fromkeys(allc[cid]))
    for cid, items in allc.items():
        kind = cid.split("-")[0]
        need = 2 if kind == "context" else 1  # a lone CJK character in an English snippet is usually a name
        todo[kind].update(dict.fromkeys(x for x in items if (kind, x) not in have and sum(map(foreign_letter, x)) >= need))
    fills = s.setdefault("fill", {})
    ids = []
    for kind, items in todo.items():
        items, n = list(items), FILL_PER_REQUEST[kind]
        for j in range(0, len(items), n):
            cid = f"{kind}-fill{len(fills):05d}"
            fills[cid] = items[j:j + n]
            ids.append(cid)
        print(kind, len(items), "items", flush=True)
    if not ids:
        print("nothing to fill")
        return
    b = c.messages.batches.create(requests=[request(cid, fills[cid]) for cid in ids])
    s["batches"][b.id] = {"ids": ids, "collected": False, "fill": True}
    s["failed"] = []  # their items ride in the fill batch
    save_state(work, s)
    print("fill batch", b.id, len(ids), "requests", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("collect", "pilot", "submit", "status", "results", "retry", "fill"))
    ap.add_argument("--catalog", default=str(Path.home() / "Work/neuron-lens/data/catalog/qwen3-8b"))
    ap.add_argument("--token-counts", help="default: neuron-lens's token counts for the catalog's model")
    ap.add_argument("--tokenizer")
    ap.add_argument("--work", help="default: webui/.neuron-lens/<model>")
    ap.add_argument("--parts", default="vocab,contexts,captions,labels")
    ap.add_argument("--known", default="", help="comma-separated items.jsonl of earlier runs, whose items are skipped")
    ap.add_argument("--n", type=int, default=12, help="pilot requests per kind")
    ap.add_argument("--cap", type=float, default=25.0, help="refuse to submit if the pilot projects more dollars")
    args = ap.parse_args()
    model = Path(args.catalog).name
    args.work = args.work or str(Path(__file__).resolve().parents[2] / ".neuron-lens" / model)
    data = Path(args.catalog).parents[1]
    args.token_counts = args.token_counts or str(data / ("token_counts.pt" if model == "qwen3-8b" else f"token_counts-{model}.pt"))
    {"collect": collect, "pilot": pilot, "submit": submit, "status": status, "results": results, "retry": retry, "fill": fill}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
