"""C TVM VM checks of the static reads and `exact-readout-v2` (hook ABI v5, B1 §4.1, §4.3).

Run by `units_goldens.verify_arch` on each fixture library. Two layers:

- every primitive against its formula, in fp64 numpy over the converted (dequantized) tensors,
  on seeded inputs (`formula:` in the worst-|Δ| record);
- the goldens wherever a device primitive is involved, composed the way the host composes them:
  `neuron-push` (the registered direction is the neuron write), `unit-reads` (each neuron read's
  `w_norm` and `cos_ab`), `unit-artifacts` (every population scalar and contrast statistic of
  every lens layer that the device reads produce: R writes and fires± `ρ`, `lse`, `lse_neg`, mean,
  std; S writes and fires± `‖u‖`, `sqrt(uᵀΣ_E u)`, mean, std, S fires pooled), `slens` (the atom
  directions `K_L ẽ_v` and the footprint z top-k), `lens-reads` (every present R and S half).

The fixture lenses come from the goldens' portable generator (`units-tiny-models-v1.json`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

SEED = 0x5EED0005
LENS_ORDINAL = 1000
LENS_LAYERS = {("R", "out"): (0, 1), ("R", "mid"): (1, 3), ("S", "out"): (0, 1), ("S", "mid"): (0, 2)}
R_STD_FLOOR, S_STD_FLOOR = 1e-4, 1e-6
LOWEST = np.float32(-3.4028234663852886e38)
_MASK = 0xFFFFFFFF


def _h(x):
    x = x & _MASK
    x ^= x >> 16
    x = (x * 0x7FEB352D) & _MASK
    x ^= x >> 15
    x = (x * 0x846CA68B) & _MASK
    x ^= x >> 16
    return x


def _u(ordinal: int, count: int):
    seed = int(_h(np.array([(SEED + 0x9E3779B9 * ordinal) & _MASK], dtype=np.uint64))[0])
    return _h(np.arange(count, dtype=np.uint64) ^ np.uint64(seed))


def fixture_lenses(d: int) -> dict:
    """`(family, site, layer) → [d, d]` fp32, as `browser_units.fixture_lens` builds them."""
    lenses = {}
    for i, ((family, site), layers) in enumerate(sorted(LENS_LAYERS.items())):
        for layer in layers:
            u = _u(LENS_ORDINAL + 16 * i + layer, d * d)
            noise = (((u >> 16) % 256).astype(np.float64) - 128.0) * 2.0**-10
            lenses[(family, site, layer)] = (0.75 * np.eye(d) + noise.reshape(d, d)).astype(np.float32)
    return lenses


class Weights:
    """The fixture library's weights in fp64, dequantized from the tensor cache."""

    def __init__(self, tensors: dict, block: dict, num_layers: int):
        self.tensors, self.block, self.num_layers = tensors, block, num_layers
        self.prefix = "language_model.model." if any(
            n.startswith("language_model.") for n in tensors) else "model."

    def get(self, name: str) -> np.ndarray:
        name = self.prefix + name
        if name + ".weight" in self.tensors:
            return self.tensors[name + ".weight"].astype(np.float64)
        weight, scale = self.tensors[name + ".q_weight"], self.tensors[name + ".q_scale"]
        shifts = np.arange(8, dtype=np.uint32) * 4
        codes = ((weight[:, :, None] >> shifts) & 0xF).reshape(weight.shape[0], -1).astype(np.float64)
        return (codes - 7) * np.repeat(scale.astype(np.float64), 32, axis=1)

    def gain(self, layer: int, which: str) -> np.ndarray:
        sandwich = self.block["sandwich"]
        names = {"ai": "input_layernorm",
                 "ao": "post_attention_layernorm" if sandwich else None,
                 "mi": "pre_feedforward_layernorm" if sandwich else "post_attention_layernorm",
                 "mo": "post_feedforward_layernorm" if sandwich else None}
        if names[which] is None:
            return np.ones(self.get("embed_tokens").shape[1])
        return self.get(f"layers.{layer}.{names[which]}")

    def layer(self, layer: int) -> dict:
        b, p = self.block, f"layers.{layer}."
        hd, heads, kv, d_ff = b["head_dim"], b["n_heads"], b["n_kv_heads"], b["d_ff"]
        if self.prefix + p + "self_attn.c_attn.weight" in self.tensors or \
                self.prefix + p + "self_attn.c_attn.q_weight" in self.tensors:
            qkv = self.get(p + "self_attn.c_attn")
            q, k, v = qkv[: heads * hd], qkv[heads * hd: (heads + kv) * hd], qkv[(heads + kv) * hd:]
        else:
            q, k, v = (self.get(p + f"self_attn.{r}_proj") for r in "qkv")
        gate_up = self.get(p + "mlp.gate_up_proj")
        return {"q": q, "k": k, "v": v, "o": self.get(p + "self_attn.o_proj"),
                "gate": gate_up[:d_ff], "up": gate_up[d_ff:], "down": self.get(p + "mlp.down_proj")}

    def writes(self, layer: int) -> np.ndarray:
        """`[d_ff, d]`: `s·γ_mo ⊙ W_down[:, n]` per neuron."""
        scale = float(self.block.get("residual_scale", 1.0))
        return (scale * self.gain(layer, "mo")[:, None] * self.layer(layer)["down"]).T

    def reads(self, layer: int):
        gain = self.gain(layer, "mi")
        w = self.layer(layer)
        a, b = w["gate"] * gain, w["up"] * gain
        return (a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12),
                b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12))

    def head_write(self, layer: int, head: int) -> np.ndarray:
        hd = self.block["head_dim"]
        return self.gain(layer, "ao")[:, None] * self.layer(layer)["o"][:, head * hd:(head + 1) * hd]

    @property
    def embedding(self) -> np.ndarray:
        return self.get("embed_tokens")

    @property
    def final_gain(self) -> np.ndarray:
        return self.get("norm")


class Device:
    """Calls into the fixture library's static reads (fp32 in, numpy out)."""

    def __init__(self, vm):
        self.vm, self.tvm = vm, vm.env.tvm

    def __call__(self, name: str, *args, packed: bool = True):
        arrays = [self.tvm.runtime.tensor(np.ascontiguousarray(a)) for a in args]
        out = self.vm.vm[name](*arrays, self.vm.params) if packed else self.vm.vm[name](*arrays)
        if hasattr(out, "numpy"):
            return out.numpy()
        return [o.numpy() for o in out]

    def topk(self, x: np.ndarray, k: int):
        n = x.shape[0]
        values, indices = self("drowse_topk", x.astype(np.float32), np.zeros((n, k), np.float32),
                               np.zeros((n, k), np.int32), packed=False)
        return values, indices

    def chan(self, rows: np.ndarray, chunk: int = 37):
        """Mean and centred sum of squares of `rows` through repeated merges (chunked, so the
        merge rule itself is exercised)."""
        width = rows.shape[1]
        count, mean, m2 = 0, np.zeros(width, np.float32), np.zeros(width, np.float32)
        for start in range(0, rows.shape[0], chunk):
            part = rows[start:start + chunk].astype(np.float32)
            mean, m2 = self("drowse_chan_merge", np.array([count], np.float32), mean, m2, part,
                            packed=False)
            count += part.shape[0]
        return count, mean, m2


def mask_words(kept, vocab: int) -> np.ndarray:
    words = np.zeros((vocab + 31) // 32, np.uint32)
    for v in kept:
        words[v >> 5] |= np.uint32(1) << np.uint32(v & 31)
    return words


def exact_topk(x: np.ndarray, k: int):
    """Value desc, index asc."""
    order = np.lexsort((np.arange(x.shape[-1])[None, :].repeat(x.shape[0], 0), -x), axis=-1)[:, :k]
    return np.take_along_axis(x, order, -1), order


def _rms(x: np.ndarray, eps: float) -> np.ndarray:
    return 1.0 / np.sqrt(np.mean(np.asarray(x, np.float64) ** 2, axis=-1) + eps)


def verify_formulas(dev: Device, w: Weights, check, statistics) -> None:
    """Every primitive against its formula on seeded inputs."""
    block, L = w.block, w.num_layers
    d, d_ff = w.embedding.shape[1], block["d_ff"]
    heads, hd, kv = block["n_heads"], block["head_dim"], block["n_kv_heads"]
    vocab = w.embedding.shape[0]
    eps, scale_e = block["rms_eps"], float(block["embed_scale"])
    rng = np.random.default_rng(0xB4C)
    f32 = lambda *shape: rng.standard_normal(shape).astype(np.float32)  # noqa: E731

    units = np.array([[l, n] for l in range(L) for n in (0, 5, d_ff - 1)] + [[L - 1, 7], [0, 40]],
                     np.int32)
    rows = dev("drowse_neuron_rows", units)
    for i, (layer, n) in enumerate(units):
        a, b = w.reads(layer)
        check("formula:neuron_rows", np.stack([w.writes(layer)[n], a[n], b[n]]), rows[i])
    head_units = np.array([[l, h] for l in range(L) for h in range(heads)], np.int32)
    head = dev("drowse_head_rows", head_units)
    group = heads // kv
    for i, (layer, h) in enumerate(head_units):
        p, ai = w.layer(layer), w.gain(layer, "ai")
        g = h // group
        expect = np.stack([w.head_write(layer, h).T, ai * p["q"][h * hd:(h + 1) * hd],
                           ai * p["k"][g * hd:(g + 1) * hd], ai * p["v"][g * hd:(g + 1) * hd]])
        check("formula:head_rows", expect, head[i])

    layers = np.array([L - 1, 0, 2 % L, 0], np.int32)
    x = f32(len(layers), 3, d)
    got = dev("drowse_layer_write_scores", x, layers)
    reads = dev("drowse_layer_read_scores", x, layers)
    for j, layer in enumerate(layers):
        a, b = w.reads(layer)
        check("formula:layer_write_scores", x[j] @ w.writes(layer).T, got[j])
        check("formula:layer_read_scores", np.stack([x[j] @ a.T, x[j] @ b.T]), reads[j])

    g, act, mlp_o, z, attn_o = f32(4, d), f32(4, d_ff), f32(4, d), f32(4, heads * hd), f32(4, d)
    contributions = dev("drowse_neuron_contributions", g, act, mlp_o, layers)
    head_c = dev("drowse_head_contributions", g, z, attn_o, layers)
    grams = dev("drowse_head_write_grams")
    norms = dev("drowse_head_write_norms", z, attn_o, layers, grams, packed=False)
    for j, layer in enumerate(layers):
        r_mo = _rms(mlp_o[j], eps) if block["sandwich"] else 1.0
        r_ao = _rms(attn_o[j], eps) if block["sandwich"] else 1.0
        check("formula:neuron_contributions", act[j] * (w.writes(layer) @ g[j]) * r_mo,
              contributions[j])
        per_head = []
        norm = []
        for h in range(heads):
            bh = w.head_write(layer, h)
            zh = z[j, h * hd:(h + 1) * hd].astype(np.float64)
            per_head.append(r_ao * float((g[j] @ bh) @ zh))
            norm.append(r_ao * np.sqrt(max(0.0, float(zh @ (bh.T @ bh) @ zh))))
        check("formula:head_contributions", per_head, head_c[j])
        check("formula:head_write_norms", norm, norms[j])
    pairs = np.array([[0, 0], [L - 1, heads - 1], [1 % L, 2 % heads]], np.int32)
    gathered = dev("drowse_gather_head_grams", grams, pairs, packed=False)
    check("formula:gather_head_grams", grams[pairs[:, 0], pairs[:, 1]], gathered)

    ids = np.array([0, 3, 71, vocab - 1, 17], np.int32)
    mean_e = f32(d) * np.float32(0.1)
    E, gamma_f = w.embedding, w.final_gain
    check("formula:embedding_rows", scale_e * E[ids] - mean_e,
          dev("drowse_embedding_rows", ids, mean_e))
    unemb = dev("drowse_unembedding_rows", ids)
    check("formula:unembedding_rows", gamma_f * E[ids], unemb)
    h_final = f32(len(ids), d)
    check("formula:unembed_cotangent", _rms(h_final, eps)[:, None] * gamma_f * E[ids],
          dev("drowse_unembed_cotangent", h_final, ids))
    rows_x = f32(3, d)
    logits = dev("drowse_vocab_logits", rows_x)
    normed = rows_x * _rms(rows_x, eps)[:, None] * gamma_f
    check("formula:vocab_logits", normed @ E.T, logits)
    mean_s, sigma, words = statistics
    u = f32(3, d)
    scores = dev("drowse_footprint_scores", u, mean_s, sigma, words)
    kept = np.array([(words[v >> 5] >> (v & 31)) & 1 for v in range(vocab)], bool)
    raw = (scale_e * u.astype(np.float64) @ E.T - (u @ mean_s)[:, None]) / np.sqrt(
        np.maximum(np.einsum("ni,ij,nj->n", u, sigma.astype(np.float64), u), 1e-30))[:, None]
    check("formula:footprint_scores", raw[:, kept], scores[:, kept])
    if not np.all(scores[:, ~kept] == LOWEST):
        raise SystemExit("formula:footprint_scores: a masked id is not LOWEST")
    for flag in (0, 1):
        moments = dev("drowse_logit_moments", logits, words, np.array([flag], np.int32), packed=False)
        sel = logits[:, kept] if flag else logits
        c = sel.astype(np.float64) - sel.mean(-1, keepdims=True)
        var = np.maximum((c**2).mean(-1), 1e-30)
        check("formula:logit_moments", np.stack([(c**3).mean(-1) / var**1.5,
                                                 (c**4).mean(-1) / var**2], -1), moments)
    big = f32(3, 1500) * np.float32(4)
    big[1, 7:11] = big[1, 3]
    lse = dev("drowse_log_sum_exp", big, packed=False)
    check("formula:log_sum_exp", np.log(np.exp(big.astype(np.float64)).sum(-1)), lse)
    for k in (1, 5, 64, 300):
        values, indices = dev.topk(big, k)
        ev, ei = exact_topk(big, k)
        if not np.array_equal(indices, ei):
            raise SystemExit(f"formula:topk k={k}: indices differ from value desc, index asc")
        check("formula:topk", ev, values)
    masked = np.where(kept[None, :], logits, LOWEST)
    values, indices = dev.topk(masked, int(kept.sum()) + 3)
    if not (np.all(indices[:, -3:] >= 0) and np.all(values[:, -3:] == LOWEST)):
        raise SystemExit("formula:topk past the kept ids must return the LOWEST entries in id order")
    a, wb = f32(2, 3, 5), f32(2, 5, 4)
    check("formula:batched_matmul", a @ wb, dev("drowse_batched_matmul", a, wb, packed=False))
    check("formula:batched_matmul", a @ wb[:1], dev("drowse_batched_matmul", a, wb[:1], packed=False))
    wt = f32(1, 4, 5)
    check("formula:batched_matmul_nt", a @ wt.transpose(0, 2, 1),
          dev("drowse_batched_matmul_nt", a, wt, packed=False))
    xq, mq = f32(3, d), f32(3, d, d)
    check("formula:quadratic_form", np.einsum("bi,bij,bj->b", xq, mq, xq),
          dev("drowse_quadratic_form", xq, mq, packed=False))
    check("formula:quadratic_form", np.einsum("bi,ij,bj->b", xq, mq[0], xq),
          dev("drowse_quadratic_form", xq, mq[:1], packed=False))
    xg = f32(3, 4, 7)
    check("formula:grouped_row_norms", np.linalg.norm(xg, axis=-1),
          dev("drowse_grouped_row_norms", xg, packed=False))
    data = f32(100, 9) + np.float32(3)
    count, mean, m2 = dev.chan(data)
    check("formula:chan_merge", data.mean(0), mean)
    check("formula:chan_merge", ((data - data.mean(0)) ** 2).sum(0), m2)


class Reader:
    """The host's compositions of the device reads, as B5 composes them (fixture lenses)."""

    def __init__(self, dev: Device, w: Weights, lenses: dict, statistics):
        self.dev, self.w, self.lenses = dev, w, lenses
        self.mean_e, self.sigma, self.words = statistics
        self.L = w.num_layers
        self.eps_f = w.block["rms_eps"]

    def r_matrix(self, layer: int, site: str):
        if ("R", site, layer) in self.lenses:
            return self.lenses[("R", site, layer)]
        if site == "out" and layer == self.L - 1:
            return None  # the logit lens
        return False

    def transport_r(self, rows: np.ndarray, matrix) -> np.ndarray:
        rows = rows.astype(np.float32)
        if matrix is None:
            return rows
        return self.dev("drowse_batched_matmul_nt", rows[None], matrix[None], packed=False)[0]

    def pull_s(self, rows: np.ndarray, matrix) -> np.ndarray:
        return self.dev("drowse_batched_matmul", rows.astype(np.float32)[None], matrix[None],
                        packed=False)[0]

    def r_read(self, rows, matrix, k: int):
        logits = self.dev("drowse_vocab_logits", self.transport_r(rows, matrix))
        lse = self.dev("drowse_log_sum_exp", logits, packed=False)
        values, ids = self.dev.topk(logits, k)
        return ids, np.exp(values.astype(np.float64) - lse[:, None])

    def place(self, rows, layer: int, site: str, k: int, of: str, means: dict):
        """lens.read's two halves of `rows` at (layer, site): (ids, values) per half, or None
        where the fixture has no lens there."""
        r = s = None
        matrix = self.r_matrix(layer, site)
        if matrix is not False:
            r = self.r_read(rows, matrix, k)
        if ("S", site, layer) in self.lenses:
            centred = rows - means[site][layer] if of == "residual" else rows
            s = self.s_read(self.pull_s(centred, self.lenses[("S", site, layer)]), k)
        return {"r": r, "s": s}

    def s_read(self, u, k: int):
        scores = self.dev("drowse_footprint_scores", u.astype(np.float32), self.mean_e, self.sigma,
                          self.words)
        values, ids = self.dev.topk(scores, k)
        return ids, values


def declared_groups(doc: dict, path: str) -> list:
    """The golden's declared tie groups (positions) for the token list at `path`."""
    return [g for tie in doc.get("declared_ties", []) if tie["path"] == path for g in tie["groups"]]


def _tie_blocks(values: list, window: float, groups: list) -> list:
    """Positions whose order may differ: runs of golden values within `window` of each other,
    joined with the declared tie groups."""
    parent = list(range(len(values)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(values) - 1):
        if abs(values[i] - values[i + 1]) <= window:
            parent[find(i + 1)] = find(i)
    for group in groups:
        for p in group[1:]:
            parent[find(p)] = find(group[0])
    blocks: dict = {}
    for i in range(len(values)):
        blocks.setdefault(find(i), []).append(i)
    return list(blocks.values())


def _check_tokens(check, label: str, tokens: list, ids, values, atol: float,
                  groups: list = ()) -> None:
    """A golden token list against the device's: the same ids, each id's value within `atol`
    (+ the 6-decimal rounding), the device's values non-increasing, and each position holding the
    golden's id except inside a tie (a declared group, or golden values within the tolerance)."""
    golden = {int(t["token_id"]): float(t["value"]) for t in tokens}
    got = {int(i): float(v) for i, v in zip(ids, values)}
    if set(golden) != set(got):
        raise SystemExit(f"{label}: ids {sorted(got)} are not the golden's {sorted(golden)}")
    order = [golden[int(t["token_id"])] for t in tokens]
    check(label, order, [got[int(t["token_id"])] for t in tokens], atol=atol + 2e-6, rtol=0.0)
    device = [float(v) for v in values]
    if any(device[i] < device[i + 1] for i in range(len(device) - 1)):
        raise SystemExit(f"{label}: the device's values are not non-increasing: {device}")
    golden_ids = [int(t["token_id"]) for t in tokens]
    for block in _tie_blocks(order, 2 * (atol + 2e-6), list(groups)):
        if {golden_ids[p] for p in block} != {int(ids[p]) for p in block}:
            raise SystemExit(f"{label}: positions {block} hold {[int(ids[p]) for p in block]}, "
                             f"the golden {[golden_ids[p] for p in block]}")


def _check_place(check, tol: dict, label: str, golden: dict, got: dict, row: int, doc: dict,
                 path: str, where: str) -> None:
    """One row's present R and S halves against a place read's (R probabilities, S z)."""
    for half, atol in (("r", tol["probability_atol"]), ("s", tol["z_atol"])):
        want = golden[half]
        if want is None or want.get("absent") is not None or want.get("unit") == "layer_z":
            continue
        if got[half] is None:
            raise SystemExit(f"{label}: a present {half} half at {where} has no fixture lens")
        ids, values = got[half]
        m = len(want["tokens"])
        _check_tokens(check, f"{label}:{half}", want["tokens"], ids[row][:m], values[row][:m], atol,
                      declared_groups(doc, f"{path}.{half}.tokens"))


def verify_goldens(dev: Device, w: Weights, check, goldens: Path, arch: str, statistics) -> None:
    block, L = w.block, w.num_layers
    d, d_ff = w.embedding.shape[1], block["d_ff"]
    lenses = fixture_lenses(d)
    reader = Reader(dev, w, lenses, statistics)
    stats = json.loads((goldens / f"unit-stats-{arch}-v1.json").read_text())
    from units_goldens import decode

    means = {"out": decode(stats["out_mean"]), "mid": decode(stats["mid_mean"])}
    tol = json.loads((goldens / f"lens-reads-{arch}-v1.json").read_text())["tolerance"]["library"]

    # neuron-push: the registered profile is the neuron's write.
    push = json.loads((goldens / f"neuron-push-{arch}-v1.json").read_text())
    for layer, vector in push["profile"].items():
        unit = next(c for c in push["cases"] if "neuron/L" in json.dumps(c))
        index = int(json.dumps(unit).split("neuron/L")[1].split("n")[1].split('"')[0].split("@")[0])
        row = dev("drowse_neuron_rows", np.array([[int(layer), index]], np.int32))[0, 0]
        check("neuron-push:profile", decode(vector), row)

    # unit-reads: each neuron read's write norm and cos(â, b̂).
    reads = json.loads((goldens / f"unit-reads-{arch}-v1.json").read_text())
    for case in reads["reads"]:
        read = case["read"]
        if read["kind"] != "neuron":
            continue
        rows = dev("drowse_neuron_rows", np.array([[read["layer"], read["index"]]], np.int32))[0]
        check("unit-reads:w_norm", read["scalars"]["w_norm"], np.linalg.norm(rows[0]),
              atol=2e-6 + tol["atol"])
        check("unit-reads:cos_ab", read["scalars"]["cos_ab"], float(rows[1] @ rows[2]),
              atol=2e-6 + tol["atol"])

    # unit-artifacts: the population scalars and contrast statistics the device reads produce.
    artifacts = json.loads((goldens / f"unit-artifacts-{arch}-v1.json").read_text())
    embedded = dev("drowse_embedding_rows", np.arange(w.embedding.shape[0], dtype=np.int32),
                   reader.mean_e)
    for entry in artifacts["layers"]:
        layer, t = entry["layer"], entry["tensors"]
        rows = dev("drowse_neuron_rows", np.array([[layer, n] for n in range(d_ff)], np.int32))
        populations = {"writes": rows[:, 0], "fires+": rows[:, 1] + rows[:, 2],
                       "fires-": rows[:, 1] - rows[:, 2]}
        site_of = {"writes": "out", "fires+": "mid", "fires-": "mid"}
        for name, population in populations.items():
            key = f"R.{name}"
            if f"{key}.rho" not in t:
                continue
            matrix = reader.r_matrix(layer, site_of[name])
            x = reader.transport_r(population, matrix)
            logits = dev("drowse_vocab_logits", x)
            check(f"unit-artifacts:{key}.rho", decode(t[f"{key}.rho"]), _rms(x, reader.eps_f))
            check(f"unit-artifacts:{key}.lse", decode(t[f"{key}.lse"]),
                  dev("drowse_log_sum_exp", logits, packed=False))
            if f"{key}.lse_neg" in t:
                check(f"unit-artifacts:{key}.lse_neg", decode(t[f"{key}.lse_neg"]),
                      dev("drowse_log_sum_exp", -logits, packed=False))
            count, mean, m2 = dev.chan(logits)
            check(f"unit-artifacts:{key}.mean", decode(t[f"{key}.mean"]), mean)
            check(f"unit-artifacts:{key}.std", decode(t[f"{key}.std"]),
                  np.sqrt(np.maximum(m2 / count, R_STD_FLOOR**2)))
        pooled = []
        for name, population in populations.items():
            key = f"S.{name}"
            if f"{key}.norm" not in t:
                continue
            u = reader.pull_s(population, lenses[("S", site_of[name], layer)])
            norm = dev("drowse_grouped_row_norms", u[:, None, :], packed=False)[:, 0]
            check(f"unit-artifacts:{key}.norm", decode(t[f"{key}.norm"]), norm)
            root = np.sqrt(np.maximum(dev("drowse_quadratic_form", u, reader.sigma[None],
                                          packed=False), 1e-30))
            check(f"unit-artifacts:{key}.footprint_std", decode(t[f"{key}.footprint_std"]), root)
            unit_u = (u / np.maximum(norm, 1e-12)[:, None]).astype(np.float32)
            raw = dev("drowse_batched_matmul_nt", unit_u[None], embedded[None], packed=False)[0]
            if name == "writes":
                count, mean, m2 = dev.chan(raw)
                check(f"unit-artifacts:{key}.mean", decode(t[f"{key}.mean"]), mean)
                check(f"unit-artifacts:{key}.std", decode(t[f"{key}.std"]),
                      np.sqrt(np.maximum(m2 / count, S_STD_FLOOR**2)))
            else:
                pooled.append(raw)
        if pooled:
            count, mean, m2 = dev.chan(np.concatenate(pooled))
            check("unit-artifacts:S.fires.mean", decode(t["S.fires.mean"]), mean)
            check("unit-artifacts:S.fires.std", decode(t["S.fires.std"]),
                  np.sqrt(np.maximum(m2 / count, S_STD_FLOOR**2)))

    # slens: the atom directions `K_L ẽ_v` and the footprint z top-k.
    slens = json.loads((goldens / f"slens-{arch}-v1.json").read_text())
    words = dict(zip(slens["probe_words"], slens["probe_word_ids"]))
    for atom in slens["atoms"]:
        name = atom["atom"].split("/", 1)[1]
        token = int(name[1:]) if name.startswith("#") else words[name]
        e = dev("drowse_embedding_rows", np.array([token], np.int32), reader.mean_e)
        for layer, vector in atom["layers"].items():
            got = dev("drowse_batched_matmul_nt", e[None], lenses[("S", "out", int(layer))][None],
                      packed=False)[0, 0]
            check("slens:atom", decode(vector), got)
    for f, footprint in enumerate(slens["footprints"]):
        ids, z = reader.s_read(decode(footprint["u"]), footprint["k"])
        for row, top in enumerate(footprint["top"]):
            _check_tokens(check, "slens:footprint", [{"token_id": t["token_id"], "value": t["z"]}
                                                      for t in top], ids[row], z[row], tol["z_atol"],
                          declared_groups(slens, f"footprints[{f}].top[{row}]"))

    # lens-reads: every present half.
    lens_reads = json.loads((goldens / f"lens-reads-{arch}-v1.json").read_text())
    for q, request in enumerate(lens_reads["requests"]):
        if request.get("lens") is not None:
            continue  # a session with no S lens prepared: the absent text only
        layer, site = request["layer"], request["site"]
        got = reader.place(decode(request["rows"]), layer, site, request["k"], request["of"], means)
        for i, read in enumerate(request["reads"]):
            _check_place(check, tol, "lens-reads", read, got, i, lens_reads,
                         f"requests[{q}].reads[{i}]", f"L{layer} {site}")

    # unit-reads poles: a neuron's writes (w, −w at its out) and firing directions (â ± b̂ at its
    # mid) read through lens.read, and each pole's logit-lens skew and kurtosis over the kept
    # tokens. Hub-hidden halves (unit layer_z) rank by contrast statistics, B5's host composition
    # rather than a device read; every raw half is checked, hub-hidden read or not.
    for c, case in enumerate(reads["reads"]):
        read = case["read"]
        if read["kind"] != "neuron":
            continue
        rows = dev("drowse_neuron_rows", np.array([[read["layer"], read["index"]]], np.int32))[0]
        w, a, b = rows[0], rows[1], rows[2]
        moments = dev("drowse_logit_moments", dev("drowse_vocab_logits", w[None]),
                      reader.words, np.array([1], np.int32), packed=False)[0]
        places = {"writes": reader.place(np.stack([w, -w]), read["layer"], "out", case["k"],
                                         "direction", means),
                  "fires_on": reader.place(np.stack([a + b, a - b]), read["layer"], "mid",
                                           case["k"], "direction", means)}
        for p, pole in enumerate(read["poles"]):
            sign = 1.0 if pole["pole"] == "+" else -1.0
            check("unit-reads:vocab_moments", [pole["vocab_skew"], pole["vocab_kurt"]],
                  [sign * float(moments[0]), float(moments[1])], atol=2e-6 + tol["atol"])
            for where in ("writes", "fires_on"):
                _check_place(check, tol, f"unit-reads:{where}", pole[where], places[where], p,
                             reads, f"reads[{c}].read.poles[{p}].{where}",
                             f"L{read['layer']} {pole[where]['site']}")


def verify(env, vm, tensors: dict, goldens: Path, arch: str, check) -> None:
    if os.environ.get("DROWSE_READS_CONTINUE"):
        strict, failures = check, []

        def check(label, golden, actual, **kwargs):  # noqa: F811
            try:
                strict(label, golden, actual, **kwargs)
            except SystemExit as error:
                failures.append(str(error))
                print("FAIL", error, flush=True)
    w = Weights(tensors, vm.block, vm.num_layers)
    dev = Device(vm)
    slens = json.loads((goldens / f"slens-{arch}-v1.json").read_text())
    from units_goldens import decode

    statistics = (decode(slens["mean_E"]).astype(np.float32),
                  decode(slens["sigma_E"]).astype(np.float32),
                  mask_words(slens["mask"]["kept"], w.embedding.shape[0]))
    verify_formulas(dev, w, check, statistics)
    verify_goldens(dev, w, check, goldens, arch, statistics)
    if os.environ.get("DROWSE_READS_CONTINUE") and failures:
        raise SystemExit(f"{len(failures)} static-read checks failed")
