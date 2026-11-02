"""Production q4 parity of a units library's sites, static reads and activity (hook ABI v5).

B1 §4.3 item 2: the browser against Python on the dequantized MLC tensors. A `standard-v4`
library's capture carries, beside its residuals, the block sites `act`, `z` and `mid` at the same
layers and positions (through the replay), and a set of static reads. Python recomputes each from
the Transformers model `mlc_q4_torch` builds over the same tensors:

- per layer, the relative error of `act`, `z` and `mid` against the activation scale
  (`‖browser − python‖ / ‖python‖`, the q4 gate's residual measure) is at most 1%;
- each static read `vocabReadoutTopK(unitWeightRows("neuron_write", [unit]), 8)` (the softmax of
  `W_U · N_f(s·γ_mo ⊙ W_down[:, n])` over the full vocabulary) shares at least 7 of its top 8 ids
  with Python's, and each shared id's probability is within 1e-3 relative;
- unit activity: the top 32 neurons by peak `|act|` and the top 32 heads by peak write norm
  `‖(γ_ao ⊙ W_O^h) z_h‖ · r_ao` (B1 §2.2's tap value) over the captured layers and positions, ranked
  the same way from each side's sites, have Jaccard index at least 0.9. The rank is unscaled: the
  reading scales come from a units pack and divide both sides alike.

A `standard-v4-residual` library has none of these sites; its capture carries no `units` member.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

SITE_TOLERANCE = 0.01
STATIC_READ_TOLERANCE = 1e-3
STATIC_READ_TOP = 8
MINIMUM_STATIC_READ_OVERLAP = 7
MAXIMUM_STATIC_READS = 64
ACTIVITY_TOP = 32
MINIMUM_ACTIVITY_JACCARD = 0.9
SITES = ("act", "z", "mid")
_VOCAB_CHUNK = 16384


def _overlay_unit_architectures() -> tuple[str, ...]:
    # The overlay's own list (forks/overlay_checks.py), as the lock tooling reads it.
    path = Path(__file__).resolve().parent / "forks" / "overlay_checks.py"
    spec = importlib.util.spec_from_file_location("drowse_overlay_checks", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load forks/overlay_checks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tuple(module.UNIT_ARCHITECTURES)


UNIT_ARCHITECTURES = _overlay_unit_architectures()


def has_unit_sites(architecture: str) -> bool:
    return architecture in UNIT_ARCHITECTURES


def block_shape(model: torch.nn.Module) -> dict[str, int]:
    config = model.config
    return {
        "d_ff": int(config.intermediate_size),
        "n_heads": int(config.num_attention_heads),
        "head_dim": int(config.head_dim),
    }


def validate_units_member(
    units: object,
    *,
    layers: list[int],
    num_layers: int | None = None,
) -> dict[str, Any]:
    """The capture's `units` member: the block shape, the sites file digest and the static reads."""
    if not isinstance(units, dict) or set(units) != {"block", "sites_sha256", "static_reads"}:
        raise ValueError("browser units capture has an invalid schema")
    block = units["block"]
    if (
        not isinstance(block, dict)
        or set(block) != {"d_ff", "n_heads", "head_dim"}
        or any(not isinstance(block[key], int) or block[key] < 1 for key in block)
    ):
        raise ValueError("browser units capture block shape is invalid")
    digest = units["sites_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("browser units capture sites digest is invalid")
    reads = units["static_reads"]
    if not isinstance(reads, list) or not 1 <= len(reads) <= MAXIMUM_STATIC_READS:
        raise ValueError(f"browser units capture needs 1 to {MAXIMUM_STATIC_READS} static reads")
    seen = set()
    for read in reads:
        if (
            not isinstance(read, dict)
            or set(read) != {"layer", "neuron", "ids", "probabilities"}
            or not isinstance(read["layer"], int)
            or not 0 <= read["layer"] < (num_layers if num_layers is not None else 1 << 30)
            or not isinstance(read["neuron"], int)
            or not 0 <= read["neuron"] < block["d_ff"]
            or (read["layer"], read["neuron"]) in seen
            or not isinstance(read["ids"], list)
            or len(read["ids"]) != STATIC_READ_TOP
            or len(set(read["ids"])) != STATIC_READ_TOP
            or any(not isinstance(value, int) or value < 0 for value in read["ids"])
            or not isinstance(read["probabilities"], list)
            or len(read["probabilities"]) != STATIC_READ_TOP
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1
                for value in read["probabilities"]
            )
            or any(a < b for a, b in zip(read["probabilities"], read["probabilities"][1:]))
        ):
            raise ValueError("browser units capture static read is invalid")
        seen.add((read["layer"], read["neuron"]))
    if not {layer for layer, _ in seen} >= set(layers):
        raise ValueError("browser units capture needs a static read at every captured layer")
    return units


def site_width(block: dict[str, int], hidden_size: int) -> int:
    return block["d_ff"] + block["n_heads"] * block["head_dim"] + hidden_size


def split_sites(values: np.ndarray, block: dict[str, int], hidden_size: int) -> dict[str, np.ndarray]:
    """`[layers, positions, d_ff + H·hd + d]` (act, then z, then mid) into the three sites."""
    d_ff, z_width = block["d_ff"], block["n_heads"] * block["head_dim"]
    if values.shape[-1] != d_ff + z_width + hidden_size:
        raise ValueError("browser units sites have the wrong width")
    return {
        "act": values[..., :d_ff],
        "z": values[..., d_ff:d_ff + z_width],
        "mid": values[..., d_ff + z_width:],
    }


def _mid_norm(layer: torch.nn.Module, architecture: str) -> torch.nn.Module:
    # N_mi's input is h_mid: Gemma's pre_feedforward_layernorm, Qwen3's post_attention_layernorm.
    return layer.pre_feedforward_layernorm if architecture == "gemma3_text" else layer.post_attention_layernorm


def capture_torch_sites(
    model: torch.nn.Module,
    architecture: str,
    layers: list[int],
    positions: list[int],
    input_ids: list[int],
    device: str,
) -> dict[str, np.ndarray]:
    """`act` (down_proj's input), `z` (o_proj's input) and `mid` (N_mi's input) at each layer and
    position, ``[layers, positions, width]`` fp32."""
    captured: dict[tuple[str, int], torch.Tensor] = {}
    handles = []

    def keep(site: str, layer: int):
        def hook(_module: torch.nn.Module, args: tuple[object, ...]) -> None:
            value = args[0]
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"Transformers layer {layer} {site} input is not a tensor")
            captured[(site, layer)] = value.detach().to(torch.float32).cpu()

        return hook

    model_layers = model.model.layers
    for layer in layers:
        block = model_layers[layer]
        handles.append(block.mlp.down_proj.register_forward_pre_hook(keep("act", layer)))
        handles.append(block.self_attn.o_proj.register_forward_pre_hook(keep("z", layer)))
        handles.append(_mid_norm(block, architecture).register_forward_pre_hook(keep("mid", layer)))
    try:
        tokens = torch.tensor([input_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            model(tokens, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return {
        site: np.stack([captured[(site, layer)][0, positions].numpy() for layer in layers])
        for site in SITES
    }


def site_relative_errors(
    python: dict[str, np.ndarray],
    browser: dict[str, np.ndarray],
    layers: list[int],
) -> dict[str, dict[str, float]]:
    out = {}
    for site in SITES:
        expected, actual = python[site].astype(np.float64), browser[site].astype(np.float64)
        if expected.shape != actual.shape:
            raise ValueError(f"browser {site} site has the wrong shape")
        out[site] = {
            str(layer): float(
                np.linalg.norm(actual[index] - expected[index]) / max(np.linalg.norm(expected[index]), 1e-12)
            )
            for index, layer in enumerate(layers)
        }
    return out


def _weight(module: torch.nn.Module) -> np.ndarray:
    return module.weight.detach().to(torch.float64).cpu().numpy()


def _norm_parameters(module: torch.nn.Module) -> tuple[np.ndarray, float]:
    return _weight(module), float(module.variance_epsilon)


def neuron_write(model: torch.nn.Module, architecture: str, layer: int, neuron: int) -> np.ndarray:
    """`s·γ_mo ⊙ W_down[:, n]` (s = 1 on both unit architectures; γ_mo is Gemma's
    post_feedforward_layernorm, identity on Qwen3)."""
    block = model.model.layers[layer]
    column = block.mlp.down_proj.weight[:, neuron].detach().to(torch.float64).cpu().numpy()
    if architecture == "gemma3_text":
        column = column * _weight(block.post_feedforward_layernorm)
    return column


def vocab_readout_topk(model: torch.nn.Module, row: np.ndarray, k: int) -> tuple[list[int], np.ndarray]:
    """Top `k` of `softmax(W_U · N_f(row))` over the full vocabulary, in fp64 (ties by id)."""
    gain, epsilon = _norm_parameters(model.model.norm)
    normalized = row / math.sqrt(float(np.mean(row * row)) + epsilon) * gain
    unembedding = model.lm_head.weight
    logits = np.concatenate([
        unembedding[start:start + _VOCAB_CHUNK].detach().to(torch.float64).cpu().numpy() @ normalized
        for start in range(0, unembedding.shape[0], _VOCAB_CHUNK)
    ])
    shifted = logits - logits.max()
    probabilities = np.exp(shifted) / np.exp(shifted).sum()
    order = np.lexsort((np.arange(len(probabilities)), -probabilities))[:k]
    return [int(value) for value in order], probabilities


def static_read_parity(
    model: torch.nn.Module,
    architecture: str,
    reads: list[dict[str, Any]],
) -> dict[str, Any]:
    worst_relative, least_overlap, details = 0.0, STATIC_READ_TOP, []
    for read in reads:
        python_ids, probabilities = vocab_readout_topk(
            model, neuron_write(model, architecture, read["layer"], read["neuron"]), STATIC_READ_TOP,
        )
        shared = [index for index, token in enumerate(read["ids"]) if token in python_ids]
        overlap = len(shared)
        relative = max(
            (
                abs(read["probabilities"][index] - probabilities[read["ids"][index]])
                / max(probabilities[read["ids"][index]], 1e-30)
                for index in shared
            ),
            default=math.inf,
        )
        worst_relative = max(worst_relative, float(relative))
        least_overlap = min(least_overlap, overlap)
        details.append({
            "layer": read["layer"], "neuron": read["neuron"], "overlap": overlap,
            "maxRelativeError": float(relative), "pythonIds": python_ids,
        })
    return {"maxRelativeError": worst_relative, "minOverlap": least_overlap, "reads": details}


def _top(scores: np.ndarray, top: int) -> set[int]:
    flat = scores.reshape(-1)
    order = np.lexsort((np.arange(flat.size), -flat))
    return {int(index) for index in order[:top]}


def head_write_norms(
    model: torch.nn.Module,
    architecture: str,
    layers: list[int],
    z: np.ndarray,
) -> np.ndarray:
    """`‖(γ_ao ⊙ W_O^h) z_h‖ · r_ao` per layer, position and head (`r_ao` is Gemma's
    `rsqrt(mean(attn_o²) + ε_ao)`, 1 on Qwen3) → ``[layers, positions, H]``."""
    shape = block_shape(model)
    heads, head_dim = shape["n_heads"], shape["head_dim"]
    out = np.zeros((len(layers), z.shape[1], heads))
    for slot, layer in enumerate(layers):
        block = model.model.layers[layer]
        output = _weight(block.self_attn.o_proj)
        values = z[slot].astype(np.float64)
        scale = np.ones(values.shape[0])
        gain = np.ones(output.shape[0])
        if architecture == "gemma3_text":
            gain, epsilon = _norm_parameters(block.post_attention_layernorm)
            attention_output = values @ output.T
            scale = 1 / np.sqrt(np.mean(attention_output * attention_output, axis=-1) + epsilon)
        for head in range(heads):
            write = gain[:, None] * output[:, head * head_dim:(head + 1) * head_dim]
            out[slot, :, head] = np.linalg.norm(values[:, head * head_dim:(head + 1) * head_dim] @ write.T, axis=-1) * scale
    return out


def activity_jaccard(
    model: torch.nn.Module,
    architecture: str,
    layers: list[int],
    python: dict[str, np.ndarray],
    browser: dict[str, np.ndarray],
) -> dict[str, float]:
    """Top-32 neurons by peak |act| and top-32 heads by peak write norm, Jaccard per kind."""
    out = {}
    for kind in ("neuron", "head"):
        sets = []
        for sites in (python, browser):
            if kind == "neuron":
                scores = np.abs(sites["act"].astype(np.float64)).max(axis=1)
            else:
                scores = head_write_norms(model, architecture, layers, sites["z"]).max(axis=1)
            sets.append(_top(scores, ACTIVITY_TOP))
        out[kind] = len(sets[0] & sets[1]) / len(sets[0] | sets[1])
    return out


def unit_parity(
    model: torch.nn.Module,
    architecture: str,
    units: dict[str, Any],
    browser_sites: dict[str, np.ndarray],
    layers: list[int],
    positions: list[int],
    input_ids: list[int],
    device: str,
) -> dict[str, Any]:
    """Every B1 §4.3 production measure for a units library; raises on a failed tolerance."""
    if units["block"] != block_shape(model):
        raise ValueError("browser units capture block shape does not match the model")
    python_sites = capture_torch_sites(model, architecture, layers, positions, input_ids, device)
    errors = site_relative_errors(python_sites, browser_sites, layers)
    worst = {site: max(errors[site].values()) for site in SITES}
    for site, value in worst.items():
        if value > SITE_TOLERANCE:
            raise ValueError(
                f"dequantized MLC q4 {site} site parity failed: relative error {value:.6g} exceeds {SITE_TOLERANCE}"
            )
    reads = static_read_parity(model, architecture, units["static_reads"])
    if reads["minOverlap"] < MINIMUM_STATIC_READ_OVERLAP:
        raise ValueError(
            f"dequantized MLC q4 static read parity failed: top-{STATIC_READ_TOP} overlap {reads['minOverlap']}"
            f" is below {MINIMUM_STATIC_READ_OVERLAP}"
        )
    if reads["maxRelativeError"] > STATIC_READ_TOLERANCE:
        raise ValueError(
            "dequantized MLC q4 static read parity failed: relative error "
            f"{reads['maxRelativeError']:.6g} exceeds {STATIC_READ_TOLERANCE}"
        )
    jaccard = activity_jaccard(model, architecture, layers, python_sites, browser_sites)
    least = min(jaccard.values())
    if least < MINIMUM_ACTIVITY_JACCARD:
        raise ValueError(
            f"dequantized MLC q4 unit activity parity failed: top-{ACTIVITY_TOP} Jaccard {least:.6g}"
            f" is below {MINIMUM_ACTIVITY_JACCARD}"
        )
    return {
        "maxActRelativeError": worst["act"],
        "maxZRelativeError": worst["z"],
        "maxMidRelativeError": worst["mid"],
        "maxStaticReadRelativeError": reads["maxRelativeError"],
        "minStaticReadTopOverlap": reads["minOverlap"],
        "minUnitActivityJaccard": least,
        "siteRelativeErrorByLayer": errors,
        "staticReads": reads["reads"],
        "unitActivityJaccard": jaccard,
    }


UNIT_RESULT_KEYS = (
    "maxActRelativeError",
    "maxZRelativeError",
    "maxMidRelativeError",
    "maxStaticReadRelativeError",
    "minStaticReadTopOverlap",
    "minUnitActivityJaccard",
)
