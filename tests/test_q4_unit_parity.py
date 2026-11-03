from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


_RUNTIME_ROOT = Path(__file__).parents[1] / "browser-runtime"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _RUNTIME_ROOT / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_MLC = _load("mlc_q4_torch")
_UNITS = _load("q4_unit_parity")
INPUT_IDS = [2, 5, 9, 11, 3, 17, 23, 8, 40, 1]
POSITIONS = [0, 4, 9]


def tiny_model(architecture: str) -> torch.nn.Module:
    """A tiny Transformers model in the q4 reference's form: MlcRMSNorm gains as the MLC tensor
    cache holds them (Gemma's `1 + w`)."""
    torch.manual_seed(11)
    common = {
        "vocab_size": 64, "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 3,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8, "rms_norm_eps": 1e-6,
        "max_position_embeddings": 64, "tie_word_embeddings": True,
    }
    if architecture == "qwen3":
        from transformers import Qwen3Config, Qwen3ForCausalLM

        model = Qwen3ForCausalLM(Qwen3Config(**common))
    else:
        from transformers import Gemma3ForCausalLM, Gemma3TextConfig

        model = Gemma3ForCausalLM(Gemma3TextConfig(
            **common, query_pre_attn_scalar=8, sliding_window=4,
            layer_types=["sliding_attention", "sliding_attention", "full_attention"],
        ))
    model.config._attn_implementation = "eager"
    with torch.no_grad():
        for module in model.modules():
            if type(module).__name__.endswith("RMSNorm"):
                module.weight.add_(torch.randn_like(module.weight) * 0.1)
                if architecture == "gemma3_text":
                    module.weight.add_(1)
                    module.variance_epsilon = module.eps
    _MLC.replace_rms_norms(model)
    model.requires_grad_(False)
    return model.eval()


@pytest.mark.parametrize("architecture", ["qwen3", "gemma3_text"])
def test_sites_are_the_block_equations(architecture: str) -> None:
    """`z` feeds o_proj, `mid = h + N_ao(W_O z)` and `act` feeds down_proj, with the block out
    `mid + N_mo(W_down act)` (B1 §1.1)."""
    model = tiny_model(architecture)
    layers = [0, 2]
    sites = _UNITS.capture_torch_sites(model, architecture, layers, POSITIONS, INPUT_IDS, "cpu")
    inputs, outputs = {}, {}
    handles = []
    for layer in layers:
        block = model.model.layers[layer]
        handles.append(block.register_forward_pre_hook(
            lambda _m, args, kwargs, layer=layer: inputs.__setitem__(layer, (args[0] if args else kwargs["hidden_states"]).detach()),
            with_kwargs=True,
        ))
        handles.append(block.register_forward_hook(
            lambda _m, _a, output, layer=layer: outputs.__setitem__(layer, (output[0] if isinstance(output, tuple) else output).detach()),
        ))
    with torch.inference_mode():
        model(torch.tensor([INPUT_IDS]), use_cache=False)
    for handle in handles:
        handle.remove()
    for slot, layer in enumerate(layers):
        block = model.model.layers[layer]
        z, act, mid = (torch.from_numpy(sites[name][slot]) for name in ("z", "act", "mid"))
        attention_output = block.self_attn.o_proj(z)
        mlp_output = block.mlp.down_proj(act)
        if architecture == "gemma3_text":
            attention_output = block.post_attention_layernorm(attention_output)
            mlp_output = block.post_feedforward_layernorm(mlp_output)
        torch.testing.assert_close(inputs[layer][0, POSITIONS] + attention_output, mid, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(mid + mlp_output, outputs[layer][0, POSITIONS], rtol=1e-5, atol=1e-6)
        assert sites["act"].shape[-1] == 32 and sites["z"].shape[-1] == 16 and sites["mid"].shape[-1] == 16


@pytest.mark.parametrize("architecture", ["qwen3", "gemma3_text"])
def test_vocab_readout_is_the_softmax_of_the_final_norm_logits(architecture: str) -> None:
    model = tiny_model(architecture)
    row = _UNITS.neuron_write(model, architecture, 1, 5)
    block = model.model.layers[1]
    expected_row = block.mlp.down_proj.weight[:, 5].double()
    if architecture == "gemma3_text":
        expected_row = expected_row * block.post_feedforward_layernorm.weight.double()
    np.testing.assert_allclose(row, expected_row.numpy(), rtol=1e-12)
    ids, probabilities = _UNITS.vocab_readout_topk(model, row, 8)
    with torch.inference_mode():
        logits = model.lm_head(model.model.norm(torch.from_numpy(row).float()[None]))[0].double()
    expected = torch.softmax(logits, -1).numpy()
    np.testing.assert_allclose(probabilities, expected, rtol=1e-5)
    assert ids == [int(i) for i in np.argsort(-expected, kind="stable")[:8]]


def browser_capture(model, architecture, layers, perturb=0.0):
    sites = _UNITS.capture_torch_sites(model, architecture, layers, POSITIONS, INPUT_IDS, "cpu")
    rng = np.random.default_rng(3)
    browser = {
        name: (value * (1 + perturb * rng.standard_normal(value.shape))).astype(np.float32)
        for name, value in sites.items()
    }
    reads = []
    for layer in range(len(model.model.layers)):
        ids, probabilities = _UNITS.vocab_readout_topk(model, _UNITS.neuron_write(model, architecture, layer, 3), 8)
        reads.append({
            "layer": layer, "neuron": 3, "ids": ids,
            "probabilities": [float(np.float32(probabilities[i])) for i in ids],
        })
    units = {"block": _UNITS.block_shape(model), "sites_sha256": "0" * 64, "static_reads": reads}
    return units, browser


@pytest.mark.parametrize("architecture", ["qwen3", "gemma3_text"])
def test_unit_parity_reports_every_production_measure(architecture: str) -> None:
    model = tiny_model(architecture)
    layers = [0, 1, 2]
    units, browser = browser_capture(model, architecture, layers, perturb=1e-4)
    _UNITS.validate_units_member(units, layers=layers, num_layers=3)
    width = _UNITS.site_width(units["block"], 16)
    packed = np.concatenate([browser["act"], browser["z"], browser["mid"]], axis=-1)
    assert packed.shape[-1] == width
    split = _UNITS.split_sites(packed, units["block"], 16)
    result = _UNITS.unit_parity(model, architecture, units, split, layers, POSITIONS, INPUT_IDS, "cpu")
    assert set(_UNITS.UNIT_RESULT_KEYS) <= set(result)
    assert 0 < result["maxActRelativeError"] < 1e-3
    assert result["minStaticReadTopOverlap"] == 8
    assert result["maxStaticReadRelativeError"] < 1e-6
    assert result["minUnitActivityJaccard"] >= 0.9


@pytest.mark.parametrize("site", ["act", "z", "mid"])
def test_unit_parity_refuses_a_site_off_by_more_than_one_percent(site: str) -> None:
    model = tiny_model("qwen3")
    layers = [0, 2]
    units, browser = browser_capture(model, "qwen3", layers)
    browser[site] = browser[site] * 1.05
    with pytest.raises(ValueError, match=f"{site} site parity failed"):
        _UNITS.unit_parity(model, "qwen3", units, browser, layers, POSITIONS, INPUT_IDS, "cpu")


def test_unit_parity_refuses_static_reads_that_disagree() -> None:
    model = tiny_model("gemma3_text")
    layers = [1]
    units, browser = browser_capture(model, "gemma3_text", layers)
    # Two ids outside Python's top 8: overlap 6.
    read = units["static_reads"][0]
    outside = [token for token in range(64) if token not in read["ids"]][:2]
    read["ids"][-2:] = outside
    with pytest.raises(ValueError, match="overlap 6 is below 7"):
        _UNITS.unit_parity(model, "gemma3_text", units, browser, layers, POSITIONS, INPUT_IDS, "cpu")
    units, browser = browser_capture(model, "gemma3_text", layers)
    units["static_reads"][0]["probabilities"][0] *= 1.01
    with pytest.raises(ValueError, match="static read parity failed: relative error"):
        _UNITS.unit_parity(model, "gemma3_text", units, browser, layers, POSITIONS, INPUT_IDS, "cpu")


def test_activity_jaccard_counts_the_ranked_sets() -> None:
    model = tiny_model("qwen3")
    layers = [0, 1, 2]
    _, python = browser_capture(model, "qwen3", layers)
    swapped = {name: value.copy() for name, value in python.items()}
    # Silence the browser's five strongest neurons: five of the top 32 change.
    strongest = np.argsort(-np.abs(python["act"]).max(axis=1).reshape(-1), kind="stable")[:5]
    flat = swapped["act"].reshape(len(layers), len(POSITIONS), -1)
    for index in strongest:
        flat[index // 32, :, index % 32] = 0
    jaccard = _UNITS.activity_jaccard(model, "qwen3", layers, python, swapped)
    assert jaccard["neuron"] == pytest.approx(27 / 37)
    assert jaccard["head"] == 1


def test_units_member_is_validated() -> None:
    good = {
        "block": {"d_ff": 32, "n_heads": 2, "head_dim": 8}, "sites_sha256": "a" * 64,
        "static_reads": [{"layer": 0, "neuron": 1, "ids": list(range(8)), "probabilities": [0.5] + [0.05] * 7}],
    }
    assert _UNITS.validate_units_member(good, layers=[0], num_layers=2) is good
    for broken in (
        {**good, "extra": 1},
        {**good, "static_reads": []},
        {**good, "static_reads": [{**good["static_reads"][0], "neuron": 32}]},
        {**good, "static_reads": [{**good["static_reads"][0], "ids": [0] * 8}]},
        {**good, "static_reads": [{**good["static_reads"][0], "probabilities": [0.05] * 7 + [0.5]}]},
    ):
        with pytest.raises(ValueError):
            _UNITS.validate_units_member(broken, layers=[0], num_layers=2)
    with pytest.raises(ValueError, match="every captured layer"):
        _UNITS.validate_units_member(good, layers=[0, 1], num_layers=2)


def test_units_follow_the_overlay_architectures() -> None:
    assert _UNITS.has_unit_sites("qwen3") and _UNITS.has_unit_sites("gemma3_text")
    assert not any(_UNITS.has_unit_sites(name) for name in ("qwen3_5", "gpt2", "gpt_neox"))
