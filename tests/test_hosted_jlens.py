from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "browser-runtime"))

from fit_hosted_jlens import (
    _capture_rows,
    _load_words,
    _stabilize_tensor_names,
    dense_linears,
    fit_capture,
    residual_parity,
    token_rows_sha256,
)
from mlc_q4_torch import MlcLinear, MlcRMSNorm, replace_linears, replace_rms_norms


def test_capture_rows_require_one_contiguous_exact_closure() -> None:
    rows = _capture_rows(
        [
            {
                "input_ids": list(range(18)),
                "byte_offset": 0,
                "byte_length": 2 * 18 * 3 * 4,
            },
            {
                "input_ids": list(range(20)),
                "byte_offset": 2 * 18 * 3 * 4,
                "byte_length": 2 * 20 * 3 * 4,
            },
        ],
        2 * (18 + 20) * 3 * 4,
        layer_count=2,
        hidden_size=3,
        sequence_limit=20,
        skip_first=16,
    )
    assert [row["byte_offset"] for row in rows] == [0, 432]


@pytest.mark.parametrize(
    "field,value",
    [
        ("byte_offset", 4),
        ("byte_length", 8),
        ("input_ids", list(range(17))),
    ],
)
def test_capture_rows_reject_invalid_extent_or_short_prompt(
    field: str, value: object,
) -> None:
    row = {
        "input_ids": list(range(18)),
        "byte_offset": 0,
        "byte_length": 2 * 18 * 3 * 4,
    }
    row[field] = value
    with pytest.raises(ValueError, match="row 0"):
        _capture_rows(
            [row],
            2 * 18 * 3 * 4,
            layer_count=2,
            hidden_size=3,
            sequence_limit=20,
            skip_first=16,
        )


def test_words_must_be_unique_trimmed_strings(tmp_path: Path) -> None:
    valid = tmp_path / "valid.json"
    valid.write_text('["ocean", "calm"]')
    assert _load_words(valid) == ["ocean", "calm"]
    invalid = tmp_path / "invalid.json"
    invalid.write_text('["ocean", "ocean"]')
    with pytest.raises(ValueError, match="unique trimmed"):
        _load_words(invalid)


def test_tensor_generation_names_are_stabilized(tmp_path: Path) -> None:
    (tmp_path / "random-a.safetensors").write_bytes(b"a")
    (tmp_path / "random-b.safetensors").write_bytes(b"b")
    manifest = {
        "format_version": 7,
        "sites": {
            "out": {
                "layers": [2, 4],
                "tensor_files": {"2": "random-a.safetensors", "4": "random-b.safetensors"},
                "tensor_sha256": {"2": "a" * 64, "4": "b" * 64},
            },
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    _stabilize_tensor_names(tmp_path)
    rewritten = json.loads((tmp_path / "manifest.json").read_text())
    assert rewritten["sites"]["out"]["tensor_files"] == {
        "2": "out.layer-2.safetensors",
        "4": "out.layer-4.safetensors",
    }
    assert rewritten["sites"]["out"]["tensor_sha256"] == {"2": "a" * 64, "4": "b" * 64}
    assert (tmp_path / "out.layer-2.safetensors").read_bytes() == b"a"
    assert (tmp_path / "out.layer-4.safetensors").read_bytes() == b"b"
    assert not (tmp_path / "random-a.safetensors").exists()


def test_stabilize_rejects_a_v6_manifest(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({
        "source_layers": [2],
        "tensor_files": {"2": "random-a.safetensors"},
    }))
    with pytest.raises(ValueError, match="invalid tensor pointers"):
        _stabilize_tensor_names(tmp_path)


def test_token_corpus_digest_is_drowses_token_ids_v1() -> None:
    from drowse.lens.lifecycle import _rows_sha256

    rows = [[1, 2, 3], [40, 50]]
    assert token_rows_sha256(rows) == _rows_sha256(rows)


def test_drowse_writer_produces_a_stable_v7_r_pack(tmp_path: Path) -> None:
    from drowse.base.integrity import hash_file
    from drowse.lens.jlens import JacobianLens
    from drowse.lens.key import LensKey
    from drowse.lens.store import LensProvenance, save_lens_at

    lens = JacobianLens(
        {0: torch.eye(3), 1: 2 * torch.eye(3)}, n_prompts=4, d_model=3, key=LensKey("R"),
    )
    root = tmp_path / "packs" / "jlens"
    save_lens_at(lens, root, LensProvenance(
        corpus_spec="fixture", corpus_sha256="1" * 64, seq_len=32, dim_batch=8,
        raw_corpus_sha256="2" * 64, raw_prompt_count=4, usable_prompt_count=4,
        model_layer_count=3, model_fingerprint="3" * 64, model_source_fingerprint="4" * 40,
    ), split_half=None)
    _stabilize_tensor_names(root)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["format_version"] == 7
    assert manifest["kind"] == "R"
    assert manifest["estimator_policy"]["direction"] == "output"
    assert manifest["checkpoint"] is False
    assert manifest["consumed_prefix_sha256"] is None
    assert "source_layers" not in manifest and "method" not in manifest
    out = manifest["sites"]["out"]
    assert set(out) == {"layers", "tensor_files", "tensor_sha256"}
    assert out["layers"] == [0, 1]
    assert out["tensor_files"] == {"0": "out.layer-0.safetensors", "1": "out.layer-1.safetensors"}
    for layer, name in out["tensor_files"].items():
        assert hash_file(root / name) == out["tensor_sha256"][layer]
    assert sorted(path.name for path in root.glob("*.safetensors")) == [
        "out.layer-0.safetensors", "out.layer-1.safetensors",
    ]


# -- the fit on drowse's transport estimator ---------------------------------


def _mlc_shaped_model() -> torch.nn.Module:
    """A tiny Qwen3 with the MLC modules ``build_transformers_model`` installs."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=64, tie_word_embeddings=True,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).eval()
    replace_rms_norms(model)
    replace_linears(model)
    model.requires_grad_(False)
    return model


def _rows() -> list[list[int]]:
    generator = torch.Generator().manual_seed(1)
    return [
        torch.randint(3, 64, (length,), generator=generator).tolist()
        for length in (20, 22, 19, 24)
    ]


@torch.no_grad()
def _captured(model: torch.nn.Module, rows: list[list[int]]) -> list[np.ndarray]:
    """What the browser capture holds: every block's output, every token."""
    outputs: list[torch.Tensor] = []

    def keep(_module: torch.nn.Module, _inputs: object, output: object) -> None:
        outputs.append((output[0] if isinstance(output, tuple) else output)[0])

    handles = [layer.register_forward_hook(keep) for layer in model.model.layers]
    captured = []
    try:
        for row in rows:
            outputs.clear()
            model(torch.tensor([row]), use_cache=False)
            captured.append(torch.stack(outputs).float().numpy())
    finally:
        for handle in handles:
            handle.remove()
    return captured


def test_dense_linears_restore_describable_projections() -> None:
    from drowse.model.blocks import verify_model_spec

    model = _mlc_shaped_model()
    assert any(isinstance(module, MlcLinear) for module in model.modules())
    assert any(isinstance(module, MlcRMSNorm) for module in model.modules())
    dense_linears(model)
    assert not any(isinstance(module, MlcLinear) for module in model.modules())
    assert model.lm_head.weight is model.model.embed_tokens.weight
    spec = verify_model_spec(model)
    assert spec.n_layers == 3 and spec.d_model == 32


def test_fit_capture_runs_drowses_estimator_on_the_captured_rows() -> None:
    from drowse.lens.fit import fit_lens
    from drowse.model.blocks import verify_model_spec

    model = _mlc_shaped_model()
    dense_linears(model)
    spec = verify_model_spec(model)
    rows = _rows()
    captured = _captured(model, rows)
    fitted, parity = fit_capture(
        model, spec, rows, captured.__getitem__,
        kind="R", source_layers=[0, 1], dim_batch=16, seq_len=32, skip_first=16,
        max_residual_error=0.01,
    )
    assert sorted(parity) == [0, 1, 2]
    assert max(parity.values()) < 1e-5
    lens = fitted.lens
    assert lens.kind == "R" and lens.sites == ("out",)
    assert lens.layers("out") == [0, 1]
    assert lens.n_prompts == len(rows)
    assert sorted(fitted.split_half) == ["out"]
    assert sorted(fitted.split_half["out"]) == [0, 1]
    direct = fit_lens(
        model, None, ["row"] * len(rows), kind="R", sites=("out",),
        layers={"out": [0, 1]}, dim_batch=16, max_seq_len=32,
        input_id_rows=rows, spec=spec,
    ).lens
    for layer in (0, 1):
        assert torch.equal(lens.matrix(layer), direct.matrix(layer))


def test_fit_capture_refuses_a_model_that_does_not_reproduce_the_browser() -> None:
    from drowse.model.blocks import verify_model_spec

    model = _mlc_shaped_model()
    dense_linears(model)
    spec = verify_model_spec(model)
    rows = _rows()
    captured = _captured(model, rows)
    captured[2] = captured[2].copy()
    captured[2][1] *= 1.2
    with pytest.raises(ValueError, match="layer 1 relative error"):
        fit_capture(
            model, spec, rows, captured.__getitem__,
            kind="R", source_layers=[0, 1], dim_batch=16, seq_len=32, skip_first=16,
            max_residual_error=0.01,
        )


def test_residual_parity_reads_every_block_output() -> None:
    from drowse.model.blocks import verify_model_spec

    model = _mlc_shaped_model()
    dense_linears(model)
    spec = verify_model_spec(model)
    rows = _rows()[:1]
    captured = _captured(model, rows)
    with pytest.raises(ValueError, match="wrong residual shape"):
        residual_parity(model, spec, rows, lambda _index: captured[0][:2])
