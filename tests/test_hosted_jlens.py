from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "browser-runtime"))

from fit_hosted_jlens import _capture_rows, _load_words, _stabilize_tensor_names


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


def test_drowse_writer_produces_a_stable_v7_pack(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from drowse.base.integrity import hash_file
    from drowse.lens.jlens import JacobianLens
    from drowse.lens.store import LensProvenance, save_lens_at

    lens = JacobianLens({0: torch.eye(3), 1: 2 * torch.eye(3)}, n_prompts=4, d_model=3)
    root = tmp_path / "packs" / "jlens"
    save_lens_at(lens, root, LensProvenance(
        corpus_spec="fixture", corpus_sha256="1" * 64, seq_len=32, dim_batch=8,
        raw_corpus_sha256="2" * 64, raw_prompt_count=4, usable_prompt_count=4,
        model_layer_count=3, model_fingerprint="3" * 64, model_source_fingerprint="4" * 40,
    ))
    _stabilize_tensor_names(root)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["format_version"] == 7
    assert manifest["kind"] == "J"
    assert manifest["checkpoint"] is False
    assert manifest["consumed_prefix_sha256"] is None
    assert "source_layers" not in manifest and "method" not in manifest
    out = manifest["sites"]["out"]
    assert out["layers"] == [0, 1]
    assert out["tensor_files"] == {"0": "out.layer-0.safetensors", "1": "out.layer-1.safetensors"}
    for layer, name in out["tensor_files"].items():
        assert hash_file(root / name) == out["tensor_sha256"][layer]
    assert sorted(path.name for path in root.glob("*.safetensors")) == [
        "out.layer-0.safetensors", "out.layer-1.safetensors",
    ]
