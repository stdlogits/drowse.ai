"""The hosted pack builder against drowse's tiny ABI v5 fixture: every shipped
tensor is the session's own (and the unit-artifacts golden's), every binding
names the identity it averages over."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "browser-runtime"))
sys.path.insert(0, str(ROOT / "drowse"))

from build_hosted_packs import (  # noqa: E402
    KINDS,
    NativeModel,
    PackBuildError,
    build,
    mask_words,
    sha256_file,
    split_population,
    unpack_mask_words,
)

GOLDENS = ROOT / "drowse" / "tests" / "fixtures" / "browser"


def runtime_record(model: NativeModel) -> dict:
    identity = {
        "sourceModel": "fixture/units-tiny",
        "sourceRevision": "f" * 40,
        "convertedManifestSha256": "0" * 64,
        "quantization": "q0f32",
        "tokenizerSha256": "1" * 64,
        "chatTemplateSha256": "2" * 64,
        "modelLibrarySha256": "3" * 64,
        "runtimeAbi": "drowse-web-runtime-v2",
        "hookAbi": "block-sites-v5",
        "hiddenSize": model.d_model,
        "layerMap": list(range(model.n_layers)),
    }
    return {"modelId": "units-tiny", "runtimeIdentity": identity, "runtimeIdentitySha256": "a" * 64}


def test_mask_words_set_bit_v_of_word_v_over_32() -> None:
    keep = torch.zeros(70, dtype=torch.bool)
    keep[[0, 31, 32, 69]] = True
    words = mask_words(keep)
    assert words.dtype == torch.uint32 and words.shape == (3,)
    assert [int(w) for w in words.to(torch.int64)] == [1 | (1 << 31), 1, 1 << 5]
    assert torch.equal(unpack_mask_words(words, 70), keep)


def test_split_population_keeps_vocabulary_statistics_apart() -> None:
    names = [
        "R.writes.rho", "R.writes.lse", "R.writes.mean", "R.writes.std", "S.fires.mean",
        "S.fires+.norm", "R.heads.ov.std", "heads.norm_derivative.query", "neurons.write_norms",
    ]
    reads, contrast = split_population({name: torch.zeros(1) for name in names})
    assert sorted(contrast) == ["R.heads.ov.std", "R.writes.mean", "R.writes.std", "S.fires.mean"]
    assert sorted(reads) == [
        "R.writes.lse", "R.writes.rho", "S.fires+.norm", "heads.norm_derivative.query", "neurons.write_norms",
    ]


@pytest.fixture(scope="module", params=["gemma3", "qwen3"])
def built(request, tmp_path_factory):
    from tests.support.browser_units import fixture

    with fixture(request.param) as fx:
        model = NativeModel(fx.session)
        runtime = runtime_record(model)
        out = tmp_path_factory.mktemp(request.param) / "packs"
        record = build(model, runtime, out, KINDS, full_depth=False, words=["a", "e", "t"])
        yield request.param, fx, model, runtime, out, record


def test_lens_packs_bind_the_runtime_and_ship_the_native_shards(built) -> None:
    from drowse.lens.store import lens_identity

    _, _, model, runtime, out, record = built
    for family, kind in (("R", "jlens"), ("S", "slens")):
        pack = out / kind / "packs" / kind
        manifest_text = (pack / "manifest.json").read_text()
        manifest = json.loads(manifest_text)
        native, _, native_identity = model.lens_source(family)
        assert manifest["kind"] == family
        assert manifest["model_fingerprint"] == runtime["runtimeIdentitySha256"]
        assert manifest["model_source_fingerprint"] == runtime["runtimeIdentity"]["sourceRevision"]
        bound = ("sites", "model_fingerprint", "model_source_fingerprint", "model_layer_count")
        assert manifest["model_layer_count"] == model.n_layers
        assert {k: v for k, v in manifest.items() if k not in bound} == {
            k: v for k, v in native.items() if k not in bound
        }
        assert record["packs"][kind]["lens_identity"] == lens_identity(manifest)
        assert record["packs"][kind]["native"]["lens_identity"] == native_identity
        for site, entry in manifest["sites"].items():
            assert entry["layers"] == native["sites"][site]["layers"]
            assert entry["tensor_sha256"] == native["sites"][site]["tensor_sha256"]
            for layer in entry["layers"]:
                assert entry["tensor_files"][str(layer)] == f"{site}.layer-{layer}.safetensors"
                tensors = load_file(pack / entry["tensor_files"][str(layer)])
                assert list(tensors) == [f"layer_{layer}"]
                lens = model.session.lens.resident() if family == "R" else model.session.lens.resident_input()
                assert torch.equal(tensors[f"layer_{layer}"], lens.matrix(layer, site).float().cpu())


def test_slens_pack_carries_the_sessions_embedding_statistics(built) -> None:
    from drowse.lens.token_counts import tokenizer_fingerprint

    _, _, model, runtime, out, _ = built
    pack = out / "slens" / "packs" / "slens"
    record = json.loads((pack / "embedding-statistics.json").read_text())
    tensors = load_file(pack / record["tensor_file"])
    emb = model.session.lens.token_embeddings()
    raw = (pack / record["mask_file"]).read_bytes()
    assert len(raw) == 4 * record["mask_words"]
    words = torch.from_numpy(np.frombuffer(raw, dtype="<u4").astype(np.int64))
    assert torch.equal(unpack_mask_words(words, record["vocab_rows"]), emb.keep.cpu())
    assert torch.equal(tensors["mean_E"], emb.mean.cpu())
    assert torch.equal(tensors["sigma_E"], emb.cov.cpu())
    assert record["mask"] == emb.describe()
    assert record["tokenizer_fingerprint"] == tokenizer_fingerprint(model.session.tokenizer)
    assert record["tokenizer_sha256"] == runtime["runtimeIdentity"]["tokenizerSha256"]
    assert record["embedding_scale"] == pytest.approx(emb.scale)


def test_units_pack_is_the_unit_artifacts_golden(built) -> None:
    from tests.support.browser_units import decode

    arch, _, model, _, out, record = built
    pack = out / "units" / "packs" / "units"
    units = json.loads((pack / "units.json").read_text())
    assert units["binding"]["r_lens_identity"] == record["packs"]["jlens"]["lens_identity"]
    assert units["binding"]["s_lens_identity"] == record["packs"]["slens"]["lens_identity"]
    assert units["binding"]["embedding_statistics_sha256"] == record["packs"]["slens"]["embedding_statistics"]["sha256"]
    golden = json.loads((GOLDENS / f"unit-artifacts-{arch}-v1.json").read_text())
    tolerance = golden["tolerance"]["python"]
    assert len(units["layers"]) == len(golden["layers"]) == model.n_layers
    for entry in golden["layers"]:
        shipped = {}
        for part in ("reads", "contrast"):
            spec = units["layers"][str(entry["layer"])][part]
            tensors = load_file(pack / spec["file"])
            assert sorted(tensors) == sorted(spec["shapes"])
            assert all(value.ndim <= 2 for value in tensors.values())
            shipped.update({name: value.reshape(spec["shapes"][name]) for name, value in tensors.items()})
        assert sorted(shipped) == sorted(entry["tensors"])
        for name, value in entry["tensors"].items():
            np.testing.assert_allclose(
                shipped[name].numpy(), decode(value).astype(np.float32),
                atol=tolerance["atol"], rtol=tolerance["rtol"], err_msg=f"L{entry['layer']} {name}",
            )
    null = load_file(pack / units["null"]["file"])["null_basis"].double()
    assert units["null"]["rank"] == golden["null"]["rank"]
    np.testing.assert_allclose((null @ null.T).numpy(), decode(golden["null"]["projector"]), atol=1e-5)
    emb = model.session.lens.token_embeddings()
    kept = emb.keep.nonzero().squeeze(1)
    expected = kept[kept < units["head_kept"]["unembedding_rows"]]
    assert expected.tolist() == decode(golden["head_kept_ids"]).tolist()
    assert units["head_kept"]["count"] == expected.numel()


def test_core_statistics_stack_the_neutral_pass(built) -> None:
    _, fx, model, runtime, out, _ = built
    record = json.loads((out / "core" / "neutral-statistics.json").read_text())
    tensors = {
        name: value.reshape(record["shapes"][name])
        for name, value in load_file(out / "core" / record["tensor_file"]).items()
    }
    stats = fx.session.units.runtime.stats()
    assert record["n_documents"] == stats.n_documents
    for layer in range(model.n_layers):
        assert torch.equal(tensors["neuron_scale"][layer], stats.layer(layer).neuron_scale)
        assert torch.equal(tensors["head_mean"][layer], stats.layer(layer).head_mean)
        assert torch.equal(tensors["mid_mean"][layer], fx.session._neutral.site_means("mid")[layer])
        assert torch.equal(tensors["out_mean"][layer], fx.session.layer_means[layer])


def test_core_rows_are_the_whiteners_and_units_bind_them(built) -> None:
    """The core builder fits the core pack's whitener from these rows, so the units pack's
    write norms and the browser's whitener share one metric; units bind the rows' digest."""
    from drowse.geometry.whitener import LayerWhitener

    _, fx, model, _, out, record = built
    core = json.loads((out / "core" / "neutral-statistics.json").read_text())
    rows_file = out / "core" / core["neutral_rows"]["file"]
    assert core["neutral"]["rows_sha256"] == sha256_file(rows_file)
    rows = load_file(rows_file)
    assert sorted(rows) == sorted(f"layer_{layer}" for layer in range(model.n_layers))
    rebuilt = LayerWhitener.from_neutral_activations(
        {layer: rows[f"layer_{layer}"] for layer in range(model.n_layers)}, fx.session.layer_means,
    )
    session = fx.session._neutral.whitener()
    probe = torch.linspace(-1.0, 1.0, model.d_model)
    for layer in range(model.n_layers):
        assert rows[f"layer_{layer}"].shape[0] == core["neutral_rows"]["rows"]
        assert rebuilt.mahalanobis_norm(layer, probe) == pytest.approx(session.mahalanobis_norm(layer, probe), rel=1e-6)
    units = json.loads((out / "units" / "packs" / "units" / "units.json").read_text())
    assert units["binding"]["neutral"] == core["neutral"] == record["packs"]["core"]["neutral"]


def test_units_need_the_core_statistics_and_the_write_norms(built, tmp_path, monkeypatch) -> None:
    _, _, model, runtime, _, _ = built
    with pytest.raises(PackBuildError, match="core statistics"):
        build(model, runtime, tmp_path / "a", ["jlens", "slens", "units"], full_depth=False, words=["a"])
    reads = model.session.units.reads
    original = reads.population_artifacts

    def without_write_norms(layer):
        return {k: v for k, v in original(layer).items() if k != "neurons.write_norms"}

    monkeypatch.setattr(reads, "population_artifacts", without_write_norms)
    with pytest.raises(PackBuildError, match="write_norms"):
        build(model, runtime, tmp_path / "b", KINDS, full_depth=False, words=["a"])
    assert not list(tmp_path.iterdir())


def test_release_packs_refuse_a_lens_short_of_full_depth(built, tmp_path) -> None:
    _, _, model, runtime, _, _ = built
    with pytest.raises(PackBuildError, match="every layer|shape"):
        build(model, runtime, tmp_path / "packs", ["jlens"])
    assert not list(tmp_path.iterdir())
