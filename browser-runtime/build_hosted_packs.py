#!/usr/bin/env python3
"""The hosted lens, units and core-statistics packs, built from drowse's own artifacts.

drowse.ai reads the model the browser runs.  The native artifacts a pack carries
(fit_lens R and S lenses at ``out`` and ``mid``, the corpus token counts, the
neutral pass) are fitted by drowse on the dequantized browser weights in fp32
(``browser-runtime/README.md``, "Lens, units and core packs"); this builder turns
them into browser packs bound to one runtime identity, without refitting:

- ``jlens`` (the R family): every layer of R at ``out`` (L0..n-2) and ``mid``
  (L0..n-1), one shard per (site, layer), plus the curated browser vocabulary;
- ``slens`` (the S family): every layer of S at both sites, plus the S embedding
  statistics (token mask bitmap, ``mean_E``, ``Σ_E``, mask description and the
  counting tokenizer's fingerprint);
- ``units``: per layer the unit reads (population scalars, head-norm derivatives,
  whitened write norms) and the contrast statistics, both from
  ``UnitReader.population_artifacts``; the null basis model-wide;
- ``core``: the neutral statistics a core pack adds (unit statistics, ``mid`` and
  ``out`` means), from the same neutral pass.

The packs bind the runtime identity they are built for (``--runtime-identity``,
written by ``webui/scripts/hosted-runtime-identity.mjs``) and, for units, the
identities of the lens packs and statistics they average over.  Lens shards are
the native files byte for byte, so a pack shard's digest is the native manifest's.
Output goes to a staging directory; ``build-hosted-packs.sh`` assembles and
validates it.  Nothing is published from here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

FORMAT = "drowse-hosted-packs-v1"
LENS_SITES = ("out", "mid")
FAMILY_KIND = {"R": "jlens", "S": "slens"}
EMBEDDING_STATISTICS = "embedding-statistics"
UNITS_MODEL = "units-model.safetensors"
CORE_STATISTICS = "neutral-statistics"
VOCABULARY_FORMAT = "drowse-jlens-vocabulary-v2"
WORD_SEEDS = Path(__file__).resolve().parent / "release-corpus" / "jlens-word-seeds.txt"
# population_artifacts keys whose last axis is the vocabulary (or the head-read
# vocabulary): the contrast statistics ``hide_hubs`` needs, shipped apart from the
# per-unit scalars every unit read needs.
CONTRAST_SUFFIXES = (".mean", ".std")


class PackBuildError(RuntimeError):
    """The native artifacts do not make a complete, consistent pack."""


# --------------------------------------------------------------------------- utilities

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    """sha256 of sorted-key, compact JSON (Python's ``json.dumps``)."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_tensors(path: Path, tensors: Mapping[str, torch.Tensor]) -> tuple[str, dict[str, list[int]]]:
    """Save finite fp32 tensors as the browser's safetensors reader takes them (rank 1 or
    2: a deeper tensor is stored row-major as ``[prod(shape[:-1]), shape[-1]]``); returns
    the file's sha256 and each tensor's logical shape, which the pack's JSON records."""
    host: dict[str, torch.Tensor] = {}
    shapes: dict[str, list[int]] = {}
    for name, value in sorted(tensors.items()):
        value = value.detach().to("cpu", torch.float32).contiguous()
        if value.ndim == 0 or not bool(torch.isfinite(value).all()):
            raise PackBuildError(f"{path.name}: {name} is a scalar or not finite")
        shapes[name] = list(value.shape)
        host[name] = value.reshape(-1, value.shape[-1]) if value.ndim > 2 else value
    save_file(host, str(path))
    return sha256_file(path), shapes


def mask_words(keep: torch.Tensor) -> torch.Tensor:
    """``[V]`` bool → ``[ceil(V/32)]`` uint32 words, bit ``v & 31`` of word ``v >> 5``
    set when ``v`` is kept (the bitmap ``BlockSitesEmbeddingStatistics.mask`` reads)."""
    bits = keep.to("cpu", torch.bool).reshape(-1)
    words = math.ceil(bits.numel() / 32)
    padded = torch.zeros(words * 32, dtype=torch.int64)
    padded[: bits.numel()] = bits.to(torch.int64)
    weights = torch.tensor([1 << i for i in range(32)], dtype=torch.int64)
    packed = (padded.reshape(words, 32) * weights).sum(dim=1)
    return packed.to(torch.uint32)


def unpack_mask_words(words: torch.Tensor, rows: int) -> torch.Tensor:
    values = words.to(torch.int64).reshape(-1, 1)
    bits = (values >> torch.arange(32, dtype=torch.int64)) & 1
    return bits.reshape(-1)[:rows].to(torch.bool)


# --------------------------------------------------------------------------- identity

def load_runtime_identity(path: Path) -> dict[str, Any]:
    """The ``{modelId, runtimeIdentity, runtimeIdentitySha256}`` record
    ``hosted-runtime-identity.mjs`` writes (the catalog's own digest)."""
    record = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(record, dict) or set(record) != {"modelId", "runtimeIdentity", "runtimeIdentitySha256"}
        or not isinstance(record["runtimeIdentity"], dict)
        or not isinstance(record["runtimeIdentitySha256"], str) or len(record["runtimeIdentitySha256"]) != 64
    ):
        raise PackBuildError(f"{path} is not a runtime identity record")
    identity = record["runtimeIdentity"]
    for key in ("sourceModel", "sourceRevision", "hiddenSize", "layerMap", "quantization", "tokenizerSha256"):
        if key not in identity:
            raise PackBuildError(f"runtime identity lacks {key}")
    return record


class NativeModel:
    """A drowse session over the browser weights, with the artifacts its home holds."""

    def __init__(self, session: Any) -> None:
        from drowse.model.blocks import model_spec

        self.session = session
        self.model_id: str = session.model_id
        self.spec = model_spec(session.model)
        self.n_layers = len(self.spec.blocks)
        self.d_model = int(session.model.get_input_embeddings().weight.shape[1])

    # -- lenses --
    def lens_source(self, family: str) -> tuple[dict[str, Any], Path, str]:
        """The active local lens of ``family``: its manifest, its directory and the
        identity the session holds it under (checked equal to the manifest's)."""
        from drowse.lens.sources import LENS_SOURCES
        from drowse.lens.store import lens_identity, load_local_lens_sidecar

        ref = LENS_SOURCES.active(self.model_id, family)
        if ref is None or not ref.local:
            raise PackBuildError(f"{self.model_id} has no active local {family} lens")
        manifest = load_local_lens_sidecar(self.model_id, ref.name)
        if manifest is None:
            raise PackBuildError(f"{self.model_id} local:{ref.name} has no durable manifest")
        identity = lens_identity(manifest)
        runtime = self.session.lens.runtime if family == "R" else self.session.lens.input_runtime
        resident = self.session.lens.resident() if family == "R" else self.session.lens.resident_input()
        if resident is None or runtime.identity != identity:
            raise PackBuildError(f"the session's {family} lens is not local:{ref.name}'s manifest")
        return manifest, LENS_SOURCES.local_dir(self.model_id, ref.name), identity

    def neutral_identity(self) -> dict[str, Any]:
        from drowse.store.neutral import neutral_cache_identity, validate_neutral_cache_metadata

        self.session.layer_means  # the neutral pass, loaded (or made) as every read makes it
        identity = neutral_cache_identity(validate_neutral_cache_metadata(self.model_id))
        return {"capture_sha256": identity["capture_sha256"], "sha256": canonical_digest(identity),
                "n_prompts": identity["n_prompts"]}


def expected_layers(family: str, site: str, n_layers: int) -> list[int]:
    """Full depth (PLAN.md): R stops one short of the last block at ``out``,
    whose read is the logit lens; everything else reads every layer."""
    last = n_layers - 1 if (family == "R" and site == "out") else n_layers
    return list(range(last))


# --------------------------------------------------------------------------- lens packs

def build_lens_pack(
    model: NativeModel, family: str, runtime: Mapping[str, Any], root: Path, *, full_depth: bool = True,
    words: Iterable[str] | None = None,
) -> dict[str, Any]:
    """``<root>/packs/<kind>/``: the native manifest bound to the runtime identity
    and every (site, layer) shard; returns the pack's binding summary.  A release
    pack is full depth (PLAN.md); ``full_depth=False`` takes the lens's layers as
    fitted (the tiny fixture's sparse lenses)."""
    from drowse.lens.store import lens_identity

    kind = FAMILY_KIND[family]
    manifest, directory, native_identity = model.lens_source(family)
    identity = runtime["runtimeIdentity"]
    if manifest.get("kind") != family or manifest.get("format_version") != 7:
        raise PackBuildError(f"the {family} source is not a v7 {family} lens")
    if manifest.get("checkpoint") is not False or manifest.get("dtype") != "float32":
        raise PackBuildError(f"the {family} lens is a checkpoint or not fp32")
    # A lens from `drowse lens fit` records its model's depth; a hand-made one (the tiny
    # fixture's) may not, and only outside a release is that filled from the runtime.
    depth = manifest["model_layer_count"]
    if manifest["d_model"] != identity["hiddenSize"] or (
        depth != len(identity["layerMap"]) and (full_depth or depth is not None)
    ):
        raise PackBuildError(f"the {family} lens does not match the runtime's shape")
    if model.n_layers != len(identity["layerMap"]):
        raise PackBuildError("the loaded model's depth differs from the runtime identity")
    pack = root / "packs" / kind
    pack.mkdir(parents=True)
    sites: dict[str, Any] = {}
    for site in LENS_SITES:
        native = manifest["sites"].get(site)
        layers = expected_layers(family, site, model.n_layers)
        if native is None or (full_depth and list(native["layers"]) != layers):
            raise PackBuildError(f"the {family} lens does not cover {site} at every layer ({layers[0]}..{layers[-1]})")
        if not set(native["layers"]) <= set(layers):
            raise PackBuildError(f"the {family} lens has {site} layers outside the model")
        layers = list(native["layers"])
        files: dict[str, str] = {}
        digests: dict[str, str] = {}
        for layer in layers:
            source = directory / native["tensor_files"][str(layer)]
            name = f"{site}.layer-{layer}.safetensors"
            shutil.copyfile(source, pack / name)
            digest = sha256_file(pack / name)
            if digest != native["tensor_sha256"][str(layer)]:
                raise PackBuildError(f"{source} does not match its manifest digest")
            files[str(layer)] = name
            digests[str(layer)] = digest
        sites[site] = {"layers": layers, "tensor_files": files, "tensor_sha256": digests}
        if "split_half" in native:
            sites[site]["split_half"] = dict(native["split_half"])
    bound = {
        **manifest,
        "sites": sites,
        "model_fingerprint": runtime["runtimeIdentitySha256"],
        "model_source_fingerprint": identity["sourceRevision"],
        "model_layer_count": len(identity["layerMap"]),
    }
    manifest_text = json.dumps(bound, indent=2, sort_keys=True) + "\n"
    (pack / "manifest.json").write_text(manifest_text, encoding="utf-8")
    pack_identity = lens_identity(json.loads(manifest_text))
    summary = {
        "format": FORMAT,
        "kind": kind,
        "family": family,
        "runtime_identity_sha256": runtime["runtimeIdentitySha256"],
        "lens_identity": pack_identity,
        "native": {
            "lens_identity": native_identity,
            "model_fingerprint": manifest["model_fingerprint"],
            "model_source_fingerprint": manifest["model_source_fingerprint"],
        },
        "sites": {site: {"layers": len(value["layers"])} for site, value in sites.items()},
    }
    if family == "R":
        summary["vocabulary"] = build_browser_vocabulary(model, pack, identity["hiddenSize"], words)
    else:
        summary["embedding_statistics"] = build_embedding_statistics(model, pack, runtime)
    write_json(pack / "pack.json", summary)
    return summary


def lens_words() -> list[str]:
    words = [line.strip() for line in WORD_SEEDS.read_text(encoding="utf-8").splitlines()]
    return [word for word in words if word and not word.startswith("#")]


def single_token_id(tokenizer: Any, word: str) -> int | None:
    for candidate in (f" {word}", word):
        token_ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(token_ids) == 1 and tokenizer.decode(token_ids).strip() == word:
            return int(token_ids[0])
    return None


def build_browser_vocabulary(
    model: NativeModel, pack: Path, hidden_size: int, words_in: Iterable[str] | None = None,
) -> dict[str, Any]:
    """The R pack's curated words: each seed word that is one token, with its
    unembedding row of the browser weights (``browser-vocabulary.*``)."""
    unembedding = model.session.model.get_output_embeddings().weight.detach()
    rows: list[torch.Tensor] = []
    words: list[dict[str, Any]] = []
    seen: set[int] = set()
    for word in (lens_words() if words_in is None else words_in):
        token_id = single_token_id(model.session.tokenizer, word)
        if token_id is None or token_id in seen or token_id >= unembedding.shape[0]:
            continue
        seen.add(token_id)
        rows.append(unembedding[token_id].to("cpu", torch.float32))
        words.append({"word": word, "row": len(words), "token_id": token_id})
    if not words:
        raise PackBuildError("no lens seed word is a single token for this tokenizer")
    tensor = pack / "browser-vocabulary.safetensors"
    digest, _ = save_tensors(tensor, {"unembedding": torch.stack(rows)})
    write_json(pack / "browser-vocabulary.json", {
        "format": VOCABULARY_FORMAT,
        "hidden_size": hidden_size,
        "tensor_file": tensor.name,
        "tensor_sha256": digest,
        "words": words,
    })
    return {"words": len(words)}


def build_embedding_statistics(model: NativeModel, pack: Path, runtime: Mapping[str, Any]) -> dict[str, Any]:
    """B1 §5.1's S statistics, exactly the session's ``TokenEmbeddings``: the
    mask bitmap, ``mean_E``, ``Σ_E``, the mask description and the fingerprint of
    the tokenizer the counts were taken with (the browser refuses another)."""
    from drowse.lens.token_counts import tokenizer_fingerprint

    emb = model.session.lens.token_embeddings()
    keep = emb.keep.to("cpu", torch.bool)
    words = mask_words(keep)
    if not torch.equal(unpack_mask_words(words, keep.numel()), keep):
        raise PackBuildError("the mask bitmap does not round-trip")
    # The bitmap is raw little-endian u32 words (the browser's safetensors reader is fp32 only).
    mask_file = pack / f"{EMBEDDING_STATISTICS}.mask.u32"
    mask_file.write_bytes(words.to(torch.int64).numpy().astype("<u4").tobytes())
    tensor = pack / f"{EMBEDDING_STATISTICS}.safetensors"
    digest, _ = save_tensors(tensor, {
        "mean_E": emb.mean.to("cpu", torch.float32),
        "sigma_E": emb.cov.to("cpu", torch.float32),
    })
    describe = emb.describe()
    record = {
        "format": "drowse-s-embedding-statistics-v1",
        "runtime_identity_sha256": runtime["runtimeIdentitySha256"],
        "vocab_rows": int(keep.numel()),
        "d_emb": int(emb.d_emb),
        "embedding_scale": float(emb.scale),
        "mask": {"source": describe["source"], "detail": describe["detail"], "kept": int(describe["kept"])},
        "mask_sha256": hashlib.sha256(keep.to(torch.uint8).numpy().tobytes()).hexdigest(),
        "mask_file": mask_file.name,
        "mask_words": int(words.numel()),
        "mask_file_sha256": sha256_file(mask_file),
        "tokenizer_fingerprint": tokenizer_fingerprint(model.session.tokenizer),
        "tensor_file": tensor.name,
        "tensor_sha256": digest,
    }
    write_json(pack / f"{EMBEDDING_STATISTICS}.json", record)
    return {"sha256": canonical_digest(record), "kept": record["mask"]["kept"], "tensor_sha256": digest}


# --------------------------------------------------------------------------- units

def split_population(artifacts: Mapping[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """(reads, contrast): per-unit scalars, norm derivatives and write norms apart
    from the ``[V]`` contrast statistics only ``hide_hubs`` reads."""
    reads: dict[str, torch.Tensor] = {}
    contrast: dict[str, torch.Tensor] = {}
    for name, value in artifacts.items():
        family = name.split(".", 1)[0]
        target = contrast if family in ("R", "S") and name.endswith(CONTRAST_SUFFIXES) else reads
        target[name] = value
    return reads, contrast


def build_units_pack(
    model: NativeModel,
    runtime: Mapping[str, Any],
    root: Path,
    binding: Mapping[str, Any],
    layers: Iterable[int] | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    """``<root>/packs/units/``: per layer ``reads.layer-L`` and ``contrast.layer-L``
    (``UnitReader.population_artifacts``), the null basis, and ``units.json``,
    which binds the runtime, the R and S lens packs, the S embedding statistics
    and the neutral pass every statistic averages over."""
    reads_api = model.session.units.reads
    pack = root / "packs" / "units"
    pack.mkdir(parents=True)
    layer_list = list(range(model.n_layers)) if layers is None else sorted(set(layers))
    if layer_list != list(range(model.n_layers)):
        raise PackBuildError("a units pack covers every layer")
    entries: dict[str, Any] = {}
    for layer in layer_list:
        artifacts = reads_api.population_artifacts(layer)
        reads, contrast = split_population(artifacts)
        entry: dict[str, Any] = {}
        for part, tensors in (("reads", reads), ("contrast", contrast)):
            if not tensors:
                raise PackBuildError(f"layer {layer} has no {part} artifacts")
            name = f"{part}.layer-{layer}.safetensors"
            digest, shapes = save_tensors(pack / name, tensors)
            entry[part] = {"file": name, "sha256": digest, "shapes": shapes}
        entries[str(layer)] = entry
        if progress is not None:
            progress(f"units layer {layer}: {len(reads)} reads, {len(contrast)} contrast tensors")
    null = reads_api.null_basis().to("cpu", torch.float32)
    head_kept = reads_api.head_kept_ids().to("cpu", torch.int64)
    emb = model.session.lens.token_embeddings()
    unembed_rows = int(model.session.model.get_output_embeddings().weight.shape[0])
    kept = emb.keep.to("cpu").nonzero().squeeze(1)
    if not torch.equal(head_kept, kept[kept < unembed_rows]):
        raise PackBuildError("the head-read vocabulary is not the mask's kept ids inside the unembedding")
    model_digest, _ = save_tensors(pack / UNITS_MODEL, {"null_basis": null})
    from drowse.units.reads import NULL_SHARE

    blocks = model.spec.blocks
    record = {
        "format": "drowse-units-pack-v1",
        "runtime_identity_sha256": runtime["runtimeIdentitySha256"],
        "binding": dict(binding),
        "model_layer_count": model.n_layers,
        "d_model": model.d_model,
        "d_ff": int(blocks[0].mlp.d_ff),
        "n_heads": int(blocks[0].attn.n_heads),
        "head_dim": int(blocks[0].attn.head_dim),
        "null": {"rank": int(null.shape[1]), "share": NULL_SHARE, "file": UNITS_MODEL,
                 "key": "null_basis", "sha256": model_digest},
        "head_kept": {"count": int(head_kept.numel()), "unembedding_rows": unembed_rows,
                      "rule": "the S mask's kept ids below the unembedding's rows"},
        "layers": entries,
    }
    write_json(pack / "units.json", record)
    return {
        "layers": len(entries),
        "reads_bytes": sum((pack / e["reads"]["file"]).stat().st_size for e in entries.values()),
        "contrast_bytes": sum((pack / e["contrast"]["file"]).stat().st_size for e in entries.values()),
    }


# --------------------------------------------------------------------------- core statistics

def build_core_statistics(model: NativeModel, runtime: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """``<root>/core/neutral-statistics.*``: the neutral pass's unit statistics
    and the ``out`` and ``mid`` means, stacked over layers, for the core pack."""
    neutral = model.session._neutral
    stats = model.session.units.runtime.stats()
    out_means = neutral.site_means("out")
    mid_means = neutral.site_means("mid")
    layers = list(range(model.n_layers))
    if sorted(out_means) != layers or sorted(mid_means) != layers or sorted(stats.layers) != layers:
        raise PackBuildError("the neutral pass does not cover every layer at both sites")

    def stack(get: Any) -> torch.Tensor:
        return torch.stack([get(layer).to("cpu", torch.float32) for layer in layers])

    tensors = {
        "neuron_mean": stack(lambda layer: stats.layer(layer).neuron_mean),
        "neuron_scale": stack(lambda layer: stats.layer(layer).neuron_scale),
        "head_mean": stack(lambda layer: stats.layer(layer).head_mean),
        "head_scale": stack(lambda layer: stats.layer(layer).head_scale),
        "mid_mean": stack(lambda layer: mid_means[layer]),
        "out_mean": stack(lambda layer: out_means[layer]),
    }
    core = root / "core"
    core.mkdir(parents=True)
    tensor = core / f"{CORE_STATISTICS}.safetensors"
    digest, shapes = save_tensors(tensor, tensors)
    neutral_identity = model.neutral_identity()
    record = {
        "format": "drowse-neutral-statistics-v1",
        "runtime_identity_sha256": runtime["runtimeIdentitySha256"],
        "model_layer_count": model.n_layers,
        "n_documents": int(stats.n_documents),
        "neutral": neutral_identity,
        "shapes": shapes,
        "tensor_file": tensor.name,
        "tensor_sha256": digest,
    }
    write_json(core / f"{CORE_STATISTICS}.json", record)
    return {"sha256": canonical_digest(record), "bytes": tensor.stat().st_size, "neutral": neutral_identity}


# --------------------------------------------------------------------------- driver

KINDS = ("jlens", "slens", "units", "core")


def check_weights(model: NativeModel, runtime: Mapping[str, Any], model_path: str) -> dict[str, Any]:
    """The loaded weights are the runtime's: an export's marker names the same
    source revision, quantization and tokenizer, and the loaded tokenizer is the
    lock's; a Hub model must be at the pinned revision of a q0f32 runtime."""
    identity = runtime["runtimeIdentity"]
    marker_path = Path(model_path) / "DROWSE-PARITY-EXPORT.json"
    if marker_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            marker.get("source", [None, None])[1] != identity["sourceRevision"]
            or marker.get("quantization") != identity["quantization"]
            or marker.get("tokenizer_sha256_lock") != identity["tokenizerSha256"]
            or marker.get("verified", {}).get("tokenizer_sha256") != identity["tokenizerSha256"]
        ):
            raise PackBuildError("the export marker does not match the runtime identity")
        return {"export": marker_path.name, "marker_sha256": sha256_file(marker_path)}
    if identity["quantization"] != "q0f32":
        raise PackBuildError("a quantized runtime's packs need its dequantized export (DROWSE-PARITY-EXPORT.json)")
    if model_path != identity["sourceModel"]:
        raise PackBuildError(f"{model_path} is not the runtime's source model {identity['sourceModel']}")
    from huggingface_hub import hf_hub_download

    tokenizer = Path(hf_hub_download(identity["sourceModel"], "tokenizer.json",
                                     revision=identity["sourceRevision"]))
    if sha256_file(tokenizer) != identity["tokenizerSha256"]:
        raise PackBuildError("the source tokenizer at the pinned revision is not the runtime's")
    return {"source": identity["sourceModel"], "revision": identity["sourceRevision"]}


def build(model: NativeModel, runtime: Mapping[str, Any], output: Path, kinds: Iterable[str],
          progress: Any = None, provenance: Mapping[str, Any] | None = None, *,
          full_depth: bool = True, words: Iterable[str] | None = None) -> dict[str, Any]:
    """Build ``kinds`` into ``output`` (which must not exist) through a partial
    directory renamed at the end; returns the build record."""
    kinds = list(dict.fromkeys(kinds))
    if any(kind not in KINDS for kind in kinds):
        raise PackBuildError(f"unknown pack kinds {kinds}")
    if "units" in kinds and not {"jlens", "slens"} <= set(kinds):
        raise PackBuildError("a units pack is built with the lens packs it binds")
    if output.exists():
        raise PackBuildError(f"{output} exists")
    partial = output.with_name(f"{output.name}.partial-{uuid.uuid4().hex}")
    partial.mkdir(parents=True)
    say = progress or (lambda message: None)
    try:
        record: dict[str, Any] = {
            "format": FORMAT,
            "model_id": runtime["modelId"],
            "runtime_identity_sha256": runtime["runtimeIdentitySha256"],
            "provenance": dict(provenance or {}),
            "packs": {},
        }
        lenses: dict[str, dict[str, Any]] = {}
        for family in ("R", "S"):
            kind = FAMILY_KIND[family]
            if kind in kinds:
                say(f"{kind}: copying {family} shards")
                lenses[family] = build_lens_pack(model, family, runtime, partial / kind, full_depth=full_depth, words=words)
                record["packs"][kind] = lenses[family]
        if "units" in kinds:
            binding = {
                "r_lens_identity": lenses["R"]["lens_identity"],
                "s_lens_identity": lenses["S"]["lens_identity"],
                "embedding_statistics_sha256": lenses["S"]["embedding_statistics"]["sha256"],
                "neutral": model.neutral_identity(),
            }
            say("units: population artifacts per layer")
            record["packs"]["units"] = build_units_pack(model, runtime, partial / "units", binding, progress=say)
        if "core" in kinds:
            say("core: neutral statistics")
            record["packs"]["core"] = build_core_statistics(model, runtime, partial)
        write_json(partial / "build.json", record)
        partial.rename(output)
        return record
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("model", help="the browser weights: an export directory, or the Hub id of a q0f32 runtime")
    parser.add_argument("--runtime-identity", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--kinds", default=",".join(KINDS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    if not os.environ.get("DROWSE_HOME"):
        raise SystemExit("set DROWSE_HOME to the home holding the native fits")
    runtime = load_runtime_identity(args.runtime_identity)
    import drowse
    from drowse.session.session import DrowseSession

    session = DrowseSession.from_pretrained(args.model, device=args.device, dtype="float32", probes=[])
    try:
        model = NativeModel(session)
        provenance = {
            "weights": check_weights(model, runtime, args.model),
            "drowse_version": drowse.__version__,
            "drowse_model_id": model.model_id,
        }
        record = build(model, runtime, args.output, args.kinds.split(","),
                       progress=lambda message: print(message, file=sys.stderr, flush=True),
                       provenance=provenance)
    finally:
        session.close()
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
