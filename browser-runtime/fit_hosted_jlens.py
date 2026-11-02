"""Fit a hosted R-family lens pack from a browser capture.

The capture (``build-hosted-jlens.mjs``) runs the release corpus through the
exact q4 browser runtime and keeps, per document, the token ids it rendered
and every block's output residual (``capture.json`` and ``residuals.f32``).

drowse 6.0's estimator is transport (:func:`drowse.lens.fit.fit_lens`): each
document's primal is recomputed from the model's weights through its
:class:`~drowse.model.blocks.ModelSpec`, and every block's map is written out
by hand, with no autograd and no forward hooks.  A captured residual
therefore cannot be the linearization point any more (5.x forward-corrected
each block onto the captured trajectory and took the dequantized graph's
derivative).  The data path is now:

- the fit runs on the dequantized MLC tensor cache (the same weights the
  browser runs), as plain ``nn.Linear`` projections so the block description
  reads real weights, over the captured token rows;
- the captured residuals are the parity gate: the dequantized model's block
  outputs must reproduce the browser's at every primal position, per layer,
  within ``max_residual_error`` (relative Frobenius error, the bound
  ``validate_mlc_q4_parity.py`` holds captures to), or the fit refuses.

The pack is lens format v7 at the ``out`` site, written by ``save_lens_at``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

from mlc_q4_torch import MlcLinear, build_transformers_model

from drowse.base.integrity import hash_file
from drowse.lens.fit import LensFit, fit_lens
from drowse.lens.jlens import DEFAULT_DIM_BATCH, SKIP_FIRST_POSITIONS, resolve_word_token
from drowse.lens.store import LensProvenance, save_lens_at
from drowse.model.blocks import ModelSpec, verify_model_spec

#: The output-family kinds a hosted pack may hold: the RelP R lens (drowse's
#: default) and the standard J lens.
OUTPUT_KINDS = ("R", "J")
#: The browser-parity bound on each layer's relative residual error.
MAX_RESIDUAL_ERROR = 0.01


def fit_pack(
    model_directory: Path,
    capture_directory: Path,
    words_path: Path,
    output_directory: Path,
    *,
    kind: str = "R",
    device: str,
    dim_batch: int = DEFAULT_DIM_BATCH,
    source_layers: list[int] | None,
    max_residual_error: float = MAX_RESIDUAL_ERROR,
) -> dict[str, object]:
    if output_directory.exists():
        raise ValueError(f"output directory already exists: {output_directory}")
    if kind not in OUTPUT_KINDS:
        raise ValueError(f"a hosted lens pack holds an R-family kind ({', '.join(OUTPUT_KINDS)})")
    if dim_batch <= 0:
        raise ValueError("dim_batch must be positive")
    metadata, rows, residuals = _load_capture(capture_directory)
    layers = metadata["layer_map"]
    hidden_size = metadata["hidden_size"]
    seq_len = metadata["seq_len"]
    skip_first = metadata["skip_first_positions"]

    model = build_transformers_model(model_directory, device)
    dense_linears(model)
    spec = verify_model_spec(model)
    if layers != list(range(spec.n_layers)) or hidden_size != spec.d_model:
        raise ValueError("the lens capture does not match the dequantized model shape")
    requested_layers = (
        list(range(spec.n_layers - 1))
        if source_layers is None
        else sorted(set(source_layers))
    )
    fitted, parity = fit_capture(
        model,
        spec,
        [row["input_ids"] for row in rows],
        lambda index: _row_residuals(residuals, rows[index], len(layers), hidden_size),
        kind=kind,
        source_layers=requested_layers,
        dim_batch=dim_batch,
        seq_len=seq_len,
        skip_first=skip_first,
        max_residual_error=max_residual_error,
        on_progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    lens = fitted.lens

    words = _load_words(words_path)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_directory), local_files_only=True, trust_remote_code=False,
    )
    vocabulary = []
    token_ids = []
    for row, word in enumerate(words):
        token_id = resolve_word_token(tokenizer, word)
        vocabulary.append({"word": word, "row": row, "token_id": token_id})
        token_ids.append(token_id)
    output_directory.mkdir(parents=True)
    try:
        root = output_directory / "packs" / "jlens"
        root.mkdir(parents=True)
        save_lens_at(
            lens,
            root,
            LensProvenance(
                corpus_spec=metadata["corpus_spec"],
                corpus_sha256=token_rows_sha256([row["input_ids"] for row in rows]),
                seq_len=seq_len,
                dim_batch=dim_batch,
                corpus_hash_kind="token_ids_v1",
                skip_first=skip_first,
                raw_corpus_sha256=metadata["raw_corpus_sha256"],
                raw_prompt_count=len(rows),
                usable_prompt_count=lens.n_prompts,
                model_layer_count=len(layers),
                model_fingerprint=metadata["runtime_identity_sha256"],
                model_source_fingerprint=metadata["model_source_fingerprint"],
            ),
            split_half=None,
        )
        shutil.rmtree(root / ".locks", ignore_errors=True)
        _stabilize_tensor_names(root)
        weight = getattr(model.get_output_embeddings(), "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise ValueError("dequantized model does not expose its language-head weight")
        selected = weight[token_ids].detach().to(device="cpu", dtype=torch.float32).contiguous()
        vocabulary_tensor = root / "browser-vocabulary.safetensors"
        save_file({"unembedding": selected}, vocabulary_tensor)
        (root / "browser-vocabulary.json").write_text(
            json.dumps(
                {
                    "format": "drowse-jlens-vocabulary-v2",
                    "hidden_size": hidden_size,
                    "tensor_file": vocabulary_tensor.name,
                    "tensor_sha256": hash_file(vocabulary_tensor),
                    "words": vocabulary,
                },
                indent=2,
            ) + "\n"
        )
    except BaseException:
        shutil.rmtree(output_directory, ignore_errors=True)
        raise
    return {
        "modelId": metadata["model_id"],
        "runtimeIdentitySha256": metadata["runtime_identity_sha256"],
        "kind": lens.kind,
        "family": "R",
        "site": "out",
        "prompts": lens.n_prompts,
        "layers": lens.layers("out"),
        "hiddenSize": lens.d_model,
        "words": len(words),
        "splitHalf": {str(layer): value for layer, value in fitted.split_half.get("out", {}).items()},
        "residualRelativeError": {str(layer): value for layer, value in parity.items()},
        "outputDirectory": str(output_directory),
    }


def fit_capture(
    model: Any,
    spec: ModelSpec,
    input_id_rows: Sequence[Sequence[int]],
    reference: Callable[[int], np.ndarray],
    *,
    kind: str,
    source_layers: Sequence[int],
    dim_batch: int,
    seq_len: int,
    skip_first: int,
    max_residual_error: float,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[LensFit, dict[int, float]]:
    """Gate the captured rows on residual parity, then fit ``kind`` at ``out``.

    ``reference(i)`` is document ``i``'s captured ``[layers, tokens, d]``
    block outputs.  Returns the fit and each layer's relative residual error.
    """
    if not (max_residual_error > 0 and np.isfinite(max_residual_error)):
        raise ValueError("max_residual_error must be finite and positive")
    parity = residual_parity(model, spec, input_id_rows, reference)
    worst = max(parity, key=parity.__getitem__)
    if parity[worst] > max_residual_error:
        raise ValueError(
            f"the dequantized model does not reproduce the browser residuals: layer {worst} "
            f"relative error {parity[worst]:.6g} exceeds {max_residual_error:.6g}"
        )
    fitted = fit_lens(
        model,
        None,
        [f"browser-capture-row-{index}" for index in range(len(input_id_rows))],
        kind=kind,
        sites=("out",),
        layers={"out": list(source_layers)},
        dim_batch=dim_batch,
        max_seq_len=seq_len,
        skip_first=skip_first,
        input_id_rows=[list(row) for row in input_id_rows],
        on_progress=on_progress,
        spec=spec,
    )
    if fitted.lens.n_prompts != len(input_id_rows):
        raise ValueError("the lens fit dropped a captured row; every row must be usable")
    return fitted, parity


@torch.no_grad()
def residual_parity(
    model: Any,
    spec: ModelSpec,
    input_id_rows: Sequence[Sequence[int]],
    reference: Callable[[int], np.ndarray],
) -> dict[int, float]:
    """Each layer's ``‖h_model − h_browser‖ / ‖h_browser‖`` over every row.

    Compared at the positions transport's primal keeps (all but the last
    token, which no read position attends to).
    """
    outputs: dict[int, torch.Tensor] = {}

    def keep(layer: int) -> Callable[..., None]:
        def hook(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
            value = output[0] if isinstance(output, tuple) else output
            outputs[layer] = value[0].detach().to(device="cpu", dtype=torch.float32)

        return hook

    device = spec.embed.weight.device
    error = torch.zeros(spec.n_layers, dtype=torch.float64)
    scale = torch.zeros(spec.n_layers, dtype=torch.float64)
    handles = [block.module.register_forward_hook(keep(block.layer)) for block in spec.blocks]
    try:
        for index, row in enumerate(input_id_rows):
            outputs.clear()
            ids = torch.tensor([list(row)], dtype=torch.long, device=device)
            model(ids, use_cache=False, logits_to_keep=1)
            captured = torch.from_numpy(np.asarray(reference(index), dtype=np.float32))
            if captured.shape != (spec.n_layers, len(row), spec.d_model):
                raise ValueError(f"lens capture row {index} has the wrong residual shape")
            kept = len(row) - 1
            for layer in range(spec.n_layers):
                actual = outputs[layer][:kept].double()
                expected = captured[layer, :kept].double()
                error[layer] += (actual - expected).square().sum()
                scale[layer] += expected.square().sum()
    finally:
        for handle in handles:
            handle.remove()
    return {
        layer: float((error[layer] / scale[layer].clamp_min(1e-24)).sqrt())
        for layer in range(spec.n_layers)
    }


def dense_linears(module: torch.nn.Module) -> None:
    """Swap each MLC linear back to ``nn.Linear`` on the same parameters.

    The MLC kernels reproduce the browser's fp16 accumulation for a forward
    pass; transport reads the projection weights directly, and the block
    description takes only dense ``nn.Linear`` projections.
    """
    for name, child in tuple(module.named_children()):
        if isinstance(child, MlcLinear):
            out_features, in_features = child.weight.shape
            linear = torch.nn.Linear(
                in_features, out_features, bias=child.bias is not None, device="meta",
            )
            linear.weight = child.weight
            if child.bias is not None:
                linear.bias = child.bias
            setattr(module, name, linear)
        else:
            dense_linears(child)


def token_rows_sha256(rows: Sequence[Sequence[int]]) -> str:
    """The ``token_ids_v1`` corpus identity, as drowse's lens fits write it."""
    return hashlib.sha256(repr([list(row) for row in rows]).encode("utf-8")).hexdigest()


def _load_capture(
    capture_directory: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.memmap]:
    metadata_path = capture_directory / "capture.json"
    residual_path = capture_directory / "residuals.f32"
    metadata = json.loads(metadata_path.read_text())
    required = {
        "schema_version", "model_id", "source_model_id",
        "model_source_fingerprint", "runtime_identity_sha256", "layer_map",
        "hidden_size", "seq_len", "skip_first_positions", "corpus_spec",
        "raw_corpus_sha256", "capture_plan_sha256", "residual_sha256", "rows",
    }
    if not isinstance(metadata, dict) or set(metadata) != required or metadata["schema_version"] != 1:
        raise ValueError("lens capture metadata has an invalid schema")
    layers = _integer_list(metadata["layer_map"], "layer_map")
    hidden_size = _positive_integer(metadata["hidden_size"], "hidden_size")
    seq_len = _positive_integer(metadata["seq_len"], "seq_len")
    skip_first = _nonnegative_integer(
        metadata["skip_first_positions"], "skip_first_positions",
    )
    if skip_first != SKIP_FIRST_POSITIONS:
        raise ValueError(
            f"lens capture uses skip_first_positions={skip_first}; "
            f"the lens format requires {SKIP_FIRST_POSITIONS}"
        )
    for field in (
        "model_id", "source_model_id", "model_source_fingerprint",
        "runtime_identity_sha256", "corpus_spec", "raw_corpus_sha256",
        "capture_plan_sha256", "residual_sha256",
    ):
        if not isinstance(metadata[field], str) or not metadata[field]:
            raise ValueError(f"lens capture has an invalid {field}")
    for field in (
        "runtime_identity_sha256", "raw_corpus_sha256",
        "capture_plan_sha256", "residual_sha256",
    ):
        if not _sha256(metadata[field]):
            raise ValueError(f"lens capture has an invalid {field}")
    if not _hex_digest(metadata["model_source_fingerprint"], 40):
        raise ValueError("lens capture has an invalid model_source_fingerprint")
    if hash_file(residual_path) != metadata["residual_sha256"]:
        raise ValueError("lens capture residual digest does not match")
    rows = _capture_rows(
        metadata["rows"],
        residual_path.stat().st_size,
        layer_count=len(layers),
        hidden_size=hidden_size,
        sequence_limit=seq_len,
        skip_first=skip_first,
    )
    plan = hashlib.sha256()
    for row in rows:
        plan.update(_canonical_json({"inputIds": row["input_ids"]}))
    if plan.hexdigest() != metadata["capture_plan_sha256"]:
        raise ValueError("lens capture plan digest does not match")
    return metadata, rows, np.memmap(residual_path, dtype="<f4", mode="r")


def _row_residuals(
    residuals: np.ndarray, row: dict[str, Any], layer_count: int, hidden_size: int,
) -> np.ndarray:
    count = len(row["input_ids"])
    start = row["byte_offset"] // 4
    stop = start + layer_count * count * hidden_size
    values = np.asarray(residuals[start:stop]).reshape(layer_count, count, hidden_size)
    if not np.isfinite(values).all():
        raise ValueError("lens capture residuals are non-finite")
    return values


def _capture_rows(
    value: object,
    file_bytes: int,
    *,
    layer_count: int,
    hidden_size: int,
    sequence_limit: int,
    skip_first: int,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("lens capture rows must be a non-empty list")
    rows = []
    expected_offset = 0
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {
            "input_ids", "byte_offset", "byte_length",
        }:
            raise ValueError(f"lens capture row {index} has an invalid schema")
        input_ids = item["input_ids"]
        if (
            not isinstance(input_ids, list)
            or len(input_ids) <= skip_first + 1
            or len(input_ids) > sequence_limit
            or any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in input_ids)
        ):
            raise ValueError(f"lens capture row {index} has invalid input IDs")
        offset = _nonnegative_integer(item["byte_offset"], f"rows[{index}].byte_offset")
        length = _positive_integer(item["byte_length"], f"rows[{index}].byte_length")
        expected_length = layer_count * len(input_ids) * hidden_size * 4
        if offset != expected_offset or length != expected_length:
            raise ValueError(f"lens capture row {index} has an invalid extent")
        rows.append({"input_ids": input_ids, "byte_offset": offset, "byte_length": length})
        expected_offset += length
    if expected_offset != file_bytes:
        raise ValueError("lens capture rows do not close the residual file")
    return rows


def _load_words(path: Path) -> list[str]:
    value = json.loads(path.read_text())
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(word, str) or not word or word.strip() != word for word in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError("lens words must be a non-empty JSON array of unique trimmed strings")
    return value


def _stabilize_tensor_names(root: Path) -> None:
    """Rename each site's generation shards to ``<site>.layer-<L>.safetensors``.

    The v7 writer names shards ``<site>.layer-<L>.gen-<uuid>``; a pack ships
    fixed names so a rebuild of the same lens is byte-identical.
    """
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    sites = manifest.get("sites")
    if not isinstance(sites, dict) or not sites:
        raise ValueError("the lens writer produced invalid tensor pointers")
    for site, entry in sites.items():
        tensor_files = entry.get("tensor_files") if isinstance(entry, dict) else None
        layers = entry.get("layers") if isinstance(entry, dict) else None
        if not isinstance(tensor_files, dict) or not isinstance(layers, list):
            raise ValueError(f"the lens writer produced invalid {site} tensor pointers")
        stable = {}
        for layer in layers:
            source_name = tensor_files.get(str(layer))
            if not isinstance(source_name, str):
                raise ValueError(f"the lens writer omitted {site} tensor layer {layer}")
            target_name = f"{site}.layer-{layer}.safetensors"
            os.replace(root / source_name, root / target_name)
            stable[str(layer)] = target_name
        entry["tensor_files"] = stable
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _integer_list(value: object, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in value)
        or any(value[index - 1] >= value[index] for index in range(1, len(value)))
    ):
        raise ValueError(f"lens capture has an invalid {label}")
    return value


def _positive_integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"lens capture has an invalid {label}")
    return value


def _nonnegative_integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"lens capture has an invalid {label}")
    return value


def _sha256(value: object) -> bool:
    return _hex_digest(value, 64)


def _hex_digest(value: object, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a hosted R-family lens pack (out site) from a browser capture.",
    )
    parser.add_argument("model_directory", type=Path)
    parser.add_argument("capture_directory", type=Path)
    parser.add_argument("words", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--kind", choices=OUTPUT_KINDS, default="R")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dim-batch", type=int, default=DEFAULT_DIM_BATCH)
    parser.add_argument("--source-layers", type=str)
    parser.add_argument("--max-residual-error", type=float, default=MAX_RESIDUAL_ERROR)
    arguments = parser.parse_args()
    source_layers = (
        None
        if arguments.source_layers is None
        else [int(value) for value in arguments.source_layers.split(",") if value]
    )
    result = fit_pack(
        arguments.model_directory,
        arguments.capture_directory,
        arguments.words,
        arguments.output_directory,
        kind=arguments.kind,
        device=arguments.device,
        dim_batch=arguments.dim_batch,
        source_layers=source_layers,
        max_residual_error=arguments.max_residual_error,
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
