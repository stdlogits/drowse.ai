#!/usr/bin/env python3
"""Build an MLC model directory from one of drowse's hook ABI v5 fixture checkpoints.

The checkpoint is `write_checkpoint(arch, path)` from drowse's `tests/support/browser_units.py`
(the generator behind every `*-v1.json` units golden). This script binds it to the goldens'
model file (`units-tiny-models-v1.json`: every tensor's fp32 sha256), maps the recorded HF
config to MLC's config (names and the Gemma sliding pattern MLC cannot read from `layer_types`),
converts the weights with MLC's own converter, and proves the result exact: q0f32 tensors equal
the source bit for bit, and every q4f32_1 tensor dequantizes to the source bit for bit (the
fixture weights are q4-exact, so MLC's `scale = max|w| / 7` must reproduce them). Libraries are
compiled from the written config with `compile-tiny-webgpu.py --model-config`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from overlay_checks import (  # noqa: E402
    bootstrap_mlc_package,
    ensure_import_roots,
    require_llvm,
    verify_sources,
)

FIXTURE_MODELS = "units-tiny-models-v1"
ARCHES = {"qwen3": "qwen3", "gemma3_text": "gemma3"}
QUANTIZATIONS = ("q0f32", "q4f32_1")


def main() -> None:
    args = parse_args()
    mlc_repository = args.mlc_repository.resolve(strict=True)
    tvm_repository = args.tvm_repository.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    verify_sources(mlc_repository, tvm_repository)
    ensure_import_roots(mlc_repository, tvm_repository)
    os.environ.setdefault("SKIP_LOADING_MLCLLM_SO", "1")

    model_file = json.loads(args.model_file.read_text(encoding="utf-8"))
    if (
        model_file.get("golden") != "units-tiny-models"
        or model_file.get("fixtureModels", {}).get("name") != FIXTURE_MODELS
        or model_file.get("hookAbi") != "block-sites-v5"
    ):
        raise SystemExit(f"{args.model_file} is not the {FIXTURE_MODELS} model file")
    entry = model_file["arches"][ARCHES[args.architecture]]
    if entry["family"] != args.architecture:
        raise SystemExit(f"model file family {entry['family']} is not {args.architecture}")
    source = verify_checkpoint(checkpoint, entry)

    import tvm
    from safetensors.numpy import load_file

    bootstrap_mlc_package(mlc_repository, "fixture", model_registry=True)
    from mlc_llm.interface.convert_weight import convert_weight
    from mlc_llm.model import MODELS
    from mlc_llm.quantization import QUANTIZATION

    require_llvm(tvm)
    production = load_production_tool()
    model = MODELS[args.architecture]
    if args.architecture == "gemma3_text":
        model = production.gemma_conversion_model(model)
    quantization = QUANTIZATION[args.quantization]
    mlc_config = mlc_source_config(args.architecture, entry, args.context_window_size)

    stage = Path(tempfile.mkdtemp(prefix=f"{output.name}.partial-", dir=output.parent))
    try:
        config_path = stage / "source-config.json"
        config_path.write_text(json.dumps(mlc_config, indent=2, sort_keys=True) + "\n")
        convert_weight(
            config=config_path,
            quantization=quantization,
            model=model,
            device=tvm.runtime.cpu(),
            source=checkpoint / "model.safetensors",
            source_format="huggingface-safetensor",
            output=stage,
            encode_format="raw",
        )
        config_path.unlink()
        model_config = model.config.from_dict(mlc_config)
        mapping = model.source["huggingface-safetensor"](model_config, quantization)
        report = verify_conversion(stage, load_file(str(checkpoint / "model.safetensors")), mapping,
                                   args.quantization)
        write_chat_config(stage, checkpoint, args, entry, production.browser_model_config(
            args.architecture, model_config))
        shutil.copyfile(checkpoint / "tokenizer.json", stage / "tokenizer.json")
        (stage / "drowse-units-fixture.json").write_text(json.dumps({
            "fixtureModels": FIXTURE_MODELS,
            "fixtureModelsSha256": hashlib.sha256(args.model_file.read_bytes()).hexdigest(),
            "arch": ARCHES[args.architecture],
            "architecture": args.architecture,
            "quantization": args.quantization,
            "contextWindowSize": args.context_window_size,
            "sourceTensors": source,
            "conversion": report,
        }, indent=2, sort_keys=True) + "\n")
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(json.dumps({"output": str(output), "architecture": args.architecture,
                      "quantization": args.quantization, **report}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlc-repository", required=True, type=Path)
    parser.add_argument("--tvm-repository", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-file", required=True, type=Path)
    parser.add_argument("--architecture", required=True, choices=sorted(ARCHES))
    parser.add_argument("--quantization", required=True, choices=QUANTIZATIONS)
    parser.add_argument("--context-window-size", type=int, default=128)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_production_tool():
    path = Path(__file__).resolve().with_name("build-production-webgpu.py")
    spec = importlib.util.spec_from_file_location("drowse_build_production_webgpu", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_checkpoint(checkpoint: Path, entry: dict) -> int:
    """Every tensor the model file lists, with its recorded fp32 digest; nothing else but the
    tied head."""
    from safetensors.numpy import load_file

    tensors = load_file(str(checkpoint / "model.safetensors"))
    listed = {row["name"]: row for row in entry["tensors"]}
    extra = sorted(set(tensors) - set(listed) - {"lm_head.weight"})
    if extra:
        raise SystemExit(f"checkpoint has tensors the model file does not list: {extra[:4]}")
    for name, row in listed.items():
        value = tensors.get(name)
        if value is None:
            raise SystemExit(f"checkpoint lacks {name}")
        if value.dtype != np.float32 or list(value.shape) != row["shape"]:
            raise SystemExit(f"checkpoint tensor {name} is not fp32 {row['shape']}")
        digest = hashlib.sha256(value.astype("<f4").tobytes()).hexdigest()
        if digest != row["sha256"]:
            raise SystemExit(f"checkpoint tensor {name} differs from {FIXTURE_MODELS}")
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    for key, value in entry["config"].items():
        if config.get(key) != value:
            raise SystemExit(f"checkpoint config {key} differs from {FIXTURE_MODELS}")
    return len(listed)


def sliding_pattern(layer_types: list[str]) -> int:
    """MLC's Gemma3 rule is `local = (L + 1) % pattern != 0`; find the pattern that reproduces
    the HF `layer_types` exactly."""
    count = len(layer_types)
    for pattern in range(1, count + 1):
        if all((layer_types[i] == "sliding_attention") == ((i + 1) % pattern != 0)
               for i in range(count)):
            return pattern
    raise SystemExit(f"layer_types {layer_types} has no MLC sliding pattern")


def mlc_source_config(architecture: str, entry: dict, context_window_size: int) -> dict:
    hf = entry["config"]
    common = {
        "hidden_size": hf["hidden_size"],
        "intermediate_size": hf["intermediate_size"],
        "num_hidden_layers": hf["num_hidden_layers"],
        "num_attention_heads": hf["num_attention_heads"],
        "num_key_value_heads": hf["num_key_value_heads"],
        "head_dim": hf["head_dim"],
        "rms_norm_eps": hf["rms_norm_eps"],
        "vocab_size": hf["vocab_size"],
        "context_window_size": context_window_size,
        "prefill_chunk_size": context_window_size,
        "max_batch_size": 1,
    }
    if architecture == "qwen3":
        rope = hf["rope_parameters"]
        if rope.get("rope_type") != "default" or hf["sliding_window"] is not None:
            raise SystemExit("the qwen3 fixture must use default RoPE and no sliding window")
        return {**common, "model_type": "qwen3", "hidden_act": hf["hidden_act"],
                "attention_bias": False, "rope_theta": rope["rope_theta"],
                "tie_word_embeddings": hf["tie_word_embeddings"]}
    rope = hf["rope_parameters"]
    if any(rope[kind].get("rope_type") != "default" for kind in rope):
        raise SystemExit("the gemma3 fixture must use default RoPE")
    if hf["attn_logit_softcapping"] is not None or hf["final_logit_softcapping"] is not None:
        raise SystemExit("the gemma3 fixture must not soft-cap logits")
    text = {**common, "hidden_activation": hf["hidden_activation"],
            "query_pre_attn_scalar": hf["query_pre_attn_scalar"],
            "position_embedding_base": rope["full_attention"]["rope_theta"],
            "rope_local_base_freq": rope["sliding_attention"]["rope_theta"],
            "sliding_window_size": hf["sliding_window"],
            "sliding_window_pattern": sliding_pattern(hf["layer_types"])}
    return {"model_type": "gemma3_text", "text_config": text, "vocab_size": hf["vocab_size"],
            "context_window_size": context_window_size, "prefill_chunk_size": context_window_size,
            "sliding_window_size": -1, "max_batch_size": 1, "is_text_model": True}


def load_tensor_cache(directory: Path) -> dict[str, np.ndarray]:
    cache = json.loads((directory / "tensor-cache.json").read_text(encoding="utf-8"))
    tensors = {}
    for shard in cache["records"]:
        data = (directory / shard["dataPath"]).read_bytes()
        for record in shard["records"]:
            if record.get("format") != "raw":
                raise SystemExit(f"{record['name']} is not stored raw ({record.get('format')})")
            chunk = data[record["byteOffset"]: record["byteOffset"] + record["nbytes"]]
            tensors[record["name"]] = np.frombuffer(chunk, dtype=record["dtype"]).reshape(
                record["shape"])
    return tensors


def dequantize_q4(weight: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """MLC group quantization (q4f32_1): eight 4-bit codes per uint32, low nibble first, groups of
    32 along the last axis, value = (code − 7) · scale."""
    rows = weight.shape[0]
    shifts = np.arange(8, dtype=np.uint32) * 4
    codes = (weight[:, :, None] >> shifts) & np.uint32(0xF)
    codes = codes.reshape(rows, -1).astype(np.float32)
    columns = codes.shape[1]
    if scale.shape != (rows, columns // 32):
        raise SystemExit(f"q4 scale shape {scale.shape} does not match codes {codes.shape}")
    return (codes - np.float32(7)) * np.repeat(scale.astype(np.float32), 32, axis=1)


def verify_conversion(directory: Path, source: dict, mapping, quantization: str) -> dict:
    converted = load_tensor_cache(directory)
    exact = q4 = 0
    for mlc_name, hf_names in mapping.param_map.items():
        arrays = [source[name] for name in hf_names]
        expected = np.asarray(mapping.map_func[mlc_name](*arrays), dtype=np.float32)
        base = mlc_name[: -len(".weight")] if mlc_name.endswith(".weight") else mlc_name
        if quantization == "q4f32_1" and f"{base}.q_weight" in converted:
            weight, scale = converted[f"{base}.q_weight"], converted[f"{base}.q_scale"]
            if scale.dtype != np.float32:
                raise SystemExit(f"{base}.q_scale is {scale.dtype}, not float32")
            actual = dequantize_q4(weight, scale)[:, : expected.shape[-1]]
            grouped = np.abs(expected.astype(np.float64)).reshape(expected.shape[0], -1, 32)
            if not np.array_equal(scale, (grouped.max(-1) / 7).astype(np.float32)):
                raise SystemExit(f"{base}.q_scale is not max|w| / 7 of the source")
            q4 += 1
        else:
            actual = converted.get(mlc_name)
            if actual is None:
                raise SystemExit(f"converted cache lacks {mlc_name}")
        if actual.shape != expected.shape or not np.array_equal(
            actual.astype(np.float32).view(np.uint32), expected.view(np.uint32)
        ):
            worst = float(np.max(np.abs(actual.astype(np.float64) - expected))) \
                if actual.shape == expected.shape else float("inf")
            raise SystemExit(f"{mlc_name} is not exact after conversion (max |Δ| {worst:g})")
        exact += 1
    return {"exactTensors": exact, "q4Tensors": q4}


def write_chat_config(directory: Path, checkpoint: Path, args, entry: dict, model_config: dict):
    tokenizer = entry["tokenizer"]
    vocab = json.loads((checkpoint / "tokenizer.json").read_text(encoding="utf-8"))
    eos = tokenizer["eos"]
    config = {
        "model_type": args.architecture,
        "quantization": args.quantization,
        "model_config": model_config,
        "vocab_size": entry["config"]["vocab_size"],
        "active_vocab_size": tokenizer["size"],
        "context_window_size": args.context_window_size,
        "sliding_window_size": -1,
        "prefill_chunk_size": args.context_window_size,
        "attention_sink_size": -1,
        "tensor_parallel_shards": 1,
        "pipeline_parallel_stages": 1,
        "tokenizer_files": ["tokenizer.json"],
        "tokenizer_info": {"token_postproc_method": "byte_level", "prepend_space_in_encode": False,
                           "strip_space_in_decode": False},
        "conv_template": {
            "name": "drowse-units-fixture", "system_template": "", "system_message": "",
            "roles": {"user": "", "assistant": ""}, "seps": [""], "role_content_sep": "",
            "role_empty_sep": "", "stop_str": [], "stop_token_ids": [eos],
        },
        "bos_token_id": tokenizer["bos"] if tokenizer["bos"] is not None else eos,
        "eos_token_id": eos,
        "pad_token_id": tokenizer["pad"],
        # Raw completions start at the family's BOS, as the goldens' short prompt does.
        "drowse_completion_prefix_token_ids": [] if tokenizer["bos"] is None else [tokenizer["bos"]],
        "drowse_capture_special_token_ids": sorted(
            vocab["model"]["vocab"][token] if token in vocab["model"]["vocab"] else
            next(item["id"] for item in vocab["added_tokens"] if item["content"] == token)
            for token in tokenizer["specials"]
        ),
        "repetition_penalty": 1.0, "frequency_penalty": 0.0, "presence_penalty": 0.0,
        "top_p": 1.0, "temperature": 0.0,
    }
    (directory / "mlc-chat-config.json").write_text(json.dumps(config, indent=2, sort_keys=True)
                                                     + "\n")


if __name__ == "__main__":
    main()
