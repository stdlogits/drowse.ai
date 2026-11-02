#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from overlay_checks import (  # noqa: E402
    bootstrap_mlc_package,
    ensure_import_roots,
    require_llvm,
    verify_sources,
)


def main() -> None:
    args = parse_args()
    mlc_repository = args.mlc_repository.resolve(strict=True)
    tvm_repository = args.tvm_repository.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    if not output.parent.is_dir():
        raise SystemExit(f"output parent does not exist: {output.parent}")
    verify_sources(mlc_repository, tvm_repository)
    ensure_import_roots(mlc_repository, tvm_repository)
    os.environ.setdefault("SKIP_LOADING_MLCLLM_SO", "1")

    try:
        import tvm
        from tvm.contrib import tvmjs

        bootstrap_mlc_package(mlc_repository, "fixture")
        from mlc_llm.model.gemma3.gemma3_model import (
            Gemma3Config,
            Gemma3ForCausalLM,
            Gemma3TextConfig,
        )
        from mlc_llm.model.llama.llama_model import LlamaConfig, LlamaForCausalLM
        from mlc_llm.model.qwen3.qwen3_model import Qwen3Config, Qwen3LMHeadModel
        from mlc_llm.quantization import QUANTIZATION, make_quantization_functions
    except (ImportError, OSError) as error:
        raise SystemExit(f"exact MLC/TVM fixture toolchain is unavailable: {error}") from error
    require_llvm(tvm)

    model, model_type = create_model(
        args.architecture,
        Qwen3Config,
        Qwen3LMHeadModel,
        LlamaConfig,
        LlamaForCausalLM,
        Gemma3Config,
        Gemma3TextConfig,
        Gemma3ForCausalLM,
        args.quantization,
        QUANTIZATION,
        make_quantization_functions,
        args.context_window_size,
    )
    _, named_params, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    params = {
        name: fixture_parameter(name, parameter)
        for name, parameter in named_params
    }

    output.mkdir()
    tvmjs.dump_tensor_cache(
        params,
        str(output),
        encode_format="raw",
        meta_data={
            "fixture": "drowse-tiny-webgpu-v2",
            "architecture": args.architecture,
            "contextWindowSize": args.context_window_size,
            "quantization": args.quantization,
            "parameterCount": sum(value.size for value in params.values()),
        },
        shard_cap_mb=8,
        show_progress=False,
    )
    write_json(
        output / "mlc-chat-config.json",
        chat_config(model_type, args.quantization, args.context_window_size),
    )
    write_json(output / "tokenizer.json", tokenizer_config())
    print(
        json.dumps(
            {
                "architecture": args.architecture,
                "contextWindowSize": args.context_window_size,
                "files": sorted(path.name for path in output.iterdir()),
                "output": str(output),
                "parameterCount": sum(value.size for value in params.values()),
                "quantization": args.quantization,
            },
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlc-repository", required=True, type=Path)
    parser.add_argument("--tvm-repository", required=True, type=Path)
    parser.add_argument(
        "--architecture", required=True, choices=["qwen3", "llama", "gemma3_text"]
    )
    parser.add_argument(
        "--quantization",
        choices=["q0f32", "q4f32_1", "q4f16_1"],
        default="q0f32",
    )
    parser.add_argument("--context-window-size", type=int, choices=[16, 2048, 4096], default=16)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def create_model(
    architecture,
    qwen_config_class,
    qwen_model_class,
    llama_config_class,
    llama_model_class,
    gemma_config_class,
    gemma_text_config_class,
    gemma_model_class,
    quantization_name,
    quantization_registry,
    make_quantization_functions,
    context_window_size,
):
    common = {
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-6,
        "vocab_size": 16,
        "context_window_size": context_window_size,
        "prefill_chunk_size": context_window_size,
        "head_dim": 32,
    }
    if architecture == "qwen3":
        config = qwen_config_class(
            hidden_act="silu",
            attention_bias=False,
            rope_theta=10000,
            **common,
        )
        model_class = qwen_model_class
        model_type = "qwen3"
    elif architecture == "llama":
        config = llama_config_class(**common)
        model_class = llama_model_class
        model_type = "llama"
    else:
        text_config = gemma_text_config_class(
            hidden_size=common["hidden_size"],
            intermediate_size=common["intermediate_size"],
            num_attention_heads=common["num_attention_heads"],
            num_hidden_layers=common["num_hidden_layers"],
            num_key_value_heads=common["num_key_value_heads"],
            rms_norm_eps=common["rms_norm_eps"],
            context_window_size=common["context_window_size"],
            prefill_chunk_size=common["prefill_chunk_size"],
            head_dim=common["head_dim"],
            sliding_window_size=common["context_window_size"],
            query_pre_attn_scalar=common["head_dim"],
        )
        config = gemma_config_class(
            text_config=text_config,
            vocab_size=common["vocab_size"],
            is_text_model=True,
        )
        model_class = gemma_model_class
        model_type = "gemma3_text"
    quantization = quantization_registry[quantization_name]
    quantize = make_quantization_functions(model_class)[quantization.kind]
    model, _ = quantize(config, quantization)
    return model, model_type


def numpy_dtype(dtype: str):
    if dtype == "float32":
        return np.float32
    if dtype == "float16":
        return np.float16
    if dtype == "uint32":
        return np.uint32
    raise SystemExit(f"unsupported tiny fixture parameter dtype: {dtype}")


def fixture_parameter(name, parameter):
    shape = tuple(dimension_value(dimension) for dimension in parameter.shape)
    values = np.zeros(shape, numpy_dtype(str(parameter.dtype)))
    if name == "model.embed_tokens.weight":
        values[:, 0] = 0.5
        values[:, 1] = np.linspace(-0.25, 0.25, shape[0], dtype=np.float32).astype(
            values.dtype
        )
    elif name == "model.embed_tokens.q_weight":
        values.fill(np.uint32(0x77777777))
        values[:, 0] = np.uint32(0x777777AE)
    elif name == "model.embed_tokens.q_scale":
        values.fill(np.float32(0.5 / 7))
    elif name == "lm_head.q_weight":
        values.fill(np.uint32(0x77777777))
        for token_id in range(shape[0]):
            values[token_id, 0] = np.uint32(0x77777770 | min(token_id, 15))
    elif name == "lm_head.q_scale":
        values.fill(np.float32(0.125))
    elif name.endswith("norm.weight"):
        values.fill(1)
    elif name == "lm_head.weight":
        values[:, 0] = np.linspace(-0.5, 0.5, shape[0], dtype=np.float32).astype(
            values.dtype
        )
        values[:, 1] = np.linspace(0.25, -0.25, shape[0], dtype=np.float32).astype(
            values.dtype
        )
    return values


def dimension_value(dimension) -> int:
    if isinstance(dimension, int):
        return dimension
    value = getattr(dimension, "value", None)
    if isinstance(value, int):
        return value
    if getattr(dimension, "name", None) == "vocab_size":
        return 16
    raise SystemExit(f"tiny fixture parameter has a dynamic dimension: {dimension}")


def chat_config(model_type: str, quantization: str, context_window_size: int):
    return {
        "tokenizer_files": ["tokenizer.json"],
        "tokenizer_info": {
            "token_postproc_method": "raw",
            "prepend_space_in_encode": False,
            "strip_space_in_decode": False,
        },
        "vocab_size": 16,
        "conv_template": {
            "system_template": "{system_message}",
            "system_message": "",
            "roles": {"user": "user", "assistant": "assistant", "tool": "tool"},
            "seps": ["\n"],
            "role_content_sep": ": ",
            "role_empty_sep": ":",
            "stop_str": [],
            "stop_token_ids": [2],
        },
        "context_window_size": context_window_size,
        "model_config": {
            "hidden_size": 64,
            "num_hidden_layers": 2,
        },
        "sliding_window_size": -1,
        "attention_sink_size": -1,
        "repetition_penalty": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "top_p": 1.0,
        "temperature": 0.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 3,
        "drowse_capture_special_token_ids": [0, 1, 2, 3],
        "model_type": model_type,
        "quantization": quantization,
    }


def tokenizer_config():
    vocabulary = {
        "<unk>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<pad>": 3,
        "hello": 4,
        "world": 5,
        "user": 6,
        "assistant": 7,
        ":": 8,
        "test": 9,
        "a": 10,
        "b": 11,
        "c": 12,
        "d": 13,
        "e": 14,
        "f": 15,
    }
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": vocabulary, "unk_token": "<unk>"},
    }


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
