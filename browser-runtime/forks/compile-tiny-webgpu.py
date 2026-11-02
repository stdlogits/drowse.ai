#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from overlay_checks import (  # noqa: E402
    UNIT_FUNCTIONS,
    required_functions,
    bootstrap_mlc_package,
    ensure_import_roots,
    require_llvm,
    verify_emcc,
    verify_geometry_shader_bindings,
    verify_library_shaders,
    verify_sources,
)


def main() -> None:
    args = parse_args()
    mlc_repository = args.mlc_repository.resolve(strict=True)
    tvm_repository = args.tvm_repository.resolve(strict=True)
    output = args.output.resolve()
    validate_output(output)
    fixture_config = (
        json.loads(args.model_config.read_text(encoding="utf-8"))
        if args.model_config is not None
        else None
    )
    verify_sources(mlc_repository, tvm_repository)
    emcc = verify_emcc()
    verify_wasm_runtime(mlc_repository)
    ensure_import_roots(mlc_repository, tvm_repository)
    os.environ["MLC_LLM_SOURCE_DIR"] = str(mlc_repository)
    os.environ.setdefault("SKIP_LOADING_MLCLLM_SO", "1")

    try:
        import tvm
        from tvm import relax
        from tvm.s_tir.dlight.base import transform as dlight_transform

        bootstrap_mlc_package(mlc_repository, "compiler")
        from mlc_llm import compiler_pass as _compiler_pass  # noqa: F401
        from mlc_llm import op as op_ext
        from mlc_llm.interface.compiler_flags import IPCAllReduceStrategyType
        from mlc_llm.model.gemma3.gemma3_model import (
            Gemma3Config,
            Gemma3ForCausalLM,
            Gemma3TextConfig,
        )
        from mlc_llm.model.llama.llama_model import LlamaConfig, LlamaForCausalLM
        from mlc_llm.model.qwen3.qwen3_model import Qwen3Config, Qwen3LMHeadModel
        from mlc_llm.quantization import QUANTIZATION, make_quantization_functions
    except (ImportError, OSError) as error:
        raise SystemExit(f"exact MLC/TVM compiler toolchain is unavailable: {error}") from error
    require_llvm(tvm)
    install_dlight_diagnostic(dlight_transform)

    target = tvm.target.Target(
        {
            "kind": "webgpu",
            "host": {
                "kind": "llvm",
                "mtriple": "wasm32-unknown-unknown-wasm",
            },
        }
    )
    with target:
        op_ext.enable(
            target=target,
            flashinfer=False,
            faster_transformer=False,
            cutlass=False,
        )
        model, model_type, config = create_model(
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
            fixture_config,
        )
        module, named_params, ext_mods = model.export_tvm(
            spec=model.get_default_spec(),
            allow_extern=True,
        )

    verify_exports(module, args.architecture)
    if args.ir_stats_only:
        functions = []
        for global_var in module.get_global_vars():
            name = global_var.name_hint
            function = module[global_var]
            functions.append({
                "name": name,
                "scriptBytes": len(function.script(show_meta=False).encode()),
                "type": type(function).__name__,
            })
        print(json.dumps({
            "architecture": args.architecture,
            "functionCount": len(functions),
            "functions": sorted(functions, key=lambda item: item["scriptBytes"], reverse=True),
            "quantization": args.quantization,
        }, sort_keys=True))
        return
    prepare_parameters(named_params)
    metadata = create_metadata(model_type, config, named_params, args.quantization)
    num_hidden_layers = (
        config.num_hidden_layers
        if hasattr(config, "num_hidden_layers")
        else config.text_config.num_hidden_layers
    )
    variable_bounds = {
        "total_seq_len": config.context_window_size,
        "seq_len": config.prefill_chunk_size,
        "batch_size": 1,
        "jlens_layers": num_hidden_layers,
        "sae_features": 16384,
    }
    cuda_graph_hints = {
        "batch_decode": ["batch_size"],
        "batch_decode_to_last_hidden_states": ["batch_size"],
        "batch_verify": ["batch_size", "seq_len"],
        "batch_verify_to_last_hidden_states": ["batch_size", "seq_len"],
    }
    pipeline = relax.get_pipeline(
        "mlc_llm",
        target=target,
        flashinfer=False,
        cublas_gemm=False,
        faster_transformer=False,
        allreduce_strategy=IPCAllReduceStrategyType.NONE,
        variable_bounds=variable_bounds,
        cuda_graph_symbolic_capture_hints=cuda_graph_hints,
        additional_tirs={},
        ext_mods=ext_mods,
        metadata=metadata,
        debug_dump=None,
    )
    runtime_bitcode = mlc_repository / "web/dist/wasm/mlc_wasm_runtime.bc"
    with tvm.transform.PassContext(config={"tirx.disable_cse_tir": True}):
        executable = relax.build(
            module,
            target=target,
            relax_pipeline=pipeline,
            system_lib=True,
        )
        executable.export_library(str(output), libs=[str(runtime_bitcode)])

    wasm = output.read_bytes()
    if len(wasm) == 0 or wasm[:4] != b"\x00asm":
        output.unlink(missing_ok=True)
        raise SystemExit("compiler did not produce a valid WebAssembly container")
    verify_wasm(output, wasm, emcc, args.architecture)
    digest = hashlib.sha256(wasm).hexdigest()
    print(
        json.dumps(
            {
                "architecture": args.architecture,
                "bytes": output.stat().st_size,
                "contextWindowSize": config.context_window_size,
                "output": str(output),
                "quantization": args.quantization,
                "sha256": digest,
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
    parser.add_argument(
        "--model-config",
        type=Path,
        help="mlc-chat-config.json of a build-units-fixtures.py model to compile instead of the "
        "synthetic tiny config (its context window replaces --context-window-size)",
    )
    parser.add_argument("--ir-stats-only", action="store_true")
    return parser.parse_args()


def validate_output(output: Path) -> None:
    if output.suffix != ".wasm":
        raise SystemExit("output must use the .wasm extension")
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    if not output.parent.is_dir():
        raise SystemExit(f"output parent does not exist: {output.parent}")


def verify_wasm(output: Path, wasm: bytes, emcc: Path, architecture: str) -> None:
    try:
        verify_library_shaders(wasm, required_functions(architecture))
    except SystemExit:
        output.unlink(missing_ok=True)
        raise
    wasm_opt = emcc.parents[1] / "bin/wasm-opt"
    if not wasm_opt.is_file():
        output.unlink(missing_ok=True)
        raise SystemExit(f"exact Emscripten Binaryen validator is missing: {wasm_opt}")
    validation = subprocess.run(
        [str(wasm_opt), str(output), "--enable-bulk-memory", "-o", os.devnull],
        text=True,
        capture_output=True,
        check=False,
    )
    if validation.returncode != 0:
        output.unlink(missing_ok=True)
        detail = validation.stderr.strip().splitlines()
        raise SystemExit("WebAssembly validation failed: " + (detail[-1] if detail else "unknown error"))


def verify_wasm_runtime(repository: Path) -> None:
    runtime = repository / "web/dist/wasm/mlc_wasm_runtime.bc"
    if not runtime.is_file() or runtime.stat().st_size == 0:
        raise SystemExit(
            "missing mlc_wasm_runtime.bc; run the exact MLC web/prep_emcc_deps.sh first"
        )


def install_dlight_diagnostic(dlight_transform) -> None:
    original = dlight_transform._apply_rules

    def apply_with_context(func, target, rules, tunable):
        try:
            return original(func, target, rules, tunable)
        except Exception as error:
            symbol = func.attrs.get("global_symbol", "<anonymous>")
            body = func.script(show_meta=False)
            raise RuntimeError(
                f"WebGPU scheduling failed for TIR function {symbol}: {body[:1200]}"
            ) from error

    dlight_transform._apply_rules = apply_with_context


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
    fixture_config=None,
):
    if fixture_config is not None:
        # A converted fixture model (build-units-fixtures.py): compile exactly its config.
        if (
            fixture_config["model_type"] != architecture
            or fixture_config["quantization"] != quantization_name
            or architecture not in ("qwen3", "gemma3_text")
        ):
            raise SystemExit("--model-config does not describe this architecture and quantization")
        config_class, model_class = (
            (qwen_config_class, qwen_model_class)
            if architecture == "qwen3"
            else (gemma_config_class, gemma_model_class)
        )
        config = config_class.from_dict(fixture_config["model_config"])
        quantization = quantization_registry[quantization_name]
        quantize = make_quantization_functions(model_class)[quantization.kind]
        model, _ = quantize(config, quantization)
        return model, architecture, config
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
    return model, model_type, config


def verify_exports(module, architecture: str) -> None:
    names = {global_var.name_hint for global_var in module.get_global_vars()}
    missing = sorted(required_functions(architecture) - names)
    unexpected = sorted(UNIT_FUNCTIONS & names) if architecture == "llama" else []
    if unexpected:
        raise SystemExit(f"{architecture} is a residual architecture but exports {unexpected[0]}")
    if missing:
        raise SystemExit(f"{architecture} export lacks Drowse functions: {', '.join(missing)}")


def prepare_parameters(named_params) -> None:
    for _, parameter in named_params:
        parameter.attrs["preprocs"] = list(parameter.attrs.get("preprocs", []))
        parameter.attrs["pipeline_stages"] = [0]


def create_metadata(model_type, config, named_params, quantization):
    def parameter_metadata(name, parameter):
        return {
            "name": name,
            "shape": [dimension if isinstance(dimension, int) else dimension.name for dimension in parameter.shape],
            "dtype": str(parameter.dtype),
            "preprocs": parameter.attrs["preprocs"],
            "pipeline_stages": parameter.attrs["pipeline_stages"],
        }

    return {
        "model_type": model_type,
        "quantization": quantization,
        "context_window_size": config.context_window_size,
        "sliding_window_size": getattr(config, "sliding_window_size", -1),
        "attention_sink_size": getattr(config, "attention_sink_size", -1),
        "prefill_chunk_size": config.prefill_chunk_size,
        "tensor_parallel_shards": config.tensor_parallel_shards,
        "pipeline_parallel_stages": getattr(config, "pipeline_parallel_stages", 1),
        "disaggregation": getattr(config, "disaggregation", False),
        "kv_state_kind": "kv_cache",
        "max_batch_size": 1,
        "active_vocab_size": None,
        "model_task": "chat",
        "params": [parameter_metadata(name, parameter) for name, parameter in named_params],
    }


if __name__ == "__main__":
    main()
