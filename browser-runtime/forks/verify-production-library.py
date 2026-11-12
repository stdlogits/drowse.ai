#!/usr/bin/env python3
"""One compiled production library (hook ABI v5) checked on the CPU (W4a; B1 §4.3, §8.2).

- The export and shader checks the compile runs: every function of `required_functions(arch)`
  is present and every drowse kernel keeps at most 8 storage bindings.
- The descriptor (schema 4) the model's `drowse_hook_profile` returns, built for the C TVM VM and
  written to the report, for `expectedHookProfileDescriptor` (C0) to check against the lock's
  block spec.
- Units models (`standard-v5`), with `--parity`: one forward of the parity input through
  `drowse_block` on the C TVM VM against PyTorch on the dequantized MLC tensors (`mlc_q4_torch`):
  the per-layer relative error of `act`, `z`, `mid` and `out` and of the last position's logits
  at most 1% (B1 §4.3, the q4 parity gate's measure); greedy tokens through the units pair equal
  to PyTorch's; static reads over the resident weights (neuron and head rows, embedding and
  unembedding rows, layer write scores and vocabulary logits of the forward's own residuals)
  within 1e-3 relative of fp64 numpy with top-8 overlap at least 7 of 8.

The C TVM VM runs the library's Relax graph and drowse kernels (serialized), not the WebGPU
shaders: this checks the conversion, the graph and the constants, and leaves the WGSL to the
WebGPU harness. The PyTorch reference runs in its own process (`--reference-python`, CPU only),
since `mlc_q4_torch` needs Transformers 5 and the compiler toolchain pins 4.52.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import types as pytypes
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
from overlay_checks import (  # noqa: E402
    UNIT_ARCHITECTURES,
    bootstrap_mlc_package,
    ensure_import_roots,
    require_llvm,
    required_functions,
    verify_library_shaders,
    verify_sources,
)

SITE_TOLERANCE = 1e-2
READ_TOLERANCE = 1e-3
TOP_K, TOP_K_OVERLAP = 8, 7
BASE_ARCHITECTURES = ("gpt2", "gpt_neox", "qwen3_5")


def main() -> None:
    args = parse_args()
    if args.reference_out is not None:
        reference(args.model.resolve(strict=True), json.loads(args.input.read_text()),
                  args.greedy_tokens, args.reference_out)
        return
    if args.mlc_repository is None or args.tvm_repository is None:
        raise SystemExit("--mlc-repository and --tvm-repository are required")
    mlc_repository = args.mlc_repository.resolve(strict=True)
    tvm_repository = args.tvm_repository.resolve(strict=True)
    model_dir = args.model.resolve(strict=True)
    library = (args.library or model_dir / "model.wasm").resolve(strict=True)
    chat = json.loads((model_dir / "mlc-chat-config.json").read_text())
    build = json.loads((model_dir / "drowse-build.json").read_text())
    arch = chat["model_type"]
    # The CPU checks run the graph the model's own config and conversion produce (the C TVM VM
    # cannot run the WGSL); these digests bind the report to the library and model it checked.
    report: dict = {"model": str(model_dir), "library": str(library), "architecture": arch,
                    "quantization": chat["quantization"],
                    "structuredHookProfile": build["structuredHookProfile"],
                    "sha256": {name: file_sha256(path) for name, path in (
                        ("model.wasm", library),
                        ("drowse-build.json", model_dir / "drowse-build.json"),
                        ("mlc-chat-config.json", model_dir / "mlc-chat-config.json"),
                        ("tensor-cache.json", model_dir / "tensor-cache.json"))},
                    "checks": {}}
    started = time.time()

    verify_library_shaders(library.read_bytes(), required_functions(arch))
    report["checks"]["exportAndShaders"] = {"passed": True,
                                            "functions": sorted(required_functions(arch))}

    verify_sources(mlc_repository, tvm_repository)
    ensure_import_roots(mlc_repository, tvm_repository)
    os.environ.setdefault("SKIP_LOADING_MLCLLM_SO", "1")
    bootstrap_mlc_package(mlc_repository, "production verification", model_registry=True)
    if arch in BASE_ARCHITECTURES:
        from base_model_adapters import register_candidate_models

        register_candidate_models(include_hybrid=arch == "qwen3_5")
    import tvm
    from tvm import relax
    from tvm.relax.frontend import nn
    from tvm.relax.frontend.nn import op

    require_llvm(tvm)
    env = environment(tvm, relax, nn, op)
    report["descriptor"] = library_descriptor(env, chat)
    report["checks"]["descriptor"] = {"passed": True, "fields": len(report["descriptor"]),
                                      "source": "the model class's drowse_hook_profile on the C TVM VM"}

    failures: list[str] = []
    if args.parity:
        if arch not in UNIT_ARCHITECTURES:
            raise SystemExit(f"{arch} attests standard-v5-residual: no block sites to compare")
        parity_input = json.loads(args.input.read_text())
        report["checks"].update(parity(env, model_dir, args.input, parity_input, args.greedy_tokens,
                                       args.reference_python, failures))
    report["seconds"] = round(time.time() - started, 1)
    report["passed"] = not failures
    report["failures"] = failures
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.report is not None:
        args.report.write_text(text + "\n")
    print(json.dumps({key: report[key] for key in ("model", "passed", "failures", "seconds")}))
    if failures:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlc-repository", type=Path)
    parser.add_argument("--tvm-repository", type=Path)
    parser.add_argument("--model", required=True, type=Path,
                        help="a build-production-webgpu.py model directory")
    parser.add_argument("--library", type=Path, help="defaults to MODEL/model.wasm")
    parser.add_argument("--parity", action="store_true",
                        help="units models: block sites, greedy tokens and static reads")
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parents[1]
                        / "fixtures/production-q4-parity-input-v1.json")
    parser.add_argument("--greedy-tokens", type=int, default=4)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--reference-python", type=Path, default=Path(sys.executable),
                        help="the interpreter of the PyTorch reference (Transformers 5)")
    parser.add_argument("--reference-out", type=Path,
                        help="internal: run only the PyTorch reference and write it here (.npz)")
    return parser.parse_args()


def environment(tvm, relax, nn, op):
    """What `units_goldens.FixtureVM` needs (the same namespace verify-mlc-hook.py builds)."""
    from mlc_llm import compiler_pass as _compiler_pass  # noqa: F401  (registers the pipeline)
    from mlc_llm import op as op_ext
    from mlc_llm.interface.compiler_flags import IPCAllReduceStrategyType
    from mlc_llm.model import MODELS, drowse_reads, drowse_units
    from mlc_llm.model.drowse_hooks import structured_curve_program_spec
    from mlc_llm.quantization import QUANTIZATION, make_quantization_functions

    forks = Path(__file__).resolve().parent
    verifier = load_module("drowse_verify_mlc_hook", forks / "verify-mlc-hook.py")
    compiler = load_module("drowse_compile_tiny", forks / "compile-tiny-webgpu.py")
    op_ext.enable(target=tvm.target.Target("llvm"), flashinfer=False, faster_transformer=False,
                  cutlass=False)
    return pytypes.SimpleNamespace(
        tvm=tvm, relax=relax, nn=nn, op=op, MODELS=MODELS,
        models={name: (info.config, info.model) for name, info in MODELS.items()},
        QUANTIZATION=QUANTIZATION, make_quantization_functions=make_quantization_functions,
        strip_gpu_thread_bindings=verifier.strip_gpu_thread_bindings,
        prepare_parameters=compiler.prepare_parameters, create_metadata=compiler.create_metadata,
        IPCAllReduceStrategyType=IPCAllReduceStrategyType,
        structured_curve_program_spec=structured_curve_program_spec,
        unit_program_spec=drowse_units.unit_program_spec,
        unit_meta_layout=drowse_units.unit_meta_layout,
        unit_coeff_layout=drowse_units.unit_coeff_layout,
        UnitSites=drowse_units.UnitSites,
        drowse_units=drowse_units,
        drowse_reads=drowse_reads,
        UNIT_MAX_GROUPS=drowse_units.UNIT_MAX_GROUPS,
        UNIT_MAX_HEAD_TAPS=drowse_units.UNIT_MAX_HEAD_TAPS,
    )


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def library_descriptor(env, chat: dict) -> list[int]:
    """`drowse_hook_profile()` of the model class the compile used, on the C TVM VM."""
    tvm, relax, nn = env.tvm, env.relax, env.nn
    info = env.MODELS[chat["model_type"]]
    config = info.config.from_dict(chat["model_config"])
    quantization = env.QUANTIZATION[chat["quantization"]]
    with tvm.target.Target({"kind": "llvm"}):
        model, _ = info.quantize[quantization.kind](config, quantization)
        spec = nn.spec.ModuleSpec.from_raw(
            {"drowse_hook_profile": {"$": {"param_mode": "none", "effect_mode": "none"}}}, model)
        module, _, _ = model.export_tvm(spec=spec, allow_extern=True)
    executable = relax.build(module, target=tvm.target.Target({"kind": "llvm"}))
    vm = relax.VirtualMachine(executable, tvm.cpu())
    return [int(value) for value in vm["drowse_hook_profile"]().numpy().reshape(-1)]


# --------------------------------------------------------------------------- parity (§4.3) ---

def parity(env, model_dir: Path, input_path: Path, parity_input: dict, greedy_tokens: int,
           reference_python: Path, failures: list) -> dict:
    import subprocess
    import tempfile

    import units_goldens as ug
    from static_reads import Device

    ids = [int(value) for value in parity_input["inputIds"]]
    checks: dict = {}

    timer = time.time()
    vm = ug.FixtureVM(env, model_dir)
    checks["cvmBuildSeconds"] = round(time.time() - timer, 1)
    block, layers = vm.block, vm.num_layers
    caps = {"groups": env.UNIT_MAX_GROUPS, "head_taps": env.UNIT_MAX_HEAD_TAPS}
    empty = ug.UnitProgram(env.unit_meta_layout(layers), env.unit_coeff_layout(block["head_dim"]),
                           layers, block["head_dim"], caps).buffers([])
    timer = time.time()
    vm_logits, sites = vm.forward_sites(vm.new_cache(), ids, empty)
    checks["cvmForwardSeconds"] = round(time.time() - timer, 1)

    timer = time.time()
    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch) / "reference.npz"
        subprocess.run([str(reference_python), "-I", str(Path(__file__).resolve()), "--model", str(model_dir),
                        "--input", str(input_path), "--greedy-tokens", str(greedy_tokens),
                        "--reference-out", str(out)],
                       check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": ""})
        with np.load(out) as saved:
            torch_sites = {name: saved[name] for name in ("act", "z", "mid", "out")}
            torch_logits = saved["logits"]
            torch_greedy = [int(v) for v in saved["greedy"]]
            eos_ids = {int(v) for v in saved["eos"]}
            checks["reference"] = {"transformers": str(saved["transformers"]), "torch": str(saved["torch"])}
    checks["torchReferenceSeconds"] = round(time.time() - timer, 1)

    site_errors = {}
    for name in ("act", "z", "mid", "out"):
        errors = [relative(sites[name][layer], torch_sites[name][layer]) for layer in range(layers)]
        site_errors[name] = {"max": max(errors), "byLayer": errors}
        if max(errors) > SITE_TOLERANCE:
            failures.append(f"sites:{name} relative error {max(errors):.3g} > {SITE_TOLERANCE}")
    logit_error = relative(vm_logits, torch_logits)
    overlap = top_overlap(vm_logits, torch_logits)
    site_errors["logits"] = {"max": logit_error, "top8Overlap": overlap}
    if logit_error > SITE_TOLERANCE or overlap < TOP_K_OVERLAP:
        failures.append(f"sites:logits relative error {logit_error:.3g}, top-8 overlap {overlap}")
    checks["sites"] = site_errors

    timer = time.time()
    vm_greedy = cvm_greedy(vm, ids, empty, greedy_tokens, eos_ids)
    checks["greedy"] = {"cvm": vm_greedy, "torch": torch_greedy, "equal": vm_greedy == torch_greedy,
                        "seconds": round(time.time() - timer, 1)}
    if vm_greedy != torch_greedy:
        failures.append(f"greedy tokens differ: cvm {vm_greedy}, torch {torch_greedy}")

    timer = time.time()
    tensors = ug.load_tensor_cache(model_dir)
    weights = ProductionWeights(tensors, block, layers)
    checks["staticReads"] = static_reads(Device(vm), weights, sites, failures)
    checks["staticReads"]["seconds"] = round(time.time() - timer, 1)
    return checks


def reference(model_dir: Path, parity_input: dict, greedy_tokens: int, out: Path) -> None:
    """The PyTorch half: sites, last logits and greedy tokens on the dequantized MLC tensors."""
    import torch
    import transformers

    from mlc_q4_torch import build_transformers_model
    from validate_mlc_q4_parity import greedy_token_ids

    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "4")))
    ids = [int(value) for value in parity_input["inputIds"]]
    model = build_transformers_model(model_dir, "cpu")
    chat = json.loads((model_dir / "mlc-chat-config.json").read_text())
    sites, logits = torch_forward(model, ids, chat["model_type"] == "gemma3_text")
    eos = model.config.eos_token_id
    eos_ids = [int(eos)] if isinstance(eos, int) else [int(value) for value in eos or []]
    greedy = greedy_token_ids(model, ids, greedy_tokens, "cpu")
    np.savez(out, **sites, logits=logits, greedy=np.asarray(greedy, np.int64),
             eos=np.asarray(eos_ids, np.int64), transformers=transformers.__version__,
             torch=torch.__version__)


def torch_forward(model, ids: list[int], sandwich: bool):
    """`act` (down-projection input), `z` (output-projection input), `mid` (the residual after
    attention: the input of the MLP's pre-norm) and `out` (the block output) per layer."""
    import torch

    rows: dict = {"act": {}, "z": {}, "mid": {}, "out": {}}
    handles = []

    def pre(name, layer):
        def hook(_module, inputs):
            rows[name][layer] = inputs[0][0].detach().to(torch.float32).numpy().copy()
        return hook

    def post(layer):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            rows["out"][layer] = value[0].detach().to(torch.float32).numpy().copy()
        return hook

    for layer, block in enumerate(model.model.layers):
        mid_norm = block.pre_feedforward_layernorm if sandwich else block.post_attention_layernorm
        handles += [block.mlp.down_proj.register_forward_pre_hook(pre("act", layer)),
                    block.self_attn.o_proj.register_forward_pre_hook(pre("z", layer)),
                    mid_norm.register_forward_pre_hook(pre("mid", layer)),
                    block.register_forward_hook(post(layer))]
    try:
        with torch.inference_mode():
            logits = model(torch.tensor([ids], dtype=torch.long), use_cache=False).logits[0, -1]
    finally:
        for handle in handles:
            handle.remove()
    count = len(model.model.layers)
    sites = {name: np.stack([rows[name][layer] for layer in range(count)]) for name in rows}
    return sites, logits.to(torch.float32).numpy()


def cvm_greedy(vm, ids: list[int], empty, count: int, eos_ids: set) -> list[int]:
    cache = vm.new_cache()
    logits, _, _ = vm.forward_units(cache, ids, empty, prefill=True)
    generated: list[int] = []
    for _ in range(count):
        token = int(np.argmax(logits))
        if token in eos_ids:
            break
        generated.append(token)
        if len(generated) == count:
            break
        logits, _, _ = vm.forward_units(cache, [token], empty, prefill=False)
    return generated


# ------------------------------------------------------------------------- static reads ---

class ProductionWeights:
    """`static_reads.Weights` over a production tensor cache, one layer resident at a time and the
    vocabulary dequantized in chunks (a 4B model's embedding is 5.4 GB in fp64)."""

    CHUNK = 16384

    def __init__(self, tensors: dict, block: dict, num_layers: int):
        from static_reads import Weights

        self.inner = Weights(tensors, block, num_layers)
        self.tensors, self.block, self.prefix = tensors, block, self.inner.prefix
        self._layer = (None, None)

    def layer(self, layer: int) -> dict:
        if self._layer[0] != layer:
            self._layer = (None, None)
            self._layer = (layer, self.inner.layer(layer))
        return self._layer[1]

    def gain(self, layer: int, which: str) -> np.ndarray:
        return self.inner.gain(layer, which)

    def writes(self, layer: int) -> np.ndarray:
        scale = float(self.block.get("residual_scale", 1.0))
        return (scale * self.gain(layer, "mo")[:, None] * self.layer(layer)["down"]).T

    def reads(self, layer: int):
        gain, w = self.gain(layer, "mi"), self.layer(layer)
        a, b = w["gate"] * gain, w["up"] * gain
        return (a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12),
                b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12))

    def head_write(self, layer: int, head: int) -> np.ndarray:
        hd = self.block["head_dim"]
        return self.gain(layer, "ao")[:, None] * self.layer(layer)["o"][:, head * hd:(head + 1) * hd]

    @property
    def vocab(self) -> int:
        name = self.prefix + "embed_tokens"
        tensor = self.tensors.get(name + ".weight", self.tensors.get(name + ".q_weight"))
        return int(tensor.shape[0])

    def embedding_rows(self, start: int, stop: int) -> np.ndarray:
        name = self.prefix + "embed_tokens"
        if name + ".weight" in self.tensors:
            return self.tensors[name + ".weight"][start:stop].astype(np.float64)
        weight = self.tensors[name + ".q_weight"][start:stop]
        scale = self.tensors[name + ".q_scale"][start:stop]
        shifts = np.arange(8, dtype=np.uint32) * 4
        codes = ((weight[:, :, None] >> shifts) & 0xF).reshape(weight.shape[0], -1).astype(np.float64)
        return (codes - 7) * np.repeat(scale.astype(np.float64), 32, axis=1)

    @property
    def final_gain(self) -> np.ndarray:
        return self.inner.final_gain


def static_reads(dev, w: ProductionWeights, sites: dict, failures: list) -> dict:
    block = w.block
    L = sites["out"].shape[0]
    d_ff, heads, hd, kv = block["d_ff"], block["n_heads"], block["head_dim"], block["n_kv_heads"]
    eps, scale_e = block["rms_eps"], float(block["embed_scale"])
    group = heads // kv
    layers = sorted({0, L // 2, L - 1})
    worst: dict = {}

    def check(label: str, expected, actual) -> None:
        error = relative(np.asarray(actual), np.asarray(expected))
        worst[label] = max(worst.get(label, 0.0), error)

    units = np.array([[l, n] for l in layers for n in (0, d_ff // 2, d_ff - 1)], np.int32)
    rows = dev("drowse_neuron_rows", units)
    for i, (layer, n) in enumerate(units):
        a, b = w.reads(int(layer))
        for k, expected in enumerate((w.writes(int(layer))[n], a[n], b[n])):
            check("neuron_rows", expected, rows[i, k])
    head_units = np.array([[l, h] for l in layers for h in (0, heads - 1)], np.int32)
    head = dev("drowse_head_rows", head_units)
    for i, (layer, h) in enumerate(head_units):
        p, ai, g = w.layer(int(layer)), w.gain(int(layer), "ai"), int(h) // group
        expected = (w.head_write(int(layer), int(h)).T, ai * p["q"][h * hd:(h + 1) * hd],
                    ai * p["k"][g * hd:(g + 1) * hd], ai * p["v"][g * hd:(g + 1) * hd])
        for k, value in enumerate(expected):
            check("head_rows", value, head[i, k])

    overlaps: dict = {}
    selected = np.array(layers, np.int32)
    x = np.stack([sites["mid"][layer][-2:] for layer in layers]).astype(np.float32)
    scores = dev("drowse_layer_write_scores", x, selected)
    for j, layer in enumerate(layers):
        expected = x[j].astype(np.float64) @ w.writes(layer).T
        for r in range(expected.shape[0]):
            check("layer_write_scores", expected[r], scores[j, r])
            overlaps.setdefault("layer_write_scores", []).append(top_overlap(scores[j, r], expected[r]))

    vocab = w.vocab
    ids = np.array([0, 1, vocab // 2, vocab - 1], np.int32)
    zero = np.zeros(sites["out"].shape[-1], np.float32)
    embedded = dev("drowse_embedding_rows", ids, zero)
    unembedded = dev("drowse_unembedding_rows", ids)
    gamma_f = w.final_gain
    for i, v in enumerate(ids):
        row = w.embedding_rows(int(v), int(v) + 1)[0]
        check("embedding_rows", scale_e * row, embedded[i])
        check("unembedding_rows", gamma_f * row, unembedded[i])

    last = sites["out"][L - 1][-2:].astype(np.float32)
    logits = dev("drowse_vocab_logits", last)
    normed = last.astype(np.float64) * rms(last, eps)[:, None] * gamma_f
    expected = np.concatenate([normed @ w.embedding_rows(start, min(start + w.CHUNK, vocab)).T
                               for start in range(0, vocab, w.CHUNK)], axis=1)
    for r in range(expected.shape[0]):
        check("vocab_logits", expected[r], logits[r][:vocab])
        overlaps.setdefault("vocab_logits", []).append(top_overlap(logits[r][:vocab], expected[r]))

    for label, error in worst.items():
        if error > READ_TOLERANCE:
            failures.append(f"static reads:{label} relative error {error:.3g} > {READ_TOLERANCE}")
    for label, values in overlaps.items():
        if min(values) < TOP_K_OVERLAP:
            failures.append(f"static reads:{label} top-8 overlap {min(values)} < {TOP_K_OVERLAP}")
    return {"worstRelative": worst, "top8Overlap": {k: min(v) for k, v in overlaps.items()},
            "layers": layers}


def rms(x: np.ndarray, eps: float) -> np.ndarray:
    return 1.0 / np.sqrt(np.mean(np.asarray(x, np.float64) ** 2, axis=-1) + eps)


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(actual: np.ndarray, expected: np.ndarray) -> float:
    """‖a − e‖ / ‖e‖; infinite (so every gate fails) on a shape mismatch or any non-finite value."""
    actual = np.asarray(actual, np.float64)
    expected = np.asarray(expected, np.float64)
    if actual.shape != expected.shape or not (np.isfinite(actual).all() and np.isfinite(expected).all()):
        return float("inf")
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-12))


def top_overlap(actual: np.ndarray, expected: np.ndarray) -> int:
    """Shared ids among the two top-8 sets; 0 on a shape mismatch or any non-finite value."""
    actual = np.asarray(actual, np.float64).reshape(-1)
    expected = np.asarray(expected, np.float64).reshape(-1)
    if actual.shape != expected.shape or not (np.isfinite(actual).all() and np.isfinite(expected).all()):
        return 0
    a = set(np.argsort(-actual, kind="stable")[:TOP_K].tolist())
    e = set(np.argsort(-expected, kind="stable")[:TOP_K].tolist())
    return len(a & e)


if __name__ == "__main__":
    main()
