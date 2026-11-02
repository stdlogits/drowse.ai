"""Pinned toolchain identities and overlay checks shared by every fork script.

`compile-tiny-webgpu.py`, `create-tiny-webgpu-model.py`, `build-production-webgpu.py` and
`verify-mlc-hook.py` import this module, so the source pins, the overlay result digests and the
emitted-shader checks have one definition. The pins come from `manifest.json` beside this file
(the document the runtime lock binds by digest), never from a second copy.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

MANIFEST_PATH = Path(__file__).resolve().with_name("manifest.json")
MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
_OVERLAYS = {overlay["id"]: overlay for overlay in MANIFEST["overlays"]}

HOOK_ABI = MANIFEST["hookAbi"]
STRUCTURED_HOOK_PROFILE = MANIFEST["structuredHookProfile"]
EXACT_READOUT_ABI = MANIFEST["exactReadoutAbi"]
MLC_BASE_COMMIT = _OVERLAYS["mlc-llm-drowse"]["baseCommit"]
TVM_COMMIT = MANIFEST["tvmCommit"]
TVM_FFI_COMMIT = MANIFEST["tvmFfiCommit"]
EMCC_VERSION = MANIFEST["emccVersion"]
EMCC_REVISION = MANIFEST["emccRevision"]
RESULT_DIGESTS = dict(_OVERLAYS["mlc-llm-drowse"]["resultFiles"])
TVM_RESULT_DIGESTS = dict(_OVERLAYS["tvm-webgpu-readonly"]["resultFiles"])

#: VM functions every eligible (dense, unit-site) model library exports.
REQUIRED_FUNCTIONS = {
    "drowse_hook_profile",
    "drowse_prefill",
    "drowse_decode",
    "drowse_batch_prefill",
    "drowse_batch_decode",
    "drowse_capture_prefill",
    "drowse_capture_decode",
    "drowse_capture_batch_prefill",
    "drowse_capture_batch_decode",
    "drowse_rank_one_capture_prefill_v1",
    "drowse_rank_one_capture_decode_v1",
    "drowse_rank_one_capture_batch_prefill_v1",
    "drowse_rank_one_capture_batch_decode_v1",
    "drowse_structured_batch_prefill",
    "drowse_structured_batch_decode",
    "drowse_geometry_batch_prefill",
    "drowse_geometry_batch_decode",
    "drowse_curved_batch_prefill",
    "drowse_curved_batch_decode",
    "drowse_jlens_probabilities",
    "drowse_jlens_readout_accumulate",
    "drowse_jlens_readout_topk",
    "drowse_jlens_directions",
    "drowse_sae_readout_accumulate",
    "drowse_sae_jump_relu_readout_accumulate",
}

GEOMETRY_SHADER_PREFIX = b"// Function: drowse_geometry_measurements"
TOPK_TILE_SHADER_PREFIX = b"// Function: drowse_exact_top8_tiles"
TOPK_MERGE_SHADER_PREFIX = b"// Function: drowse_exact_top8_merge"
JLENS_TRANSPORT_SHADER_PREFIX = b"// Function: drowse_jlens_transport"
TOPK_WORKGROUP_LINE = b"@compute @workgroup_size(64, 1, 1)"
JLENS_TRANSPORT_WORKGROUP_LINE = b"@compute @workgroup_size(128, 1, 1)"
MAX_WEBGPU_STORAGE_BINDINGS_PER_STAGE = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_commit(repository: Path, expected: str, label: str) -> None:
    actual = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected:
        raise SystemExit(f"{label} checkout must be exact commit {expected}, found {actual}")


def verify_files(repository: Path, expected_files: dict[str, str], label: str) -> None:
    for relative_path, expected in expected_files.items():
        if sha256(repository / relative_path) != expected:
            raise SystemExit(f"{label} result digest differs for {relative_path}")


def verify_sources(mlc_repository: Path, tvm_repository: Path | None) -> None:
    """The exact MLC (and, when given, TVM) checkouts with the manifest's overlays applied."""
    require_commit(mlc_repository, MLC_BASE_COMMIT, "MLC-LLM")
    verify_files(mlc_repository, RESULT_DIGESTS, "MLC overlay")
    verify_portable_topk_source(mlc_repository)
    if tvm_repository is not None:
        require_commit(tvm_repository, TVM_COMMIT, "TVM")
        require_commit(tvm_repository / "3rdparty/tvm-ffi", TVM_FFI_COMMIT, "TVM-FFI")
        verify_files(tvm_repository, TVM_RESULT_DIGESTS, "TVM WebGPU overlay")


def verify_portable_topk_source(repository: Path) -> None:
    source = (repository / "python/mlc_llm/op/moe_misc.py").read_text(encoding="utf-8")
    if (
        source.count("TX = 1024") < 2
        or "T.buffer_store(local_top_k_index, -1, indices=[t])" not in source
        or "T.buffer_store(local_top_k_index, t, indices=[-1])" in source
    ):
        raise SystemExit("generic MLC MoE top-k differs from upstream scheduling or safe initialization")
    hooks = (repository / "python/mlc_llm/model/drowse_hooks.py").read_text(encoding="utf-8")
    for fragment in (
        "EXACT_READOUT_TOPK_BLOCK_SIZE = 256",
        "candidate_count = T.ceildiv(column_count, EXACT_READOUT_TOPK_BLOCK_SIZE)",
        '"drowse_exact_top8_tiles"',
        '"drowse_exact_top8_merge"',
        '"drowse_jlens_transport"',
        "_exact_readout_is_better",
        "while candidate_count > 1:",
        'for thread in T.thread_binding(0, 64, "threadIdx.x"):',
        "T.And(",
        "T.Or(",
    ):
        if fragment not in hooks:
            raise SystemExit(f"Drowse exact tiled top-8 omits {fragment}")
    exact_source = hooks.split("def exact_readout_topk", 1)[-1].split(
        "def structured_geometry_payload_layout", 1
    )[0]
    if 'scope="shared"' not in exact_source or "tvm_storage_sync" not in exact_source:
        raise SystemExit("Drowse exact tiled top-8 requires bounded cooperative reduction")
    for relative_path in (
        "python/mlc_llm/model/llama/llama_model.py",
        "python/mlc_llm/model/qwen3/qwen3_model.py",
    ):
        model = (repository / relative_path).read_text(encoding="utf-8")
        if (
            model.count("exact_readout_topk") != 5
            or model.count("transport_jlens_hidden") != 2
            or "op_ext.moe_misc.gating_topk(" in model
        ):
            raise SystemExit(f"{relative_path} does not route exact readouts through Drowse top-8")


def ensure_import_roots(mlc_repository: Path, tvm_repository: Path) -> None:
    expected = [(tvm_repository / "python").resolve(), (mlc_repository / "python").resolve()]
    import_roots = {Path(path).resolve() for path in sys.path if path}
    missing = [str(path) for path in expected if path not in import_roots]
    if missing:
        raise SystemExit(
            "exact source roots are missing from PYTHONPATH: " + os.pathsep.join(missing)
        )


def bootstrap_mlc_package(mlc_repository: Path, purpose: str) -> None:
    """Import `mlc_llm` from the checkout's sources only (no compiled `libmlc_llm`)."""
    if "mlc_llm" in sys.modules:
        raise SystemExit(f"mlc_llm was imported before the source-only {purpose} bootstrap")
    package = types.ModuleType("mlc_llm")
    package.__path__ = [str(mlc_repository / "python/mlc_llm")]
    package.__package__ = "mlc_llm"
    sys.modules["mlc_llm"] = package
    model_package = types.ModuleType("mlc_llm.model")
    model_package.__path__ = [str(mlc_repository / "python/mlc_llm/model")]
    model_package.__package__ = "mlc_llm.model"
    sys.modules["mlc_llm.model"] = model_package


def verify_emcc() -> Path:
    executable = shutil.which("emcc")
    if executable is None:
        raise SystemExit(f"emcc {EMCC_VERSION} is required in PATH")
    try:
        output = subprocess.check_output([executable, "--version"], text=True)
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"emcc {EMCC_VERSION} is required in PATH") from error
    first_line = output.splitlines()[0] if output else ""
    if f" {EMCC_VERSION} " not in f" {first_line} " or EMCC_REVISION not in first_line:
        raise SystemExit(
            f"emcc must be exact version {EMCC_VERSION} ({EMCC_REVISION}), found: {first_line}"
        )
    return Path(executable).resolve()


def require_llvm(tvm) -> None:
    if tvm.target.codegen.llvm_version_major() != 18:
        raise SystemExit("TVM compiler must be built against LLVM 18")


def vm_function_names_missing(wasm: bytes, functions) -> list[str]:
    """Functions whose VM metadata (length-prefixed name) the library lacks."""
    missing = []
    for function in sorted(functions):
        encoded = function.encode()
        if len(encoded).to_bytes(8, "little") + encoded not in wasm:
            missing.append(function)
    return missing


def verify_library_shaders(wasm: bytes, functions=REQUIRED_FUNCTIONS) -> None:
    """Every check on a compiled model library's container and emitted WGSL."""
    if not wasm.startswith(b"\x00asm"):
        raise SystemExit("compiler did not produce a valid WebAssembly container")
    missing = vm_function_names_missing(wasm, functions)
    if missing:
        raise SystemExit(f"compiled WebAssembly lacks Drowse VM function metadata: {missing[0]}")
    if b"local_top_k_index[-1i]" in wasm:
        raise SystemExit("compiled WebAssembly contains an out-of-bounds top-k shader store")
    if b"@compute @workgroup_size(1024" in wasm:
        raise SystemExit("dense launch-model WebAssembly contains an unexpected 1024-thread shader")
    verify_topk_shader_workgroups(wasm)
    verify_jlens_transport_shader(wasm)
    verify_geometry_shader_bindings(wasm)
    verify_drowse_shader_bindings(wasm)


def _shaders(wasm: bytes, prefix: bytes):
    """Yield `(name, header, body)` for each emitted WGSL kernel whose comment starts with prefix."""
    position = 0
    while True:
        start = wasm.find(prefix, position)
        if start < 0:
            return
        line_end = wasm.find(b"\n", start)
        next_kernel = wasm.find(b"// Function:", start + len(prefix))
        end = next_kernel if next_kernel >= 0 else min(len(wasm), start + 131072)
        compute = wasm.find(b"@compute", line_end, end)
        if line_end < 0 or compute < 0:
            raise SystemExit("compiled WebGPU shader metadata is truncated")
        name = wasm[start + len(b"// Function: ") : line_end].decode("ascii", "replace")
        yield name, wasm[start:compute], wasm[start:end]
        position = end


def verify_drowse_shader_bindings(wasm: bytes) -> None:
    """B1 §2.4: no Drowse kernel's emitted WGSL binds more than eight storage buffers."""
    kernels = 0
    for name, header, _ in _shaders(wasm, b"// Function: drowse_"):
        bindings = header.count(b"var<storage")
        if bindings > MAX_WEBGPU_STORAGE_BINDINGS_PER_STAGE:
            raise SystemExit(
                f"compiled WebGPU shader {name} uses {bindings} storage bindings; "
                f"the limit is {MAX_WEBGPU_STORAGE_BINDINGS_PER_STAGE}"
            )
        kernels += 1
    if kernels == 0:
        raise SystemExit("compiled WebAssembly lacks Drowse WebGPU shaders")


def verify_topk_shader_workgroups(wasm: bytes) -> None:
    counts = {}
    for label, prefix, expected_bindings in (
        ("tile", TOPK_TILE_SHADER_PREFIX, 3),
        ("merge", TOPK_MERGE_SHADER_PREFIX, 4),
    ):
        kernels = 0
        for name, header, shader in _shaders(wasm, prefix):
            if not name.endswith("_kernel"):
                raise SystemExit(f"compiled exact top-8 {label} shader name is invalid")
            workgroup = shader.find(b"@compute @workgroup_size(", 0, 2048)
            line_end = shader.find(b"\n", workgroup, 2048)
            if workgroup < 0 or line_end < 0 or shader[workgroup:line_end] != TOPK_WORKGROUP_LINE:
                raise SystemExit(f"compiled exact top-8 {label} shader does not use 64-thread tiles")
            bindings = shader[:workgroup].count(b"var<storage")
            if (
                bindings != expected_bindings
                or b"var<workgroup>" not in shader
                or b"workgroupBarrier();" not in shader
                or b"candidate_thread" in shader
                or b"array<f32, 2048>" in shader
                or b"array<i32, 2048>" in shader
                or b"var<workgroup> best_values : array<f32, 64>;" not in shader
                or b"var<workgroup> best_indices : array<i32, 64>;" not in shader
                or b"rank < 8i" not in shader
            ):
                raise SystemExit(
                    f"compiled exact top-8 {label} shader violates the bounded cooperative tiled "
                    f"schedule: bindings={bindings}"
                )
            kernels += 1
        counts[label] = kernels
    if any(count == 0 for count in counts.values()):
        raise SystemExit(f"compiled WebAssembly lacks exact tiled top-8 shaders: {counts}")


def verify_jlens_transport_shader(wasm: bytes) -> None:
    kernels = 0
    for _, _, shader in _shaders(wasm, JLENS_TRANSPORT_SHADER_PREFIX):
        workgroup = shader.find(b"@compute @workgroup_size(", 0, 2048)
        line_end = shader.find(b"\n", workgroup, 2048)
        bindings = shader[:workgroup].count(b"var<storage")
        if (
            workgroup < 0
            or line_end < 0
            or shader[workgroup:line_end] != JLENS_TRANSPORT_WORKGROUP_LINE
            or bindings != 3
            or b"var<workgroup>" not in shader
            or b"workgroupBarrier();" not in shader
            or b"fma(" not in shader
            or b"var<workgroup> partial : array<f32, 128>;" not in shader
        ):
            raise SystemExit(
                "compiled J-lens transport shader violates the bounded cooperative fp32 schedule: "
                f"bindings={bindings}"
            )
        kernels += 1
    if kernels == 0:
        raise SystemExit("compiled WebAssembly lacks the Drowse J-lens transport shader")


def verify_geometry_shader_bindings(wasm: bytes) -> None:
    kernels = [
        (name, header.count(b"var<storage"))
        for name, header, _ in _shaders(wasm, GEOMETRY_SHADER_PREFIX)
    ]
    if not kernels:
        raise SystemExit("compiled WebAssembly lacks a Drowse geometry WebGPU shader")
    for name, bindings in kernels:
        if bindings != MAX_WEBGPU_STORAGE_BINDINGS_PER_STAGE:
            raise SystemExit(
                f"compiled WebGPU shader {name} uses {bindings} storage bindings; "
                f"expected {MAX_WEBGPU_STORAGE_BINDINGS_PER_STAGE}"
            )
