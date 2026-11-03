# Browser compiler image

Build the exact native Linux feasibility toolchain from the repository root:

```bash
docker build \
  --file browser-runtime/compiler/Dockerfile \
  --tag drowse-browser-compiler:feasibility \
  browser-runtime
```

The recipe supports `linux/amd64` and `linux/arm64` builds. `linux/arm64` links TVM against the
pinned official LLVM 18.1.8 archive. `linux/amd64` builds LLVM from tag `llvmorg-18.1.8`
(commit `3b5b5c1e`, `forks/manifest.json` `llvmSourceBuild`) with the pinned x86_64 archive's own
clang, configured like the archive (Release, its 19 targets, assertions/RTTI/EH off): that
archive's libraries were compiled against GCC 7's libstdc++, whose `std::optional` ABI no C++20
libstdc++ shares, so a TVM linked against them segfaults on import. The lock binds the tag and
commit (`toolchain.llvmSourceBuild`) and `check-runtime-lock.mjs` requires the Dockerfile's
`LLVM_SOURCE_*_AMD64` arguments to match. The jobe toolchain the hook ABI v5 libraries were
compiled with is the same build (parity handoff `B40-toolchain.md` §3, §8); the image itself has
not been rebuilt since.
Emscripten 3.1.56 does not publish Linux ARM64 binaries, so the ARM64 image runs only
the pinned Emscripten tools through AMD64 translation. On Apple silicon with Colima,
use Rosetta for that narrow compatibility layer:

```bash
colima stop
colima start --vm-type vz --vz-rosetta
```

The build pins the Ubuntu base digest, LLVM archives and source, Emscripten SDK commit,
TVM/TVM-FFI sources, MLC source, and both Drowse overlays. Its final build gate runs the C VM
structured, geometry and readout goldens, then compiles fp32 and q4 Qwen3, Llama and Gemma 3
WebGPU fixture libraries under the v5 export check. The units goldens need the fixture models
`forks/build-units-fixtures.py` makes from drowse's checkpoints; they run outside the image
(`browser-runtime/record-cvm-goldens.py`, README "Release").

The local image ID is feasibility evidence only. A production runtime lock must
name an immutable registry reference and manifest digest for the published
image; it must not use a local image ID.

The image also contains `/opt/drowse/build-production-webgpu.py`. Its `convert`
command accepts only an exact Hugging Face snapshot whose required files carry
the requested 40-character revision in the local Hub metadata. It writes a new
q4f16 tensor cache, tokenizer/config closure, source notices, the complete
model-license text, and a
content-addressed `drowse-build.json`; it never overwrites an output directory.
The `compile` command re-verifies that complete manifest, compiles into a
same-filesystem staging directory, rejects a library missing any Drowse hook or
capture export, and promotes only the validated WASM to the requested path.

The pinned TVM/MLC pipeline does not currently emit byte-identical WASM across
otherwise identical invocations: serialized VM and generated-function ordering
can vary while the verified model behavior remains unchanged. Treat the emitted
SHA-256 as the identity of that exact release candidate. Do not promote the
runtime lock until either the compiler output is canonicalized reproducibly or
the release process records and physically validates one immutable emitted
library per model variant.

Every hook ABI v5 library is converted from its pinned source snapshot and compiled with this
overlay (no upstream MLC closure is adopted): `convert` records `drowse-web-runtime-v2`,
`block-sites-v5` and the architecture's profile (`standard-v4` for Gemma 3 and Qwen3,
`standard-v4-residual` otherwise) in `drowse-build.json`, and `compile` refuses a manifest from
another ABI:

```bash
docker run --rm -v /absolute/snapshot:/source:ro -v /absolute/output:/output \
  drowse-browser-compiler:feasibility \
  'python /opt/drowse/build-production-webgpu.py \
    --mlc-repository /opt/mlc-llm --tvm-repository /opt/tvm convert \
    --source /source --source-repository Qwen/Qwen3-4B --source-revision SOURCE_REVISION \
    --architecture qwen3 --quantization q4f32_1 \
    --context-window-size 4096 --prefill-chunk-size 2048 --output /output/qwen3-4b \
  && python /opt/drowse/build-production-webgpu.py \
    --mlc-repository /opt/mlc-llm --tvm-repository /opt/tvm compile \
    --model /output/qwen3-4b --architecture qwen3 --quantization q4f32_1 \
    --output /output/qwen3-4b/qwen3-4b-webgpu.wasm'
```

The compile runs the export check (the overlay's v5 function sets, `overlay_checks.py`) and
the shader checks. The production recipe per model (architecture, quantization, conversation
template) and its gates are in the parity handoff (`lanes/B4.md`, "Production compile recipe").
