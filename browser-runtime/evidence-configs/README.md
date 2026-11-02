# Release evidence producer configs

These files pin the exact observer and arguments used by the release evidence
recorder. Their repository path and SHA-256 are stored in each evidence record,
and their bytes are included in the producer tool version. Editing or renaming a
config invalidates evidence made with the old config.

Four authoring gates have direct, deterministic local observers:

- `fitting-kernel-parity.json`
- `topology-orchestration.json`
- `manifold-serialization.json`
- `shared-golden-results.json`

Hook ABI v5's runtime records (browser-runtime/README.md, "Release evidence") have one config
each, recorded at the commit that adopts the runtime lock (`evidence-configs.test.mjs` checks the
set against the lock):

- `cvm-goldens.json` (`cVmGoldens`): `record-cvm-goldens.py` on jobe, with the source-built
  toolchain's Python (`~/drowse-toolchain/venv`), W4a's MLC and TVM worktrees (overlays applied,
  pattern cap 16) and the four fixture models in `~/drowse-toolchain/work/w4a/fixtures`. Run it
  from a clean checkout of the adopting commit on jobe.
- `webgpu-goldens.json` (`webGpuGoldens`): `browser-units-goldens.mjs --report` once per fixture
  library through the vendored bridge, over `~/.drowse-parity/units-webgpu-p16` on the M5 (the
  four fixture models and their pattern-cap-16 libraries, `libs/SHA256SUMS`).
- `production-library-<id>.json` (`productionLibrary`), one per lock model, optional (a release
  records only the two golden gates; these diagnose one library): the physical harness
  at the model's smallest advertised context, without SAE; a `standard-v4` model with its R, S
  and units packs, a residual one with none. They run Chrome Beta (`--browser-channel
  chrome-beta`), the only Chrome on the M5; with stable Chrome installed, drop the flag.
- `production-q4-parity-<id>.json` (`productionQ4Parity`), one per q4f32_1 model, optional:
  `validate_mlc_q4_parity.py` on the CPU over a capture-schema-5 capture made beforehand with the
  repository lock (`npm run capture:hosted:residuals -- ID MODEL MODEL.wasm WEBLLM.js
  ../browser-runtime/fixtures/production-q4-parity-input-v1.json CAPTURE`). The producer runs
  `python3` from `PATH`, which must be the environment with Transformers 5 and torch.

The per-model configs read a fixed layout on the M5, one directory per lock model:

```
~/.drowse-parity/release/<id>/model        assembled publish directory (hosted-artifacts.json, model.wasm)
~/.drowse-parity/release/<id>/core         core pack (hosted-pack-artifacts.json)
~/.drowse-parity/release/<id>/jlens        R-lens pack   (standard-v4)
~/.drowse-parity/release/<id>/slens        S-lens pack   (standard-v4)
~/.drowse-parity/release/<id>/units        units pack    (standard-v4)
~/.drowse-parity/release/<id>/q4-capture   capture schema 5 (q4f32_1 models)
```

Each entry may be a symlink to where the release pipeline built it (for example
`~/.drowse-parity/packs/<id>/<sha16>/jlens`): the harness follows it, while the pack validators
(`validate-hosted-instrument-pack.mjs`) refuse a symlinked directory and take the real path. The
capture for a q4 config runs on the same browser (`--browser-channel chrome-beta`). Observers write nothing but their report: the
producer refuses any stderr output, so the harness echoes the page console only with
`--verbose`, which no config passes.

Run the observer without recording evidence by invoking the registered producer:

```bash
node webui/scripts/hosted-release-evidence-producer.mjs \
  --evidence-type fittingKernelParity \
  --config browser-runtime/evidence-configs/fitting-kernel-parity.json \
  --nonce 0000000000000000000000000000000000000000000000000000000000000000
```

Creating a candidate evidence record requires a committed clean worktree and
an immutable tested Drowse revision. Physical observer configs also require
the matching compiled model artifacts and hardware. These are optional
engineering records; the current CI workflow has no evidence-target dispatch
input or evidence attestation job.

The shared-golden observer runs both the Python and browser consumers for the
structured-program, measurement, loom-transcript, topology/RBF, sampling, and
reciprocal artifact fixtures. Use the recorder from a clean, committed revision to capture its results.
