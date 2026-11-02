# Drowse browser runtime

The hosted application must not infer with stock WebLLM. Production loading is
enabled only when `runtime-lock.json` has `status: "verified"` and the exact
package, overlay, toolchain, model, and compiled-library closure is pinned.

The runtime lock is `verified`. Exact converted-model revisions, compiled
libraries, package archive, overlay closure, runtime ABI, hook ABI, tokenizer,
and chat-template identities are bound by digest. The public signed catalog
contains only immutable packages that match those identities.

The browser core-pack compiler produces the required neutral whitener and seed
manifold pack for each release model. Every steering expression lowers to one
drowse-structured-v4 program (native's `synthesize_subspace` lowering, which
needs the neutral whitener); it shares Python goldens for Mahalanobis
normalization, mean-normalized per-layer share, and uncapped affine gain, and
lowers independent push/ablation groups, prior-step gates, response phases,
SAE and R-lens programs, box/custom/sphere curved manifolds, in-block unit
ablation, and exact geometry reads.
Curved steering retains its separate safety cap. Every published pack passes the
semantic validator and a physical WebGPU runtime check before catalog inclusion.

The checked-in TypeScript parser accepts the full Python steering grammar. The
current compiler supports single and composed affine pushes, mixed or
multiaxis mean-replacement ablation, Mahalanobis `~`/`|` projections,
response/prompt/count/thinking phases, prior-step scalar gates, and up to four
curved terms per layer. Curves may use bounded, unbounded, periodic, or spherical
domains and may have up to four intrinsic dimensions. Curved steering can itself
be phase-gated or controlled by a prior SAE, R-lens probability, scalar affine
probe, or exact geometry channel from the preceding token. The `standard-v2`
and `standard-v3` profiles emit `drowse-structured-v3`: in addition to four independently gated
affine groups, rank eight, eight scalar probes, four curves, thirty-two nodes per
curve, four intrinsic dimensions, and eight embedded dimensions, it reserves
eight geometry probes, whitener rank 96, thirty-three candidates per probe, and
a separate 41-float output record per layer and geometry probe. The older
`standard-v1` profile continues to emit `drowse-structured-v2` and fails closed
for multidimensional, curved, membership, and node-distance geometry reads.
`standard-v3` additionally attests `exact-readout-v1`, a fixed top-eight
full-vocabulary R-lens readout and chunked full-dictionary SAE readout. A
`standard-v2` library remains valid for geometry but is rejected before loading
a precomputed R-lens or SAE pack.

The v3 geometry path produces domain coordinates, subspace fraction,
normalized off-surface residual, tube membership, and raw whitened distance to
every candidate. The host derives normalized nearest-label distances and the
Gaussian assignment posterior from fixed, validated schema data. Coordinate,
fraction, membership, distance, and assignment gates therefore use exact
previous-token values instead of scalar approximations. Sphere fits use an
explicit domain tag and the same hyperspherical embedding, Jacobian, clamping,
log/parallel-transport/exp translation, and nearest-foot solve as the Python
domain. These TypeScript and compiled-kernel contracts are implemented but do
not by themselves constitute production-model or physical-device verification.

Hosted browsers start optional live R-lens and SAE readouts off on every device.
Users enable them through the instrument Live controls. Explicit probes, steering
gates, and token replay still request their required computations. Plain chat
therefore avoids full-vocabulary and full-dictionary analysis on every token,
including when the instrument sidebar is hidden; model sampling is unchanged.

Optional instrument packs are executable inputs rather than capability
placeholders. A local SAE v1 fp32 artifact provides `W_dec[id]` steering and the
exact `relu((h - b_dec) @ W_enc + b_enc)` probe used by `sae/<id>` gates. A
pack may also include `packs/sae/features.json`, an exact v1 sidecar with
`{format_version, model_id, release, features}` and sparse
`{label, max_act}` rows. The loader rejects the complete pack when any row is
malformed or outside the SAE width. When `max_act` exists, the GPU probe, gate,
and scalar reading use `activation / max_act`; discovery rows retain the raw
activation and carry the same label and unit. Packs without the sidecar retain
the Python-compatible raw-activation/null-metadata behavior. Live discovery
evaluates the complete installed encoder on the GPU in bounded feature chunks
and reports the exact global top eight; it does not train or modify the SAE. A
local R-lens v7 artifact provides its unchanged per-layer shards. The worker uses
the model's complete quantized or fp32 language head to compute an exact
full-vocabulary softmax on the GPU, reports exact per-layer and layer-aggregated
top-eight rows, and selects requested token probabilities for probes and gates.
Steering selectors are not limited to the pack's curated word
list: before synchronous lowering, the worker resolves each requested word
with the loaded tokenizer, requires the Python-compatible leading-space or bare
form to round-trip as exactly one token, and derives `W_U[v] @ J_l` through the
loaded GPU runtime in batches of at most eight token IDs. Partial R lenses are
passed in fitted source-layer order and partitioned into dynamic
`[chunk, hidden, hidden]` inputs bounded by the smaller of the adapter's buffer
and storage-binding limits. The runtime conservatively uses one matrix per
chunk because larger dynamic R-lens batches are not stable across supported MLC
WebGPU VMs. The worker uploads every validated fitted matrix exactly once when
it installs the structured hook program and keeps the chunks GPU-resident for
that program's lifetime. Each probability read or gate then makes one exact
full-vocabulary call per chunk, combines all readbacks behind one device
synchronization, and scatters the rows back to model-layer order.
Program replacement, clearing, unload, device loss, and worker termination
dispose the resident inputs. Loading an R-lens pack fails before generation if
even one fp32 Jacobian cannot fit either adapter limit. Instrument artifacts are
checked against the catalog closure, schema, tensor keys, fp32 shapes, layer
map, and digests before lowering. During program installation the bridge
borrows the validated host array only while constructing the bounded chunks; it
neither retains the complete lens nor re-uploads Jacobians on token reads.

Qwen chat models carry tokenizer-derived `<think>` delimiter IDs in the runtime
lock. Loading retokenizes the delimiters with the installed tokenizer and
attests the result before the generation state machine enables thinking. This
permits exact `@thinking` and `@after` control changes at streamed token
boundaries. Models without an attested thinking profile reject thinking-specific
programs before generation. The browser derives required features from every
expression and fails before generation when a loaded library or pack cannot
implement them exactly.

## Runtime source overlays

`forks/manifest.json` binds the current Drowse MLC-LLM, WebLLM, and TVM WebGPU
changes to exact upstream commits, base trees, patch bytes, and resulting file
digests. The `a9lim/mlc-llm-drowse` and `a9lim/web-llm-drowse` repositories may
mirror those result trees, but a mirror commit is not part of the executable
release identity. The checked-in overlay closure is reproducible without
depending on mutable repository state.

The runtime lock hashes this manifest and the compiler Dockerfile, repeats the
exact TVM, TVM-FFI, Emscripten, and LLVM identities, and binds the vendored
WebLLM archive by its exact SHA-256, package name, version, and repository. The
converted model manifests bind the compiled WebGPU libraries by SHA-256. A
published compiler image or fork mirror can be recorded when one exists, but
neither is required to run or reproduce the checked-in overlay and artifact
closure.

Apply an overlay only to a clean checkout at its exact upstream commit:

```bash
node browser-runtime/forks/apply-fork-overlay.mjs \
  mlc-llm-drowse /path/to/mlc-llm --check
node browser-runtime/forks/apply-fork-overlay.mjs \
  mlc-llm-drowse /path/to/mlc-llm --apply
```

Use `web-llm-drowse` for the WebLLM overlay. Apply the TVM backport before
building the compiler:

```bash
node browser-runtime/forks/apply-fork-overlay.mjs \
  tvm-webgpu-readonly /path/to/tvm --apply
```

The command rejects a different commit or tree, any tracked or untracked
worktree change, a modified patch, and any resulting file whose digest differs
from the manifest.

## Hosted repository provisioning

The runtime and distribution locks target the `stdlogits` Hugging Face namespace.
Changing this configuration does not transfer repositories or publish artifacts.
The pinned revisions must exist there, and the catalog must be signed for the new
artifact URLs before downloads can work. Check repositories without mutation, or
create any missing public repositories with the authenticated owner account:

```bash
cd webui
npm run provision:hosted:check
npm run provision:hosted:create
```

Weights do not depend on the hook ABI. A model whose conversion is already published moves to a
new runtime ABI without reconverting: `forks/build-production-webgpu.py rebind --model PUBLISHED_DIR
--model-id ID --architecture ARCH --quantization Q --output DIR` checks that the directory is the
runtime lock's conversion (its `tensor-cache.json` is the lock's `manifestSha256`), copies every file
its `drowse-build.json` lists, checked by SHA-256, and rewrites only that manifest; then
`compile --model DIR`. The four instruct q4
models (Gemma 3 1B and 4B, Qwen3 1.7B and 4B) ship this way: their published weights were adopted
and precision-promoted (`CONVERSION-SOURCE.json`), and `convert --quantization q4f32_1` would
produce different, bf16-stored scales and norms. Check a compiled library on the CPU with
`forks/verify-production-library.py --model DIR [--parity --reference-python PYTHON]` (the PyTorch
reference needs Transformers 5).

After converting and compiling a model, assemble its exact runtime closure and
validate it against `runtime-lock.json`:

```bash
npm run assemble:hosted:model -- CONVERTED_DIR MODEL.wasm PUBLISH_DIR
DROWSE_HOME=FITS_HOME ../browser-runtime/build-hosted-packs.sh MODEL_ID   # R, S, units, core statistics
npm run build:hosted:whitener -- MODEL_ID PUBLISH_DIR MODEL.wasm WEBLLM.js CORE_DIR \
  $(for c in ../browser-runtime/release-inputs/core-manifolds/*; do printf -- '--manifold-source %s ' "$c"; done) \
  --neutral-statistics PACKS/MODEL_ID/SHA16/core
npm run validate:hosted:core-pack -- MODEL_ID CORE_DIR
npm run build:hosted:sae -- MODEL_ID PUBLISH_DIR MODEL.wasm WEBLLM.js CORPUS.json SAE_DIR
npm run capture:hosted:residuals -- MODEL_ID PUBLISH_DIR MODEL.wasm WEBLLM.js INPUT.json CAPTURE_DIR
python3 ../browser-runtime/validate_mlc_q4_parity.py PUBLISH_DIR CAPTURE_DIR
npm run assemble:hosted:pack -- PACK_DIR
npm run validate:hosted:instrument-pack -- MODEL_ID sae|jlens|slens PACK_DIR
npm run validate:hosted:instrument-pack -- MODEL_ID units UNITS_DIR --with JLENS_DIR --with SLENS_DIR
npm run publish:hosted:model -- MODEL_ID PUBLISH_DIR
npm run publish:hosted:pack -- MODEL_ID core|sae|jlens|slens|units PACK_DIR
```

The production-q4 comparator checks the unsteered residuals of every layer and
position against the dequantized Transformers model, then the steered read on
hook ABI v5 (capture schema 4): one drowse-structured-v4 program with linear
probes on a fixed orthonormal sketch at every layer (the first direction the
push of a rank-1 affine group at the middle layer), measured at every captured
position with and without the group (`measurePreparedPositions`). Probe error
is reported against the activation scale: the absolute projection difference
over the larger residual norm times the probe-direction norm, so the 1% gate
stays well-defined when a projection is near zero. It separately requires the
control probes to reproduce the projection of the plain capture, rejects any
pre-enabled-layer delta between the control and steered probes, and compares
the steering delta's direction in the sketch.

`webui/scripts/browser-core-build.mjs` runs the core builder (the unchanged
`hosted-whitener.worker.ts`), the plain and measured captures and the
injection-arithmetic oracle on B4's tiny v5 fixture libraries:
`node scripts/browser-core-build.mjs --webllm WEB_LLM.js --fixtures DIR
--arch qwen3|gemma3 --quantization q0f32|q4f32_1 --work SCRATCH`.

Candidate libraries that do not yet match the checked-in production lock can be
captured against an explicit engineering lock without editing repository state:

```bash
npm run prepare:hosted:candidate-lock -- MODEL_ID PUBLISH_DIR /tmp/candidate-runtime-lock.json
npm run capture:hosted:residuals -- MODEL_ID PUBLISH_DIR MODEL.wasm WEBLLM.js INPUT.json CAPTURE_DIR \
  --runtime-lock /tmp/candidate-runtime-lock.json
npm run build:hosted:whitener -- MODEL_ID PUBLISH_DIR MODEL.wasm WEBLLM.js CORE_DIR \
  $(for c in ../browser-runtime/release-inputs/core-manifolds/*; do printf -- '--manifold-source %s ' "$c"; done) \
  --neutral-statistics PACKS/MODEL_ID/SHA16/core \
  --runtime-lock /tmp/candidate-runtime-lock.json
npm run validate:hosted:core-pack -- MODEL_ID CORE_DIR \
  --runtime-lock /tmp/candidate-runtime-lock.json
```

The preparation tool starts from the repository lock, validates the assembled
model closure, and writes a new lock with only that model's library digest
replaced. It refuses to overwrite an existing file. The capture records both the
exact-byte and canonical status-free lock digests and marks the source as
`override`. It revalidates the complete lock schema, model identity, build
provenance, and every local artifact before Chrome starts. An override capture
can be compared locally, but the parity validator refuses to turn it into a
release-evidence receipt.

Instrument builders can optionally record a corpus closure produced by
`prepare_hosted_release_corpus.py` for research bookkeeping:

```bash
npm run build:hosted:sae -- MODEL_ID PUBLISH_DIR MODEL.wasm WEBLLM.js \
  RELEASE_DIR/sae-corpus.json SAE_DIR \
  --release-corpus RELEASE_DIR/release-corpus.json \
  --release-corpus-source APPROVED-SOURCE.json \
  --release-license-evidence LICENSE-EVIDENCE
npm run validate:hosted:instrument-pack -- MODEL_ID sae|jlens|slens PACK_DIR --release
```

When supplied, the builders verify and copy those bookkeeping files. They are
not required for catalog generation or upload. Production admission instead
uses the ordinary instrument validator: exact runtime/context compatibility,
safe tensor shapes and dtypes, finite values, readable manifests, and successful
loading by the browser instrument registry.

The offline SAE pack builder mirrors Python's local-training default: it rounds 65%
model depth with Python's ties-to-even rule, then resolves that slot through the
runtime lock's `layerMap`. The current locks select layer 18 for Qwen3-1.7B and
layer 23 for Qwen3-4B. An explicit `--layer` overrides the default and must name
a layer in that map.

The publisher is check-only unless `--upload` is supplied. Uploading creates a
Hugging Face commit, so it must be an explicit release action:

```bash
npm run publish:hosted:model -- MODEL_ID PUBLISH_DIR --upload
npm run publish:hosted:pack -- MODEL_ID core|sae|jlens|slens|units PACK_DIR --upload
```

Record the returned immutable revision as `convertedRevision`, generate the
catalog artifact list from that revision, and rerun the distribution preflight.
Model publication alone is insufficient: the matching required core pack,
signed catalog, and distribution lock must also be published before production
loading is enabled. Precomputed R-lens, S-lens, units and SAE packs are optional
model-specific capabilities and are listed only when a compatible release already exists.

The whitener builder is a physical WebGPU release tool. It fits the reduced
Mahalanobis representation with the shipped Rust/WASM kernel
(`src/hosted/fitting/neutralWhitener.ts`) and writes separately bound 2K and 4K
`neutral-whitener` pairs. For Gemma 3 and Qwen 3, which drowse fits natively,
`--neutral-statistics` is required: it names `build_hosted_packs.py`'s `core/`
(unit statistics and the `out` and `mid` means of drowse's neutral pass, and the pass's
activation rows, `neutral-rows.safetensors`), and the whitener is fitted from those
native rows, the ones drowse's own whitener and so the units pack's write norms are built
from. The shipped `neutral-statistics.json` then names the rows and each context's
whitener digest (`whitener`), and the units pack binds the rows' digest, so the browser
refuses units over any other whitener. For GPT-2, Pythia and Qwen3.5 base, which have no
native neutral pass, the builder renders the bundled 96-row neutral corpus through the
exact q4 browser runtime and fits from those captures. It refuses an existing output
directory and checks the model manifest, runtime identity, layer map, transfer digest,
and context bindings. Each `--manifold-source` (repeat it, one per concept) names a v10
PCA discover folder; the builder captures its nodes through the same q4 runtime and fits
them as the in-app fit does (a one-node concept folds against the neutral mean,
`fitting/monopolar.ts`, as drowse's `_fit_monopolar` does), producing one browser-fitted
`.drowse` per concept and context, the core concepts first-run steering and the probe
rack start from. The independent validator (`validate:hosted:core-pack`) checks every
context closure, that each context carries exactly the core roster
(`release-inputs/core-manifolds/`), that a natively fitted model carries the neutral
statistics bound to every context's whitener, and every browser archive before the pack
manifest is assembled; `--neutral-rows PACKS/MODEL_ID/SHA16/core/neutral-rows.safetensors`
also refits the whitener from the rows and requires the pack's.

The core concepts are drowse 6.0's default roster: every bundled concept tagged in a
default probe category (epistemic, alignment, register, cultural; `session.py`
`PROBE_CATEGORIES`), seventeen in all, among them the one-node `deceptive`,
`refusing` and `manipulative`, which fold against the neutral mean.
`webui/scripts/sync-core-concepts.mjs` copies them from the pinned submodule into
`release-inputs/core-manifolds/` (rerun it after a pin bump; the release-tool tests
fail while the copies drift). Beside them sits the hosted starter `welcoming.detached`
(original text, CC0-1.0), which the recorded landing demo steers. Every source's
responses are aligned to the canonical baseline prompts, carry machine-readable
authorship and license provenance, and contain no fitted model tensors. Each build
fits them in the browser runtime itself, so every archive is bound to that runtime's
identity and context and to the whitener the same build writes.

The SAE builder is offline developer release tooling and is never shipped as a
hosted user feature. It renders a release
corpus through the same exact q4 browser runtime, captures explicit token
positions at the selected post-block layer, streams the fp32 activations to a
bounded local file, trains only the ReLU SAE parameters outside the hosted
application, emits the existing local-v1 schema, assembles its immutable file
closure, and reopens the first and last features through the production browser
registry. `--feature-metadata FEATURES.json` adds the validated optional
sidecar to that closure. Raw activation rows remain local and are removed after
the validated pack is produced.

## Lens, units and core packs

Hosted lenses are drowse's own fits. For each model with units (the six whose blocks
drowse 6.0 describes: the four Gemma 3 and two Qwen3 models), drowse fits R and S at
`out` and `mid` (`drowse lens fit`, `fit_lens`) over the pinned FineWeb-Edu corpus
(`release-corpus/native-lens-corpus.json`: 64 documents of 128 tokens), counts the
corpus's tokens (`lens counts`) and runs the neutral pass (`unit stats`), all in fp32 on
the weights the browser runs: the dequantized q4 export, or for the q0f32 Gemma 3 270M
its source checkpoint, bit for bit. The recipe and its runbook are in the parity
handoff (`B2-fits.md`); lens fitting is never part of the hosted application.

`build_hosted_packs.py` turns one model's native artifacts into packs bound to one
runtime identity, without refitting; `build-hosted-packs.sh MODEL_ID` runs it with the
identity `webui/scripts/hosted-runtime-identity.mjs` reads from the runtime lock,
assembles each pack, reads it back through the browser registry and records sizes
(`hosted-pack-sizes.mjs`). Output goes to a staging directory,
`~/.drowse-parity/packs/MODEL_ID/<runtime sha256 prefix>/`; nothing is published.
W4's recompiled libraries change the runtime identity: rerun the script with the new
lock. The costly part, the units contrast statistics, is cached by drowse in its home,
so a rebind takes seconds.

- **`jlens`** (the R family): every layer of R, `out` L0..n−2 (the last block's `out`
  read is the logit lens) and `mid` L0..n−1, one shard per (site, layer)
  (`<site>.layer-<L>.safetensors`, key `layer_<L>`), the native files byte for byte, so
  each shard's digest is the native manifest's. The v7 manifest is drowse's with three
  fields bound to the runtime: `model_fingerprint` (the runtime identity's digest),
  `model_source_fingerprint` (its source revision) and `model_layer_count`. The pack's
  `pack.json` keeps the native identities. The curated browser vocabulary rides along.
- **`slens`** (the S family): every layer of S at both sites, and the S embedding
  statistics (B1 §5.1): `mean_E`, `Σ_E` (`embedding-statistics.safetensors`), the token
  mask as little-endian u32 words, bit `v & 31` of word `v >> 5` set when token `v` is
  kept (`embedding-statistics.mask.u32`), and `embedding-statistics.json` with the mask's
  description (`{source: "corpus", detail, kept}`), the counting tokenizer's
  fingerprint and file digest (`tokenizer_sha256`, the runtime's: the browser refuses the
  statistics under another tokenizer) and the embedding scale. They are exactly the
  session's `TokenEmbeddings`.
- **`units`**: per layer `reads.layer-<L>` (population scalars per lens and population,
  head-norm derivatives, whitened write norms) and `contrast.layer-<L>` (the `[V]`
  statistics only `hide_hubs` reads), both straight from
  `UnitReader.population_artifacts(layer)`; the null basis model-wide; `units.json` binds
  the runtime, the R and S packs (`lens_identity` of their manifests), the S statistics
  and the neutral pass every statistic averages over, with the digest of its rows (the
  write norms' whitener, and the core pack's, are built from them). `build_hosted_packs.py`
  refuses a layer without `neurons.write_norms`. Rank-3 tensors are stored
  `[prod(shape[:-1]), shape[-1]]` with their logical shapes recorded. On the ABI v5 tiny
  fixture every shipped units tensor equals the `unit-artifacts` golden on both
  architectures (`tests/test_hosted_packs.py`).
- **core statistics**: `core/neutral-statistics.*` (unit statistics, `out` and `mid`
  means, stacked over layers) and `core/neutral-rows.safetensors` (the neutral pass's
  rows, a build input that does not ship), added to the core pack by the whitener
  builder above, which fits the core whitener from the rows.

**Full depth.** Every layer of R and S at both sites ships, so unit reads, the anatomy
and every head read reach every layer as natively; the eight-layer cap is gone. A lens or
units pack installs its manifests, model-wide statistics and the live readout's R:out
layers (up to eight, evenly spaced over the blocks before the last, within 448 MiB:
`lensReadoutLayers`, the readout's rule since the first lens packs), so the readout runs
offline after an install as before. Every other layer file is fetched the first time a
read needs it, verified against the signed catalog closure and added to the pack's
install (`packFileResidency`, `VerifiedArtifactDownloader.downloadPackFile`, the
worker's `fetchFile`); the registry keeps recently read layers decoded
(`BrowserInstrumentRegistry.lensMatrix`, `embeddingStatistics`, `unitsLayer`).

| model | R pack (installs) | S pack (installs) | units | core statistics |
|---|---|---|---|---|
| gemma3-270m-instruct | 55.0 MiB (12.8) | 57.9 MiB (1.6) | 230 MiB (0.1) | 0.6 MiB |
| gemma3-1b-instruct, gemma3-1b-pt | 259 MiB (41.1) | 268 MiB (5.1) | 339 MiB (0.1) | 2.4 MiB |
| qwen3-1.7b | 881 MiB (129) | 912 MiB (16) | 240 MiB (0.2) | 2.6 MiB |

One lens layer is `d² · 4` bytes (1.6, 5.1, 16 and 25 MiB for d = 640, 1152, 2048, 2560); a
units layer is about 11.4 MiB of contrast statistics on Gemma's 262k vocabulary (6 MiB on
Qwen's 152k) plus its reads. The 4B models' rows follow their fits (the parity handoff's
`lanes/B6.md`).

After building a model, core pack, and R-lens pack, the physical instrument
harness loads all three through the same browser loaders, enables live R-lens
probabilities, transfers the complete per-layer Jacobian program once into
bounded GPU-resident chunks, and generates with an R-lens steering term without
making an external request:

```bash
npm run test:browser-runtime:instruments -- \
  --model-id qwen3-1.7b \
  --model-directory MODEL_DIR --model-library MODEL.wasm --webllm WEBLLM.js \
  --core-directory CORE_DIR --jlens-directory JLENS_DIR \
  --sae-directory SAE_DIR
```

Use the full production-model harness for one fail-closed Chrome pass over the
complete local runtime path:

```bash
npm run test:browser-runtime:full -- \
  --model-id qwen3-1.7b \
  --model-directory MODEL_DIR --model-library MODEL.wasm --webllm WEBLLM.js \
  --core-directory CORE_DIR --jlens-directory JLENS_DIR \
  --sae-directory SAE_DIR --context-tokens 2048 \
  --output /tmp/drowse-real-browser-e2e.json
```

The command rehashes every local manifest object before opening Chrome, requires
a confirmed hardware adapter with at least 10 storage buffers per shader stage,
then measures ordinary generation, selective
post-block capture, exact forced-token replay and compact cross-token scoring,
an alternate-token fork, flat steering gated independently by geometry, R-lens
probability, and SAE activation signals, R-lens and SAE steering/readout,
a small spectral fit backed by the OPFS activation spool, nearest-node role
selection and named-role generation, curved steering,
streamed stop, unload/reload, and a final reload from reopened OPFS files while
all HTTP(S) fetches are blocked. It fails when any phase is absent, reordered,
unsupported, unmeasured, or emits a WebGPU/device-loss error. There is no
`--skip-fitting` or missing-instrument mode.

Its JSON is deliberately marked `releaseEvidence: false`: one local desktop
run is useful physical integration evidence, but it cannot claim Android,
cross-platform, numeric-parity, or full release-matrix results. Release records
still go through `record:hosted:evidence` and the revision-bound evidence
schemas.

Build the catalog from an exact release spec only after every model and pack has
an immutable revision. Model directories contain `hosted-artifacts.json`; each
core or optional-instrument directory contains `hosted-pack-artifacts.json`
with exact relative paths, byte sizes, and SHA-256 digests. The builder rehashes
every file, closes all runtime-lock context profiles, derives runtime/context
identity digests, and validates the completed catalog with the browser parser:

```bash
npm run build:hosted:catalog -- RELEASE_SPEC.json catalog.json
node scripts/sign-catalog.mjs catalog.json PRIVATE_KEY.pem KEY_ID catalog.sig.json
npm run publish:hosted:catalog -- catalog.json catalog.sig.json
```

Apply `tvm-webgpu-readonly` to the pinned TVM checkout before building its
compiler. Its read-only buffer fix was merged as Apache TVM PR 20113. Without
that fix, typed buffer aliases are emitted as `read_write`, and current Chrome rejects
the generated paged-KV shader because a `workgroupBarrier` appears under control
flow derived from that storage binding.

The same overlay also carries a local attention numerical fix, separate from
that upstream PR: GPU prefill and decode initialize their running maximum at
the minimum finite float32 value and assign masked positions exactly zero
softmax weight. The former `-50000` cutoff is above valid attention scores in
Pythia-70M, corrupting both attention normalization and masked-token exclusion.
The compiler verifies the exact hashes of both modified kernel sources.

MLC commit `9fa644f5` merged typed-buffer compiler passes after its checked-in
TVM gitlink was last updated. Those passes explicitly depend on Apache TVM
`68ba2b31`, so the feasibility toolchain pins that external TVM commit and its
`tvm-ffi` submodule rather than incorrectly compiling against MLC's older
gitlink. The overlay manifest records the exact TVM base, upstream WebGPU
backport, TVM-FFI, Emscripten, and LLVM identities.

The optional exact-source MLC check requires that pinned source-built TVM
compiler. It fails when the toolchain is unavailable instead of reporting a
skipped gate:

```bash
TVM_DEVICE_BACKEND_AUTOLOAD=0 \
TVM_LIBRARY_PATH=/path/to/exact-tvm-build/lib \
SKIP_LOADING_MLCLLM_SO=1 \
PYTHONPATH=/path/to/tvm/python:/path/to/mlc-llm/python \
  /path/to/venv/bin/python browser-runtime/forks/verify-mlc-hook.py \
    --mlc-repository /path/to/mlc-llm \
    --tvm-repository /path/to/tvm
```

That check proves the exact Qwen3, Llama, and Gemma 3 graph exports contain the four
steering and four selective-capture entry points and executes the shared
rank-one and fp32 capture goldens through a compiled C TVM VM. It does not
prove WebGPU, q4, lifecycle, long-prefill, or physical-device feasibility.

For the WebGPU compile gate, first source exact Emscripten 3.1.56 and rebuild
both support runtimes against the pinned TVM checkout:

```bash
export TVM_SOURCE_DIR=/path/to/tvm
make -C /path/to/mlc-llm/web clean
make -C /path/to/tvm/web clean
(cd /path/to/mlc-llm && ./web/prep_emcc_deps.sh)
```

Then compile each architecture fixture with the LLVM-18-enabled TVM compiler:

```bash
TVM_DEVICE_BACKEND_AUTOLOAD=0 \
TVM_LIBRARY_PATH=/path/to/tvm-build/lib \
PYTHONPATH=/path/to/tvm/python:/path/to/mlc-llm/python \
  /path/to/venv/bin/python browser-runtime/forks/compile-tiny-webgpu.py \
    --mlc-repository /path/to/mlc-llm --tvm-repository /path/to/tvm \
    --architecture qwen3 --output /tmp/drowse-tiny-qwen3.wasm
```

Repeat with `--architecture llama` and `--architecture gemma3_text` for the
architecture regression fixtures. The compiler requires a new output path,
validates the resulting WebAssembly with the matching Emscripten Binaryen tool,
and verifies that every rank-one, structured, curved, capture, and R-lens VM
function remains in the library. The three launch models use `q4f16_1`; pass
`--context-window-size 4096` for the long-prefill Qwen3 fixture.

Create the matching deterministic tensor cache with the same architecture,
quantization, and context arguments. The fixture uses nonzero embeddings and
unit normalization weights so the browser gate exercises an actual rank-one
residual change; q4 caches use valid offset-packed int4 values and scales:

```bash
TVM_DEVICE_BACKEND_AUTOLOAD=0 \
TVM_LIBRARY_PATH=/path/to/tvm-build/lib \
PYTHONPATH=/path/to/tvm/python:/path/to/mlc-llm/python \
  /path/to/venv/bin/python browser-runtime/forks/create-tiny-webgpu-model.py \
    --mlc-repository /path/to/mlc-llm --tvm-repository /path/to/tvm \
    --architecture qwen3 --quantization q4f16_1 \
    --output /tmp/drowse-tiny-qwen3-model
```

After building the WebLLM overlay, run the compiled library through a dedicated
browser worker:

```bash
node webui/scripts/browser-runtime-feasibility.mjs \
  --webllm /path/to/web-llm/lib/index.js \
  --model /tmp/drowse-tiny-qwen3-model \
  --library /tmp/drowse-tiny-qwen3.wasm \
  --architecture qwen3 --quantization q4f16_1 --reloads 3
```

Pass `--browser-channel chrome` or `--browser-channel msedge` to run the same
gate against an installed production browser instead of Playwright's bundled
Chromium.

The harness fails on uncaptured WebGPU errors and checks ordinary generation,
rank-one and structured affine injection, SAE probes, live gates, simultaneous
curves, multidimensional and periodic curves, full-vocabulary R-lens
probabilities, hook clearing, streamed stop, repeated load/unload, and worker
termination. For the shape-cache regression, compile the Qwen3 q4f16 fixture
at a 4096-token context and run separate
`--prompt-tokens 2048` and `--prompt-tokens 4095` cases. That result belongs to
the Qwen long-prefill regression; Qwen3-1.7B and Qwen3-4B also have independent
tiny-fp32 and production-q4 parity coverage.

Pass `--verified-artifacts` to preload each fixture object exactly once and
provide it to the fork through a caller-owned, read-only artifact cache. This
exercises configuration, tokenizer, model-library, and tensor-shard loading
without allowing WebLLM to create a second browser cache. The harness rejects
any missing object, any unexpected URL, or a repeated network fetch. This mode
uses a worker-side configured `MLCEngine`; caller-owned caches are deliberately
not sent through structured-clone worker messages.

Add `--selected-adapter --reloads 1` to prove that the production device is
requested from the exact adapter supplied by Drowse. WebGPU adapters are
single-use for device creation in current Chromium, so the hosted capability
gate destroys its calibration device, then acquires a fresh adapter and rejects
it unless the hardware status, identity, features, and bucketed limits match the
checked adapter. Repeated load/unload testing likewise requests a fresh adapter
per load.

These deterministic tiny checks prove fp32 and q4 browser execution only. They
do not satisfy production-model numeric parity, Android, device-loss, or the
physical release matrix.

Production-size compilation must also exercise a launch model with its real
layer count. A two-layer fixture cannot detect per-stage WebGPU binding growth.
Probe and capture aggregation therefore write every layer into one in-place
output tensor; returning a single `concat` over all layer tensors is forbidden
because a 28-layer Qwen model would require 29 storage bindings in one shader,
above the baseline WebGPU per-stage limit. The physical harness discovers all
tokenizer files and tensor shards from the converted metadata and requires each
caller-owned artifact to be fetched exactly once.

Every converted `mlc-chat-config.json` must also contain a sorted, unique
`drowse_capture_special_token_ids` array produced from the source tokenizer's
complete `all_special_ids` and `added_tokens_encoder` union. Browser loading
checks that this list covers every `tokenizer.json` added-token ID and each
configured BOS, EOS, and padding ID before capture can run. This prevents
last-content pooling from silently depending on tokenizer heuristics.

`distribution-lock.json` independently gates the signed catalog and download
origins. It embeds the current and next Ed25519 public keys, the minimum catalog
sequence, and the exact Hugging Face artifact redirect origin exercised by the
full distribution preflight. Its verified state authorizes catalog downloads.

`runtime-feasibility-evidence.json`, `authoring-evidence.json`, and
`benchmark-evidence.json` remain optional engineering records. They are useful
for regression tracking and support-policy decisions, but their status,
producer registrations, source ancestry, and matrix completeness do not enable
or block downloads, generation, fitting, or a release build.

The current launch catalog contains Gemma 3 270M, Qwen3-1.7B, and Qwen3-4B. Each release model
must include an already-released compatible R-lens, selected automatically in
the first-run flow. The UI advertises only context profiles physically exercised for each
model. Authoring-evidence ledgers and a fixed 48-cell benchmark matrix are not
download gates, but each published model still requires a real-browser load,
generation, steering, unload, and offline-reload pass. Models that list an
existing R-lens or SAE pack also require the matching browser instrument pass.

## Release transaction

1. Exercise the advertised context profiles for all three launch models.
2. Publish each model's immutable converted files, required core pack, and an
   already-released exact-model R-lens that passes the browser pack validator.
   Add compatible precomputed SAE packs when available; never fit either
   instrument as part of the browser release process.
3. Fill the model revision and artifact SHA-256 fields in `runtime-lock.json`
   and set its status to `verified`.
4. Generate and sign the catalog, fill the exact Hugging Face redirect origins,
   and set `distribution-lock.json` to `verified`.
5. Run the release checker and distribution preflight:

   ```bash
   node webui/scripts/check-runtime-lock.mjs --release
   node webui/scripts/preflight-hosted-distribution.mjs
   ```

Parity, authoring, lifecycle, and cross-platform runs remain ordinary regression
tests and local reports; they are not part of the release transaction.

`runtimeIdentitySha256` is SHA-256 over canonical, recursively key-sorted JSON
for the runtime identity record. A context binding is the same digest over
`{"contextTokens":N,"runtimeIdentitySha256":"..."}`. Catalog validation
recomputes both values before a pack can be activated.

Each runtime identity digest is bound to exactly one catalog artifact with the
matching `converted_manifest`, `tokenizer`, `chat_template`, or
`model_library` role. Weight shards are closed by the converted manifest; a
signed identity claim that does not match those artifact digests is rejected.

The catalog signing key is never stored in this repository. The signing tool
accepts an Ed25519 PKCS#8 PEM key path and signs the exact `catalog.json` bytes.
The publisher verifies the currently published catalog against the lock's keys
before it advances the sequence. A rotation that cannot sign with the outgoing
key replaces both keys in the lock and passes the lock the published catalog was
signed under: `--previous-lock OLD_DISTRIBUTION_LOCK`. Builds with the old lock
reject the new catalog, so deploy the new build right after publishing.
## Cloudflare Pages deployment

Cloudflare credentials, Pages project creation, Access policy, DNS, deployment,
and production publication are external release gates. The repository builds and
verifies static output but does not perform any of those actions automatically.
`webui/wrangler.jsonc` names `webui/dist-hosted` as the upload directory and
contains no credentials.

### Access-controlled preview

1. From the repository root, run `npm ci`, then `npm run build:hosted` from `webui`. This always emits the
   feasibility-preview copy, `X-Robots-Tag: noindex, nofollow`, and a blocking
   `robots.txt`.
2. Upload `webui/dist-hosted` to a Cloudflare Pages preview deployment and put
   the preview hostname behind a Cloudflare Access policy. Do not promote this
   build to the public production hostname.
3. Verify the deployed hostname, including Pages-applied headers, protection of
   `/`, `/app`, the manifest, and service worker from unauthenticated requests,
   and offline navigation:

   ```bash
   npx playwright install chromium
   export CF_ACCESS_CLIENT_ID=...
   export CF_ACCESS_CLIENT_SECRET=...
   npm run verify:hosted:deployment -- https://PREVIEW_HOST --channel preview
   ```

   For non-interactive verification behind Access, provision a scoped service
   token outside the repository and expose both `CF_ACCESS_CLIENT_ID` and
   `CF_ACCESS_CLIENT_SECRET` only to this command. The verifier sends the pair
   to same-origin requests, browser navigation, and service-worker requests;
   cross-origin browser requests are blocked so the token cannot leave the
   preview origin, and the verifier never prints it.

### Public release

1. Complete the runtime artifact, signed distribution, and hosted security
   checks above. Authoring evidence, source ancestry attestations, and a complete
   physical benchmark matrix are not required.
2. From the repository root, run `npm ci`, then `npm run build:hosted:release` from
   `webui`. The release
   build fails closed unless the locks and distribution preflight pass, emits
   public indexing policy, and records the exact 40-character source revision.
3. Upload the resulting `webui/dist-hosted` without rebuilding it to an
   Access-protected Pages preview. Configure a preview-host-only Cloudflare
   Response Header Transform Rule that sets `X-Robots-Tag: noindex, nofollow`;
   do not apply that rule to the production hostname. Provision a scoped Access
   service token outside the repository and run the candidate check. This
   verifies release metadata while requiring both Access protection and the
   externally applied noindex header:

   ```bash
   npx playwright install chromium
   export CF_ACCESS_CLIENT_ID=...
   export CF_ACCESS_CLIENT_SECRET=...
   npm run verify:hosted:deployment -- https://RELEASE_CANDIDATE_HOST \
     --channel candidate
   ```

4. Publish the same byte-identical `dist-hosted` artifact to the production
   environment and remove Access only from the intended public hostname. Unset
   the service-token environment variables, then prove the public deployment is
   anonymously reachable and no longer carries noindex:

   ```bash
   unset CF_ACCESS_CLIENT_ID CF_ACCESS_CLIENT_SECRET
   npm run verify:hosted:deployment -- https://RELEASE_HOST --channel release
   ```

The post-deploy verifier accepts only an HTTPS origin. Preview and candidate
checks require Access service credentials and prove that the shell's entry
routes and bootstrap assets deny unauthenticated requests. Candidates must have
release metadata, an independently supplied source revision, and noindex.
Public release checks forbid Access credentials and noindex. `--revision` is an
optional spot check, not a release prerequisite. Every mode checks `/`, `/app`,
the complete deployed COOP/COEP/CORP/CSP/Permissions-Policy set,
indexing policy, the exact required PWA icon set, asset MIME and cache headers,
route rendering, `crossOriginIsolated`, service-worker readiness, and a
never-before-fetched offline `/app` URL. A successful local build does not
substitute for this check.
