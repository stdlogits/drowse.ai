# Hook ABI v5: `block-sites-v5`

The contract between drowse.ai's TypeScript, the WebLLM bridge and the compiled MLC model
libraries. v5 reads and writes inside each transformer block (the head site, `mid`, the neuron
site), replays nodes with anatomy, activity and pattern taps, and runs static reads over the
resident weights. It replaces `post-block-residual-v4`; the post-block hook and every v3
structured buffer carry over unchanged.

Derivations and alternatives are in the parity handoff, `B1-hook-abi-v5.md` (cited as §n), with
the toolchain corrections of `B40-toolchain.md` §7. Where the two differ, this document follows
the decisions taken since: one hook ABI for all nine libraries, full-depth lens packs fetched per
layer, patterns mandatory (no `standard-v4-nopatterns`), one units entry-point pair on the curved
superset, and one runtime path. As built: the rank-one functions are gone from the MLC overlay
(`drowse_prefill`/`decode`, the batch pair, the rank-one capture v1 functions), from the bridge
(rank-one programs and captures, `drowse-structured-v2`, the legacy alias table) and from the
adapter's runtime, which compiles and installs `drowse-structured-v4` programs under the v4
profiles only. The v1-v3 profile records stay in `structuredHookProfile.ts` while the lock, the
catalog and the release tooling still name them (W4 rewrites those); no v5 code path accepts them.

The TypeScript side of this contract, pinned by script tests:

| file | holds | test |
|---|---|---|
| `webui/src/hosted/runtime/structuredHookProfile.ts` | ABI names, profiles, `BlockSpec`, descriptor schema 4 | `structured-hook-units.test.mjs` |
| `webui/src/hosted/runtime/structuredHookProgram.ts` | `drowse-structured-v4`: unit tables, encoder, controls, CPU ablation reference | `structured-hook-units.test.mjs` |
| `webui/src/hosted/runtime/modelBackend.ts` | `BrowserBlockSitesBackend` (the device surface), typed tensors, replay layouts | `block-sites-backend.test.mjs` |
| `webui/src/hosted/runtime/webLlmBlockSites.ts`, `webLlmStaticReads.ts` | the surface over the bridge (replay, residents, static reads) | `web-llm-block-sites.test.mjs`; WebGPU `browser-units-goldens.mjs --suite all` |
| `browser-runtime/forks/units_goldens.py`, `static_reads.py` | the C TVM VM checks against the goldens | `verify-mlc-hook.py --units-fixtures` |
| `browser-runtime/forks/verify-production-library.py` | a production library on the CPU: export and shader checks, its descriptor, and (`--parity`) its block sites, greedy tokens and static reads on the C TVM VM against PyTorch on the dequantized MLC tensors (B1 §4.3) | run per library before release |

## 1. Names (§8.1)

| layer | v4 | v5 |
|---|---|---|
| hook ABI (`DROWSE_HOOK_ABI`, lock `hookAbi`) | `post-block-residual-v4` | `block-sites-v5` |
| runtime ABI (lock `runtimeAbi`) | `drowse-web-runtime-v1` | `drowse-web-runtime-v2` |
| readout ABI (lock `exactReadoutAbi`) | `exact-readout-v1` | `exact-readout-v2` |
| structured format | `drowse-structured-v3` (and v2) | `drowse-structured-v4` |
| capacity profiles | `standard-v1`…`-v3` | `standard-v4` (units), `standard-v4-residual` (no units) |
| descriptor schema | 3 (31 fields) | 4 (67 fields) |

Eligible architectures (`gemma3_text`, `qwen3`: dense gated MLP, RMSNorm, RoPE, separate q/k/v
in the HF checkpoint) compile under `standard-v4`. `gpt2`, `gpt_neox` and `qwen3_5` compile under
`standard-v4-residual`: same hook ABI, `UNIT_SITES = 0`, and every unit, anatomy, activity and
`mid` call answers native's `UnsupportedBlockError` (422) with the lock's `unitRefusal` text.

## 2. Sites (§1)

Every eligible block, with `N(x) = x · rsqrt(mean(x²) + ε) · (o + w)`:

```text
a      = N_ai(h)
q_h    = RoPE(N_q(W_Q^h a)),  k_g = RoPE(N_k(W_K^g a)),  v_g = W_V^g a        (g = kv_head(h))
z_h    = Σ_j softmax_j(q_h·k_g(j) · scaling + mask)_j · v_g(j)                  [T, H·hd]   HEAD SITE
attn_o = W_O z                                                                  [T, d]
h_mid  = h + N_ao(attn_o)                                                                   MID SITE
act    = φ(W_gate N_mi(h_mid)) ⊙ (W_up N_mi(h_mid))                             [T, d_ff]   NEURON SITE
mlp_o  = W_down act                                                             [T, d]
h_out  = h_mid + N_mo(mlp_o)                                                                OUT (v4 hook after)
```

| | gemma3_text | qwen3 |
|---|---|---|
| norm offset `o` | 1 | 0 |
| sandwich `N_ao`, `N_mo` | `post_attention_layernorm`, `post_feedforward_layernorm` | none |
| `N_mi` | `pre_feedforward_layernorm` | `post_attention_layernorm` |
| `φ` | tanh-GELU | SiLU |
| `N_q`, `N_k` | per-head RMSNorm | per-head RMSNorm |
| embedding | `√d · E[v]` | `E[v]` |
| local layers | `(L + 1) % 6 ≠ 0`, window `i − W < j ≤ i` | none |

A neuron `n` of block `L` is `act[:, n]`; a head `h` is `z[:, h·hd:(h+1)·hd]`, its write
`W_O^h z_h` with Gemma's `N_ao` gain and the position's RMS factor `r_ao` held constant.

**In the MLC graph** (B40 §7.1, read on jobe): `drowse_block` replaces each
`layer(hidden_states, paged_kv_cache, layer_id)` inside the existing `drowse_*_forward` functions
and exposes the sites as no-op graph calls unless their section is compiled in. The down
projection's input (`φ(x1) * x2` after `gate_up_proj`'s split) and the output projection's input
(the reshaped attention output) are plain Relax tensors in every layout. The q/k/v side differs
by architecture: Gemma3 has **separate** `q_proj`, `k_proj`, `v_proj`, normed and concatenated
before `attention_with_fused_qkv`; Qwen3 has one `c_attn`, split, normed and re-concatenated. The
fused projections inside MLC are no obstacle; native's refusal of fused projections concerns the
HF checkpoint.

Ablation cannot be corrected after the block (§1.3): Gemma's sandwich norms make a unit's effect
nonlinear in the whole write, a Qwen3 head's effect passes through the MLP, and prefill ablation
changes later layers' K/V.

## 3. Entry points (§3.4)

| function | `standard-v4` | `-residual` | role |
|---|---|---|---|
| `drowse_structured_batch_{prefill,decode}` | yes | yes | v4-identical: affine and probes |
| `drowse_curved_batch_{prefill,decode}` | yes | yes | v4-identical: and curves |
| `drowse_units_batch_{prefill,decode}` | yes | no | curved superset, unit sites, unit tap |
| `drowse_replay_batch_prefill` | yes | no | units superset, anatomy, activity, pattern taps |
| unit static reads (§8 below) | yes | no | |
| vocabulary, footprint, lens and utility functions (§8) | yes | yes | `exact-readout-v2` |
| `drowse_hook_profile` | yes | yes | descriptor schema 4 |

Routing: a program with no unit section runs the v4-identical functions and pays nothing; a
program with one runs the units pair. A units library must export every function marked yes, a
residual library none of the unit ones; either mismatch fails the load (`HOOK_ABI_MISMATCH`).

```text
drowse_units_batch_prefill(
  input_embeds [1, seq_len, d] f32, logit_positions [n] i32, paged_kv_cache,
  affine_active, affine_basis, affine_neutral, affine_target, affine_along, affine_kappa,
  probe_kind, probe_direction, probe_bias, probe_threshold,
  curve_basis, curve_neutral, curve_domain_kind, curve_parameters, curve_feet,     # as v4 curved
  unit_meta u32 [UNIT_META_LEN], unit_coeff f32 [UNIT_COEFF_LEN], unit_active f32 [4],
  tap_grams f32 [16, hd, hd],
) -> (logits, paged_kv_cache, measurements [L, 8], next_feet [L, 4, 4], jlens_hidden [L, d],
      unit_taps [64])
```

Decode drops `logit_positions` and takes `[1, 1, d]`. `seq_len ≤ PREFILL_CHUNK_SIZE` on every
call.

```text
drowse_replay_batch_prefill(
  <every units-prefill argument>,
  replay_meta i32 [REPLAY_META_LEN],
  unit_scales f32 [L, 2·d_ff + H],
  head_grams f32 [L, H, hd, hd],                # resident, built once per model
  activity_state f32 [L, 3·d_ff + 3·H], traces f32 [32, trace_rows], population f32 [L, d_ff + H],
  pattern_keys f32 [16, KV, pattern_tokens, hd],
) -> (logits, paged_kv_cache, measurements, next_feet, jlens_hidden, unit_taps,
      anatomy_taps [L, STRIDE], activity_state', traces', population',
      pattern_keys', pattern_q [16, H, hd])

drowse_attention_rows(pattern_keys [16, KV, pattern_tokens, hd], pattern_q [16, H, hd],
                      rows_meta i32 [16, 4]) -> rows f32 [16, pattern_tokens]
drowse_head_write_grams() -> head_grams f32 [L, H, hd, hd]
```

`trace_rows` and `pattern_tokens` are symbolic: the caller sizes `traces` to the node's rows and
`pattern_keys` to the positions the replay feeds (1 when not asked), both at most the context
window, so a replay holds only what it reads (a 4B model's keys at full context would be 256 MiB).
`activity_state`, `traces`, `population` and `pattern_keys` are written in place and returned
(the outputs alias the inputs), so they thread through the replay's forwards; `anatomy_taps` and
`pattern_q` are fresh per forward, and the caller keeps the ones of the forward holding the
anatomy position. `rows_meta` rows are `(slot, layer, head, p)`, slot −1 unused.

## 4. `drowse-structured-v4` (§3.1, §3.4)

A strict superset of v3: every v3 buffer is unchanged in shape and meaning, plus one unit section
that is present whole or absent. Without it a v4 program's buffers are byte-identical to the v3
program the same layers compile to (pinned). A `standard-v4-residual` library rejects any unit
section (`UNIT_SECTION_UNSUPPORTED`, 422).

Host program (`StructuredHookProgramBuffers`, `format: "drowse-structured-v4"`,
`hookAbi: "block-sites-v5"`) adds:

```ts
unitShape: { dFF; nHeads; headDim };
unitMeta: Uint32Array;  unitCoeff: Float32Array;  unitActive: Float32Array;   // tables below
tapGrams: Float32Array;                                                     // [16, hd, hd]
controls.units: (StructuredHookControl | null)[];                           // length 4
measurementSchema.unitTaps: (StructuredUnitTap | null)[];                   // length 64, by slot
```

**Semantics** (native `UnitOp`, `UnitHook`, `_lower_unit_terms`). One term per `α !neuron/…` or
`α !head/…`: `keep = 1 − α`, `bias = α · mean` (the neutral `neuron_mean[n]` or `head_mean[h]`),
built in fp32. For every row `t` of every forward, prefill and decode alike:

```text
if pos_t < UNIT_POSITIONS_FROM (= 1): skip          # the sink keeps its own value
for each entry e of this layer and site (CSR range):
    if unit_active[group(e)] != 0:
        x[t, span(e)] = x[t, span(e)] · keep(e) + bias(e, :)    # fp32
```

`pos_t` is the absolute position from `paged_kv_cache.get_query_positions`, so chunked prefill
needs nothing more. Coordinates without an entry are untouched (native's `x·1 + 0` is exact). A
neuron's span is `[n, n+1)`, a head's `[h·hd, (h+1)·hd)` with `hd` bias values. Any finite `α`.

A composed stack holds each unit at most once (the grammar sums equal triggers and refuses
differing ones), so no coordinate is touched twice; the encoder fails closed with
`UNIT_DUPLICATE` rather than compose.

**Tables** (u32 `unit_meta`, f32 `unit_coeff`; `structuredUnitSectionLayout`):

```text
unit_meta  = neuron_offsets[L+1] | neuron_index[N_ABL] | neuron_group[N_ABL]
           | head_offsets[L+1]   | head_index[H_ABL]   | head_group[H_ABL]
           | tap_offsets_neuron[L+1] | tap_neuron_index[TAP] | tap_neuron_slot[TAP]
           | tap_offsets_head[L+1]   | tap_head_index[TAP_HEADS] | tap_head_slot[TAP_HEADS]
unit_coeff = neuron_keep[N_ABL] | neuron_bias[N_ABL] | head_keep[H_ABL] | head_bias[H_ABL, hd]
unit_active f32 [G_U]
N_ABL = 256, H_ABL = 64, G_U = 4, TAP = 64, TAP_HEADS = 16
UNIT_META_LEN = 4(L+1) + 800, UNIT_COEFF_LEN = 576 + 64·hd
```

Entries are CSR by layer (`offsets[L]` is the first entry of layer `L`, `offsets[L+1]` one past its
last) and in registration order within a layer; capacity past the last entry is zero. Groups are
the distinct triggers in first-registration order. The upload is the arrays' bytes, little-endian.

**Triggers.** `unit_active` is uploaded before each forward beside `affine_active` from the
host's trigger evaluation (phase, generated-token count, thinking state, the prior step's gate
scores; `createStructuredHookControlEvaluator` returns it as `unitActive`). Prefill rows see the
prefill state; a decode forward one state for all its rows. A unit gate reads the prior step's
scaled unit reading through `priorScores[unit]`. Native's step count (`loom/replay.py`): the
prefill is step 0 and the decode forward that reads the k-th generated token (1-based) is
`gen_step = k`, so `@first:2` covers the first decode forward only and `@after:n` starts at the
n-th; the goldens' `triggers` case pins this on both arches.

**Position targeting** (browser only). `position_spans`/`position_indices` gate unit terms as they
gate affine and curve terms: prefill splits chunks at span boundaries and outside a span the
bridge uploads `unit_active` zeroed (`updateDrowseStructuredControls` takes `unitActive` and keeps
a `drowsePositionUnit` copy). Inside a span the trigger state applies; position 0 stays excluded.
Unit taps and replay readings are measurements and are never gated. Pinned by a bridge test
(`!neuron/…` with spans `[[3, 5]]` ablates only rows 3 and 4).

**Prefix reuse.** Any generation with a hook program resets the chat and prefills from position
0, so the sink is always row 0 of the first chunk. A future prefix-reuse path must treat a
prefill-active unit term as prefill steering.

Neuron pushes (`0.3 neuron/L8n344`) are not unit terms: they lower to the additive residual path
as a one-layer direction `s·γ_mo ⊙ W_down[:, n]` read by `unitWeightRows("neuron_write")`, like a
`jlens/` atom. `slens/` atoms lower the same way through `footprintDirections`.

## 5. The live unit tap (§2.2)

The units pair writes `unit_taps f32 [64]` per forward at its last row, in slot order, after any
ablation:

```text
neuron slot = act[last, n]
head slot   = sqrt(max(0, z_hᵀ G_h z_h)) · r_ao
              G_h = (γ_ao ⊙ W_O^h)ᵀ (γ_ao ⊙ W_O^h) [hd, hd],  γ_ao = 1 + w (Gemma), 1 (Qwen3)
              r_ao = rsqrt(mean(attn_o[last]²) + ε_ao) on Gemma, 1 on Qwen3
```

`tap_grams[i]` holds `G_h` of the `i`-th tap-head entry in CSR order, built by
`drowse_head_write_grams` when the program is bound (`bindStructuredProgram`; until that static
read exists the caller supplies the grams and an all-zero gram fails the bind). The tap rides the
existing measurement-bundle readback (one sync per token, `BlockSitesForwardReadback`). The host
divides by the soft floored reading scale (neuron pole by `sign(act)`); scales never reach the GPU
on the live path. 64 slots, at most 16 heads; more is `UNIT_TAP_CAPACITY` before generation.

`slens/` probes and gates run post-forward over the existing `[L, d]` capture
(`drowse_slens_probes`, §8). There is no live `mid` read: native 6.0 has none. The capture reaches
the host only when the installed program sets `hiddenCapture` (a units program with attached
`slens/` probes): the bridge then returns it as the measurement bundle's `hidden` in the same sync
as the probe slots and unit taps. `slens/` reads cover attached probes only; a gate naming a probe
nothing attached never fires (native `resolve_gates`), so there are no gate-only S reads.

**Stream window.** Between a generation's tokens the stream holds the model lock. Block-sites
`read`, `uploadResident`, `release`, `headGrams`, `uploadUnitScales` and `attentionRows` requests
that arrive then run inside the stream's token window, through one serialized queue per model (each
may hold TVM scopes across a device sync; the queue drains before the stream resumes), so a
forward's `slens/` probes read its capture before the next token. Replays (`openReplay`,
`replayForward`, `finishReplay`, `cancelReplay`) own the KV cache and wait for the model lock.

**Envelope.** Per token, attached R-lens and SAE probe readings ride the token's measurement
envelope only while their family's live readout is on (native's token payload merges the live
readout step's readings), as do `slens/` readings (lens family) and unit readings (units family);
gates read every attached probe on every forward, and the reply's aggregate carries every reading
(`tokenMeasurementEnvelope` in `webLlmGeneration.ts`).

## 6. Descriptor schema 4 and attestation (§3.5)

`drowse_hook_profile` returns int32 `[67]` in this order (`HOOK_PROFILE_DESCRIPTOR_V4_FIELDS`;
`expectedHookProfileDescriptor` builds the expected vector). Fields 0-30 are v3's, unchanged in
meaning:

```text
 0 MAGIC 0x53414B4C            1 SCHEMA_VERSION 4         2 PROFILE 4
 3 HOOK_ABI_VERSION 5          4 FORMAT_VERSION 4         5 LAYER_COUNT
 6 HIDDEN_SIZE                 7 MAX_AFFINE_GROUPS 4      8 MAX_RANK 8
 9 MAX_PROBES 8               10 MAX_CURVES 4            11 MAX_CURVE_NODES 32
12 MAX_INTRINSIC_DIM 4        13 MAX_EMBED_DIM 8         14 CURVE_PARAMETER_STRIDE 671
15 MIN_COMPUTE_WORKGROUP_STORAGE_SIZE 32768               16 MAX_GEOMETRY_PROBES 8
17 MAX_WHITENER_RANK 96       18 MAX_GEOMETRY_CANDIDATES 33                 19 GEOMETRY_OUTPUT_STRIDE 41
20 GEOMETRY_FOOT_RESTARTS 3   21 GEOMETRY_FOOT_ITERATIONS 12                22 GEOMETRY_WARM_RESTARTS 2
23 GEOMETRY_WARM_ITERATIONS 4 24 REQUIRED_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE 8
25 GEOMETRY_KERNEL_STORAGE_BINDINGS 8                     26 GEOMETRY_HEADER_STRIDE 7
27 GEOMETRY_PAYLOAD_ELEMENTS  28 EXACT_READOUT_ABI_VERSION 2                 29 READOUT_TOP_K 8
30 MAX_SAE_FEATURES_PER_CHUNK 16384
31 UNIT_SITES (0|1)           32 D_FF                    33 N_Q_HEADS
34 N_KV_HEADS                 35 HEAD_DIM                36 SANDWICH (0|1)
37 NORM_OFFSET (0|1)          38 ACTIVATION (1 silu, 2 gelu_tanh)
39 EMBED_SCALE_F32_BITS       40 RMS_EPS_F32_BITS        41 QK_NORM (0|1)
42 ATTN_SCALING_F32_BITS      43 ATTN_LOGIT_SOFTCAP_F32_BITS                 44 FINAL_LOGIT_SOFTCAP_F32_BITS
45 ROPE_THETA_F32_BITS        46 ROPE_LOCAL_THETA_F32_BITS                   47 ROPE_GLOBAL_SCALE_F32_BITS
48 ROPE_IN_GRAPH (0|1)        49 SLIDING_WINDOW          50 SLIDING_PATTERN
51 PREFILL_CHUNK_SIZE         52 MAX_UNIT_GROUPS 4       53 MAX_NEURON_ABLATIONS 256
54 MAX_HEAD_ABLATIONS 64      55 MAX_UNIT_TAPS 64        56 MAX_HEAD_TAPS 16
57 MAX_ACTIVITY_TRACES 32     58 MAX_ACTIVITY_SPANS 4    59 MAX_PATTERN_LAYERS 16
60 MAX_PATTERN_HEADS 16       61 ANATOMY_STRIDE          62 UNIT_POSITIONS_FROM 1
63 MAX_LENS_READ_TOP_K 256    64 MAX_ANATOMY_UNITS 64    65 DROWSE_KERNEL_STORAGE_BINDINGS 8
66 CONTEXT_TOKENS
```

- `*_F32_BITS` hold the IEEE-754 bits of `Math.fround(value)` reinterpreted as a signed int32
  (`float32Bits`); 0 means none (no softcap, no local θ). The library must emit the same
  reinterpretation (`op.full` takes int32).
- `PROFILE` is 4 for both v4 profiles; `UNIT_SITES` says which. A residual library reports 0 in
  fields 32-50, every unit capacity (52-60), `ANATOMY_STRIDE`, `UNIT_POSITIONS_FROM` and
  `MAX_ANATOMY_UNITS`; it keeps `MAX_LENS_READ_TOP_K = 256` (lens reads at `out` run everywhere).
- `ANATOMY_STRIDE = 6·d + d_ff + H·hd`. `CONTEXT_TOKENS` is the compiled context window, the
  largest of the lock's `contextProfiles`. `PREFILL_CHUNK_SIZE` is the chunk the library was
  compiled with, which the lock records per model (`models[].prefillChunkSize`, its
  `mlc-chat-config.json`'s `prefill_chunk_size`): a positive integer no larger than the smallest
  context profile. The five instruct libraries use 2048 against [2048, 4096]; gpt2-base 1024
  against 1024; pythia-70m-base, gemma3-1b-pt and qwen35-2b-base 1024 against their one 2048
  context (every published build of them). The adapter attests the lock's value, never one derived
  from the contexts, so a library compiled with another chunk fails with `HOOK_ABI_MISMATCH` at
  `PREFILL_CHUNK_SIZE`, and so does a load whose lock entry records none.

**Attestation.** Before any upload the bridge compares the descriptor element by element with the
expected vector built from the TypeScript profile and the runtime lock's model entry: the profile
id and every capacity, plus every block and attention field against the lock's `blockSpec`, which
B2 recorded from native `model_spec` and the source config (`ropeTheta`, `ropeLocalTheta` and the
rope scaling factor come from the HF config, not `model_spec`). A mismatch names its fields
(`hookProfileDescriptorMismatches`) and fails the load with `HOOK_ABI_MISMATCH`.

The descriptor catches a library compiled with the wrong constants. It cannot see a wrong graph
with the right constants (a swapped `gate_up` split, a misplaced norm): the `block-sites` goldens on
the tiny libraries and the production q4 parity gate's per-layer `act`/`z`/`mid` error catch those.

**Lock** (`runtime-lock.schema.json`, §8.2): `hookAbi: "block-sites-v5"`,
`exactReadoutAbi: "exact-readout-v2"`, `runtimeAbi: "drowse-web-runtime-v2"`;
`models[].structuredHookProfile ∈ {standard-v4, standard-v4-residual}`; `models[].blockSpec`
(`BLOCK_SPEC_KEYS`: `dFF, nHeads, nKvHeads, headDim, sandwich, normOffset, activation, embedScale,
rmsEps, qkNorm, attnScaling, attnLogitSoftcap, finalLogitSoftcap, ropeTheta, ropeLocalTheta,
ropeGlobalScale, ropeInGraph, slidingWindow, slidingPattern`; booleans for the flags,
`"silu" | "gelu_tanh"`, `null` for none) required with `standard-v4` and forbidden with
`-residual`; `models[].unitRefusal` (native's full message) required with `-residual`;
`models[].prefillChunkSize` required everywhere (`validateRuntimeLock` bounds it by the smallest
context). `prepare-hosted-candidate-lock.mjs` refuses a publish directory whose
`hosted-artifacts.json` or `mlc-chat-config.json` (top level or `model_config`) names another
prefill chunk, and `publish-hosted-model.mjs` refuses the closure likewise.

## 7. Replay (§2.3)

One entry point runs every replay forward. The host drives native's `replay_segments` and
`drive_segments`: one cacheless forward when every open term is `Trigger.BOTH`, otherwise prefill
then decode runs, one position per forward under a probe gate, each forward's controls evaluated
from the previous forward's readback. `BrowserBlockSitesBackend.openReplay` opens a fresh sequence
at position 0 with the captures asked for; the host feeds logical segments one by one
(`forward(segment)`: tokens and one control state, `affineActive`, `curveActive`, `unitActive`),
reads each segment's last probe slots and unit taps back for the next gate, then `finish()` reads
the captures and drops the sequence (`cancel()` drops it unread).

- **Chunking.** No forward carries more than `PREFILL_CHUNK_SIZE` rows. A longer segment runs as
  consecutive chunks with its controls held constant. The KV cache, curve feet, activity state,
  traces and pattern keys persist across chunks and segments; trigger and gate state change only
  at segment boundaries. The anatomy taps are written by whichever chunk holds the position.
  Position-targeted nodes replay with their recorded spans as separate segments.
- **Anatomy taps** at one absolute position `p` (`anatomyTapLayout`):
  `anatomy_taps f32 [L, STRIDE]`, per layer `block_in | mid | out_pre | out | attn_o | mlp_o | act | z`.
  `block_in[0]` is the prepared input embedding (Gemma `√d · E[v]`); `out_pre` is the block output
  before the post-block hook, `out` after it. Native's `TokenTaps` exactly
  (`attention_write = mid − block_in`, `mlp_write = out_pre − mid`, `steering_write = out − out_pre`).
- **Unit activity** (native `_activity`), reduced on the device per layer into
  `activity_state` (`activityStateLayout`: `neuron_peak_value | _row | _rank` `[d_ff]` each, then
  the head triple `[H]`; initial value 0, row −1, rank −1). A node's token rows map to forward
  columns through ≤ 4 spans `(first_row, rows, first_position)`: row `first_row + i` is column
  `first_position + i`. Columns outside every span are not node rows and are skipped. A row whose
  column is below `UNIT_POSITIONS_FROM` is excluded: value 0, rank −1 (the sink, and rows of a span
  with negative `first_position` that no column holds; the host writes their trace zeros). Values
  are scaled: neuron `act / S_pole` (pole by sign), head `sqrt(max(0, z_hᵀG_h z_h))·r_ao / S_h`,
  from `unit_scales` (`unitScalesLayout`: `S+ [d_ff] | S− [d_ff] | S_head [H]`). Rank is `|v|`; a
  strictly greater rank wins, ties keep the earlier row. Traces `[32, context]` hold every row of
  the requested units; `population [L, d_ff + H]` the values at one row. The host ranks the state
  (top per kind over native's flat order, value descending, flat index ascending; rank −1 drops).
- **Head grams** `head_grams [L, H, hd, hd]` for every head are built once per loaded model
  (`residentHeadGrams`, `drowse_head_write_grams`: one TIR kernel per layer over the dequantized
  `W_O`, `γ_ao` applied on a sandwich block; a Relax matmul there trips MLC's dequantize-matmul
  fusion), kept resident and read per layer at a compile-time offset (zero-copy, B40 §7.4);
  dropped with the model. 18 MiB (270m) to 72 MiB (qwen3-4b).
- **Peaks, as built.** One thread per unit walks the forward's rows in order with a strict `>`,
  so within a forward the first maximum wins and across forwards the earlier one keeps a tie:
  native's `rank.max(dim=0)` then `best > held` exactly.
- **`replay_meta`** (i32, `replayMetaLayout`/`encodeReplayMeta`, 134 values under `standard-v4`):
  `[0]` anatomy position (−1 none), `[1]` span count, `[2..18)` four spans of
  `(first_row, rows, first_position, unproduced)`, `[18]` population row (−1 none), `[19]` trace
  count, `[20..116)` 32 traces of `(layer, kind 0 neuron | 1 head, index)`, `[116]` pattern layer
  count, `[117..133)` sixteen pattern layers ascending, `[133]` flags (1 anatomy, 2 activity,
  4 patterns). Unused entries are −1. Patterns need the anatomy flag (queries are captured there).
- **Bridge (B4).** The vendored WebLLM takes one request per call, `drowseBlockSites(request)`
  (one worker message kind): `headGrams`, `uploadUnitScales`, `release`, `openReplay`
  (`replayMeta`, `unitScales` resident id, `rowCount`, `tokenCount`), `replayForward` (tokens and
  one control state), `finishReplay`, `cancelReplay`, `attentionRows`. `openReplay` needs a
  drowse-structured-v4 program with a unit section installed (the adapter installs the node's,
  adding an empty unit section, or an empty program for an unsteered replay), validates the meta
  fail-closed, resets the KV cache and allocates the threaded buffers; a forward is chunked at
  `PREFILL_CHUNK_SIZE` and reads back its last chunk's probe slots and unit taps in one sync;
  `finishReplay` reads the captures in one sync, keeps the pattern capture resident and resets the
  KV cache. Generation and capture refuse (`REPLAY_BUSY`) while a replay is open. The adapter is
  `webui/src/hosted/runtime/webLlmBlockSites.ts`; `BlockSitesReplayRequest.tokenCount` (optional,
  default the context) sizes the pattern keys.
- **Replay readback (bridge .46).** A replay forward is generation's forward: curve feet thread
  through its chunks, and the geometry readout runs on every chunk (the prefill solve on a chunk,
  the warm solve on one row) with its feet threaded. The segment's last chunk returns, in the one
  sync that reads its probe slots and unit taps, the `geometry` readout, the `[L, d]` `hidden`
  capture when the program sets `hiddenCapture`, and the J-lens probabilities of kind-3 probe
  slots (the full-vocabulary readout of `jLensTokenIds`). Node replays score every gate generation
  scores from that readback (unit taps, geometry, attached `slens/` over `hidden`, J-lens and SAE
  slots); a gate the readback cannot score refuses with native's 422 instead of staying shut.

## 8. Static reads and `exact-readout-v2` (§4)

A Relax function cannot index per-layer parameters by a runtime layer, so every weight primitive
is all-layers and masked: it loops the layers statically and each layer's launch handles only the
requests naming that layer (`units [n, 2] = (layer, index)` or `layers [c]`), writing in place into
one fresh buffer. Weights are read inside the kernel: group-quantized storage is dequantized by the
quantization config's own `_dequantize`, whose TE expression is inlined into the kernel
(`drowse_reads._Weight`), so a read sees exactly the values the forward multiplies. Every value is
fp32. Functions as built (MLC overlay `drowse_reads.py`; adapter `webLlmStaticReads.ts`):

| function | inputs → output | formula | member |
|---|---|---|---|
| `drowse_neuron_rows` | `units [n, 2]` → `[n, 3, d]` | write `s·γ_mo ⊙ W_down[:, n]`; `â = γ_mi ⊙ W_gate[n] / max(‖·‖, 1e-12)`; `b̂` from `W_up` | `unitWeightRows` (`neuron_write`, `neuron_read`) |
| `drowse_head_rows` | `units [n, 2]` → `[n, 4, hd, d]` | `(γ_ao ⊙ W_O^h)ᵀ`; raw `γ_ai ⊙ W_Q^h`, `γ_ai ⊙ W_K^g`, `γ_ai ⊙ W_V^g` | `unitWeightRows` (`head_write` transposed back to `[d, hd]` on the host, `head_read`) |
| `drowse_head_write_grams` | → `[L, H, hd, hd]` | `G_h = Bᵀ B` | `residentHeadGrams` |
| `drowse_gather_head_grams` | `head_grams`, `heads [n, 2]` → `[n, hd, hd]` | gather | `headWriteGrams`; `bindStructuredProgram` fills `tapGrams` with it |
| `drowse_layer_write_scores` | `x [c, m, d]`, `layers [c]` → `[c, m, d_ff]` | `x · (s·γ_mo ⊙ W_down)` | `layerWriteScores` |
| `drowse_layer_read_scores` | `x [c, m, d]`, `layers [c]` → `[c, 2, m, d_ff]` | `(â·x, b̂·x)` | `layerReadScores` |
| `drowse_neuron_contributions` | `g [c, d]`, `act [c, d_ff]`, `mlp_o [c, d]`, `layers` → `[c, d_ff]` | `act_n · ((g ⊙ γ_mo)·W_down[:, n]) · r_mo · s` | `neuronContributions` |
| `drowse_head_contributions` | `g`, `z [c, H·hd]`, `attn_o`, `layers` → `[c, H]` | `r_ao · Σ_i ((g ⊙ γ_ao)ᵀ W_O)[h·hd+i] · z[h·hd+i]` | `headContributions` |
| `drowse_head_write_norms` | `z`, `attn_o`, `layers`, `head_grams` → `[c, H]` | `sqrt(max(0, z_hᵀ G_h z_h)) · r_ao` | `headWriteNorms` |
| `drowse_unembed_cotangent` | `h_final [n, d]`, `ids [n]` → `[n, d]` | `rsqrt(mean(h²)+ε_f) · γ_f ⊙ W_U[v]` | `unembedCotangent` (refused on a residual library) |
| `drowse_embedding_rows` | `ids [n]`, `mean_E [d]` → `[n, d]` | `ẽ_v = s_E·E[v] − mean_E` | `vocabRowsTimes` (`embed`), `footprintDirections` |
| `drowse_unembedding_rows` | `ids [n]` → `[n, d]` | `γ_f ⊙ W_U[v]` | `vocabRowsTimes` (`unembed`) |
| `drowse_vocab_logits` | `rows [n, d]` → `[n, V]` | `W_U · N_f(row)`, the model's own final norm in its dtype, the product in fp32 (native `full_logits`) | `vocabReadoutTopK`, `vocabTokenLogits`, `vocabLogitMoments` |
| `drowse_footprint_scores` | `u [n, d]`, `mean_E`, `Σ_E`, mask → `[n, V]` | `(s_E·(u·E[v]) − u·mean_E) / sqrt(max(uᵀΣ_E u, 1e-30))`, masked ids LOWEST | `footprintTopK` |
| `drowse_logit_moments` | `logits [n, m]`, mask, `flags [1]` → `[n, 2]` | skew, kurtosis over the kept ids (all when `flags[0] = 0`), variance clamped at 1e-30 | `vocabLogitMoments` |
| `drowse_log_sum_exp` | `x [n, m]` → `[n]` | | `logSumExp`, the readouts' normalizer |
| `drowse_topk` | `x [n, m]`, `values [n, k]`, `indices [n, k]` in place | exact, value desc, index asc; `k` passes of a workgroup argmax after the previous pick | `topK` (≤ 1024), the readouts (≤ 256) |
| `drowse_batched_matmul(_nt)` | `a [b, m, k]`, `w [1 or b, k, n]` (`_nt`: `[·, n, k]`) → `[b, m, n]` | | `batchedMatmul`, `lensTransport` (R `x @ Jᵀ` is `_nt`, S `x @ K` plain), `vocabRowsTimes`, `slensProbes` |
| `drowse_quadratic_form` | `x [b, n]`, `M [1 or b, n, n]` → `[b]` | | `batchedQuadraticForm`, `slensProbes` |
| `drowse_grouped_row_norms` | `x [n, g, w]` → `[n, g]` | | `groupedRowNorms` |
| `drowse_chan_merge` | `count [1]`, `mean [d]`, `m2 [d]`, `rows [r, d]` → `(mean, m2)` | Chan's merge of the rows' own two-pass statistics | `chanMerge` |

Every v5 library exports the twelve `exact-readout-v2` functions (embedding rows through Chan's
merge); exactly the libraries with unit sites export the nine unit reads (neuron rows through the
cotangent; gather included). The bridge fails a load that breaks either rule (`HOOK_ABI_MISMATCH`).
Compositions run in the adapter as one bridge request each: `drowseBlockSites({ op: "read", steps,
read })`, a list of library calls whose arguments are host tensors, residents (with an optional
view) or earlier outputs, and whose named outputs come back with one device sync. `slensProbes`
centres on the host (`h_L − μ_L`), pulls through `K_L` on the device and divides each probe's dot
by `sqrt(uᵀΣ_E u)`; `footprintDirections` is `ẽ_v @ K_Lᵀ` per layer; `vocabTokenLogits` reads the
full logits back and gathers on the host. The device writes −∞ as the lowest finite float (WGSL
has no infinity literal); the adapter reports it as `-Infinity`.

Fill buffers (the in-place bases and constant inputs) are `drowse_reads.fresh`, a fill kernel
bound to a runtime input: `op.zeros`/`op.ones` fold to module constants, the VM deduplicates equal
constants across functions, and an in-place store into one changes every function's copy.

Everything averaged over the vocabulary or the neutral corpus is offline (§4.2): unit statistics,
`mid` and `out` means, head-norm derivatives `D_query`, `D_key` (applied on the host to the raw
head read rows), whitened write norms, the null basis, population scalars, contrast statistics and
the S embedding statistics. No eigensolver runs on the GPU.

**Residency.** Lens matrices are uploaded one layer at a time as resident `[1, d, d]` chunks
(`uploadLensLayer`), fetched on a layer's first read, with the live readout keeping its own
resident set; reads take the resident handles, so a layer's matrix crosses the bus once. The S
embedding statistics (`mean_E`, `Σ_E`, the token mask bitmap) are resident while any S read may
run. Head grams are resident per model. A matrix that cannot fit `maxBufferSize` and
`maxStorageBufferBindingSize` fails before use.

## 9. Patterns (§6)

Paged attention computes an online softmax and never materializes probabilities, so a pattern row
`a_pj = softmax_j(q_p·k_j·scaling + mask)` is rebuilt from captured rows. The replay captures, at up
to sixteen asked layers, every KV head's post-RoPE key for every position into `pattern_keys`, and
every head's query at the anatomy position into `pattern_q [16, H, hd]`; `attentionRows` then
computes, for each chosen `(layer, head)`:

```text
s_j = q_p·k_j · scaling,  masked where j > p, and on a local layer where j ≤ p − W
a_j = exp(s_j − max) / Σ exp(s_j − max)       fp32, j = 0…p (no softcap on these models)
```

**As built.** `drowse_attention_rows` runs one workgroup per row: scores `q·k · scaling` (the
library's `ATTN_SCALING`) for `lo ≤ j ≤ p`, `lo = p − W + 1` on a local layer
(`(layer + 1) % SLIDING_PATTERN ≠ 0`) and 0 otherwise, then the max, `exp(s − max)` and the sum by
shared-memory reductions; masked entries read 0. GQA maps head `h` to KV head `h / (H / KV)`.

**RoPE.** Gemma3 rotates in the graph (`_gemma3_rotary`, KV cache `RopeMode.NONE`), so its
captured `k` is post-RoPE. Qwen3 rotates **inside the paged KV cache** (B40 §7.2:
`RopeMode.NORMAL`; `f_split_rotary` runs `llama_rope_with_position_map` before
`f_transpose_append_mha` writes the pages, which therefore hold post-RoPE K), so its in-graph `k` is
pre-RoPE and the capture applies the same rotation (`rope_freq_default`, θ from the descriptor,
`rope_scale = 1`), rounded to the `qkv` dtype as the cache stores it. The bridge asserts the cache runs in `kNormal`: a cache created with sliding
window support switches to `kInline` and stores un-rotated K. `attention_kv_cache_debug_get_kv`
is a debugging oracle only (every layer, K and V, and it refuses Gemma3's sliding layers).

**Layers.** With full-depth lens packs every layer can hold R:mid, so a head that ranks may sit at
any layer. A pattern request names at most `MAX_PATTERN_LAYERS = 16` layers, as many as
`MAX_PATTERN_HEADS`, so the chosen heads' layers always fit one request (a9 raised the cap from 8,
2026-11-02). The heads are known only after the anatomy's contributions, so the worker runs one
pattern pass: a second replay of the same segments asking only for those layers (§6 alternative
(a), exact).
Patterns are mandatory under `standard-v4`; a library that cannot capture them fails attestation.

## 10. Binding budget (§2.4)

Every drowse kernel's emitted WGSL keeps at most 8 storage bindings (WebGPU's default
`maxStorageBuffersPerShaderStage`), checked on the emitted shaders, not the TIR signatures. Results
are stored per layer into one shared allocation by an in-place store; a final all-layer `concat` is
forbidden. `unit_meta` packs every integer table and `unit_coeff` every float coefficient, which
keeps every unit kernel within 8:

| kernel (per layer) | bindings |
|---|---|
| `drowse_head_ablate_L` (before `o_proj`) | `z` in/out, `positions`, `unit_meta`, `unit_coeff`, `unit_active` = 5 (in place) |
| `drowse_neuron_ablate_L` (before `down_proj`) | `act` in/out, `positions`, `unit_meta`, `unit_coeff`, `unit_active` = 5 (in place) |
| `drowse_unit_tap_L` (after the ablations) | `z`, `attn_o`, `act`, `unit_meta`, `tap_grams`, `unit_taps` in/out = 6 (both families; Qwen3 binds `attn_o` without reading it) |
| anatomy store (per site; 8 per layer) | `x`, `positions`, `replay_meta`, `anatomy_taps` in/out = 4 |
| activity, neurons | `act`, `positions`, `replay_meta`, `unit_scales`, `activity_state`, `traces`, `population` = 7 |
| activity, heads A (values) | `z`, `attn_o`, `head_grams`, `unit_scales`, `replay_meta`, `head_values` out = 6 (as built: `replay_meta` lets an anatomy-only replay skip the quadratic forms; one workgroup per row, heads looped, barriers in uniform control flow) |
| activity, heads B (reduce) | `head_values`, `positions`, `replay_meta`, `activity_state`, `traces`, `population` = 6 |
| pattern K capture | `qkv`, `positions`, `replay_meta`, `pattern_keys`, `pattern_q` = 5 |
| static-read kernels (§8) | at most 7: `drowse_vocab_rows` and `drowse_vocab_scores` (ids or rows, two weight buffers, gain or shift, offset or mask, factor or flags, out); the layer projections 6; the gathers 5 |
| `drowse_attention_rows` | `pattern_keys`, `pattern_q`, `rows_meta`, rows out = 4 |
| `drowse_head_write_gram_L` | dequantized `W_O` (γ applied), `head_grams` in/out = 2 |

As built (B4, stage 1): three dispatches per layer in the units pair. The neuron and head taps
share one kernel run after both ablations, so it reads the ablated `z` and `act` and the
pre-norm `attn_o` (Gemma's `r_ao`); one workgroup per tap slot, barriers only under control flow
that depends on the workgroup id (WGSL uniformity), and the ablations are in place, so a row
whose position is below `UNIT_POSITIONS_FROM` is untouched. The upstream graph is otherwise
unchanged: every v4 kernel is byte-identical in a v5 library (binding names aside).

**The library's binding.** `libraryBindingBytes` (`modelBackend.ts`) is the largest buffer a v5
library allocates at its largest context and prefill chunk, the larger of:

- its largest weight tensor as the GPU holds it (`float32` for a bf16-encoded record);
- its units machinery (`blockSitesBindingBytes`, units libraries): `pattern_keys` whole, all 16
  slots at the full context (one buffer as built, sized to the positions a replay feeds, so only a
  full-context pattern replay reaches it), the head grams, the chunk's MLP gate/up activation
  `[prefill_chunk_size, 2·d_ff]` (MLC's static memory plan allocates it at the chunk bound; a
  replay chunk's `act` is half of it), a layer's head rows (`drowse_head_rows` over every head,
  `[H, 4, hd, d]`; the adapter splits a longer `unitWeightRows` request into calls of at most `H`
  heads or `⌊4·H·hd/3⌋` neurons, `unitRowsPerCall`) and every other `blockSitesBufferElements`
  buffer;
- a unit read's vocabulary chunk (`UNIT_VOCAB_ROWS` = 8192 rows times a factor of at most
  `max(d, H·hd)` columns), the structured program's buffers, one full-vocabulary static read of
  `VOCAB_READ_ROWS` (64) rows (`[64, V]`; the adapter splits longer
  `drowse_vocab_logits`/`drowse_footprint_scores` reads into calls of 64 rows) and one resident
  lens layer `[d, d]`. J-lens and SAE dictionaries upload in chunks of at most 32 MiB, and a
  residual capture of many positions ends each chunk before the first position whose
  `[L, positions, d]` output would outgrow the device's binding (bridge .47,
  `drowseCapturePositionsPerForward`; qwen3-4b takes 728 positions per forward at 256 MiB).

| library (MiB) | largest weight | `pattern_keys` | head grams | gate/up | binding |
|---|---|---|---|---|---|
| gemma3-270m-instruct (q0f32) | 640 (embedding) | 64 | 18 | 32 | 640 |
| gemma3-1b-instruct, gemma3-1b-pt | 144 | 64, 32 | 26 | 108, 54 | 144 |
| gemma3-4b-instruct | 320.1 | 256 | 68 | 160 | 320.1 |
| qwen3-1.7b | 148.4 | 256 | 28 | 96 | 256 |
| qwen3-4b | 185.5 | 256 | 72 | 152 | 256 |
| gpt2-base (q0f32), pythia-70m-base (q0f32), qwen35-2b-base | 147.2, 98.3, 242.5 | | | (48 on qwen35) | 147.2, 98.3, 242.5 |

The catalog builder derives every variant's `maxStorageBufferBindingSize` and `maxBufferSize` as
this binding (a release spec cannot declare them); capability admission compares the adapter's
reported limits with the variant's. So admission implies loadability for a catalog the builder
made, which is what the release signs. The browser's own catalog check is a narrower floor: it
refuses a units variant whose limits fall below its lock entry's `blockSitesBindingBytes`, the
part of the binding the lock alone determines (the weight and vocabulary terms need the
library's `tensor-cache.json` and chat config, which the lock does not record). At load the adapter derives the same
binding from the verified `tensor-cache.json` and `mlc-chat-config.json` and the lock entry
(`webLlmEngine.ts`; the variant's declared limits only gate admission), and the bridge
(0.2.84-drowse.47) asks the device for exactly it (`buffer_size_required_bytes`; WebGPU's defaults
of 128 MiB binding and 256 MiB buffer when they are larger) and refuses an adapter that reports
less before any weight is read, so every admitted adapter runs the model under the same limits and
an adapter the catalog admits can load it. (The bridge used to request 1 GiB when offered and
128 MiB otherwise, so an adapter between the catalog's 256 or 512 MiB and 1 GiB was admitted and
then could not bind the embedding.) On the M5's Chrome (adapter `maxStorageBufferBindingSize`
4294967292), all nine libraries load, prefill a full chunk and generate at exactly their binding,
and the largest buffer each page creates is the weight above (the next is the gate/up activation).
The binding is per variant, at its largest context: qwen3-1.7b at the 2,048 tokens iPhone and iPad
run needs 148.4 MiB, but its variant declares 256 MiB.

## 11. Carried from v4 unchanged

**Residual contract.** The post-block hook runs after each zero-based block on `[batch, sequence,
hidden]`; a tuple's element zero is the residual. Decode capture reads `[0, sequence − 1, :]`.
Hook-disabled execution allocates no steering or probe buffers and leaves logits unchanged.

**Affine and curved terms.** Affine injection in fp32,
`delta = along · (target − collapse · dot(h − neutral, basis))`, `h' = h + delta · basis`,
deliberately uncapped (Python's `subspace_inject`). Curved terms keep Python's safety cap, carry
the final prompt foot into the first decode step and then the previous decode foot; each curve
slot is a fixed 671-float record (rank 8, 32 RBF nodes, 4 intrinsic and 8 embedded dimensions;
domain tags 0 unused, 1 box/custom, 2 sphere; periodic axes by cos/sin embedding; spheres by
log at the neutral origin, parallel transport, scaled exponential map). Up to four curves compose
in slot order. Probes read the post-injection residual at the same boundary; dynamic terms see
phase, decode step, thinking state and prior-step gate scalars, never a partial current-step
measurement. Probe kind 1 is `dot(h, direction) + bias`, kind 2 its ReLU (the SAE encoder
channel), kind 3 a full-vocabulary J-lens probability after the fp32 transport and the native
final norm and head.

**Geometry** (v3's output `[L, 8, 41]`, records `[valid, fraction, residual, membership,
coords[4], node_distance[33]]`). One packed payload with ten flat sections and one alias view, so
the kernel binds exactly eight buffers. Low-rank inverse `Σ⁻¹h = h/ridge + U diag(c) Uᵀh`; flat
probes map the whitened reduced coordinate through `coord_map + coord_bias`; curved probes solve
the nearest foot (cold: 3 restarts × 12 iterations; warm: carried foot plus nearest nodes,
2 × 4); membership `exp(−r²/(2σ(foot)²))` with a sigma field, else 1. With
`persist_subspace_coords` the host reconstructs each layer's coordinate from the distances and the
immutable anchors and checks it reproduces every distance within fp32 tolerance.

**J-lens readout.** The live readout transports the `[L, d]` capture through resident fp32
`[1, d, d]` chunks no larger than `maxBufferSize` and `maxStorageBufferBindingSize`, runs the exact
full-vocabulary softmax once per chunk and synchronizes once; program replacement, clearing,
unload, device loss and worker termination dispose the persistent inputs.

**Events.** The worker persists each authoritative tree mutation before emitting `tree_mutated`;
`started → (tree_mutated | token)* → done`, every event with a monotonic sequence number and
generation id; a gap requires a full tree snapshot.

**Runtime identity.** An artifact binding names the source commit, converted manifest digest,
quantization, tokenizer and chat-template digests, compiled library digest, runtime ABI, this hook
ABI, layer map, hidden size and context profile. A friendly model id is never enough. Evidence
recorded against `post-block-residual-v4` does not carry to v5.
