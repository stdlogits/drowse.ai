import type {
  BrowserRuntimeClass,
  CatalogFile,
  CatalogInstrumentPack,
  CatalogModel,
  ModelVariant,
  RuntimeProgressEvent,
  RuntimeServiceRequest,
} from "../../../../drowse/webui/src/lib/runtime/contracts";
import type {
  WSGenerateRequest,
  WSServerMessage,
  WSSubmitRequest,
} from "../../../../drowse/webui/src/lib/types";
import type { ActivationSpool } from "../fitting/activationSpool";
import type { GpuAdapterLike } from "./capabilities";
import type { SteeringManifoldEntry } from "./steeringExpression";
import type {
  StructuredHookControlState,
  StructuredHookProgramBuffers,
  StructuredUnitKind,
} from "./structuredHookProgram";
import {
  anatomyStride,
  BLOCK_SITES_EXACT_READOUT_ABI,
  BLOCK_SITES_HOOK_ABI,
  BLOCK_SITES_RUNTIME_ABI,
  checkedStructuredHookProduct,
  UNIT_POSITIONS_FROM,
  validateBlockSitesProfile,
  type BlockSitesCapacityProfile,
  type BlockSpec,
  type StructuredHookCapacityProfile,
} from "./structuredHookProfile";

export interface BrowserModelArtifact {
  manifest: CatalogFile;
  file: File;
}

export interface BrowserInstrumentPackArtifacts {
  pack: CatalogInstrumentPack;
  /** The files that install with the pack (`packInstallFiles`). */
  artifacts: BrowserModelArtifact[];
  /** A layer file the pack fetches on first read (`packFileResidency` "on_demand"): the verified
   *  local copy, downloaded into the pack's install the first time. Absent where nothing can be
   *  fetched (release validation, a fixture): such a read refuses. */
  fetchFile?: (manifest: CatalogFile, signal: AbortSignal) => Promise<File>;
}

export interface BrowserModelLoadRequest {
  model: CatalogModel;
  variant: ModelVariant;
  /** The runtime lock's `blockSpec` for this variant (B1 §8.2), which a `standard-v4` library's
   *  descriptor is attested against; null or absent for a residual library (and until W4
   *  records it, when a `standard-v4` load fails closed with `HOOK_ABI_MISMATCH`). */
  blockSpec?: BlockSpec | null;
  /** The runtime lock's `unitRefusal` for a residual variant (native `UnsupportedBlockError`'s
   *  text, B1 §7): what its unit members answer (422). */
  unitRefusal?: string | null;
  requiredCorePack: CatalogInstrumentPack;
  contextTokens: number;
  runtimeClass?: BrowserRuntimeClass;
  adapter: GpuAdapterLike | null;
  artifacts: BrowserModelArtifact[];
  optionalPacks: BrowserInstrumentPackArtifacts[];
  activationSpool: ActivationSpool;
  signal: AbortSignal;
  onProgress?: (event: RuntimeProgressEvent) => void;
  onDeviceLost(failure: {
    code: string;
    message: string;
    confirmedOom: boolean;
  }): void;
}

export interface BrowserModelLoadResult {
  prefillTokensPerSecond: number | null;
  decodeTokensPerSecond: number | null;
}

export type BrowserGenerationPerformance = BrowserModelLoadResult;
export type BrowserGenerationStopReason = "user" | "internal";

export interface BrowserModelBackend {
  load(request: BrowserModelLoadRequest): Promise<BrowserModelLoadResult>;
  unload(): Promise<void>;
  refreshManifolds(signal: AbortSignal): Promise<void>;
  // Aborting must roll back staged fitting artifacts and settle before unload can proceed.
  request(
    request: RuntimeServiceRequest,
    onProgress: (event: RuntimeProgressEvent) => void,
    signal: AbortSignal,
  ): Promise<unknown>;
  generate(
    request: WSSubmitRequest | WSGenerateRequest,
    emit: (message: WSServerMessage) => void | Promise<void>,
    onAdmitted?: () => void,
  ): Promise<BrowserGenerationPerformance | void>;
  stop(reason?: BrowserGenerationStopReason): Promise<void>;
  /** The loaded compiler's installed manifolds, read without waiting on anything, for the
   *  steering checks drowse makes before its session lock; `null` when there is none to ask. */
  steeringManifolds?(): readonly SteeringManifoldEntry[] | null;
  /** The loaded library's hook ABI v5 device surface (`block-sites-v5`), read without waiting.
   *  Before a v5 library is loaded its `capabilities()` is null and every member rejects. */
  blockSites(): BrowserBlockSitesBackend;
}

export class UnavailableBrowserModelBackend implements BrowserModelBackend {
  private readonly unavailableBlockSites = rejectingBlockSitesBackend(backendUnavailable);

  blockSites(): BrowserBlockSitesBackend {
    return this.unavailableBlockSites;
  }

  async load(): Promise<never> {
    throw backendUnavailable();
  }

  async unload(): Promise<void> {}

  async refreshManifolds(): Promise<void> {}

  async request(): Promise<never> {
    throw backendUnavailable();
  }

  async generate(): Promise<never> {
    throw backendUnavailable();
  }

  async stop(): Promise<void> {}
}

function backendUnavailable(): Error & {
  code: string;
  status: number;
  recoverable: boolean;
} {
  return Object.assign(
    new Error(
      "The Drowse MLC fork with post-block steering hooks is not available; stock WebLLM is not used as a fallback",
    ),
    {
      code: "CUSTOM_MLC_BACKEND_UNAVAILABLE",
      status: 503,
      recoverable: false,
    },
  );
}

// =================================================================================
// Hook ABI v5 (`block-sites-v5`): the device surface the worker (B5) calls and the
// WebLLM bridge (B4) implements. Section numbers are B1-hook-abi-v5.md's; the contract is
// browser-runtime/HOOK_ABI.md. Every tensor crossing the seam is a typed array with explicit
// dims and dtype, row-major; every value is fp32 over the resident (dequantized q4) weights.
// =================================================================================

// --------------------------------------------------------------------- tensors ---

export type BlockSitesDType = "f32" | "i32" | "u32" | "f16";

interface TypedTensor<D extends BlockSitesDType, A> {
  readonly dtype: D;
  readonly shape: readonly number[];
  readonly data: A;
}

export type F32Tensor = TypedTensor<"f32", Float32Array>;
export type I32Tensor = TypedTensor<"i32", Int32Array>;
export type U32Tensor = TypedTensor<"u32", Uint32Array>;
/** IEEE binary16 bit patterns (native's packed `float16` little-endian population). */
export type F16Tensor = TypedTensor<"f16", Uint16Array>;
export type BlockSitesTensor = F32Tensor | I32Tensor | U32Tensor | F16Tensor;

function tensor<D extends BlockSitesDType, A extends { length: number }>(
  dtype: D,
  data: A,
  shape: readonly number[],
): TypedTensor<D, A> {
  const elements = checkedStructuredHookProduct(`${dtype} tensor`, ...shape);
  if (data.length !== elements) {
    throw new TypeError(`A ${dtype} tensor of shape [${shape.join(", ")}] needs ${elements} values`);
  }
  return Object.freeze({ dtype, shape: Object.freeze([...shape]), data });
}

export const f32Tensor = (data: Float32Array, shape: readonly number[]): F32Tensor =>
  tensor("f32", data, shape);
export const i32Tensor = (data: Int32Array, shape: readonly number[]): I32Tensor =>
  tensor("i32", data, shape);
export const u32Tensor = (data: Uint32Array, shape: readonly number[]): U32Tensor =>
  tensor("u32", data, shape);
export const f16Tensor = (data: Uint16Array, shape: readonly number[]): F16Tensor =>
  tensor("f16", data, shape);

/** Check a tensor's dtype and dims; `null` in `shape` accepts any extent there. */
export function requireTensorShape<T extends BlockSitesTensor>(
  value: T,
  dtype: T["dtype"],
  shape: readonly (number | null)[],
  label: string,
): T {
  const arrays = { f32: Float32Array, i32: Int32Array, u32: Uint32Array, f16: Uint16Array };
  if (
    value.dtype !== dtype ||
    !(value.data instanceof arrays[dtype]) ||
    value.shape.length !== shape.length ||
    value.shape.some((extent, axis) => shape[axis] !== null && extent !== shape[axis]) ||
    value.data.length !== checkedStructuredHookProduct(label, ...value.shape)
  ) {
    const want = shape.map((extent) => extent ?? "*").join(", ");
    throw new TypeError(`${label} must be ${dtype} [${want}], got ${value.dtype} [${value.shape.join(", ")}]`);
  }
  return value;
}

// ------------------------------------------------------------------- residency ---

/** Buffers that stay on the GPU across calls until released or the model unloads. */
export type BlockSitesResidentKind =
  | "head_grams"
  | "unit_scales"
  | "lens_layer"
  | "embedding_statistics"
  | "pattern_capture";

export interface BlockSitesResident<K extends BlockSitesResidentKind = BlockSitesResidentKind> {
  readonly kind: K;
  readonly id: string;
  readonly bytes: number;
  readonly shape: readonly number[];
}

export type LensFamily = "r" | "s";
export type LensSite = "out" | "mid";

/** One fp32 lens matrix, resident as a `[1, d, d]` chunk (v4's one-matrix-per-chunk rule): R's
 *  `J_L` or S's `K_L`, fetched per layer on first read (PLAN: full-depth packs, B1 §5.3). */
export interface BlockSitesLensLayer extends BlockSitesResident<"lens_layer"> {
  readonly family: LensFamily;
  readonly site: LensSite;
  readonly layer: number;
}

/** The S lens's embedding statistics (B1 §5.1): `mean_E [d]`, `Σ_E [d, d]` and the token mask
 *  (bit `v` of `mask[v >> 5]`, little-endian, set when `v` is kept). */
export interface BlockSitesEmbeddingStatistics {
  readonly meanE: F32Tensor;
  readonly sigmaE: F32Tensor;
  readonly mask: U32Tensor;
}

// ---------------------------------------------------------------- capabilities ---

export interface BlockSitesCapabilities {
  readonly hookAbi: typeof BLOCK_SITES_HOOK_ABI;
  readonly runtimeAbi: typeof BLOCK_SITES_RUNTIME_ABI;
  readonly exactReadoutAbi: typeof BLOCK_SITES_EXACT_READOUT_ABI;
  /** The attested profile (B1 §3.5); `unitSites` says whether the unit members run. */
  readonly profile: BlockSitesCapacityProfile;
  readonly layerCount: number;
  readonly hiddenSize: number;
  readonly vocabSize: number;
  /** The compiled context window (descriptor `CONTEXT_TOKENS`). */
  readonly contextTokens: number;
  /** No forward, replay included, carries more rows than this (B1 §2.3). */
  readonly prefillChunkSize: number;
  /** The lock's `blockSpec`, attested; null on a residual library. */
  readonly blockSpec: BlockSpec | null;
  /** The lock's `unitRefusal` on a residual library (native `UnsupportedBlockError`'s text, B1
   *  §7): every unit, anatomy, activity and `mid` member rejects with it (422). Null otherwise. */
  readonly unitRefusal: string | null;
}

/** Validate a capabilities record against its own profile (the bridge builds it at load). */
export function validateBlockSitesCapabilities(value: BlockSitesCapabilities): BlockSitesCapabilities {
  const profile = validateBlockSitesProfile(value.profile);
  if (
    value.hookAbi !== BLOCK_SITES_HOOK_ABI ||
    value.runtimeAbi !== BLOCK_SITES_RUNTIME_ABI ||
    value.exactReadoutAbi !== BLOCK_SITES_EXACT_READOUT_ABI
  ) {
    throw new TypeError("Block-sites capabilities name the wrong ABI");
  }
  for (const key of ["layerCount", "hiddenSize", "vocabSize", "contextTokens", "prefillChunkSize"] as const) {
    if (!Number.isSafeInteger(value[key]) || value[key] < 1) {
      throw new TypeError(`Block-sites capabilities ${key} must be a positive integer`);
    }
  }
  const units = profile.unitSites === 1;
  if (units !== (value.blockSpec !== null) || units === (typeof value.unitRefusal === "string")) {
    throw new TypeError(
      "A standard-v4 library carries a block spec and no refusal; a residual one the reverse",
    );
  }
  if (!units && value.unitRefusal!.length === 0) {
    throw new TypeError("A residual library's unit refusal must name the reason");
  }
  return value;
}

// ------------------------------------------------------------------ generation ---

/** What one forward of a generation under a bound drowse-structured-v4 program reads back in
 *  its single measurement-bundle sync (B1 §2.2). The worker divides each tap by its schema
 *  entry's reading scale; gates read the prior step's scaled value through `priorScores`. */
export interface BlockSitesForwardReadback {
  /** `[L, maxProbes]` probe slots, as the measurement bundle's `scalar` (a kind-3 slot holds
   *  its J-lens probability; NaN where the backend cannot read one). */
  readonly measurements: F32Tensor;
  /** `[maxUnitTaps]` raw tap values at the forward's last row in slot order (`act[last, n]`;
   *  `sqrt(max(0, z_hᵀ G_h z_h))·r_ao`), null when the program has no unit section. */
  readonly unitTaps: F32Tensor | null;
  /** `[L, maxGeometryProbes, geometryOutputStride]` geometry readout of the forward's last row
   *  (feet threaded through the replay as generation threads them), null when the program has
   *  no geometry probes. */
  readonly geometry: F32Tensor | null;
  /** `[L, d]` capture of the forward's last row when the program sets `hiddenCapture` (its
   *  `slens/` reads), else null. */
  readonly hidden: F32Tensor | null;
}

// ---------------------------------------------------------------------- replay ---

/** One logical segment of native's `replay_segments` (B1 §2.3): its tokens are fed at the next
 *  absolute positions (the first segment at 0, a fresh sequence) under one control state. The
 *  bridge runs a segment as consecutive chunks of at most `prefillChunkSize` rows with the
 *  controls held constant; the KV cache, curve feet, activity state, traces and pattern keys
 *  persist across chunks and segments. The host evaluates each segment's controls from the
 *  previous segment's readback, as native `drive_segments` does (a gated decode run is one
 *  position per segment). Position-targeted spans are split into segments by the caller
 *  (controls zeroed outside a span). */
export interface BlockSitesReplaySegment {
  readonly tokenIds: Int32Array;
  readonly controls: StructuredHookControlState;
}

/** A node's token rows read at consecutive forward columns (native `ReplayForward.spans()`):
 *  row `firstRow + i` is column `firstPosition + i`. A negative `firstPosition` marks the span's
 *  first `−firstPosition` rows unproduced: no column holds them and they read 0. */
export interface BlockSitesActivitySpan {
  readonly firstRow: number;
  readonly rows: number;
  readonly firstPosition: number;
}

export interface BlockSitesUnitRef {
  readonly kind: StructuredUnitKind;
  readonly layer: number;
  readonly index: number;
}

/** Unit activity over a node (B1 §2.3, native `_activity`): per row the scaled value (neuron
 *  `act / S_pole`, head write norm `/ S_h`), excluded (0, rank −1) where the row's column is below
 *  `UNIT_POSITIONS_FROM`; peaks keep the strictly greater `|v|`, ties the earlier row. Columns
 *  outside every span are not node rows and are skipped. */
export interface BlockSitesActivityRequest {
  readonly spans: readonly BlockSitesActivitySpan[];
  /** Token rows of the node: the trace length. */
  readonly rowCount: number;
  readonly scales: BlockSitesResident<"unit_scales">;
  /** Units traced row by row (≤ `maxActivityTraces`). */
  readonly traces: readonly BlockSitesUnitRef[];
  /** The token row whose whole population to return, or null. */
  readonly populationRow: number | null;
}

/** What a replay captures; its segments are fed one by one through `BrowserBlockSitesReplay`. */
export interface BlockSitesReplayRequest {
  /** The bound program the node ran under; null replays unsteered (empty tables). */
  readonly program: StructuredHookProgramBuffers | null;
  /** Absolute position whose anatomy taps to capture, or null (B1 §2.3). */
  readonly anatomyPosition: number | null;
  readonly activity: BlockSitesActivityRequest | null;
  /** Capture post-RoPE keys of every KV head at these layers (≤ `maxPatternLayers`) for every
   *  position, and every head's query at `anatomyPosition`, for `attentionRows` (B1 §6). */
  readonly patternLayers: readonly number[] | null;
  /** Positions the segments will feed in all (B4: sizes the pattern keys to the replay, and
   *  the bridge refuses a segment past it); absent, the compiled context. */
  readonly tokenCount?: number;
}

export interface BlockSitesReplayResult {
  /** `[L, ANATOMY_STRIDE]` at `anatomyPosition` (`anatomyTapLayout`), or null. */
  readonly anatomyTaps: F32Tensor | null;
  readonly activity: {
    /** `[L, 3·dFF + 3·H]`, `activityStateLayout` order; host ranks it (B1 §2.3 tie order). */
    readonly state: F32Tensor;
    /** `[traces, rowCount]` scaled values; excluded and unproduced rows read 0. */
    readonly traces: F32Tensor;
    /** `[L, dFF + H]` scaled values at `populationRow`, or null. */
    readonly population: F32Tensor | null;
  } | null;
  /** Keys and queries kept resident for `attentionRows`; the caller releases it. */
  readonly patterns: BlockSitesResident<"pattern_capture"> | null;
}

/** One replay in progress (B1 §2.3, §3.4): a fresh KV sequence and the capture buffers, held
 *  until `finish` or `cancel`. Calls are sequential; another replay or a generation waits. */
export interface BrowserBlockSitesReplay {
  /** Feed one logical segment (chunked at `prefillChunkSize`) and return its last forward's
   *  probe slots and unit taps, from which the host evaluates the next segment's controls. */
  forward(segment: BlockSitesReplaySegment): Promise<BlockSitesForwardReadback>;
  /** Read the captures back and drop the sequence. */
  finish(): Promise<BlockSitesReplayResult>;
  /** Drop the sequence and every capture buffer without reading them. */
  cancel(): Promise<void>;
}

// --------------------------------------------------------------- static reads ---

export type UnitWeightRowsKind = "neuron_write" | "neuron_read" | "head_write" | "head_read";

/** The device surface. Members whose formula touches a unit site or a block weight reject with
 *  `UnsupportedBlockError` (the lock's refusal) on a residual library; vocabulary, footprint, lens
 *  and utility members run on every v5 library. Arguments are validated before any dispatch. */
export interface BrowserBlockSitesBackend {
  /** The attested v5 surface, or null before a v5 library is loaded. */
  capabilities(): BlockSitesCapabilities | null;

  // ---- generation (§2.2, §3.4) ----
  /** §2.2/§3.4: check a drowse-structured-v4 program against the attested profile (a unit section
   *  on a residual library is the refusal), fill `tapGrams` with `G_h = Bᵀ B` of each tapped head
   *  (`drowse_head_write_grams`) and return the program to put in the generation plan. A program
   *  with a unit section runs `drowse_units_batch_*`; each forward reads back
   *  `BlockSitesForwardReadback`. */
  bindStructuredProgram(program: StructuredHookProgramBuffers): Promise<StructuredHookProgramBuffers>;

  // ---- residency (§2.3, §5.3) ----
  /** §2.3: `head_grams [L, H, hd, hd]`, `G_h = Bᵀ B` for every head, built once per loaded model. */
  residentHeadGrams(signal?: AbortSignal): Promise<BlockSitesResident<"head_grams">>;
  /** §2.3: soft floored reading scales `[L, 2·dFF + H]` (`unitScalesLayout`) for activity. */
  uploadUnitScales(scales: F32Tensor): Promise<BlockSitesResident<"unit_scales">>;
  /** §5.3: one lens matrix `[d, d]` (R `J_L` or S `K_L`) at `family`/`site`/`layer`. */
  uploadLensLayer(
    family: LensFamily,
    site: LensSite,
    layer: number,
    matrix: F32Tensor,
  ): Promise<BlockSitesLensLayer>;
  /** §5.1: the S embedding statistics, resident while any S read may run. */
  uploadEmbeddingStatistics(
    statistics: BlockSitesEmbeddingStatistics,
  ): Promise<BlockSitesResident<"embedding_statistics">>;
  /** Free a resident buffer; head grams are dropped with the model, not here. */
  release(resident: BlockSitesResident): Promise<void>;

  // ---- replay (§2.3, §3.4, §6) ----
  /** §2.3/§3.4 `drowse_replay_batch_prefill`: open a KV-continuing replay with the anatomy,
   *  activity and pattern taps asked for; the host feeds its segments one by one. */
  openReplay(request: BlockSitesReplayRequest, signal?: AbortSignal): Promise<BrowserBlockSitesReplay>;
  /** §6: `a_j = softmax_j(q_p·k_j·scaling)` over `j = 0…p` (`j > p` and, on a local layer,
   *  `j ≤ p − W` masked to 0) for each `(layer, head)` of the capture → `[n, p + 1]`. */
  attentionRows(
    capture: BlockSitesResident<"pattern_capture">,
    heads: readonly { layer: number; head: number }[],
  ): Promise<F32Tensor>;

  // ---- static reads over the resident weights (§4.1) ----
  /** §4.1 `drowse_unit_weight_rows`: `neuron_write` `s·γ_mo ⊙ W_down[:, n]` → `[n, d]`;
   *  `neuron_read` `(â, b̂)`, `â = γ_mi ⊙ W_gate[n] / max(‖·‖, 1e-12)` → `[n, 2, d]`;
   *  `head_write` `γ_ao ⊙ W_O[:, h·hd:(h+1)·hd]` → `[n, d, hd]`; `head_read` raw
   *  `(γ_ai ⊙ W_Q^h, γ_ai ⊙ W_K^g, γ_ai ⊙ W_V^g)` → `[n, 3, hd, d]`, units in request order. */
  unitWeightRows(kind: UnitWeightRowsKind, units: readonly BlockSitesUnitRef[]): Promise<F32Tensor>;
  /** §2.2/§4.1 `drowse_head_write_grams`: `G_h = Bᵀ B` per head → `[n, hd, hd]`. */
  headWriteGrams(heads: readonly { layer: number; head: number }[]): Promise<F32Tensor>;
  /** §4.1 `drowse_layer_write_scores`: `x [c, m, d]` at `layers [c]` → `x · (s·γ_mo ⊙ W_down)`
   *  `[c, m, dFF]`. */
  layerWriteScores(x: F32Tensor, layers: Int32Array): Promise<F32Tensor>;
  /** §4.1 `drowse_layer_read_scores`: `(â·x, b̂·x)` → `[c, 2, m, dFF]`. */
  layerReadScores(x: F32Tensor, layers: Int32Array): Promise<F32Tensor>;
  /** §4.1 `drowse_neuron_contributions`: `c_n = act_n · ((g ⊙ γ_mo)·W_down[:, n]) · r_mo · s`,
   *  `r_mo = rsqrt(mean(mlp_o²) + ε)` (Gemma; 1 on Qwen3); `g, mlpO [c, d]`, `act [c, dFF]` →
   *  `[c, dFF]`. */
  neuronContributions(
    g: F32Tensor,
    act: F32Tensor,
    mlpO: F32Tensor,
    layers: Int32Array,
  ): Promise<F32Tensor>;
  /** §4.1 `drowse_head_contributions`: `c_h = r_ao · Σ_i ((g ⊙ γ_ao)ᵀ W_O)[h·hd+i] · z[h·hd+i]`;
   *  `g, attnO [c, d]`, `z [c, H·hd]` → `[c, H]`. */
  headContributions(
    g: F32Tensor,
    z: F32Tensor,
    attnO: F32Tensor,
    layers: Int32Array,
  ): Promise<F32Tensor>;
  /** §4.1 `drowse_head_write_norms`: `sqrt(max(0, z_hᵀ G_h z_h)) · r_ao` → `[c, H]`. */
  headWriteNorms(z: F32Tensor, attnO: F32Tensor, layers: Int32Array): Promise<F32Tensor>;
  /** §4.1 `drowse_vocab_rows_times`: rows `ẽ_v = s_E·E[v] − mean_E` (`embed`, needs the
   *  statistics) or `γ_f ⊙ W_U[v]` (`unembed`) for `ids [n]`, times `factor [d, m]` or a
   *  resident lens matrix (`transpose` picks `M` or `Mᵀ`) → `[n, m]`. */
  vocabRowsTimes(request: {
    readonly kind: "embed" | "unembed";
    readonly ids: Int32Array;
    readonly factor: F32Tensor | { readonly lens: BlockSitesLensLayer; readonly transpose: boolean };
    readonly statistics?: BlockSitesResident<"embedding_statistics">;
  }): Promise<F32Tensor>;
  /** §4.1 `drowse_vocab_readout_topk`: `softmax(W_U · N_f(rows))` over the full vocabulary,
   *  fp32 normalizer; top `k ≤ maxLensReadTopK` by logit desc, id asc. `rows [n, d]`. */
  vocabReadoutTopK(rows: F32Tensor, k: number): Promise<{
    readonly ids: I32Tensor;
    readonly logits: F32Tensor;
    readonly logSumExp: F32Tensor;
  }>;
  /** §4.1 `drowse_vocab_token_logits`: the same path at `ids [n, j]` → logits `[n, j]`. */
  vocabTokenLogits(rows: F32Tensor, ids: I32Tensor): Promise<{
    readonly logits: F32Tensor;
    readonly logSumExp: F32Tensor;
  }>;
  /** §4.1 `drowse_vocab_logit_moments`: skew and kurtosis of `W_U · N_f(row)` over the kept ids
   *  (the statistics' mask; every id when absent) → `[n, 2]`. */
  vocabLogitMoments(
    rows: F32Tensor,
    statistics?: BlockSitesResident<"embedding_statistics">,
  ): Promise<F32Tensor>;
  /** §4.1/§5.1 `drowse_footprint_topk`: `z_v = (s_E·(u·E[v]) − u·mean_E) / sqrt(max(uᵀΣ_E u,
   *  1e-30))`, masked ids `−∞`; top `k ≤ maxLensReadTopK`, z desc, id asc. `u [n, d]`. */
  footprintTopK(
    u: F32Tensor,
    k: number,
    statistics: BlockSitesResident<"embedding_statistics">,
  ): Promise<{ readonly ids: I32Tensor; readonly z: F32Tensor }>;
  /** §4.1 `drowse_unembed_cotangent`: `u = rsqrt(mean(h²) + ε_f) · γ_f ⊙ W_U[v]`;
   *  `hFinal [n, d]`, `ids [n]` → `[n, d]`. */
  unembedCotangent(hFinal: F32Tensor, ids: Int32Array): Promise<F32Tensor>;
  /** §5.1 `drowse_slens_probes`: `z_L(v)` of `u = (h_L − μ_L) @ K_L` for each S:out layer and
   *  probe token; `hidden, meansOut [L, d]`, `eTilde [P, d]` → `[c, P]`. */
  slensProbes(request: {
    readonly hidden: F32Tensor;
    readonly meansOut: F32Tensor;
    readonly lens: readonly BlockSitesLensLayer[];
    readonly eTilde: F32Tensor;
    readonly statistics: BlockSitesResident<"embedding_statistics">;
  }): Promise<F32Tensor>;
  /** §3.3/§5.3 `drowse_footprint_directions`: `K_L ẽ_v` per S:out layer and id → `[c, n, d]`. */
  footprintDirections(
    lens: readonly BlockSitesLensLayer[],
    ids: Int32Array,
    statistics: BlockSitesResident<"embedding_statistics">,
  ): Promise<F32Tensor>;

  // ---- exact-readout-v2 utilities (weight-free, §4.1) ----
  /** Rows `[c, m, d]` through resident lens matrices: `x @ M` or, with `transpose`, `x @ Mᵀ`
   *  (R's `x @ Jᵀ`, S's `x @ K`) → `[c, m, d]`. */
  lensTransport(
    rows: F32Tensor,
    lens: readonly BlockSitesLensLayer[],
    transpose: boolean,
  ): Promise<F32Tensor>;
  /** `a [b, m, k] @ b [b, k, n]` → `[b, m, n]`. */
  batchedMatmul(a: F32Tensor, b: F32Tensor): Promise<F32Tensor>;
  /** `xᵢᵀ Mᵢ xᵢ` for `x [b, n]`, `M [b, n, n]`, `[n, n]` shared, or the resident `Σ_E` → `[b]`. */
  batchedQuadraticForm(
    x: F32Tensor,
    matrix: F32Tensor | BlockSitesResident<"embedding_statistics">,
  ): Promise<F32Tensor>;
  /** Euclidean norms of consecutive `width`-wide groups of `x [n, g·width]` → `[n, g]`. */
  groupedRowNorms(x: F32Tensor, width: number): Promise<F32Tensor>;
  /** Chan's streamed merge of `rows [r, d]` into `(count, mean [d], m2 [d])`. */
  chanMerge(
    state: { readonly count: number; readonly mean: F32Tensor; readonly m2: F32Tensor },
    rows: F32Tensor,
  ): Promise<{ readonly count: number; readonly mean: F32Tensor; readonly m2: F32Tensor }>;
  /** Row-wise log-sum-exp of `x [n, m]` → `[n]`. */
  logSumExp(x: F32Tensor): Promise<F32Tensor>;
  /** Exact row-wise top `k ≤ 1024` of `x [n, m]`: value desc, index asc. */
  topK(x: F32Tensor, k: number): Promise<{ readonly values: F32Tensor; readonly indices: I32Tensor }>;
}

/** Every `BrowserBlockSitesBackend` member except `capabilities`, for wrappers and tests. */
export const BLOCK_SITES_BACKEND_MEMBERS = Object.freeze([
  "bindStructuredProgram",
  "residentHeadGrams",
  "uploadUnitScales",
  "uploadLensLayer",
  "uploadEmbeddingStatistics",
  "release",
  "openReplay",
  "attentionRows",
  "unitWeightRows",
  "headWriteGrams",
  "layerWriteScores",
  "layerReadScores",
  "neuronContributions",
  "headContributions",
  "headWriteNorms",
  "vocabRowsTimes",
  "vocabReadoutTopK",
  "vocabTokenLogits",
  "vocabLogitMoments",
  "footprintTopK",
  "unembedCotangent",
  "slensProbes",
  "footprintDirections",
  "lensTransport",
  "batchedMatmul",
  "batchedQuadraticForm",
  "groupedRowNorms",
  "chanMerge",
  "logSumExp",
  "topK",
] as const satisfies readonly Exclude<keyof BrowserBlockSitesBackend, "capabilities">[]);

/** Members that need block sites: on a residual library they reject with the refusal. */
export const BLOCK_SITES_UNIT_MEMBERS = Object.freeze([
  "residentHeadGrams",
  "uploadUnitScales",
  "openReplay",
  "attentionRows",
  "unitWeightRows",
  "headWriteGrams",
  "layerWriteScores",
  "layerReadScores",
  "neuronContributions",
  "headContributions",
  "headWriteNorms",
] as const satisfies readonly (typeof BLOCK_SITES_BACKEND_MEMBERS)[number][]);

// ---------------------------------------------------------------------- errors ---

/** Native `drowse.errors.UnsupportedBlockError` (422): a block outside what a `BlockSpec`
 *  describes, raised by every unit read, anatomy, unit activity and `mid` read on such a model. */
export class UnsupportedBlockError extends Error {
  readonly code = "UnsupportedBlockError";
  readonly status = 422;
  readonly recoverable = false;
  readonly layer: number | null;
  readonly reason: string;

  constructor(reason: string, options: { layer?: number | null } = {}) {
    const layer = options.layer ?? null;
    super(layer === null ? reason : `layer ${layer}: ${reason}`);
    this.name = "UnsupportedBlockError";
    this.layer = layer;
    this.reason = reason;
  }
}

/** A surface with no v5 library behind it: `capabilities()` is null and every member rejects
 *  with `error()` (fixture runtimes, the unavailable backend, a v4 library). */
export function rejectingBlockSitesBackend(error: () => Error): BrowserBlockSitesBackend {
  const backend: Record<string, unknown> = { capabilities: () => null };
  for (const member of BLOCK_SITES_BACKEND_MEMBERS) {
    backend[member] = () => Promise.reject(error());
  }
  return Object.freeze(backend) as unknown as BrowserBlockSitesBackend;
}

// --------------------------------------------------------------------- layouts ---

/** Per-layer anatomy taps (B1 §2.3): `block_in | mid | out_pre | out | attn_o | mlp_o | act | z`;
 *  `block_in[0]` is the prepared input embedding, `out_pre` the block output before the v4
 *  post-block hook, `out` after it. Offsets are floats within one layer's stride. */
export function anatomyTapLayout(hiddenSize: number, spec: BlockSpec): {
  readonly stride: number;
  readonly blockIn: number;
  readonly mid: number;
  readonly outPre: number;
  readonly out: number;
  readonly attnO: number;
  readonly mlpO: number;
  readonly act: number;
  readonly z: number;
} {
  const d = hiddenSize;
  return Object.freeze({
    stride: anatomyStride(hiddenSize, spec),
    blockIn: 0,
    mid: d,
    outPre: 2 * d,
    out: 3 * d,
    attnO: 4 * d,
    mlpO: 5 * d,
    act: 6 * d,
    z: 6 * d + spec.dFF,
  });
}

/** Per-layer activity state (B1 §2.3): `neuron_peak_value | neuron_peak_row | neuron_peak_rank`
 *  `[dFF]` each, then `head_peak_value | head_peak_row | head_peak_rank` `[H]` each, fp32 (rows
 *  are exact below 2²⁴). Initial: value 0, row −1, rank −1. */
export function activityStateLayout(spec: BlockSpec): {
  readonly stride: number;
  readonly neuronValue: number;
  readonly neuronRow: number;
  readonly neuronRank: number;
  readonly headValue: number;
  readonly headRow: number;
  readonly headRank: number;
} {
  const f = spec.dFF;
  const h = spec.nHeads;
  return Object.freeze({
    stride: 3 * f + 3 * h,
    neuronValue: 0,
    neuronRow: f,
    neuronRank: 2 * f,
    headValue: 3 * f,
    headRow: 3 * f + h,
    headRank: 3 * f + 2 * h,
  });
}

export function initialActivityState(layerCount: number, spec: BlockSpec): Float32Array {
  const layout = activityStateLayout(spec);
  const state = new Float32Array(checkedStructuredHookProduct("activity state", layerCount, layout.stride));
  for (let layer = 0; layer < layerCount; layer += 1) {
    const base = layer * layout.stride;
    state.fill(-1, base + layout.neuronRow, base + layout.headValue);
    state.fill(-1, base + layout.headRow, base + layout.stride);
  }
  return state;
}

/** Per-layer reading scales for activity (B1 §2.3): `S+ [dFF] | S− [dFF] | S_head [H]`, each
 *  `sqrt(top² + (0.25·m)²)` floored at 1e-6 by the host (native `stats.reading_scales`). */
export function unitScalesLayout(spec: BlockSpec): {
  readonly stride: number;
  readonly positive: number;
  readonly negative: number;
  readonly head: number;
} {
  return Object.freeze({
    stride: 2 * spec.dFF + spec.nHeads,
    positive: 0,
    negative: spec.dFF,
    head: 2 * spec.dFF,
  });
}

/** `replay_meta` (i32), B1 §3.4, for capacity profile `profile`:
 *  `[0]` anatomy position (−1 none), `[1]` span count, then `maxActivitySpans` ×
 *  `(first_row, rows, first_position, unproduced)`, then population row (−1 none), trace count,
 *  `maxActivityTraces` × `(layer, kind 0 neuron | 1 head, index)` (−1 unused), pattern layer
 *  count, `maxPatternLayers` layers (−1 unused), and a flags word (bit 0 anatomy, bit 1
 *  activity, bit 2 patterns). */
export function replayMetaLayout(input: StructuredHookCapacityProfile): {
  readonly length: number;
  readonly anatomyPosition: number;
  readonly spanCount: number;
  readonly spans: number;
  readonly populationRow: number;
  readonly traceCount: number;
  readonly traces: number;
  readonly patternLayerCount: number;
  readonly patternLayers: number;
  readonly flags: number;
} {
  const profile = validateBlockSitesProfile(input);
  if (profile.unitSites !== 1) {
    throw new UnsupportedBlockError(`the ${profile.id} profile has no replay entry point`);
  }
  const spans = 2;
  const populationRow = spans + 4 * profile.maxActivitySpans;
  const traces = populationRow + 2;
  const patternLayerCount = traces + 3 * profile.maxActivityTraces;
  const patternLayers = patternLayerCount + 1;
  const flags = patternLayers + profile.maxPatternLayers;
  return Object.freeze({
    length: flags + 1,
    anatomyPosition: 0,
    spanCount: 1,
    spans,
    populationRow,
    traceCount: populationRow + 1,
    traces,
    patternLayerCount,
    patternLayers,
    flags,
  });
}

export const REPLAY_FLAG_ANATOMY = 1;
export const REPLAY_FLAG_ACTIVITY = 2;
export const REPLAY_FLAG_PATTERNS = 4;

/** Pack a replay request's metadata for `drowse_replay_batch_prefill`, checking every value
 *  against the profile's capacities and the model (positions and rows below the compiled context,
 *  unit indices below `dFF` or `nHeads`), so nothing wraps in the int32 table. */
export function encodeReplayMeta(
  input: StructuredHookCapacityProfile,
  model: {
    readonly layerCount: number;
    readonly contextTokens: number;
    readonly blockSpec: BlockSpec;
  },
  request: Pick<BlockSitesReplayRequest, "anatomyPosition" | "activity" | "patternLayers">,
): Int32Array {
  const profile = validateBlockSitesProfile(input);
  const layout = replayMetaLayout(profile);
  const { layerCount, contextTokens, blockSpec } = model;
  if (!Number.isSafeInteger(contextTokens) || contextTokens < 1 || contextTokens >= 2 ** 24) {
    throw new RangeError("Replay context must be a positive count below 2^24");
  }
  const meta = new Int32Array(layout.length).fill(-1);
  const index = (value: number, bound: number, label: string): number => {
    if (!Number.isSafeInteger(value) || value < 0 || value >= bound) {
      throw new RangeError(`Replay ${label} ${value} is out of range`);
    }
    return value;
  };
  let flags = 0;
  meta[layout.anatomyPosition] = -1;
  if (request.anatomyPosition !== null) {
    meta[layout.anatomyPosition] = index(request.anatomyPosition, contextTokens, "anatomy position");
    flags |= REPLAY_FLAG_ANATOMY;
  }
  const activity = request.activity;
  meta[layout.spanCount] = 0;
  meta[layout.traceCount] = 0;
  if (activity !== null) {
    if (activity.spans.length === 0 || activity.spans.length > profile.maxActivitySpans) {
      throw new RangeError(`Unit activity takes 1 to ${profile.maxActivitySpans} spans`);
    }
    if (activity.traces.length > profile.maxActivityTraces) {
      throw new RangeError(`Unit activity traces at most ${profile.maxActivityTraces} units`);
    }
    if (
      !Number.isSafeInteger(activity.rowCount) || activity.rowCount < 1 ||
      activity.rowCount > contextTokens
    ) {
      throw new RangeError("Unit activity rows must fit the compiled context");
    }
    meta[layout.spanCount] = activity.spans.length;
    activity.spans.forEach((span, at) => {
      if (
        !Number.isSafeInteger(span.firstRow) || span.firstRow < 0 ||
        !Number.isSafeInteger(span.rows) || span.rows < 1 ||
        span.firstRow + span.rows > activity.rowCount ||
        !Number.isSafeInteger(span.firstPosition) ||
        span.firstPosition <= -contextTokens ||
        span.firstPosition + span.rows > contextTokens
      ) {
        throw new RangeError("Unit activity span is outside the node's rows or the context");
      }
      const base = layout.spans + 4 * at;
      meta[base] = span.firstRow;
      meta[base + 1] = span.rows;
      meta[base + 2] = span.firstPosition;
      meta[base + 3] = span.firstPosition < 0 ? 1 : 0;
    });
    meta[layout.populationRow] = activity.populationRow === null
      ? -1
      : index(activity.populationRow, activity.rowCount, "population row");
    meta[layout.traceCount] = activity.traces.length;
    activity.traces.forEach((unit, at) => {
      const base = layout.traces + 3 * at;
      meta[base] = index(unit.layer, layerCount, "trace layer");
      if (unit.kind !== "neuron" && unit.kind !== "head") throw new RangeError("Replay trace kind is invalid");
      meta[base + 1] = unit.kind === "neuron" ? 0 : 1;
      meta[base + 2] = index(
        unit.index,
        unit.kind === "neuron" ? blockSpec.dFF : blockSpec.nHeads,
        "trace unit",
      );
    });
    flags |= REPLAY_FLAG_ACTIVITY;
  }
  meta[layout.patternLayerCount] = 0;
  if (request.patternLayers !== null) {
    const layers = request.patternLayers;
    if (request.anatomyPosition === null) {
      throw new RangeError("Pattern capture reads the query at the anatomy position");
    }
    if (layers.length === 0 || layers.length > profile.maxPatternLayers) {
      throw new RangeError(`Pattern capture takes 1 to ${profile.maxPatternLayers} layers`);
    }
    layers.forEach((layer, at) => {
      if (at > 0 && layer <= layers[at - 1]) throw new RangeError("Pattern layers must ascend");
      meta[layout.patternLayers + at] = index(layer, layerCount, "pattern layer");
    });
    meta[layout.patternLayerCount] = layers.length;
    flags |= REPLAY_FLAG_PATTERNS;
  }
  meta[layout.flags] = flags;
  return meta;
}

/** Element counts of every v5 buffer for one model (B1 §2.3, §2.4, §3.1, §6), for the binding
 *  budget (`maxStorageBufferBindingSize`) and the catalog's limits. `pattern_keys` is one slot. */
export function blockSitesBufferElements(
  input: StructuredHookCapacityProfile,
  model: {
    readonly layerCount: number;
    readonly hiddenSize: number;
    readonly contextTokens: number;
    readonly prefillChunkSize: number;
    readonly blockSpec: BlockSpec;
  },
): Readonly<Record<
  | "unitMeta" | "unitCoeff" | "unitActive" | "tapGrams" | "unitTaps" | "headGrams"
  | "anatomyTaps" | "activityState" | "traces" | "population" | "unitScales"
  | "patternKeysSlot" | "patternQ" | "replayMeta" | "chunkAct" | "chunkHeadValues",
  number
>> {
  const profile = validateBlockSitesProfile(input);
  const spec = model.blockSpec;
  const l = model.layerCount;
  const hd = spec.headDim;
  const size = (label: string, ...factors: number[]) =>
    checkedStructuredHookProduct(label, ...factors);
  const offsets = l + 1;
  return Object.freeze({
    unitMeta: 4 * offsets + 2 * (profile.maxNeuronAblations + profile.maxHeadAblations +
      profile.maxUnitTaps + profile.maxHeadTaps),
    unitCoeff: 2 * profile.maxNeuronAblations + profile.maxHeadAblations +
      size("head bias", profile.maxHeadAblations, hd),
    unitActive: profile.maxUnitGroups,
    tapGrams: size("tap grams", profile.maxHeadTaps, hd, hd),
    unitTaps: profile.maxUnitTaps,
    headGrams: size("head grams", l, spec.nHeads, hd, hd),
    anatomyTaps: size("anatomy taps", l, anatomyStride(model.hiddenSize, spec)),
    activityState: size("activity state", l, activityStateLayout(spec).stride),
    traces: size("traces", profile.maxActivityTraces, model.contextTokens),
    population: size("population", l, spec.dFF + spec.nHeads),
    unitScales: size("unit scales", l, unitScalesLayout(spec).stride),
    patternKeysSlot: size("pattern keys", spec.nKvHeads, model.contextTokens, hd),
    patternQ: size("pattern queries", profile.maxPatternLayers, spec.nHeads, hd),
    replayMeta: replayMetaLayout(profile).length,
    chunkAct: size("chunk act", model.prefillChunkSize, spec.dFF),
    chunkHeadValues: size("chunk head values", model.prefillChunkSize, spec.nHeads),
  });
}

/** Rows below this absolute position are never ablated or ranked (re-exported for callers). */
export { UNIT_POSITIONS_FROM };
