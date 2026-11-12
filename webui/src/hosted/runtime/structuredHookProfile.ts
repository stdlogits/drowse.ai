import type { StructuredHookProfileId } from "../../../../drowse/webui/src/lib/runtime/contracts";

export const WEB_LLM_RUNTIME_MIN_STORAGE_BUFFERS_PER_SHADER_STAGE = 10;
export const EXACT_READOUT_MIN_COMPUTE_WORKGROUP_SIZE_X = 256;
export const EXACT_READOUT_MIN_COMPUTE_INVOCATIONS_PER_WORKGROUP = 256;
export const EXACT_READOUT_REQUIRED_WEBGPU_LIMITS = Object.freeze({
  maxComputeWorkgroupSizeX: EXACT_READOUT_MIN_COMPUTE_WORKGROUP_SIZE_X,
  maxComputeInvocationsPerWorkgroup:
    EXACT_READOUT_MIN_COMPUTE_INVOCATIONS_PER_WORKGROUP,
} as const);

/** A hook ABI v6 capacity profile (B1 §3.5): `standard-v5` (unit sites) or
 *  `standard-v5-residual`, whose library attests `unitSites: 0` and every unit capacity 0. */
export interface StructuredHookCapacityProfile {
  readonly id: BlockSitesProfileId;
  readonly maxAffineGroups: number;
  readonly maxRank: number;
  readonly maxProbes: number;
  readonly maxCurves: number;
  readonly maxCurveNodes: number;
  readonly maxIntrinsicDim: number;
  readonly maxEmbedDim: number;
  readonly maxGeometryProbes: number;
  readonly maxWhitenerRank: number;
  readonly maxGeometryCandidates: number;
  readonly geometryOutputStride: number;
  readonly geometryFootRestarts: number;
  readonly geometryFootIterations: number;
  readonly geometryWarmRestarts: number;
  readonly geometryWarmIterations: number;
  readonly requiredMaxStorageBuffersPerShaderStage: number;
  readonly geometryKernelStorageBindings: number;
  readonly geometryHeaderStride: number;
  readonly maxComputeWorkgroupStorageSize: number;
  readonly unitSites: 0 | 1;
  readonly maxUnitGroups: number;
  readonly maxNeuronAblations: number;
  readonly maxHeadAblations: number;
  readonly maxUnitTaps: number;
  readonly maxHeadTaps: number;
  readonly maxActivityTraces: number;
  readonly maxActivitySpans: number;
  readonly maxPatternLayers: number;
  readonly maxPatternHeads: number;
  readonly maxLensReadTopK: number;
  readonly maxAnatomyUnits: number;
}

export interface StructuredHookMemoryRequirements {
  readonly curveParameterStride: number;
  readonly hostProgramBytes: number;
  readonly gpuInputBytes: number;
  readonly maxBufferSize: number;
  readonly maxStorageBufferBindingSize: number;
  readonly maxStorageBuffersPerShaderStage: number;
  readonly maxComputeWorkgroupStorageSize: number;
  readonly geometryHeaderBytes: number;
  readonly geometryPayloadBytes: number;
}

// ------------------------------------------------------------ hook ABI v6 ---
// v5 (B1-hook-abi-v5.md §3.5, §8.1) plus typed frames; browser-runtime/HOOK_ABI.md is the contract.

/** Hook ABI of every v6 library: in-block unit sites, replay, static reads, typed frames. */
export const BLOCK_SITES_HOOK_ABI = "block-sites-v6" as const;
/** Runtime ABI of the v6 WebLLM bridge. */
export const BLOCK_SITES_RUNTIME_ABI = "drowse-web-runtime-v3" as const;
/** Readout ABI of every v6 library (B1 §8.1, §4.1): lens reads at both sites, k ≤ 256. */
export const BLOCK_SITES_EXACT_READOUT_ABI = "exact-readout-v2" as const;

/** The hook ABI v6 profiles. */
export type BlockSitesProfileId = "standard-v5" | "standard-v5-residual";

/** A v6 profile id where drowse's shared types want `StructuredHookProfileId`, which still names
 *  the v5 profiles (`standard-v4*`, contracts.ts) until a drowse release renames them. */
export function sharedProfileId(id: BlockSitesProfileId): StructuredHookProfileId {
  return id as string as StructuredHookProfileId;
}

/** `UNIT_POSITIONS_FROM`: rows whose producing position is below it are excluded from unit
 *  ablation and activity (the sink at 0; unproduced rows below 0). B1 §2.3, §3.1. */
export const UNIT_POSITIONS_FROM = 1;
/** Storage bindings every drowse kernel's emitted WGSL stays within (B1 §2.4). */
export const DROWSE_KERNEL_STORAGE_BINDINGS = 8;

/** The steering, probe and geometry capacities both v6 profiles share (unchanged since the v4
 *  ABI's geometry profile). */
const STRUCTURED_HOOK_CAPACITIES = Object.freeze({
  maxAffineGroups: 4,
  maxRank: 8,
  maxProbes: 8,
  maxCurves: 4,
  maxCurveNodes: 32,
  maxIntrinsicDim: 4,
  maxEmbedDim: 8,
  maxGeometryProbes: 8,
  maxWhitenerRank: 96,
  maxGeometryCandidates: 33,
  geometryOutputStride: 41,
  geometryFootRestarts: 3,
  geometryFootIterations: 12,
  geometryWarmRestarts: 2,
  geometryWarmIterations: 4,
  requiredMaxStorageBuffersPerShaderStage: 8,
  geometryKernelStorageBindings: 8,
  geometryHeaderStride: 7,
  maxComputeWorkgroupStorageSize: 32 * 1024,
});

export const STANDARD_STRUCTURED_HOOK_PROFILE_V5: StructuredHookCapacityProfile = Object.freeze({
  id: "standard-v5",
  ...STRUCTURED_HOOK_CAPACITIES,
  unitSites: 1,
  maxUnitGroups: 4,
  maxNeuronAblations: 256,
  maxHeadAblations: 64,
  maxUnitTaps: 64,
  maxHeadTaps: 16,
  maxActivityTraces: 32,
  maxActivitySpans: 4,
  maxPatternLayers: 16,
  maxPatternHeads: 16,
  maxLensReadTopK: 256,
  maxAnatomyUnits: 64,
});

/** For `gpt2`, `gpt_neox` and `qwen3_5` (B1 §7). Lens reads stay k ≤ 256 (exact-readout-v2). */
export const STANDARD_STRUCTURED_HOOK_PROFILE_V5_RESIDUAL: StructuredHookCapacityProfile =
  Object.freeze({
    id: "standard-v5-residual",
    ...STRUCTURED_HOOK_CAPACITIES,
    unitSites: 0,
    maxUnitGroups: 0,
    maxNeuronAblations: 0,
    maxHeadAblations: 0,
    maxUnitTaps: 0,
    maxHeadTaps: 0,
    maxActivityTraces: 0,
    maxActivitySpans: 0,
    maxPatternLayers: 0,
    maxPatternHeads: 0,
    maxLensReadTopK: 256,
    maxAnatomyUnits: 0,
  });

const UNIT_PROFILE_KEYS = Object.freeze([
  "unitSites",
  "maxUnitGroups",
  "maxNeuronAblations",
  "maxHeadAblations",
  "maxUnitTaps",
  "maxHeadTaps",
  "maxActivityTraces",
  "maxActivitySpans",
  "maxPatternLayers",
  "maxPatternHeads",
  "maxLensReadTopK",
  "maxAnatomyUnits",
] as const satisfies readonly (keyof StructuredHookCapacityProfile)[]);

const PROFILE_KEYS = Object.freeze([
  "id",
  "maxAffineGroups",
  "maxRank",
  "maxProbes",
  "maxCurves",
  "maxCurveNodes",
  "maxIntrinsicDim",
  "maxEmbedDim",
  "maxGeometryProbes",
  "maxWhitenerRank",
  "maxGeometryCandidates",
  "geometryOutputStride",
  "geometryFootRestarts",
  "geometryFootIterations",
  "geometryWarmRestarts",
  "geometryWarmIterations",
  "requiredMaxStorageBuffersPerShaderStage",
  "geometryKernelStorageBindings",
  "geometryHeaderStride",
  "maxComputeWorkgroupStorageSize",
] as const);

const PROFILES: ReadonlyMap<string, StructuredHookCapacityProfile> = new Map([
  [STANDARD_STRUCTURED_HOOK_PROFILE_V5.id, STANDARD_STRUCTURED_HOOK_PROFILE_V5],
  [STANDARD_STRUCTURED_HOOK_PROFILE_V5_RESIDUAL.id, STANDARD_STRUCTURED_HOOK_PROFILE_V5_RESIDUAL],
]);

/** The hook ABI v6 profile named `id` (`standard-v5` or `standard-v5-residual`); any other id,
 *  including the v1-v3 profiles of retired libraries, is a TypeError. */
export function resolveStructuredHookProfile(
  id: unknown,
): StructuredHookCapacityProfile {
  if (typeof id !== "string" || !PROFILES.has(id)) {
    throw new TypeError(`Unsupported structured hook capacity profile ${String(id)}`);
  }
  return PROFILES.get(id)!;
}

export function validateStructuredHookProfile(
  value: unknown,
): StructuredHookCapacityProfile {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError("Structured hook capacity profile must be an object");
  }
  const candidate = value as Record<string, unknown>;
  const profile = resolveStructuredHookProfile(candidate.id);
  const canonical = [...PROFILE_KEYS, ...UNIT_PROFILE_KEYS];
  const keys = Object.keys(candidate).sort();
  const expectedKeys = [...canonical].sort();
  if (
    keys.length !== expectedKeys.length ||
    keys.some((key, index) => key !== expectedKeys[index])
  ) {
    throw new TypeError("Structured hook capacity profile has an unsupported shape");
  }
  for (const key of canonical) {
    if (candidate[key] !== (profile as unknown as Record<string, unknown>)[key]) {
      throw new TypeError(`Structured hook capacity profile ${profile.id} was modified`);
    }
  }
  validateProfileInvariants(profile);
  return profile;
}

// -------------------------------------------------------------- BlockSpec ---

export type BlockActivation = "silu" | "gelu_tanh";

/** The lock's per-model block description (B1 §8.2), recorded from native `model_spec` and the
 *  source checkpoint's config; required with `standard-v5`, forbidden with `-residual`. Every
 *  field is attested by the library's descriptor (schema 4). `null` means "none". */
export interface BlockSpec {
  readonly dFF: number;
  readonly nHeads: number;
  readonly nKvHeads: number;
  readonly headDim: number;
  /** Gemma's sandwich norms around the attention and MLP writes (`N_ao`, `N_mo`). */
  readonly sandwich: boolean;
  /** RMSNorm weight offset: 1 for Gemma's `1 + w`, 0 for Qwen3's `w`. */
  readonly normOffset: 0 | 1;
  readonly activation: BlockActivation;
  /** Input embedding scale `s_E` (Gemma `√d`, Qwen3 1). */
  readonly embedScale: number;
  readonly rmsEps: number;
  readonly qkNorm: boolean;
  /** Attention score scale (Gemma `query_pre_attn_scalar^-0.5`, Qwen3 `head_dim^-0.5`). */
  readonly attnScaling: number;
  readonly attnLogitSoftcap: number | null;
  readonly finalLogitSoftcap: number | null;
  /** RoPE θ of the global layers. */
  readonly ropeTheta: number;
  /** RoPE θ of the local (sliding) layers, or null when every layer uses `ropeTheta`. */
  readonly ropeLocalTheta: number | null;
  /** 1 / the global layers' rope scaling factor (1 when unscaled). */
  readonly ropeGlobalScale: number;
  /** Gemma rotates q/k in the graph (`_gemma3_rotary`); Qwen3's paged KV cache rotates them. */
  readonly ropeInGraph: boolean;
  /** Local layers attend to `i − W < j ≤ i`. */
  readonly slidingWindow: number | null;
  /** Layer `L` is global when `(L + 1) % pattern == 0`, local otherwise. */
  readonly slidingPattern: number | null;
}

/** B1 §8.2's field order, which is also the order the descriptor attests them in. */
export const BLOCK_SPEC_KEYS = Object.freeze([
  "dFF",
  "nHeads",
  "nKvHeads",
  "headDim",
  "sandwich",
  "normOffset",
  "activation",
  "embedScale",
  "rmsEps",
  "qkNorm",
  "attnScaling",
  "attnLogitSoftcap",
  "finalLogitSoftcap",
  "ropeTheta",
  "ropeLocalTheta",
  "ropeGlobalScale",
  "ropeInGraph",
  "slidingWindow",
  "slidingPattern",
] as const satisfies readonly (keyof BlockSpec)[]);

export function validateBlockSpec(value: unknown): BlockSpec {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError("Block spec must be an object");
  }
  const candidate = value as Record<string, unknown>;
  const keys = Object.keys(candidate).sort();
  const expected = [...BLOCK_SPEC_KEYS].sort();
  if (keys.length !== expected.length || keys.some((key, index) => key !== expected[index])) {
    throw new TypeError("Block spec has an unsupported shape");
  }
  for (const key of ["dFF", "nHeads", "nKvHeads", "headDim"] as const) {
    positiveInteger(candidate[key], `block spec ${key}`);
  }
  for (const key of ["sandwich", "qkNorm", "ropeInGraph"] as const) {
    if (typeof candidate[key] !== "boolean") throw new TypeError(`block spec ${key} must be a boolean`);
  }
  if (candidate.normOffset !== 0 && candidate.normOffset !== 1) {
    throw new TypeError("block spec normOffset must be 0 or 1");
  }
  if (candidate.activation !== "silu" && candidate.activation !== "gelu_tanh") {
    throw new TypeError("block spec activation must be silu or gelu_tanh");
  }
  for (const key of ["embedScale", "rmsEps", "attnScaling", "ropeTheta", "ropeGlobalScale"] as const) {
    positiveFloat32(candidate[key], `block spec ${key}`);
  }
  for (const key of ["attnLogitSoftcap", "finalLogitSoftcap", "ropeLocalTheta"] as const) {
    if (candidate[key] !== null) positiveFloat32(candidate[key], `block spec ${key}`);
  }
  for (const key of ["slidingWindow", "slidingPattern"] as const) {
    if (candidate[key] !== null) positiveInteger(candidate[key], `block spec ${key}`);
  }
  if ((candidate.slidingWindow === null) !== (candidate.slidingPattern === null)) {
    throw new TypeError("block spec slidingWindow and slidingPattern are set together");
  }
  if ((candidate.nHeads as number) % (candidate.nKvHeads as number) !== 0) {
    throw new TypeError("block spec nHeads must be a multiple of nKvHeads");
  }
  return Object.freeze({ ...candidate }) as unknown as BlockSpec;
}

/** Floats per layer of the replay's anatomy taps: `block_in | mid | out_pre | out | attn_o |
 *  mlp_o | act | z` = `6·d + d_ff + H·hd` (B1 §2.3). */
export function anatomyStride(hiddenSize: number, spec: BlockSpec): number {
  positiveInteger(hiddenSize, "anatomy hidden size");
  return checkedStructuredHookSum("anatomy stride", [
    checkedStructuredHookProduct("anatomy residual fields", 6, hiddenSize),
    spec.dFF,
    checkedStructuredHookProduct("anatomy head site", spec.nHeads, spec.headDim),
  ]);
}

/** Whether model layer `layer` is a local (sliding-window) layer under `spec`. */
export function isLocalAttentionLayer(spec: BlockSpec, layer: number): boolean {
  return spec.slidingPattern !== null && (layer + 1) % spec.slidingPattern !== 0;
}

// ------------------------------------------------------ descriptor schema 4 ---

export const HOOK_PROFILE_DESCRIPTOR_MAGIC = 0x53414b4c;
export const HOOK_PROFILE_DESCRIPTOR_SCHEMA_VERSION = 4;
/** Descriptor field 2 for both v5 profiles; `UNIT_SITES` says which of the two. */
export const HOOK_PROFILE_DESCRIPTOR_PROFILE_V5 = 5;
export const BLOCK_SITES_HOOK_ABI_VERSION = 6;
export const STRUCTURED_HOOK_FORMAT_V5_VERSION = 5;
export const BLOCK_SITES_EXACT_READOUT_ABI_VERSION = 2;
/** The live readout's top-k (descriptor field 29, unchanged from v3). */
export const LIVE_READOUT_TOP_K = 8;
export const MAX_SAE_FEATURES_PER_CHUNK = 16384;

/** `drowse_hook_profile`, schema 4 (B1 §3.5): an int32 vector in this order. The first 31
 *  fields are v3's, unchanged in meaning (hook ABI 6, format 5, exact readout 2); `*_F32_BITS`
 *  fields hold the IEEE-754 bits of `Math.fround(value)` as a signed int32, 0 meaning none. */
export const HOOK_PROFILE_DESCRIPTOR_V4_FIELDS = Object.freeze([
  "MAGIC",
  "SCHEMA_VERSION",
  "PROFILE",
  "HOOK_ABI_VERSION",
  "FORMAT_VERSION",
  "LAYER_COUNT",
  "HIDDEN_SIZE",
  "MAX_AFFINE_GROUPS",
  "MAX_RANK",
  "MAX_PROBES",
  "MAX_CURVES",
  "MAX_CURVE_NODES",
  "MAX_INTRINSIC_DIM",
  "MAX_EMBED_DIM",
  "CURVE_PARAMETER_STRIDE",
  "MIN_COMPUTE_WORKGROUP_STORAGE_SIZE",
  "MAX_GEOMETRY_PROBES",
  "MAX_WHITENER_RANK",
  "MAX_GEOMETRY_CANDIDATES",
  "GEOMETRY_OUTPUT_STRIDE",
  "GEOMETRY_FOOT_RESTARTS",
  "GEOMETRY_FOOT_ITERATIONS",
  "GEOMETRY_WARM_RESTARTS",
  "GEOMETRY_WARM_ITERATIONS",
  "REQUIRED_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE",
  "GEOMETRY_KERNEL_STORAGE_BINDINGS",
  "GEOMETRY_HEADER_STRIDE",
  "GEOMETRY_PAYLOAD_ELEMENTS",
  "EXACT_READOUT_ABI_VERSION",
  "READOUT_TOP_K",
  "MAX_SAE_FEATURES_PER_CHUNK",
  "UNIT_SITES",
  "D_FF",
  "N_Q_HEADS",
  "N_KV_HEADS",
  "HEAD_DIM",
  "SANDWICH",
  "NORM_OFFSET",
  "ACTIVATION",
  "EMBED_SCALE_F32_BITS",
  "RMS_EPS_F32_BITS",
  "QK_NORM",
  "ATTN_SCALING_F32_BITS",
  "ATTN_LOGIT_SOFTCAP_F32_BITS",
  "FINAL_LOGIT_SOFTCAP_F32_BITS",
  "ROPE_THETA_F32_BITS",
  "ROPE_LOCAL_THETA_F32_BITS",
  "ROPE_GLOBAL_SCALE_F32_BITS",
  "ROPE_IN_GRAPH",
  "SLIDING_WINDOW",
  "SLIDING_PATTERN",
  "PREFILL_CHUNK_SIZE",
  "MAX_UNIT_GROUPS",
  "MAX_NEURON_ABLATIONS",
  "MAX_HEAD_ABLATIONS",
  "MAX_UNIT_TAPS",
  "MAX_HEAD_TAPS",
  "MAX_ACTIVITY_TRACES",
  "MAX_ACTIVITY_SPANS",
  "MAX_PATTERN_LAYERS",
  "MAX_PATTERN_HEADS",
  "ANATOMY_STRIDE",
  "UNIT_POSITIONS_FROM",
  "MAX_LENS_READ_TOP_K",
  "MAX_ANATOMY_UNITS",
  "DROWSE_KERNEL_STORAGE_BINDINGS",
  "CONTEXT_TOKENS",
] as const);

export type HookProfileDescriptorField = (typeof HOOK_PROFILE_DESCRIPTOR_V4_FIELDS)[number];
export const HOOK_PROFILE_DESCRIPTOR_V4_LENGTH = HOOK_PROFILE_DESCRIPTOR_V4_FIELDS.length;

/** What the descriptor is checked against besides the profile: the model's lock entry. */
export interface HookProfileDescriptorModel {
  readonly layerCount: number;
  readonly hiddenSize: number;
  /** The compiled context window: the largest of the lock's `contextProfiles`. */
  readonly contextTokens: number;
  /** The compiled prefill chunk: the lock's `prefillChunkSize` (at most its smallest context). */
  readonly prefillChunkSize: number;
  /** Required with `standard-v5`, null with `standard-v5-residual`. */
  readonly blockSpec: BlockSpec | null;
}

/** The IEEE-754 single-precision bits of `Math.fround(value)` as a signed int32. */
export function float32Bits(value: number): number {
  const bits = new Int32Array(new Float32Array([value]).buffer);
  return bits[0];
}

/** The exact descriptor a library compiled under `profile` for `model` must report; the bridge
 *  compares element by element and fails the load with `HOOK_ABI_MISMATCH` (B1 §3.5). A residual
 *  library reports 0 in every block field, every unit capacity, `ANATOMY_STRIDE`,
 *  `UNIT_POSITIONS_FROM` and `MAX_ANATOMY_UNITS`. */
export function expectedHookProfileDescriptor(
  input: StructuredHookCapacityProfile,
  model: HookProfileDescriptorModel,
): Int32Array {
  const profile = validateStructuredHookProfile(input);
  positiveInteger(model.layerCount, "descriptor layer count");
  positiveInteger(model.hiddenSize, "descriptor hidden size");
  positiveInteger(model.contextTokens, "descriptor context tokens");
  positiveInteger(model.prefillChunkSize, "descriptor prefill chunk size");
  if (model.prefillChunkSize > model.contextTokens) {
    throw new RangeError("descriptor prefill chunk size exceeds the context window");
  }
  const units = profile.unitSites === 1;
  if (units !== (model.blockSpec !== null)) {
    throw new TypeError(
      units
        ? "standard-v5 requires the model's block spec"
        : "standard-v5-residual forbids a block spec",
    );
  }
  const spec = model.blockSpec === null ? null : validateBlockSpec(model.blockSpec);
  const bits = (value: number | null): number => (value === null ? 0 : float32Bits(value));
  const flag = (value: boolean): number => (value ? 1 : 0);
  const fields: Record<HookProfileDescriptorField, number> = {
    MAGIC: HOOK_PROFILE_DESCRIPTOR_MAGIC,
    SCHEMA_VERSION: HOOK_PROFILE_DESCRIPTOR_SCHEMA_VERSION,
    PROFILE: HOOK_PROFILE_DESCRIPTOR_PROFILE_V5,
    HOOK_ABI_VERSION: BLOCK_SITES_HOOK_ABI_VERSION,
    FORMAT_VERSION: STRUCTURED_HOOK_FORMAT_V5_VERSION,
    LAYER_COUNT: model.layerCount,
    HIDDEN_SIZE: model.hiddenSize,
    MAX_AFFINE_GROUPS: profile.maxAffineGroups,
    MAX_RANK: profile.maxRank,
    MAX_PROBES: profile.maxProbes,
    MAX_CURVES: profile.maxCurves,
    MAX_CURVE_NODES: profile.maxCurveNodes,
    MAX_INTRINSIC_DIM: profile.maxIntrinsicDim,
    MAX_EMBED_DIM: profile.maxEmbedDim,
    CURVE_PARAMETER_STRIDE: structuredCurveParameterStride(profile),
    MIN_COMPUTE_WORKGROUP_STORAGE_SIZE: profile.maxComputeWorkgroupStorageSize,
    MAX_GEOMETRY_PROBES: profile.maxGeometryProbes,
    MAX_WHITENER_RANK: profile.maxWhitenerRank,
    MAX_GEOMETRY_CANDIDATES: profile.maxGeometryCandidates,
    GEOMETRY_OUTPUT_STRIDE: profile.geometryOutputStride,
    GEOMETRY_FOOT_RESTARTS: profile.geometryFootRestarts,
    GEOMETRY_FOOT_ITERATIONS: profile.geometryFootIterations,
    GEOMETRY_WARM_RESTARTS: profile.geometryWarmRestarts,
    GEOMETRY_WARM_ITERATIONS: profile.geometryWarmIterations,
    REQUIRED_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE: profile.requiredMaxStorageBuffersPerShaderStage,
    GEOMETRY_KERNEL_STORAGE_BINDINGS: profile.geometryKernelStorageBindings,
    GEOMETRY_HEADER_STRIDE: profile.geometryHeaderStride,
    GEOMETRY_PAYLOAD_ELEMENTS: structuredGeometryPayloadLayout(
      profile,
      model.layerCount,
      model.hiddenSize,
    ).elements,
    EXACT_READOUT_ABI_VERSION: BLOCK_SITES_EXACT_READOUT_ABI_VERSION,
    READOUT_TOP_K: LIVE_READOUT_TOP_K,
    MAX_SAE_FEATURES_PER_CHUNK,
    UNIT_SITES: profile.unitSites,
    D_FF: spec?.dFF ?? 0,
    N_Q_HEADS: spec?.nHeads ?? 0,
    N_KV_HEADS: spec?.nKvHeads ?? 0,
    HEAD_DIM: spec?.headDim ?? 0,
    SANDWICH: spec === null ? 0 : flag(spec.sandwich),
    NORM_OFFSET: spec?.normOffset ?? 0,
    ACTIVATION: spec === null ? 0 : spec.activation === "silu" ? 1 : 2,
    EMBED_SCALE_F32_BITS: bits(spec?.embedScale ?? null),
    RMS_EPS_F32_BITS: bits(spec?.rmsEps ?? null),
    QK_NORM: spec === null ? 0 : flag(spec.qkNorm),
    ATTN_SCALING_F32_BITS: bits(spec?.attnScaling ?? null),
    ATTN_LOGIT_SOFTCAP_F32_BITS: bits(spec?.attnLogitSoftcap ?? null),
    FINAL_LOGIT_SOFTCAP_F32_BITS: bits(spec?.finalLogitSoftcap ?? null),
    ROPE_THETA_F32_BITS: bits(spec?.ropeTheta ?? null),
    ROPE_LOCAL_THETA_F32_BITS: bits(spec?.ropeLocalTheta ?? null),
    ROPE_GLOBAL_SCALE_F32_BITS: bits(spec?.ropeGlobalScale ?? null),
    ROPE_IN_GRAPH: spec === null ? 0 : flag(spec.ropeInGraph),
    SLIDING_WINDOW: spec?.slidingWindow ?? 0,
    SLIDING_PATTERN: spec?.slidingPattern ?? 0,
    PREFILL_CHUNK_SIZE: model.prefillChunkSize,
    MAX_UNIT_GROUPS: profile.maxUnitGroups,
    MAX_NEURON_ABLATIONS: profile.maxNeuronAblations,
    MAX_HEAD_ABLATIONS: profile.maxHeadAblations,
    MAX_UNIT_TAPS: profile.maxUnitTaps,
    MAX_HEAD_TAPS: profile.maxHeadTaps,
    MAX_ACTIVITY_TRACES: profile.maxActivityTraces,
    MAX_ACTIVITY_SPANS: profile.maxActivitySpans,
    MAX_PATTERN_LAYERS: profile.maxPatternLayers,
    MAX_PATTERN_HEADS: profile.maxPatternHeads,
    ANATOMY_STRIDE: spec === null ? 0 : anatomyStride(model.hiddenSize, spec),
    UNIT_POSITIONS_FROM: units ? UNIT_POSITIONS_FROM : 0,
    MAX_LENS_READ_TOP_K: profile.maxLensReadTopK,
    MAX_ANATOMY_UNITS: profile.maxAnatomyUnits,
    DROWSE_KERNEL_STORAGE_BINDINGS,
    CONTEXT_TOKENS: model.contextTokens,
  };
  return Int32Array.from(HOOK_PROFILE_DESCRIPTOR_V4_FIELDS, (name) => fields[name]);
}

/** Names of the fields where a library's descriptor differs from the expected one (a length
 *  mismatch names `LENGTH`), for the `HOOK_ABI_MISMATCH` message. Empty means attested. */
export function hookProfileDescriptorMismatches(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
): string[] {
  if (actual.length !== HOOK_PROFILE_DESCRIPTOR_V4_LENGTH || expected.length !== actual.length) {
    return ["LENGTH"];
  }
  const mismatches: string[] = [];
  for (let index = 0; index < actual.length; index += 1) {
    if (actual[index] !== expected[index]) mismatches.push(HOOK_PROFILE_DESCRIPTOR_V4_FIELDS[index]);
  }
  return mismatches;
}

/** Which v5 profile a schema-4 descriptor claims (`UNIT_SITES` decides), before the full check. */
export function hookProfileDescriptorProfileId(values: ArrayLike<number>): BlockSitesProfileId {
  const at = (name: HookProfileDescriptorField): number =>
    values[HOOK_PROFILE_DESCRIPTOR_V4_FIELDS.indexOf(name)];
  if (
    values.length !== HOOK_PROFILE_DESCRIPTOR_V4_LENGTH ||
    at("MAGIC") !== HOOK_PROFILE_DESCRIPTOR_MAGIC ||
    at("SCHEMA_VERSION") !== HOOK_PROFILE_DESCRIPTOR_SCHEMA_VERSION ||
    at("PROFILE") !== HOOK_PROFILE_DESCRIPTOR_PROFILE_V5 ||
    (at("UNIT_SITES") !== 0 && at("UNIT_SITES") !== 1)
  ) {
    throw new TypeError("The model library's hook profile descriptor is not schema 4");
  }
  return at("UNIT_SITES") === 1 ? "standard-v5" : "standard-v5-residual";
}

export function structuredCurveParameterStride(
  input: StructuredHookCapacityProfile,
): number {
  const profile = validateStructuredHookProfile(input);
  return checkedStructuredHookSum("curve parameter stride", [
    checkedStructuredHookProduct(
      "curve node parameter fields",
      profile.maxCurveNodes,
      profile.maxEmbedDim,
    ),
    checkedStructuredHookProduct(
      "curve RBF fields",
      profile.maxCurveNodes,
      profile.maxRank,
    ),
    checkedStructuredHookProduct(
      "curve polynomial fields",
      profile.maxEmbedDim + 1,
      profile.maxRank,
    ),
    profile.maxEmbedDim,
    profile.maxEmbedDim,
    profile.maxIntrinsicDim,
    profile.maxIntrinsicDim,
    checkedStructuredHookProduct(
      "curve bound fields",
      profile.maxIntrinsicDim,
      2,
    ),
    profile.maxIntrinsicDim,
    profile.maxIntrinsicDim,
    profile.maxCurveNodes,
    profile.maxEmbedDim + 1,
    6,
  ]);
}

export interface StructuredGeometryPayloadLayout {
  readonly offsets: readonly number[];
  readonly elements: number;
}

export function structuredGeometryPayloadLayout(
  input: StructuredHookCapacityProfile,
  layerCount: number,
  hiddenSize: number,
): StructuredGeometryPayloadLayout {
  const profile = validateStructuredHookProfile(input);
  positiveInteger(layerCount, "structured hook layer count");
  positiveInteger(hiddenSize, "structured hook hidden size");
  if (profile.maxGeometryProbes === 0) {
    return Object.freeze({ offsets: Object.freeze([]), elements: 0 });
  }
  const slots = checkedStructuredHookProduct(
    "geometry payload slots",
    layerCount,
    profile.maxGeometryProbes,
  );
  const fieldElements = [
    checkedStructuredHookProduct("geometry payload means", slots, hiddenSize),
    checkedStructuredHookProduct("geometry payload inverse means", slots, hiddenSize),
    checkedStructuredHookProduct("geometry payload basis", slots, profile.maxRank, hiddenSize),
    checkedStructuredHookProduct("geometry payload Gram inverse", slots, profile.maxRank, profile.maxRank),
    checkedStructuredHookProduct("geometry payload Cholesky", slots, profile.maxRank, profile.maxRank),
    checkedStructuredHookProduct(
      "geometry payload candidate coordinates",
      slots,
      profile.maxGeometryCandidates,
      profile.maxRank,
    ),
    checkedStructuredHookProduct(
      "geometry payload coordinate map",
      slots,
      profile.maxIntrinsicDim,
      profile.maxRank,
    ),
    checkedStructuredHookProduct("geometry payload coordinate bias", slots, profile.maxIntrinsicDim),
    checkedStructuredHookProduct(
      "geometry payload curve parameters",
      slots,
      structuredCurveParameterStride(profile),
    ),
    checkedStructuredHookProduct(
      "geometry payload curve node coordinates",
      slots,
      profile.maxCurveNodes,
      profile.maxIntrinsicDim,
    ),
    checkedStructuredHookProduct(
      "geometry payload curve node values",
      slots,
      profile.maxCurveNodes,
      profile.maxRank,
    ),
  ];
  const offsets: number[] = [];
  let elements = 0;
  for (const field of fieldElements) {
    const byteOffset = checkedStructuredHookProduct(
      "geometry payload byte offset",
      elements,
      Float32Array.BYTES_PER_ELEMENT,
    );
    if (byteOffset % Float32Array.BYTES_PER_ELEMENT !== 0) {
      throw new RangeError("geometry payload fields are not float32-aligned");
    }
    offsets.push(elements);
    elements = checkedStructuredHookSum("geometry payload elements", [elements, field]);
  }
  return Object.freeze({ offsets: Object.freeze(offsets), elements });
}

export function structuredHookMemoryRequirements(
  input: StructuredHookCapacityProfile,
  layerCount: number,
  hiddenSize: number,
  options: { includeJlens?: boolean } = {},
): StructuredHookMemoryRequirements {
  const profile = validateStructuredHookProfile(input);
  positiveInteger(layerCount, "structured hook layer count");
  positiveInteger(hiddenSize, "structured hook hidden size");

  const l = layerCount;
  const h = hiddenSize;
  const g = profile.maxAffineGroups;
  const r = profile.maxRank;
  const p = profile.maxProbes;
  const c = profile.maxCurves;
  const n = profile.maxCurveNodes;
  const i = profile.maxIntrinsicDim;
  const e = profile.maxEmbedDim;
  const geometryProbes = profile.maxGeometryProbes;
  const whitenerRank = profile.maxWhitenerRank;
  const geometryCandidates = profile.maxGeometryCandidates;
  const stride = structuredCurveParameterStride(profile);
  const lg = checkedStructuredHookProduct("affine slots", l, g);
  const lp = checkedStructuredHookProduct("probe slots", l, p);
  const lc = checkedStructuredHookProduct("curve slots", l, c);
  const geometrySlots = checkedStructuredHookProduct(
    "geometry slots",
    l,
    geometryProbes,
  );
  const geometryPayload = structuredGeometryPayloadLayout(profile, l, h);
  const geometryHeaderElements = checkedStructuredHookProduct(
    "geometry header elements",
    geometrySlots,
    profile.geometryHeaderStride,
  );

  const hostElements = [
    lg,
    checkedStructuredHookProduct("affine basis elements", l, g, r, h),
    checkedStructuredHookProduct("affine neutral elements", l, g, h),
    checkedStructuredHookProduct("affine target elements", l, g, r),
    lg,
    checkedStructuredHookProduct("affine kappa elements", l, g, r),
    lp,
    checkedStructuredHookProduct("probe direction elements", l, p, h),
    lp,
    checkedStructuredHookProduct("curve scalar metadata elements", lc, 5),
    checkedStructuredHookProduct("curve basis elements", l, c, r, h),
    checkedStructuredHookProduct("curve neutral elements", l, c, h),
    checkedStructuredHookProduct("curve domain kind elements", l, c),
    checkedStructuredHookProduct("curve node parameter elements", l, c, n, e),
    checkedStructuredHookProduct("curve RBF elements", l, c, n, r),
    checkedStructuredHookProduct("curve polynomial elements", l, c, e + 1, r),
    checkedStructuredHookProduct("curve coordinate elements", l, c, e, 2),
    checkedStructuredHookProduct("curve point elements", l, c, i, 2),
    checkedStructuredHookProduct("curve along and onto elements", lc, 2),
    checkedStructuredHookProduct("curve bound elements", l, c, i, 2),
    checkedStructuredHookProduct("curve axis metadata elements", l, c, i, 2),
    lc,
    checkedStructuredHookProduct("curve sigma RBF elements", l, c, n),
    checkedStructuredHookProduct("curve sigma polynomial elements", l, c, e + 1),
    lc,
  ];
  if (geometryProbes > 0) {
    hostElements.push(
      l,
      l,
      checkedStructuredHookProduct(
        "geometry whitener basis elements",
        l,
        whitenerRank,
        h,
      ),
      checkedStructuredHookProduct(
        "geometry whitener correction elements",
        l,
        whitenerRank,
      ),
      checkedStructuredHookProduct("geometry scalar metadata elements", geometrySlots, 7),
      checkedStructuredHookProduct("geometry mean elements", geometrySlots, h),
      checkedStructuredHookProduct("geometry inverse mean elements", geometrySlots, h),
      checkedStructuredHookProduct("geometry basis elements", geometrySlots, r, h),
      checkedStructuredHookProduct("geometry Gram inverse elements", geometrySlots, r, r),
      checkedStructuredHookProduct("geometry Cholesky elements", geometrySlots, r, r),
      checkedStructuredHookProduct(
        "geometry node white elements",
        geometrySlots,
        geometryCandidates,
        r,
      ),
      checkedStructuredHookProduct("geometry coordinate map elements", geometrySlots, i, r),
      checkedStructuredHookProduct("geometry coordinate bias elements", geometrySlots, i),
      checkedStructuredHookProduct("geometry curve parameter elements", geometrySlots, stride),
      checkedStructuredHookProduct("geometry curve node coordinate elements", geometrySlots, n, i),
      checkedStructuredHookProduct("geometry curve node value elements", geometrySlots, n, r),
      checkedStructuredHookProduct("geometry foot elements", geometrySlots, i),
    );
  }
  if (options.includeJlens === true) {
    hostElements.push(
      checkedStructuredHookProduct("R-lens matrix elements", l, h, h),
      p,
    );
  }

  const gpuInputElements = [
    ...hostElements.slice(0, 9),
    checkedStructuredHookProduct("GPU curve basis elements", l, c, r, h),
    checkedStructuredHookProduct("GPU curve neutral elements", l, c, h),
    checkedStructuredHookProduct("GPU curve domain kind elements", l, c),
    checkedStructuredHookProduct("GPU curve parameter elements", l, c, stride),
    checkedStructuredHookProduct("GPU curve foot elements", l, c, i),
  ];
  if (geometryProbes > 0) {
    gpuInputElements.push(
      l,
      l,
      checkedStructuredHookProduct(
        "GPU geometry whitener basis elements",
        l,
        whitenerRank,
        h,
      ),
      checkedStructuredHookProduct(
        "GPU geometry whitener correction elements",
        l,
        whitenerRank,
      ),
      geometryHeaderElements,
      geometryPayload.elements,
      checkedStructuredHookProduct("GPU geometry foot elements", geometrySlots, i),
    );
  }
  if (options.includeJlens === true) {
    gpuInputElements.push(
      checkedStructuredHookProduct("GPU R-lens matrix elements", h, h),
      p,
      1,
    );
  }
  const outputElements = [
    lp,
    ...(geometryProbes > 0
      ? [
          checkedStructuredHookProduct(
            "GPU geometry output elements",
            geometrySlots,
            profile.geometryOutputStride,
          ),
        ]
      : []),
    ...(options.includeJlens === true
      ? [
          checkedStructuredHookProduct("GPU R-lens hidden elements", l, h),
          p,
          checkedStructuredHookProduct("GPU R-lens direction elements", p, h),
        ]
      : []),
  ];
  const largestElements = Math.max(...gpuInputElements, ...outputElements);
  const largestBytes = checkedStructuredHookProduct(
    "largest structured hook buffer bytes",
    largestElements,
    Float32Array.BYTES_PER_ELEMENT,
  );
  const geometryHeaderBytes = checkedStructuredHookProduct(
    "geometry header bytes",
    geometryHeaderElements,
    Uint32Array.BYTES_PER_ELEMENT,
  );
  const geometryPayloadBytes = checkedStructuredHookProduct(
    "geometry payload bytes",
    geometryPayload.elements,
    Float32Array.BYTES_PER_ELEMENT,
  );
  return {
    curveParameterStride: stride,
    hostProgramBytes: checkedStructuredHookProduct(
      "structured hook host program bytes",
      checkedStructuredHookSum("structured hook host program elements", hostElements),
      Float32Array.BYTES_PER_ELEMENT,
    ),
    gpuInputBytes: checkedStructuredHookProduct(
      "structured hook GPU input bytes",
      checkedStructuredHookSum("structured hook GPU input elements", gpuInputElements),
      Float32Array.BYTES_PER_ELEMENT,
    ),
    maxBufferSize: largestBytes,
    maxStorageBufferBindingSize: largestBytes,
    maxStorageBuffersPerShaderStage: profile.requiredMaxStorageBuffersPerShaderStage,
    maxComputeWorkgroupStorageSize: profile.maxComputeWorkgroupStorageSize,
    geometryHeaderBytes,
    geometryPayloadBytes,
  };
}

export function checkedStructuredHookProduct(
  label: string,
  ...factors: number[]
): number {
  let value = 1;
  for (const factor of factors) {
    if (!Number.isSafeInteger(factor) || factor < 0) {
      throw new RangeError(`${label} has an invalid factor`);
    }
    value *= factor;
    if (!Number.isSafeInteger(value)) {
      throw new RangeError(`${label} exceeds safe integer arithmetic`);
    }
  }
  return value;
}

function checkedStructuredHookSum(label: string, terms: readonly number[]): number {
  let value = 0;
  for (const term of terms) {
    if (!Number.isSafeInteger(term) || term < 0) {
      throw new RangeError(`${label} has an invalid term`);
    }
    value += term;
    if (!Number.isSafeInteger(value)) {
      throw new RangeError(`${label} exceeds safe integer arithmetic`);
    }
  }
  return value;
}

function validateProfileInvariants(profile: StructuredHookCapacityProfile): void {
  for (const key of PROFILE_KEYS.slice(1)) {
    positiveInteger(profile[key], `structured hook profile ${key}`);
  }
  if (profile.maxCurveNodes < 2) {
    throw new TypeError("Structured hook profiles require at least two curve nodes");
  }
  if (profile.maxRank < profile.maxIntrinsicDim) {
    throw new TypeError("Structured hook profiles require rank to cover intrinsic dimensions");
  }
  if (profile.maxEmbedDim < profile.maxIntrinsicDim * 2) {
    throw new TypeError("Structured hook profiles require two embedding slots per intrinsic axis");
  }
  if (profile.requiredMaxStorageBuffersPerShaderStage !== 8) {
    throw new TypeError("Structured hook profiles must fit WebGPU's guaranteed storage binding limit");
  }
  if (profile.maxGeometryCandidates !== profile.maxCurveNodes + 1) {
    throw new TypeError("Structured hook geometry requires one neutral candidate slot");
  }
  if (
    profile.geometryOutputStride !==
      4 + profile.maxIntrinsicDim + profile.maxGeometryCandidates
  ) {
    throw new TypeError("Structured hook geometry output stride is inconsistent");
  }
  if (
    profile.geometryHeaderStride !== 7 ||
    profile.geometryKernelStorageBindings !== 8 ||
    profile.geometryKernelStorageBindings > profile.requiredMaxStorageBuffersPerShaderStage
  ) {
    throw new TypeError("Structured hook geometry exceeds the WebGPU storage binding budget");
  }
  validateUnitInvariants(profile);
}

function validateUnitInvariants(profile: StructuredHookCapacityProfile): void {
  positiveInteger(profile.maxLensReadTopK, "structured hook profile maxLensReadTopK");
  const unitKeys = UNIT_PROFILE_KEYS.filter((key) => key !== "maxLensReadTopK");
  if (profile.unitSites === 0) {
    if (unitKeys.some((key) => profile[key] !== 0)) {
      throw new TypeError("A residual structured hook profile cannot declare unit capacities");
    }
    return;
  }
  if (profile.unitSites !== 1) throw new TypeError("Structured hook profile unitSites must be 0 or 1");
  for (const key of unitKeys) positiveInteger(profile[key], `structured hook profile ${key}`);
  if (profile.maxHeadTaps > profile.maxUnitTaps) {
    throw new TypeError("Structured hook head taps must fit within the unit tap slots");
  }
  if (profile.maxPatternHeads < profile.maxPatternLayers) {
    throw new TypeError("Structured hook pattern heads must cover every pattern layer");
  }
}

function positiveInteger(value: unknown, label: string): asserts value is number {
  if (!Number.isSafeInteger(value) || (value as number) < 1) {
    throw new TypeError(`${label} must be a positive integer`);
  }
}

function positiveFloat32(value: unknown, label: string): asserts value is number {
  if (
    typeof value !== "number" ||
    !Number.isFinite(value) ||
    value <= 0 ||
    !Number.isFinite(Math.fround(value)) ||
    Math.fround(value) === 0
  ) {
    throw new TypeError(`${label} must be a positive finite float32`);
  }
}
