// The runtime lock's per-model `blockSpec` (B1 §8.2) for Node tools: its key set is read from
// the browser's own definition (`BLOCK_SPEC_KEYS` in structuredHookProfile.ts, the order the
// schema-4 descriptor attests), and `blockSpecFromMlcConfig` mirrors the MLC overlay's
// `_drowse_block_spec()` (Gemma 3 and Qwen3) over a converted model's `mlc-chat-config.json`, so a
// candidate lock can be checked against the configuration the library was compiled from before
// any browser attests its descriptor.

import { readFileSync } from "node:fs";

const profileSource = readFileSync(
  new URL("../src/hosted/runtime/structuredHookProfile.ts", import.meta.url),
  "utf8",
);

function blockSpecKeys(source) {
  const match = /export const BLOCK_SPEC_KEYS = Object\.freeze\(\[([^\]]*)\]/u.exec(source);
  if (!match) throw new Error("structuredHookProfile.ts does not define BLOCK_SPEC_KEYS");
  const keys = [...match[1].matchAll(/"([A-Za-z]+)"/gu)].map((item) => item[1]);
  if (keys.length === 0 || new Set(keys).size !== keys.length) {
    throw new Error("structuredHookProfile.ts BLOCK_SPEC_KEYS is empty or repeats a key");
  }
  return Object.freeze(keys);
}

export const BLOCK_SPEC_KEYS = blockSpecKeys(profileSource);

const INTEGER_KEYS = ["dFF", "nHeads", "nKvHeads", "headDim"];
const BOOLEAN_KEYS = ["sandwich", "qkNorm", "ropeInGraph"];
const FLOAT_KEYS = ["embedScale", "rmsEps", "attnScaling", "ropeTheta", "ropeGlobalScale"];
const NULLABLE_FLOAT_KEYS = ["attnLogitSoftcap", "finalLogitSoftcap", "ropeLocalTheta"];
const NULLABLE_INTEGER_KEYS = ["slidingWindow", "slidingPattern"];

function positiveInteger(value) {
  return Number.isSafeInteger(value) && value > 0;
}

function positiveFloat32(value) {
  return typeof value === "number" && Number.isFinite(Math.fround(value)) && Math.fround(value) > 0;
}

/** Throws unless `value` is a lock block spec with exactly BLOCK_SPEC_KEYS (the browser's
 *  `validateBlockSpec` rules); returns it. */
export function validateLockBlockSpec(value, label = "block spec") {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  const keys = Object.keys(value).sort();
  const expected = [...BLOCK_SPEC_KEYS].sort();
  if (keys.length !== expected.length || keys.some((key, index) => key !== expected[index])) {
    throw new Error(`${label} has unknown or missing fields`);
  }
  if (
    INTEGER_KEYS.some((key) => !positiveInteger(value[key])) ||
    BOOLEAN_KEYS.some((key) => typeof value[key] !== "boolean") ||
    (value.normOffset !== 0 && value.normOffset !== 1) ||
    (value.activation !== "silu" && value.activation !== "gelu_tanh") ||
    FLOAT_KEYS.some((key) => !positiveFloat32(value[key])) ||
    NULLABLE_FLOAT_KEYS.some((key) => value[key] !== null && !positiveFloat32(value[key])) ||
    NULLABLE_INTEGER_KEYS.some((key) => value[key] !== null && !positiveInteger(value[key])) ||
    (value.slidingWindow === null) !== (value.slidingPattern === null) ||
    value.nHeads % value.nKvHeads !== 0
  ) {
    throw new Error(`${label} is invalid`);
  }
  return value;
}

/** Equality as the descriptor attests it: float fields as float32, everything else exactly. */
export function sameBlockSpec(left, right) {
  const floats = new Set([...FLOAT_KEYS, ...NULLABLE_FLOAT_KEYS]);
  return BLOCK_SPEC_KEYS.every((key) => {
    const a = left?.[key];
    const b = right?.[key];
    return floats.has(key) && typeof a === "number" && typeof b === "number"
      ? Math.fround(a) === Math.fround(b)
      : a === b;
  });
}

/** The block spec an architecture's library attests, from its converted `mlc-chat-config.json`
 *  (the overlay's `_drowse_block_spec`); null for an architecture without unit sites. */
export function blockSpecFromMlcConfig(architecture, chatConfig) {
  const config = chatConfig?.model_config;
  if (!config || typeof config !== "object") {
    throw new Error("mlc-chat-config.json has no model_config");
  }
  if (architecture === "gemma3_text") {
    const text = config.text_config ?? config;
    const factor = text.rope_scaling?.factor ?? 1;
    return validateLockBlockSpec({
      dFF: text.intermediate_size,
      nHeads: text.num_attention_heads,
      nKvHeads: text.num_key_value_heads,
      headDim: text.head_dim,
      sandwich: true,
      normOffset: 1,
      activation: "gelu_tanh",
      embedScale: text.hidden_size ** 0.5,
      rmsEps: text.rms_norm_eps,
      qkNorm: true,
      attnScaling: text.query_pre_attn_scalar ** -0.5,
      attnLogitSoftcap: null,
      finalLogitSoftcap: null,
      ropeTheta: text.position_embedding_base,
      ropeLocalTheta: text.rope_local_base_freq,
      ropeGlobalScale: 1 / factor,
      ropeInGraph: true,
      slidingWindow: text.sliding_window_size,
      slidingPattern: text.sliding_window_pattern,
    }, "gemma3_text block spec");
  }
  if (architecture === "qwen3") {
    if (config.rope_scaling || config.use_sliding_window === true) {
      throw new Error("qwen3 libraries rotate in the cache without scaling or a window");
    }
    return validateLockBlockSpec({
      dFF: config.intermediate_size,
      nHeads: config.num_attention_heads,
      nKvHeads: config.num_key_value_heads,
      headDim: config.head_dim,
      sandwich: false,
      normOffset: 0,
      activation: "silu",
      embedScale: 1,
      rmsEps: config.rms_norm_eps,
      qkNorm: true,
      attnScaling: config.head_dim ** -0.5,
      attnLogitSoftcap: null,
      finalLogitSoftcap: null,
      ropeTheta: config.rope_theta,
      ropeLocalTheta: null,
      ropeGlobalScale: 1,
      ropeInGraph: false,
      slidingWindow: null,
      slidingPattern: null,
    }, "qwen3 block spec");
  }
  return null;
}
