// The units-tiny-models-v1 fixture models (`../fixtures/units-tiny-models-v1.json`, from
// drowse's former browser-units generator): the portable generator, bit for bit in 32-bit
// integer arithmetic, the byte-level tokenizer and the hand-built lenses. Test-only: the CPU
// reference backend and the fixture runtimes build their weights from the model file's integers;
// nothing here reaches the GPU.
import type { BlockSpec } from "../structuredHookProfile";
import type { DenseBlockWeights, DenseModelWeights } from "./denseModel";

/** The model file's `arches.<arch>` record (`units-tiny-models-v1.json`). */
export interface TinyArchRecord {
  readonly family: string;
  readonly config: Record<string, unknown> & {
    readonly vocab_size: number;
    readonly hidden_size: number;
    readonly intermediate_size: number;
    readonly num_hidden_layers: number;
    readonly num_attention_heads: number;
    readonly num_key_value_heads: number;
    readonly head_dim: number;
    readonly rms_norm_eps: number;
    readonly layer_types: readonly string[];
  };
  readonly tensors: readonly {
    readonly name: string;
    readonly shape: readonly number[];
    readonly kind: "q4" | "norm";
    readonly exponent: number | null;
    readonly sha256: string;
  }[];
  readonly tokenizer: {
    readonly size: number;
    readonly specials: readonly string[];
    readonly bos: number | null;
    readonly eos: number;
    readonly pad: number;
  };
  readonly short_prompt: readonly number[];
  readonly lenses: Readonly<Record<string, readonly number[]>>;
}

export interface TinyModelFile {
  readonly generator: { readonly seed: number; readonly golden_ratio: number; readonly group: number; readonly group_scale_tag: number };
  readonly arches: Readonly<Record<"gemma3" | "qwen3", TinyArchRecord>>;
  readonly band: readonly [number, number];
}

const MASK_GOLDEN = 0x9e3779b9;
const LENS_ORDINAL = 1000;

/** lowbias32. */
export function lowbias32(input: number): number {
  let x = input >>> 0;
  x ^= x >>> 16;
  x = Math.imul(x, 0x7feb352d) >>> 0;
  x ^= x >>> 15;
  x = Math.imul(x, 0x846ca68b) >>> 0;
  x ^= x >>> 16;
  return x >>> 0;
}

function seedOf(seed: number, ordinal: number): number {
  return lowbias32((seed + Math.imul(MASK_GOLDEN | 0, ordinal)) >>> 0);
}

function uniform(seed: number, ordinal: number, count: number): Uint32Array {
  const s = seedOf(seed, ordinal);
  const out = new Uint32Array(count);
  for (let i = 0; i < count; i += 1) out[i] = lowbias32((i ^ s) >>> 0);
  return out;
}

/** A q4-exact tensor: groups of 32 along the last axis, ±7 first in every group, scales
 *  `2^−(E + bit)`. */
export function quantizedTensor(seed: number, ordinal: number, count: number, exponent: number): Float32Array {
  const u = uniform(seed, ordinal, count);
  const s = seedOf(seed, ordinal);
  const out = new Float32Array(count);
  for (let g = 0; g < count / 32; g += 1) {
    const bit = lowbias32((g ^ s ^ 0x80000000) >>> 0) % 2;
    const scale = 2 ** -(exponent + bit);
    for (let j = 0; j < 32; j += 1) {
      const i = 32 * g + j;
      const q = j === 0 ? (u[i] >= 0x80000000 ? 7 : -7) : (u[i] % 15) - 7;
      out[i] = q * scale;
    }
  }
  return out;
}

export function normTensor(seed: number, ordinal: number, count: number, gemma: boolean): Float32Array {
  const u = uniform(seed, ordinal, count);
  const out = new Float32Array(count);
  for (let i = 0; i < count; i += 1) {
    const frac = (u[i] >>> 9) * 2 ** -24;
    out[i] = gemma ? frac - 0.25 : 0.75 + frac;
  }
  return out;
}

/** `0.75·δ_ij + ((u >> 16) mod 256 − 128)·2⁻¹⁰` `[d, d]`. */
export function lensMatrix(seed: number, ordinal: number, d: number): Float32Array {
  const u = uniform(seed, ordinal, d * d);
  const out = new Float32Array(d * d);
  for (let i = 0; i < d * d; i += 1) {
    out[i] = (((u[i] >>> 16) % 256) - 128) * 2 ** -10 + (i % (d + 1) === 0 ? 0.75 : 0);
  }
  return out;
}

/** The fixture's lenses by `"R:out"` etc: layer → `[d, d]` (the sorted (family, site) ordinal). */
export function tinyLenses(file: TinyModelFile, arch: "gemma3" | "qwen3"): Map<string, Map<number, Float32Array>> {
  const record = file.arches[arch];
  const d = record.config.hidden_size;
  const keys = Object.keys(record.lenses).sort((a, b) => {
    const [fa, sa] = a.split(":");
    const [fb, sb] = b.split(":");
    return fa === fb ? (sa < sb ? -1 : sa > sb ? 1 : 0) : fa < fb ? -1 : 1;
  });
  const out = new Map<string, Map<number, Float32Array>>();
  keys.forEach((key, i) => {
    const layers = new Map<number, Float32Array>();
    for (const layer of record.lenses[key]) {
      layers.set(layer, lensMatrix(file.generator.seed, LENS_ORDINAL + 16 * i + layer, d));
    }
    out.set(key, layers);
  });
  return out;
}

export function tinyBlockSpec(record: TinyArchRecord): BlockSpec {
  const config = record.config as TinyArchRecord["config"] & {
    query_pre_attn_scalar?: number;
    sliding_window?: number | null;
    rope_parameters: Record<string, unknown>;
  };
  const gemma = record.family === "gemma3_text";
  const rope = config.rope_parameters as Record<string, { rope_theta: number }> & { rope_theta?: number };
  return Object.freeze({
    dFF: config.intermediate_size,
    nHeads: config.num_attention_heads,
    nKvHeads: config.num_key_value_heads,
    headDim: config.head_dim,
    sandwich: gemma,
    normOffset: gemma ? 1 : 0,
    activation: gemma ? "gelu_tanh" : "silu",
    embedScale: gemma ? Math.sqrt(config.hidden_size) : 1,
    rmsEps: config.rms_norm_eps,
    qkNorm: true,
    attnScaling: (gemma ? config.query_pre_attn_scalar! : config.head_dim) ** -0.5,
    attnLogitSoftcap: null,
    finalLogitSoftcap: null,
    ropeTheta: gemma ? rope.full_attention.rope_theta : rope.rope_theta!,
    ropeLocalTheta: gemma ? rope.sliding_attention.rope_theta : null,
    ropeGlobalScale: 1,
    ropeInGraph: gemma,
    slidingWindow: gemma ? config.sliding_window! : null,
    slidingPattern: gemma ? 2 : null,
  } satisfies BlockSpec);
}

/** Every tensor from the generator in the file's order (`name → fp32`). */
export function tinyTensors(file: TinyModelFile, arch: "gemma3" | "qwen3"): Map<string, Float32Array> {
  const record = file.arches[arch];
  const out = new Map<string, Float32Array>();
  record.tensors.forEach((entry, ordinal) => {
    const count = entry.shape.reduce((a, b) => a * b, 1);
    out.set(entry.name, entry.kind === "q4"
      ? quantizedTensor(file.generator.seed, ordinal, count, entry.exponent!)
      : normTensor(file.generator.seed, ordinal, count, arch === "gemma3"));
  });
  return out;
}

/** The fixture model as dense fp32 weights, norm gains with their offset folded in. */
export function tinyModelWeights(file: TinyModelFile, arch: "gemma3" | "qwen3"): DenseModelWeights {
  const record = file.arches[arch];
  const spec = tinyBlockSpec(record);
  const tensors = tinyTensors(file, arch);
  const get = (name: string): Float32Array => {
    const value = tensors.get(name);
    if (value === undefined) throw new Error(`tiny ${arch} lacks ${name}`);
    return value;
  };
  const gain = (name: string): Float32Array =>
    Float32Array.from(get(name), (w) => w + spec.normOffset);
  const blocks: DenseBlockWeights[] = [];
  for (let layer = 0; layer < record.config.num_hidden_layers; layer += 1) {
    const p = `model.layers.${layer}.`;
    const gemma = spec.sandwich;
    blocks.push({
      inputNorm: gain(`${p}input_layernorm.weight`),
      q: get(`${p}self_attn.q_proj.weight`),
      k: get(`${p}self_attn.k_proj.weight`),
      v: get(`${p}self_attn.v_proj.weight`),
      o: get(`${p}self_attn.o_proj.weight`),
      qNorm: gain(`${p}self_attn.q_norm.weight`),
      kNorm: gain(`${p}self_attn.k_norm.weight`),
      attnOutNorm: gemma ? gain(`${p}post_attention_layernorm.weight`) : null,
      mlpInNorm: gain(gemma ? `${p}pre_feedforward_layernorm.weight` : `${p}post_attention_layernorm.weight`),
      mlpOutNorm: gemma ? gain(`${p}post_feedforward_layernorm.weight`) : null,
      gate: get(`${p}mlp.gate_proj.weight`),
      up: get(`${p}mlp.up_proj.weight`),
      down: get(`${p}mlp.down_proj.weight`),
    });
  }
  return {
    spec,
    layerCount: record.config.num_hidden_layers,
    hiddenSize: record.config.hidden_size,
    vocabSize: record.config.vocab_size,
    embed: get("model.embed_tokens.weight"),
    finalNorm: gain("model.norm.weight"),
    blocks,
  };
}

// ------------------------------------------------------------------ tokenizer ---

/** GPT-2's byte → symbol map (`ByteLevel.alphabet()`). */
function byteSymbols(): string[] {
  const printable: number[] = [];
  for (let b = 33; b <= 126; b += 1) printable.push(b);
  for (let b = 161; b <= 172; b += 1) printable.push(b);
  for (let b = 174; b <= 255; b += 1) printable.push(b);
  const symbols: string[] = new Array(256);
  let extra = 0;
  for (let b = 0; b < 256; b += 1) {
    symbols[b] = printable.includes(b) ? String.fromCodePoint(b) : String.fromCodePoint(256 + extra++);
  }
  return symbols;
}

/** drowse's `byte_tokenizer(family)`: specials, then the 256 byte symbols sorted by code point.
 *  `decode` is the ByteLevel decoder on one id (UTF-8 with replacement, specials verbatim). */
export interface TinyTokenizer {
  readonly size: number;
  readonly bos: number | null;
  readonly eos: number;
  readonly specials: readonly string[];
  /** id → byte, or −1 for a special. */
  readonly byteOf: Int16Array;
  idOfByte(byte: number): number;
  encode(text: string): number[];
  decode(ids: readonly number[]): string;
}

/** `ChatFamily.specials` in id order: pad, bos, eos, the header opener. */
const FAMILY_SPECIALS: Readonly<Record<string, readonly string[]>> = {
  gemma3_text: ["<pad>", "<bos>", "<end_of_turn>", "<start_of_turn>"],
  qwen3: ["<|endoftext|>", "<|im_end|>", "<|im_start|>"],
};

export function tinyTokenizer(record: TinyArchRecord): TinyTokenizer {
  const symbols = byteSymbols();
  const order = Array.from({ length: 256 }, (_, b) => b).sort(
    (a, b) => symbols[a].codePointAt(0)! - symbols[b].codePointAt(0)!,
  );
  const specials = FAMILY_SPECIALS[record.family];
  if (specials === undefined || [...specials].sort().join() !== [...record.tokenizer.specials].sort().join()) {
    throw new Error(`no tiny tokenizer for ${record.family}`);
  }
  const byteOf = new Int16Array(specials.length + 256).fill(-1);
  const idOf = new Int32Array(256);
  order.forEach((byte, rank) => {
    byteOf[specials.length + rank] = byte;
    idOf[byte] = specials.length + rank;
  });
  const decoder = new TextDecoder("utf-8", { fatal: false });
  return {
    size: specials.length + 256,
    bos: record.tokenizer.bos,
    eos: record.tokenizer.eos,
    specials,
    byteOf,
    idOfByte: (byte) => idOf[byte],
    encode: (text) => [...new TextEncoder().encode(text)].map((byte) => idOf[byte]),
    decode(ids) {
      let text = "";
      let bytes: number[] = [];
      const flush = () => {
        if (bytes.length) text += decoder.decode(Uint8Array.from(bytes));
        bytes = [];
      };
      for (const id of ids) {
        if (byteOf[id] >= 0) bytes.push(byteOf[id]);
        else {
          flush();
          text += specials[id];
        }
      }
      flush();
      return text;
    },
  };
}
