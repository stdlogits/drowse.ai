// A dense fp32 transformer of the six eligible architectures (B1 §1.1), plain loops over
// Float32Arrays: the CPU reference the reference backend runs. Test-only; no GPU path uses it.
import type { BlockSpec } from "../structuredHookProfile";
import { isLocalAttentionLayer } from "../structuredHookProfile";

/** One block's weights, row-major `[out, in]` as HF stores them; norm gains with the offset
 *  folded in (`γ = o + w`). */
export interface DenseBlockWeights {
  readonly inputNorm: Float32Array;
  readonly q: Float32Array;
  readonly k: Float32Array;
  readonly v: Float32Array;
  readonly o: Float32Array;
  readonly qNorm: Float32Array;
  readonly kNorm: Float32Array;
  /** Gemma's `N_ao`; null on Qwen3. */
  readonly attnOutNorm: Float32Array | null;
  readonly mlpInNorm: Float32Array;
  /** Gemma's `N_mo`; null on Qwen3. */
  readonly mlpOutNorm: Float32Array | null;
  readonly gate: Float32Array;
  readonly up: Float32Array;
  readonly down: Float32Array;
}

export interface DenseModelWeights {
  readonly spec: BlockSpec;
  readonly layerCount: number;
  readonly hiddenSize: number;
  readonly vocabSize: number;
  /** `[V, d]`, tied: the unembedding too. */
  readonly embed: Float32Array;
  readonly finalNorm: Float32Array;
  readonly blocks: readonly DenseBlockWeights[];
}

/** `y [T, out] = x [T, in] · Wᵀ`, `W [out, in]`. */
export function linear(x: ArrayLike<number>, w: Float32Array, rows: number, inner: number, out: number): Float32Array {
  const y = new Float32Array(rows * out);
  for (let t = 0; t < rows; t += 1) {
    const xo = t * inner;
    for (let o = 0; o < out; o += 1) {
      const wo = o * inner;
      let s = 0;
      for (let i = 0; i < inner; i += 1) s += x[xo + i] * w[wo + i];
      y[t * out + o] = s;
    }
  }
  return y;
}

/** RMSNorm of each `width` row: `x · rsqrt(mean(x²) + ε) · γ`. */
export function rmsNorm(x: ArrayLike<number>, gain: Float32Array, width: number, eps: number): Float32Array {
  const rows = x.length / width;
  const y = new Float32Array(x.length);
  for (let t = 0; t < rows; t += 1) {
    let s = 0;
    for (let i = 0; i < width; i += 1) s += x[t * width + i] ** 2;
    const r = 1 / Math.sqrt(s / width + eps);
    for (let i = 0; i < width; i += 1) y[t * width + i] = x[t * width + i] * r * gain[i];
  }
  return y;
}

/** `rsqrt(mean(x²) + ε)` of one row. */
export function rmsScale(x: ArrayLike<number>, offset: number, width: number, eps: number): number {
  let s = 0;
  for (let i = 0; i < width; i += 1) s += x[offset + i] ** 2;
  return 1 / Math.sqrt(s / width + eps);
}

export function activation(kind: BlockSpec["activation"], x: number): number {
  if (kind === "silu") return x / (1 + Math.exp(-x));
  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
}

function rope(x: Float32Array, rows: number, heads: number, hd: number, positions: ArrayLike<number>, theta: number, scale: number): void {
  const half = hd / 2;
  const inv = new Float64Array(half);
  for (let i = 0; i < half; i += 1) inv[i] = scale / theta ** ((2 * i) / hd);
  for (let t = 0; t < rows; t += 1) {
    for (let i = 0; i < half; i += 1) {
      const angle = Math.fround(positions[t] * Math.fround(inv[i]));
      const c = Math.cos(angle);
      const s = Math.sin(angle);
      for (let h = 0; h < heads; h += 1) {
        const at = (t * heads + h) * hd;
        const a = x[at + i];
        const b = x[at + half + i];
        x[at + i] = a * c - b * s;
        x[at + half + i] = b * c + a * s;
      }
    }
  }
}

/** The hook points of one block forward (B1 §1.2); each may replace its tensor. */
export interface DenseBlockSites {
  /** `z [T, H·hd]` before `o_proj` (head ablation). */
  head?(layer: number, z: Float32Array, positions: Int32Array): Float32Array;
  /** `act [T, dFF]` before `down_proj` (neuron ablation). */
  neuron?(layer: number, act: Float32Array, positions: Int32Array): Float32Array;
  /** The block output before the v4 post-block hook → after it. */
  postBlock?(layer: number, outPre: Float32Array, positions: Int32Array): Float32Array;
  /** Every site, after its hooks, for capture. */
  capture?(layer: number, sites: DenseBlockCapture): void;
  /** Softmax rows `[H, T, ctx]` per layer (patterns), keyed by row. */
  attention?(layer: number, rows: Float32Array, positions: Int32Array, context: number): void;
}

export interface DenseBlockCapture {
  readonly blockIn: Float32Array;
  readonly z: Float32Array;
  readonly attnO: Float32Array;
  readonly mid: Float32Array;
  readonly act: Float32Array;
  readonly mlpO: Float32Array;
  readonly outPre: Float32Array;
  readonly out: Float32Array;
  /** Post-RoPE keys `[T, KV·hd]`, and queries `[T, H·hd]`, of this forward's rows. */
  readonly keys: Float32Array;
  readonly queries: Float32Array;
}

/** A KV-continuing sequence: keys (post-RoPE) and values per layer. */
export class DenseSequence {
  readonly keys: Float32Array[];
  readonly values: Float32Array[];
  length = 0;

  constructor(readonly model: DenseModelWeights, readonly capacity: number) {
    const kv = model.spec.nKvHeads * model.spec.headDim;
    this.keys = model.blocks.map(() => new Float32Array(capacity * kv));
    this.values = model.blocks.map(() => new Float32Array(capacity * kv));
  }

  /** Feed `ids` at the next positions; returns every row's final hidden state (before the
   *  final norm) `[T, d]`. */
  forward(ids: ArrayLike<number>, sites: DenseBlockSites = {}): Float32Array {
    const model = this.model;
    const { spec } = model;
    const d = model.hiddenSize;
    const T = ids.length;
    if (this.length + T > this.capacity) throw new RangeError("The sequence is past its capacity");
    const positions = Int32Array.from({ length: T }, (_, t) => this.length + t);
    let h: Float32Array = new Float32Array(T * d);
    for (let t = 0; t < T; t += 1) {
      for (let i = 0; i < d; i += 1) h[t * d + i] = model.embed[ids[t] * d + i] * spec.embedScale;
    }
    const H = spec.nHeads;
    const KV = spec.nKvHeads;
    const hd = spec.headDim;
    const group = H / KV;
    model.blocks.forEach((block, layer) => {
      const blockIn = h;
      const a = rmsNorm(h, block.inputNorm, d, spec.rmsEps);
      const q = rmsNorm(linear(a, block.q, T, d, H * hd), block.qNorm, hd, spec.rmsEps);
      const k = rmsNorm(linear(a, block.k, T, d, KV * hd), block.kNorm, hd, spec.rmsEps);
      const v = linear(a, block.v, T, d, KV * hd);
      const local = isLocalAttentionLayer(spec, layer);
      const theta = local && spec.ropeLocalTheta !== null ? spec.ropeLocalTheta : spec.ropeTheta;
      const scale = local ? 1 : spec.ropeGlobalScale;
      rope(q, T, H, hd, positions, theta, scale);
      rope(k, T, KV, hd, positions, theta, scale);
      this.keys[layer].set(k, this.length * KV * hd);
      this.values[layer].set(v, this.length * KV * hd);
      const context = this.length + T;
      const keys = this.keys[layer];
      const values = this.values[layer];
      let z: Float32Array = new Float32Array(T * H * hd);
      const probs = sites.attention ? new Float32Array(H * T * context) : null;
      const scores = new Float64Array(context);
      for (let t = 0; t < T; t += 1) {
        const p = positions[t];
        const lo = local && spec.slidingWindow !== null ? Math.max(0, p - spec.slidingWindow + 1) : 0;
        for (let head = 0; head < H; head += 1) {
          const g = Math.floor(head / group);
          let max = -Infinity;
          for (let j = lo; j <= p; j += 1) {
            let s = 0;
            for (let i = 0; i < hd; i += 1) s += q[(t * H + head) * hd + i] * keys[(j * KV + g) * hd + i];
            s *= spec.attnScaling;
            if (spec.attnLogitSoftcap !== null) s = Math.tanh(s / spec.attnLogitSoftcap) * spec.attnLogitSoftcap;
            scores[j] = s;
            if (s > max) max = s;
          }
          let total = 0;
          for (let j = lo; j <= p; j += 1) {
            scores[j] = Math.exp(scores[j] - max);
            total += scores[j];
          }
          for (let j = lo; j <= p; j += 1) {
            const w = scores[j] / total;
            if (probs) probs[(head * T + t) * context + j] = w;
            for (let i = 0; i < hd; i += 1) z[(t * H + head) * hd + i] += w * values[(j * KV + g) * hd + i];
          }
        }
      }
      if (probs) sites.attention!(layer, probs, positions, context);
      if (sites.head) z = sites.head(layer, z, positions);
      const attnO = linear(z, block.o, T, H * hd, d);
      const attnWrite = block.attnOutNorm ? rmsNorm(attnO, block.attnOutNorm, d, spec.rmsEps) : attnO;
      const mid = new Float32Array(T * d);
      for (let i = 0; i < T * d; i += 1) mid[i] = h[i] + attnWrite[i];
      const m = rmsNorm(mid, block.mlpInNorm, d, spec.rmsEps);
      const gate = linear(m, block.gate, T, d, spec.dFF);
      const up = linear(m, block.up, T, d, spec.dFF);
      let act: Float32Array = new Float32Array(T * spec.dFF);
      for (let i = 0; i < act.length; i += 1) act[i] = activation(spec.activation, gate[i]) * up[i];
      if (sites.neuron) act = sites.neuron(layer, act, positions);
      const mlpO = linear(act, block.down, T, spec.dFF, d);
      const mlpWrite = block.mlpOutNorm ? rmsNorm(mlpO, block.mlpOutNorm, d, spec.rmsEps) : mlpO;
      const outPre = new Float32Array(T * d);
      for (let i = 0; i < T * d; i += 1) outPre[i] = mid[i] + mlpWrite[i];
      const out = sites.postBlock ? sites.postBlock(layer, outPre, positions) : outPre;
      sites.capture?.(layer, { blockIn, z, attnO, mid, act, mlpO, outPre, out, keys: k, queries: q });
      h = out;
    });
    this.length += T;
    return h;
  }
}

/** Logits `[V]` of one final hidden row: `W_U · N_f(h)` (no softcap on these models). */
export function finalLogits(model: DenseModelWeights, hidden: ArrayLike<number>, offset = 0): Float32Array {
  const d = model.hiddenSize;
  const row = rmsNorm(Float32Array.prototype.slice.call(hidden, offset, offset + d) as Float32Array, model.finalNorm, d, model.spec.rmsEps);
  const logits = linear(row, model.embed, 1, d, model.vocabSize);
  if (model.spec.finalLogitSoftcap !== null) {
    const cap = model.spec.finalLogitSoftcap;
    for (let v = 0; v < logits.length; v += 1) logits[v] = Math.tanh(logits[v] / cap) * cap;
  }
  return logits;
}
