// Unit statistics and reading scales (native `drowse/units/stats.py`): the neutral
// statistics B6's core pack carries, and the soft floored scale every unit reading divides by.
import type { BlockSpec } from "../structuredHookProfile";

/** One model's neutral unit statistics (`UnitStats`, fp32), plus the neutral `out` and `mid`
 *  means (the S centring at each site, `[L, d]`). */
export interface UnitStatistics {
  readonly documents: number;
  /** `[L, dFF]`. */
  readonly neuronMean: Float32Array;
  /** `[L, 2, dFF]`: row 0 the `+` pole's `act_top`, row 1 the `−` pole's. */
  readonly neuronScale: Float32Array;
  /** `[L, H, hd]`. */
  readonly headMean: Float32Array;
  /** `[L, H]`. */
  readonly headScale: Float32Array;
}

export const SCALE_FLOOR_SHARE = 0.25;

/** The reading scales (`UnitScales`): source, detail and per layer the `act_top` tables and
 *  the soft scales a reading divides by (`neuron [L, 2, dFF]`, `head [L, H]`). */
export interface UnitScales {
  readonly source: "neutral" | "catalog";
  readonly detail: string;
  readonly neuronTop: Float32Array;
  readonly headTop: Float32Array;
  readonly neuron: Float32Array;
  readonly head: Float32Array;
  readonly typical: {
    readonly neuron: Float32Array;
    readonly head: Float32Array;
    readonly modelNeuron: number;
    readonly modelHead: number;
  };
}

/** torch's `median()` of the positive entries: the lower middle for an even count; 0 when none
 *  is positive. */
export function typicalScale(values: ArrayLike<number>): number {
  const positive: number[] = [];
  for (let i = 0; i < values.length; i += 1) if (values[i] > 0) positive.push(values[i]);
  if (positive.length === 0) return 0;
  positive.sort((a, b) => a - b);
  return positive[Math.floor((positive.length - 1) / 2)];
}

/** `sqrt(s² + (c·m)²)` in fp32. */
export function softScale(scale: number, base: number): number {
  const floor = Math.fround(SCALE_FLOOR_SHARE * base);
  return Math.fround(Math.sqrt(Math.fround(Math.fround(scale * scale) + Math.fround(floor * floor))));
}

/** Native `reading_scales` over the neutral statistics: each layer's floor base is the larger of
 *  its typical scale and the model's. */
export function readingScales(
  stats: UnitStatistics,
  spec: Pick<BlockSpec, "dFF" | "nHeads">,
  layers: number,
  source: UnitScales["source"] = "neutral",
  detail = `${stats.documents} neutral statements`,
): UnitScales {
  const F = spec.dFF;
  const H = spec.nHeads;
  const modelNeuron = source === "neutral" ? typicalScale(stats.neuronScale) : 0;
  const modelHead = source === "neutral" ? typicalScale(stats.headScale) : 0;
  const neuron = new Float32Array(stats.neuronScale.length);
  const head = new Float32Array(stats.headScale.length);
  const typicalNeuron = new Float32Array(layers);
  const typicalHead = new Float32Array(layers);
  for (let layer = 0; layer < layers; layer += 1) {
    const n = stats.neuronScale.subarray(layer * 2 * F, (layer + 1) * 2 * F);
    const h = stats.headScale.subarray(layer * H, (layer + 1) * H);
    typicalNeuron[layer] = typicalScale(n);
    typicalHead[layer] = typicalScale(h);
    const nBase = Math.max(typicalNeuron[layer], modelNeuron);
    const hBase = Math.max(typicalHead[layer], modelHead);
    for (let i = 0; i < 2 * F; i += 1) neuron[layer * 2 * F + i] = softScale(n[i], nBase);
    for (let i = 0; i < H; i += 1) head[layer * H + i] = softScale(h[i], hBase);
  }
  return {
    source,
    detail,
    neuronTop: stats.neuronScale,
    headTop: stats.headScale,
    neuron,
    head,
    typical: { neuron: typicalNeuron, head: typicalHead, modelNeuron, modelHead },
  };
}

/** The activity kernel's scale table `[L, 2·dFF + H]` (`unitScalesLayout`: `S+ | S− | S_head`),
 *  floored at 1e-6 (native `clamp_min`). */
export function activityScaleTable(scales: UnitScales, spec: Pick<BlockSpec, "dFF" | "nHeads">, layers: number): Float32Array {
  const F = spec.dFF;
  const H = spec.nHeads;
  const out = new Float32Array(layers * (2 * F + H));
  for (let layer = 0; layer < layers; layer += 1) {
    for (let i = 0; i < 2 * F; i += 1) out[layer * (2 * F + H) + i] = Math.max(scales.neuron[layer * 2 * F + i], 1e-6);
    for (let h = 0; h < H; h += 1) out[layer * (2 * F + H) + 2 * F + h] = Math.max(scales.head[layer * H + h], 1e-6);
  }
  return out;
}
