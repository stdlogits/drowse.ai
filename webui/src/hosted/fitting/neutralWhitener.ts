import { decodeFp32Safetensors, encodeFp32Safetensors } from "../artifacts/safetensors";
import type { SerializedMahalanobisWhitener } from "./workerContracts";

/** The fitting kernel's whitener entry point (`drowse_fitting_wasm`). */
export interface NeutralWhitenerKernel {
  fitMahalanobisWhitener(
    rows: Float64Array,
    rowCount: number,
    columns: number,
    ridgeScale: number,
  ): {
    readonly rank: number;
    readonly ridge: number;
    mean(): Float64Array;
    basis(): Float64Array;
    eigenvalues(): Float64Array;
    inverseScales(): Float64Array;
    free(): void;
  };
}

export interface FittedNeutralWhitener {
  /** `neutral-whitener.safetensors`: per layer `mean`, `basis`, `eigenvalues`, `inverse_scales`. */
  readonly tensorBytes: Uint8Array;
  readonly ridgePerLayer: Record<string, number>;
  readonly whiteners: ReadonlyMap<number, SerializedMahalanobisWhitener>;
}

/**
 * The neutral rows `build_hosted_packs.py` writes (`neutral-rows.safetensors`, `layer_<L>`
 * `[N, D]` fp32): the activations drowse's own whitener is built from. One row-major matrix per
 * runtime layer, widened to the kernel's f64.
 */
export function decodeNeutralRows(
  bytes: Uint8Array,
  layerMap: readonly number[],
  columns: number,
): { rows: number; matrices: Float64Array[] } {
  const reader = decodeFp32Safetensors(bytes, "neutral-rows.safetensors");
  const expected = layerMap.map((layer) => `layer_${layer}`).sort();
  const keys = [...reader.keys].sort();
  if (keys.length !== expected.length || keys.some((key, index) => key !== expected[index])) {
    throw new Error("the neutral rows do not cover exactly the runtime's layers");
  }
  const rows = reader.description.shapes.get(`layer_${layerMap[0]}`)?.[0] ?? 0;
  if (rows < 2) throw new Error("the neutral rows hold fewer than two rows");
  const matrices = layerMap.map((layer) => Float64Array.from(reader.tensor(`layer_${layer}`, [rows, columns]).data));
  return { rows, matrices };
}

/**
 * The core pack's neutral whitener: the kernel's Mahalanobis fit (drowse's ridge rule, scale 1)
 * per layer over the neutral rows. The core builder runs it on the native rows, and the core
 * validator reruns it to check a pack's whitener is the one those rows give.
 */
export function fitNeutralWhitener(
  kernel: NeutralWhitenerKernel,
  matrices: readonly Float64Array[],
  rows: number,
  columns: number,
  layerMap: readonly number[],
): FittedNeutralWhitener {
  if (matrices.length !== layerMap.length) throw new Error("one neutral matrix per runtime layer is required");
  const tensors: Record<string, { shape: number[]; data: Float32Array }> = {};
  const ridges = new Map<string, number>();
  const whiteners = new Map<number, SerializedMahalanobisWhitener>();
  for (let slot = 0; slot < layerMap.length; slot += 1) {
    const layer = layerMap[slot];
    if (matrices[slot].length !== rows * columns) throw new Error(`layer ${layer}: the neutral matrix has the wrong size`);
    const fit = kernel.fitMahalanobisWhitener(matrices[slot], rows, columns, 1);
    try {
      const rank = fit.rank;
      const mean = fit.mean();
      const basis = fit.basis();
      const eigenvalues = fit.eigenvalues();
      const inverseScales = fit.inverseScales();
      tensors[`layer_${layer}.mean`] = { shape: [columns], data: Float32Array.from(mean) };
      tensors[`layer_${layer}.basis`] = { shape: [rank, columns], data: Float32Array.from(basis) };
      tensors[`layer_${layer}.eigenvalues`] = { shape: [rank], data: Float32Array.from(eigenvalues) };
      tensors[`layer_${layer}.inverse_scales`] = { shape: [rank], data: Float32Array.from(inverseScales) };
      ridges.set(String(layer), fit.ridge);
      whiteners.set(layer, { columns, rank, ridge: fit.ridge, mean, basis, eigenvalues, inverseScales });
    } finally {
      fit.free();
    }
  }
  return {
    tensorBytes: encodeFp32Safetensors(tensors, { path: "neutral-whitener.safetensors" }),
    ridgePerLayer: Object.fromEntries(ridges),
    whiteners,
  };
}
