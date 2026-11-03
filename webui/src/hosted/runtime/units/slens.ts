// The S readout (native `lens/footprint.py`, the lens instrument's `slens/` path; B1 §3.3,
// §5.1): footprint z of residuals, `slens/` probe rows and the `slens/` atom directions
// `K^{out}_L ẽ_v`, with native's refusals.
import { f32Tensor, type BlockSitesLensLayer } from "../modelBackend";
import { layerDepth, readS, requireLensMatrix, sWeights, type UnitsReadContext } from "./lensReads";

/** Native `InvalidTokenIdError` (400). */
export class InvalidTokenIdError extends Error {
  readonly code = "InvalidTokenIdError";
  readonly status = 400;
  readonly recoverable = true;

  constructor(message: string) {
    super(message);
    this.name = "InvalidTokenIdError";
  }
}

/** Native `LensNotFittedError` for an S use with no S lens (404). */
export function noInputLens(context: UnitsReadContext): Error {
  return Object.assign(new Error(context.noLens.S), { code: "LensNotFittedError", status: 404, recoverable: true });
}

/** Refuse a token the S mask drops (`_input_token`). */
export async function requireKeptToken(context: UnitsReadContext, word: string, tokenId: number): Promise<void> {
  const embeddings = await context.embeddings();
  if (!embeddings.keep[tokenId]) {
    throw new InvalidTokenIdError(
      `slens/${word} names token ${tokenId}, which the S read's token mask drops (special, added and byte-fallback pieces, embedding padding rows and tokens too rare in the counted text never rank in an S read; mask: ${embeddings.mask.source})`,
    );
  }
}

/** The S:out lens layers, refusing when there is no S lens (or no `out` site). */
export async function sOutLayers(context: UnitsReadContext): Promise<{ layers: number[]; matrices: BlockSitesLensLayer[] }> {
  const lens = context.s;
  if (lens === null) throw noInputLens(context);
  if (!lens.sites.includes("out")) requireLensMatrix(lens, "out", -1);
  const layers = [...lens.layers("out")];
  return { layers, matrices: await Promise.all(layers.map((layer) => lens.matrix("out", layer))) };
}

/** `slens/<word>`'s profile: `{L: K^{out}_L ẽ_v}` over every S:out layer (`register_input_direction`). */
export async function slensAtomDirections(
  context: UnitsReadContext,
  word: string,
  tokenId: number,
): Promise<Map<number, Float32Array>> {
  const { layers, matrices } = await sOutLayers(context);
  await requireKeptToken(context, word, tokenId);
  const embeddings = await context.embeddings();
  const directions = await context.backend.footprintDirections(matrices, Int32Array.of(tokenId), embeddings.resident);
  const d = context.hiddenSize;
  return new Map(layers.map((layer, c) => [layer, directions.data.slice(c * d, (c + 1) * d)]));
}

/** `ẽ_v = s_E·E[v] − mean_E` rows for `ids` `[n, d]` (an identity factor through the device). */
export async function centredEmbeddings(context: UnitsReadContext, ids: Int32Array): Promise<Float32Array> {
  const d = context.hiddenSize;
  const embeddings = await context.embeddings();
  const identity = new Float32Array(d * d);
  for (let i = 0; i < d; i += 1) identity[i * d + i] = 1;
  const out = await context.backend.vocabRowsTimes({ kind: "embed", ids, factor: f32Tensor(identity, [d, d]), statistics: embeddings.resident });
  return out.data;
}

/** Native `footprint_probe_rows`: `z [L, P]` per-layer footprint z, `depths`/`weights` `[L]` →
 *  `[3 + L, P]`: `Σ w_L z_L`, the depth center and spread under mass `w_L·max(z_L, 0)`, every `z_L`. */
export function footprintProbeRows(z: Float32Array, layers: number, depths: ArrayLike<number>, weights: ArrayLike<number>): Float32Array {
  const P = z.length / layers;
  const out = new Float32Array((3 + layers) * P);
  for (let p = 0; p < P; p += 1) {
    let sum = 0;
    let total = 0;
    let center = 0;
    for (let l = 0; l < layers; l += 1) {
      const weighted = Math.fround(weights[l] * z[l * P + p]);
      sum += weighted;
      const mass = Math.max(weighted, 0);
      total += mass;
      center += mass * depths[l];
    }
    total = Math.max(total, 1e-12);
    center /= total;
    let variance = 0;
    for (let l = 0; l < layers; l += 1) {
      const mass = Math.max(Math.fround(weights[l] * z[l * P + p]), 0);
      variance += mass * (depths[l] - center) ** 2;
    }
    out[p] = sum;
    out[P + p] = center;
    out[2 * P + p] = Math.sqrt(Math.max(variance / total, 0));
    for (let l = 0; l < layers; l += 1) out[(3 + l) * P + p] = z[l * P + p];
  }
  return out;
}

/** `slens/` probe rows for residuals `hidden [L, d]` (every layer's `out`) and probe tokens:
 *  per probe `[3 + c, 1]` over the S:out layers (`drowse_slens_probes` + `footprint_probe_rows`). */
export async function slensProbeRows(
  context: UnitsReadContext,
  hidden: Float32Array,
  tokenIds: Int32Array,
): Promise<{ layers: number[]; depths: number[]; weights: number[]; rows: Float32Array[] }> {
  const { layers, matrices } = await sOutLayers(context);
  const embeddings = await context.embeddings();
  const means = context.means.out;
  if (means === null) throw new Error("the slens/ probes need the neutral out means");
  const eTilde = await centredEmbeddings(context, tokenIds);
  const d = context.hiddenSize;
  const z = await context.backend.slensProbes({
    hidden: f32Tensor(hidden, [context.nLayers, d]),
    meansOut: f32Tensor(means, [context.nLayers, d]),
    lens: matrices,
    eTilde: f32Tensor(eTilde, [tokenIds.length, d]),
    statistics: embeddings.resident,
  });
  const depths = layers.map((layer) => Math.fround(layerDepth(layer, context.nLayers)));
  const weights = sWeights(layers, context.nLayers, context.band).map(Math.fround);
  const P = tokenIds.length;
  const rows = Array.from({ length: P }, (_, p) => {
    const column = Float32Array.from({ length: layers.length }, (_, l) => z.data[l * P + p]);
    return footprintProbeRows(column, layers.length, depths, weights);
  });
  return { layers, depths, weights, rows };
}

/** Footprint z top-k of residuals `[n, d]` at an S:out layer, centred on the neutral out mean. */
export async function footprintTop(
  context: UnitsReadContext,
  residuals: Float32Array,
  layer: number,
  k: number,
): Promise<{ ids: Int32Array; z: Float32Array; width: number }> {
  const lens = context.s;
  if (lens === null) throw noInputLens(context);
  requireLensMatrix(lens, "out", layer);
  const d = context.hiddenSize;
  const means = context.means.out!;
  const x = Float32Array.from(residuals, (value, i) => value - means[layer * d + (i % d)]);
  const result = await readS(context, x, await lens.matrix("out", layer), k);
  return { ids: result.ids, z: result.values, width: result.width };
}
