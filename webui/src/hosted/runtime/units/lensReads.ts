// The four-lens read (native `drowse/lens/read.py`, `depth.py`): one R half (probabilities
// through `J`, the logit lens at the last block's `out`) and one S half (footprint z through
// `K`, residuals centred on the neutral mean at their site) per row, each with the depth
// rule's weight and its lens or the reason it is absent. Device work goes through the
// block-sites backend; the host assembles the records as native's `to_dict` writes them.
import {
  f32Tensor,
  type BlockSitesEmbeddingStatistics,
  type BlockSitesLensLayer,
  type BlockSitesResident,
  type BrowserBlockSitesBackend,
  type LensSite,
} from "../modelBackend";

export type LensFamilyName = "R" | "S";
export type ReadOf = "residual" | "direction";

// ------------------------------------------------------------------- depth rule ---

/** The S→R crossover band (native `LensBand`, default 0.15–0.35 of depth). */
export interface LensBand {
  readonly lo: number;
  readonly hi: number;
}

export const DEFAULT_LENS_BAND: LensBand = Object.freeze({ lo: 0.15, hi: 0.35 });

export function sWeight(band: LensBand, depth: number): number {
  if (depth <= band.lo) return 1;
  if (depth >= band.hi) return 0;
  return (band.hi - depth) / (band.hi - band.lo);
}

export function layerDepth(layer: number, nLayers: number): number {
  return nLayers <= 1 ? 0 : layer / (nLayers - 1);
}

/** `(w_S, w_R)` for every layer. */
export function kindWeights(nLayers: number, band: LensBand = DEFAULT_LENS_BAND): [number, number][] {
  return Array.from({ length: nLayers }, (_, layer) => {
    const ws = sWeight(band, layerDepth(layer, nLayers));
    return [ws, 1 - ws];
  });
}

/** `w_S` at `layers`, normalized to sum 1 (all zeros when none has S weight). */
export function sWeights(layers: readonly number[], nLayers: number, band: LensBand = DEFAULT_LENS_BAND): number[] {
  const table = kindWeights(nLayers, band);
  const raw = layers.map((layer) => table[layer][0]);
  const total = raw.reduce((a, b) => a + b, 0);
  return total > 0 ? raw.map((w) => w / total) : raw.map(() => 0);
}

/** Python's `round(x, 6)` for record fields (half-even on the decimal; the goldens compare
 *  rounded fields within 2e-6, never by string). */
export function round6(x: number): number {
  return Math.round(x * 1e6) / 1e6;
}

/** Python's `str(list)` of ints, as native's messages spell a layer list. */
export function pyList(values: readonly number[]): string {
  return `[${values.join(", ")}]`;
}

// ------------------------------------------------------------------ the lenses ---

/** One selected lens (R or S) as the host sees it: what it holds, and its matrices made
 *  resident on first use (full-depth packs, fetched per layer). */
export interface UnitsLens {
  readonly family: LensFamilyName;
  /** The sites the lens holds. */
  readonly sites: readonly LensSite[];
  /** The fitted layers at `site`, ascending. */
  layers(site: LensSite): readonly number[];
  /** The matrix at `(site, layer)` on the device. */
  matrix(site: LensSite, layer: number): Promise<BlockSitesLensLayer>;
}

/** Native `LensNotFittedError` (404). */
export class LensNotFittedError extends Error {
  readonly code = "LensNotFittedError";
  readonly status = 404;
  readonly recoverable = true;

  constructor(message: string) {
    super(message);
    this.name = "LensNotFittedError";
  }
}

export function lensLabel(family: LensFamilyName, site: LensSite): string {
  return `${family}:${site}`;
}

/** `JacobianLens.matrix`'s refusals, verbatim. */
export function requireLensMatrix(lens: UnitsLens, site: LensSite, layer: number): void {
  if (!lens.sites.includes(site)) {
    const held = lens.sites.join(", ") || "none";
    throw new LensNotFittedError(
      `this ${lens.family} lens has no '${site}' site (it holds: ${held}); fit or fetch a ${lens.family} lens that carries ${site}`,
    );
  }
  const fitted = lens.layers(site);
  if (!fitted.includes(layer)) {
    throw new LensNotFittedError(
      `layer ${layer} not in fitted ${lens.family}:${site} lens layers ${pyList(fitted.slice(0, 3))}..${pyList(fitted.slice(-3))}`,
    );
  }
}

export function holds(lens: UnitsLens | null, site: LensSite, layer: number): lens is UnitsLens {
  return lens !== null && lens.sites.includes(site) && lens.layers(site).includes(layer);
}

export const LOGIT_LENS = "logit";

/** Everything a units read needs from the loaded model, its packs and its tokenizer. */
export interface UnitsReadContext {
  readonly backend: BrowserBlockSitesBackend;
  readonly modelId: string;
  readonly nLayers: number;
  readonly hiddenSize: number;
  readonly vocabSize: number;
  readonly band: LensBand;
  readonly r: UnitsLens | null;
  readonly s: UnitsLens | null;
  /** The absent texts when no lens of a family is selected (native's instrument wording). */
  readonly noLens: { readonly R: string; readonly S: string };
  /** The S embedding statistics, resident (loads with the S lens). */
  embeddings(): Promise<UnitsEmbeddings>;
  /** The neutral means a residual's S half centres on: `out` and `mid`, `[L, d]`. */
  readonly means: { readonly out: Float32Array | null; readonly mid: Float32Array | null };
  decode(tokenId: number): string;
}

export interface UnitsEmbeddings {
  readonly resident: BlockSitesResident<"embedding_statistics">;
  readonly statistics: BlockSitesEmbeddingStatistics;
  /** Kept ids, ascending. */
  readonly kept: Int32Array;
  readonly keep: Uint8Array;
  readonly mask: { readonly source: string; readonly detail: string; readonly kept: number };
}

/** The R transport at `(layer, site)` (native `output_transport`): `null` matrix and the
 *  `logit` label at the last block's `out`, else the lens's matrix. Throws as native. */
export async function outputTransport(
  context: UnitsReadContext,
  lens: UnitsLens,
  layer: number,
  site: LensSite,
): Promise<{ matrix: BlockSitesLensLayer | null; label: string }> {
  if (site === "out" && layer === context.nLayers - 1) {
    requireLensSite(lens, "out");
    return { matrix: null, label: LOGIT_LENS };
  }
  requireLensMatrix(lens, site, layer);
  return { matrix: await lens.matrix(site, layer), label: lensLabel(lens.family, site) };
}

function requireLensSite(lens: UnitsLens, site: LensSite): void {
  if (!lens.sites.includes(site)) requireLensMatrix(lens, site, -1);
}

// ------------------------------------------------------------------- the reads ---

export interface TokenScoreJSON {
  token_id: number;
  token: string;
  value: number;
}

export interface FamilyReadJSON {
  family: LensFamilyName;
  weight: number;
  unit: "probability" | "footprint_z";
  lens: string | null;
  centred: boolean;
  tokens: TokenScoreJSON[];
  absent: string | null;
}

export interface LensReadJSON {
  layer: number;
  site: LensSite;
  depth: number;
  of: ReadOf;
  r: FamilyReadJSON;
  s: FamilyReadJSON;
}

export interface VectorReadRequest {
  /** `[n, d]` rows. */
  readonly rows: Float32Array;
  readonly layer: number;
  readonly site: LensSite;
  readonly of: ReadOf;
}

/** Rows `[n, d]` times a resident lens matrix: `x @ Mᵀ` (`transpose`) or `x @ M`. */
export async function transport(
  context: UnitsReadContext,
  rows: Float32Array,
  matrix: BlockSitesLensLayer,
  transpose: boolean,
): Promise<Float32Array> {
  const n = rows.length / context.hiddenSize;
  const out = await context.backend.lensTransport(f32Tensor(rows, [1, n, context.hiddenSize]), [matrix], transpose);
  return out.data;
}

/** R top-k `(p, ids)` of rows `[n, d]` through `J` (`x @ Jᵀ`) or the identity. */
export async function readR(
  context: UnitsReadContext,
  rows: Float32Array,
  matrix: BlockSitesLensLayer | null,
  k: number,
): Promise<{ values: Float32Array; ids: Int32Array; width: number }> {
  const x = matrix === null ? rows : await transport(context, rows, matrix, true);
  const n = x.length / context.hiddenSize;
  const width = Math.min(k, context.vocabSize);
  const top = await context.backend.vocabReadoutTopK(f32Tensor(x, [n, context.hiddenSize]), width);
  const values = new Float32Array(n * width);
  for (let r = 0; r < n; r += 1) {
    for (let j = 0; j < width; j += 1) values[r * width + j] = Math.exp(top.logits.data[r * width + j] - top.logSumExp.data[r]);
  }
  return { values, ids: top.ids.data, width };
}

/** S top-k `(z, ids)` of rows `[n, d]` (already centred) through `K` (`u = x @ K`). */
export async function readS(
  context: UnitsReadContext,
  rows: Float32Array,
  matrix: BlockSitesLensLayer,
  k: number,
): Promise<{ values: Float32Array; ids: Int32Array; width: number }> {
  const u = await transport(context, rows, matrix, false);
  const n = u.length / context.hiddenSize;
  const embeddings = await context.embeddings();
  const top = await context.backend.footprintTopK(f32Tensor(u, [n, context.hiddenSize]), Math.min(k, embeddings.kept.length), embeddings.resident);
  return { values: top.z.data, ids: top.ids.data, width: top.ids.shape[1] };
}

/** Native `plan_reads` → `PendingReads.finish`: per request, one `LensReadJSON` per row. */
export async function readVectors(
  context: UnitsReadContext,
  requests: readonly VectorReadRequest[],
  k: number,
): Promise<LensReadJSON[][]> {
  const weights = kindWeights(context.nLayers, context.band);
  const d = context.hiddenSize;
  const out: LensReadJSON[][] = [];
  for (const request of requests) {
    const layer = request.layer;
    if (!(layer >= 0 && layer < context.nLayers)) {
      throw new RangeError(`layer ${layer} is outside the model's ${context.nLayers} layers`);
    }
    const n = request.rows.length / d;
    const [ws, wr] = weights[layer];
    const half = async (
      family: LensFamilyName,
      weight: number,
      unit: FamilyReadJSON["unit"],
      run: (matrix: BlockSitesLensLayer | null) => Promise<{ values: Float32Array; ids: Int32Array; width: number }>,
      centred: boolean,
    ): Promise<FamilyReadJSON[]> => {
      const lens = family === "R" ? context.r : context.s;
      const absent = (label: string | null, why: string): FamilyReadJSON[] =>
        Array.from({ length: n }, () => ({ family, weight: round6(weight), unit, lens: label, centred: false, tokens: [], absent: why }));
      if (lens === null) return absent(null, context.noLens[family]);
      let label = lensLabel(family, request.site);
      let matrix: BlockSitesLensLayer | null;
      try {
        if (family === "R") {
          ({ matrix, label } = await outputTransport(context, lens, layer, request.site));
        } else {
          requireLensMatrix(lens, request.site, layer);
          matrix = await lens.matrix(request.site, layer);
        }
      } catch (error) {
        if (error instanceof LensNotFittedError) return absent(label, error.message);
        throw error;
      }
      const result = await run(matrix);
      return Array.from({ length: n }, (_, row) => ({
        family,
        weight: round6(weight),
        unit,
        lens: label,
        centred,
        tokens: Array.from({ length: result.width }, (_, j) => {
          const id = result.ids[row * result.width + j];
          return { token_id: id, token: context.decode(id), value: round6(result.values[row * result.width + j]) };
        }),
        absent: null,
      }));
    };
    const r = await half("R", wr, "probability", (matrix) => readR(context, request.rows, matrix, k), false);
    const s = await half("S", ws, "footprint_z", async (matrix) => {
      let x = request.rows;
      if (request.of === "residual") {
        const means = context.means[request.site];
        if (means === null) throw new Error("a residual read needs the neutral means its S half centres on");
        x = Float32Array.from(x, (value, i) => value - means[layer * d + (i % d)]);
      }
      return readS(context, x, matrix!, k);
    }, request.of === "residual");
    const depth = round6(layerDepth(layer, context.nLayers));
    out.push(Array.from({ length: n }, (_, row) => ({
      layer,
      site: request.site,
      depth,
      of: request.of,
      r: r[row],
      s: s[row],
    })));
  }
  return out;
}
