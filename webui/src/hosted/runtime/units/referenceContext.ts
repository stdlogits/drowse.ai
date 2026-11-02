// A units read context over host-held lens matrices and statistics (the reference backend's
// tests, the fixture runtimes): matrices upload on first use, as B6's per-layer packs do.
import { f32Tensor, u32Tensor, type BlockSitesLensLayer, type BrowserBlockSitesBackend, type LensSite } from "../modelBackend";
import type { DenseModelWeights } from "./denseModel";
import type { LensBand, LensFamilyName, UnitsEmbeddings, UnitsLens, UnitsReadContext } from "./lensReads";

/** The absent text for a lens family with no lens installed. Native's names CLI commands
 *  (`drowse lens fetch …`); the browser names the pack to install, in its own copy rules. */
export function noLensText(family: LensFamilyName, modelId: string): string {
  return family === "S"
    ? `no S-lens is installed for ${modelId} (install the model's S lens pack)`
    : `no R-family lens is selected for ${modelId} (install the model's R lens pack)`;
}

export function hostLens(
  backend: BrowserBlockSitesBackend,
  family: LensFamilyName,
  tables: ReadonlyMap<LensSite, ReadonlyMap<number, Float32Array>>,
  d: number,
): UnitsLens {
  const resident = new Map<string, Promise<BlockSitesLensLayer>>();
  const sites = (["out", "mid"] as const).filter((site) => tables.has(site));
  return {
    family,
    sites,
    layers: (site) => [...(tables.get(site)?.keys() ?? [])].sort((a, b) => a - b),
    matrix(site, layer) {
      const key = `${site}:${layer}`;
      let found = resident.get(key);
      if (!found) {
        const matrix = tables.get(site)?.get(layer);
        if (!matrix) throw new Error(`no ${family}:${site} matrix at L${layer}`);
        found = backend.uploadLensLayer(family === "R" ? "r" : "s", site, layer, f32Tensor(matrix, [d, d]));
        resident.set(key, found);
      }
      return found;
    },
  };
}

/** The S embedding statistics over `keep` (native `TokenEmbeddings.build`, fp32 accumulated in
 *  double): `mean_E = s_E · mean_{kept} E[v]`, `Σ_E = mean_{kept} ẽ ẽᵀ`. */
export function embeddingStatistics(model: DenseModelWeights, keep: Uint8Array): { meanE: Float32Array; sigmaE: Float32Array; mask: Uint32Array } {
  const d = model.hiddenSize;
  const s = model.spec.embedScale;
  const kept = [...keep.keys()].filter((v) => keep[v]);
  const total = new Float64Array(d);
  for (const v of kept) for (let i = 0; i < d; i += 1) total[i] += model.embed[v * d + i];
  const meanE = Float32Array.from(total, (x) => (x * s) / kept.length);
  const cov = new Float64Array(d * d);
  const e = new Float64Array(d);
  for (const v of kept) {
    for (let i = 0; i < d; i += 1) e[i] = model.embed[v * d + i] * s - meanE[i];
    for (let i = 0; i < d; i += 1) for (let j = 0; j < d; j += 1) cov[i * d + j] += e[i] * e[j];
  }
  const mask = new Uint32Array(Math.ceil(keep.length / 32));
  for (const v of kept) mask[v >>> 5] |= 1 << (v & 31);
  return { meanE, sigmaE: Float32Array.from(cov, (x) => x / kept.length), mask };
}

export interface ReferenceContextOptions {
  readonly backend: BrowserBlockSitesBackend;
  readonly model: DenseModelWeights;
  readonly modelId: string;
  readonly band: LensBand;
  /** `"R:out"` → layer → `[d, d]`; a family with no entry is not selected. */
  readonly lenses: ReadonlyMap<string, ReadonlyMap<number, Float32Array>>;
  readonly keep: Uint8Array;
  readonly mask: { readonly source: string; readonly detail: string };
  readonly means: { readonly out: Float32Array | null; readonly mid: Float32Array | null };
  readonly decode: (tokenId: number) => string;
  /** Override the statistics (a pack's), else computed from `keep`. */
  readonly statistics?: { meanE: Float32Array; sigmaE: Float32Array; mask: Uint32Array };
}

export function referenceReadContext(options: ReferenceContextOptions): UnitsReadContext {
  const d = options.model.hiddenSize;
  const family = (name: LensFamilyName): UnitsLens | null => {
    const tables = new Map<LensSite, ReadonlyMap<number, Float32Array>>();
    for (const site of ["out", "mid"] as const) {
      const table = options.lenses.get(`${name}:${site}`);
      if (table) tables.set(site, table);
    }
    return tables.size ? hostLens(options.backend, name, tables, d) : null;
  };
  let embeddings: Promise<UnitsEmbeddings> | null = null;
  return {
    backend: options.backend,
    modelId: options.modelId,
    nLayers: options.model.layerCount,
    hiddenSize: d,
    vocabSize: options.model.vocabSize,
    band: options.band,
    r: family("R"),
    s: family("S"),
    noLens: { R: noLensText("R", options.modelId), S: noLensText("S", options.modelId) },
    means: options.means,
    tokenTexts: async (ids) => ids.map((id) => options.decode(id)),
    embeddings() {
      embeddings ??= (async () => {
        const stats = options.statistics ?? embeddingStatistics(options.model, options.keep);
        const statistics = {
          meanE: f32Tensor(stats.meanE, [d]),
          sigmaE: f32Tensor(stats.sigmaE, [d, d]),
          mask: u32Tensor(stats.mask, [stats.mask.length]),
        };
        const kept = Int32Array.from([...options.keep.keys()].filter((v) => options.keep[v]));
        return {
          resident: await options.backend.uploadEmbeddingStatistics(statistics),
          statistics,
          kept,
          keep: options.keep,
          mask: { ...options.mask, kept: kept.length },
        };
      })();
      return embeddings;
    },
  };
}
