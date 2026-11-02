// Static unit reads (native `drowse/units/reads.py::UnitReader`): what a neuron or a head is
// about, from its weights, through the selected R and S lenses (B1 §4.1, §4.2). Weight and
// vocabulary work runs on the block-sites backend; everything averaged over the vocabulary or
// the neutral corpus comes from the offline artifacts (`population_artifacts`' keys).
import type { SerializedMahalanobisWhitener } from "../../fitting/workerContracts";
import { f32Tensor, i32Tensor, type BlockSitesLensLayer } from "../modelBackend";
import type { BlockSpec } from "../structuredHookProfile";
import { LayerSelectorError, resolveLayerSelector, type LayerSelector } from "../steeringExpression";
import { copyScore, solve } from "./linalg";
import {
  holds,
  kindWeights,
  layerDepth,
  LOGIT_LENS,
  lensLabel,
  outputTransport,
  readVectors,
  round6,
  type LensFamilyName,
  type LensReadJSON,
  type UnitsReadContext,
} from "./lensReads";
import type { UnitScales, UnitStatistics } from "./scales";

export const MAX_UNIT_K = 256;
const R_STD_FLOOR = 1e-4;
const S_STD_FLOOR = 1e-6;
const HEAD_STD_FLOOR = 1e-6;
const VOCAB_CHUNK = 8192;
const HEAD_ROLES = ["query", "key", "value"] as const;

/** The neuron-lens catalogs native knows; no hosted model has one (PLAN, scope facts). */
export const CATALOGUED_MODELS = "Qwen/Qwen3-8B, google/gemma-4-12B";
export const CATALOG_REPO = "stdlogits/neuron-lens";
export const CATALOG_REVISION = "b5bf11a2b9e35c9100a0fa0995ff7a4e57f5b777";
export const NO_CATALOG = `no neuron-lens catalog for this model (catalogued: ${CATALOGUED_MODELS})`;

/** `units/catalog`: native `catalog_status` for an uncatalogued model (disk only). */
export function unitCatalogStatus(modelId: string): Record<string, unknown> {
  return {
    model_id: modelId,
    available: false,
    bound: false,
    slug: null,
    repo: CATALOG_REPO,
    revision: CATALOG_REVISION,
    credit: null,
    reason: NO_CATALOG,
    binding: null,
  };
}

/** Native `RequestError` (400) with its text verbatim. */
export class UnitRequestError extends Error {
  readonly status = 400;
  readonly recoverable = true;

  constructor(readonly code: string, message: string) {
    super(message);
    this.name = code;
  }
}

/** Native `WhitenerError` (422-shaped refusal of a decomposition without the metric). */
export class UnitWhitenerError extends Error {
  readonly code = "WhitenerError";
  readonly status = 422;
  readonly recoverable = false;

  constructor(message: string) {
    super(message);
    this.name = "WhitenerError";
  }
}

// ---------------------------------------------------------------------- inputs ---

/** One layer's offline artifacts, keyed as native `UnitReader.population_artifacts` keys them. */
export type UnitLayerArtifacts = ReadonlyMap<string, Float32Array>;

/** What the reader needs besides the device: B6's units and core packs (or, in tests, drowse's
 *  unit-stats / unit-artifacts goldens and the fixture whitener). */
export interface UnitsArtifacts {
  readonly statistics: UnitStatistics | null;
  /** Why `statistics` is null (native `UnitsUnavailableError`'s text). */
  readonly statisticsAbsent: string | null;
  readonly scales: UnitScales | null;
  layer(layer: number): Promise<UnitLayerArtifacts>;
  /** The null basis's projector `N Nᵀ` `[d, d]`. */
  nullProjector(): Promise<Float32Array>;
  /** The head-read vocabulary (`head_kept_ids`). */
  headKeptIds(): Promise<Int32Array>;
  /** The neutral whitener at `layer` (eigen form), or null. */
  whitener(layer: number): Promise<SerializedMahalanobisWhitener | null>;
}

export interface UnitRef {
  readonly kind: "neuron" | "head";
  readonly layer: number;
  readonly index: number;
}

export function unitName(ref: UnitRef): string {
  return ref.kind === "neuron" ? `neuron/L${ref.layer}n${ref.index}` : `head/L${ref.layer}h${ref.index}`;
}

const UNIT_RE = /^[Ll]([0-9]+)([NnHh])([0-9]+)$/;

/** Native `parse_unit_ref` + `runtime.check`: the text and range refusals (400). */
export function parseUnit(unit: string, spec: Pick<BlockSpec, "dFF" | "nHeads">, nLayers: number): UnitRef {
  const slash = unit.indexOf("/");
  const namespace = slash < 0 ? unit : unit.slice(0, slash);
  const name = slash < 0 ? "" : unit.slice(slash + 1);
  if (slash < 0 || (namespace !== "neuron" && namespace !== "head")) {
    throw new UnitRequestError("RequestError", `a unit is neuron/L<layer>n<index> or head/L<layer>h<index>, got ${pyRepr(unit)}`);
  }
  const letter = namespace === "neuron" ? "n" : "h";
  const match = UNIT_RE.exec(name);
  if (match === null || match[2].toLowerCase() !== letter) {
    const example = namespace === "neuron" ? "L8n344" : "L20h17";
    throw new UnitRequestError(
      "UnitRefError",
      `'${namespace}/${name}' is not a ${namespace} reference; write L<layer>${letter}<index> (e.g. '${namespace}/${example}')`,
    );
  }
  const ref: UnitRef = { kind: namespace, layer: Number(match[1]), index: Number(match[3]) };
  return checkUnit(ref, spec, nLayers);
}

export function checkUnit(ref: UnitRef, spec: Pick<BlockSpec, "dFF" | "nHeads">, nLayers: number): UnitRef {
  if (!(ref.layer >= 0 && ref.layer < nLayers)) {
    throw new UnitRequestError("UnitIndexError", `layer ${ref.layer} is outside the model (${nLayers} blocks, 0..${nLayers - 1})`);
  }
  const count = ref.kind === "neuron" ? spec.dFF : spec.nHeads;
  if (!(ref.index >= 0 && ref.index < count)) {
    throw new UnitRequestError("UnitIndexError", `${ref.kind} ${ref.index} is outside block ${ref.layer} (${count} ${ref.kind}s, 0..${count - 1})`);
  }
  return ref;
}

/** Python's `repr` of a str (single quotes unless it holds one and no double quote). */
/** A layer-selector refusal as the units reads raise it (`RequestError`, 400). */
export function layerRequest<T>(resolve: () => T): T {
  try {
    return resolve();
  } catch (error) {
    if (error instanceof LayerSelectorError) throw new UnitRequestError("RequestError", error.message);
    throw error;
  }
}

export function pyRepr(text: string): string {
  const quote = text.includes("'") && !text.includes('"') ? '"' : "'";
  const body = [...text].map((ch) => {
    if (ch === "\\") return "\\\\";
    if (ch === quote) return `\\${ch}`;
    if (ch === "\n") return "\\n";
    if (ch === "\r") return "\\r";
    if (ch === "\t") return "\\t";
    const code = ch.codePointAt(0)!;
    if (code < 0x20 || code === 0x7f) return `\\x${code.toString(16).padStart(2, "0")}`;
    return ch;
  }).join("");
  return `${quote}${body}${quote}`;
}

// ------------------------------------------------------------- the records ---

export interface UnitHalfJSON {
  family: LensFamilyName;
  weight: number;
  unit: "probability" | "footprint_z" | "layer_z" | "own_token_z" | "energy_z";
  lens: string | null;
  tokens: { token_id: number; token: string; value: number }[];
  absent: string | null;
}

export interface PlaceReadJSON {
  layer: number;
  site: "out" | "mid";
  depth: number;
  hubs: "shown" | "hidden";
  s: UnitHalfJSON;
  r: UnitHalfJSON;
}

function half(read: LensReadJSON["r"]): UnitHalfJSON {
  return { family: read.family, weight: read.weight, unit: read.unit, lens: read.lens, tokens: read.tokens, absent: read.absent };
}

function place(read: LensReadJSON): PlaceReadJSON {
  return { layer: read.layer, site: read.site, depth: read.depth, hubs: "shown", s: half(read.s), r: half(read.r) };
}

const num = (value: number | null | undefined): number | null =>
  value === null || value === undefined || Number.isNaN(value) ? null : round6(value);

// ------------------------------------------------------------------- reader ---

export interface UnitReaderOptions {
  readonly context: UnitsReadContext;
  readonly artifacts: UnitsArtifacts;
  readonly spec: BlockSpec;
  /** Resolve a steerable selector to its per-layer directions (`find`'s selector target). */
  readonly resolveSelector?: (selector: string) => Promise<ReadonlyMap<number, Float32Array>>;
  /** A single-token word (or `#<id>`) to its id, for `top` (S family, the input embedding). */
  readonly resolveToken: (word: string) => number;
}

export class UnitReader {
  private readonly context: UnitsReadContext;
  private readonly artifacts: UnitsArtifacts;
  private readonly spec: BlockSpec;
  private readonly d: number;
  private readonly nLayers: number;

  constructor(private readonly options: UnitReaderOptions) {
    this.context = options.context;
    this.artifacts = options.artifacts;
    this.spec = options.spec;
    this.d = options.context.hiddenSize;
    this.nLayers = options.context.nLayers;
  }

  // ---------------------------------------------------------------- plumbing
  private checkK(k: number): number {
    if (!(Number.isSafeInteger(k) && k >= 1 && k <= MAX_UNIT_K)) {
      throw new UnitRequestError("RequestError", `k must be in [1, ${MAX_UNIT_K}]`);
    }
    return k;
  }

  private weights(layer: number): [number, number] {
    return kindWeights(this.nLayers, this.context.band)[layer];
  }

  private noLens(family: LensFamilyName, layer: number, site: string): string {
    const lens = family === "R" ? this.context.r : this.context.s;
    if (lens === null) return `no ${family}-family lens is selected for ${this.context.modelId}`;
    return `the ${family} lens holds no ${site} matrix at L${layer}`;
  }

  private async sMatrix(layer: number, site: "out" | "mid"): Promise<BlockSitesLensLayer | null> {
    return holds(this.context.s, site, layer) ? this.context.s.matrix(site, layer) : null;
  }

  private async rMatrix(layer: number, site: "out" | "mid"): Promise<BlockSitesLensLayer | null> {
    return holds(this.context.r, site, layer) ? this.context.r.matrix(site, layer) : null;
  }

  /** `_r_matrix`: the R transport and label, `(null, null)` without one. */
  private async rTransport(layer: number, site: "out" | "mid"): Promise<{ matrix: BlockSitesLensLayer | null; label: string | null }> {
    const lens = this.context.r;
    if (lens === null || !lens.sites.includes(site)) return { matrix: null, label: null };
    if (!(site === "out" && layer === this.nLayers - 1) && !lens.layers(site).includes(layer)) {
      return { matrix: null, label: null };
    }
    return outputTransport(this.context, lens, layer, site);
  }

  private catalogBlock(): { bound: false; slug: null; url: null; credit: null; reason: string } {
    return { bound: false, slug: null, url: null, credit: null, reason: NO_CATALOG };
  }

  private catalogSummary(): { bound: false; slug: null; credit: null; reason: string } {
    return { bound: false, slug: null, credit: null, reason: NO_CATALOG };
  }

  /** `units/catalog`: native `catalog_status` for an uncatalogued model. */
  catalogStatus(): Record<string, unknown> {
    return unitCatalogStatus(this.context.modelId);
  }

  /** `units/labels`: no catalog, so no rows; each unit is still parsed and checked. */
  labels(units: readonly string[]): Record<string, unknown> {
    for (const unit of units) parseUnit(unit, this.spec, this.nLayers);
    return { catalog: this.catalogSummary(), labels: [] };
  }

  private async layerArtifacts(layer: number): Promise<UnitLayerArtifacts> {
    return this.artifacts.layer(layer);
  }

  private artifact(table: UnitLayerArtifacts, key: string): Float32Array {
    const value = table.get(key);
    if (value === undefined) throw new Error(`the units pack lacks ${key}`);
    return value;
  }

  /** Vocabulary rows times a host factor `[d, m]`, streamed over `ids` → `[n, m]`. */
  private async rowsTimes(kind: "embed" | "unembed", ids: Int32Array, factor: Float32Array, m: number): Promise<Float32Array> {
    const out = new Float32Array(ids.length * m);
    const statistics = kind === "embed" ? (await this.context.embeddings()).resident : undefined;
    for (let start = 0; start < ids.length; start += VOCAB_CHUNK) {
      const chunk = ids.subarray(start, Math.min(ids.length, start + VOCAB_CHUNK));
      const part = await this.context.backend.vocabRowsTimes({ kind, ids: chunk, factor: f32Tensor(factor, [this.d, m]), ...(statistics ? { statistics } : {}) });
      out.set(part.data, start * m);
    }
    return out;
  }

  /** Vocabulary rows through a resident lens matrix (`rows @ M` or `rows @ Mᵀ`) → `[n, d]`. */
  private async rowsThrough(kind: "embed" | "unembed", ids: Int32Array, lens: BlockSitesLensLayer, transpose: boolean): Promise<Float32Array> {
    const out = new Float32Array(ids.length * this.d);
    const statistics = kind === "embed" ? (await this.context.embeddings()).resident : undefined;
    for (let start = 0; start < ids.length; start += VOCAB_CHUNK) {
      const chunk = ids.subarray(start, Math.min(ids.length, start + VOCAB_CHUNK));
      const part = await this.context.backend.vocabRowsTimes({ kind, ids: chunk, factor: { lens, transpose }, ...(statistics ? { statistics } : {}) });
      out.set(part.data, start * this.d);
    }
    return out;
  }

  /** `Fᵀ`-side product with a resident matrix: `M @ F` (`transpose`: `Mᵀ @ F`) for `F [d, m]`. */
  private async matrixTimes(lens: BlockSitesLensLayer, factor: Float32Array, m: number, transpose: boolean): Promise<Float32Array> {
    // (M F)ᵀ = Fᵀ Mᵀ: transport the rows of Fᵀ through Mᵀ (and through M for Mᵀ F).
    const ft = new Float32Array(m * this.d);
    for (let i = 0; i < this.d; i += 1) for (let j = 0; j < m; j += 1) ft[j * this.d + i] = factor[i * m + j];
    const moved = await this.context.backend.lensTransport(f32Tensor(ft, [1, m, this.d]), [lens], !transpose);
    const out = new Float32Array(this.d * m);
    for (let j = 0; j < m; j += 1) for (let i = 0; i < this.d; i += 1) out[i * m + j] = moved.data[j * this.d + i];
    return out;
  }

  /** Top `k` of `values` (`[n, width]`, value desc, index asc; −∞ skipped), one host pass. */
  private topTokens(values: Float64Array | Float32Array, rows: number, k: number, ids?: Int32Array): { token_id: number; token: string; value: number }[][] {
    const width = values.length / rows;
    return Array.from({ length: rows }, (_, r) => {
      const order = Array.from({ length: width }, (_, i) => i).filter((i) => Number.isFinite(values[r * width + i]));
      order.sort((a, b) => values[r * width + b] - values[r * width + a] || a - b);
      return order.slice(0, Math.min(k, width)).map((i) => {
        const id = ids ? ids[i] : i;
        return { token_id: id, token: this.context.decode(id), value: round6(values[r * width + i]) };
      });
    });
  }

  // ----------------------------------------------------------------- neurons
  async neuron(ref: UnitRef, k: number, hideHubs: boolean): Promise<Record<string, unknown>> {
    k = this.checkK(k);
    checkUnit(ref, this.spec, this.nLayers);
    const d = this.d;
    const units = [{ kind: "neuron" as const, layer: ref.layer, index: ref.index }];
    const w = (await this.context.backend.unitWeightRows("neuron_write", units)).data;
    const ab = (await this.context.backend.unitWeightRows("neuron_read", units)).data;
    const plus = Float32Array.from({ length: d }, (_, i) => ab[i] + ab[d + i]);
    const minus = Float32Array.from({ length: d }, (_, i) => ab[i] - ab[d + i]);
    let normPlus = 0;
    let normMinus = 0;
    for (let i = 0; i < d; i += 1) {
      normPlus += plus[i] ** 2;
      normMinus += minus[i] ** 2;
    }
    const cosAb = (normPlus - normMinus) / 4;
    const writesRows = new Float32Array(2 * d);
    writesRows.set(w, 0);
    writesRows.set(w.map((x) => -x), d);
    const firesRows = new Float32Array(2 * d);
    firesRows.set(plus, 0);
    firesRows.set(minus, d);
    const [writesReads, firesReads] = await readVectors(this.context, [
      { rows: writesRows, layer: ref.layer, site: "out", of: "direction" },
      { rows: firesRows, layer: ref.layer, site: "mid", of: "direction" },
    ], k);
    let writes = writesReads.map(place);
    let fires = firesReads.map(place);
    if (hideHubs) {
      writes = await this.hubless(writes, writesRows, [1, -1], k, [["writes", "writes"], ["writes", "writes"]], "writes");
      fires = await this.hubless(fires, firesRows, [1, 1], k, [["fires+", "fires-"], ["fires", "fires"]], "fires");
    }
    const embeddings = await this.context.embeddings();
    const moments = await this.context.backend.vocabLogitMoments(f32Tensor(w, [1, d]), embeddings.resident);
    const projector = await this.artifacts.nullProjector();
    let wNorm = 0;
    let quad = 0;
    for (let i = 0; i < d; i += 1) {
      wNorm += w[i] ** 2;
      let row = 0;
      for (let j = 0; j < d; j += 1) row += projector[i * d + j] * w[j];
      quad += w[i] * row;
    }
    const nullShare = quad / Math.max(wNorm, 1e-30);
    const skew = moments.data[0];
    const kurt = moments.data[1];
    const firing = this.neuronFiring(ref);
    const stats = this.artifacts.statistics;
    return {
      unit: unitName(ref),
      kind: "neuron",
      layer: ref.layer,
      index: ref.index,
      depth: round6(layerDepth(ref.layer, this.nLayers)),
      scalars: {
        w_norm: round6(Math.sqrt(wNorm)),
        cos_ab: round6(cosAb),
        null_share: round6(nullShare),
        mean_act: stats ? round6(stats.neuronMean[ref.layer * this.spec.dFF + ref.index]) : null,
      },
      poles: (["+", "-"] as const).map((pole, i) => ({
        pole,
        writes: writes[i],
        fires_on: fires[i],
        vocab_skew: round6(pole === "+" ? skew : -skew),
        vocab_kurt: round6(kurt),
        firing: firing[i],
        label: null,
      })),
      catalog: this.catalogBlock(),
    };
  }

  /** `_hubless`: re-rank each present half by the layer-contrastive z. */
  private async hubless(
    places: PlaceReadJSON[],
    rows: Float32Array,
    signs: readonly number[],
    k: number,
    populations: readonly [readonly string[], readonly string[]],
    where: "writes" | "fires",
  ): Promise<PlaceReadJSON[]> {
    const { layer, site } = places[0];
    const d = this.d;
    const n = places.length;
    const table = await this.layerArtifacts(layer);
    const embeddings = await this.context.embeddings();
    const V = this.context.vocabSize;
    const masked = (z: Float64Array) => {
      for (let r = 0; r < n; r += 1) for (let v = 0; v < V; v += 1) if (!embeddings.keep[v]) z[r * V + v] = -Infinity;
      return z;
    };
    let rHalves = places.map((p) => p.r);
    let sHalves = places.map((p) => p.s);
    const [rNames, sNames] = populations;
    if (rHalves.some((h) => h.absent === null)) {
      const { matrix, label } = await this.rTransport(layer, site);
      if (label !== null && table.has(`R.${rNames[0]}.mean`)) {
        const x = matrix === null ? rows : (await this.context.backend.lensTransport(f32Tensor(rows, [1, n, d]), [matrix], true)).data;
        const all = Int32Array.from({ length: n * V }, (_, i) => i % V);
        const logits = (await this.context.backend.vocabTokenLogits(f32Tensor(x, [n, d]), i32Tensor(all, [n, V]))).logits.data;
        const z = new Float64Array(n * V);
        for (let r = 0; r < n; r += 1) {
          const mean = this.artifact(table, `R.${rNames[r]}.mean`);
          const std = this.artifact(table, `R.${rNames[r]}.std`);
          for (let v = 0; v < V; v += 1) z[r * V + v] = (logits[r * V + v] - signs[r] * mean[v]) / std[v];
        }
        const lists = this.topTokens(masked(z), n, k);
        rHalves = rHalves.map((h, r) => h.absent === null ? { family: "R", weight: h.weight, unit: "layer_z", lens: h.lens ?? label, tokens: lists[r], absent: null } : h);
      }
    }
    if (sHalves.some((h) => h.absent === null)) {
      const matrix = await this.sMatrix(layer, site);
      if (matrix !== null && table.has(`S.${sNames[0]}.mean`)) {
        const u = (await this.context.backend.lensTransport(f32Tensor(rows, [1, n, d]), [matrix], false)).data;
        const unit = new Float32Array(d * n);
        for (let r = 0; r < n; r += 1) {
          let norm = 0;
          for (let i = 0; i < d; i += 1) norm += u[r * d + i] ** 2;
          norm = Math.max(Math.sqrt(norm), 1e-12);
          for (let i = 0; i < d; i += 1) unit[i * n + r] = u[r * d + i] / norm;
        }
        const ids = Int32Array.from({ length: V }, (_, v) => v);
        const scores = await this.rowsTimes("embed", ids, unit, n);
        const z = new Float64Array(n * V);
        for (let r = 0; r < n; r += 1) {
          const mean = this.artifact(table, `S.${sNames[r]}.mean`);
          const std = this.artifact(table, `S.${sNames[r]}.std`);
          for (let v = 0; v < V; v += 1) z[r * V + v] = (scores[v * n + r] - signs[r] * mean[v]) / std[v];
        }
        const lists = this.topTokens(masked(z), n, k);
        sHalves = sHalves.map((h, r) => h.absent === null ? { family: "S", weight: h.weight, unit: "layer_z", lens: h.lens, tokens: lists[r], absent: null } : h);
      }
    }
    return places.map((p, i) => ({ ...p, hubs: "hidden" as const, s: sHalves[i], r: rHalves[i] }));
  }

  private scaleBlock(top: number, value: number): Record<string, unknown> {
    const scales = this.artifacts.scales!;
    return { source: scales.source, detail: scales.detail, top: round6(top), value: round6(value) };
  }

  private neuronFiring(ref: UnitRef): Record<string, unknown>[] {
    const stats = this.artifacts.statistics;
    const scales = this.artifacts.scales;
    const F = this.spec.dFF;
    return [0, 1].map((row) => {
      let neutral: Record<string, number> | null = null;
      let scale: Record<string, unknown> | null = null;
      if (stats !== null && scales !== null) {
        const at = (ref.layer * 2 + row) * F + ref.index;
        neutral = {
          act_top: round6(stats.neuronScale[at]),
          mean: round6(stats.neuronMean[ref.layer * F + ref.index]),
          documents: stats.documents,
        };
        scale = this.scaleBlock(scales.neuronTop[at], scales.neuron[at]);
      }
      return firing(neutral, stats === null ? this.artifacts.statisticsAbsent : null, scale);
    });
  }

  private headFiring(ref: UnitRef): Record<string, unknown> {
    const stats = this.artifacts.statistics;
    const scales = this.artifacts.scales;
    const H = this.spec.nHeads;
    const hd = this.spec.headDim;
    let neutral: Record<string, number> | null = null;
    let scale: Record<string, unknown> | null = null;
    if (stats !== null && scales !== null) {
      const at = ref.layer * H + ref.index;
      let norm = 0;
      for (let i = 0; i < hd; i += 1) norm += stats.headMean[at * hd + i] ** 2;
      neutral = { write_top: round6(stats.headScale[at]), mean_z_norm: round6(Math.sqrt(norm)), documents: stats.documents };
      scale = this.scaleBlock(scales.headTop[at], scales.head[at]);
    }
    return firing(neutral, stats === null ? this.artifacts.statisticsAbsent : null, scale);
  }

  // ------------------------------------------------------------------- heads
  /** Head `h`'s `role` rows through its head norm's mean derivative, `[hd, d]`. */
  private projection(rows: Float32Array, role: number, ref: UnitRef, table: UnitLayerArtifacts): Float32Array {
    const hd = this.spec.headDim;
    const d = this.d;
    const raw = rows.subarray(role * hd * d, (role + 1) * hd * d);
    const name = HEAD_ROLES[role];
    const derivative = table.get(`heads.norm_derivative.${name}`);
    if (derivative === undefined) return Float32Array.from(raw);
    const index = role === 0 ? ref.index : Math.floor(ref.index / (this.spec.nHeads / this.spec.nKvHeads));
    const D = derivative.subarray(index * hd * hd, (index + 1) * hd * hd);
    const out = new Float32Array(hd * d);
    for (let a = 0; a < hd; a += 1) {
      for (let c = 0; c < hd; c += 1) {
        const x = D[a * hd + c];
        if (x === 0) continue;
        for (let i = 0; i < d; i += 1) out[a * d + i] += x * raw[c * d + i];
      }
    }
    return out;
  }

  /** Per head of `heads`: `[d, hd]` value factor (`Pᵀ`) side by side → B factor `[d, H'·hd]`
   *  through `K^{out⊤}_{L−1}` (or the embedding itself at L0); null without that matrix. */
  private async bFactor(layer: number, heads: readonly number[], table: UnitLayerArtifacts): Promise<Float32Array | null> {
    const hd = this.spec.headDim;
    const d = this.d;
    const through = layer - 1 < 0 ? null : await this.sMatrix(layer - 1, "out");
    if (layer - 1 >= 0 && through === null) return null;
    const rows = (await this.context.backend.unitWeightRows("head_read", heads.map((h) => ({ kind: "head" as const, layer, index: h })))).data;
    const width = heads.length * hd;
    const factor = new Float32Array(d * width);
    heads.forEach((h, n) => {
      const value = this.projection(rows.subarray(n * 3 * hd * d, (n + 1) * 3 * hd * d), 2, { kind: "head", layer, index: h }, table);
      for (let a = 0; a < hd; a += 1) for (let i = 0; i < d; i += 1) factor[i * width + n * hd + a] = value[a * d + i];
    });
    // Kᵀ F: the footprint X_v = ẽ_v Kᵀ, then the value rows.
    return through === null ? factor : this.matrixTimes(through, factor, width, true);
  }

  /** `(rows, factor, label)` reading each output token in the heads' writes at `mid`. */
  private async aFactor(layer: number, heads: readonly number[], family: LensFamilyName): Promise<{ kind: "embed" | "unembed"; factor: Float32Array; label: string } | null> {
    const hd = this.spec.headDim;
    const d = this.d;
    const writes = (await this.context.backend.unitWeightRows("head_write", heads.map((h) => ({ kind: "head" as const, layer, index: h })))).data;
    const width = heads.length * hd;
    const write = new Float32Array(d * width);
    heads.forEach((_, n) => {
      for (let i = 0; i < d; i += 1) for (let a = 0; a < hd; a += 1) write[i * width + n * hd + a] = writes[(n * d + i) * hd + a];
    });
    if (family === "R") {
      const j = await this.rMatrix(layer, "mid");
      if (j === null) return null;
      return { kind: "unembed", factor: await this.matrixTimes(j, write, width, false), label: lensLabel("R", "mid") };
    }
    const kMid = await this.sMatrix(layer, "mid");
    if (kMid === null) return null;
    return { kind: "embed", factor: await this.matrixTimes(kMid, write, width, true), label: lensLabel("S", "mid") };
  }

  /** `_ov_maps`: per family each head's own-token z `[H', V']` and `Bᵀ A` `[H', hd, hd]`. */
  private async ovMaps(
    aFactors: ReadonlyMap<LensFamilyName, { kind: "embed" | "unembed"; factor: Float32Array }>,
    bFactor: Float32Array,
    ids: Int32Array,
    heads: number,
  ): Promise<Map<LensFamilyName, { z: Float64Array; bta: Float64Array }>> {
    const hd = this.spec.headDim;
    const width = heads * hd;
    const n = ids.length;
    const B = await this.rowsTimes("embed", ids, bFactor, width);
    const out = new Map<LensFamilyName, { z: Float64Array; bta: Float64Array }>();
    for (const [family, a] of aFactors) {
      const A = await this.rowsTimes(a.kind, ids, a.factor, width);
      const mean = new Float64Array(width);
      for (let v = 0; v < n; v += 1) for (let j = 0; j < width; j += 1) mean[j] += A[v * width + j];
      for (let j = 0; j < width; j += 1) mean[j] /= n;
      const cov = new Float64Array(heads * hd * hd);
      const bta = new Float64Array(heads * hd * hd);
      for (let v = 0; v < n; v += 1) {
        for (let h = 0; h < heads; h += 1) {
          const base = v * width + h * hd;
          for (let i = 0; i < hd; i += 1) {
            const ci = A[base + i] - mean[h * hd + i];
            const bi = B[base + i];
            for (let j = 0; j < hd; j += 1) {
              cov[(h * hd + i) * hd + j] += ci * (A[base + j] - mean[h * hd + j]);
              bta[(h * hd + i) * hd + j] += bi * A[base + j];
            }
          }
        }
      }
      for (let i = 0; i < cov.length; i += 1) cov[i] /= n;
      const z = new Float64Array(heads * n);
      for (let v = 0; v < n; v += 1) {
        for (let h = 0; h < heads; h += 1) {
          const base = v * width + h * hd;
          let own = 0;
          let variance = 0;
          for (let i = 0; i < hd; i += 1) {
            own += (A[base + i] - mean[h * hd + i]) * B[base + i];
            let row = 0;
            for (let j = 0; j < hd; j += 1) row += cov[(h * hd + i) * hd + j] * B[base + j];
            variance += B[base + i] * row;
          }
          z[h * n + v] = own / Math.sqrt(Math.max(variance, 1e-30));
        }
      }
      out.set(family, { z, bta });
    }
    return out;
  }

  async head(ref: UnitRef, k: number, hideHubs: boolean): Promise<Record<string, unknown>> {
    k = this.checkK(k);
    checkUnit(ref, this.spec, this.nLayers);
    const { layer } = ref;
    const hd = this.spec.headDim;
    const hubs = hideHubs ? "hidden" : "shown";
    const depth = round6(layerDepth(layer, this.nLayers));
    const ids = await this.artifacts.headKeptIds();
    const table = await this.layerArtifacts(layer);
    const [ws, wr] = this.weights(layer);
    const b = await this.bFactor(layer, [ref.index], table);
    const bAbsent = b !== null ? null
      : `a head's input side reads through the S lens at L${layer - 1} out, and ${this.context.s === null ? "no S-family lens is selected" : "it holds no such matrix"}`;
    const aFactors = new Map<LensFamilyName, { kind: "embed" | "unembed"; factor: Float32Array; label: string }>();
    for (const family of ["R", "S"] as const) {
      const a = await this.aFactor(layer, [ref.index], family);
      if (a !== null) aFactors.set(family, a);
    }
    const maps = b === null || aFactors.size === 0 ? new Map<LensFamilyName, { z: Float64Array; bta: Float64Array }>() : await this.ovMaps(aFactors, b, ids, 1);
    const copies = new Map<LensFamilyName, UnitHalfJSON>();
    const suppresses = new Map<LensFamilyName, UnitHalfJSON>();
    const scores: Record<string, number | null> = { R: null, S: null };
    for (const [family, weight] of [["R", wr], ["S", ws]] as const) {
      const label = aFactors.get(family)?.label ?? null;
      const map = maps.get(family);
      if (map === undefined) {
        const why = b === null ? bAbsent! : this.noLens(family, layer, "mid");
        const absent: UnitHalfJSON = { family, weight: round6(weight), unit: "own_token_z", lens: label, tokens: [], absent: why };
        copies.set(family, absent);
        suppresses.set(family, absent);
        continue;
      }
      let z = map.z;
      scores[family] = copyScore(map.bta, hd);
      let unit: UnitHalfJSON["unit"] = "own_token_z";
      if (hideHubs) {
        const { mean, std } = await this.headContrast(layer, family, table, ids);
        z = Float64Array.from(z, (value, v) => (value - mean[v]) / std[v]);
        unit = "layer_z";
      }
      const stacked = new Float64Array(2 * z.length);
      stacked.set(z, 0);
      stacked.set(z.map((x) => -x), z.length);
      const [up, down] = this.topTokens(stacked, 2, k, ids);
      copies.set(family, { family, weight: round6(weight), unit, lens: label, tokens: up, absent: null });
      suppresses.set(family, { family, weight: round6(weight), unit, lens: label, tokens: down.map((t) => ({ ...t, value: round6(-t.value) })), absent: null });
    }
    return {
      unit: unitName(ref),
      kind: "head",
      layer,
      index: ref.index,
      kv_head: Math.floor(ref.index / (this.spec.nHeads / this.spec.nKvHeads)),
      depth,
      copies: { layer, site: "mid", depth, hubs, s: copies.get("S"), r: copies.get("R") },
      suppresses: { layer, site: "mid", depth, hubs, s: suppresses.get("S"), r: suppresses.get("R") },
      inputs: await this.headInputs(ref, table, k, ids),
      copy_score: { R: num(scores.R), S: num(scores.S) },
      firing: this.headFiring(ref),
      label: null,
      catalog: this.catalogBlock(),
    };
  }

  /** `_head_contrast`: mean and unbiased std across the layer's heads of each token's own-token z. */
  private async headContrast(layer: number, family: LensFamilyName, table: UnitLayerArtifacts, ids: Int32Array): Promise<{ mean: ArrayLike<number>; std: ArrayLike<number> }> {
    const mean = table.get(`${family}.heads.ov.mean`);
    const std = table.get(`${family}.heads.ov.std`);
    if (mean !== undefined && std !== undefined) return { mean, std };
    const heads = Array.from({ length: this.spec.nHeads }, (_, h) => h);
    const b = await this.bFactor(layer, heads, table);
    const a = await this.aFactor(layer, heads, family);
    if (b === null || a === null) throw new Error("head contrast needs both OV factors");
    const z = (await this.ovMaps(new Map([[family, a]]), b, ids, heads.length)).get(family)!.z;
    const n = ids.length;
    const H = heads.length;
    const m = new Float64Array(n);
    const s = new Float64Array(n);
    for (let v = 0; v < n; v += 1) {
      let total = 0;
      for (let h = 0; h < H; h += 1) total += z[h * n + v];
      m[v] = total / H;
      let sq = 0;
      for (let h = 0; h < H; h += 1) sq += (z[h * n + v] - m[v]) ** 2;
      s[v] = Math.max(Math.sqrt(sq / (H - 1)), HEAD_STD_FLOOR);
    }
    return { mean: m, std: s };
  }

  /** `_head_inputs`: what each projection picks up at `L − 1`'s output. */
  private async headInputs(ref: UnitRef, table: UnitLayerArtifacts, k: number, ids: Int32Array): Promise<Record<string, PlaceReadJSON>> {
    const { layer } = ref;
    const at = layer - 1;
    const siteLayer = Math.max(at, 0);
    const [ws, wr] = this.weights(siteLayer);
    const depth = round6(layerDepth(siteLayer, this.nLayers));
    const hd = this.spec.headDim;
    const d = this.d;
    const rows = (await this.context.backend.unitWeightRows("head_read", [{ kind: "head", layer, index: ref.index }])).data;
    const width = 3 * hd;
    const projections = new Float32Array(d * width);
    for (let role = 0; role < 3; role += 1) {
      const p = this.projection(rows, role, ref, table);
      for (let a = 0; a < hd; a += 1) for (let i = 0; i < d; i += 1) projections[i * width + role * hd + a] = p[a * d + i];
    }
    const energies = new Map<LensFamilyName, Float64Array>();
    const sThrough = at < 0 ? null : await this.sMatrix(at, "out");
    if (at < 0 || sThrough !== null) energies.set("S", await this.inputEnergies("embed", sThrough, true, projections, ids));
    const jOut = at < 0 ? null : await this.rMatrix(at, "out");
    if (jOut !== null) energies.set("R", await this.inputEnergies("unembed", jOut, false, projections, ids));
    const labels: Record<LensFamilyName, string | null> = {
      S: at < 0 ? "embedding" : this.context.s === null ? null : lensLabel("S", "out"),
      R: this.context.r === null || at < 0 ? null : lensLabel("R", "out"),
    };
    const out: Record<string, PlaceReadJSON> = {};
    HEAD_ROLES.forEach((role, index) => {
      const halves = {} as Record<LensFamilyName, UnitHalfJSON>;
      for (const [family, weight] of [["S", ws], ["R", wr]] as const) {
        const energy = energies.get(family);
        if (energy === undefined) {
          const why = family === "R" && at < 0
            ? "the embedding has no R read (a head at L0 reads the token embeddings)"
            : this.noLens(family, at, "out");
          halves[family] = { family, weight: round6(weight), unit: "energy_z", lens: labels[family], tokens: [], absent: why };
          continue;
        }
        const z = energy.subarray(index * ids.length, (index + 1) * ids.length);
        halves[family] = { family, weight: round6(weight), unit: "energy_z", lens: labels[family], tokens: this.topTokens(z, 1, k, ids)[0], absent: null };
      }
      out[role] = { layer: siteLayer, site: "out", depth, hubs: "shown", s: halves.S, r: halves.R };
    });
    return out;
  }

  /** `_input_energies`: per projection the vocabulary z of `‖P x̂_v‖`, `x_v = rows_v · through`. */
  private async inputEnergies(
    kind: "embed" | "unembed",
    through: BlockSitesLensLayer | null,
    throughTransposed: boolean,
    projections: Float32Array,
    ids: Int32Array,
  ): Promise<Float64Array> {
    const hd = this.spec.headDim;
    const d = this.d;
    const width = 3 * hd;
    // through as a right factor: S `rows @ Kᵀ`, R `rows @ J`.
    const factor = through === null ? projections : await this.matrixTimes(through, projections, width, throughTransposed);
    const projected = await this.rowsTimes(kind, ids, factor, width);
    const base = through === null
      ? await this.rowsTimes(kind, ids, identity(d), d)
      : await this.rowsThrough(kind, ids, through, throughTransposed);
    const n = ids.length;
    const energy = new Float64Array(3 * n);
    for (let v = 0; v < n; v += 1) {
      let norm = 0;
      for (let i = 0; i < d; i += 1) norm += base[v * d + i] ** 2;
      norm = Math.max(Math.sqrt(norm), 1e-12);
      for (let role = 0; role < 3; role += 1) {
        let e = 0;
        for (let a = 0; a < hd; a += 1) e += projected[v * width + role * hd + a] ** 2;
        energy[role * n + v] = Math.sqrt(e) / norm;
      }
    }
    for (let role = 0; role < 3; role += 1) {
      const row = energy.subarray(role * n, (role + 1) * n);
      let mean = 0;
      for (const x of row) mean += x;
      mean /= n;
      let variance = 0;
      for (const x of row) variance += (x - mean) ** 2;
      const std = Math.max(Math.sqrt(variance / n), 1e-12);
      for (let v = 0; v < n; v += 1) row[v] = (row[v] - mean) / std;
    }
    return energy;
  }

  // -------------------------------------------------------------------- read
  async read(unit: string, k = 8, hideHubs = false): Promise<Record<string, unknown>> {
    const ref = parseUnit(unit, this.spec, this.nLayers);
    return ref.kind === "neuron" ? this.neuron(ref, k, hideHubs) : this.head(ref, k, hideHubs);
  }

  // --------------------------------------------------------------------- top
  async top(token: string, by: "writes" | "fires" = "writes", layers: LayerSelector = null, k = 20): Promise<Record<string, unknown>> {
    if (by !== "writes" && by !== "fires") {
      throw new UnitRequestError("RequestError", `by must be 'writes' or 'fires', got ${pyRepr(String(by))}`);
    }
    k = this.checkK(k);
    const all = Array.from({ length: this.nLayers }, (_, layer) => layer);
    const selected = layerRequest(() => resolveLayerSelector(layers, all, this.nLayers, false));
    for (const layer of selected) {
      if (!(layer >= 0 && layer < this.nLayers)) {
        throw new UnitRequestError("RequestError", `layer ${layer} is outside the model's ${this.nLayers} layers`);
      }
    }
    const tokenId = this.options.resolveToken(token);
    const embeddings = await this.context.embeddings();
    if (!embeddings.keep[tokenId]) {
      throw new UnitRequestError(
        "RequestError",
        `${pyRepr(token)} is token ${tokenId}, which the token mask drops (special, added and byte-fallback pieces and tokens too rare in the counted text never rank; mask: ${embeddings.mask.source})`,
      );
    }
    const site = by === "writes" ? "out" : "mid";
    const candidates: { score: number; layer: number; index: number; pole: number; r: Record<string, unknown>; s: Record<string, unknown> }[] = [];
    const info: Record<string, unknown> = {};
    for (const layer of [...new Set(selected)].sort((a, b) => a - b)) {
      const scored = await this.topLayer(layer, site, by, tokenId);
      info[String(layer)] = scored.info;
      candidates.push(...scored.rows);
    }
    candidates.sort((a, b) => b.score - a.score);
    return {
      token_id: tokenId,
      token: this.context.decode(tokenId),
      by,
      hits: candidates.slice(0, k).map((row) => ({
        unit: `neuron/L${row.layer}n${row.index}`,
        layer: row.layer,
        index: row.index,
        pole: row.pole === 0 ? "+" : "-",
        score: round6(row.score),
        r: row.r,
        s: row.s,
        label: null,
      })),
      layers: info,
    };
  }

  /** `_top_layer` through B1 §4.2's shortcut: each statistic `top` reads at token `v` is the
   *  population's mean and floored std of the per-row score it already computes. */
  private async topLayer(layer: number, site: "out" | "mid", by: "writes" | "fires", token: number) {
    const F = this.spec.dFF;
    const d = this.d;
    const [ws, wr] = this.weights(layer);
    const table = await this.layerArtifacts(layer);
    const layers = Int32Array.of(layer);
    const stats = (values: Float64Array, floor: number) => {
      let sum = 0;
      let sq = 0;
      for (const x of values) {
        sum += x;
        sq += x * x;
      }
      const mean = sum / values.length;
      return { mean, std: Math.sqrt(Math.max(sq / values.length - mean * mean, floor * floor)) };
    };
    let rZ: Float64Array[] | null = null;
    let rP: Float64Array[] | null = null;
    let sZ: Float64Array[] | null = null;
    let sFz: Float64Array[] | null = null;
    const { matrix: rMatrix, label: rLabel } = await this.rTransport(layer, site);
    if (rLabel !== null) {
      // g = (γ_f ⊙ W_U[v]) J (the identity at the logit lens): lg_n = ρ_n · g·x_n.
      const g = rMatrix === null
        ? await this.rowsTimes("unembed", Int32Array.of(token), identity(d), d)
        : await this.rowsThrough("unembed", Int32Array.of(token), rMatrix, false);
      if (by === "writes") {
        const dots = (await this.context.backend.layerWriteScores(f32Tensor(g, [1, 1, d]), layers)).data;
        const rho = this.artifact(table, "R.writes.rho");
        const lg = Float64Array.from({ length: F }, (_, n) => rho[n] * dots[n]);
        const { mean, std } = stats(lg, R_STD_FLOOR);
        const zPlus = lg.map((x) => (x - mean) / std);
        const lse = this.artifact(table, "R.writes.lse");
        const lseNeg = this.artifact(table, "R.writes.lse_neg");
        rZ = [zPlus, zPlus.map((x) => -x)];
        rP = [lg.map((x, n) => Math.exp(x - lse[n])), lg.map((x, n) => Math.exp(-x - lseNeg[n]))];
      } else {
        const read = (await this.context.backend.layerReadScores(f32Tensor(g, [1, 1, d]), layers)).data;
        rZ = [];
        rP = [];
        for (const [name, sign] of [["fires+", 1], ["fires-", -1]] as const) {
          const rho = this.artifact(table, `R.${name}.rho`);
          const lse = this.artifact(table, `R.${name}.lse`);
          const lg = Float64Array.from({ length: F }, (_, n) => rho[n] * (read[n] + sign * read[F + n]));
          const { mean, std } = stats(lg, R_STD_FLOOR);
          rZ.push(lg.map((x) => (x - mean) / std));
          rP.push(lg.map((x, n) => Math.exp(x - lse[n])));
        }
      }
    }
    const sMatrix = await this.sMatrix(layer, site);
    let sLabel: string | null = null;
    if (sMatrix !== null) {
      sLabel = lensLabel("S", site);
      // x = K ẽ_v: s_n = (row_n · x) / ‖u_n‖, footprint z = (row_n · x) / sqrt(u_nᵀ Σ_E u_n).
      const x = await this.rowsThrough("embed", Int32Array.of(token), sMatrix, true);
      if (by === "writes") {
        const raw = Float64Array.from((await this.context.backend.layerWriteScores(f32Tensor(x, [1, 1, d]), layers)).data);
        const norm = this.artifact(table, "S.writes.norm");
        const footprint = this.artifact(table, "S.writes.footprint_std");
        const s = raw.map((value, n) => value / norm[n]);
        const { mean, std } = stats(s, S_STD_FLOOR);
        const zPlus = s.map((value) => (value - mean) / std);
        sZ = [zPlus, zPlus.map((value) => -value)];
        const fz = raw.map((value, n) => value / footprint[n]);
        sFz = [fz, fz.map((value) => -value)];
      } else {
        const read = (await this.context.backend.layerReadScores(f32Tensor(x, [1, 1, d]), layers)).data;
        const poles = ([["fires+", 1], ["fires-", -1]] as const).map(([name, sign]) => {
          const raw = Float64Array.from({ length: F }, (_, n) => read[n] + sign * read[F + n]);
          const norm = this.artifact(table, `S.${name}.norm`);
          const footprint = this.artifact(table, `S.${name}.footprint_std`);
          return { s: raw.map((value, n) => value / norm[n]), fz: raw.map((value, n) => value / footprint[n]) };
        });
        const pooled = new Float64Array(2 * F);
        pooled.set(poles[0].s, 0);
        pooled.set(poles[1].s, F);
        const { mean, std } = stats(pooled, S_STD_FLOOR);
        sZ = poles.map((pole) => pole.s.map((value) => (value - mean) / std));
        sFz = poles.map((pole) => pole.fz);
      }
    }
    const presentR = rZ !== null;
    const presentS = sZ !== null;
    const info = {
      weights: { S: round6(ws), R: round6(wr) },
      lens: { S: sLabel, R: rLabel },
      absent: {
        S: presentS ? null : this.noLens("S", layer, site),
        R: presentR ? null : this.noLens("R", layer, site),
      },
    };
    if (!presentR && !presentS) return { info, rows: [] };
    let wR = presentR ? wr : 0;
    let wS = presentS ? ws : 0;
    let total = wR + wS;
    if (total <= 0) {
      wR = Number(presentR);
      wS = Number(presentS);
      total = wR + wS;
    }
    const score = new Float64Array(2 * F);
    for (let p = 0; p < 2; p += 1) {
      for (let n = 0; n < F; n += 1) {
        let value = 0;
        if (rZ) value += wR * rZ[p][n];
        if (sZ) value += wS * sZ[p][n];
        score[p * F + n] = value / total;
      }
    }
    const keep = Math.min(MAX_UNIT_K, 2 * F);
    const order = Array.from({ length: 2 * F }, (_, i) => i).sort((a, b) => score[b] - score[a] || a - b).slice(0, keep);
    const rows = order.map((flat) => {
      const pole = Math.floor(flat / F);
      const index = flat % F;
      return {
        score: score[flat],
        layer,
        index,
        pole,
        r: { z: num(rZ?.[pole][index]), probability: num(rP?.[pole][index]), weight: round6(wr), lens: rLabel },
        s: { z: num(sZ?.[pole][index]), footprint_z: num(sFz?.[pole][index]), weight: round6(ws), lens: sLabel },
      };
    });
    return { info, rows };
  }

  // -------------------------------------------------------------------- find
  /** `find`: split per-layer directions over each layer's neuron writes (the `neurons`
   *  dictionary, Mahalanobis, `sparse_decompose`). */
  async find(
    target: { selector: string } | { vector: Float32Array; layer: number },
    k = 16,
    layers: LayerSelector = null,
  ): Promise<Record<string, unknown>> {
    k = this.checkK(k);
    let label: string;
    let directions: ReadonlyMap<number, Float32Array>;
    if ("selector" in target) {
      if (!this.options.resolveSelector) throw new UnitRequestError("RequestError", "find cannot resolve selectors here");
      label = target.selector;
      directions = await this.options.resolveSelector(target.selector);
    } else {
      if (target.vector.length !== this.d) {
        throw new UnitRequestError("RequestError", `find takes residual-width vectors (d = ${this.d}), got shape (${target.vector.length},)`);
      }
      label = "vector";
      directions = new Map([[target.layer, target.vector]]);
    }
    const available = [...directions.keys()].sort((a, b) => a - b);
    const requested = layerRequest(() => resolveLayerSelector(layers, available, this.nLayers))
      .filter((layer) => directions.has(layer));
    if (requested.length === 0) {
      throw new UnitRequestError("RequestError", `no overlap between ${pyRepr(label)}'s layers and the requested layers`);
    }
    for (const layer of requested) {
      if (!(layer >= 0 && layer < this.nLayers)) {
        throw new UnitRequestError("RequestError", `layer ${layer} is outside the model's ${this.nLayers} layers`);
      }
    }
    const whiteners = new Map<number, SerializedMahalanobisWhitener>();
    const missing: number[] = [];
    for (const layer of requested) {
      const whitener = await this.artifacts.whitener(layer);
      if (whitener === null) missing.push(layer);
      else whiteners.set(layer, whitener);
    }
    if (missing.length) {
      throw new UnitWhitenerError(
        `decomposition needs a Mahalanobis whitener covering layers [${missing.join(", ")}] (regenerate the neutral activation cache)`,
      );
    }
    const rows: Record<string, unknown> = {};
    const shares: number[] = [];
    for (const layer of requested) {
      const { share, atoms } = await this.decompose(layer, directions.get(layer)!, whiteners.get(layer)!, k);
      shares.push(round6(share));
      rows[String(layer)] = {
        share: round6(share),
        neurons: atoms.map(({ index, coeff }) => ({
          unit: `neuron/L${layer}n${index}`,
          index,
          pole: coeff > 0 ? "+" : "-",
          coeff: round6(coeff),
          label: null,
        })),
      };
    }
    return {
      target: label,
      k,
      mean_share: shares.length ? round6(shares.reduce((a, b) => a + b, 0) / shares.length) : 0,
      layers: rows,
      catalog: this.catalogSummary(),
    };
  }

  /** `sparse_decompose` against `NeuronWrites` (signed, OLS re-solve), scores on the device. */
  private async decompose(
    layer: number,
    target: Float32Array,
    whitener: SerializedMahalanobisWhitener,
    k: number,
  ): Promise<{ share: number; atoms: { index: number; coeff: number }[] }> {
    const d = this.d;
    const t = whitenRows(whitener, target, 1);
    let tNormSq = 0;
    for (const x of t) tNormSq += x * x;
    if (tNormSq === 0) return { share: 0, atoms: [] };
    const table = await this.layerArtifacts(layer);
    const norms = this.artifact(table, "neurons.write_norms");
    const selected: number[] = [];
    const atoms: Float32Array[] = [];
    let coeffs: Float64Array = new Float64Array(0);
    let residual = Float32Array.from(t);
    for (let step = 0; step < k; step += 1) {
      // (W w_n)·r = w_n·(W r): W is symmetric.
      const wr = whitenRows(whitener, residual, 1);
      const raw = (await this.context.backend.layerWriteScores(f32Tensor(wr, [1, 1, d]), Int32Array.of(layer))).data;
      let best = -1;
      let bestScore = -Infinity;
      for (let n = 0; n < raw.length; n += 1) {
        if (selected.includes(n)) continue;
        const score = Math.abs(raw[n] / Math.max(norms[n], 1e-12));
        if (score > bestScore) {
          bestScore = score;
          best = n;
        }
      }
      if (!(bestScore > 0)) break;
      selected.push(best);
      const write = (await this.context.backend.unitWeightRows("neuron_write", [{ kind: "neuron", layer, index: best }])).data;
      atoms.push(whitenRows(whitener, write, 1));
      const s = atoms.length;
      const gram = new Float64Array(s * s);
      const b = new Float64Array(s);
      for (let i = 0; i < s; i += 1) {
        for (let j = 0; j < s; j += 1) gram[i * s + j] = Math.fround(dot(atoms[i], atoms[j]));
        gram[i * s + i] += 1e-7;
        b[i] = Math.fround(dot(atoms[i], t));
      }
      coeffs = solve(gram, b, s);
      residual = Float32Array.from(t);
      for (let i = 0; i < s; i += 1) for (let j = 0; j < d; j += 1) residual[j] -= coeffs[i] * atoms[i][j];
    }
    let rNormSq = 0;
    for (const x of residual) rNormSq += x * x;
    const share = Math.max(0, 1 - rNormSq / tNormSq);
    const pairs = selected
      .map((index, i) => ({ index, coeff: Math.fround(coeffs[i]) }))
      .filter((pair) => pair.coeff !== 0)
      .sort((a, b) => Math.abs(b.coeff) - Math.abs(a.coeff));
    return { share, atoms: pairs };
  }
}

function firing(neutral: Record<string, number> | null, neutralAbsent: string | null, scale: Record<string, unknown> | null): Record<string, unknown> {
  return {
    rate: null,
    rate_absent: NO_CATALOG,
    scale,
    neutral,
    neutral_absent: neutralAbsent,
    catalog: null,
    catalog_absent: NO_CATALOG,
  };
}

function identity(d: number): Float32Array {
  const out = new Float32Array(d * d);
  for (let i = 0; i < d; i += 1) out[i * d + i] = 1;
  return out;
}

function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i += 1) s += a[i] * b[i];
  return s;
}

/** `W x` for each row, `W = Σ^{-1/2}` from the eigen form: `x/√λ + Σ_k (1/√(e_k+λ) − 1/√λ)
 *  (v_k·x) v_k` (native `whiten`, `(1/√λ)(I − V diag(g) Vᵀ)`, `g = 1 − √(λ/(λ+e))`). */
export function whitenRows(whitener: SerializedMahalanobisWhitener, values: ArrayLike<number>, rows: number): Float32Array {
  const d = whitener.columns;
  const out = new Float32Array(rows * d);
  const inv = 1 / Math.sqrt(whitener.ridge);
  for (let r = 0; r < rows; r += 1) {
    const x = Array.prototype.slice.call(values, r * d, (r + 1) * d) as number[];
    const y = x.map((value) => value * inv);
    for (let c = 0; c < whitener.rank; c += 1) {
      let coord = 0;
      for (let i = 0; i < d; i += 1) coord += whitener.basis[c * d + i] * x[i];
      const factor = coord * (1 / Math.sqrt(whitener.eigenvalues[c] + whitener.ridge) - inv);
      for (let i = 0; i < d; i += 1) y[i] += factor * whitener.basis[c * d + i];
    }
    out.set(y, r * d);
  }
  return out;
}
