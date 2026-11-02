// The worker side of drowse 6.0's units members (`instruments.tokenAnatomy`, `unitActivity`,
// `unitRead`, `unitTop`, `unitFind`, `unitCatalog`, `unitLabels`): request parsing as native's
// routes do it, the host ports over the loaded library's block-sites backend, B6's packs, and
// the loom's node replays. A library without unit sites answers native's 422 refusal.
import type { SerializedMahalanobisWhitener } from "../../fitting/workerContracts";
import type { ScalarReadingJSON } from "../../../../../drowse/webui/src/lib/types";
import {
  f32Tensor,
  u32Tensor,
  UnsupportedBlockError,
  type BlockSitesLensLayer,
  type BlockSitesResident,
  type BrowserBlockSitesBackend,
  type LensSite,
} from "../modelBackend";
import type { StructuredHookProgramBuffers } from "../structuredHookProgram";
import type { BlockSpec } from "../structuredHookProfile";
import { checkAnatomyOptions, tokenAnatomy, type AnatomyForward } from "./anatomy";
import { DEFAULT_LENS_BAND, type LensBand, type LensFamilyName, type UnitsEmbeddings, type UnitsLens, type UnitsReadContext } from "./lensReads";
import { checkActivityOptions, unitActivity, type NodeReplayForward, type ReplayGateScores } from "./nodeReplay";
import { noLensText } from "./referenceContext";
import { readingScales, type UnitScales, type UnitStatistics } from "./scales";
import { parseLayerSelector, parseSteeringExpression, resolveModelLayers, type LayerSelector } from "../steeringExpression";
import { slensAtomDirections } from "./slens";
import { centreDigest } from "./probeDigest";
import { lensProbeReadings, slensReadings, UNIT_MEAN_FOOTPRINT_Z, type LensProbe } from "./replayProbes";
import { unitGateScores, type BrowserUnitProgramSource } from "./unitProgram";
import { layerRequest, NO_CATALOG, parseUnit, unitCatalogStatus, UnitReader, UnitRequestError, type UnitLayerArtifacts } from "./unitReader";

export const UNITS_SERVICE_METHODS = Object.freeze([
  "tokenAnatomy",
  "unitActivity",
  "unitRead",
  "unitTop",
  "unitFind",
  "unitCatalog",
  "unitLabels",
] as const);
export type UnitsServiceMethod = (typeof UNITS_SERVICE_METHODS)[number];

export function isUnitsServiceMethod(method: string): method is UnitsServiceMethod {
  return (UNITS_SERVICE_METHODS as readonly string[]).includes(method);
}

/** Units one `unitLabels` request may name (native `MAX_LABEL_UNITS`). */
export const MAX_LABEL_UNITS = 128;

/** One family's lens as a pack holds it: its sites and layers, each matrix fetched (and made
 *  resident) on first read (PLAN: full-depth packs sharded per layer). */
export interface UnitsPackLens {
  /** The source label the anatomy's binding names (`local:r`, a pack id). */
  readonly source: string;
  /** Native `lens_identity(sidecar)` (the pack registry's `lensIdentity(manifest)`): what an
   *  `slens/` probe's digest binds. */
  readonly identity: string;
  readonly sites: readonly LensSite[];
  layers(site: LensSite): readonly number[];
  /** `[d, d]` fp32 (`J_L` or `K_L`). */
  fetch(site: LensSite, layer: number): Promise<Float32Array>;
}

/** What B6's packs give the units members (core pack: statistics, means, whitener; lens
 *  packs; the units-reads pack: per-layer artifacts keyed as native `population_artifacts`). */
export interface UnitsPackSource {
  readonly band: LensBand;
  statistics(): Promise<{ statistics: UnitStatistics | null; absent: string | null }>;
  means(): Promise<{ out: Float32Array | null; mid: Float32Array | null }>;
  lens(family: LensFamilyName): UnitsPackLens | null;
  /** The S embedding statistics (S pack): `mean_E`, `Σ_E`, the mask bitmap and its record. */
  embeddingStatistics(): Promise<{
    meanE: Float32Array;
    sigmaE: Float32Array;
    mask: Uint32Array;
    source: string;
    detail: string;
    /** Native `LensInstrument._mask_digest()`: the first 16 hex of sha256 over the counts
     *  (int64 LE) and `|{min_count}`, or `"tokenizer"` without counts. */
    maskDigest: string;
  }>;
  /** One layer's `reads` or `contrast` artifacts (the contrast only `hide_hubs` reads). */
  layerArtifacts(layer: number, part: "reads" | "contrast"): Promise<UnitLayerArtifacts>;
  /** `‖Nᵀ w‖²` over the null basis. */
  nullProjection(w: Float32Array): Promise<number>;
  headKeptIds(): Promise<Int32Array>;
  whitener(layer: number): Promise<SerializedMahalanobisWhitener | null>;
}

/** A loom node's producing forward, as the loom renders it (`TokenReplayer.forward`). */
export interface UnitsNodeReplay {
  readonly forward: AnatomyForward;
  /** The program the node ran under (null: unsteered, or `steered: false`). */
  readonly program: StructuredHookProgramBuffers | null;
  readonly steering: string | null;
  readonly canThink: boolean;
}

export interface UnitsNodeReplayPort {
  /** The forward producing `rawIndex` of `nodeId` (its last row when null: the whole node). */
  nodeForward(
    nodeId: string,
    rawIndex: number | null,
    options: { steered: boolean; raw: boolean; label: string; produced: boolean },
  ): Promise<UnitsNodeReplay>;
}

export interface BrowserUnitsServiceOptions {
  readonly modelId: string;
  readonly blockSites: () => BrowserBlockSitesBackend;
  readonly packs: () => UnitsPackSource | null;
  /** Each id's own text (native `tokenizer.decode([id])`). */
  readonly tokenTexts: (tokenIds: readonly number[]) => Promise<string[]>;
  /** `slens/`-family token resolution (`resolve_token(word, family="S")`). */
  readonly resolveToken: (word: string) => Promise<number>;
  /** A steerable selector to its per-layer directions (`find`'s selector). */
  readonly resolveSelector?: (selector: string) => Promise<ReadonlyMap<number, Float32Array>>;
  readonly replay: () => UnitsNodeReplayPort | null;
  /** A replayed forward's geometry gate scores from its `[L, maxGeometryProbes, stride]`
   *  readout, as generation's measurement envelope scores them. */
  readonly geometryGateScores?: (program: StructuredHookProgramBuffers, geometry: Float32Array) => Readonly<Record<string, number>>;
}

function badRequest(message: string): UnitRequestError {
  return new UnitRequestError("RequestError", message);
}

function optionalRecord(value: unknown, label: string): Record<string, unknown> {
  if (value === undefined || value === null) return {};
  if (typeof value !== "object" || Array.isArray(value)) throw badRequest(`${label} must be an object`);
  return value as Record<string, unknown>;
}

function integer(value: unknown, fallback: number, label: string): number {
  if (value === undefined || value === null) return fallback;
  if (typeof value !== "number" || !Number.isSafeInteger(value)) throw badRequest(`${label} must be an integer`);
  return value;
}

function bool(value: unknown, fallback: boolean, label: string): boolean {
  if (value === undefined || value === null) return fallback;
  if (typeof value !== "boolean") throw badRequest(`${label} must be a boolean`);
  return value;
}

function text(value: unknown, label: string): string {
  if (typeof value !== "string" || value.trim() === "") throw badRequest(`${label} must be a non-empty string`);
  return value;
}

/** The units routes' `layers` (native `_layers`/`_parse_layers`: `parse_layer_selector` with
 *  `sample` allowed, a malformed selector a 400). */
export function routeLayerSelector(value: unknown): LayerSelector {
  if (value === undefined || value === null) return null;
  if (typeof value !== "string") throw badRequest("layers must be a layer selector string");
  return layerRequest(() => parseLayerSelector(value, true));
}

/** Resident lens matrices over a pack lens (each uploaded once, on first read); `track` sees
 *  every upload, so the owner can release them. */
export function residentLens(
  backend: BrowserBlockSitesBackend,
  family: LensFamilyName,
  lens: UnitsPackLens,
  d: number,
  track: (upload: Promise<BlockSitesResident>) => void = () => undefined,
): UnitsLens {
  const resident = new Map<string, Promise<BlockSitesLensLayer>>();
  return {
    family,
    sites: lens.sites,
    layers: (site) => lens.layers(site),
    matrix(site, layer) {
      const key = `${site}:${layer}`;
      let found = resident.get(key);
      if (!found) {
        found = lens.fetch(site, layer).then((matrix) =>
          backend.uploadLensLayer(family === "R" ? "r" : "s", site, layer, f32Tensor(matrix, [d, d])));
        resident.set(key, found);
        track(found);
        found.catch(() => resident.delete(key));
      }
      return found;
    },
  };
}

/** The units members over one loaded model. */
export class BrowserUnitsService {
  private session: {
    backend: BrowserBlockSitesBackend;
    packs: UnitsPackSource;
    context: UnitsReadContext;
    reader: UnitReader;
    spec: BlockSpec;
    scales: UnitScales | null;
    sources: Record<LensFamilyName, string | null>;
    /** Every resident this session uploaded (lens matrices, embedding statistics). */
    residents: Set<Promise<BlockSitesResident>>;
  } | null = null;

  constructor(private readonly options: BrowserUnitsServiceOptions) {}

  /** Drop every cached read and release the session's residents (a model or pack change). */
  async reset(): Promise<void> {
    const session = this.session;
    this.session = null;
    if (session === null) return;
    const settled = await Promise.allSettled(session.residents);
    for (const result of settled) {
      if (result.status !== "fulfilled") continue;
      // A release that fails (the model went with its residents) has nothing left to free.
      await session.backend.release(result.value).catch(() => undefined);
    }
  }

  /** Native's 422 for a library without unit sites, or the unavailable surface. */
  private requireUnits(): { backend: BrowserBlockSitesBackend; spec: BlockSpec } {
    const backend = this.options.blockSites();
    const capabilities = backend.capabilities();
    if (capabilities === null) {
      throw new UnsupportedBlockError("the loaded model library predates hook ABI block-sites-v5");
    }
    if (capabilities.blockSpec === null) throw new UnsupportedBlockError(capabilities.unitRefusal!);
    return { backend, spec: capabilities.blockSpec };
  }

  private async open() {
    const { backend, spec } = this.requireUnits();
    if (this.session !== null && this.session.backend === backend) return this.session;
    const packs = this.options.packs();
    if (packs === null) {
      throw Object.assign(new Error("The units pack for this model is not installed"), {
        code: "UNITS_PACK_UNAVAILABLE",
        status: 503,
        recoverable: true,
      });
    }
    const capabilities = backend.capabilities()!;
    const d = capabilities.hiddenSize;
    const { statistics, absent } = await packs.statistics();
    const scales = statistics === null ? null : readingScales(statistics, spec, capabilities.layerCount);
    const means = await packs.means();
    const residents = new Set<Promise<BlockSitesResident>>();
    const track = (upload: Promise<BlockSitesResident>) => {
      residents.add(upload);
    };
    const lensOf = (family: LensFamilyName) => {
      const lens = packs.lens(family);
      return lens === null ? null : residentLens(backend, family, lens, d, track);
    };
    let embeddings: Promise<UnitsEmbeddings> | null = null;
    const context: UnitsReadContext = {
      backend,
      modelId: this.options.modelId,
      nLayers: capabilities.layerCount,
      hiddenSize: d,
      vocabSize: capabilities.vocabSize,
      band: packs.band ?? DEFAULT_LENS_BAND,
      r: lensOf("R"),
      s: lensOf("S"),
      noLens: { R: noLensText("R", this.options.modelId), S: noLensText("S", this.options.modelId) },
      means,
      tokenTexts: this.options.tokenTexts,
      embeddings() {
        embeddings ??= (async () => {
          const value = await packs.embeddingStatistics();
          const statistics = {
            meanE: f32Tensor(value.meanE, [d]),
            sigmaE: f32Tensor(value.sigmaE, [d, d]),
            mask: u32Tensor(value.mask, [value.mask.length]),
          };
          const keep = new Uint8Array(capabilities.vocabSize);
          for (let v = 0; v < keep.length; v += 1) keep[v] = (value.mask[v >>> 5] >>> (v & 31)) & 1;
          const kept = Int32Array.from([...keep.keys()].filter((v) => keep[v]));
          const upload = backend.uploadEmbeddingStatistics(statistics);
          track(upload);
          return {
            resident: await upload,
            statistics,
            kept,
            keep,
            mask: { source: value.source, detail: value.detail, kept: kept.length },
          };
        })();
        embeddings.catch(() => {
          embeddings = null;
        });
        return embeddings;
      },
    };
    const reader = new UnitReader({
      context,
      spec,
      artifacts: {
        statistics,
        statisticsAbsent: absent,
        scales,
        layer: (layer, part) => packs.layerArtifacts(layer, part),
        nullProjection: (w) => packs.nullProjection(w),
        headKeptIds: () => packs.headKeptIds(),
        whitener: (layer) => packs.whitener(layer),
      },
      resolveToken: this.options.resolveToken,
      ...(this.options.resolveSelector ? { resolveSelector: this.options.resolveSelector } : {}),
    });
    this.session = {
      backend,
      packs,
      context,
      reader,
      spec,
      scales,
      sources: { R: packs.lens("R")?.source ?? null, S: packs.lens("S")?.source ?? null },
      residents,
    };
    return this.session;
  }

  /** A replayed forward's name-scored gates, as generation scores them after a forward: unit
   *  taps, the geometry readout and the `slens/` reads over the `[L, d]` capture. */
  private readonly replayGateScores: ReplayGateScores = async (program, readback) => {
    const scores: Record<string, number> = { ...unitGateScores(program, readback) };
    if (readback.geometry !== null && this.options.geometryGateScores) {
      Object.assign(scores, this.options.geometryGateScores(program, readback.geometry.data));
    }
    const slens = program.measurementSchema?.slensProbes ?? [];
    if (slens.length > 0 && readback.hidden !== null) {
      Object.assign(scores, (await this.slensReads(readback.hidden.data, slens)).values);
    }
    return scores;
  };

  /** A generation forward's `slens/` reads over its `[L, d]` capture (native
   *  `_InputBlock.readings`) and the S source label the lens binding names. */
  async slensReads(
    hidden: Float32Array,
    probes: readonly { readonly name: string; readonly tokenId: number }[],
  ): Promise<{ readings: Record<string, ScalarReadingJSON>; values: Record<string, number>; source: string | null }> {
    const session = await this.open();
    return { ...await slensReadings(session.context, hidden, probes), source: session.sources.S };
  }

  /** The structured compiler's unit source (`BrowserFeasibilityCorePackCompiler.setUnitSource`)
   *  over the loaded library: its block spec or refusal, the unit statistics, and the
   *  directions `prepare(expression)` reads off the device before a compile (each neuron push's
   *  write, each `slens/` atom's footprints over the S:out layers). */
  async programSource(): Promise<BrowserUnitProgramSource> {
    const backend = this.options.blockSites();
    const capabilities = backend.capabilities();
    if (capabilities === null || capabilities.blockSpec === null) {
      return {
        spec: null,
        refusal: capabilities?.unitRefusal ?? "the loaded model library predates hook ABI block-sites-v5",
        layerCount: capabilities?.layerCount ?? 0,
        statistics: null,
        scales: null,
        neuronWrite: () => null,
        slensAtom: () => null,
        slensIdentity: () => null,
        prepare: async () => undefined,
        prepareProbe: async () => undefined,
      };
    }
    const session = await this.open();
    const writes = new Map<string, Float32Array>();
    const atoms = new Map<string, { tokenId: number; directions: ReadonlyMap<number, Float32Array> }>();
    let identity: { lens: string; mask: string; means: Float32Array } | null = null;
    const { statistics } = await session.packs.statistics();
    const d = session.context.hiddenSize;
    const prepareSlens = async (word: string) => {
      if (!atoms.has(word)) {
        const tokenId = await this.options.resolveToken(word);
        atoms.set(word, { tokenId, directions: await slensAtomDirections(session.context, word, tokenId) });
      }
      if (identity === null) {
        const lens = session.packs.lens("S");
        const means = session.context.means.out;
        if (lens === null || means === null) return;
        const { maskDigest } = await session.packs.embeddingStatistics();
        identity = { lens: lens.identity, mask: maskDigest, means };
      }
    };
    return {
      spec: session.spec,
      refusal: null,
      layerCount: session.context.nLayers,
      statistics,
      scales: session.scales,
      neuronWrite: (layer, index) => writes.get(`${layer}:${index}`) ?? null,
      slensAtom: (word) => atoms.get(word) ?? null,
      slensIdentity: (layers) => identity === null
        ? null
        : { lens: identity.lens, centre: centreDigest(identity.means, d, layers), mask: identity.mask },
      prepare: async (expression) => {
        if (!expression.trim()) return;
        const parsed = parseSteeringExpression(expression);
        for (const term of parsed.terms) {
          for (const atom of [term.selector.base, term.selector.onto]) {
            if (atom === null) continue;
            if (atom.namespace === "neuron" && !term.ablation) {
              const ref = parseUnit(`neuron/${atom.concept}`, session.spec, session.context.nLayers);
              const key = `${ref.layer}:${ref.index}`;
              if (!writes.has(key)) {
                const rows = await session.backend.unitWeightRows("neuron_write", [{ kind: "neuron", layer: ref.layer, index: ref.index }]);
                writes.set(key, Float32Array.from(rows.data));
              }
            }
            if (atom.namespace === "slens") await prepareSlens(atom.concept);
          }
          const gate = term.trigger?.gate?.probe;
          if (gate?.startsWith("slens/")) await prepareSlens(gate.slice("slens/".length));
        }
      },
      prepareProbe: async (selector) => {
        const trimmed = selector.trim();
        if (trimmed.startsWith("slens/")) await prepareSlens(trimmed.slice("slens/".length));
      },
    };
  }

  private replay(): UnitsNodeReplayPort {
    const port = this.options.replay();
    if (port === null) {
      throw Object.assign(new Error("The loaded browser model cannot replay a loom node"), {
        code: "TOKEN_REPLAY_UNAVAILABLE",
        status: 409,
        recoverable: true,
      });
    }
    return port;
  }

  /** `context.lensProbes`: the session's attached lens probes, which the anatomy envelope scores
   *  over the replay (the instrument runtime passes its compiler's). */
  async request(
    method: UnitsServiceMethod,
    args: readonly unknown[],
    context: { readonly lensProbes?: readonly LensProbe[] } = {},
  ): Promise<unknown> {
    switch (method) {
      case "unitCatalog":
        return this.catalog();
      case "unitLabels":
        return this.labels(args[0]);
      case "unitRead": {
        const options = optionalRecord(args[1], "unit read options");
        const session = await this.open();
        return session.reader.read(text(args[0], "unit"), integer(options.k, 8, "k"), bool(options.hideHubs, false, "hideHubs"));
      }
      case "unitTop": {
        const options = optionalRecord(args[1], "unit top options");
        const layers = routeLayerSelector(options.layers);
        const session = await this.open();
        const by = options.by ?? "writes";
        return session.reader.top(text(args[0], "token"), by as "writes" | "fires", layers, integer(options.k, 20, "k"));
      }
      case "unitFind":
        return this.find(args[0]);
      case "tokenAnatomy":
        return this.anatomy(args[0], args[1], args[2], context.lensProbes ?? []);
      case "unitActivity":
        return this.activity(args[0], args[1]);
    }
  }

  /** Disk only natively (no block description), so it answers on every model. */
  private catalog(): Record<string, unknown> {
    return unitCatalogStatus(this.options.modelId);
  }

  private labels(value: unknown): Record<string, unknown> {
    if (!Array.isArray(value) || value.some((name) => typeof name !== "string")) {
      throw badRequest("units must be a list of unit names");
    }
    const names = value.map((name: string) => name.trim()).filter((name) => name !== "");
    if (names.length === 0) throw badRequest("units names at least one neuron/… or head/… unit");
    if (names.length > MAX_LABEL_UNITS) throw badRequest(`units names at most ${MAX_LABEL_UNITS} units`);
    // Each unit is parsed and checked against the model (native `_parse`); no catalog binds
    // a hosted model, so there are no rows.
    const { spec } = this.requireUnits();
    const layers = this.options.blockSites().capabilities()!.layerCount;
    for (const name of names) parseUnit(name, spec, layers);
    return { catalog: { bound: false, slug: null, credit: null, reason: NO_CATALOG }, labels: [] };
  }

  private async find(value: unknown): Promise<Record<string, unknown>> {
    const body = optionalRecord(value, "unit find body");
    const hasSelector = body.selector !== undefined && body.selector !== null;
    const hasVector = body.vector !== undefined && body.vector !== null;
    if (hasSelector === hasVector) throw badRequest("find takes exactly one of selector or vector");
    if (hasVector && (body.layer === undefined || body.layer === null)) throw badRequest("a vector needs the layer it lives at");
    const k = integer(body.k, 16, "k");
    if (!(k >= 1 && k <= 256)) throw badRequest("k must be in [1, 256]");
    const layers = routeLayerSelector(body.layers);
    const session = await this.open();
    if (hasSelector) return session.reader.find({ selector: text(body.selector, "selector") }, k, layers);
    if (!Array.isArray(body.vector) || body.vector.length === 0 || body.vector.some((x) => typeof x !== "number" || !Number.isFinite(x))) {
      throw badRequest("vector must be a non-empty list of finite numbers");
    }
    const layer = integer(body.layer, 0, "layer");
    if (layer < 0) throw badRequest("layer must be at least 0");
    return session.reader.find({ vector: Float32Array.from(body.vector as number[]), layer }, k, layers);
  }

  private async anatomy(nodeId: unknown, rawIndex: unknown, value: unknown, lensProbes: readonly LensProbe[]): Promise<unknown> {
    const options = optionalRecord(value, "token anatomy options");
    const knobs = {
      k: integer(options.k, 8, "k"),
      units: integer(options.units, 8, "units"),
      patterns: integer(options.patterns, 4, "patterns"),
    };
    checkAnatomyOptions(knobs);
    const index = integer(rawIndex, -1, "rawIndex");
    if (index < 0) throw badRequest("raw_index must be a non-negative integer");
    const session = await this.open();
    const node = await this.replay().nodeForward(text(nodeId, "node id"), index, {
      steered: bool(options.steered, true, "steered"),
      raw: bool(options.raw, false, "raw"),
      label: "session.token_anatomy",
      produced: true,
    });
    const { anatomy, out } = await tokenAnatomy(session.context, { spec: session.spec, scales: session.scales, backend: session.backend }, node.forward, node.program, knobs, {
      sources: session.sources,
      steering: node.steering,
    }, { canThink: node.canThink, gateScores: this.replayGateScores });
    // The attached lens probes read the residual each block wrote, after the steering, as the
    // live capture does (native `build_measurements(lens_readings=...)`).
    const readings = await lensProbeReadings(session.context, out, lensProbes);
    const families = [...new Set(Object.values(readings).map((reading) => reading.unit === UNIT_MEAN_FOOTPRINT_Z ? "S" : "R"))].sort();
    const instruments = Object.keys(readings).length === 0 ? {} : {
      lens: {
        readings,
        binding: { sources: Object.fromEntries(families.map((family) => [family, session.sources[family as "R" | "S"]])), steering: node.steering },
      },
    };
    return {
      measurements: {
        version: 2,
        scope: "replay",
        provenance: "replayed",
        instruments,
        ...(Object.keys(readings).length === 0 ? {} : { scores: Object.fromEntries(Object.entries(readings).map(([name, reading]) => [name, reading.value])) }),
        anatomy,
      },
    };
  }

  private async activity(nodeId: unknown, value: unknown): Promise<unknown> {
    const options = optionalRecord(value, "unit activity options");
    const selector = routeLayerSelector(options.layers);
    const session = await this.open();
    if (session.scales === null) {
      const { absent } = await session.packs.statistics();
      throw Object.assign(new Error(absent ?? "the model has no unit statistics"), { code: "UnitsUnavailableError", status: 422, recoverable: false });
    }
    const top = integer(options.top, 32, "top");
    const rawIndex = options.rawIndex === undefined || options.rawIndex === null ? null : integer(options.rawIndex, 0, "rawIndex");
    if (options.units !== undefined && options.units !== null && (!Array.isArray(options.units) || options.units.some((u) => typeof u !== "string"))) {
      throw badRequest("units must be a list of unit names");
    }
    const units = ((options.units as string[] | undefined) ?? []).map((name) => parseUnit(name.trim(), session.spec, session.context.nLayers));
    checkActivityOptions({ top, units, rawIndex, layers: selector });
    const layers = rawIndex === null ? null : layerRequest(() => resolveModelLayers(selector, session.context.nLayers));
    // The forward anchors at the node's last row, whose producing forward covers every row
    // (native `_last_row`); `rawIndex` only names the population's row.
    const node = await this.replay().nodeForward(text(nodeId, "node id"), null, {
      steered: bool(options.steered, true, "steered"),
      raw: bool(options.raw, false, "raw"),
      label: "session.unit_activity",
      produced: rawIndex !== null,
    });
    if (rawIndex !== null && !(rawIndex >= 0 && rawIndex < node.forward.rawIds.length)) {
      throw badRequest(`raw_index ${rawIndex} is outside node ${String(nodeId)}'s ${node.forward.rawIds.length} tokens`);
    }
    const result = await unitActivity(session.backend, { nLayers: session.context.nLayers, spec: session.spec }, session.scales, node.forward, node.program, {
      top, units, rawIndex, layers,
    }, this.options.tokenTexts, { canThink: node.canThink, gateScores: this.replayGateScores });
    return { ...result, node_id: nodeId, steering: node.steering };
  }
}

export type { NodeReplayForward };
