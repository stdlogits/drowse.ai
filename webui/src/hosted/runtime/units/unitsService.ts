// The worker side of drowse 6.0's units members (`instruments.tokenAnatomy`, `unitActivity`,
// `unitRead`, `unitTop`, `unitFind`, `unitCatalog`, `unitLabels`): request parsing as native's
// routes do it, the host ports over the loaded library's block-sites backend, B6's packs, and
// the loom's node replays. A library without unit sites answers native's 422 refusal.
import type { SerializedMahalanobisWhitener } from "../../fitting/workerContracts";
import {
  f32Tensor,
  u32Tensor,
  UnsupportedBlockError,
  type BlockSitesLensLayer,
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
import { parseSteeringExpression } from "../steeringExpression";
import { slensAtomDirections } from "./slens";
import type { BrowserUnitProgramSource } from "./unitProgram";
import { NO_CATALOG, parseUnit, unitCatalogStatus, UnitReader, UnitRequestError, type UnitLayerArtifacts } from "./unitReader";

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
  }>;
  layerArtifacts(layer: number): Promise<UnitLayerArtifacts>;
  nullProjector(): Promise<Float32Array>;
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
  readonly decode: (tokenId: number) => string;
  /** `slens/`-family token resolution (`resolve_token(word, family="S")`). */
  readonly resolveToken: (word: string) => number;
  /** A steerable selector to its per-layer directions (`find`'s selector). */
  readonly resolveSelector?: (selector: string) => Promise<ReadonlyMap<number, Float32Array>>;
  readonly replay: () => UnitsNodeReplayPort | null;
  readonly gateScores?: ReplayGateScores;
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

/** A layer selector as the units routes take it: `all`, `3`, `3,7`, `20-31` (inclusive). */
export function parseLayerSelector(value: unknown, nLayers: number): number[] | null {
  if (value === undefined || value === null || value === "") return null;
  if (typeof value !== "string") throw badRequest("layers must be a layer selector string");
  const spec = value.trim();
  if (spec === "all") return null;
  const out = new Set<number>();
  for (const part of spec.split(",")) {
    const piece = part.trim();
    const range = /^(-?\d+)\s*-\s*(-?\d+)$/.exec(piece);
    if (range) {
      const [a, b] = [Number(range[1]), Number(range[2])];
      if (b < a) throw badRequest(`bad layers ${JSON.stringify(value)}: range ${piece} runs backwards`);
      for (let layer = a; layer <= b; layer += 1) out.add(layer);
    } else if (/^-?\d+$/.test(piece)) {
      out.add(Number(piece));
    } else {
      throw badRequest(`bad layers ${JSON.stringify(value)}: ${JSON.stringify(piece)} is not a layer`);
    }
  }
  const layers = [...out].sort((a, b) => a - b);
  for (const layer of layers) {
    if (!(layer >= 0 && layer < nLayers)) throw badRequest(`layer ${layer} is outside the model's ${nLayers} layers`);
  }
  return layers;
}

/** Resident lens matrices over a pack lens (each uploaded once, on first read). */
export function residentLens(backend: BrowserBlockSitesBackend, family: LensFamilyName, lens: UnitsPackLens, d: number): UnitsLens {
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
  } | null = null;

  constructor(private readonly options: BrowserUnitsServiceOptions) {}

  /** Drop every resident and cached read (a model or pack change). */
  reset(): void {
    this.session = null;
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
    const lensOf = (family: LensFamilyName) => {
      const lens = packs.lens(family);
      return lens === null ? null : residentLens(backend, family, lens, d);
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
      decode: this.options.decode,
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
          return {
            resident: await backend.uploadEmbeddingStatistics(statistics),
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
        layer: (layer) => packs.layerArtifacts(layer),
        nullProjector: () => packs.nullProjector(),
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
    };
    return this.session;
  }

  /** The structured compiler's unit source (`BrowserFeasibilityCorePackCompiler.setUnitSource`)
   *  over the loaded library: its block spec or refusal, the unit statistics, and the
   *  directions `prepare(expression)` reads off the device before a compile (each neuron push's
   *  write, each `slens/` atom's footprints over the S:out layers). */
  async programSource(): Promise<BrowserUnitProgramSource & { prepare(expression: string): Promise<void> }> {
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
        prepare: async () => undefined,
      };
    }
    const session = await this.open();
    const writes = new Map<string, Float32Array>();
    const atoms = new Map<string, { tokenId: number; directions: ReadonlyMap<number, Float32Array> }>();
    const { statistics } = await session.packs.statistics();
    return {
      spec: session.spec,
      refusal: null,
      layerCount: session.context.nLayers,
      statistics,
      scales: session.scales,
      neuronWrite: (layer, index) => writes.get(`${layer}:${index}`) ?? null,
      slensAtom: (word) => atoms.get(word) ?? null,
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
            if (atom.namespace === "slens" && !atoms.has(atom.concept)) {
              const tokenId = this.options.resolveToken(atom.concept);
              atoms.set(atom.concept, { tokenId, directions: await slensAtomDirections(session.context, atom.concept, tokenId) });
            }
          }
        }
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

  async request(method: UnitsServiceMethod, args: readonly unknown[]): Promise<unknown> {
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
        const session = await this.open();
        const by = options.by ?? "writes";
        return session.reader.top(
          text(args[0], "token"),
          by as "writes" | "fires",
          parseLayerSelector(options.layers, session.context.nLayers),
          integer(options.k, 20, "k"),
        );
      }
      case "unitFind":
        return this.find(args[0]);
      case "tokenAnatomy":
        return this.anatomy(args[0], args[1], args[2]);
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
    const session = await this.open();
    const layers = parseLayerSelector(body.layers, session.context.nLayers);
    if (hasSelector) return session.reader.find({ selector: text(body.selector, "selector") }, k, layers);
    if (!Array.isArray(body.vector) || body.vector.length === 0 || body.vector.some((x) => typeof x !== "number" || !Number.isFinite(x))) {
      throw badRequest("vector must be a non-empty list of finite numbers");
    }
    const layer = integer(body.layer, 0, "layer");
    if (layer < 0) throw badRequest("layer must be at least 0");
    return session.reader.find({ vector: Float32Array.from(body.vector as number[]), layer }, k, layers);
  }

  private async anatomy(nodeId: unknown, rawIndex: unknown, value: unknown): Promise<unknown> {
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
    const anatomy = await tokenAnatomy(session.context, { spec: session.spec, scales: session.scales, backend: session.backend }, node.forward, node.program, knobs, {
      sources: session.sources,
      steering: node.steering,
    }, { canThink: node.canThink, ...(this.options.gateScores ? { gateScores: this.options.gateScores } : {}) });
    return {
      measurements: {
        version: 2,
        scope: "replay",
        provenance: "replayed",
        instruments: {},
        anatomy,
      },
    };
  }

  private async activity(nodeId: unknown, value: unknown): Promise<unknown> {
    const options = optionalRecord(value, "unit activity options");
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
    const layers = parseLayerSelector(options.layers, session.context.nLayers);
    checkActivityOptions({ top, units, rawIndex, layers });
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
    }, this.options.decode, { canThink: node.canThink, ...(this.options.gateScores ? { gateScores: this.options.gateScores } : {}) });
    return { ...result, node_id: nodeId, steering: node.steering };
  }
}

export type { NodeReplayForward };
