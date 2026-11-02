// The units members' packs on a loaded model (B6): the core pack's neutral statistics and
// whitener through the steering compiler, the R and S lens packs, the S embedding statistics and
// the units pack through its instrument registry. Every read is the registry's (fetched,
// verified and cached per layer on first use); this file only reshapes them into what the host
// ports read (`UnitsPackSource`).
import type { SerializedMahalanobisWhitener } from "../../fitting/workerContracts";
import type {
  BrowserInstrumentRegistry,
  BrowserLensFamily,
  BrowserNeutralStatistics,
} from "../browserInstrumentPacks";
import { DEFAULT_LENS_BAND, type LensFamilyName } from "./lensReads";
import type { UnitStatistics } from "./scales";
import { pythonRepr } from "./probeDigest";
import { pythonStrip } from "../steeringExpression";
import { InvalidTokenIdError } from "./slens";
import type { UnitLayerArtifacts } from "./unitReader";
import type { UnitsPackLens, UnitsPackSource } from "./unitsService";
import type { LensSite } from "../modelBackend";

/** What the pack source reads off the steering compiler (`BrowserFeasibilityCorePackCompiler`). */
export interface UnitsPackCompiler {
  instrumentRegistry(): BrowserInstrumentRegistry;
  neutralStatistics(signal: AbortSignal): Promise<BrowserNeutralStatistics | null>;
  neutralWhitener(layer: number): SerializedMahalanobisWhitener | null;
}

const SITES: readonly LensSite[] = ["out", "mid"];

function packError(code: string, message: string): Error & { code: string; status: number; recoverable: boolean } {
  return Object.assign(new Error(message), { code, status: 409, recoverable: true });
}

/**
 * `UnitsPackSource` over the session's compiler (`compiler()` is the current one: a manifold
 * refresh replaces it, the registry and core pack stay). `signal` aborts every read when the
 * model unloads.
 */
export function compilerUnitsPackSource(
  compiler: () => UnitsPackCompiler,
  modelId: string,
  signal: AbortSignal,
): UnitsPackSource {
  const registry = () => compiler().instrumentRegistry();
  let kept: { mask: Uint32Array; ids: Promise<Int32Array> } | null = null;
  const embeddingStatistics = () => registry().embeddingStatistics(signal);
  const neutral = async (): Promise<BrowserNeutralStatistics | null> => compiler().neutralStatistics(signal);
  const tensor = (stats: BrowserNeutralStatistics, name: string): Float32Array => {
    const value = stats.tensors.get(name);
    if (value === undefined) {
      throw packError("CORE_NEUTRAL_STATISTICS_INVALID", `The core pack's neutral statistics lack ${name}`);
    }
    return value.data;
  };
  const lens = (family: LensFamilyName): UnitsPackLens | null => {
    const packs = registry();
    const identity = packs.lensIdentity(family as BrowserLensFamily);
    const source = packs.lensSource(family as BrowserLensFamily);
    if (identity === null || source === null) return null;
    return {
      source,
      identity,
      sites: SITES.filter((site) => packs.lensSiteLayers(family, site).length > 0),
      layers: (site) => packs.lensSiteLayers(family, site),
      async fetch(site, layer) {
        const matrix = await packs.lensMatrix(family, site, layer, signal);
        if (matrix === null) {
          throw packError("LENS_LAYER_UNAVAILABLE", `The ${family}-lens pack holds no ${site} matrix at layer ${layer}`);
        }
        return matrix;
      },
    };
  };
  const units = () => {
    const descriptor = registry().unitsPack();
    if (descriptor === null) {
      throw packError("UNITS_PACK_UNAVAILABLE", "Install the units pack for this model to read neurons and heads");
    }
    return descriptor;
  };
  return {
    band: DEFAULT_LENS_BAND,
    async statistics() {
      const stats = await neutral();
      if (stats === null) {
        return {
          statistics: null,
          absent: `units are unavailable for ${modelId}: the core pack carries no neutral unit statistics`,
        };
      }
      const statistics: UnitStatistics = {
        documents: stats.nDocuments,
        neuronMean: tensor(stats, "neuron_mean"),
        neuronScale: tensor(stats, "neuron_scale"),
        headMean: tensor(stats, "head_mean"),
        headScale: tensor(stats, "head_scale"),
      };
      return { statistics, absent: null };
    },
    async means() {
      const stats = await neutral();
      return stats === null
        ? { out: null, mid: null }
        : { out: tensor(stats, "out_mean"), mid: tensor(stats, "mid_mean") };
    },
    lens,
    async embeddingStatistics() {
      const stats = await embeddingStatistics();
      return {
        meanE: stats.meanE,
        sigmaE: stats.sigmaE,
        mask: stats.mask,
        source: stats.maskDescription.source,
        detail: stats.maskDescription.detail,
        maskDigest: stats.maskDigest,
      };
    },
    async layerArtifacts(layer, part): Promise<UnitLayerArtifacts> {
      const tensors = await registry().unitsLayer(part, layer, signal);
      return new Map([...tensors].map(([name, value]) => [name, value.data]));
    },
    async nullProjection(w) {
      const { dModel: d, nullRank: rank } = units();
      const basis = await registry().unitsNullBasis(signal);
      let energy = 0;
      for (let c = 0; c < rank; c += 1) {
        let coordinate = 0;
        for (let i = 0; i < d; i += 1) coordinate += basis[i * rank + c] * w[i];
        energy += coordinate * coordinate;
      }
      return energy;
    },
    async headKeptIds() {
      const descriptor = units();
      const stats = await embeddingStatistics();
      if (kept === null || kept.mask !== stats.mask) {
        // Native `_head_kept`: the mask's kept ids inside the unembedding the R lens reads.
        const ids: number[] = [];
        for (let v = 0; v < Math.min(stats.vocabRows, descriptor.unembeddingRows); v += 1) {
          if ((stats.mask[v >>> 5] >>> (v & 31)) & 1) ids.push(v);
        }
        if (ids.length !== descriptor.headKeptCount) {
          throw packError(
            "UNITS_PACK_BINDING_MISMATCH",
            "The units pack's head vocabulary is not the S token mask's",
          );
        }
        kept = { mask: stats.mask, ids: Promise.resolve(Int32Array.from(ids)) };
      }
      return kept.ids;
    },
    async whitener(layer) {
      return compiler().neutralWhitener(layer);
    },
  };
}

/** The tokenizer ports the units members use, over the loaded engine's tokenizer. */
export interface UnitsTokenizer {
  tokenizeText(text: string): Promise<number[]>;
  /** Each id's own text (`decode([id])`). */
  tokenTexts(tokenIds: readonly number[]): Promise<string[]>;
}

/** Native `resolve_token(word, family="S")` (`resolve_token_atom` bounded by the input
 *  embedding's rows): `#<id>` addresses the id, any other word must be one token whose text,
 *  stripped, is the word (the leading-space piece first, then the bare one). */
export function sTokenResolver(tokenizer: UnitsTokenizer, embeddingRows: () => number): (word: string) => Promise<number> {
  return async (word) => {
    if (word.startsWith("#")) {
      const digits = word.slice(1);
      if (!/^[0-9]+$/u.test(digits)) {
        throw new InvalidTokenIdError(`token-id atom ${pythonRepr(word)} must be '#' followed by a non-negative integer`);
      }
      const tokenId = Number(digits);
      const rows = embeddingRows();
      if (!Number.isSafeInteger(tokenId) || tokenId >= rows) {
        throw new InvalidTokenIdError(`token id ${digits} is outside this vocabulary (size ${rows})`);
      }
      return tokenId;
    }
    let pieces: string[] = [];
    for (const candidate of [` ${word}`, word]) {
      const ids = await tokenizer.tokenizeText(candidate);
      if (ids.length === 1 && pythonStrip((await tokenizer.tokenTexts(ids))[0]) === word) return ids[0];
      if (pieces.length === 0 && ids.length > 1) pieces = await tokenizer.tokenTexts(ids);
    }
    throw Object.assign(
      new Error(
        `${pythonRepr(word)} is not a single token in this vocabulary` +
          (pieces.length > 0 ? ` (pieces: ${pythonRepr({ list: pieces })})` : "") +
          "; the Jacobian lens can only address single-token words",
      ),
      { code: "MultiTokenWordError", status: 400, recoverable: true },
    );
  };
}
