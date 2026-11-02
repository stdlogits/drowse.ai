import { decodeFp32Safetensors, type Fp32SafetensorsReader } from "../artifacts";
import { sha256 } from "@noble/hashes/sha2.js";
import { bytesToHex } from "@noble/hashes/utils.js";
import type { ProbeInfo } from "../../../../drowse/webui/src/lib/types";
import { runtimeIdentitySha256 } from "../../lib/runtime/catalog";
import { LEGACY_PRODUCT_SLUGS } from "../../../../drowse/webui/src/lib/brandMigration";
import type {
  BrowserInstrumentPackArtifacts,
  BrowserModelArtifact,
  BrowserModelLoadRequest,
} from "./modelBackend";
import type { BrowserSaeFitIdentity } from "../artifacts/fittedAuthoring";
import type {
  BrowserCentroidTransform,
  BrowserFittingLayerCentroids,
} from "../fitting/coordinator";
import { jLensTokenDirection } from "./structuredHookProgram";
import { verifiedSaeDescriptionSource, type SaeDescriptionSource } from "../../../../drowse/webui/src/lib/saeDescriptions";
import type { CatalogFile } from "../../../../drowse/webui/src/lib/runtime/contracts";
import { packFileResidency, packInstallFiles } from "./optionalPackCompatibility";

const LOCAL_SAE_V1_KEYS = [
  "activation", "batch_size", "corpus_sha256", "corpus_spec", "d_model", "d_sae",
  "dead_feature_threshold", "format_version", "kind", "l1_coefficient", "layer",
  "learning_rate", "model_fingerprint", "model_id", "model_source_fingerprint", "name",
  "release", "seq_len", "tensor_file", "tensor_sha256", "tokens_trained",
].sort();
// drowse's lens format v7 (drowse/lens/store.py `_MANIFEST_FIELDS`): one lens
// kind per manifest, its matrices grouped by residual site.
const LOCAL_LENS_V7_KEYS = [
  "checkpoint", "consumed_prefix_sha256", "corpus_hash_kind", "corpus_sha256",
  "corpus_spec", "d_model", "dim_batch", "dtype", "estimator_policy", "format_version",
  "kind", "model_fingerprint", "model_layer_count", "model_source_fingerprint",
  "n_prompts", "raw_corpus_sha256", "raw_prompt_count", "seq_len", "sites",
  "usable_prompt_count",
].sort();
const LOCAL_LENS_V7_SITES = ["out", "mid"];
const LOCAL_LENS_V7_SITE_KEYS = ["layers", "tensor_files", "tensor_sha256"];
// The v7 reader rule frozen with drowse 6.0: a site may also carry its fit's
// `split_half` ({layer: difference}). `odd` belongs only to a checkpoint, so a
// durable manifest (every pack's) that carries it is malformed.
const LOCAL_LENS_V7_OPTIONAL_SITE_KEYS = ["split_half"];
// A lens pack holds one family: the R family (the RelP R or the standard J) in a `jlens` pack,
// the S family in an `slens` pack, each at `out` (required) and `mid`. A full-depth pack holds
// every layer, one shard per (site, layer); the live readout's R:out layers install with the
// pack and stay resident, every other layer is fetched and decoded on its first read.
const FAMILY_KINDS = { R: new Set(["J", "R"]), S: new Set(["S"]) } as const;
const JLENS_DIRECTION_CACHE_MAX_ENTRIES = 64;
const JLENS_DIRECTION_CACHE_MAX_BYTES = 32 * 1024 * 1024;
/** Decoded on-demand lens layers kept in memory across both families (the GPU keeps its own
 *  resident copies); the least recently read go first. */
const LENS_LAYER_CACHE_MAX_BYTES = 256 * 1024 * 1024;
/** Decoded units layer files (reads and contrast statistics) kept in memory. */
const UNITS_LAYER_CACHE_MAX_BYTES = 96 * 1024 * 1024;
const S_EMBEDDING_STATISTICS_FORMAT = "drowse-s-embedding-statistics-v1";
const UNITS_PACK_FORMAT = "drowse-units-pack-v1";

export type BrowserLensFamily = "R" | "S";
export type BrowserLensSite = "out" | "mid";

export interface BrowserSaeFeature {
  readonly key: string;
  readonly layer: number;
  readonly label: string | null;
  readonly maxAct: number | null;
  readonly activation: "relu" | "jump_relu";
  readonly threshold: number;
  readonly direction: Float32Array;
  readonly encoderDirection: Float32Array;
  readonly encoderInputBias: Float32Array;
  readonly featureBias: number;
}

export interface BrowserSaeGpuDictionary {
  readonly bindingId: string;
  readonly hiddenSize: number;
  readonly runtimeLayerIndex: number;
  readonly featureCount: number;
  readonly activation: "relu" | "jump_relu";
  readonly encoder: Float32Array;
  readonly encoderBias: Float32Array;
  readonly encoderThreshold: Float32Array;
  readonly decoderBias: Float32Array;
}

export interface BrowserJlensGpuDictionary {
  readonly bindingId: string;
  readonly hiddenSize: number;
  readonly layerIndices: Int32Array;
  readonly matrices: readonly Float32Array[];
}

export interface BrowserJlensToken {
  readonly key: string;
  readonly word: string;
  readonly tokenId: number;
  readonly directions: ReadonlyMap<number, Float32Array>;
}

interface BrowserSae {
  readonly descriptionSource: SaeDescriptionSource | null;
  readonly source: string;
  readonly displayName: string;
  readonly release: string;
  readonly revision: string;
  readonly fingerprint: string;
  readonly layer: number;
  readonly hiddenSize: number;
  readonly featureCount: number;
  readonly activation: "relu" | "jump_relu";
  readonly featureMetadata: ReadonlyMap<number, {
    readonly label: string | null;
    readonly maxAct: number | null;
  }>;
  readonly encoder: Float32Array;
  readonly decoder: Float32Array;
  readonly encoderBias: Float32Array;
  readonly encoderThreshold: Float32Array;
  readonly encoderInputBias: Float32Array;
  readonly decoderBias: Float32Array;
}

export interface BrowserExactSaeFitting {
  readonly layers: readonly number[];
  readonly transformCentroids: BrowserCentroidTransform;
  readonly provenance: BrowserSaeFitIdentity;
}

export interface BrowserJlensRuntimeResolver {
  tokenizeText(text: string): Promise<number[]>;
  /** The loaded model's vocabulary size; a `jlens/#<id>` id must fall below it. */
  vocabularySize(): number;
  decodeTokens(tokenIds: readonly number[]): Promise<string>;
  resolveJlensTokenDirections(
    bindingId: string,
    layerIndices: readonly number[],
    tokenIds: readonly number[],
  ): Promise<Float32Array>;
  setJlensDictionary?(dictionary: BrowserJlensGpuDictionary | null): Promise<void>;
}

export function saeDecoderDirection(
  decoder: ArrayLike<number>,
  hiddenSize: number,
  feature: number,
): Float32Array {
  if (
    !Number.isSafeInteger(hiddenSize) || hiddenSize < 1 ||
    !Number.isSafeInteger(feature) || feature < 0 ||
    decoder.length % hiddenSize !== 0 || feature >= decoder.length / hiddenSize
  ) {
    throw new TypeError("SAE decoder feature selection is invalid");
  }
  const offset = feature * hiddenSize;
  const direction = Float32Array.from(
    { length: hiddenSize },
    (_, index) => decoder[offset + index],
  );
  if (direction.some((value) => !Number.isFinite(value))) {
    throw new TypeError("SAE decoder direction contains a non-finite value");
  }
  return direction;
}

interface BrowserJlens {
  readonly source: string;
  readonly displayName: string;
  /** The lens kind: the RelP R or the standard J, both of the R family. */
  readonly kind: "J" | "R";
  readonly nPrompts: number;
  /** The fit's split-half difference per out-site layer, when recorded. */
  readonly splitHalf: Readonly<Record<string, number>> | null;
  readonly hiddenSize: number;
  readonly layerMap: readonly number[];
  readonly runtimeLayerMap: readonly number[];
  readonly bindingId: string;
  /** drowse's `lens_identity` of the pack's manifest. */
  readonly identity: string;
  /** Every layer the pack holds, per site (the source row's coverage). */
  readonly sites: ReadonlyMap<string, LensManifestSite>;
  readonly files: PackFiles;
  readonly manifestPath: string;
  /** The live readout's R:out matrices, resident while the pack is active. */
  readonly jacobians: ReadonlyMap<number, Float32Array>;
  readonly words: Map<string, { row: number; tokenId: number }>;
  readonly unembedding: Float32Array;
  readonly cache: Map<string, BrowserJlensToken>;
  readonly resolvedByTokenId: Map<number, ReadonlyMap<number, Float32Array>>;
  readonly pinnedTokenIds: Set<number>;
  resolvedBytes: number;
}

/** One site of an installed lens pack: the layers it holds and the fit's split-half difference
 *  at each, when recorded. */
export interface LensSiteCoverage {
  readonly layers: readonly number[];
  readonly splitHalf: Readonly<Record<string, number>> | null;
}

/** An installed lens pack as its source row describes it: its family and kind, and its coverage
 *  (layers and split-half per site, documents) once the manifest also passes the loader's site
 *  checks. */
export interface LensSourceSummary {
  readonly family: BrowserLensFamily;
  readonly kind: "J" | "R" | "S";
  readonly coverage: {
    readonly sites: Readonly<Partial<Record<BrowserLensSite, LensSiteCoverage>>>;
    readonly nPrompts: number;
  } | null;
}

/** An S pack: its manifest and, decoded on first use, the S embedding statistics. */
interface BrowserSlens {
  readonly source: string;
  readonly displayName: string;
  readonly nPrompts: number;
  readonly identity: string;
  readonly bindingId: string;
  readonly sites: ReadonlyMap<string, LensManifestSite>;
  readonly files: PackFiles;
  readonly manifestPath: string;
  readonly statisticsPath: string;
  /** Python's canonical digest of `embedding-statistics.json`, which a units pack binds. */
  readonly statisticsIdentity: string;
  readonly statisticsRecord: Record<string, unknown>;
  statistics: Promise<BrowserEmbeddingStatistics> | null;
}

/** B1 §5.1's S statistics (`BlockSitesEmbeddingStatistics` before upload). */
export interface BrowserEmbeddingStatistics {
  /** `[d]` mean of the scaled kept embeddings. */
  readonly meanE: Float32Array;
  /** `[d, d]` covariance of the centred kept embeddings, row-major. */
  readonly sigmaE: Float32Array;
  /** Bit `v & 31` of word `v >> 5` is set when token `v` is kept. */
  readonly mask: Uint32Array;
  readonly vocabRows: number;
  readonly embeddingScale: number;
  readonly maskDescription: { readonly source: string; readonly detail: string; readonly kept: number };
  readonly tokenizerFingerprint: string;
}

/** A units pack's model-wide record (`units.json`). */
export interface BrowserUnitsPackDescriptor {
  readonly source: string;
  readonly displayName: string;
  readonly layers: number;
  readonly dModel: number;
  readonly dFF: number;
  readonly nHeads: number;
  readonly headDim: number;
  readonly nullRank: number;
  readonly headKeptCount: number;
  readonly unembeddingRows: number;
  /** What the statistics average over: the R and S lens packs (their `lens_identity`), the S
   *  embedding statistics and the neutral pass (the core pack's `neutral-statistics.json`), whose
   *  rows the write norms' whitener is built from. */
  readonly binding: {
    readonly rLensIdentity: string;
    readonly sLensIdentity: string;
    readonly embeddingStatisticsSha256: string;
    readonly neutralSha256: string;
    readonly neutralRowsSha256: string;
  };
}

/** One units layer file decoded: each tensor at its logical shape (row-major). */
export type BrowserUnitsLayer = ReadonlyMap<string, { readonly shape: readonly number[]; readonly data: Float32Array }>;

interface BrowserUnits {
  readonly descriptor: BrowserUnitsPackDescriptor;
  readonly files: PackFiles;
  readonly root: string;
  readonly nullFile: { readonly path: string; readonly sha256: string };
  /** The core pack's neutral pass (`neutral-statistics.json`), with the rows its whitener was
   *  fitted from, or null without a valid one. */
  readonly coreNeutral: CoreNeutralBinding | null;
  readonly layers: ReadonlyMap<number, Readonly<Record<"reads" | "contrast", {
    readonly path: string;
    readonly sha256: string;
    readonly shapes: Readonly<Record<string, readonly number[]>>;
  }>>>;
  nullBasis: Promise<Float32Array> | null;
}

/** A pack's files by path: an install file from its artifacts, an on-demand layer file through
 *  the worker's fetcher (downloaded into the install and verified on first use). */
class PackFiles {
  private readonly byPath: Map<string, BrowserModelArtifact>;
  private readonly closure: Map<string, CatalogFile>;

  constructor(private readonly entry: BrowserInstrumentPackArtifacts) {
    this.byPath = new Map(entry.artifacts.map((artifact) => [artifact.manifest.path, artifact]));
    this.closure = new Map(entry.pack.files.map((file) => [file.path, file]));
  }

  get displayName(): string {
    return this.entry.pack.displayName;
  }

  /** The closure's digest for `path` (the signed catalog's). */
  sha256(path: string): string {
    const file = this.closure.get(path);
    if (file === undefined) {
      throw instrumentError("INSTRUMENT_PACK_INVALID", `${this.displayName} has no file ${path}`);
    }
    return file.sha256;
  }

  async file(path: string, signal: AbortSignal): Promise<BrowserModelArtifact> {
    signal.throwIfAborted();
    const installed = this.byPath.get(path);
    if (installed !== undefined) return installed;
    const manifest = this.closure.get(path);
    if (manifest === undefined) {
      throw instrumentError("INSTRUMENT_PACK_INVALID", `${this.displayName} has no file ${path}`);
    }
    if (this.entry.fetchFile === undefined) {
      throw instrumentError(
        "INSTRUMENT_PACK_LAYER_UNAVAILABLE",
        `${path} of ${this.displayName} is fetched on first read, and nothing here can fetch it`,
      );
    }
    const file = await this.entry.fetchFile(manifest, signal);
    signal.throwIfAborted();
    if (file.size !== manifest.bytes) {
      throw instrumentError("INSTRUMENT_PACK_ARTIFACT_MISMATCH", `${path} of ${this.displayName} has the wrong size`);
    }
    return { manifest, file };
  }
}

/** Least-recently-read decoded files, bounded by bytes, one decode per key at a time. Readers of
 *  a key share its load; the load (a fetch, possibly) runs while any of them still waits and is
 *  aborted when the last gives up, so nothing outlives the reads that wanted it. */
class DecodedCache<T> {
  private readonly entries = new Map<string, { value: T; bytes: number }>();
  private readonly pending = new Map<string, {
    readonly promise: Promise<T>;
    readonly controller: AbortController;
    waiters: number;
  }>();
  private bytes = 0;

  constructor(private readonly maxBytes: number) {}

  async get(
    key: string,
    signal: AbortSignal,
    load: (signal: AbortSignal) => Promise<{ value: T; bytes: number }>,
  ): Promise<T> {
    signal.throwIfAborted();
    const hit = this.entries.get(key);
    if (hit !== undefined) {
      this.entries.delete(key);
      this.entries.set(key, hit);
      return hit.value;
    }
    const running = this.pending.get(key) ?? this.start(key, load);
    running.waiters += 1;
    return new Promise<T>((resolve, reject) => {
      let waiting = true;
      const leave = () => {
        if (!waiting) return;
        waiting = false;
        signal.removeEventListener("abort", abort);
        running.waiters -= 1;
      };
      const abort = () => {
        leave();
        if (running.waiters === 0) {
          // The last reader left: stop the load and forget it, so a reader that comes next
          // starts its own instead of joining one that is already aborted.
          running.controller.abort(signal.reason);
          if (this.pending.get(key) === running) this.pending.delete(key);
        }
        reject(signal.reason);
      };
      signal.addEventListener("abort", abort, { once: true });
      running.promise.then(
        (value) => { leave(); resolve(value); },
        (error) => { leave(); reject(error); },
      );
    });
  }

  private start(
    key: string,
    load: (signal: AbortSignal) => Promise<{ value: T; bytes: number }>,
  ): { promise: Promise<T>; controller: AbortController; waiters: number } {
    const controller = new AbortController();
    const promise = load(controller.signal).then(({ value, bytes }) => {
      if (controller.signal.aborted) return value;
      const previous = this.entries.get(key);
      if (previous !== undefined) this.bytes -= previous.bytes;
      this.entries.delete(key);
      this.entries.set(key, { value, bytes });
      this.bytes += bytes;
      while (this.bytes > this.maxBytes && this.entries.size > 1) {
        const [oldest, entry] = this.entries.entries().next().value!;
        this.entries.delete(oldest);
        this.bytes -= entry.bytes;
      }
      return value;
    }).finally(() => {
      if (this.pending.get(key) === running) this.pending.delete(key);
    });
    promise.catch(() => undefined);
    const running = { promise, controller, waiters: 0 };
    this.pending.set(key, running);
    return running;
  }

  clear(prefix: string): void {
    for (const [key, entry] of [...this.entries]) {
      if (!key.startsWith(prefix)) continue;
      this.entries.delete(key);
      this.bytes -= entry.bytes;
    }
  }
}

export class BrowserInstrumentRegistry {
  private readonly lensLayers = new DecodedCache<Float32Array>(LENS_LAYER_CACHE_MAX_BYTES);
  private readonly unitsLayers = new DecodedCache<BrowserUnitsLayer>(UNITS_LAYER_CACHE_MAX_BYTES);

  private constructor(
    private readonly sae: BrowserSae | null,
    private readonly jlensPacks: ReadonlyMap<string, BrowserInstrumentPackArtifacts>,
    private readonly jlensSummaries: ReadonlyMap<string, LensSourceSummary | null>,
    private readonly loadRequest: BrowserModelLoadRequest,
    private jlens: BrowserJlens | null,
    private readonly slens: BrowserSlens | null,
    private readonly units: BrowserUnits | null,
  ) {}

  static async load(request: BrowserModelLoadRequest): Promise<BrowserInstrumentRegistry> {
    const optionalPacks = request.optionalPacks ?? [];
    for (const { pack } of optionalPacks) {
      if (pack.runtimeIdentitySha256 !== request.variant.runtimeIdentitySha256) {
        throw instrumentError(
          "INSTRUMENT_PACK_RUNTIME_MISMATCH",
          `${pack.displayName} was built for a different browser model runtime`,
        );
      }
    }
    const saePacks = optionalPacks.filter(({ pack }) => pack.kind === "sae");
    const jlensPacks = optionalPacks.filter(({ pack }) => pack.kind === "jlens");
    const slensPacks = optionalPacks.filter(({ pack }) => pack.kind === "slens");
    const unitsPacks = optionalPacks.filter(({ pack }) => pack.kind === "units");
    for (const [packs, label] of [[saePacks, "SAE"], [slensPacks, "S-lens"], [unitsPacks, "units"]] as const) {
      if (packs.length > 1) {
        throw instrumentError(
          "INSTRUMENT_PACK_AMBIGUOUS",
          `Only one active ${label} pack may be loaded at a time`,
        );
      }
    }
    const jlensBySource = new Map(jlensPacks.map((candidate) => [candidate.pack.id, candidate]));
    const initialJlens = jlensPacks.length === 0 ? null : await loadJlens(jlensPacks[0], request);
    // Every installed source's row carries its coverage, active or not (the Lens room lists them
    // all); the loaded one's comes from what was loaded, the rest from their manifests.
    const summaries = new Map<string, LensSourceSummary | null>();
    for (const candidate of jlensPacks) {
      summaries.set(
        candidate.pack.id,
        initialJlens?.source === candidate.pack.id
          ? lensSummaryOf("R", initialJlens.kind, initialJlens.sites, initialJlens.nPrompts)
          : await lensSourceSummary(candidate, request),
      );
    }
    const slens = slensPacks.length === 1 ? await loadSlens(slensPacks[0], request) : null;
    return new BrowserInstrumentRegistry(
      saePacks.length === 1 ? await loadSae(saePacks[0], request) : null,
      jlensBySource,
      summaries,
      request,
      initialJlens,
      slens,
      unitsPacks.length === 1 ? await loadUnits(unitsPacks[0], request) : null,
    );
  }

  /** Every installed lens source's row, the R packs (one active) and the S pack (active when
   *  installed: one S source per model). */
  jlensSources(): Array<{
    source: string;
    displayName: string;
    active: boolean;
    summary: LensSourceSummary | null;
  }> {
    const rows = [...this.jlensPacks].map(([source, candidate]) => ({
      source,
      displayName: candidate.pack.displayName,
      active: source === this.jlens?.source,
      summary: source === this.jlens?.source
        ? lensSummaryOf("R", this.jlens.kind, this.jlens.sites, this.jlens.nPrompts)
        : this.jlensSummaries.get(source) ?? null,
    }));
    const slens = this.slens;
    return slens === null ? rows : [...rows, {
      source: slens.source,
      displayName: slens.displayName,
      active: true,
      summary: lensSummaryOf("S", "S", slens.sites, slens.nPrompts),
    }];
  }

  async activateJlensSource(source: string): Promise<void> {
    if (this.jlens?.source === source) return;
    const candidate = this.jlensPacks.get(source);
    if (candidate === undefined) {
      throw instrumentError(
        "JLENS_SOURCE_UNAVAILABLE",
        `Lens source ${source} is not installed for this model`,
      );
    }
    const previous = this.jlens;
    this.jlens = await loadJlens(candidate, this.loadRequest);
    if (previous !== null) this.lensLayers.clear(`R:${previous.bindingId}:`);
  }

  capabilities(): { sae: boolean; jlens: boolean; slens: boolean; units: boolean } {
    return {
      sae: this.sae !== null,
      jlens: this.jlens !== null,
      slens: this.slens !== null,
      units: this.units !== null,
    };
  }

  // ---------------------------------------------------------------- lens layers ---

  /** The layers the active `family` pack holds at `site` (empty when none is installed). */
  lensSiteLayers(family: BrowserLensFamily, site: BrowserLensSite): readonly number[] {
    const lens = family === "R" ? this.jlens : this.slens;
    return lens?.sites.get(site)?.layers ?? [];
  }

  /** `family`'s lens identity (`lens_identity` of the pack manifest), or null. */
  lensIdentity(family: BrowserLensFamily): string | null {
    return (family === "R" ? this.jlens : this.slens)?.identity ?? null;
  }

  /**
   * One lens matrix `[d, d]` (R's `J_L` or S's `K_L`, row-major as fitted) at `site`/`layer`:
   * the readout's resident copy, else the layer's shard, fetched and verified against the
   * manifest on first read and kept while memory allows. `null` when the pack holds no such
   * layer, which is that read's absent half. Refuses when the family has no pack.
   */
  async lensMatrix(
    family: BrowserLensFamily,
    site: BrowserLensSite,
    layer: number,
    signal: AbortSignal,
  ): Promise<Float32Array | null> {
    const lens = family === "R" ? this.jlens : this.slens;
    if (lens === null) {
      throw instrumentError(
        family === "R" ? "JLENS_PACK_UNAVAILABLE" : "SLENS_PACK_UNAVAILABLE",
        `Install a compatible ${family}-lens pack for this model`,
      );
    }
    const entry = lens.sites.get(site);
    if (entry === undefined || !entry.layers.includes(layer)) return null;
    if (family === "R" && site === "out") {
      const resident = (lens as BrowserJlens).jacobians.get(layer);
      if (resident !== undefined) return resident;
    }
    const hiddenSize = this.loadRequest.variant.runtimeIdentity.hiddenSize;
    return this.lensLayers.get(`${family}:${lens.bindingId}:${site}:${layer}`, signal, async (shared) => {
      const data = await readLensShard(lens.files, lens.manifestPath, entry, layer, hiddenSize, shared);
      return { value: data, bytes: data.byteLength };
    });
  }

  /** The S embedding statistics of the S pack, decoded on first use. */
  async embeddingStatistics(signal: AbortSignal): Promise<BrowserEmbeddingStatistics> {
    const slens = this.slens;
    if (slens === null) {
      throw instrumentError("SLENS_PACK_UNAVAILABLE", "Install a compatible S-lens pack for this model");
    }
    slens.statistics ??= readEmbeddingStatistics(slens, this.loadRequest, signal).catch((error) => {
      slens.statistics = null;
      throw error;
    });
    return slens.statistics;
  }

  // ---------------------------------------------------------------- units ---

  /** The units pack's record, or null. Its statistics are valid only with the lenses it binds:
   *  `unitsLayer` refuses while another R or S source is active. */
  unitsPack(): BrowserUnitsPackDescriptor | null {
    return this.units?.descriptor ?? null;
  }

  /** The null basis `[d, rank]` (row-major), decoded on first use. */
  async unitsNullBasis(signal: AbortSignal): Promise<Float32Array> {
    const units = this.requireUnits();
    units.nullBasis ??= (async () => {
      const artifact = await units.files.file(units.nullFile.path, signal);
      const reader = await tensorArtifactReader(artifact, signal);
      if (!sameStrings(reader.keys, ["null_basis"])) {
        throw instrumentError("UNITS_PACK_INVALID", "The units null basis file has unexpected tensors");
      }
      return reader.tensor("null_basis", [units.descriptor.dModel, units.descriptor.nullRank]).data;
    })().catch((error) => {
      units.nullBasis = null;
      throw error;
    });
    return units.nullBasis;
  }

  /**
   * One layer's units file, decoded: `reads` (population scalars, head-norm derivatives,
   * whitened write norms) or `contrast` (the `[V]` statistics `hide_hubs` reads), keyed as
   * drowse's `UnitReader.population_artifacts` names them. Fetched on first use.
   */
  async unitsLayer(part: "reads" | "contrast", layer: number, signal: AbortSignal): Promise<BrowserUnitsLayer> {
    const units = this.requireUnits();
    const binding = units.descriptor.binding;
    if (
      this.jlens?.identity !== binding.rLensIdentity || this.slens?.identity !== binding.sLensIdentity ||
      this.slens.statisticsIdentity !== binding.embeddingStatisticsSha256
    ) {
      throw instrumentError(
        "UNITS_PACK_BINDING_MISMATCH",
        "The units pack was built over other R and S lenses than the ones in use",
      );
    }
    if (
      units.coreNeutral?.sha256 !== binding.neutralSha256 ||
      units.coreNeutral.rowsSha256 !== binding.neutralRowsSha256
    ) {
      throw instrumentError(
        "UNITS_PACK_BINDING_MISMATCH",
        "The units pack was built over another neutral pass or whitener than the core pack's",
      );
    }
    const entry = units.layers.get(layer)?.[part];
    if (entry === undefined) {
      throw instrumentError("UNITS_LAYER_UNAVAILABLE", `The units pack has no layer ${layer}`);
    }
    return this.unitsLayers.get(`${part}:${layer}`, signal, async (shared) => {
      if (units.files.sha256(entry.path) !== entry.sha256) {
        throw instrumentError("UNITS_PACK_INVALID", `units ${part} layer ${layer} has a mismatched digest`);
      }
      const reader = await tensorArtifactReader(await units.files.file(entry.path, shared), shared);
      if (!sameStrings(reader.keys, Object.keys(entry.shapes).sort())) {
        throw instrumentError("UNITS_PACK_INVALID", `units ${part} layer ${layer} has unexpected tensors`);
      }
      const tensors = new Map<string, { shape: readonly number[]; data: Float32Array }>();
      let bytes = 0;
      for (const [name, shape] of Object.entries(entry.shapes)) {
        const stored = shape.length > 2
          ? [shape.slice(0, -1).reduce((total, size) => total * size, 1), shape[shape.length - 1]]
          : shape;
        const data = reader.tensor(name, stored).data;
        tensors.set(name, { shape, data });
        bytes += data.byteLength;
      }
      return { value: tensors, bytes };
    });
  }

  private requireUnits(): BrowserUnits {
    if (this.units === null) {
      throw instrumentError("UNITS_PACK_UNAVAILABLE", "Install the units pack for this model to read neurons and heads");
    }
    return this.units;
  }

  descriptor(): {
    sae: { source: string; displayName: string; layer: number; features: number; modelLayers: number[]; descriptionSource: SaeDescriptionSource | null } | null;
    jlens: {
      source: string;
      displayName: string;
      kind: "J" | "R";
      layers: number[];
      nPrompts: number;
      splitHalf: Record<string, number> | null;
      words: Array<{ word: string; tokenId: number }>;
    } | null;
  } {
    return {
      sae: this.sae === null
        ? null
        : {
            source: this.sae.source,
            displayName: this.sae.displayName,
            layer: this.sae.layer,
            features: this.sae.featureCount,
            modelLayers: [...this.loadRequest.variant.runtimeIdentity.layerMap],
            descriptionSource: this.sae.descriptionSource,
          },
      jlens: this.jlens === null
        ? null
        : {
            source: this.jlens.source,
            displayName: this.jlens.displayName,
            kind: this.jlens.kind,
            layers: [...this.jlens.layerMap],
            nPrompts: this.jlens.nPrompts,
            splitHalf: this.jlens.splitHalf === null ? null : { ...this.jlens.splitHalf },
            words: [...this.jlens.words].map(([word, value]) => ({
              word,
              tokenId: value.tokenId,
            })),
          },
    };
  }

  saeGpuDictionary(runtimeLayerMap: readonly number[]): BrowserSaeGpuDictionary | null {
    const sae = this.sae;
    if (sae === null) return null;
    const runtimeLayerIndex = runtimeLayerMap.indexOf(sae.layer);
    if (runtimeLayerIndex < 0) {
      throw instrumentError(
        "SAE_PACK_LAYER_MISMATCH",
        `SAE layer ${sae.layer} is outside the loaded runtime layer map`,
      );
    }
    return {
      bindingId: sae.fingerprint,
      hiddenSize: sae.hiddenSize,
      runtimeLayerIndex,
      featureCount: sae.featureCount,
      activation: sae.activation,
      encoder: sae.encoder,
      encoderBias: sae.encoderBias,
      encoderThreshold: sae.encoderThreshold,
      // The GPU ABI calls the encoder input offset decoderBias.
      decoderBias: sae.encoderInputBias,
    };
  }

  jlensGpuDictionary(
    runtimeLayerMap: readonly number[],
  ): BrowserJlensGpuDictionary | null {
    const jlens = this.jlens;
    if (jlens === null) return null;
    const fitted: Array<{ layerIndex: number; matrix: Float32Array }> = [];
    runtimeLayerMap.forEach((layer, layerIndex) => {
      const matrix = jlens.jacobians.get(layer);
      if (matrix !== undefined) fitted.push({ layerIndex, matrix });
    });
    if (fitted.length !== jlens.layerMap.length) {
      throw instrumentError(
        "JLENS_PACK_LAYER_MISMATCH",
        "The installed R lens's layer map does not match the loaded runtime",
      );
    }
    return {
      bindingId: jlens.bindingId,
      hiddenSize: jlens.hiddenSize,
      layerIndices: Int32Array.from(fitted, ({ layerIndex }) => layerIndex),
      matrices: fitted.map(({ matrix }) => matrix),
    };
  }

  setPinnedJlensWords(words: readonly string[]): void {
    const jlens = this.jlens;
    if (jlens === null) return;
    const pinned = new Set<number>();
    for (const value of words) {
      const word = normalizeJlensWord(value);
      const token = jlens.cache.get(word);
      if (token === undefined) {
        throw instrumentError(
          "JLENS_TOKEN_UNAVAILABLE",
          `Lens token ${JSON.stringify(word)} must be prepared before it is pinned`,
        );
      }
      pinned.add(token.tokenId);
      touchJlensWord(jlens, word, token);
    }
    jlens.pinnedTokenIds.clear();
    for (const tokenId of pinned) jlens.pinnedTokenIds.add(tokenId);
    trimJlensDirectionCache(jlens);
  }

  saeFeatureMetadata(): Record<string, { label: string | null; maxAct: number | null }> {
    const sae = this.sae;
    if (sae === null) return {};
    return Object.fromEntries([...sae.featureMetadata].map(([featureId, metadata]) => [
      String(featureId),
      { ...metadata },
    ]));
  }

  probeHash(
    modelId: string,
    probe: Exclude<ProbeInfo, { family: "geometry" }>,
  ): string {
    if (probe.family === "units") {
      throw instrumentError(
        "INSTRUMENT_PACK_UNAVAILABLE",
        `Unit probe ${probe.name} needs unit reads, which this browser runtime does not have`,
      );
    }
    if (probe.family === "lens") {
      if (probe.token_id === null) {
        throw instrumentError(
          "JLENS_TOKEN_UNAVAILABLE",
          `Lens probe ${probe.name} has no resolved token identity`,
        );
      }
      if (probe.namespace === "slens" || this.jlens === null) {
        throw instrumentError(
          "JLENS_PACK_UNAVAILABLE",
          `Lens probe ${probe.name} needs ${probe.namespace === "slens" ? "an S" : "an R"} lens, which is not loaded`,
        );
      }
      // drowse's `LensInstrument.probe_hash`: the readout channel binds the
      // resident lens, so the same word through another lens is another probe.
      return pythonTupleDigest([
        "jlens-readout-v5",
        modelId,
        probe.word,
        probe.token_id,
        [...probe.layers],
        this.jlens.identity,
      ]);
    }
    const sae = this.sae;
    if (sae === null || probe.layers.length !== 1 || probe.layers[0] !== sae.layer) {
      throw instrumentError(
        "SAE_PACK_UNAVAILABLE",
        `SAE probe ${probe.name} does not match the active browser SAE`,
      );
    }
    return pythonTupleDigest([
      "sae-readout-v2",
      modelId,
      sae.fingerprint,
      sae.release,
      sae.layer,
      probe.feature_id,
      sae.featureMetadata.get(probe.feature_id)?.maxAct === null ||
          sae.featureMetadata.get(probe.feature_id)?.maxAct === undefined
        ? null
        : { pythonFloat: sae.featureMetadata.get(probe.feature_id)!.maxAct! },
    ]);
  }

  saeFeature(value: string): BrowserSaeFeature {
    const sae = this.sae;
    if (sae === null) {
      throw instrumentError("SAE_PACK_UNAVAILABLE", "Install the compatible SAE pack to use sae/<feature>");
    }
    const feature = Number(value);
    if (!Number.isSafeInteger(feature) || feature < 0 || feature >= sae.featureCount) {
      throw instrumentError(
        "SAE_FEATURE_UNAVAILABLE",
        `SAE feature ${value} is outside the installed pack's 0-${sae.featureCount - 1} range`,
      );
    }
    const encoderDirection = new Float32Array(sae.hiddenSize);
    for (let index = 0; index < sae.hiddenSize; index += 1) {
      encoderDirection[index] = sae.encoder[index * sae.featureCount + feature];
    }
    return {
      key: `sae/${feature}`,
      layer: sae.layer,
      label: sae.featureMetadata.get(feature)?.label ?? null,
      maxAct: sae.featureMetadata.get(feature)?.maxAct ?? null,
      activation: sae.activation,
      threshold: sae.encoderThreshold[feature],
      direction: saeDecoderDirection(sae.decoder, sae.hiddenSize, feature),
      encoderDirection,
      encoderInputBias: sae.encoderInputBias,
      featureBias: sae.encoderBias[feature],
    };
  }

  exactSaeFitting(selector: string): BrowserExactSaeFitting | null {
    const sae = this.sae;
    if (sae === null || selector !== sae.source && selector !== sae.release) return null;
    const idsByLayer = new Map([[sae.layer, `${sae.release}:layer-${sae.layer}`]]);
    return {
      layers: [sae.layer],
      transformCentroids: (layer, centroids) =>
        reconstructSaeCentroids(sae, layer, centroids),
      provenance: {
        release: sae.release,
        revision: sae.revision,
        fingerprint: sae.fingerprint,
        idsByLayer,
        fullCoverage: false,
      },
    };
  }

  async prepareJlensWords(
    words: readonly string[],
    resolver: BrowserJlensRuntimeResolver,
  ): Promise<void> {
    const requested = [...new Set(words.map(normalizeJlensWord))];
    if (requested.length === 0) return;
    if (requested.length > JLENS_DIRECTION_CACHE_MAX_ENTRIES) {
      throw instrumentError(
        "JLENS_PREPARATION_LIMIT",
        `Prepare at most ${JLENS_DIRECTION_CACHE_MAX_ENTRIES} lens words at once`,
      );
    }
    const jlens = this.jlens;
    if (jlens === null) {
      throw instrumentError(
        "JLENS_PACK_UNAVAILABLE",
        "Install a compatible R-lens pack to use jlens/<word>",
      );
    }
    const tokenIds = new Map<string, number>();
    for (const word of requested) {
      const tokenId = await resolveJlensWordToken(resolver, word);
      const curated = jlens.words.get(word);
      if (curated !== undefined && curated.tokenId !== tokenId) {
        throw instrumentError(
          "JLENS_VOCABULARY_INVALID",
          `The installed R lens's vocabulary token for ${JSON.stringify(word)} does not match the loaded tokenizer`,
        );
      }
      tokenIds.set(word, tokenId);
    }
    const uniqueIds = [...new Set(tokenIds.values())].filter(
      (tokenId) => !jlens.resolvedByTokenId.has(tokenId),
    );
    if (uniqueIds.length > 0) {
      const layerIndices = jlens.layerMap.map((layer) =>
        jlens.runtimeLayerMap.indexOf(layer)
      );
      for (let start = 0; start < uniqueIds.length; start += 8) {
        const batch = uniqueIds.slice(start, start + 8);
        const values = await resolver.resolveJlensTokenDirections(
          jlens.bindingId,
          layerIndices,
          batch,
        );
        const expected = jlens.layerMap.length * batch.length * jlens.hiddenSize;
        if (
          !(values instanceof Float32Array) || values.length !== expected ||
          values.some((value) => !Number.isFinite(value))
        ) {
          throw instrumentError(
            "JLENS_DIRECTION_INVALID",
            "The loaded model runtime returned invalid R-lens token directions",
          );
        }
        batch.forEach((tokenId, tokenIndex) => {
          const directions = new Map<number, Float32Array>();
          jlens.layerMap.forEach((layer, layerIndex) => {
            const offset = (layerIndex * batch.length + tokenIndex) * jlens.hiddenSize;
            directions.set(layer, values.slice(offset, offset + jlens.hiddenSize));
          });
          storeJlensDirections(jlens, tokenId, directions);
        });
      }
    }
    for (const [word, tokenId] of tokenIds) {
      const directions = touchJlensDirections(jlens, tokenId)!;
      touchJlensWord(jlens, word, {
        key: `jlens/${word}`,
        word,
        tokenId,
        directions,
      });
    }
    trimJlensDirectionCache(jlens);
  }

  jlensToken(word: string): BrowserJlensToken {
    const jlens = this.jlens;
    if (jlens === null) {
      throw instrumentError("JLENS_PACK_UNAVAILABLE", "Install a compatible R-lens pack to use jlens/<word>");
    }
    const normalized = normalizeJlensWord(word);
    const cached = jlens.cache.get(normalized);
    if (cached !== undefined) {
      touchJlensWord(jlens, normalized, cached);
      touchJlensDirections(jlens, cached.tokenId);
      return cached;
    }
    const entry = jlens.words.get(normalized);
    if (entry === undefined) {
      throw instrumentError(
        "JLENS_TOKEN_UNAVAILABLE",
        `Lens token ${JSON.stringify(normalized)} has not been prepared with the loaded tokenizer`,
      );
    }
    const resolved = touchJlensDirections(jlens, entry.tokenId);
    let directions: ReadonlyMap<number, Float32Array>;
    if (resolved === undefined) {
      const unembedding = jlens.unembedding.subarray(
        entry.row * jlens.hiddenSize,
        (entry.row + 1) * jlens.hiddenSize,
      );
      const fallback = new Map<number, Float32Array>();
      for (const layer of jlens.layerMap) {
        fallback.set(layer, jLensTokenDirection(unembedding, jlens.jacobians.get(layer)!));
      }
      directions = fallback;
      storeJlensDirections(jlens, entry.tokenId, directions);
    } else {
      directions = resolved;
    }
    const token = {
      key: `jlens/${normalized}`,
      word: normalized,
      tokenId: entry.tokenId,
      directions,
    };
    touchJlensWord(jlens, normalized, token);
    trimJlensDirectionCache(jlens);
    return token;
  }

  jlensProgram(
    probeNames: readonly string[],
    runtimeLayerMap: readonly number[],
    probeCount: number,
    includeFullVocabularyReadout = false,
  ): { bindingId: string; layerIndices: Int32Array; tokenIds: Int32Array } | null {
    const names = probeNames.filter((name) => name.startsWith("jlens/"));
    if (names.length === 0 && !includeFullVocabularyReadout) return null;
    const jlens = this.jlens;
    if (jlens === null) {
      throw instrumentError("JLENS_PACK_UNAVAILABLE", "Install a compatible R-lens pack to use probability readouts");
    }
    const fitted: number[] = [];
    runtimeLayerMap.forEach((layer, programIndex) => {
      if (!jlens.jacobians.has(layer)) return;
      fitted.push(programIndex);
    });
    if (fitted.length === 0) {
      throw instrumentError(
        "JLENS_PACK_LAYER_MISMATCH",
        "The installed R lens has no fitted layer in the loaded runtime",
      );
    }
    const layerIndices = Int32Array.from(fitted);
    const tokenIds = new Int32Array(probeCount);
    probeNames.forEach((name, probeIndex) => {
      if (!name.startsWith("jlens/")) return;
      tokenIds[probeIndex] = this.jlensToken(name.slice("jlens/".length)).tokenId;
    });
    return { bindingId: jlens.bindingId, layerIndices, tokenIds };
  }
}

async function loadSae(
  pack: BrowserInstrumentPackArtifacts,
  request: BrowserModelLoadRequest,
): Promise<BrowserSae> {
  assertArtifactClosure(pack, request);
  const manifestArtifact = uniqueArtifact(pack.artifacts, (artifact) =>
    artifact.manifest.path.endsWith("/manifest.json"), "SAE manifest");
  const manifest = await jsonArtifact(manifestArtifact, request.signal);
  const hiddenSize = request.variant.runtimeIdentity.hiddenSize;
  if (
    !sameStrings(Object.keys(manifest).filter((key) => key !== "apply_b_dec_to_input").sort(), LOCAL_SAE_V1_KEYS) ||
    ("apply_b_dec_to_input" in manifest && typeof manifest.apply_b_dec_to_input !== "boolean") ||
    !(
      manifest.format_version === 1 && manifest.activation === "relu" ||
      manifest.format_version === 2 && manifest.activation === "jump_relu"
    ) || manifest.kind !== "local" ||
    !nonnegativeInteger(manifest.layer) || !positiveInteger(manifest.d_model) ||
    !positiveInteger(manifest.d_sae) || manifest.d_model !== hiddenSize ||
    manifest.model_id !== request.variant.runtimeIdentity.sourceModel ||
    !matchesRuntimeFingerprint(manifest.model_fingerprint, request) ||
    manifest.model_source_fingerprint !== request.variant.runtimeIdentity.sourceRevision ||
    typeof manifest.release !== "string" ||
    !/^local:[A-Za-z][A-Za-z0-9_-]*$/u.test(manifest.release) ||
    typeof manifest.tensor_file !== "string" || !safeBasename(manifest.tensor_file) ||
    typeof manifest.tensor_sha256 !== "string" || !/^[a-f0-9]{64}$/u.test(manifest.tensor_sha256)
  ) {
    throw instrumentError("SAE_PACK_INVALID", "The SAE pack does not match the local v1 fp32 schema");
  }
  if (!request.variant.runtimeIdentity.layerMap.includes(manifest.layer)) {
    throw instrumentError("SAE_PACK_LAYER_MISMATCH", `SAE layer ${manifest.layer} is outside this model's layer map`);
  }
  const tensorArtifact = artifactAt(pack.artifacts, siblingPath(
    manifestArtifact.manifest.path,
    manifest.tensor_file,
  ));
  if (tensorArtifact.manifest.sha256 !== manifest.tensor_sha256) {
    throw instrumentError("SAE_PACK_INVALID", "The SAE tensor digest does not match its manifest");
  }
  const tensors = await tensorArtifactReader(tensorArtifact, request.signal);
  const expectedKeys = manifest.activation === "jump_relu"
    ? ["W_dec", "W_enc", "b_dec", "b_enc", "threshold"]
    : ["W_dec", "W_enc", "b_dec", "b_enc"];
  if (!sameStrings(tensors.keys, expectedKeys)) {
    throw instrumentError(
      "SAE_PACK_INVALID",
      manifest.activation === "jump_relu"
        ? "The JumpReLU SAE tensor must contain W_enc, W_dec, b_enc, b_dec, and threshold"
        : "The SAE tensor must contain exactly W_enc, W_dec, b_enc, and b_dec",
    );
  }
  const featureMetadataArtifacts = pack.artifacts.filter((artifact) =>
    artifact.manifest.path.endsWith("/features.json")
  );
  if (featureMetadataArtifacts.length > 1) {
    throw instrumentError(
      "SAE_PACK_INVALID",
      "The SAE pack may contain at most one feature metadata file",
    );
  }
  const featureMetadata = featureMetadataArtifacts.length === 0
    ? new Map<number, { label: string | null; maxAct: number | null }>()
    : await loadSaeFeatureMetadata(
        featureMetadataArtifacts[0],
        request.signal,
        manifest.model_id,
        manifest.release,
        manifest.d_sae,
      );
  const gemmaScope2 = manifest.activation === "jump_relu" &&
    typeof manifest.corpus_spec === "string" &&
    /^provider:google\/gemma-scope-2-(270m|1b|4b|12b|27b)-(it|pt)$/u.test(manifest.corpus_spec);
  const applyDecoderBias = manifest.apply_b_dec_to_input ?? (gemmaScope2 ? false : undefined);
  if (applyDecoderBias === undefined && String(manifest.corpus_spec).startsWith("provider:")) {
    throw instrumentError("SAE_PACK_INVALID", "Provider SAE packs must declare apply_b_dec_to_input");
  }
  const decoderBias = tensors.tensor("b_dec", [hiddenSize]).data;
  const centered = applyDecoderBias ?? true;
  return {
    source: pack.pack.id,
    descriptionSource: verifiedSaeDescriptionSource(manifest),
    displayName: pack.pack.displayName,
    release: manifest.release,
    revision: manifest.tensor_sha256,
    fingerprint: centered ? manifest.tensor_sha256 : bytesToHex(sha256(new TextEncoder().encode(
      `sae-uncentered-v1:${manifest.tensor_sha256}`,
    ))),
    layer: manifest.layer,
    hiddenSize,
    featureCount: manifest.d_sae,
    activation: manifest.activation,
    featureMetadata,
    encoder: tensors.tensor("W_enc", [hiddenSize, manifest.d_sae]).data,
    decoder: tensors.tensor("W_dec", [manifest.d_sae, hiddenSize]).data,
    encoderBias: tensors.tensor("b_enc", [manifest.d_sae]).data,
    encoderThreshold: manifest.activation === "jump_relu"
      ? tensors.tensor("threshold", [manifest.d_sae]).data
      : new Float32Array(manifest.d_sae),
    encoderInputBias: centered ? decoderBias : new Float32Array(hiddenSize),
    decoderBias,
  };
}

async function loadSaeFeatureMetadata(
  artifact: BrowserModelArtifact,
  signal: AbortSignal,
  modelId: string,
  release: string,
  featureCount: number,
): Promise<Map<number, { label: string | null; maxAct: number | null }>> {
  const payload = await jsonArtifact(artifact, signal);
  if (
    !sameStrings(Object.keys(payload).sort(), [
      "features", "format_version", "model_id", "release",
    ]) ||
    payload.format_version !== 1 || payload.model_id !== modelId ||
    payload.release !== release || !isRecord(payload.features)
  ) {
    throw instrumentError(
      "SAE_FEATURE_METADATA_INVALID",
      "The SAE feature metadata does not match the browser v1 schema",
    );
  }
  const metadata = new Map<number, { label: string | null; maxAct: number | null }>();
  for (const [key, value] of Object.entries(payload.features)) {
    const featureId = Number(key);
    if (
      !/^(0|[1-9]\d*)$/u.test(key) || !Number.isSafeInteger(featureId) ||
      featureId < 0 || featureId >= featureCount || !isRecord(value) ||
      !sameStrings(Object.keys(value).sort(), ["label", "max_act"])
    ) {
      throw instrumentError(
        "SAE_FEATURE_METADATA_INVALID",
        "The SAE feature metadata contains an invalid feature row",
      );
    }
    const label = value.label;
    const maxAct = value.max_act;
    if (
      label !== null && (typeof label !== "string" || label.trim() === "") ||
      maxAct !== null && (typeof maxAct !== "number" || !Number.isFinite(maxAct) || maxAct <= 0)
    ) {
      throw instrumentError(
        "SAE_FEATURE_METADATA_INVALID",
        `SAE feature ${featureId} has invalid label or max_act metadata`,
      );
    }
    metadata.set(featureId, { label, maxAct });
  }
  return metadata;
}

function reconstructSaeCentroids(
  sae: BrowserSae,
  layer: number,
  centroids: BrowserFittingLayerCentroids,
): BrowserFittingLayerCentroids {
  if (
    layer !== sae.layer || centroids.columns !== sae.hiddenSize ||
    !Number.isSafeInteger(centroids.rows) || centroids.rows < 1 ||
    centroids.values.length !== centroids.rows * centroids.columns ||
    centroids.values.some((value) => !Number.isFinite(value))
  ) {
    throw instrumentError(
      "SAE_FITTING_INPUT_INVALID",
      `The installed SAE cannot transform layer ${layer} centroids`,
    );
  }
  const reconstructed = new Float64Array(centroids.values.length);
  for (let row = 0; row < centroids.rows; row += 1) {
    const offset = row * sae.hiddenSize;
    reconstructed.set(sae.decoderBias, offset);
    for (let feature = 0; feature < sae.featureCount; feature += 1) {
      let activation = sae.encoderBias[feature];
      for (let column = 0; column < sae.hiddenSize; column += 1) {
        activation += (
          centroids.values[offset + column] - sae.encoderInputBias[column]
        ) * sae.encoder[column * sae.featureCount + feature];
      }
      if (activation <= sae.encoderThreshold[feature]) continue;
      const decoderOffset = feature * sae.hiddenSize;
      for (let column = 0; column < sae.hiddenSize; column += 1) {
        reconstructed[offset + column] += activation * sae.decoder[decoderOffset + column];
      }
    }
  }
  if (reconstructed.some((value) => !Number.isFinite(value))) {
    throw instrumentError(
      "SAE_FITTING_RESULT_INVALID",
      "The installed SAE produced non-finite reconstructed centroids",
    );
  }
  return {
    rows: centroids.rows,
    columns: centroids.columns,
    values: reconstructed,
  };
}

/** One site of a lens v7 manifest: its layers, their shards, and the fit's
 *  split-half difference per layer when the fit recorded one. */
export interface LensManifestSite {
  readonly layers: number[];
  readonly tensorFiles: Readonly<Record<string, string>>;
  readonly tensorDigests: Readonly<Record<string, string>>;
  readonly splitHalf: Readonly<Record<string, number>> | null;
}

/**
 * The sites of a durable lens v7 manifest under the reader rule drowse 6.0
 * froze (`store.py::_validate_manifest`): each site holds exactly the
 * required keys plus, optionally, `split_half` (every layer, finite and
 * non-negative); `odd` is a checkpoint's and never valid here. `null` when
 * any site is malformed.
 */
export function lensManifestSites(
  manifest: Record<string, unknown>,
): Map<string, LensManifestSite> | null {
  const sites = manifest.sites;
  if (
    !isRecord(sites) || Object.keys(sites).length === 0 ||
    Object.keys(sites).some((site) => !LOCAL_LENS_V7_SITES.includes(site))
  ) return null;
  const parsed = new Map<string, LensManifestSite>();
  for (const name of LOCAL_LENS_V7_SITES) {
    if (!(name in sites)) continue;
    const site = sites[name];
    if (!isRecord(site)) return null;
    const keys = Object.keys(site);
    if (
      LOCAL_LENS_V7_SITE_KEYS.some((key) => !keys.includes(key)) ||
      keys.some((key) => !LOCAL_LENS_V7_SITE_KEYS.includes(key) &&
        !LOCAL_LENS_V7_OPTIONAL_SITE_KEYS.includes(key))
    ) return null;
    const { layers, tensor_files: tensorFiles, tensor_sha256: tensorDigests } = site;
    if (!integerList(layers) || !isRecord(tensorFiles) || !isRecord(tensorDigests)) return null;
    const layerKeys = layers.map(String).sort();
    if (
      !sameStrings(Object.keys(tensorFiles).sort(), layerKeys) ||
      !sameStrings(Object.keys(tensorDigests).sort(), layerKeys) ||
      Object.values(tensorFiles).some((file) => typeof file !== "string" || !safeBasename(file)) ||
      Object.values(tensorDigests).some((digest) =>
        typeof digest !== "string" || !/^[0-9a-f]{64}$/u.test(digest))
    ) return null;
    let splitHalf: Record<string, number> | null = null;
    if ("split_half" in site) {
      const values = site.split_half;
      if (
        !isRecord(values) || !sameStrings(Object.keys(values).sort(), layerKeys) ||
        Object.values(values).some((value) =>
          typeof value !== "number" || !Number.isFinite(value) || value < 0)
      ) return null;
      splitHalf = values as Record<string, number>;
    }
    parsed.set(name, {
      layers,
      tensorFiles: tensorFiles as Record<string, string>,
      tensorDigests: tensorDigests as Record<string, string>,
      splitHalf,
    });
  }
  return parsed;
}

/**
 * drowse's `lens_identity` (`lens/store.py`) of a manifest file's text: the
 * sha256 of `json.dumps(json.loads(text), sort_keys=True, separators=(",", ":"))`.
 * `JSON.parse` loses Python's int/float split, so this reads the text itself
 * and takes each number's type from its spelling, as `json.loads` does: a
 * fraction or an exponent makes a float, anything else an int.
 */
export function lensIdentity(manifestText: string): string {
  return bytesToHex(sha256(new TextEncoder().encode(pythonCanonicalJson(manifestText))));
}

function pythonCanonicalJson(text: string): string {
  let index = 0;
  const skipSpace = () => {
    while (index < text.length && " \t\n\r".includes(text[index])) index += 1;
  };
  const expect = (character: string) => {
    skipSpace();
    if (text[index] !== character) invalidManifestJson();
    index += 1;
  };
  const string = (): string => {
    skipSpace();
    const start = index;
    if (text[index] !== '"') invalidManifestJson();
    index += 1;
    while (index < text.length && text[index] !== '"') index += text[index] === "\\" ? 2 : 1;
    if (index >= text.length) invalidManifestJson();
    index += 1;
    try {
      return JSON.parse(text.slice(start, index)) as string;
    } catch {
      return invalidManifestJson();
    }
  };
  // Each container ends at its closing bracket or continues past a comma.
  const more = (close: string): boolean => {
    skipSpace();
    if (text[index] === close) {
      index += 1;
      return false;
    }
    expect(",");
    return true;
  };
  const value = (): string => {
    skipSpace();
    const character = text[index];
    if (character === "{") {
      index += 1;
      skipSpace();
      const entries = new Map<string, string>();
      if (text[index] === "}") index += 1;
      else {
        do {
          const key = string();
          expect(":");
          entries.set(key, value());
        } while (more("}"));
      }
      return `{${[...entries.keys()].sort(compareCodePoints).map((key) =>
        `${pythonJsonString(key)}:${entries.get(key)}`).join(",")}}`;
    }
    if (character === "[") {
      index += 1;
      skipSpace();
      const items: string[] = [];
      if (text[index] === "]") index += 1;
      else {
        do items.push(value());
        while (more("]"));
      }
      return `[${items.join(",")}]`;
    }
    if (character === '"') return pythonJsonString(string());
    for (const literal of ["true", "false", "null"]) {
      if (text.startsWith(literal, index)) {
        index += literal.length;
        return literal;
      }
    }
    const number = /-?(?:0|[1-9]\d*)(\.\d+)?([eE][-+]?\d+)?/uy;
    number.lastIndex = index;
    const match = number.exec(text);
    if (match === null) return invalidManifestJson();
    index = number.lastIndex;
    return match[1] === undefined && match[2] === undefined
      ? BigInt(match[0]).toString()
      : pythonNumberRepr(Number(match[0]), true);
  };
  const canonical = value();
  skipSpace();
  if (index !== text.length) invalidManifestJson();
  return canonical;
}

function invalidManifestJson(): never {
  throw instrumentError("INSTRUMENT_PROBE_IDENTITY_INVALID", "The lens manifest is not JSON Python reads");
}

// Python orders str keys by code point; `Array.prototype.sort` by UTF-16 unit.
function compareCodePoints(left: string, right: string): number {
  const a = Array.from(left, (character) => character.codePointAt(0)!);
  const b = Array.from(right, (character) => character.codePointAt(0)!);
  for (let index = 0; index < Math.min(a.length, b.length); index += 1) {
    if (a[index] !== b[index]) return a[index] - b[index];
  }
  return a.length - b.length;
}

// json.dumps's ensure_ascii string form: the short escapes, then \uXXXX
// (lowercase, UTF-16) for every character outside printable ASCII.
function pythonJsonString(value: string): string {
  const short: Record<string, string> = {
    '"': '\\"', "\\": "\\\\", "\n": "\\n", "\r": "\\r", "\t": "\\t", "\b": "\\b", "\f": "\\f",
  };
  let output = '"';
  for (let index = 0; index < value.length; index += 1) {
    const character = value[index];
    const code = value.charCodeAt(index);
    output += short[character] ??
      (code >= 0x20 && code <= 0x7e ? character : `\\u${code.toString(16).padStart(4, "0")}`);
  }
  return `${output}"`;
}

/** The `out` site of a hosted lens pack's manifest, under every check the loader makes: drowse's
 *  v7 fp32 format, an R-family kind (J or R), this model's fingerprint and depth, and layers
 *  inside its layer map. */
/** A v7 manifest of `family` bound to the loaded runtime, under the loader's checks; its sites.
 *  `out` is required (an R pack's live readout reads it); `mid` is optional (pre-6.0 packs). */
function browserLensSites(
  manifest: Record<string, unknown>,
  request: BrowserModelLoadRequest,
  family: BrowserLensFamily,
): Map<string, LensManifestSite> {
  const sites = lensManifestSites(manifest);
  const code = family === "R" ? "JLENS_PACK_INVALID" : "SLENS_PACK_INVALID";
  if (
    sites === null || !sites.has("out") ||
    !sameStrings(Object.keys(manifest).sort(), LOCAL_LENS_V7_KEYS) ||
    manifest.format_version !== 7 || !FAMILY_KINDS[family].has(manifest.kind as string) ||
    manifest.checkpoint !== false || manifest.consumed_prefix_sha256 !== null ||
    manifest.dtype !== "float32" || manifest.d_model !== request.variant.runtimeIdentity.hiddenSize ||
    !nonnegativeInteger(manifest.n_prompts) ||
    !matchesRuntimeFingerprint(manifest.model_fingerprint, request) ||
    manifest.model_source_fingerprint !== request.variant.runtimeIdentity.sourceRevision ||
    manifest.model_layer_count !== request.variant.runtimeIdentity.layerMap.length
  ) {
    throw instrumentError(
      code,
      `The lens pack is not an ${family}-family lens in drowse's v7 fp32 format with an out site`,
    );
  }
  for (const site of sites.values()) {
    if (site.layers.some((layer) => !request.variant.runtimeIdentity.layerMap.includes(layer))) {
      throw instrumentError(
        family === "R" ? "JLENS_PACK_LAYER_MISMATCH" : "SLENS_PACK_INVALID",
        `The ${family} lens has layers outside this model's layer map`,
      );
    }
  }
  return sites;
}

function lensSummaryOf(
  family: BrowserLensFamily,
  kind: "J" | "R" | "S",
  sites: ReadonlyMap<string, LensManifestSite>,
  nPrompts: number,
): LensSourceSummary {
  const coverage: Partial<Record<BrowserLensSite, LensSiteCoverage>> = {};
  for (const name of ["out", "mid"] as const) {
    const site = sites.get(name);
    if (site === undefined) continue;
    coverage[name] = {
      layers: [...site.layers],
      splitHalf: site.splitHalf === null ? null : { ...site.splitHalf },
    };
  }
  return { family, kind, coverage: { sites: coverage, nPrompts } };
}

/** What a lens source row says about an installed R pack (6.0's `InstrumentSourceJSON` fields),
 *  read from its manifest alone under the loader's checks. Native's row for a manifest that no
 *  longer validates keeps its kind (from the local name) and drops its coverage; here the kind
 *  is known once the pack's files verify and its manifest names an R-family kind (J or R), so a
 *  manifest that fails only the site checks keeps it (`coverage: null`). `null` when even the
 *  kind cannot be read: a hosted R pack selects into the R slot, so a K or S manifest names no
 *  kind it could serve. */
async function lensSourceSummary(
  pack: BrowserInstrumentPackArtifacts,
  request: BrowserModelLoadRequest,
): Promise<LensSourceSummary | null> {
  let kind: "J" | "R" | null = null;
  try {
    assertArtifactClosure(pack, request);
    const manifestArtifact = uniqueArtifact(pack.artifacts, (artifact) =>
      artifact.manifest.path.endsWith("/manifest.json"), "Lens manifest");
    const manifest = jsonObject(
      manifestArtifact,
      await textArtifact(manifestArtifact, request.signal),
    );
    if (manifest.kind !== "J" && manifest.kind !== "R") return null;
    kind = manifest.kind;
    return lensSummaryOf("R", kind, browserLensSites(manifest, request, "R"), manifest.n_prompts as number);
  } catch {
    // An unreadable inactive pack must not fail the model load; selecting it refuses loudly.
    request.signal.throwIfAborted();
    return kind === null ? null : { family: "R", kind, coverage: null };
  }
}

/** One (site, layer) matrix of a lens pack: its shard (one file per layer, or a pre-6.0 file
 *  holding several), checked against the manifest's digest and keys. */
async function readLensShard(
  files: PackFiles,
  manifestPath: string,
  site: LensManifestSite,
  layer: number,
  hiddenSize: number,
  signal: AbortSignal,
): Promise<Float32Array> {
  const path = siblingPath(manifestPath, site.tensorFiles[String(layer)]);
  if (files.sha256(path) !== site.tensorDigests[String(layer)]) {
    throw instrumentError("JLENS_PACK_INVALID", `Lens layer ${layer} has a mismatched tensor digest`);
  }
  const reader = await tensorArtifactReader(await files.file(path, signal), signal);
  const expected = site.layers
    .filter((candidate) => site.tensorFiles[String(candidate)] === site.tensorFiles[String(layer)])
    .map((candidate) => `layer_${candidate}`)
    .sort();
  if (!sameStrings(reader.keys, expected)) {
    throw instrumentError("JLENS_PACK_INVALID", `Lens shard ${path} has unexpected tensor keys`);
  }
  return reader.tensor(`layer_${layer}`, [hiddenSize, hiddenSize]).data;
}

async function loadJlens(
  pack: BrowserInstrumentPackArtifacts,
  request: BrowserModelLoadRequest,
): Promise<BrowserJlens> {
  assertArtifactClosure(pack, request);
  const manifestArtifact = uniqueArtifact(pack.artifacts, (artifact) =>
    artifact.manifest.path.endsWith("/manifest.json"), "Lens manifest");
  const vocabularyArtifact = uniqueArtifact(pack.artifacts, (artifact) =>
    artifact.manifest.path.endsWith("/browser-vocabulary.json"), "Lens browser vocabulary");
  const [manifestText, vocabulary] = await Promise.all([
    textArtifact(manifestArtifact, request.signal),
    jsonArtifact(vocabularyArtifact, request.signal),
  ]);
  const manifest = jsonObject(manifestArtifact, manifestText);
  const hiddenSize = request.variant.runtimeIdentity.hiddenSize;
  const sites = browserLensSites(manifest, request, "R");
  const out = sites.get("out")!;
  const files = new PackFiles(pack);
  // The live readout's layers: the out layers whose shard installs with the pack (every layer of
  // a pre-6.0 pack, the readout's evenly spaced set of a full-depth one).
  const layerMap = out.layers.filter((layer) => packFileResidency(
    pack.pack,
    request.variant,
    siblingPath(manifestArtifact.manifest.path, out.tensorFiles[String(layer)]),
  ) === "install");
  if (layerMap.length === 0) {
    throw instrumentError("JLENS_PACK_INVALID", "The R lens pack installs no readout layer");
  }
  const jacobians = new Map<number, Float32Array>();
  for (const layer of layerMap) {
    jacobians.set(layer, await readLensShard(
      files, manifestArtifact.manifest.path, out, layer, hiddenSize, request.signal,
    ));
  }
  if (
    ![
      "drowse-jlens-vocabulary-v2",
      ...LEGACY_PRODUCT_SLUGS.map((slug) => `${slug}-jlens-vocabulary-v2`),
    ].includes(vocabulary.format as string) ||
    vocabulary.hidden_size !== hiddenSize ||
    typeof vocabulary.tensor_file !== "string" || !safeBasename(vocabulary.tensor_file) ||
    typeof vocabulary.tensor_sha256 !== "string" || !Array.isArray(vocabulary.words) ||
    vocabulary.words.length === 0
  ) {
    throw instrumentError("JLENS_VOCABULARY_INVALID", "The lens browser vocabulary manifest is invalid");
  }
  const words = new Map<string, { row: number; tokenId: number }>();
  vocabulary.words.forEach((entry, index) => {
    if (
      !isRecord(entry) || typeof entry.word !== "string" || !entry.word ||
      entry.row !== index || !nonnegativeInteger(entry.token_id) || words.has(entry.word)
    ) {
      throw instrumentError("JLENS_VOCABULARY_INVALID", "Lens vocabulary words and rows must be unique and contiguous");
    }
    words.set(entry.word, { row: index, tokenId: entry.token_id });
  });
  const tensorArtifact = artifactAt(pack.artifacts, siblingPath(
    vocabularyArtifact.manifest.path,
    vocabulary.tensor_file,
  ));
  if (tensorArtifact.manifest.sha256 !== vocabulary.tensor_sha256) {
    throw instrumentError("JLENS_VOCABULARY_INVALID", "The lens vocabulary tensor digest does not match");
  }
  const vocabularyTensors = await tensorArtifactReader(tensorArtifact, request.signal);
  if (!sameStrings(vocabularyTensors.keys, ["unembedding"])) {
    throw instrumentError("JLENS_VOCABULARY_INVALID", "The lens vocabulary must contain one unembedding tensor");
  }
  return {
    source: pack.pack.id,
    displayName: pack.pack.displayName,
    kind: manifest.kind as "J" | "R",
    // browserLensSites checked both.
    nPrompts: manifest.n_prompts as number,
    splitHalf: out.splitHalf,
    hiddenSize,
    layerMap,
    runtimeLayerMap: [...request.variant.runtimeIdentity.layerMap],
    bindingId: manifestArtifact.manifest.sha256,
    identity: lensIdentity(manifestText),
    sites,
    files,
    manifestPath: manifestArtifact.manifest.path,
    jacobians,
    words,
    unembedding: vocabularyTensors.tensor("unembedding", [words.size, hiddenSize]).data,
    cache: new Map(),
    resolvedByTokenId: new Map(),
    pinnedTokenIds: new Set(),
    resolvedBytes: 0,
  };
}

async function loadSlens(
  pack: BrowserInstrumentPackArtifacts,
  request: BrowserModelLoadRequest,
): Promise<BrowserSlens> {
  assertArtifactClosure(pack, request);
  const manifestArtifact = uniqueArtifact(pack.artifacts, (artifact) =>
    artifact.manifest.path.endsWith("/slens/manifest.json"), "S-lens manifest");
  const statisticsArtifact = uniqueArtifact(pack.artifacts, (artifact) =>
    artifact.manifest.path.endsWith("/slens/embedding-statistics.json"), "S embedding statistics");
  const [manifestText, statisticsText] = await Promise.all([
    textArtifact(manifestArtifact, request.signal),
    textArtifact(statisticsArtifact, request.signal),
  ]);
  const manifest = jsonObject(manifestArtifact, manifestText);
  const sites = browserLensSites(manifest, request, "S");
  const record = jsonObject(statisticsArtifact, statisticsText);
  const mask = isRecord(record.mask) ? record.mask : null;
  if (
    record.format !== S_EMBEDDING_STATISTICS_FORMAT ||
    record.runtime_identity_sha256 !== request.variant.runtimeIdentitySha256 ||
    record.d_emb !== request.variant.runtimeIdentity.hiddenSize ||
    !positiveInteger(record.vocab_rows) || typeof record.embedding_scale !== "number" ||
    !(record.embedding_scale > 0) || mask === null || typeof mask.source !== "string" ||
    typeof mask.detail !== "string" || !positiveInteger(mask.kept) ||
    typeof record.tokenizer_fingerprint !== "string" ||
    typeof record.tensor_file !== "string" || !safeBasename(record.tensor_file) ||
    typeof record.mask_file !== "string" || !safeBasename(record.mask_file) ||
    record.mask_words !== Math.ceil((record.vocab_rows as number) / 32) ||
    typeof record.tensor_sha256 !== "string" || typeof record.mask_file_sha256 !== "string"
  ) {
    throw instrumentError("SLENS_PACK_INVALID", "The S embedding statistics record is invalid for this runtime");
  }
  // The token mask and counts are indexed by the tokenizer they were taken with (B1 §5.1).
  if (record.tokenizer_sha256 !== request.variant.runtimeIdentity.tokenizerSha256) {
    throw instrumentError(
      "SLENS_TOKENIZER_MISMATCH",
      "The S embedding statistics were taken with another tokenizer than the loaded runtime's",
    );
  }
  for (const [file, digest] of [[record.tensor_file, record.tensor_sha256], [record.mask_file, record.mask_file_sha256]]) {
    const artifact = artifactAt(pack.artifacts, siblingPath(statisticsArtifact.manifest.path, file as string));
    if (artifact.manifest.sha256 !== digest) {
      throw instrumentError("SLENS_PACK_INVALID", `S embedding statistics file ${String(file)} has a mismatched digest`);
    }
  }
  return {
    source: pack.pack.id,
    displayName: pack.pack.displayName,
    nPrompts: manifest.n_prompts as number,
    identity: lensIdentity(manifestText),
    bindingId: manifestArtifact.manifest.sha256,
    sites,
    files: new PackFiles(pack),
    manifestPath: manifestArtifact.manifest.path,
    statisticsPath: statisticsArtifact.manifest.path,
    statisticsIdentity: lensIdentity(statisticsText),
    statisticsRecord: record,
    statistics: null,
  };
}

async function readEmbeddingStatistics(
  slens: BrowserSlens,
  request: BrowserModelLoadRequest,
  signal: AbortSignal,
): Promise<BrowserEmbeddingStatistics> {
  const record = slens.statisticsRecord;
  const d = request.variant.runtimeIdentity.hiddenSize;
  const tensors = await tensorArtifactReader(
    await slens.files.file(siblingPath(slens.statisticsPath, record.tensor_file as string), signal),
    signal,
  );
  if (!sameStrings(tensors.keys, ["mean_E", "sigma_E"])) {
    throw instrumentError("SLENS_PACK_INVALID", "The S embedding statistics have unexpected tensors");
  }
  const maskArtifact = await slens.files.file(siblingPath(slens.statisticsPath, record.mask_file as string), signal);
  const bytes = new Uint8Array(await maskArtifact.file.arrayBuffer());
  signal.throwIfAborted();
  const words = record.mask_words as number;
  if (bytes.byteLength !== 4 * words) {
    throw instrumentError("SLENS_PACK_INVALID", "The S token mask has the wrong length");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const mask = Uint32Array.from({ length: words }, (_, index) => view.getUint32(4 * index, true));
  const vocabRows = record.vocab_rows as number;
  let kept = 0;
  for (let index = 0; index < words; index += 1) {
    let word = index === words - 1 && vocabRows % 32 !== 0
      ? mask[index] & (2 ** (vocabRows % 32) - 1)
      : mask[index];
    if (word !== mask[index]) {
      throw instrumentError("SLENS_PACK_INVALID", "The S token mask sets bits past the vocabulary");
    }
    while (word !== 0) {
      word &= word - 1;
      kept += 1;
    }
  }
  const description = record.mask as { source: string; detail: string; kept: number };
  if (kept !== description.kept) {
    throw instrumentError("SLENS_PACK_INVALID", "The S token mask does not keep the tokens it describes");
  }
  return {
    meanE: tensors.tensor("mean_E", [d]).data,
    sigmaE: tensors.tensor("sigma_E", [d, d]).data,
    mask,
    vocabRows,
    embeddingScale: record.embedding_scale as number,
    maskDescription: { source: description.source, detail: description.detail, kept: description.kept },
    tokenizerFingerprint: record.tokenizer_fingerprint as string,
  };
}

async function loadUnits(
  pack: BrowserInstrumentPackArtifacts,
  request: BrowserModelLoadRequest,
): Promise<BrowserUnits> {
  assertArtifactClosure(pack, request);
  const recordArtifact = uniqueArtifact(pack.artifacts, (artifact) =>
    artifact.manifest.path.endsWith("/units/units.json"), "Units record");
  const record = jsonObject(recordArtifact, await textArtifact(recordArtifact, request.signal));
  const identity = request.variant.runtimeIdentity;
  const invalid = (detail: string): never => {
    throw instrumentError("UNITS_PACK_INVALID", `The units pack is invalid for this runtime: ${detail}`);
  };
  const binding = isRecord(record.binding) ? record.binding : invalid("no binding");
  const neutral = isRecord(binding.neutral) ? binding.neutral : invalid("no neutral binding");
  const nullRecord = isRecord(record.null) ? record.null : invalid("no null basis");
  const headKept = isRecord(record.head_kept) ? record.head_kept : invalid("no head vocabulary");
  const layersRecord = isRecord(record.layers) ? record.layers : invalid("no layers");
  if (
    record.format !== UNITS_PACK_FORMAT ||
    record.runtime_identity_sha256 !== request.variant.runtimeIdentitySha256 ||
    record.model_layer_count !== identity.layerMap.length || record.d_model !== identity.hiddenSize ||
    !positiveInteger(record.d_ff) || !positiveInteger(record.n_heads) || !positiveInteger(record.head_dim) ||
    !positiveInteger(nullRecord.rank) || nullRecord.key !== "null_basis" ||
    typeof nullRecord.file !== "string" || !safeBasename(nullRecord.file) ||
    typeof nullRecord.sha256 !== "string" ||
    !positiveInteger(headKept.count) || !positiveInteger(headKept.unembedding_rows) ||
    [
      binding.r_lens_identity, binding.s_lens_identity, binding.embedding_statistics_sha256,
      neutral.sha256, neutral.rows_sha256,
    ].some((digest) => typeof digest !== "string" || !/^[0-9a-f]{64}$/u.test(digest))
  ) invalid("its record");
  const root = recordArtifact.manifest.path;
  const files = new PackFiles(pack);
  const layers = new Map<number, Record<"reads" | "contrast", {
    path: string; sha256: string; shapes: Record<string, number[]>;
  }>>();
  for (const layer of identity.layerMap) {
    const entry = layersRecord[String(layer)];
    if (!isRecord(entry)) invalid(`layer ${layer} is missing`);
    const parts = {} as Record<"reads" | "contrast", { path: string; sha256: string; shapes: Record<string, number[]> }>;
    for (const part of ["reads", "contrast"] as const) {
      const value = (entry as Record<string, unknown>)[part];
      if (
        !isRecord(value) || typeof value.file !== "string" || !safeBasename(value.file) ||
        value.file !== `${part}.layer-${layer}.safetensors` || typeof value.sha256 !== "string" ||
        !isRecord(value.shapes) || Object.keys(value.shapes).length === 0 ||
        Object.values(value.shapes).some((shape) =>
          !Array.isArray(shape) || shape.length === 0 || !shape.every(positiveInteger))
      ) invalid(`layer ${layer} ${part}`);
      const path = siblingPath(root, (value as Record<string, unknown>).file as string);
      if (files.sha256(path) !== (value as Record<string, unknown>).sha256) invalid(`layer ${layer} ${part} digest`);
      parts[part] = {
        path,
        sha256: (value as Record<string, unknown>).sha256 as string,
        shapes: (value as Record<string, unknown>).shapes as Record<string, number[]>,
      };
    }
    layers.set(layer, parts);
  }
  if (Object.keys(layersRecord).length !== identity.layerMap.length) invalid("layers outside the model");
  const nullPath = siblingPath(root, nullRecord.file as string);
  if (artifactAt(pack.artifacts, nullPath).manifest.sha256 !== nullRecord.sha256) invalid("null basis digest");
  // What the core pack's neutral statistics were measured on and its whitener fitted from, which
  // every units statistic must share (`readNeutralStatistics` checks the tensors).
  let coreNeutral: CoreNeutralBinding | null = null;
  try {
    coreNeutral = await readCoreNeutralBinding(
      request.artifacts.filter((artifact) => artifact.manifest.role === "core_pack"),
      request,
    );
  } catch (error) {
    if ((error as { code?: unknown })?.code !== "CORE_NEUTRAL_STATISTICS_INVALID") throw error;
  }
  return {
    descriptor: {
      source: pack.pack.id,
      displayName: pack.pack.displayName,
      layers: identity.layerMap.length,
      dModel: identity.hiddenSize,
      dFF: record.d_ff as number,
      nHeads: record.n_heads as number,
      headDim: record.head_dim as number,
      nullRank: nullRecord.rank as number,
      headKeptCount: headKept.count as number,
      unembeddingRows: headKept.unembedding_rows as number,
      binding: {
        rLensIdentity: binding.r_lens_identity as string,
        sLensIdentity: binding.s_lens_identity as string,
        embeddingStatisticsSha256: binding.embedding_statistics_sha256 as string,
        neutralSha256: neutral.sha256 as string,
        neutralRowsSha256: neutral.rows_sha256 as string,
      },
    },
    files,
    root,
    nullFile: { path: nullPath, sha256: nullRecord.sha256 as string },
    coreNeutral,
    layers,
    nullBasis: null,
  };
}

function matchesRuntimeFingerprint(
  fingerprint: unknown,
  request: BrowserModelLoadRequest,
): boolean {
  if (fingerprint === request.variant.runtimeIdentitySha256) return true;
  if (request.variant.runtimeIdentity.runtimeAbi !== "drowse-web-runtime-v1") {
    return false;
  }
  return LEGACY_PRODUCT_SLUGS.some((slug) => fingerprint === runtimeIdentitySha256({
    ...request.variant.runtimeIdentity,
    runtimeAbi: `${slug}-web-runtime-v1`,
  }));
}

function jlensDirectionsByteLength(
  directions: ReadonlyMap<number, Float32Array>,
): number {
  let bytes = 0;
  for (const direction of directions.values()) bytes += direction.byteLength;
  return bytes;
}

function touchJlensDirections(
  jlens: BrowserJlens,
  tokenId: number,
): ReadonlyMap<number, Float32Array> | undefined {
  const directions = jlens.resolvedByTokenId.get(tokenId);
  if (directions === undefined) return undefined;
  jlens.resolvedByTokenId.delete(tokenId);
  jlens.resolvedByTokenId.set(tokenId, directions);
  return directions;
}

function touchJlensWord(
  jlens: BrowserJlens,
  word: string,
  token: BrowserJlensToken,
): void {
  jlens.cache.delete(word);
  jlens.cache.set(word, token);
}

function storeJlensDirections(
  jlens: BrowserJlens,
  tokenId: number,
  directions: ReadonlyMap<number, Float32Array>,
): void {
  const previous = jlens.resolvedByTokenId.get(tokenId);
  if (previous !== undefined) {
    jlens.resolvedBytes -= jlensDirectionsByteLength(previous);
    jlens.resolvedByTokenId.delete(tokenId);
  }
  jlens.resolvedByTokenId.set(tokenId, directions);
  jlens.resolvedBytes += jlensDirectionsByteLength(directions);
  trimJlensDirectionCache(jlens);
}

function trimJlensDirectionCache(jlens: BrowserJlens): void {
  const maximumEntries = JLENS_DIRECTION_CACHE_MAX_ENTRIES + jlens.pinnedTokenIds.size;
  while (
    jlens.resolvedByTokenId.size > maximumEntries ||
    jlens.resolvedBytes > JLENS_DIRECTION_CACHE_MAX_BYTES
  ) {
    const evicted = [...jlens.resolvedByTokenId].find(
      ([tokenId]) => !jlens.pinnedTokenIds.has(tokenId),
    );
    if (evicted === undefined) break;
    const [tokenId, directions] = evicted;
    jlens.resolvedByTokenId.delete(tokenId);
    jlens.resolvedBytes -= jlensDirectionsByteLength(directions);
    for (const [word, token] of jlens.cache) {
      if (token.tokenId === tokenId) jlens.cache.delete(word);
    }
  }
  while (jlens.cache.size > maximumEntries) {
    const evicted = [...jlens.cache].find(
      ([, token]) => !jlens.pinnedTokenIds.has(token.tokenId),
    );
    if (evicted === undefined) break;
    jlens.cache.delete(evicted[0]);
  }
}

async function resolveJlensWordToken(
  resolver: BrowserJlensRuntimeResolver,
  word: string,
): Promise<number> {
  // `jlens/#<id>` names the vocab id exactly, as in drowse's grammar, and is
  // bounds-checked against the loaded model's vocabulary (drowse's
  // InvalidTokenIdError) rather than reaching the runtime out of range.
  const exact = /^#(\d+)$/u.exec(word);
  if (exact !== null) {
    const tokenId = Number(exact[1]);
    const vocabularySize = resolver.vocabularySize();
    if (!Number.isSafeInteger(tokenId) || tokenId >= vocabularySize) {
      throw instrumentError(
        "JLENS_TOKENIZATION_INVALID",
        `token id ${exact[1]} is outside this vocabulary (size ${vocabularySize})`,
      );
    }
    return tokenId;
  }
  let pieces: number[] = [];
  for (const candidate of [` ${word}`, word]) {
    const ids = await resolver.tokenizeText(candidate);
    if (
      !Array.isArray(ids) ||
      ids.some((id) => !Number.isSafeInteger(id) || id < 0 || id > 0x7fff_ffff)
    ) {
      throw instrumentError(
        "JLENS_TOKENIZATION_INVALID",
        "The loaded tokenizer returned invalid lens token IDs",
      );
    }
    if (ids.length === 1 && (await resolver.decodeTokens(ids)).trim() === word) {
      return ids[0];
    }
    if (pieces.length === 0 && ids.length > 1) pieces = [...ids];
  }
  throw instrumentError(
    "MULTI_TOKEN_WORD",
    `${JSON.stringify(word)} is not a single token in this vocabulary` +
      (pieces.length > 0 ? ` (token IDs: ${pieces.join(", ")})` : "") +
      ". A lens word must encode as one token.",
  );
}

function normalizeJlensWord(word: string): string {
  const normalized = word.trim().normalize("NFC");
  if (normalized.length === 0) {
    throw instrumentError("JLENS_TOKEN_UNAVAILABLE", "Lens words must be non-empty");
  }
  return normalized;
}

type PythonIdentityValue =
  | string
  | number
  | null
  | readonly number[]
  | { readonly pythonFloat: number };

function pythonTupleDigest(values: readonly PythonIdentityValue[]): string {
  const source = `(${values.map(pythonIdentityRepr).join(", ")}${values.length === 1 ? "," : ""})`;
  return bytesToHex(sha256(new TextEncoder().encode(source)));
}

function pythonIdentityRepr(value: PythonIdentityValue): string {
  if (value === null) return "None";
  if (Array.isArray(value)) {
    const entries = value as readonly number[];
    return `(${entries.map((entry) => pythonNumberRepr(entry)).join(", ")}${entries.length === 1 ? "," : ""})`;
  }
  if (typeof value === "number") return pythonNumberRepr(value);
  if (typeof value === "object") {
    return pythonNumberRepr((value as { readonly pythonFloat: number }).pythonFloat, true);
  }
  return pythonStringRepr(value);
}

function pythonStringRepr(value: string): string {
  const quote = value.includes("'") && !value.includes('"') ? '"' : "'";
  let output = quote;
  for (const character of value) {
    const codePoint = character.codePointAt(0)!;
    if (character === "\\" || character === quote) {
      output += `\\${character}`;
    } else if (character === "\t") {
      output += "\\t";
    } else if (character === "\n") {
      output += "\\n";
    } else if (character === "\r") {
      output += "\\r";
    } else if (pythonPrintable(character)) {
      output += character;
    } else if (codePoint <= 0xff) {
      output += `\\x${codePoint.toString(16).padStart(2, "0")}`;
    } else if (codePoint <= 0xffff) {
      output += `\\u${codePoint.toString(16).padStart(4, "0")}`;
    } else {
      output += `\\U${codePoint.toString(16).padStart(8, "0")}`;
    }
  }
  return `${output}${quote}`;
}

function pythonPrintable(value: string): boolean {
  return value === " " || !/[\p{Cc}\p{Cf}\p{Cs}\p{Co}\p{Cn}\p{Zl}\p{Zp}\p{Zs}]/u.test(value);
}

function pythonNumberRepr(value: number, forceFloat = false): string {
  if (!Number.isFinite(value)) {
    throw instrumentError("INSTRUMENT_PROBE_IDENTITY_INVALID", "Probe identity numbers must be finite");
  }
  if (Number.isSafeInteger(value) && !Object.is(value, -0) && !forceFloat) return String(value);
  if (Object.is(value, -0)) return "-0.0";
  const source = String(value);
  const exponentIndex = source.search(/[eE]/u);
  if (exponentIndex >= 0) {
    const coefficient = source.slice(0, exponentIndex);
    const exponent = Number(source.slice(exponentIndex + 1));
    return `${coefficient}e${exponent >= 0 ? "+" : "-"}${Math.abs(exponent).toString().padStart(2, "0")}`;
  }
  const negative = source.startsWith("-");
  const unsigned = negative ? source.slice(1) : source;
  const dot = unsigned.indexOf(".");
  const integer = dot < 0 ? unsigned : unsigned.slice(0, dot);
  const fraction = dot < 0 ? "" : unsigned.slice(dot + 1);
  const exponent = integer !== "0"
    ? integer.length - 1
    : -(fraction.search(/[1-9]/u) + 1);
  if (exponent >= -4 && exponent < 16) {
    return source.includes(".") ? source : `${source}.0`;
  }
  const digits = integer === "0"
    ? fraction.slice(fraction.search(/[1-9]/u)).replace(/0+$/u, "")
    : `${integer}${fraction}`.replace(/0+$/u, "");
  const coefficient = digits.length === 1 ? digits : `${digits[0]}.${digits.slice(1)}`;
  return `${negative ? "-" : ""}${coefficient}e${exponent >= 0 ? "+" : "-"}${Math.abs(exponent).toString().padStart(2, "0")}`;
}

// ---------------------------------------------------------------- core neutral statistics ---

const NEUTRAL_STATISTICS_FORMAT = "drowse-neutral-statistics-v1";
const NEUTRAL_STATISTICS_RECORD = "neutral-statistics.json";
const NEUTRAL_STATISTICS_KEYS = ["head_mean", "head_scale", "mid_mean", "neuron_mean", "neuron_scale", "out_mean"];

/** Whether a core pack file is part of its neutral statistics (not a manifold envelope). */
export function isNeutralStatisticsPath(path: string): boolean {
  const name = path.split("/").at(-1);
  return name === NEUTRAL_STATISTICS_RECORD || name === "neutral-statistics.safetensors";
}

/** The neutral pass a core pack carries (B1 §4.2): unit statistics and the `out` and `mid` means,
 *  each `[L, ...]` over the runtime's layers, row-major at its logical shape. */
export interface BrowserNeutralStatistics {
  readonly nDocuments: number;
  /** Python's canonical digest of the neutral pass's identity, which a units pack binds. */
  readonly neutralSha256: string;
  readonly tensors: ReadonlyMap<string, { readonly shape: readonly number[]; readonly data: Float32Array }>;
}

/** What a core pack's neutral statistics bind: the neutral pass and the rows its whitener was
 *  fitted from, with every context's whitener file the one the record names. */
interface CoreNeutralBinding {
  readonly sha256: string;
  readonly rowsSha256: string;
  readonly record: Record<string, unknown>;
  readonly recordArtifact: BrowserModelArtifact;
}

async function readCoreNeutralBinding(
  artifacts: readonly BrowserModelArtifact[],
  request: Pick<BrowserModelLoadRequest, "variant" | "signal">,
): Promise<CoreNeutralBinding | null> {
  const records = artifacts.filter((artifact) => artifact.manifest.path.split("/").at(-1) === NEUTRAL_STATISTICS_RECORD);
  if (records.length === 0) return null;
  const invalid = (detail: string): never => {
    throw instrumentError("CORE_NEUTRAL_STATISTICS_INVALID", `The core pack's neutral statistics are invalid: ${detail}`);
  };
  if (records.length !== 1) invalid("more than one record");
  const record = jsonObject(records[0], await textArtifact(records[0], request.signal));
  const neutral = isRecord(record.neutral) ? record.neutral : invalid("no neutral pass");
  const whitener = isRecord(record.whitener) ? record.whitener : invalid("no whitener binding");
  const tensors = isRecord(whitener.tensors_sha256) ? whitener.tensors_sha256 : invalid("no whitener digests");
  const contexts = (request.variant.contextProfiles ?? []).map((profile) => String(profile.contextTokens)).sort();
  if (
    typeof neutral.sha256 !== "string" || !/^[0-9a-f]{64}$/u.test(neutral.sha256) ||
    typeof neutral.rows_sha256 !== "string" || !/^[0-9a-f]{64}$/u.test(neutral.rows_sha256) ||
    whitener.rows_sha256 !== neutral.rows_sha256 || contexts.length === 0 ||
    !sameStrings(Object.keys(tensors).sort(), contexts)
  ) invalid("its neutral pass or whitener binding");
  for (const context of contexts) {
    const path = siblingPath(records[0].manifest.path, `${context}/neutral-whitener.safetensors`);
    const matches = artifacts.filter((artifact) => artifact.manifest.path === path);
    if (matches.length !== 1 || matches[0].manifest.sha256 !== tensors[context]) {
      invalid(`the ${context}-token whitener is not the one fitted from the neutral rows`);
    }
  }
  return {
    sha256: neutral.sha256 as string,
    rowsSha256: neutral.rows_sha256 as string,
    record,
    recordArtifact: records[0],
  };
}

/** Read a core pack's neutral statistics (`null` when the pack has none: a model whose blocks
 *  units do not cover). */
export async function readNeutralStatistics(
  artifacts: readonly BrowserModelArtifact[],
  request: Pick<BrowserModelLoadRequest, "variant" | "signal">,
): Promise<BrowserNeutralStatistics | null> {
  const binding = await readCoreNeutralBinding(artifacts, request);
  if (binding === null) return null;
  const invalid = (detail: string): never => {
    throw instrumentError("CORE_NEUTRAL_STATISTICS_INVALID", `The core pack's neutral statistics are invalid: ${detail}`);
  };
  const recordArtifact = binding.recordArtifact;
  const record = binding.record;
  const identity = request.variant.runtimeIdentity;
  const layers = identity.layerMap.length;
  const shapes = isRecord(record.shapes) ? record.shapes : invalid("no shapes");
  const neutral = isRecord(record.neutral) ? record.neutral : invalid("no neutral pass");
  if (
    record.format !== NEUTRAL_STATISTICS_FORMAT ||
    record.runtime_identity_sha256 !== request.variant.runtimeIdentitySha256 ||
    record.model_layer_count !== layers || !positiveInteger(record.n_documents) ||
    typeof neutral.sha256 !== "string" || !/^[0-9a-f]{64}$/u.test(neutral.sha256) ||
    typeof record.tensor_file !== "string" || !safeBasename(record.tensor_file) ||
    !sameStrings(Object.keys(shapes).sort(), NEUTRAL_STATISTICS_KEYS) ||
    Object.values(shapes).some((shape) => !Array.isArray(shape) || shape[0] !== layers || !shape.every(positiveInteger)) ||
    (shapes.mid_mean as number[])[1] !== identity.hiddenSize || (shapes.out_mean as number[])[1] !== identity.hiddenSize
  ) invalid("its record does not match this runtime");
  const tensorArtifact = artifactAt(artifacts, siblingPath(recordArtifact.manifest.path, record.tensor_file as string));
  if (tensorArtifact.manifest.sha256 !== record.tensor_sha256) invalid("the tensor digest");
  const reader = await tensorArtifactReader(tensorArtifact, request.signal);
  if (!sameStrings(reader.keys, NEUTRAL_STATISTICS_KEYS)) invalid("unexpected tensors");
  const tensors = new Map<string, { shape: readonly number[]; data: Float32Array }>();
  for (const name of NEUTRAL_STATISTICS_KEYS) {
    const shape = shapes[name] as number[];
    const stored = shape.length > 2
      ? [shape.slice(0, -1).reduce((total, size) => total * size, 1), shape[shape.length - 1]]
      : shape;
    tensors.set(name, { shape, data: reader.tensor(name, stored).data });
  }
  return { nDocuments: record.n_documents as number, neutralSha256: neutral.sha256 as string, tensors };
}

/** The artifacts hold every file that installs with the pack and nothing outside its catalog
 *  closure (on-demand layer files may be present too: a release validator passes them all). */
function assertArtifactClosure(pack: BrowserInstrumentPackArtifacts, request: BrowserModelLoadRequest): void {
  const closure = new Set(pack.pack.files.map((file) => `${file.path}:${file.sha256}`));
  const actual = pack.artifacts.map(({ manifest }) => `${manifest.path}:${manifest.sha256}`);
  const present = new Set(actual);
  if (
    actual.length !== present.size || actual.some((entry) => !closure.has(entry)) ||
    packInstallFiles(pack.pack, request.variant).some((file) => !present.has(`${file.path}:${file.sha256}`))
  ) {
    throw instrumentError("INSTRUMENT_PACK_ARTIFACT_MISMATCH", `${pack.pack.displayName} does not match its catalog closure`);
  }
}

function uniqueArtifact(
  artifacts: readonly BrowserModelArtifact[],
  predicate: (artifact: BrowserModelArtifact) => boolean,
  label: string,
): BrowserModelArtifact {
  const matches = artifacts.filter(predicate);
  if (matches.length !== 1) throw instrumentError("INSTRUMENT_PACK_INVALID", `${label} must occur exactly once`);
  return matches[0];
}

function artifactAt(artifacts: readonly BrowserModelArtifact[], path: string): BrowserModelArtifact {
  const matches = artifacts.filter((artifact) => artifact.manifest.path === path);
  if (matches.length !== 1) throw instrumentError("INSTRUMENT_PACK_INVALID", `Instrument artifact ${path} is missing or duplicated`);
  return matches[0];
}

async function jsonArtifact(
  artifact: BrowserModelArtifact,
  signal: AbortSignal,
): Promise<Record<string, unknown>> {
  return jsonObject(artifact, await textArtifact(artifact, signal));
}

async function textArtifact(artifact: BrowserModelArtifact, signal: AbortSignal): Promise<string> {
  signal.throwIfAborted();
  const text = await artifact.file.text();
  signal.throwIfAborted();
  return text;
}

function jsonObject(artifact: BrowserModelArtifact, text: string): Record<string, unknown> {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    throw instrumentError("INSTRUMENT_PACK_INVALID", `${artifact.manifest.path} is not valid JSON`);
  }
  if (!isRecord(value)) throw instrumentError("INSTRUMENT_PACK_INVALID", `${artifact.manifest.path} must contain an object`);
  return value;
}

async function tensorArtifactReader(
  artifact: BrowserModelArtifact,
  signal: AbortSignal,
): Promise<Fp32SafetensorsReader> {
  signal.throwIfAborted();
  const bytes = new Uint8Array(await artifact.file.arrayBuffer());
  signal.throwIfAborted();
  return decodeFp32Safetensors(bytes, artifact.manifest.path);
}

function siblingPath(path: string, basename: string): string {
  const index = path.lastIndexOf("/");
  return index < 0 ? basename : `${path.slice(0, index + 1)}${basename}`;
}

function safeBasename(value: string): boolean {
  return value.length > 0 && value !== "." && value !== ".." &&
    !value.includes("/") && !value.includes("\\") && !value.includes("\0");
}

function integerList(value: unknown): value is number[] {
  return Array.isArray(value) && value.length > 0 && value.every(nonnegativeInteger) &&
    value.every((item, index) => index === 0 || value[index - 1] < item);
}

function nonnegativeInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && Number(value) >= 0;
}

function positiveInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && Number(value) > 0;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function sameStrings(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function instrumentError(code: string, message: string): Error & {
  code: string;
  status: number;
  recoverable: boolean;
} {
  return Object.assign(new Error(message), { code, status: 409, recoverable: true });
}
