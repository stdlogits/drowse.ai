import {
  decodeFp32Safetensors,
  readVerifiedDrowseArchiveFile,
  validateDrowseArchive,
  type Fp32SafetensorsReader,
  type VerifiedDrowseArchive,
} from "../artifacts";
import { domainLabel } from "../artifacts/authoring";
import { sha256 } from "@noble/hashes/sha2.js";
import { bytesToHex } from "@noble/hashes/utils.js";
import type { BrowserModelArtifact, BrowserModelLoadRequest } from "./modelBackend";
import { LEGACY_PRODUCT_SLUGS } from "../../../../drowse/webui/src/lib/brandMigration";
import {
  BrowserInstrumentRegistry,
  isNeutralStatisticsPath,
  readNeutralStatistics,
  type BrowserNeutralStatistics,
  type BrowserJlensGpuDictionary,
  type BrowserJlensRuntimeResolver,
  type BrowserSaeGpuDictionary,
  type BrowserSaeFeature,
} from "./browserInstrumentPacks";
import type { SerializedMahalanobisWhitener } from "../fitting/workerContracts";
import {
  leaceProject,
  applySerializedInverse,
  mahalanobisNorm,
} from "../fitting/mahalanobis";
import type {
  CorrelationData,
  GeometryProbeInfo,
  ManifoldFitInfo,
  ManifoldInfo,
  PairwiseCompareResponse,
  ProbeGeometryResponse,
  ProbeInfo,
  ProbeRequest,
  ProfileListResponse,
  SteeringPartJSON,
  VectorInfo,
} from "../../../../drowse/webui/src/lib/types";
import {
  compileStructuredHookProgram,
  saeFeatureProbe,
  structuredUnitAblationTerm,
  validateStructuredHookProgram,
  withStructuredUnitSection,
  type StructuredAffineGroup,
  type StructuredCurve,
  type StructuredCurveAxis,
  type StructuredGeometryMeasurementProbe,
  type StructuredGeometryProbe,
  type StructuredGeometryWhitener,
  type StructuredHookControl,
  type StructuredHookLayer,
  type StructuredHookProgramBuffers,
  type StructuredMeasurementProbe,
  type StructuredMeasurementSlot,
  type StructuredProbe,
  type StructuredSlensProbe,
} from "./structuredHookProgram";
import {
  resolveStructuredHookProfile,
  type StructuredHookCapacityProfile,
} from "./structuredHookProfile";
import {
  canonicalConceptName,
  foldSteering,
  formatLayerClause,
  formatSteering,
  normalizeTrigger,
  parseSteeringExpression,
  RESERVED_NAMESPACES,
  selectTermLayers,
  steeringParts,
  type LayerScope,
  type SteeringAtom,
  type SteeringManifoldEntry,
  type SteeringPhase,
  type SteeringExpressionTerm,
  type SteeringTrigger,
} from "./steeringExpression";
import { instrumentAtomUnavailable } from "./hookCapabilities";
import {
  resolveUnitAtom,
  unitMean,
  unitTap,
  type BrowserUnitProgramSource,
} from "./units/unitProgram";
import { slensProbeDigest, unitProbeDigest } from "./units/probeDigest";
import { unitName, type UnitRef } from "./units/unitReader";

interface CoreManifold {
  key: string;
  selectorKey: string;
  namespace: string;
  name: string;
  featureSpace: string;
  variant: "raw" | "sae";
  variantIdentity: string | null;
  safeVariantIdentity: string | null;
  fitMode: string;
  labels: readonly string[];
  nodeRoles: readonly (string | null)[];
  nodeCoordinates: readonly (readonly number[])[];
  tensorPath: string;
  tensors: Fp32SafetensorsReader;
  mahalanobisShares: ReadonlyMap<number, number>;
  domain: Record<string, unknown>;
  originPerLayer: ReadonlyMap<number, Float32Array>;
  info: ManifoldInfo;
}

interface BrowserAnalyticsProfile {
  info: VectorInfo;
  directions: ReadonlyMap<number, Float32Array>;
}

interface ResolvedManifoldAtom {
  kind: "manifold";
  key: string;
  manifold: CoreManifold;
  labelIndex: number;
  requestedRole: string | null;
  labelAlias: boolean;
}

interface ResolvedInstrumentAtom {
  kind: "instrument";
  key: string;
  family: "sae" | "jlens" | "units" | "slens";
  directions: ReadonlyMap<number, Float32Array>;
  saeFeature: BrowserSaeFeature | null;
  /** A `neuron/…` or `head/…` atom's unit (its push direction, if any, is `directions`). */
  unit?: UnitRef;
}

type ResolvedCoreAtom = ResolvedManifoldAtom | ResolvedInstrumentAtom;

interface ResolvedCoreTerm {
  key: string;
  base: ResolvedCoreAtom;
  projection: {
    operator: "~" | "|";
    onto: ResolvedCoreAtom;
  } | null;
  coefficient: number;
  coefficients: readonly number[];
  ablation: boolean;
  trigger: SteeringTrigger | null;
  manifoldPosition: readonly number[] | string | null;
  /** The `{…}` clause's scope, and the canonical IR key the clause belongs to. */
  layers: LayerScope;
  irKey: string;
  /** The layers the term acts on (`select_term_layers`), `null` for every fitted one. Set once
   *  the expression is folded. */
  actsAt: ReadonlySet<number> | null;
}

/** Where one validated term acts, as `POST .../steering/validate` reports it. */
export interface BrowserSteeringTermLayers {
  key: string;
  fitted: number[];
  share: number[];
  layers: number[];
}

/** A compiled program, with what steering validation answers about its expression: the
 *  canonical text, each term's fields, where each acts and the model's depth. */
export type BrowserCompiledSteering = StructuredHookProgramBuffers & {
  steeringExpression?: string;
  steeringParts?: SteeringPartJSON[];
  steeringTerms?: readonly BrowserSteeringTermLayers[];
  steeringDepth?: number;
};

interface InstalledProbeProgram {
  slots: readonly StructuredMeasurementSlot[];
  geometry: StructuredGeometryMeasurementProbe | null;
}

interface StructuredAffineTermGroup {
  trigger: SteeringTrigger | null;
  terms: ResolvedCoreTerm[];
}

/** Every curved manifold's frame at one layer, stacked: write rows and their dual reads. */
interface CurvedFrame {
  basis: readonly ArrayLike<number>[];
  read: readonly ArrayLike<number>[];
}

// drowse's steering gains (steering/hooks.py), calibrated for typed frames:
// `eff_along_L = share_L · SUBSPACE_GAIN` for an affine group, `along · share_L ·
// MANIFOLD_ALONG_GAIN` for a curved one, `clamp(onto · share_L · MANIFOLD_ONTO_GAIN, 0, 1)`.
const SUBSPACE_GAIN = 8;
const MANIFOLD_ALONG_GAIN = 2;
const MANIFOLD_ONTO_GAIN = 0.5;

export class BrowserFeasibilityCorePackCompiler {
  private neutralStatisticsRead: Promise<BrowserNeutralStatistics | null> | null;

  private geometryLive = true;
  private lensLive = false;
  private lensLiveLayers: readonly number[] | null = null;
  private saeLive = false;
  private compiledProgram: { key: string; program: StructuredHookProgramBuffers } | null = null;
  private unitSource: BrowserUnitProgramSource | null = null;
  private unitSourceVersion = 0;
  private readonly attachedProbes = new Map<string, ProbeInfo>();
  /** Each lens atom's read covectors (by their map's identity) raised to steering writes. */
  private readonly raisedReads = new WeakMap<
    ReadonlyMap<number, Float32Array>,
    ReadonlyMap<number, Float32Array>
  >();

  private constructor(
    private readonly modelId: string,
    private readonly hiddenSize: number,
    private readonly layerMap: readonly number[],
    private readonly structuredHookProfile: StructuredHookCapacityProfile,
    private readonly manifolds: readonly CoreManifold[],
    private readonly whiteners: ReadonlyMap<number, SerializedMahalanobisWhitener>,
    private readonly instruments: BrowserInstrumentRegistry,
    private readonly loadRequest: BrowserModelLoadRequest,
    private readonly coreManifolds: readonly CoreManifold[],
  ) {
    this.neutralStatisticsRead = null;
    const keys = new Set<string>();
    for (const manifold of manifolds) {
      if (keys.has(manifold.selectorKey)) {
        throw coreError(
          "CORE_PACK_IDENTITY_COLLISION",
          `The core pack contains multiple fitted variants named ${manifold.selectorKey}`,
        );
      }
      keys.add(manifold.selectorKey);
    }
  }

  static async load(
    request: BrowserModelLoadRequest,
    options: {
      installedManifoldArchives?: readonly Blob[];
    } = {},
  ): Promise<BrowserFeasibilityCorePackCompiler> {
    const structuredHookProfile = resolveStructuredHookProfile(
      request.variant.structuredHookProfile,
    );
    const expected = request.requiredCorePack.files;
    const artifacts = request.artifacts.filter((artifact) => artifact.manifest.role === "core_pack");
    if (
      artifacts.length !== expected.length ||
      !expected.every((file) => artifacts.some((artifact) => artifact.manifest.sha256 === file.sha256))
    ) {
      throw coreError(
        "CORE_PACK_ARTIFACT_MISMATCH",
        "The loaded core pack does not match the selected catalog variant",
      );
    }
    const whitenerTensor = artifacts.filter((artifact) =>
      artifact.manifest.path.endsWith("/neutral-whitener.safetensors")
    );
    const whitenerSidecar = artifacts.filter((artifact) =>
      artifact.manifest.path.endsWith("/neutral-whitener.json")
    );
    const manifoldArtifacts = artifacts.filter((artifact) =>
      !whitenerTensor.includes(artifact) && !whitenerSidecar.includes(artifact) &&
      !isNeutralStatisticsPath(artifact.manifest.path)
    );
    if (manifoldArtifacts.length === 0) {
      throw coreError("CORE_PACK_MANIFOLD_INVALID", "The required core pack has no manifold");
    }
    const manifolds: CoreManifold[] = [];
    const requiredManifolds = new Set<string>();
    for (const artifact of manifoldArtifacts) {
      request.signal.throwIfAborted();
      const loaded = await loadCoreManifolds(artifact, request, structuredHookProfile);
      requiredManifolds.add(loaded.key);
      manifolds.push(...loaded.manifolds);
    }
    for (const key of requiredManifolds) {
      if (!manifolds.some((manifold) => manifold.key === key)) {
        throw coreError(
          "CORE_PACK_RUNTIME_MISMATCH",
          `Core manifold manifolds/${key} has no fit for this exact browser runtime`,
        );
      }
    }
    const coreManifolds = [...manifolds];
    for (const archive of options.installedManifoldArchives ?? []) {
      request.signal.throwIfAborted();
      manifolds.push(...await loadInstalledManifolds(archive, request, structuredHookProfile));
    }
    const whiteners = await selectCoreWhiteners(
      whitenerTensor,
      whitenerSidecar,
      request,
    );
    const instruments = await BrowserInstrumentRegistry.load(request);
    return new BrowserFeasibilityCorePackCompiler(
      request.model.id,
      request.variant.runtimeIdentity.hiddenSize,
      [...request.variant.runtimeIdentity.layerMap],
      structuredHookProfile,
      manifolds,
      whiteners,
      instruments,
      request,
      coreManifolds,
    );
  }

  async withInstalledManifolds(
    archives: readonly Blob[],
    signal: AbortSignal,
  ): Promise<BrowserFeasibilityCorePackCompiler> {
    signal.throwIfAborted();
    const request = { ...this.loadRequest, signal };
    const manifolds = [...this.coreManifolds];
    for (const archive of archives) {
      signal.throwIfAborted();
      manifolds.push(...await loadInstalledManifolds(archive, request, this.structuredHookProfile));
    }
    signal.throwIfAborted();
    const next = new BrowserFeasibilityCorePackCompiler(
      this.modelId,
      this.hiddenSize,
      this.layerMap,
      this.structuredHookProfile,
      manifolds,
      this.whiteners,
      this.instruments,
      this.loadRequest,
      this.coreManifolds,
    );
    // The loaded library's unit sites outlive a manifold refresh (unit and slens/ probes
    // re-attach to the new compiler through them).
    next.setUnitSource(this.unitSource);
    return next;
  }

  fittingWhiteners(): ReadonlyMap<number, SerializedMahalanobisWhitener> {
    if (this.whiteners.size !== this.layerMap.length) {
      throw coreError(
        "CORE_PACK_WHITENER_UNAVAILABLE",
        "The required core pack has no complete neutral whitener for browser fitting",
      );
    }
    return new Map([...this.whiteners].map(([layer, whitener]) => [
      layer,
      {
        ...whitener,
        mean: whitener.mean.slice(),
        basis: whitener.basis.slice(),
        eigenvalues: whitener.eigenvalues.slice(),
        inverseScales: whitener.inverseScales.slice(),
      },
    ]));
  }

  /** The core pack's neutral whitener at `layer` (eigen form), or null: the metric the units
   *  `find` decomposes in (native `_whitener()`, the same fit). */
  neutralWhitener(layer: number): SerializedMahalanobisWhitener | null {
    return this.whiteners.get(layer) ?? null;
  }

  /** `find`'s selector as native resolves it (`ensure_profile_registered`): one steerable atom's
   *  per-layer direction, an instrument atom's directions (prepared before this call: `jlens/`
   *  words, `neuron/` writes, `slens/` atoms) or a 2-node manifold folded to its direction. */
  selectorDirections(selector: string): ReadonlyMap<number, Float32Array> {
    const notOne = () => coreError(
      "PROFILE_NOT_FOUND",
      `No profile registered for vector '${selector}' (find takes one steerable selector)`,
    );
    const parsed = parseSteeringExpression(selector);
    const term = parsed.terms[0];
    if (
      parsed.terms.length !== 1 || term.ablation || term.trigger !== null || term.layers !== null ||
      term.selector.onto !== null || term.selector.manifoldPosition !== null
    ) throw notOne();
    const atom = this.resolveAtom(term.selector.base);
    if (atom.kind === "instrument") return atom.directions;
    const profile = this.analyticsProfile(atom.manifold, atom.key);
    if (profile === null) {
      throw coreError(
        "PROFILE_NOT_FOUND",
        `'${selector}' is a manifold that does not fold to a single steering direction (not a 2-node affine subspace)`,
      );
    }
    return profile.directions;
  }

  /** The loaded packs (lens layers, S statistics, units), read on first use (B5's members). */
  instrumentRegistry(): BrowserInstrumentRegistry {
    return this.instruments;
  }

  /** The core pack's neutral statistics (unit statistics, `out` and `mid` means), or null. */
  async neutralStatistics(signal: AbortSignal): Promise<BrowserNeutralStatistics | null> {
    this.neutralStatisticsRead ??= readNeutralStatistics(
      this.loadRequest.artifacts.filter((artifact) => artifact.manifest.role === "core_pack"),
      { variant: this.loadRequest.variant, signal },
    )
      .catch((error) => {
        this.neutralStatisticsRead = null;
        throw error;
      });
    return this.neutralStatisticsRead;
  }

  instrumentDescriptor(): ReturnType<BrowserInstrumentRegistry["descriptor"]> {
    return this.instruments.descriptor();
  }

  jlensSources(): ReturnType<BrowserInstrumentRegistry["jlensSources"]> {
    return this.instruments.jlensSources();
  }

  async activateJlensSource(source: string): Promise<BrowserJlensGpuDictionary | null> {
    await this.instruments.activateJlensSource(source);
    if (this.lensLive) this.lensLiveLayers = this.resolveJlensLiveLayers();
    return this.jlensGpuDictionary();
  }

  async refreshJlensProbes(resolver: BrowserJlensRuntimeResolver): Promise<void> {
    const probes = [...this.attachedProbes.values()].filter(
      (probe): probe is Extract<ProbeInfo, { family: "lens" }> =>
        probe.family === "lens" && probe.namespace === "jlens",
    );
    await this.instruments.prepareJlensWords(probes.map((probe) => probe.word), resolver);
    for (const probe of probes) {
      const token = this.instruments.jlensToken(probe.word);
      this.attachedProbes.set(probe.name, {
        ...probe,
        layers: [...token.directions.keys()].sort((left, right) => left - right),
        token_id: token.tokenId,
      });
    }
    this.syncPinnedJlensWords();
  }

  saeGpuDictionary(): BrowserSaeGpuDictionary | null {
    return this.instruments.saeGpuDictionary(this.layerMap);
  }

  jlensGpuDictionary(): BrowserJlensGpuDictionary | null {
    return this.instruments.jlensGpuDictionary(this.layerMap);
  }

  exactSaeFitting(
    selector: string,
  ): ReturnType<BrowserInstrumentRegistry["exactSaeFitting"]> {
    return this.instruments.exactSaeFitting(selector);
  }

  async prepareJlensWords(
    words: readonly string[],
    resolver: BrowserJlensRuntimeResolver,
  ): Promise<void> {
    await this.instruments.prepareJlensWords(words, resolver);
  }

  async prepareJlensSelectors(
    expression: string,
    resolver: BrowserJlensRuntimeResolver,
  ): Promise<void> {
    if (!expression.trim()) return;
    const words: string[] = [];
    const parsed = parseSteeringExpression(expression);
    const addAtom = (atom: SteeringAtom | null): void => {
      if (atom?.namespace === "jlens") words.push(atom.concept);
    };
    for (const term of parsed.terms) {
      addAtom(term.selector.base);
      addAtom(term.selector.onto);
      // A gate reads the attached probe its key names (native resolve_gates), never a word.
      const gate = term.trigger?.gate?.probe;
      if (gate !== undefined) {
        const attached = this.attachedProbes.get(parseGateRef(gate).probe);
        if (attached?.family === "lens" && attached.namespace === "jlens") words.push(attached.word);
      }
    }
    await this.instruments.prepareJlensWords(words, resolver);
  }

  instrumentLiveState(): {
    geometry: boolean;
    lens: boolean;
    lensLayers: number[] | null;
    sae: boolean;
  } {
    return {
      geometry: this.geometryLive,
      lens: this.lensLive,
      lensLayers: this.lensLive ? [...this.lensLiveLayers ?? []] : null,
      sae: this.saeLive,
    };
  }

  setInstrumentLive(
    family: "geometry" | "lens" | "sae",
    enabled: boolean,
    layers?: readonly number[],
  ): void {
    const available = this.instruments.capabilities();
    if (family === "lens" && enabled && !available.jlens) {
      throw coreError("JLENS_PACK_UNAVAILABLE", "Install the compatible R-lens pack first");
    }
    if (family === "sae" && enabled && !available.sae) {
      throw coreError("SAE_PACK_UNAVAILABLE", "Install the compatible SAE pack first");
    }
    if (family === "geometry") this.geometryLive = enabled;
    if (family === "lens") {
      if (enabled) {
        const resolved = this.resolveJlensLiveLayers(layers);
        this.lensLive = true;
        this.lensLiveLayers = resolved;
      } else {
        this.lensLive = false;
        this.lensLiveLayers = null;
      }
    }
    if (family === "sae") this.saeLive = enabled;
  }

  private resolveJlensLiveLayers(layers?: readonly number[]): readonly number[] {
    const fitted = this.instruments.descriptor().jlens?.layers ?? [];
    const selected = layers === undefined
      ? [...fitted]
      : [...new Set(layers)].sort((left, right) => left - right);
    if (selected.length === 0) {
      throw coreError(
        "JLENS_PACK_LAYER_MISMATCH",
        "R-lens live readout needs at least one fitted layer",
      );
    }
    const missing = selected.filter((layer) =>
      !Number.isSafeInteger(layer) || layer < 0 || !fitted.includes(layer)
    );
    if (missing.length > 0) {
      throw coreError(
        "JLENS_PACK_LAYER_MISMATCH",
        `R-lens layers ${missing.join(", ")} are not fitted in the installed pack`,
      );
    }
    return selected;
  }

  validateLensToken(word: string): { word: string; token_id: number } {
    const token = this.instruments.jlensToken(word);
    return { word: token.word, token_id: token.tokenId };
  }

  validateSaeFeature(featureId: number): {
    id: number;
    label: string | null;
    layer: number;
    max_act: number | null;
  } {
    const feature = this.instruments.saeFeature(String(featureId));
    return {
      id: featureId,
      label: feature.label,
      layer: feature.layer,
      max_act: feature.maxAct,
    };
  }

  listProbes(): ProbeInfo[] {
    return [...this.attachedProbes.values()].map((probe) => structuredClone(probe));
  }

  probeHashes(): Record<string, string> {
    const hashes: Record<string, string> = {};
    for (const [name, probe] of this.attachedProbes) {
      if (probe.family === "geometry") {
        hashes[name] = this.geometryProbeHash(probe);
      } else if (probe.family === "units") {
        hashes[name] = this.unitProbeHash(probe);
      } else if (probe.family === "lens" && probe.namespace === "slens") {
        const digest = this.slensProbeHash(probe);
        if (digest !== null) hashes[name] = digest;
      } else {
        hashes[name] = this.instruments.probeHash(this.modelId, probe);
      }
    }
    return hashes;
  }

  /** Native `UnitsInstrument.probe_hash` (`units-readout-v2`): the unit and the reading scale
   *  its reading divides by (the scales' detail and the unit's raw values). */
  private unitProbeHash(probe: Extract<ProbeInfo, { family: "units" }>): string {
    const source = this.unitSource;
    const scales = source?.scales ?? null;
    if (source === null || scales === null || source.spec === null) return unitProbeDigest(this.modelId, probe.unit, "", []);
    const raw = probe.kind === "neuron"
      ? [scales.neuron[(probe.layer * 2) * source.spec.dFF + probe.index], scales.neuron[(probe.layer * 2 + 1) * source.spec.dFF + probe.index]]
      : [scales.head[probe.layer * source.spec.nHeads + probe.index]];
    return unitProbeDigest(this.modelId, probe.unit, scales.detail, raw);
  }

  /** Native `LensInstrument.probe_hash` for an `slens/` probe (`slens-footprint-z-v3`): null when
   *  the unit source has not prepared the S identity (a source swapped since the attach). */
  private slensProbeHash(probe: Extract<ProbeInfo, { family: "lens" }>): string | null {
    if (probe.token_id === null) return null;
    const identity = this.unitSource?.slensIdentity(probe.layers) ?? null;
    return identity === null ? null : slensProbeDigest(this.modelId, probe.word, probe.token_id, probe.layers, identity);
  }

  /** Reads off the device what `expression`'s `neuron/` pushes and `slens/` atoms and gates
   *  need before the synchronous compile (the unit source's `prepare`). */
  async prepareUnitSelectors(expression: string): Promise<void> {
    await this.unitSource?.prepare(expression);
  }

  /** The same for one probe selector (an `slens/` probe's atom and digest identity). */
  async prepareUnitProbe(selector: string): Promise<void> {
    await this.unitSource?.prepareProbe(selector);
  }

  hasAttachedProbes(): boolean {
    return this.attachedProbes.size > 0;
  }

  attachProbe(request: ProbeRequest): ProbeInfo {
    const selector = request.selector?.trim();
    if (!selector) {
      throw coreError("INVALID_PROBE", "Probe selector must be a non-empty string");
    }
    const name = request.name?.trim() || selector;
    if (name.length > 256) {
      throw coreError("INVALID_PROBE", "Probe name is too long");
    }
    const existing = this.attachedProbes.get(name);
    if (existing) return structuredClone(existing);

    let probe: ProbeInfo;
    if (selector.startsWith("jlens/")) {
      const token = this.instruments.jlensToken(selector.slice("jlens/".length));
      probe = {
        family: "lens",
        namespace: "jlens",
        name,
        layers: [...token.directions.keys()].sort((left, right) => left - right),
        intrinsic_dim: 1,
        feature_space: "readout",
        word: token.word,
        token_id: token.tokenId,
      };
    } else if (selector.startsWith("slens/")) {
      // An S-lens probe reads the residuals each block wrote, post-forward on the host side of
      // the device (B1 §2.1): no GPU measurement slot, gates read its score by name.
      if (this.unitSource === null) throw instrumentAtomUnavailable("slens");
      const word = selector.slice("slens/".length);
      const prepared = this.unitSource.slensAtom(word);
      if (prepared === null) {
        throw coreError("SLENS_ATOM_UNPREPARED", `${selector} was not resolved before attaching`);
      }
      probe = {
        family: "lens",
        namespace: "slens",
        name,
        layers: [...prepared.directions.keys()].sort((left, right) => left - right),
        intrinsic_dim: 1,
        feature_space: "readout",
        word,
        token_id: prepared.tokenId,
      };
    } else if (selector.startsWith("neuron/") || selector.startsWith("head/")) {
      const namespace = selector.startsWith("neuron/") ? "neuron" : "head";
      if (this.unitSource === null) throw instrumentAtomUnavailable(namespace);
      const unit = resolveUnitAtom(this.unitSource, namespace, selector.slice(namespace.length + 1));
      probe = {
        family: "units",
        name,
        unit: unitName(unit),
        kind: unit.kind,
        layer: unit.layer,
        index: unit.index,
        layers: [unit.layer],
        intrinsic_dim: 1,
        feature_space: "unit-readout",
      };
    } else if (selector.startsWith("sae/")) {
      const rawFeature = selector.slice("sae/".length);
      if (!/^(0|[1-9]\d*)$/u.test(rawFeature)) {
        throw coreError("INVALID_SAE_FEATURE", "SAE probe selectors use sae/<feature number>");
      }
      const feature = this.instruments.saeFeature(rawFeature);
      probe = {
        family: "sae",
        name,
        layers: [feature.layer],
        intrinsic_dim: 1,
        feature_space: "sae-readout",
        feature_id: Number(rawFeature),
        label: feature.label,
        max_act: feature.maxAct,
      };
    } else {
      const manifold = this.resolveProbeManifold(selector);
      this.assertGeometryProbeLowerable(manifold);
      probe = this.geometryProbeInfo(manifold, name, request.top_n);
    }
    this.requireProbeCapacity(probe);
    this.attachedProbes.set(name, probe);
    this.syncPinnedJlensWords();
    return structuredClone(probe);
  }

  /** Each family's own capacity: a probe slot on the device for the geometry, R-lens and SAE
   *  probes (`maxProbes`), a unit tap for a unit probe (`maxUnitTaps`); `slens/` probes read on
   *  the host from the forward's capture and take neither. */
  private requireProbeCapacity(probe: ProbeInfo): void {
    const attached = [...this.attachedProbes.values()];
    if (probe.family === "units") {
      const capacity = this.structuredHookProfile.maxUnitTaps;
      if (attached.filter((entry) => entry.family === "units").length >= capacity) {
        throw coreError("CORE_PACK_PROBE_LIMIT", `The browser GPU program supports at most ${capacity} attached unit probes at once`);
      }
      return;
    }
    if (probe.family === "lens" && probe.namespace === "slens") return;
    const slotted = attached.filter((entry) => entry.family !== "units" && !(entry.family === "lens" && entry.namespace === "slens"));
    if (slotted.length >= this.structuredHookProfile.maxProbes) {
      throw coreError(
        "CORE_PACK_PROBE_LIMIT",
        `The browser GPU program supports at most ${this.structuredHookProfile.maxProbes} attached probes at once`,
      );
    }
  }

  detachProbe(name: string): void {
    this.attachedProbes.delete(name);
    this.syncPinnedJlensWords();
  }

  probeGeometry(name: string): ProbeGeometryResponse {
    const probe = this.attachedProbes.get(name);
    if (!probe) throw coreError("PROBE_NOT_FOUND", `Probe ${name} is not attached`);
    if (probe.family !== "geometry") {
      throw coreError("PROBE_GEOMETRY_UNAVAILABLE", "Only geometry probes have a manifold plot");
    }
    const manifold = this.resolveProbeManifold(probe.manifold);
    const layers: ProbeGeometryResponse["layers"] = {};
    const ranks = new Set<number>();
    for (const layer of fittedLayerNumbers(manifold.tensors)) {
      const prefix = `layer_${layer}`;
      const shape = manifold.tensors.description.shapes.get(`${prefix}.basis`);
      const whitener = this.whiteners.get(layer);
      if (!shape || shape.length !== 2 || !whitener) continue;
      const rank = shape[0];
      const basis = manifold.tensors.tensor(`${prefix}.basis`, [rank, this.hiddenSize]).data;
      const gram = restrictedInverseCovariance(basis, rank, whitener);
      const chol = lowerCholesky(gram, rank);
      const curved = manifold.tensors.keys.includes(`${prefix}.node_params`);
      const curve = curved
        ? coreGeometryCurve(
            manifold,
            layer,
            structuredCurveDomain(manifold.domain, this.structuredHookProfile),
            this.hiddenSize,
          )
        : null;
      const nodeCoordinates = curved
        ? flattenRows(
            coreNodeCoordinates(manifold.tensors, manifold.labels.length).map((coordinate) =>
              evaluateCoreCurve(curve!, coordinate)
            ),
            rank,
          )
        : manifold.tensors.tensor(
            `${prefix}.node_coords`,
            [manifold.labels.length, rank],
          ).data;
      const nodeWhite = multiplyRows(nodeCoordinates, manifold.labels.length, rank, chol);
      const neutralWhite = curve === null
        ? new Float32Array(rank)
        : multiplyRows(evaluateCoreCurve(curve, curve.origin), 1, rank, chol);
      ranks.add(rank);
      layers[String(layer)] = {
        layer,
        rank,
        intrinsic_dim: manifoldIntrinsicDimension(manifold),
        is_affine: !curved,
        node_white: rows(nodeWhite, manifold.labels.length, rank).map((row) => [...row]),
        neutral_white: [...neutralWhite],
        pca_rotation: null,
        explained_variance_pcs: null,
        mahalanobis_share: manifold.mahalanobisShares.get(layer)!,
        overlay: null,
      };
    }
    return {
      name,
      manifold: manifold.key,
      intrinsic_dim: manifoldIntrinsicDimension(manifold),
      is_affine: probe.is_affine,
      node_labels: [...manifold.labels],
      rank_uniform: ranks.size <= 1,
      layers,
    };
  }

  listManifolds(): ManifoldInfo[] {
    const groups = new Map<string, CoreManifold[]>();
    for (const manifold of this.manifolds) {
      const variants = groups.get(manifold.key) ?? [];
      variants.push(manifold);
      groups.set(manifold.key, variants);
    }
    return [...groups.values()].map((variants) => {
      const info = mergedManifoldInfo(variants);
      const { nodes: _nodes, fitted: _fitted, ...summary } = info;
      return structuredClone(summary);
    });
  }

  getManifold(namespace: string, name: string): ManifoldInfo {
    const manifold = this.manifolds.find((candidate) =>
      candidate.namespace === namespace && candidate.name === name
    );
    if (!manifold) {
      throw coreError(
        "CORE_PACK_SELECTOR_NOT_FOUND",
        `Installed browser packs do not contain manifold ${namespace}/${name}`,
      );
    }
    return structuredClone(mergedManifoldInfo(this.manifolds.filter((candidate) =>
      candidate.namespace === namespace && candidate.name === name
    )));
  }

  listProfiles(): ProfileListResponse {
    return {
      profiles: [...this.analyticsProfiles().values()]
        .map((profile) => structuredClone(profile.info))
        .sort((left, right) => left.name.localeCompare(right.name)),
    };
  }

  getProfile(name: string): VectorInfo {
    const profile = this.analyticsProfiles().get(name);
    if (!profile) {
      throw coreError("PROFILE_NOT_FOUND", `Browser profile ${name} is not installed`);
    }
    return structuredClone(profile.info);
  }

  profileCorrelation(requested?: readonly string[] | null): CorrelationData {
    const profiles = this.analyticsProfiles();
    const names = requested?.length ? [...requested] : [...profiles.keys()].sort();
    const missing = names.filter((name) => !profiles.has(name));
    if (missing.length > 0) {
      throw coreError("PROFILE_NOT_FOUND", `Browser profiles are not installed: ${missing.join(", ")}`);
    }
    const matrix: CorrelationData["matrix"] = Object.fromEntries(
      names.map((name) => [name, {}]),
    );
    const layersShared: Record<string, number> = {};
    for (let leftIndex = 0; leftIndex < names.length; leftIndex += 1) {
      const leftName = names[leftIndex];
      const left = profiles.get(leftName)!;
      for (let rightIndex = 0; rightIndex < names.length; rightIndex += 1) {
        const rightName = names[rightIndex];
        if (rightIndex < leftIndex) {
          matrix[leftName][rightName] = matrix[rightName][leftName];
          continue;
        }
        if (leftIndex === rightIndex) {
          matrix[leftName][rightName] = 1;
          continue;
        }
        const right = profiles.get(rightName)!;
        const shared = [...left.directions.keys()]
          .filter((layer) => right.directions.has(layer) && this.whiteners.has(layer))
          .sort((a, b) => a - b);
        layersShared[[leftName, rightName].sort().join("__")] = shared.length;
        let numerator = 0;
        let denominator = 0;
        for (const layer of shared) {
          const whitener = this.whiteners.get(layer)!;
          const leftDirection = left.directions.get(layer)!;
          const rightDirection = right.directions.get(layer)!;
          const inverseLeft = applySerializedInverse(whitener, leftDirection);
          const inverseRight = applySerializedInverse(whitener, rightDirection);
          const leftNormSquared = Math.max(0, dot(leftDirection, inverseLeft));
          const rightNormSquared = Math.max(0, dot(rightDirection, inverseRight));
          if (leftNormSquared < 1e-12 || rightNormSquared < 1e-12) continue;
          numerator += dot(leftDirection, inverseRight);
          denominator += Math.sqrt(leftNormSquared * rightNormSquared);
        }
        matrix[leftName][rightName] = denominator < 1e-12
          ? null
          : roundSix(numerator / denominator);
      }
    }
    return { names, matrix, layers_shared: layersShared };
  }

  profilePairwise(a: string, b: string): PairwiseCompareResponse {
    const profiles = this.analyticsProfiles();
    const profileA = profiles.get(a);
    const profileB = profiles.get(b);
    if (!profileA || !profileB) {
      const missing = [!profileA ? a : null, !profileB ? b : null].filter(Boolean);
      throw coreError("PROFILE_NOT_FOUND", `Browser profiles are not installed: ${missing.join(", ")}`);
    }
    const layersA = [...profileA.directions.keys()].sort((left, right) => left - right);
    const layersB = [...profileB.directions.keys()].sort((left, right) => left - right);
    const matrix = layersA.map((layerA) => {
      const whitener = this.whiteners.get(layerA);
      const directionA = profileA.directions.get(layerA)!;
      if (!whitener) return layersB.map(() => null);
      const inverseA = applySerializedInverse(whitener, directionA);
      const normA = Math.max(0, dot(directionA, inverseA));
      if (normA < 1e-12) return layersB.map(() => null);
      return layersB.map((layerB) => {
        const directionB = profileB.directions.get(layerB)!;
        const inverseB = applySerializedInverse(whitener, directionB);
        const normB = Math.max(0, dot(directionB, inverseB));
        const denominator = Math.sqrt(normA * normB);
        return denominator < 1e-12
          ? null
          : roundSix(dot(directionA, inverseB) / denominator);
      });
    });
    return {
      a,
      b,
      metric: "mahalanobis",
      layers_a: layersA,
      layers_b: layersB,
      matrix,
      model: this.modelId,
    };
  }

  private analyticsProfiles(): ReadonlyMap<string, BrowserAnalyticsProfile> {
    const profiles = new Map<string, BrowserAnalyticsProfile>();
    for (const manifold of this.manifolds) {
      const profile = this.analyticsProfile(manifold, manifold.selectorKey);
      if (profile) profiles.set(profile.info.name, profile);
    }
    for (const [name, probe] of this.attachedProbes) {
      if (probe.family !== "geometry" || profiles.has(name)) continue;
      const manifold = this.resolveProbeManifold(probe.manifold);
      const profile = this.analyticsProfile(manifold, name);
      if (profile) profiles.set(name, profile);
    }
    return profiles;
  }

  private geometryProbeHash(
    probe: Extract<ProbeInfo, { family: "geometry" }>,
  ): string {
    const manifold = this.resolveProbeManifold(probe.manifold);
    const layers = fittedLayerNumbers(manifold.tensors);
    const folded = layers.length > 0 && layers.every((layer) => {
      const prefix = `layer_${layer}`;
      const shape = manifold.tensors.description.shapes.get(`${prefix}.basis`);
      return shape?.length === 2 && shape[0] === 1 &&
        !manifold.tensors.keys.includes(`${prefix}.node_params`);
    });
    const hash = sha256.create();
    for (const layer of layers) {
      const prefix = `layer_${layer}`;
      const basisShape = manifold.tensors.description.shapes.get(`${prefix}.basis`)!;
      const rank = basisShape[0];
      if (folded) {
        const share = manifold.mahalanobisShares.get(layer)!;
        const basis = manifold.tensors.tensor(
          `${prefix}.basis`,
          [1, this.hiddenSize],
        ).data;
        updateFp32Hash(hash, Float32Array.from(basis, (value) => value * share));
        continue;
      }
      updateFp32Hash(hash, manifold.tensors.tensor(
        `${prefix}.mean`,
        [this.hiddenSize],
      ).data);
      updateFp32Hash(hash, manifold.tensors.tensor(
        `${prefix}.basis`,
        [rank, this.hiddenSize],
      ).data);
      updateFp32Hash(hash, manifold.tensors.tensor(
        `${prefix}.read`,
        [rank, this.hiddenSize],
      ).data);
      if (manifold.tensors.keys.includes(`${prefix}.node_coords`)) {
        updateFp32Hash(hash, manifold.tensors.tensor(
          `${prefix}.node_coords`,
          [manifold.labels.length, rank],
        ).data);
      }
    }
    return bytesToHex(hash.digest());
  }

  private analyticsProfile(
    manifold: CoreManifold,
    name: string,
  ): BrowserAnalyticsProfile | null {
    const directions = new Map<number, Float32Array>();
    for (const layer of fittedLayerNumbers(manifold.tensors)) {
      const prefix = `layer_${layer}`;
      const shape = manifold.tensors.description.shapes.get(`${prefix}.basis`);
      if (
        !shape || shape.length !== 2 || shape[0] !== 1 ||
        manifold.tensors.keys.includes(`${prefix}.node_params`)
      ) return null;
      const share = manifold.mahalanobisShares.get(layer);
      if (share === undefined || !Number.isFinite(share)) return null;
      const basis = manifold.tensors.tensor(`${prefix}.basis`, [1, this.hiddenSize]).data;
      directions.set(layer, Float32Array.from(basis, (value) => value * share));
    }
    if (directions.size === 0) return null;
    return {
      info: {
        name,
        layers: [...directions.keys()].sort((left, right) => left - right),
        metadata: {
          source: "browser-core-pack",
          manifold: manifold.key,
          selector_key: manifold.selectorKey,
          feature_space: manifold.featureSpace,
          variant: manifold.variant,
          variant_identity: manifold.variantIdentity,
        },
      },
      directions,
    };
  }

  /** The loaded library's unit sites and the core pack's unit statistics (B5): `neuron/`,
   *  `head/` and `slens/` atoms, unit probes and unit gates lower through it; without one they
   *  are refused (`instrumentAtomUnavailable`). Directions it serves are prepared by the caller
   *  (async, on the device) before a compile. */
  setUnitSource(source: BrowserUnitProgramSource | null): void {
    this.unitSource = source;
    this.unitSourceVersion += 1;
    this.compiledProgram = null;
  }

  compile(
    expression: string,
    probeRequests?: readonly ProbeRequest[],
  ): StructuredHookProgramBuffers {
    return structuredClone(this.compileForGeneration(expression, probeRequests));
  }

  /** Borrowed immutable payload; controls and warm state are owned by each generation. */
  compileForGeneration(
    expression: string,
    probeRequests?: readonly ProbeRequest[],
  ): StructuredHookProgramBuffers {
    const key = JSON.stringify([
      expression, probeRequests ?? this.listProbes(),
      this.instrumentLiveState(), this.instrumentDescriptor(), this.unitSourceVersion,
    ]);
    if (this.compiledProgram?.key === key) return this.compiledProgram.program;
    const program = this.compileUncached(expression, probeRequests);
    this.compiledProgram = { key, program };
    return program;
  }

  private compileUncached(
    expression: string,
    probeRequests?: readonly ProbeRequest[],
  ): StructuredHookProgramBuffers {
    if (probeRequests === undefined) return this.compileCurrent(expression);
    const previous = [...this.attachedProbes.entries()];
    this.attachedProbes.clear();
    try {
      for (const request of probeRequests) this.attachProbe(request);
      return this.compileCurrent(expression);
    } finally {
      this.attachedProbes.clear();
      for (const [name, probe] of previous) this.attachedProbes.set(name, probe);
      this.syncPinnedJlensWords();
    }
  }

  private compileCurrent(expression: string): BrowserCompiledSteering {
    // The grammar's own fold first, so a malformed or ambiguous expression is refused exactly as
    // drowse refuses it, before the compiler's narrower refusals.
    const parsed = expression.trim()
      ? parseSteeringExpression(expression)
      : { terms: [] };
    const ir = foldSteering(parsed, { manifolds: this.steeringManifolds() });
    const resolved = this.foldResolvedTerms(
      parsed.terms.map((term) => this.resolveTerm(term)),
    );
    const placements = this.placeTerms(resolved);
    const activeRole = this.activeRole(resolved);
    const program: BrowserCompiledSteering = this.compileStructured(resolved);
    program.activeRole = activeRole;
    program.steeringExpression = formatSteering(ir);
    program.steeringParts = steeringParts(ir);
    program.steeringTerms = placements;
    program.steeringDepth = this.layerMap.length;
    return program;
  }

  private activeRole(terms: readonly ResolvedCoreTerm[]): string | null {
    const explicitRoles = new Set<string>();
    for (const term of terms) {
      if (term.base.kind === "manifold" && term.base.requestedRole !== null) {
        explicitRoles.add(term.base.requestedRole);
      }
      if (
        term.projection?.onto.kind === "manifold" &&
        term.projection.onto.requestedRole !== null
      ) {
        explicitRoles.add(term.projection.onto.requestedRole);
      }
    }
    if (explicitRoles.size > 1) {
      throw coreError(
        "CORE_PACK_ROLE_CONFLICT",
        `A browser steering expression cannot compose different role baselines (${[...explicitRoles].join(", ")})`,
      );
    }
    const explicitRole = explicitRoles.values().next().value ?? null;
    if (explicitRole !== null) return explicitRole;

    let inferredRole: string | null = null;
    let inferredCoefficient = -1;
    for (const term of terms) {
      if (term.base.kind !== "manifold" || term.manifoldPosition === null) continue;
      let role: string | null;
      try {
        role = nearestNodeRole(term.base.manifold, term.manifoldPosition);
      } catch {
        continue;
      }
      if (role === null) continue;
      const coefficient = Math.abs(term.coefficient);
      if (inferredRole === null || coefficient > inferredCoefficient) {
        inferredRole = role;
        inferredCoefficient = coefficient;
      }
    }
    return inferredRole;
  }

  private compileStructured(
    allTerms: readonly ResolvedCoreTerm[],
  ): StructuredHookProgramBuffers {
    // Unit ablations (`!neuron/…`, `!head/…`) act inside the block (the unit section, B1 §3.1);
    // every other term lowers to the post-block program.
    const unitTerms = allTerms.filter((term) =>
      term.ablation && term.base.kind === "instrument" && term.base.family === "units"
    );
    const terms = allTerms.filter((term) => !unitTerms.includes(term));
    for (const term of terms) {
      if (term.base.kind === "instrument" && term.base.family === "units" && term.base.directions.size === 0) {
        throw coreError(
          "UNIT_DIRECTION_UNPREPARED",
          `${term.base.key}'s write was not read off the model before compiling`,
        );
      }
    }
    if (this.whiteners.size !== this.layerMap.length) {
      throw coreError(
        "CORE_PACK_WHITENER_UNAVAILABLE",
        "Structured browser steering requires the exact neutral whitener",
      );
    }
    const curved = terms.filter((term) => isCurvedPositionTerm(term));
    if (curved.length > this.structuredHookProfile.maxCurves) {
      throw coreError(
        "CORE_PACK_CURVED_GROUP_LIMIT",
        `The browser GPU program supports at most ${this.structuredHookProfile.maxCurves} simultaneous curved terms`,
      );
    }
    const affineTerms = terms.filter((term) => !isCurvedPositionTerm(term));
    const affineTermGroups = groupAffineTermsByTrigger(affineTerms);
    if (affineTermGroups.length > this.structuredHookProfile.maxAffineGroups) {
      throw coreError(
        "CORE_PACK_AFFINE_GROUP_LIMIT",
        `The browser GPU program supports at most ${this.structuredHookProfile.maxAffineGroups} independently gated affine groups`,
      );
    }
    // Native `resolve_gates`: a gate reads the attached probe its key names (any unit, attached
    // or not, through a live tap), by its verbatim scalar key; a key naming no attached probe and
    // no unit never fires.
    const gates = new Map<string, GateResolution>();
    for (const term of allTerms) {
      const key = term.trigger?.gate?.probe;
      if (key !== undefined && !gates.has(key)) gates.set(key, this.resolveGate(key));
    }
    const gatedProbes = (kind: GateResolution["kind"]) =>
      [...new Set([...gates.values()].flatMap((gate) => gate.kind === kind && gate.kind !== "never" ? [gate.probe] : []))];
    const unitGateNames = gatedProbes("unit");
    const requiredGateNames = [...gatedProbes("slot"), ...gatedProbes("geometry")];
    if (requiredGateNames.length > this.structuredHookProfile.maxProbes) {
      throw coreError(
        "CORE_PACK_PROBE_LIMIT",
        `A structured browser program supports at most ${this.structuredHookProfile.maxProbes} live gate probes`,
      );
    }
    const gateNames = this.liveProbeNames(requiredGateNames);
    const layers: StructuredHookLayer[] = this.layerMap.map(() => ({
      affineGroups: affineTermGroups.map(() => disabledStructuredGroup(this.hiddenSize)),
      probes: gateNames.map(() => disabledStructuredProbe(this.hiddenSize)),
      curves: [],
      geometryProbes: [],
    }));
    const affineControls: Array<StructuredHookControl | null> = Array.from(
      { length: this.layerMap.length * this.structuredHookProfile.maxAffineGroups },
      () => null,
    );
    const curveControls: Array<StructuredHookControl | null> = Array.from(
      { length: this.layerMap.length * this.structuredHookProfile.maxCurves },
      () => null,
    );
    const gateSlots = new Map<string, readonly StructuredMeasurementSlot[]>();
    const geometrySchemas: Array<StructuredGeometryMeasurementProbe | null> =
      Array.from({ length: this.structuredHookProfile.maxGeometryProbes }, () => null);
    gateNames.forEach((name, probeIndex) => {
      const keys = [...gates.entries()].flatMap(([key, gate]) =>
        gate.kind === "geometry" && gate.probe === name ? [key] : []
      );
      const installed = this.installGateProbe(name, probeIndex, layers, keys);
      gateSlots.set(name, installed.slots);
      if (installed.geometry !== null) {
        geometrySchemas[probeIndex] = installed.geometry;
        // A channel the readout does not produce never scores: its gate never fires (native).
        for (const key of keys) {
          if (!installed.geometry.scoreKeys.includes(key)) gates.set(key, { kind: "never" });
        }
      }
    });
    const hasActiveGeometry = geometrySchemas.some((schema) => schema !== null);
    if (hasActiveGeometry) {
      this.layerMap.forEach((layer, programIndex) => {
        layers[programIndex].geometryWhitener = this.structuredGeometryWhitener(layer);
      });
    }

    const curveMaps = curved.map((term) => this.structuredCurves(term));
    const curvedFrameByLayer = new Map<number, CurvedFrame>();
    // drowse's `OverlappingManifoldError` (steering/hooks.py), naming the layer's first curved
    // manifold and the one that overlaps it. The cosines are Mahalanobis: a frame's write rows
    // against the earlier frames' reads.
    const curvedOwner = new Map<number, string>();
    curveMaps.forEach((curves, curveIndex) => {
      const name = (curved[curveIndex].base as ResolvedManifoldAtom).manifold.name;
      for (const [programIndex, curve] of curves) {
        const previous = curvedFrameByLayer.get(programIndex) ?? { basis: [], read: [] };
        const overlap = maximumBasisOverlap(curve.basis, previous.read);
        if (overlap > 1e-3) {
          throw Object.assign(coreError(
            "CORE_PACK_CURVED_OVERLAP",
            `manifolds '${curvedOwner.get(programIndex)}' and '${name}' both cover layer ${this.layerMap[programIndex]} with non-orthogonal subspaces (max |cosine| = ${overlap.toFixed(3)} > 0.001); curved manifolds sharing a layer must be orthogonal`,
          ), { status: 422, recoverable: false });
        }
        if (!curvedOwner.has(programIndex)) curvedOwner.set(programIndex, name);
        curvedFrameByLayer.set(programIndex, {
          basis: [...previous.basis, ...curve.basis],
          read: [...previous.read, ...curve.read],
        });
      }
    });

    affineTermGroups.forEach((termGroup, groupIndex) => {
      const groupByLayer = this.structuredAffineGroups(termGroup.terms);
      for (const [programIndex, group] of groupByLayer) {
        const lowered = orthogonalizeAffineGroup(group, curvedFrameByLayer.get(programIndex));
        if (lowered === null) continue;
        (layers[programIndex].affineGroups as StructuredAffineGroup[])[groupIndex] = lowered;
        affineControls[programIndex * this.structuredHookProfile.maxAffineGroups + groupIndex] =
          hookControl(termGroup.trigger, gates, gateSlots);
      }
    });

    curveMaps.forEach((curves, curveIndex) => {
      for (const [programIndex, curve] of curves) {
        (layers[programIndex].curves as StructuredCurve[])[curveIndex] = curve;
        curveControls[programIndex * this.structuredHookProfile.maxCurves + curveIndex] =
          hookControl(curved[curveIndex].trigger, gates, gateSlots);
      }
    });
    const program = compileStructuredHookProgram(
      this.hiddenSize,
      layers,
      this.structuredHookProfile,
      this.instruments.jlensProgram(
        gateNames.map((name) => this.instrumentProgramProbeName(name)),
        this.jlensProgramLayerMap(gateNames),
        this.structuredHookProfile.maxProbes,
        this.lensLive,
      ),
    );
    if (program.jLensBindingId !== undefined) {
      const displayLayers = new Set(this.lensLiveLayers ?? this.layerMap);
      program.jLensReadoutLayerIndices = Int32Array.from(
        [...program.jLensLayerIndices!].filter((index) => this.lensLive && displayLayers.has(this.layerMap[index])),
      );
    }
    const descriptor = this.instruments.descriptor();
    program.measurementSchema = {
      layerMap: [...this.layerMap],
      modelLayerCount: this.layerMap.length,
      probes: Array.from({ length: this.structuredHookProfile.maxProbes }, (_, probeIndex) =>
        geometrySchemas[probeIndex] == null
          ? this.measurementProbe(gateNames[probeIndex])
          : null
      ),
      ...(hasActiveGeometry ? { geometryProbes: geometrySchemas } : {}),
      lensSource: descriptor.jlens?.source ?? null,
      saeSource: descriptor.sae?.source ?? null,
      saeLayer: descriptor.sae?.layer ?? null,
      saeFeatureCount: descriptor.sae?.features ?? null,
      lensReadout: this.lensLive,
      saeReadout: this.saeLive,
      saeFeatureMetadata: this.instruments.saeFeatureMetadata(),
    };
    if (this.saeLive) {
      const dictionary = this.saeGpuDictionary();
      if (dictionary === null) {
        throw coreError(
          "SAE_PACK_UNAVAILABLE",
          "Install the compatible SAE pack to use the live SAE readout",
        );
      }
      program.saeBindingId = dictionary.bindingId;
    }
    program.controls = { affine: affineControls, curve: curveControls };
    validateStructuredHookProgram(program);
    return this.withSlensReads(this.withUnitSection(program, unitTerms, unitGateNames, gates, gateSlots));
  }

  /** The post-forward `slens/` reads (HOOK_ABI §5): every attached `slens/` probe, scored over
   *  the bridge's `[L, d]` capture, which the program then asks for (`hiddenCapture`). */
  private withSlensReads(program: StructuredHookProgramBuffers): StructuredHookProgramBuffers {
    const probes: StructuredSlensProbe[] = [];
    for (const [name, probe] of this.attachedProbes) {
      if (probe.family !== "lens" || probe.namespace !== "slens" || probe.token_id === null) continue;
      probes.push({ name, word: probe.word, tokenId: probe.token_id });
    }
    if (probes.length === 0) return program;
    const out: StructuredHookProgramBuffers = {
      ...program,
      hiddenCapture: true,
      measurementSchema: { ...program.measurementSchema!, slensProbes: probes },
    };
    validateStructuredHookProgram(out);
    return out;
  }

  /** What a gate key reads (native `resolve_gates` over `parse_gate_ref`): a unit (an attached
   *  unit probe, or any `neuron/`/`head/` unit, checked against the model) through its live tap;
   *  an attached `slens/` probe by name; an attached R-lens or SAE probe through its device slots;
   *  an attached geometry probe through the geometry readout, any channel it produces. The scalar
   *  families read their one axis: another channel is native's `UnsupportedProbeChannelError`,
   *  another axis never fires. A key naming no attached probe and no unit never fires (native's
   *  gate stays inactive). */
  private resolveGate(key: string): GateResolution {
    const ref = parseGateRef(key);
    const attached = this.attachedProbes.get(ref.probe);
    const scalar = (family: string): boolean => {
      if (ref.channel.kind !== "axis") {
        throw Object.assign(new Error(
          `probe gate '${key}' references the ${GATE_CHANNEL_WORDS[ref.channel.kind]} channel, which a ${family} probe does not produce`,
        ), { code: "UnsupportedProbeChannelError", status: 400, recoverable: true });
      }
      return ref.channel.index === 0;
    };
    const unitNamespace = ref.probe.startsWith("neuron/") ? "neuron" : ref.probe.startsWith("head/") ? "head" : null;
    if (attached?.family === "units" || (attached === undefined && unitNamespace !== null)) {
      if (attached === undefined) {
        if (this.unitSource === null) throw instrumentAtomUnavailable(unitNamespace!);
        resolveUnitAtom(this.unitSource, unitNamespace!, ref.probe.slice(unitNamespace!.length + 1));
      }
      return scalar("units") ? { kind: "unit", probe: ref.probe } : { kind: "never" };
    }
    if (attached === undefined) return { kind: "never" };
    switch (attached.family) {
      case "lens":
        if (!scalar("lens")) return { kind: "never" };
        return { kind: attached.namespace === "slens" ? "slens" : "slot", probe: ref.probe };
      case "sae":
        return scalar("sae") ? { kind: "slot", probe: ref.probe } : { kind: "never" };
      case "geometry":
        return { kind: "geometry", probe: ref.probe };
      default:
        return { kind: "never" };
    }
  }

  /** The unit section (B1 §3.1, §2.2): one ablation per unit term, grouped by trigger in first
   *  registration order (≤ `maxUnitGroups`), and a live tap for each attached unit probe and
   *  each unit a gate reads. Capacities and duplicates refuse in the encoder's words. */
  private withUnitSection(
    program: StructuredHookProgramBuffers,
    unitTerms: readonly ResolvedCoreTerm[],
    unitGateNames: readonly string[],
    gates: ReadonlyMap<string, GateResolution>,
    gateSlots: ReadonlyMap<string, readonly StructuredMeasurementSlot[]>,
  ): StructuredHookProgramBuffers {
    const tapped: Array<{ name: string; unit: UnitRef }> = [];
    for (const [name, probe] of this.attachedProbes) {
      if (probe.family === "units") tapped.push({ name, unit: { kind: probe.kind, layer: probe.layer, index: probe.index } });
    }
    for (const name of unitGateNames) {
      if (tapped.some((entry) => entry.name === name)) continue;
      const namespace = name.startsWith("neuron/") ? "neuron" : "head";
      const source = this.unitSource;
      if (source === null) throw instrumentAtomUnavailable(namespace);
      tapped.push({ name, unit: resolveUnitAtom(source, namespace, name.slice(namespace.length + 1)) });
    }
    if (unitTerms.length === 0 && tapped.length === 0) return program;
    const source = this.unitSource;
    if (source === null || source.spec === null) {
      throw instrumentAtomUnavailable(unitTerms.length > 0 ? unitTerms[0].base.key.split("/")[0] : "neuron");
    }
    const triggers: string[] = [];
    const controls: Array<StructuredHookControl | null> = [];
    const ablations = unitTerms.map((term) => {
      const unit = (term.base as ResolvedInstrumentAtom).unit!;
      const key = JSON.stringify(term.trigger);
      let group = triggers.indexOf(key);
      if (group < 0) {
        group = triggers.length;
        triggers.push(key);
        controls.push(term.trigger === null ? null : hookControl(term.trigger, gates, gateSlots));
      }
      return structuredUnitAblationTerm(unit.kind, unit.layer, unit.index, term.coefficient, unitMean(source, unit), group);
    });
    const taps = tapped.map((entry, slot) => unitTap(source, entry.name, entry.unit, slot));
    return withStructuredUnitSection(
      program,
      { dFF: source.spec.dFF, nHeads: source.spec.nHeads, headDim: source.spec.headDim },
      { ablations, taps, controls },
    );
  }

  private syncPinnedJlensWords(): void {
    this.instruments.setPinnedJlensWords(
      [...this.attachedProbes.values()].flatMap((probe) =>
        probe.family === "lens" && probe.namespace === "jlens" ? [probe.word] : []
      ),
    );
  }

  private liveProbeNames(required: readonly string[]): string[] {
    const names = [...required];
    for (const [name, probe] of this.attachedProbes) {
      if (probe.family === "geometry" && !this.geometryLive) continue;
      if (probe.family === "units" || (probe.family === "lens" && probe.namespace === "slens")) continue;
      if (!names.includes(name)) names.push(name);
    }
    if (names.length > this.structuredHookProfile.maxProbes) {
      throw coreError(
        "CORE_PACK_PROBE_LIMIT",
        `The browser GPU program supports at most ${this.structuredHookProfile.maxProbes} required and attached probes at once`,
      );
    }
    return names;
  }

  private jlensProgramLayerMap(probeNames: readonly string[]): readonly number[] {
    if (probeNames.some((name) => this.instrumentProgramProbeName(name).startsWith("jlens/"))) return this.layerMap;
    if (!this.lensLive || this.lensLiveLayers === null) return this.layerMap;
    const selected = new Set(this.lensLiveLayers);
    return this.layerMap.map((layer) => selected.has(layer) ? layer : -1);
  }

  private resolveProbeManifold(selector: string): CoreManifold {
    let parsed;
    try {
      parsed = parseSteeringExpression(selector);
    } catch {
      throw coreError(
        "CORE_PACK_SELECTOR_NOT_FOUND",
        `Installed browser packs do not contain manifold ${selector}`,
      );
    }
    if (
      parsed.terms.length !== 1 || parsed.terms[0].ablation ||
      parsed.terms[0].selector.projection !== null ||
      parsed.terms[0].selector.manifoldPosition !== null ||
      parsed.terms[0].trigger !== null
    ) {
      throw coreError(
        "CORE_PACK_SELECTOR_NOT_FOUND",
        `Installed browser packs do not contain manifold ${selector}`,
      );
    }
    return this.resolveManifoldAtom(parsed.terms[0].selector.base).manifold;
  }

  private assertGeometryProbeLowerable(manifold: CoreManifold): void {
    const fitted = fittedLayerNumbers(manifold.tensors);
    if (fitted.length === 0) {
      throw coreError("CORE_PACK_GATE_PROBE_UNAVAILABLE", `Probe ${manifold.key} has no fitted layers`);
    }
    for (const layer of fitted) {
      const prefix = `layer_${layer}`;
      if (!this.layerMap.includes(layer)) {
        throw coreError(
          "CORE_PACK_LAYER_MISMATCH",
          `Probe ${manifold.key} layer ${layer} is outside the compiled model layer map`,
        );
      }
      const shape = manifold.tensors.description.shapes.get(`${prefix}.basis`);
      if (
        !shape || shape.length !== 2 || shape[1] !== this.hiddenSize ||
        shape[0] < 1 || shape[0] > this.structuredHookProfile.maxRank
      ) {
        throw coreError(
          "CORE_PACK_GATE_PROBE_UNAVAILABLE",
          `Probe ${manifold.key} layer ${layer} has a basis this browser model library cannot read`,
        );
      }
      if (!this.whiteners.has(layer)) {
        throw coreError(
          "CORE_PACK_WHITENER_UNAVAILABLE",
          `Probe ${manifold.key} requires a neutral whitener for layer ${layer}`,
        );
      }
    }
  }

  private geometryProbeInfo(
    manifold: CoreManifold,
    name: string,
    requestedTopN: number | undefined,
  ): GeometryProbeInfo {
    const layers = fittedLayerNumbers(manifold.tensors);
    const firstLayer = layers[0];
    const prefix = `layer_${firstLayer}`;
    const topN = requestedTopN === undefined ? 3 : requestedTopN;
    if (!Number.isSafeInteger(topN) || topN < 1 || topN > manifold.labels.length + 1) {
      throw coreError("INVALID_PROBE", "Probe nearest-node count is outside its supported range");
    }
    return {
      family: "geometry",
      name,
      manifold: manifold.selectorKey,
      top_n: topN,
      layers,
      node_labels: [...manifold.labels],
      node_count: manifold.labels.length,
      domain: structuredClone(manifold.domain) as GeometryProbeInfo["domain"],
      intrinsic_dim: manifoldIntrinsicDimension(manifold),
      feature_space: "residual",
      is_affine: !manifold.tensors.keys.includes(`${prefix}.node_params`),
      node_coords: coreNodeCoordinates(manifold.tensors, manifold.labels.length),
    };
  }

  private structuredAffineGroups(
    terms: readonly ResolvedCoreTerm[],
  ): ReadonlyMap<number, StructuredAffineGroup> {
    const lowered = new Map<number, {
      basis: Float64Array[];
      read: Float64Array[];
      target: Float64Array;
      requestedKappa: Float64Array;
      share: number;
    }>();

    for (const term of terms) {
      if (term.ablation && term.projection !== null) {
        throw coreError(
          "CORE_PACK_ABLATION_COMPOSITION_UNAVAILABLE",
          "Ablation does not compose with a projection",
        );
      }
      if (
        term.manifoldPosition !== null &&
        (term.base.kind !== "manifold" || isCurvedManifold(term.base.manifold))
      ) {
        throw coreError(
          "CORE_PACK_AFFINE_POSITION_INVALID",
          "Only a fitted flat manifold can be lowered as an affine position",
        );
      }
      if (term.manifoldPosition !== null && term.base.kind === "manifold") {
        this.validateAffineManifoldLayers(term.base.manifold);
      }
    }

    // Native `synthesize_subspace`, all in the Mahalanobis metric: each row's read is its `Σ⁻¹`
    // image, frames are Gram-Schmidt on write/read pairs (`orthoFrame`), push wins a shared
    // direction Mahalanobis-orthogonally, an ablation axis writes along its direction and reads
    // with its covector (a full `!` is LEACE), and the target is `read · Δ`.
    this.layerMap.forEach((layer, programIndex) => {
      const whitener = this.whiteners.get(layer)!;
      const pushRows: Float64Array[] = [];
      const pushFragments: Array<{ coefficient: number; direction: Float64Array }> = [];
      const ablationRows: Array<{ coefficient: number; direction: Float64Array }> = [];

      for (const term of terms) {
        if (Math.abs(term.coefficient) < 1e-9 || !termActsAt(term, layer)) continue;
        if (term.ablation) {
          const direction = this.atomNodeDirection(term.base, layer);
          if (direction === null || euclideanNorm(direction) < 1e-9) continue;
          ablationRows.push({ coefficient: term.coefficient, direction });
          continue;
        }

        const fragment = this.structuredPushFragment(term, layer, whitener);
        if (fragment === null) continue;
        const rowsNorm = Math.sqrt(fragment.basis.reduce((sum, row) => sum + dot(row, row), 0));
        if (rowsNorm < 1e-9) continue;
        pushRows.push(...fragment.basis);
        pushFragments.push({ coefficient: term.coefficient, direction: fragment.direction });
      }

      const inverse = (row: ArrayLike<number>) => applySerializedInverse(whitener, row);
      const metricNorm = (row: ArrayLike<number>, read: ArrayLike<number>) =>
        Math.sqrt(Math.max(dot(row, read), 0));
      const pushReads = pushRows.map(inverse);
      const ablationReads = ablationRows.map((row) => inverse(row.direction));
      const ablationNorms = ablationRows.map((row, index) =>
        metricNorm(row.direction, ablationReads[index])
      );
      const fragmentNorms = pushFragments.map((fragment) =>
        mahalanobisNorm(whitener, fragment.direction)
      );
      // A strictly positive proxy is enough for a pure-ablation share: hook-time gain
      // compensation cancels share from the ablation amount.
      const ablationShare = ablationRows.reduce(
        (sum, row, index) => sum + Math.abs(row.coefficient) * Math.max(ablationNorms[index], 1e-9),
        0,
      );

      const worldTarget = new Float64Array(this.hiddenSize);
      const rawDelta = new Float64Array(this.hiddenSize);
      pushFragments.forEach((fragment, index) => {
        const norm = fragmentNorms[index];
        for (let column = 0; column < this.hiddenSize; column += 1) {
          rawDelta[column] += fragment.coefficient * fragment.direction[column];
          if (norm > 1e-9) {
            worldTarget[column] += fragment.coefficient / norm * fragment.direction[column];
          }
        }
      });

      const hasPush = euclideanNorm(worldTarget) >= 1e-9;
      const push = hasPush ? orthoFrame(pushRows, pushReads) : { basis: [], read: [] };
      // Push wins shared directions: each Mahalanobis-unit ablation row loses its push-span
      // component (coefficients `P_push · u`), and the remaining ablation operator is
      // diagonalized in its own small span so coefficients survive exactly.
      const projectedAblations: Array<{
        coefficient: number;
        direction: Float64Array;
        read: Float64Array;
      }> = [];
      ablationRows.forEach((row, index) => {
        const norm = ablationNorms[index];
        if (norm < 1e-9) return;
        const residual = Float64Array.from(row.direction, (value) => value / norm);
        const residualRead = Float64Array.from(ablationReads[index], (value) => value / norm);
        const coefficients = push.read.map((pushRead) => dot(pushRead, residual));
        coefficients.forEach((coefficient, axis) => {
          for (let column = 0; column < this.hiddenSize; column += 1) {
            residual[column] -= coefficient * push.basis[axis][column];
            residualRead[column] -= coefficient * push.read[axis][column];
          }
        });
        const residualNorm = metricNorm(residual, residualRead);
        if (residualNorm >= 1e-9) {
          projectedAblations.push({
            coefficient: row.coefficient,
            direction: residual.map((value) => value / residualNorm),
            read: residualRead.map((value) => value / residualNorm),
          });
        }
      });

      let ablationBasis: Float64Array[] = [];
      let ablationRead: Float64Array[] = [];
      let ablationEigenvalues = new Float64Array();
      if (projectedAblations.length > 0) {
        const span = orthoFrame(
          projectedAblations.map((fragment) => fragment.direction),
          projectedAblations.map((fragment) => fragment.read),
        );
        const size = span.basis.length;
        const reduced = new Float64Array(size * size);
        for (const fragment of projectedAblations) {
          const coordinates = Float64Array.from(span.read, (read) => dot(read, fragment.direction));
          for (let left = 0; left < size; left += 1) {
            for (let right = 0; right < size; right += 1) {
              reduced[left * size + right] += fragment.coefficient *
                coordinates[left] * coordinates[right];
            }
          }
        }
        const eigensystem = symmetricEigen(reduced, size);
        const kept = Array.from(eigensystem.values.keys()).filter(
          (index) => Math.abs(eigensystem.values[index]) >= 1e-9,
        );
        ablationEigenvalues = Float64Array.from(
          kept,
          (index) => eigensystem.values[index],
        );
        const rotate = (frame: readonly Float64Array[]) => kept.map((eigenIndex) => {
          const row = new Float64Array(this.hiddenSize);
          for (let spanIndex = 0; spanIndex < size; spanIndex += 1) {
            const coefficient = eigensystem.vectors[spanIndex * size + eigenIndex];
            for (let column = 0; column < this.hiddenSize; column += 1) {
              row[column] += coefficient * frame[spanIndex][column];
            }
          }
          return row;
        });
        ablationBasis = rotate(span.basis);
        ablationRead = rotate(span.read);
      }

      const basis = [...push.basis, ...ablationBasis];
      const read = [...push.read, ...ablationRead];
      if (basis.length === 0) return;
      if (basis.length > this.structuredHookProfile.maxRank) {
        throw coreError(
          "CORE_PACK_AFFINE_RANK_LIMIT",
          `A composed affine steering group has rank ${basis.length}, above this browser model library's rank-${this.structuredHookProfile.maxRank} limit`,
        );
      }

      let share: number;
      const target = new Float64Array(basis.length);
      if (hasPush) {
        // The whitened raw displacement is the per-layer profile weight; the target is the
        // whitened-unit push's coordinate in the merged frame (ablation axes ≈ 0).
        share = mahalanobisNorm(whitener, rawDelta);
        if (share < 1e-9) {
          share = pushFragments.reduce(
            (sum, fragment, index) => sum + Math.abs(fragment.coefficient) * fragmentNorms[index],
            0,
          );
        }
        read.forEach((row, axis) => {
          target[axis] = dot(row, worldTarget);
        });
      } else {
        share = ablationShare;
      }
      if (share < 1e-9) return;
      lowered.set(programIndex, {
        basis,
        read,
        target,
        requestedKappa: Float64Array.from(
          { length: basis.length },
          (_, axis) => axis < push.basis.length
            ? 0
            : ablationEigenvalues[axis - push.basis.length],
        ),
        share,
      });
    });

    const totalShare = [...lowered.values()].reduce((sum, layer) => sum + layer.share, 0);
    const groups = new Map<number, StructuredAffineGroup>();
    for (const [programIndex, layer] of lowered) {
      const along = layer.share / totalShare * lowered.size * SUBSPACE_GAIN;
      groups.set(programIndex, {
        active: true,
        basis: layer.basis,
        read: layer.read,
        neutral: this.whiteners.get(this.layerMap[programIndex])!.mean,
        target: layer.target,
        along,
        kappa: Float64Array.from(layer.requestedKappa, (value) => value / along),
      });
    }
    return groups;
  }

  private structuredPushFragment(
    term: ResolvedCoreTerm,
    layer: number,
    whitener: SerializedMahalanobisWhitener,
  ): { basis: Float64Array[]; direction: Float64Array } | null {
    if (term.manifoldPosition === null) {
      const direction = this.termDirection(term, layer, whitener);
      return direction === null ? null : { basis: [direction], direction };
    }
    const atom = term.base as ResolvedManifoldAtom;
    const prefix = `layer_${layer}`;
    const shape = atom.manifold.tensors.description.shapes.get(`${prefix}.basis`);
    if (shape === undefined) return null;
    if (
      shape.length !== 2 || shape[0] < 1 || shape[1] !== this.hiddenSize ||
      shape[0] > this.structuredHookProfile.maxRank
    ) {
      throw coreError(
        "CORE_PACK_TENSOR_INVALID",
        `Flat manifold ${atom.manifold.key} layer ${layer} has an incompatible basis`,
      );
    }
    const weights = resolveAffinePositionWeights(atom.manifold, term.manifoldPosition);
    const direction = affineManifoldPositionDirection(
      atom,
      weights,
      layer,
      this.hiddenSize,
    );
    if (direction === null) return null;
    const basis = rows64(
      atom.manifold.tensors.tensor(`${prefix}.basis`, [shape[0], this.hiddenSize]).data,
      shape[0],
      this.hiddenSize,
    );
    return { basis, direction };
  }

  private structuredCurves(
    term: ResolvedCoreTerm,
  ): ReadonlyMap<number, StructuredCurve> {
    if (term.ablation || term.projection !== null || term.manifoldPosition === null) {
      throw coreError("CORE_PACK_CURVED_EXPRESSION_INVALID", "Invalid curved manifold term");
    }
    if (term.base.kind !== "manifold") {
      throw coreError("CORE_PACK_CURVED_EXPRESSION_INVALID", "Instrument directions are not curved manifolds");
    }
    const manifold = term.base.manifold;
    if (!isCurvedManifold(manifold)) {
      throw coreError(
        "CORE_PACK_CURVED_EXPRESSION_INVALID",
        "A fitted flat manifold cannot be lowered as a curved GPU program",
      );
    }
    const domain = structuredCurveDomain(manifold.domain, this.structuredHookProfile);
    // The mean-1 share normalizes over the layers the term acts at (a layer
    // clause's selection, else every fitted layer), as _manifold_layer_shares.
    const active = fittedLayerNumbers(manifold.tensors).filter((layer) => termActsAt(term, layer));
    const rawShares = active.map((layer) => manifold.mahalanobisShares.get(layer) ?? 0);
    const shareMean = rawShares.reduce((sum, value) => sum + value, 0) / rawShares.length;
    if (!(shareMean > 0) || rawShares.some((value) => !(Number.isFinite(value) && value > 0))) {
      throw coreError("CORE_PACK_METRIC_INVALID", "Curved manifold shares are invalid");
    }
    const target = resolveCurvePosition(manifold, term.manifoldPosition, domain.axes);
    const onto = term.coefficients[1] ?? 0;
    const periodic = domain.axes.some((axis) => axis.periodic);
    const curves = new Map<number, StructuredCurve>();
    for (const layer of active) {
      const programIndex = this.layerMap.indexOf(layer);
      if (programIndex < 0) {
        throw coreError(
          "CORE_PACK_LAYER_MISMATCH",
          `Core manifold layer ${layer} is outside the compiled model layer map`,
        );
      }
      const prefix = `layer_${layer}`;
      const basisShape = manifold.tensors.description.shapes.get(`${prefix}.basis`);
      const nodeShape = manifold.tensors.description.shapes.get(`${prefix}.node_params`);
      if (!basisShape || !nodeShape || nodeShape[1] !== domain.embedDim) {
        throw coreError(
          "CORE_PACK_CURVED_SHAPE_INVALID",
          "The fitted RBF embedding does not match the manifold domain",
        );
      }
      const rank = basisShape[0];
      const nodeCount = nodeShape[0];
      if (
        rank > this.structuredHookProfile.maxRank ||
        nodeCount > this.structuredHookProfile.maxCurveNodes
      ) {
        throw coreError(
          "CORE_PACK_CURVED_SHAPE_LIMIT",
          `Curved GPU programs support rank ${this.structuredHookProfile.maxRank} and ${this.structuredHookProfile.maxCurveNodes} nodes`,
        );
      }
      const sigmaWeights = manifold.tensors.tensor(
        `${prefix}.sigma_rbf_weights`,
        [nodeCount, 1],
      ).data;
      const sigmaPolynomial = manifold.tensors.tensor(
        `${prefix}.sigma_poly_coeffs`,
        [domain.embedDim + 1, 1],
      ).data;
      const origin = manifold.originPerLayer.get(layer);
      if (origin === undefined || origin.length !== domain.axes.length) {
        throw coreError(
          "CORE_PACK_CURVED_SHAPE_INVALID",
          `Core manifold layer ${layer} has an incompatible origin`,
        );
      }
      curves.set(programIndex, {
        active: true,
        domainKind: domain.kind,
        basis: rows(manifold.tensors.tensor(`${prefix}.basis`, [rank, this.hiddenSize]).data, rank, this.hiddenSize),
        read: rows(manifold.tensors.tensor(`${prefix}.read`, [rank, this.hiddenSize]).data, rank, this.hiddenSize),
        neutral: manifold.tensors.tensor(`${prefix}.mean`, [this.hiddenSize]).data,
        nodeParameters: rows(
          manifold.tensors.tensor(`${prefix}.node_params`, [nodeCount, domain.embedDim]).data,
          nodeCount,
          domain.embedDim,
        ),
        rbfWeights: rows(manifold.tensors.tensor(`${prefix}.rbf_weights`, [nodeCount, rank]).data, nodeCount, rank),
        polynomial: rows(
          manifold.tensors.tensor(`${prefix}.poly_coeffs`, [domain.embedDim + 1, rank]).data,
          domain.embedDim + 1,
          rank,
        ),
        coordinateOffset: manifold.tensors.tensor(
          `${prefix}.coord_offset`,
          [domain.embedDim],
        ).data,
        coordinateScale: manifold.tensors.tensor(
          `${prefix}.coord_scale`,
          [domain.embedDim],
        ).data,
        origin,
        target,
        axes: domain.axes,
        along: periodic
          ? Math.max(0, Math.min(1, Math.max(0, Math.min(1, term.coefficient)) * MANIFOLD_ALONG_GAIN))
          : Math.max(0, Math.min(1, term.coefficient)) *
            (manifold.mahalanobisShares.get(layer)! / shareMean) * MANIFOLD_ALONG_GAIN,
        onto: Math.max(0, Math.min(1, onto *
          (manifold.mahalanobisShares.get(layer)! / shareMean) * MANIFOLD_ONTO_GAIN)),
        sigmaRbfWeights: sigmaWeights,
        sigmaPolynomial,
        damping: 1e-3,
      });
    }
    return curves;
  }

  /** One attached probe on the device: an SAE or R-lens probe in its probe slots, a geometry
   *  probe in the geometry readout producing its own scores and the `gateKeys` a gate reads
   *  (a key it cannot produce is left out: that gate never fires). */
  private installGateProbe(
    name: string,
    probeIndex: number,
    layers: StructuredHookLayer[],
    gateKeys: readonly string[],
  ): InstalledProbeProgram {
    const attached = this.attachedProbes.get(name);
    if (attached === undefined || attached.family === "units" || (attached.family === "lens" && attached.namespace === "slens")) {
      throw coreError("CORE_PACK_GATE_PROBE_UNAVAILABLE", `${name} is not an attached device probe`);
    }
    if (attached.family === "sae") {
      const saeFeatureId = String(attached.feature_id);
      const feature = this.instruments.saeFeature(saeFeatureId);
      const programIndex = this.layerMap.indexOf(feature.layer);
      if (programIndex < 0) {
        throw coreError(
          "SAE_PACK_LAYER_MISMATCH",
          `SAE feature ${name} is attached to a layer outside the loaded runtime`,
        );
      }
      (layers[programIndex].probes as StructuredProbe[])[probeIndex] = saeFeatureProbe(
        feature.encoderDirection,
        feature.featureBias,
        feature.encoderInputBias,
        feature.maxAct,
        feature.activation,
        feature.threshold,
      );
      return { slots: [{ layer: programIndex, probe: probeIndex }], geometry: null };
    }
    if (attached.family === "lens") {
      const token = this.instruments.jlensToken(attached.word);
      const slots: StructuredMeasurementSlot[] = [];
      for (const layer of token.directions.keys()) {
        const programIndex = this.layerMap.indexOf(layer);
        if (programIndex < 0) continue;
        (layers[programIndex].probes as StructuredProbe[])[probeIndex] = {
          kind: "jlens",
          direction: new Float32Array(this.hiddenSize),
          bias: 0,
        };
        slots.push({ layer: programIndex, probe: probeIndex });
      }
      if (slots.length === 0) {
        throw coreError(
          "JLENS_PACK_LAYER_MISMATCH",
          `R-lens token ${name} has no layer in the loaded runtime`,
        );
      }
      return { slots, geometry: null };
    }
    let manifold: CoreManifold;
    try {
      manifold = this.resolveProbeManifold(attached.manifold);
    } catch {
      throw coreError(
        "CORE_PACK_GATE_PROBE_UNAVAILABLE",
        `The installed browser packs do not provide live probe ${name}`,
      );
    }
    const geometryTarget = { name: attached.name, topN: attached.top_n };
    this.assertGeometryProbeLowerable(manifold);
    const built = this.structuredGeometryProbe(manifold);
    for (const [programIndex, probe] of built.layers) {
      (layers[programIndex].geometryProbes as StructuredGeometryProbe[])[probeIndex] = probe;
    }
    // The readout's key family (native `flat_scalars`): every axis, the fraction, the membership,
    // and a distance and an assignment per label.
    const produced = new Set([
      ...(built.intrinsicDim > 0 ? [name] : []),
      ...Array.from({ length: built.intrinsicDim }, (_, axis) => `${name}[${axis}]`),
      `${name}:fraction`,
      `${name}:membership`,
      ...built.labels.flatMap((label) => [`${name}@${label}`, `${name}~${label}`]),
    ]);
    return {
      slots: [],
      geometry: {
        name: geometryTarget.name,
        scoreKeys: [...new Set([name, ...gateKeys])].filter((key) => produced.has(key)),
        manifold: manifold.selectorKey,
        labels: built.labels,
        topN: geometryTarget.topN,
        intrinsicDim: built.intrinsicDim,
        rank: built.rank,
        shareWeights: built.shareWeights,
        assignBandwidth: built.assignBandwidth,
        assignLogVolumeBias: built.assignLogVolumeBias,
        labelScale: built.labelScale,
      },
    };
  }

  private measurementProbe(name: string | undefined): StructuredMeasurementProbe | null {
    if (name === undefined) return null;
    const attached = this.attachedProbes.get(name);
    if (attached?.family === "lens") {
      return { name, family: "lens", tokenId: attached.token_id ?? undefined };
    }
    if (attached?.family === "sae") {
      return {
        name,
        family: "sae",
        featureId: attached.feature_id,
        label: attached.label,
        maxAct: attached.max_act,
      };
    }
    if (attached?.family === "geometry") return { name, family: "geometry" };
    return measurementProbe(name, this.instruments);
  }

  private structuredGeometryWhitener(layer: number): StructuredGeometryWhitener {
    const whitener = this.whiteners.get(layer);
    if (!whitener || whitener.rank > this.structuredHookProfile.maxWhitenerRank) {
      throw coreError(
        "CORE_PACK_WHITENER_UNAVAILABLE",
        `Geometry readout requires a supported whitener for layer ${layer}`,
      );
    }
    return {
      rank: whitener.rank,
      ridge: whitener.ridge,
      basis: rows64(whitener.basis, whitener.rank, whitener.columns),
      correction: Float64Array.from(whitener.eigenvalues, (value) =>
        1 / (value + whitener.ridge) - 1 / whitener.ridge
      ),
    };
  }

  private structuredGeometryProbe(manifold: CoreManifold): {
    layers: ReadonlyMap<number, StructuredGeometryProbe>;
    labels: string[];
    intrinsicDim: number;
    rank: number;
    shareWeights: number[];
    assignBandwidth: number[];
    assignLogVolumeBias: number[];
    labelScale: number;
  } {
    const fitted = fittedLayerNumbers(manifold.tensors);
    const schemaRank = manifold.tensors.description.shapes.get(`layer_${fitted[0]}.basis`)![0];
    const intrinsicDim = manifoldIntrinsicDimension(manifold);
    const injectNeutral = !manifold.labels.includes("neutral");
    const labels = [...manifold.labels, ...(injectNeutral ? ["neutral"] : [])];
    const rawShares = fitted.map((layer) => Math.max(
      manifold.mahalanobisShares.get(layer) ?? 0,
      1e-6,
    ));
    const totalShare = rawShares.reduce((sum, value) => sum + value, 0);
    const normalizedShares = rawShares.map((value) => value / totalShare);
    const shareWeights = this.layerMap.map((layer) => {
      const index = fitted.indexOf(layer);
      return index < 0 ? 0 : normalizedShares[index];
    });
    const layers = new Map<number, StructuredGeometryProbe>();
    const bandwidthPerLayer: Array<{ values: number[]; weight: number }> = [];
    for (let fittedIndex = 0; fittedIndex < fitted.length; fittedIndex += 1) {
      const layer = fitted[fittedIndex];
      const programIndex = this.layerMap.indexOf(layer);
      if (programIndex < 0) continue;
      const prefix = `layer_${layer}`;
      const rank = manifold.tensors.description.shapes.get(`${prefix}.basis`)![0];
      const whitener = this.whiteners.get(layer)!;
      const basis = manifold.tensors.tensor(
        `${prefix}.basis`,
        [rank, this.hiddenSize],
      ).data;
      const basisRows = rows(basis, rank, this.hiddenSize);
      const gram = restrictedInverseCovariance(basis, rank, whitener);
      const cholesky = lowerCholesky(gram, rank);
      const gramInverse = inversePositiveDefinite(cholesky, rank);
      const mean = manifold.tensors.tensor(`${prefix}.mean`, [this.hiddenSize]).data;
      const inverseMean = applySerializedInverse(whitener, mean);
      const isCurved = manifold.tensors.keys.includes(`${prefix}.node_params`);
      let reducedNodes: Float32Array;
      let coordinateMap: Float64Array[];
      let coordinateBias: Float64Array;
      let curve: StructuredCurve | null = null;
      let curveNodeCoordinates: Float32Array[] | undefined;
      let curveNodeValues: Float32Array[] | undefined;
      let foot: Float32Array | null = null;
      if (isCurved) {
        const domain = structuredCurveDomain(manifold.domain, this.structuredHookProfile);
        curve = coreGeometryCurve(manifold, layer, domain, this.hiddenSize);
        curveNodeCoordinates = rows(
          manifold.tensors.tensor(
            "node_coords",
            [manifold.labels.length, intrinsicDim],
          ).data,
          manifold.labels.length,
          intrinsicDim,
        );
        curveNodeValues = curveNodeCoordinates.map((coordinate) =>
          evaluateCoreCurve(curve!, coordinate)
        );
        reducedNodes = flattenRows(curveNodeValues, rank);
        coordinateMap = Array.from({ length: intrinsicDim }, () => new Float64Array(rank));
        coordinateBias = new Float64Array(intrinsicDim);
        foot = manifold.originPerLayer.get(layer)?.slice() ?? null;
        if (foot === null) {
          throw coreError(
            "CORE_PACK_CURVED_SHAPE_INVALID",
            `Curved probe ${manifold.key} has no origin for layer ${layer}`,
          );
        }
      } else {
        reducedNodes = manifold.tensors.tensor(
          `${prefix}.node_coords`,
          [manifold.labels.length, rank],
        ).data;
        const authorCoordinates = manifold.tensors.tensor(
          "node_coords",
          [manifold.labels.length, intrinsicDim],
        ).data;
        const affine = affineCoordinateMap(
          reducedNodes,
          authorCoordinates,
          manifold.labels.length,
          rank,
          intrinsicDim,
        );
        coordinateMap = affine.map;
        coordinateBias = affine.bias;
      }
      const nodeWhite = multiplyRows(
        reducedNodes,
        manifold.labels.length,
        rank,
        cholesky,
      );
      let neutralWhite: Float32Array = new Float32Array(rank);
      if (curve !== null && foot !== null) {
        neutralWhite = multiplyRows(
          evaluateCoreCurve(curve, foot),
          1,
          rank,
          cholesky,
        );
      }
      const candidates = rows(nodeWhite, manifold.labels.length, rank);
      if (injectNeutral) candidates.push(neutralWhite);
      const bandwidth = geometryLayerBandwidth(
        candidates.slice(0, manifold.labels.length),
        neutralWhite,
        injectNeutral,
        curve,
        curveNodeCoordinates,
        gram,
      );
      bandwidthPerLayer.push({ values: bandwidth, weight: normalizedShares[fittedIndex] });
      layers.set(programIndex, {
        active: true,
        mean,
        inverseMean,
        basis: basisRows,
        gramInverse,
        cholesky,
        candidateWhite: candidates,
        coordinateMap,
        coordinateBias,
        curve,
        curveNodeCoordinates,
        curveNodeValues,
        foot,
      });
    }
    const assignBandwidth = new Array(labels.length).fill(0);
    bandwidthPerLayer.forEach(({ values, weight }) => {
      for (let candidate = 0; candidate < labels.length; candidate += 1) {
        assignBandwidth[candidate] += weight * values[candidate];
      }
    });
    const medianBandwidth = Math.max(1e-6, median(assignBandwidth));
    for (let index = 0; index < assignBandwidth.length; index += 1) {
      assignBandwidth[index] = Math.max(assignBandwidth[index], 1e-3 * medianBandwidth);
    }
    const assignLogVolumeBias = assignBandwidth.map((value) => -schemaRank * Math.log(value));
    const labelScale = Math.max(1e-8, median(assignBandwidth.slice(0, manifold.labels.length)));
    return {
      layers,
      labels,
      intrinsicDim,
      rank: schemaRank,
      shareWeights,
      assignBandwidth,
      assignLogVolumeBias,
      labelScale,
    };
  }

  private instrumentProgramProbeName(name: string): string {
    const attached = this.attachedProbes.get(name);
    return attached?.family === "lens" ? `jlens/${attached.word}` : name;
  }

  private termDirection(
    term: ResolvedCoreTerm,
    layer: number,
    whitener: SerializedMahalanobisWhitener,
  ): Float64Array | null {
    if (term.projection === null) return this.atomNodeDirection(term.base, layer);
    const base = this.atomProfileDirection(term.base, layer);
    if (base === null) return null;
    const onto = this.atomProfileDirection(term.projection.onto, layer);
    if (onto === null) return term.projection.operator === "|" ? base : null;
    const projected = leaceProject(whitener, base, onto, term.projection.operator);
    return euclideanNorm(projected) < 1e-9 ? null : projected;
  }

  private atomNodeDirection(atom: ResolvedCoreAtom, layer: number): Float64Array | null {
    if (atom.kind === "instrument") {
      const direction = atom.directions.get(layer);
      return direction === undefined ? null : Float64Array.from(direction);
    }
    const values = this.atomLayerValues(atom, layer);
    if (values === null) return null;
    return Float64Array.from(values.basis, (value) => value * values.nodeCoordinate);
  }

  private atomProfileDirection(atom: ResolvedCoreAtom, layer: number): Float64Array | null {
    if (atom.kind === "instrument") {
      const direction = atom.directions.get(layer);
      return direction === undefined ? null : Float64Array.from(direction);
    }
    const values = this.atomLayerValues(atom, layer);
    if (values === null) return null;
    return Float64Array.from(values.basis, (value) => value * values.mahalanobisShare);
  }

  private atomLayerValues(
    atom: ResolvedManifoldAtom,
    layer: number,
  ): { basis: Float32Array; nodeCoordinate: number; mahalanobisShare: number } | null {
    const prefix = `layer_${layer}`;
    if (!atom.manifold.tensors.keys.includes(`${prefix}.basis`)) return null;
    if (atom.manifold.tensors.keys.includes(`${prefix}.node_params`)) {
      throw coreError(
        "CORE_PACK_CURVED_UNAVAILABLE",
        "The current browser model library cannot execute curved manifold steering",
      );
    }
    const basis = atom.manifold.tensors.tensor(`${prefix}.basis`, [1, this.hiddenSize]).data;
    const coordinates = atom.manifold.tensors.tensor(
      `${prefix}.node_coords`,
      [atom.manifold.labels.length, 1],
    ).data;
    const mahalanobisShare = atom.manifold.mahalanobisShares.get(layer);
    if (mahalanobisShare === undefined) {
      throw coreError(
        "CORE_PACK_METRIC_MISSING",
        `Core manifold layer ${layer} has no Mahalanobis share`,
      );
    }
    return { basis, nodeCoordinate: coordinates[atom.labelIndex], mahalanobisShare };
  }

  private validateAffineManifoldLayers(manifold: CoreManifold): void {
    const fittedLayers = fittedLayerNumbers(manifold.tensors);
    if (fittedLayers.length === 0) {
      throw coreError("CORE_PACK_TENSOR_INVALID", "The selected core manifold has no fitted layers");
    }
    for (const layer of fittedLayers) {
      if (!this.layerMap.includes(layer)) {
        throw coreError(
          "CORE_PACK_LAYER_MISMATCH",
          `Core manifold layer ${layer} is outside the compiled model layer map`,
        );
      }
    }
    if (
      manifold.labels.length + (manifold.labels.includes("neutral") ? 0 : 1) >
        this.structuredHookProfile.maxGeometryCandidates
    ) {
      throw coreError(
        "CORE_PACK_GEOMETRY_CANDIDATE_LIMIT",
        `Probe ${manifold.key} has too many node candidates for this browser model library`,
      );
    }
  }

  steeringDelta(parent: string | null, child: string | null): string {
    const parentTerms = this.alphaMap(parent);
    const childTerms = this.alphaMap(child);
    const keys = new Set([...parentTerms.keys(), ...childTerms.keys()]);
    const deltas = [...keys].map((key) => ({
      key,
      value: (childTerms.get(key) ?? 0) - (parentTerms.get(key) ?? 0),
    })).filter((entry) => Math.abs(entry.value) > 1e-12)
      .sort((left, right) => Math.abs(right.value) - Math.abs(left.value) || left.key.localeCompare(right.key));
    return deltas.map((entry, index) => {
      const value = formatCoefficient(entry.value);
      return `${index > 0 && entry.value >= 0 ? "+" : ""}${value} ${entry.key}`;
    }).join(" ");
  }

  private alphaMap(expression: string | null): Map<string, number> {
    const output = new Map<string, number>();
    if (!expression?.trim()) return output;
    for (const term of parseSteeringExpression(expression).terms) {
      const key = this.steeringDeltaKey(term);
      output.set(key, (output.get(key) ?? 0) + term.coefficients[0]);
    }
    return output;
  }

  private steeringDeltaKey(term: SteeringExpressionTerm): string {
    const base = term.selector.base;
    const onto = term.selector.onto;
    const containsJlens = base.namespace === "jlens" || onto?.namespace === "jlens";
    if (!containsJlens || term.selector.manifoldPosition !== null) {
      return this.resolveTerm(term).key;
    }
    if (term.coefficients.length !== 1 || term.ablation && onto !== null) {
      throw coreError(
        "CORE_PACK_EXPRESSION_UNAVAILABLE",
        "The browser GPU compiler cannot lower this steering expression",
      );
    }
    for (const atom of [base, onto]) {
      if (atom === null) continue;
      if (atom.namespace !== "jlens") {
        this.resolveAtom(atom);
        continue;
      }
      if (atom.variant !== "raw") {
        throw coreError(
          "CORE_PACK_VARIANT_UNAVAILABLE",
          "R-lens atoms do not take manifold variants",
        );
      }
      if (!this.instruments.capabilities().jlens) {
        throw coreError(
          "JLENS_PACK_UNAVAILABLE",
          "Install the compatible R-lens pack to use jlens/<word>",
        );
      }
    }
    const clause = formatLayerClause(term.layers === "all" ? null : term.layers);
    return onto === null
      ? `${term.ablation ? "!" : ""}${formatAtom(base)}${clause}`
      : `${formatAtom(base)}${term.selector.projection}${formatAtom(onto)}${clause}`;
  }

  private resolveTerm(term: SteeringExpressionTerm): ResolvedCoreTerm {
    if (term.ablation && term.selector.projection !== null) {
      throw coreError(
        "CORE_PACK_EXPRESSION_UNAVAILABLE",
        "The browser GPU compiler cannot lower this steering expression",
      );
    }
    const base = term.selector.manifoldPosition === null
      ? this.resolveAtom(term.selector.base)
      : this.resolveManifoldAtom(term.selector.base);
    const projection = term.selector.projection === null
      ? null
      : {
          operator: term.selector.projection,
          onto: this.resolveAtom(term.selector.onto!),
        };
    // A plain term on a manifold's label or 2-node name is a `%` term, so it takes `along,onto`.
    const position = term.selector.manifoldPosition !== null ||
      (projection === null && !term.ablation && base.kind === "manifold");
    if (term.coefficients.length > (position ? 2 : 1)) {
      throw coreError(
        "CORE_PACK_EXPRESSION_UNAVAILABLE",
        "The browser GPU compiler cannot lower this steering expression",
      );
    }
    const layers: LayerScope = term.layers === null || term.layers === "all" ? null : term.layers;
    const manifoldPosition = term.selector.manifoldPosition === null &&
        projection === null && !term.ablation && base.kind === "manifold" &&
        base.labelAlias && manifoldLabelNeedsPosition(base.manifold)
      ? base.manifold.labels[base.labelIndex]
      : term.selector.manifoldPosition;
    const baseKey = term.selector.manifoldPosition === null
      ? base.key
      : `${base.key}%${formatManifoldPosition(term.selector.manifoldPosition)}`;
    return {
      key: (projection === null
        ? `${term.ablation ? "!" : ""}${baseKey}`
        : `${formatAtom(term.selector.base)}${projection.operator}${formatAtom(term.selector.onto!)}`) +
        formatLayerClause(layers),
      base,
      projection,
      coefficient: term.coefficients[0],
      coefficients: term.coefficients,
      ablation: term.ablation,
      trigger: normalizeTrigger(term.trigger),
      manifoldPosition,
      layers,
      irKey: foldSteering({ terms: [term] }, { manifolds: this.steeringManifolds() }).terms[0].key,
      actsAt: null,
    };
  }

  /** The installed manifolds as the grammar's bare-atom ladder sees them, one per folder (also
   *  what the loom's `inverted` regen resolves against). */
  steeringManifolds(): SteeringManifoldEntry[] {
    const entries = new Map<string, SteeringManifoldEntry>();
    for (const manifold of this.manifolds) {
      const key = `${manifold.namespace}/${manifold.name}`;
      if (!entries.has(key)) {
        entries.set(key, {
          namespace: manifold.namespace,
          name: manifold.name,
          fitMode: manifold.fitMode,
          labels: manifold.labels,
        });
      }
    }
    return [...entries.values()];
  }

  /** The layers a resolved term can act on: its atom's fitted layers (a `~` projection's where
   *  both operands are fitted and the projection is not zero, as `project_profile` drops a
   *  degenerate `onto`). */
  private termFittedLayers(term: ResolvedCoreTerm): number[] {
    const atomLayers = (atom: ResolvedCoreAtom): number[] => atom.kind === "instrument"
      ? atom.unit !== undefined ? [atom.unit.layer] : [...atom.directions.keys()]
      : fittedLayerNumbers(atom.manifold.tensors);
    const base = atomLayers(term.base);
    if (term.projection?.operator !== "~") return [...base].sort((a, b) => a - b);
    const ontoLayers = atomLayers(term.projection.onto);
    const onto = new Set(ontoLayers);
    const fitted = base.filter((layer) => {
      if (!onto.has(layer)) return false;
      // Without a whitener the term is refused at lowering, so the layer stays listed.
      const whitener = this.whiteners.get(layer);
      return whitener === undefined || this.termDirection(term, layer, whitener) !== null;
    }).sort((a, b) => a - b);
    if (fitted.length === 0) {
      const list = (layers: number[]) => `[${[...layers].sort((a, b) => a - b).join(", ")}]`;
      throw Object.assign(
        coreError(
          "STEERING_COMPOSITION",
          `project_profile: no layers produced for operator '~' (base layers: ${list(base)}, ` +
            `onto layers: ${list(ontoLayers)})`,
        ),
        { status: 422, recoverable: false },
      );
    }
    return fitted;
  }

  /** Select each folded term's layers (`select_term_layers`: 422 on a clause naming an unfitted
   *  layer) and report where each acts, weighed alone as composition weighs it (`term_layers`). */
  private placeTerms(terms: ResolvedCoreTerm[]): BrowserSteeringTermLayers[] {
    return terms.map((term, index) => {
      const fitted = this.termFittedLayers(term);
      let chosen: number[] | null;
      try {
        chosen = selectTermLayers(term.irKey, fitted, term.layers, this.layerMap.length);
      } catch (error) {
        throw Object.assign(
          coreError("STEERING_COMPOSITION", error instanceof Error ? error.message : String(error)),
          { status: 422, recoverable: false },
        );
      }
      terms[index] = { ...term, actsAt: chosen === null ? null : new Set(chosen) };
      const raw = fitted.map((layer) => this.termLayerShare(term, layer));
      const total = raw.reduce((sum, value) => sum + value, 0);
      const share = Number.isFinite(total) && total > 0
        ? raw.map((value) => value / total * fitted.length)
        : fitted.map(() => 1);
      return { key: term.irKey, fitted, share, layers: chosen ?? fitted };
    });
  }

  /** A term's raw weight at `layer`: an ablation's is even, a curved position's its baked
   *  Mahalanobis share, a push's its whitened displacement. */
  private termLayerShare(term: ResolvedCoreTerm, layer: number): number {
    if (term.ablation) return 1;
    if (
      term.manifoldPosition !== null && term.base.kind === "manifold" &&
      isCurvedManifold(term.base.manifold)
    ) {
      return Math.max(0, term.base.manifold.mahalanobisShares.get(layer) ?? 0);
    }
    const whitener = this.whiteners.get(layer);
    if (whitener === undefined) return 0;
    const fragment = this.structuredPushFragment(term, layer, whitener);
    return fragment === null ? 0 : Math.max(0, mahalanobisNorm(whitener, fragment.direction));
  }

  /** Merge the terms that share an IR key, as the grammar's fold (which ran first and checked
   *  their triggers and positions) merges them. A label term and an explicit-position term on one
   *  node keep different keys (`local/demo%alert`, `demo%alert`), so they stay two terms. */
  private foldResolvedTerms(
    terms: readonly ResolvedCoreTerm[],
  ): ResolvedCoreTerm[] {
    const folded = new Map<string, ResolvedCoreTerm>();
    for (const term of terms) {
      const previous = folded.get(term.irKey);
      if (previous === undefined) {
        folded.set(term.irKey, term);
        continue;
      }
      const coefficient = previous.coefficient + term.coefficient;
      const manifold = !term.ablation && term.projection === null &&
        (term.manifoldPosition !== null || term.base.kind === "manifold");
      folded.set(term.irKey, {
        ...previous,
        coefficient,
        coefficients: manifold
          ? [coefficient, (previous.coefficients[1] ?? 0) + (term.coefficients[1] ?? 0)]
          : [coefficient],
      });
    }
    return [...folded.values()];
  }

  private resolveManifoldAtom(atom: SteeringAtom): ResolvedManifoldAtom {
    const matches = this.manifolds.filter((manifold) =>
      manifold.name === atom.concept &&
      (atom.namespace === null || manifold.namespace === atom.namespace)
    );
    const identities = new Set(matches.map((manifold) => manifold.key));
    if (identities.size !== 1) {
      throw coreError(
        identities.size === 0 ? "CORE_PACK_SELECTOR_NOT_FOUND" : "CORE_PACK_SELECTOR_AMBIGUOUS",
        `Core manifold ${atom.namespace ? `${atom.namespace}/` : ""}${atom.concept} is not unique`,
      );
    }
    const selected = this.selectManifoldVariant(matches, atom.variant, true);
    return {
      kind: "manifold",
      key: selected.key,
      manifold: selected.manifold,
      labelIndex: 0,
      requestedRole: selected.requestedRole,
      labelAlias: false,
    };
  }

  /** Native `LensInstrument._raised`: a `jlens/` atom (`W_U[v] J_l`) or an `slens/` atom
   *  (`K_l ẽ_v`) is a read, and its steering profile is the vector it raises to, `Σu`: the write
   *  that moves the reading furthest per unit of Mahalanobis displacement. Folded, it reads with
   *  `u` again, so `!` zeroes exactly the token's reading (LEACE). Probes keep the reads. */
  private raisedLensReads(
    key: string,
    reads: ReadonlyMap<number, Float32Array>,
  ): ReadonlyMap<number, Float32Array> {
    const memo = this.raisedReads.get(reads);
    if (memo !== undefined) return memo;
    const layers = [...reads.keys()].sort((left, right) => left - right);
    if (layers.some((layer) => !this.whiteners.has(layer))) {
      throw coreError(
        "CORE_PACK_WHITENER_UNAVAILABLE",
        `${key} needs a Mahalanobis whitener covering its layers [${layers.join(", ")}]: a lens atom is a read, and steering writes along its whitener-raised vector`,
      );
    }
    const raised = new Map(layers.map((layer) => [
      layer,
      Float32Array.from(raiseLensRead(this.whiteners.get(layer)!, reads.get(layer)!)),
    ]));
    this.raisedReads.set(reads, raised);
    return raised;
  }

  private resolveAtom(atom: SteeringAtom): ResolvedCoreAtom {
    if (atom.namespace === "sae") {
      if (atom.variant !== "raw") {
        throw coreError("CORE_PACK_VARIANT_UNAVAILABLE", "SAE atoms do not take manifold variants");
      }
      const feature = this.instruments.saeFeature(atom.concept);
      return {
        kind: "instrument",
        key: feature.key,
        family: "sae",
        directions: new Map([[feature.layer, feature.direction]]),
        saeFeature: feature,
      };
    }
    if (atom.namespace === "jlens") {
      if (atom.variant !== "raw") {
        throw coreError("CORE_PACK_VARIANT_UNAVAILABLE", "R-lens atoms do not take manifold variants");
      }
      const token = this.instruments.jlensToken(atom.concept);
      return {
        kind: "instrument",
        key: token.key,
        family: "jlens",
        directions: this.raisedLensReads(token.key, token.directions),
        saeFeature: null,
      };
    }
    if (atom.namespace === "neuron" || atom.namespace === "head") {
      const source = this.unitSource;
      if (source === null) throw instrumentAtomUnavailable(atom.namespace);
      if (atom.variant !== "raw") {
        throw coreError("CORE_PACK_VARIANT_UNAVAILABLE", "Unit atoms do not take manifold variants");
      }
      const unit = resolveUnitAtom(source, atom.namespace, atom.concept);
      const write = unit.kind === "neuron" ? source.neuronWrite(unit.layer, unit.index) : null;
      return {
        kind: "instrument",
        key: unitName(unit),
        family: "units",
        directions: write === null ? new Map() : new Map([[unit.layer, write]]),
        saeFeature: null,
        unit,
      };
    }
    if (atom.namespace === "slens") {
      const source = this.unitSource;
      if (source === null) throw instrumentAtomUnavailable(atom.namespace);
      if (atom.variant !== "raw") {
        throw coreError("CORE_PACK_VARIANT_UNAVAILABLE", "S-lens atoms do not take manifold variants");
      }
      const prepared = source.slensAtom(atom.concept);
      if (prepared === null) {
        throw coreError("SLENS_ATOM_UNPREPARED", `slens/${atom.concept} was not resolved before compiling`);
      }
      return {
        kind: "instrument",
        key: `slens/${atom.concept}`,
        family: "slens",
        directions: this.raisedLensReads(`slens/${atom.concept}`, prepared.directions),
        saeFeature: null,
      };
    }
    if (atom.namespace !== null && RESERVED_NAMESPACES.has(atom.namespace)) {
      throw instrumentAtomUnavailable(atom.namespace);
    }
    if (
      atom.namespace === null && atom.variant === "raw" &&
      !atom.concept.includes(".")
    ) {
      const identities = new Map<string, { variants: CoreManifold[]; labelIndex: number; label: string }>();
      for (const manifold of this.manifolds) {
        const labelIndex = manifold.labels.indexOf(atom.concept);
        if (labelIndex < 0) continue;
        const prior = identities.get(manifold.key);
        if (prior) prior.variants.push(manifold);
        else identities.set(manifold.key, { variants: [manifold], labelIndex, label: atom.concept });
      }
      const labels = [...identities.values()]
        .filter((candidate) => candidate.label === atom.concept);
      if (labels.length > 1) {
        throw coreError(
          "CORE_PACK_SELECTOR_AMBIGUOUS",
          `Core manifold label ${atom.concept} matches multiple installed manifolds`,
        );
      }
      if (labels.length === 1) {
        const match = labels[0];
        const selected = this.selectManifoldVariant(match.variants, atom.variant, true);
        return {
          kind: "manifold",
          key: `${selected.key}%${match.label}`,
          manifold: selected.manifold,
          labelIndex: match.labelIndex,
          requestedRole: selected.requestedRole,
          labelAlias: true,
        };
      }
    }
    const canonicalConcept = canonicalConceptName(atom.concept);
    const nameVariants = this.manifolds.filter((manifold) =>
      manifold.name === canonicalConcept &&
      (atom.namespace === null || manifold.namespace === atom.namespace) &&
      manifold.fitMode === "pca" && manifold.labels.length === 2
    );
    const names = new Map<string, CoreManifold[]>();
    for (const manifold of nameVariants) {
      const variants = names.get(manifold.key) ?? [];
      variants.push(manifold);
      names.set(manifold.key, variants);
    }
    if (names.size > 1) {
      throw coreError(
        "CORE_PACK_SELECTOR_AMBIGUOUS",
        `Core manifold name ${atom.concept} exists in multiple namespaces`,
      );
    }
    if (names.size === 1) {
      const selected = this.selectManifoldVariant([...names.values()][0], atom.variant);
      return {
        kind: "manifold",
        key: `${selected.key}%${selected.manifold.labels[0]}`,
        manifold: selected.manifold,
        labelIndex: 0,
        requestedRole: selected.requestedRole,
        labelAlias: false,
      };
    }
    throw coreError(
      "CORE_PACK_SELECTOR_NOT_FOUND",
      `The required core pack does not contain ${atom.namespace ? `${atom.namespace}/` : ""}${atom.concept}`,
    );
  }

  private selectManifoldVariant(
    candidates: readonly CoreManifold[],
    variant: string,
    allowNodeRoleInference = false,
  ): { manifold: CoreManifold; key: string; requestedRole: string | null } {
    let matches: CoreManifold[];
    let requestedRole: string | null = null;
    if (variant === "raw") {
      matches = candidates.filter((candidate) => candidate.variant === "raw");
      if (
        !allowNodeRoleInference && matches.length === 1 &&
        matches[0].nodeRoles.some((role) => role !== null)
      ) {
        throw coreError(
          "CORE_PACK_ROLE_VARIANT_REQUIRED",
          `Manifold ${matches[0].key} was fitted with a role baseline; select it with :role-<name>`,
        );
      }
    } else if (variant === "role" || variant.startsWith("role-")) {
      const requested = variant === "role" ? null : variant.slice("role-".length);
      matches = candidates.filter((candidate) => {
        if (candidate.variant !== "raw" || candidate.nodeRoles.length === 0) return false;
        const role = candidate.nodeRoles[0];
        return role !== null && candidate.nodeRoles.every((value) => value === role) &&
          (requested === null || role === requested);
      });
      if (matches.length === 1) requestedRole = matches[0].nodeRoles[0];
    } else if (variant === "sae" || variant.startsWith("sae-")) {
      const release = variant === "sae" ? null : variant.slice("sae-".length);
      matches = candidates.filter((candidate) => candidate.variant === "sae" &&
        (release === null || candidate.variantIdentity === release || candidate.safeVariantIdentity === release));
    } else {
      throw coreError(
        "CORE_PACK_VARIANT_UNAVAILABLE",
        `The browser runtime does not support manifold variant ${variant}`,
      );
    }
    if (matches.length !== 1) {
      throw coreError(
        matches.length === 0 ? "CORE_PACK_VARIANT_UNAVAILABLE" : "CORE_PACK_SELECTOR_AMBIGUOUS",
        matches.length === 0
          ? `The required core pack has no ${variant} fit for ${candidates[0]?.key ?? "this manifold"}`
          : `Manifold variant ${variant} is not unique for ${candidates[0]?.key ?? "this manifold"}`,
      );
    }
    const manifold = matches[0];
    const key = requestedRole !== null
      ? `${manifold.key}:role-${requestedRole}`
      : manifold.selectorKey;
    return { manifold, key, requestedRole };
  }
}

async function loadCoreManifolds(
  artifact: BrowserModelArtifact,
  request: BrowserModelLoadRequest,
  profile: StructuredHookCapacityProfile,
): Promise<{ key: string; manifolds: CoreManifold[] }> {
  if (
    !artifact.manifest.path.endsWith(".drowse") &&
    !LEGACY_PRODUCT_SLUGS.some((slug) => artifact.manifest.path.endsWith(`.${slug}`)) &&
    !artifact.manifest.path.endsWith(".saklaspack")
  ) {
    throw coreError(
      "CORE_PACK_FORMAT_UNSUPPORTED",
      `Core artifact ${artifact.manifest.path} is not a supported manifold envelope`,
    );
  }
  const verified = await validateDrowseArchive(artifact.file, { signal: request.signal });
  const fitted = matchingFittedArtifacts(verified, request);
  return {
    key: verified.primaryIdentity.join("/"),
    manifolds: await Promise.all(fitted.map((candidate) =>
      loadVerifiedManifold(verified, request, profile, candidate)
    )),
  };
}

async function loadInstalledManifolds(
  archive: Blob,
  request: BrowserModelLoadRequest,
  profile: StructuredHookCapacityProfile,
): Promise<CoreManifold[]> {
  const verified = await validateDrowseArchive(archive, { signal: request.signal });
  const matches = matchingFittedArtifacts(verified, request, true);
  // A fit from before typed frames is a miss (native's stale fit), not a broken install.
  const loaded = await Promise.all(matches.map((candidate) =>
    loadVerifiedManifold(verified, request, profile, candidate).catch((error: unknown) => {
      if ((error as { code?: unknown } | null)?.code === "CORE_PACK_FIT_STALE") return null;
      throw error;
    })
  ));
  return loaded.filter((manifold): manifold is CoreManifold => manifold !== null);
}

async function loadVerifiedManifold(
  verified: VerifiedDrowseArchive,
  request: BrowserModelLoadRequest,
  profile: StructuredHookCapacityProfile,
  fitted: VerifiedDrowseArchive["fittedArtifacts"][number],
): Promise<CoreManifold> {
  const tensorBytes = await readVerifiedDrowseArchiveFile(
    verified,
    fitted.tensorPath,
    request.signal,
  );
  const sidecarBytes = await readVerifiedDrowseArchiveFile(
    verified,
    fitted.sidecarPath,
    request.signal,
  );
  // Shares are Mahalanobis (the sidecar no longer says so: drowse 6.1 dropped `share_metric`).
  const sidecar = JSON.parse(new TextDecoder().decode(sidecarBytes)) as Record<string, unknown>;
  const rawShares = sidecar.mahalanobis_share_per_layer;
  if (typeof rawShares !== "object" || rawShares === null || Array.isArray(rawShares)) {
    throw coreError("CORE_PACK_METRIC_INVALID", "Core manifold shares are invalid");
  }
  const mahalanobisShares = new Map<number, number>();
  for (const [rawLayer, rawShare] of Object.entries(rawShares)) {
    const layer = Number(rawLayer);
    if (
      !Number.isSafeInteger(layer) || layer < 0 ||
      typeof rawShare !== "number" || !Number.isFinite(rawShare) || rawShare <= 0
    ) {
      throw coreError("CORE_PACK_METRIC_INVALID", "Core manifold shares are invalid");
    }
    mahalanobisShares.set(layer, rawShare);
  }
  if (!isRecord(sidecar.domain) || !isRecord(sidecar.origin_per_layer)) {
    throw coreError("CORE_PACK_MANIFOLD_INVALID", "Core manifold geometry is invalid");
  }
  const originPerLayer = new Map<number, Float32Array>();
  for (const [rawLayer, rawOrigin] of Object.entries(sidecar.origin_per_layer)) {
    const layer = Number(rawLayer);
    if (
      !Number.isSafeInteger(layer) || layer < 0 || !Array.isArray(rawOrigin) ||
      rawOrigin.length === 0 || rawOrigin.length > profile.maxIntrinsicDim ||
      rawOrigin.some((value) => typeof value !== "number" || !Number.isFinite(value))
    ) {
      throw coreError(
        "CORE_PACK_CURVED_DIMENSION_UNAVAILABLE",
        `Browser curves support between one and ${profile.maxIntrinsicDim} authoring coordinates`,
      );
    }
    originPerLayer.set(layer, Float32Array.from(rawOrigin as number[]));
  }
  const nodes = verified.manifold.nodes;
  if (!Array.isArray(nodes)) {
    throw coreError("CORE_PACK_MANIFOLD_INVALID", "Core manifold nodes are unavailable");
  }
  const labels = nodes.map((node) => {
    if (typeof node !== "object" || node === null || typeof (node as { label?: unknown }).label !== "string") {
      throw coreError("CORE_PACK_MANIFOLD_INVALID", "Core manifold node labels are invalid");
    }
    return (node as { label: string }).label;
  });
  const [namespace, name] = verified.primaryIdentity;
  const tensors = decodeFp32Safetensors(tensorBytes, fitted.tensorPath);
  // Typed frames (fit policy 2): every layer's write rows carry their dual read rows.
  for (const layer of fittedLayerNumbers(tensors)) {
    const basisShape = tensors.description.shapes.get(`layer_${layer}.basis`);
    const readShape = tensors.description.shapes.get(`layer_${layer}.read`);
    if (readShape === undefined || readShape.join() !== basisShape?.join()) {
      throw coreError(
        "CORE_PACK_FIT_STALE",
        `Manifold ${namespace}/${name} layer ${layer} has no read rows beside its basis (fitted before typed frames); refit it`,
      );
    }
  }
  const nodeStatements = await readCoreNodeStatements(verified, request.signal, labels.length);
  const nodeCoordinates = coreNodeCoordinates(tensors, labels.length);
  const fitMode = String(verified.manifold.fit_mode);
  const resolvedFitMode = fitMode === "auto"
    ? originPerLayer.size > 0 ? "spectral" : "pca"
    : fitMode;
  const domain = sidecar.domain;
  const intrinsicDimension = domainIntrinsicDimension(domain, `${namespace}/${name}`);
  const description = typeof verified.manifold.description === "string"
    ? verified.manifold.description
    : "";
  const source = typeof verified.manifold.source === "string" ? verified.manifold.source : "local";
  const tags = Array.isArray(verified.manifold.tags) &&
      verified.manifold.tags.every((value) => typeof value === "string")
    ? [...verified.manifold.tags] as string[]
    : [];
  const templateReference = verified.manifold.template_ref === null ||
      typeof verified.manifold.template_ref === "string"
    ? verified.manifold.template_ref as string | null
    : null;
  const nodeRoles = nodes.map((node) =>
    typeof node === "object" && node !== null &&
        ((node as { role?: unknown }).role === null || typeof (node as { role?: unknown }).role === "string")
      ? (node as { role: string | null }).role
      : null
  );
  const nodeKinds = nodes.map((node) =>
    typeof node === "object" && node !== null &&
        ((node as { kind?: unknown }).kind === null || typeof (node as { kind?: unknown }).kind === "string")
      ? (node as { kind: string | null }).kind
      : null
  );
  const info: ManifoldInfo = {
    namespace,
    name,
    description,
    source,
    tags,
    template_ref: templateReference,
    fit_mode: fitMode,
    is_discover: fitMode === "pca" || fitMode === "spectral" || fitMode === "auto",
    domain: structuredClone(domain) as ManifoldInfo["domain"],
    domain_label: domainLabel(domain),
    intrinsic_dim: intrinsicDimension,
    min_nodes: intrinsicDimension > 0 ? 2 * intrinsicDimension + 1 : null,
    node_count: labels.length,
    node_labels: [...labels],
    node_coords: nodeCoordinates,
    node_roles: nodeRoles,
    node_kinds: nodeKinds,
    hyperparams: isRecord(verified.manifold.hyperparams)
      ? structuredClone(verified.manifold.hyperparams) as Record<string, number | string>
      : {},
    fitted_models: [request.variant.runtimeIdentity.sourceModel],
    tensor_variants: {
      [request.variant.runtimeIdentity.sourceModel]: [
        fitted.variant === "raw" ? "raw" : `sae-${fitted.variantIdentity}`,
      ],
    },
    fitted_for_session: true,
    stale: false,
    resolved_fit_mode: resolvedFitMode,
    nodes: labels.map((label, index) => ({
      label,
      coords: nodeCoordinates[index] ? [...nodeCoordinates[index]] : null,
      statements: nodeStatements[index],
      role: nodeRoles[index],
    })),
    fitted: [{
      stem: fitted.tensorPath.split("/").at(-1)!.replace(/\.safetensors$/u, ""),
      method: typeof sidecar.method === "string" ? sidecar.method : fitMode,
      feature_space: typeof sidecar.feature_space === "string" ? sidecar.feature_space : "raw",
      node_count: labels.length,
      nodes_sha256: typeof sidecar.nodes_sha256 === "string" ? sidecar.nodes_sha256 : null,
      fit_mode: fitMode,
      hyperparams: structuredClone(sidecar.hyperparams) as Record<string, number | string>,
      diagnostics: structuredClone(sidecar.diagnostics) as ManifoldFitInfo["diagnostics"],
    }],
  };
  const safeVariantIdentity = fitted.variantIdentity === null
    ? null
    : encodeVariantIdentity(fitted.variantIdentity);
  const uniformRole = nodeRoles.length > 0 && nodeRoles[0] !== null &&
      nodeRoles.every((role) => role === nodeRoles[0])
    ? nodeRoles[0]
    : null;
  return {
    key: `${namespace}/${name}`,
    selectorKey: fitted.variant === "raw"
      ? uniformRole === null
        ? `${namespace}/${name}`
        : `${namespace}/${name}:role-${uniformRole}`
      : `${namespace}/${name}:sae-${safeVariantIdentity}`,
    namespace,
    name,
    featureSpace: String(sidecar.feature_space),
    variant: fitted.variant as "raw" | "sae",
    variantIdentity: fitted.variantIdentity,
    safeVariantIdentity,
    fitMode,
    labels,
    nodeRoles,
    nodeCoordinates,
    tensorPath: fitted.tensorPath,
    tensors,
    mahalanobisShares,
    domain,
    originPerLayer,
    info,
  };
}

async function readCoreNodeStatements(
  verified: VerifiedDrowseArchive,
  signal: AbortSignal,
  nodeCount: number,
): Promise<string[][]> {
  const prefix = `${verified.manifest.primary}/nodes/`;
  const paths = Object.keys(verified.files)
    .filter((path) => path.startsWith(prefix) && path.endsWith(".json"))
    .sort();
  if (paths.length !== nodeCount) {
    throw coreError(
      "CORE_PACK_MANIFOLD_INVALID",
      `Core manifold ${verified.manifest.primary} has an incomplete node corpus`,
    );
  }
  return Promise.all(paths.map(async (path) => {
    const bytes = await readVerifiedDrowseArchiveFile(verified, path, signal);
    const value: unknown = JSON.parse(new TextDecoder().decode(bytes));
    if (!Array.isArray(value) || !value.every((item) => typeof item === "string")) {
      throw coreError("CORE_PACK_MANIFOLD_INVALID", `Core manifold node file ${path} is invalid`);
    }
    return value as string[];
  }));
}

function coreNodeCoordinates(
  tensors: Fp32SafetensorsReader,
  nodeCount: number,
): number[][] {
  const shape = tensors.description.shapes.get("node_coords");
  if (!shape || shape.length !== 2 || shape[0] !== nodeCount || shape[1] < 1) {
    throw coreError("CORE_PACK_MANIFOLD_INVALID", "Core manifold node coordinates are invalid");
  }
  return rows(
    tensors.tensor("node_coords", [nodeCount, shape[1]]).data,
    nodeCount,
    shape[1],
  ).map((row) => [...row]);
}

function nearestNodeRole(
  manifold: CoreManifold,
  position: readonly number[] | string,
): string | null {
  if (typeof position === "string") {
    const index = manifold.labels.indexOf(position);
    if (index < 0) {
      throw coreError(
        "CORE_PACK_MANIFOLD_POSITION_INVALID",
        `Manifold ${manifold.key} has no node labeled ${position}`,
      );
    }
    return manifold.nodeRoles[index] ?? null;
  }
  const dimensions = domainIntrinsicDimension(manifold.domain, manifold.key);
  if (
    position.length !== dimensions ||
    position.some((coordinate) => !Number.isFinite(coordinate))
  ) {
    throw coreError(
      "CORE_PACK_MANIFOLD_POSITION_INVALID",
      `Manifold ${manifold.key} expects ${dimensions} authoring coordinates`,
    );
  }
  const embeddedPosition = embedCoreDomainPoint(
    manifold.domain,
    clampCoreDomainPoint(manifold.domain, position, manifold.key),
    manifold.key,
  );
  let nearestIndex = -1;
  let nearestDistance = Number.POSITIVE_INFINITY;
  manifold.nodeCoordinates.forEach((coordinates, index) => {
    const embeddedNode = embedCoreDomainPoint(
      manifold.domain,
      clampCoreDomainPoint(manifold.domain, coordinates, manifold.key),
      manifold.key,
    );
    let squaredDistance = 0;
    for (let dimension = 0; dimension < embeddedPosition.length; dimension += 1) {
      squaredDistance += (embeddedPosition[dimension] - embeddedNode[dimension]) ** 2;
    }
    if (squaredDistance < nearestDistance) {
      nearestDistance = squaredDistance;
      nearestIndex = index;
    }
  });
  if (nearestIndex < 0) {
    throw coreError(
      "CORE_PACK_MANIFOLD_POSITION_INVALID",
      `Manifold ${manifold.key} has no authored nodes`,
    );
  }
  return manifold.nodeRoles[nearestIndex] ?? null;
}

function clampCoreDomainPoint(
  domain: Record<string, unknown>,
  coordinates: readonly number[],
  key: string,
): number[] {
  if (domain.type === "custom") return [...coordinates];
  if (domain.type === "box" && Array.isArray(domain.axes)) {
    const axes = domain.axes;
    return coordinates.map((coordinate, index) => {
      const axis = axes[index];
      if (
        !isRecord(axis) || typeof axis.periodic !== "boolean" ||
        typeof axis.period !== "number" || !Number.isFinite(axis.period) || axis.period <= 0 ||
        typeof axis.lo !== "number" || !Number.isFinite(axis.lo) ||
        typeof axis.hi !== "number" || !Number.isFinite(axis.hi) || axis.lo >= axis.hi
      ) {
        throw coreError("CORE_PACK_MANIFOLD_INVALID", `Manifold ${key} has an invalid box axis`);
      }
      if (axis.periodic) return positiveRemainder(coordinate, axis.period);
      return Math.min(axis.hi, Math.max(axis.lo, coordinate));
    });
  }
  if (domain.type === "sphere" && Number.isSafeInteger(domain.dim) && (domain.dim as number) > 0) {
    return coordinates.map((coordinate, index) =>
      index === coordinates.length - 1
        ? positiveRemainder(coordinate, 2 * Math.PI)
        : Math.min(Math.PI, Math.max(0, coordinate))
    );
  }
  throw coreError("CORE_PACK_MANIFOLD_INVALID", `Manifold ${key} has an invalid domain`);
}

function embedCoreDomainPoint(
  domain: Record<string, unknown>,
  coordinates: readonly number[],
  key: string,
): number[] {
  if (domain.type === "custom") return [...coordinates];
  if (domain.type === "box" && Array.isArray(domain.axes)) {
    const embedded: number[] = [];
    domain.axes.forEach((axis, index) => {
      if (!isRecord(axis) || typeof axis.periodic !== "boolean") {
        throw coreError("CORE_PACK_MANIFOLD_INVALID", `Manifold ${key} has an invalid box axis`);
      }
      if (axis.periodic) {
        if (typeof axis.period !== "number" || !Number.isFinite(axis.period) || axis.period <= 0) {
          throw coreError("CORE_PACK_MANIFOLD_INVALID", `Manifold ${key} has an invalid box period`);
        }
        const angle = 2 * Math.PI * coordinates[index] / axis.period;
        embedded.push(Math.cos(angle), Math.sin(angle));
      } else {
        embedded.push(coordinates[index]);
      }
    });
    return embedded;
  }
  if (domain.type === "sphere" && Number.isSafeInteger(domain.dim) && (domain.dim as number) > 0) {
    const embedded = new Array<number>((domain.dim as number) + 1);
    let running = 1;
    for (let dimension = 0; dimension < (domain.dim as number); dimension += 1) {
      embedded[dimension] = running * Math.cos(coordinates[dimension]);
      running *= Math.sin(coordinates[dimension]);
    }
    embedded[domain.dim as number] = running;
    return embedded;
  }
  throw coreError("CORE_PACK_MANIFOLD_INVALID", `Manifold ${key} has an invalid domain`);
}

function positiveRemainder(value: number, modulus: number): number {
  return ((value % modulus) + modulus) % modulus;
}

function domainIntrinsicDimension(domain: Record<string, unknown>, key: string): number {
  if (domain.type === "box" && Array.isArray(domain.axes)) return domain.axes.length;
  if (domain.type === "sphere" && Number.isSafeInteger(domain.dim)) return domain.dim as number;
  if (domain.type === "custom" && Number.isSafeInteger(domain.embed_dim)) {
    return domain.embed_dim as number;
  }
  throw coreError("CORE_PACK_MANIFOLD_INVALID", `Manifold ${key} has an invalid domain`);
}

function mergedManifoldInfo(variants: readonly CoreManifold[]): ManifoldInfo {
  if (variants.length === 0) {
    throw coreError("CORE_PACK_SELECTOR_NOT_FOUND", "The browser manifold is not installed");
  }
  const info = structuredClone(variants[0].info);
  const fittedModels = new Set<string>();
  const fitted = new Map<string, NonNullable<ManifoldInfo["fitted"]>[number]>();
  const tensorVariants = new Map<string, Set<string>>();
  for (const variant of variants) {
    for (const model of variant.info.fitted_models) fittedModels.add(model);
    for (const entry of variant.info.fitted ?? []) fitted.set(entry.stem, structuredClone(entry));
    for (const [model, names] of Object.entries(variant.info.tensor_variants)) {
      const merged = tensorVariants.get(model) ?? new Set<string>();
      for (const name of names) merged.add(name);
      tensorVariants.set(model, merged);
    }
  }
  info.fitted_models = [...fittedModels].sort();
  info.fitted = [...fitted.values()];
  info.tensor_variants = Object.fromEntries(
    [...tensorVariants].map(([model, names]) => [model, [...names].sort()]),
  );
  return info;
}

export async function loadCoreWhiteners(
  tensorArtifact: BrowserModelArtifact,
  sidecarArtifact: BrowserModelArtifact,
  request: BrowserModelLoadRequest,
): Promise<ReadonlyMap<number, SerializedMahalanobisWhitener>> {
  request.signal.throwIfAborted();
  const sidecar = JSON.parse(await sidecarArtifact.file.text()) as unknown;
  if (!isRecord(sidecar)) {
    throw coreError("CORE_PACK_WHITENER_INVALID", "The neutral whitener sidecar is invalid");
  }
  const expectedKeys = [
    "format_version",
    "runtime_identity_sha256",
    "context_binding_sha256",
    "hidden_size",
    "layer_map",
    "tensors_sha256",
    "ridge_per_layer",
  ].sort();
  const keys = Object.keys(sidecar).sort();
  const context = request.variant.contextProfiles?.find(
    (profile) => profile.contextTokens === request.contextTokens,
  );
  const compatibleBindings = runtimeBindings(request, request.contextTokens);
  if (
    keys.length !== expectedKeys.length ||
    keys.some((key, index) => key !== expectedKeys[index]) ||
    sidecar.format_version !== 1 ||
    !context || !compatibleBindings.some((binding) =>
      sidecar.runtime_identity_sha256 === binding.runtimeIdentitySha256 &&
      sidecar.context_binding_sha256 === binding.contextBindingSha256
    ) ||
    sidecar.hidden_size !== request.variant.runtimeIdentity.hiddenSize ||
    !sameLayerMap(sidecar.layer_map, request.variant.runtimeIdentity.layerMap) ||
    sidecar.tensors_sha256 !== tensorArtifact.manifest.sha256 ||
    !isRecord(sidecar.ridge_per_layer)
  ) {
    throw coreError(
      "CORE_PACK_WHITENER_INVALID",
      "The neutral whitener is not bound to the selected browser runtime and context",
    );
  }
  const tensors = decodeFp32Safetensors(
    new Uint8Array(await tensorArtifact.file.arrayBuffer()),
    tensorArtifact.manifest.path,
  );
  const hiddenSize = request.variant.runtimeIdentity.hiddenSize;
  const layerMap = request.variant.runtimeIdentity.layerMap;
  const expectedTensorKeys: string[] = [];
  const result = new Map<number, SerializedMahalanobisWhitener>();
  for (const layer of layerMap) {
    const prefix = `layer_${layer}`;
    expectedTensorKeys.push(
      `${prefix}.mean`,
      `${prefix}.basis`,
      `${prefix}.eigenvalues`,
      `${prefix}.inverse_scales`,
    );
    const basisShape = tensors.description.shapes.get(`${prefix}.basis`);
    const ridge = sidecar.ridge_per_layer[String(layer)];
    if (
      !basisShape || basisShape.length !== 2 || basisShape[1] !== hiddenSize ||
      typeof ridge !== "number" || !Number.isFinite(ridge) || ridge <= 0
    ) {
      throw coreError("CORE_PACK_WHITENER_INVALID", `Neutral whitener layer ${layer} is invalid`);
    }
    const rank = basisShape[0];
    const mean = tensors.tensor(`${prefix}.mean`, [hiddenSize]).data;
    const basis = tensors.tensor(`${prefix}.basis`, [rank, hiddenSize]).data;
    const eigenvalues = tensors.tensor(`${prefix}.eigenvalues`, [rank]).data;
    const inverseScales = tensors.tensor(`${prefix}.inverse_scales`, [rank]).data;
    if (
      eigenvalues.some((value) => value <= 0) ||
      inverseScales.some((value) => value <= 0)
    ) {
      throw coreError("CORE_PACK_WHITENER_INVALID", `Neutral whitener layer ${layer} is invalid`);
    }
    result.set(layer, {
      columns: hiddenSize,
      rank,
      ridge,
      mean: Float64Array.from(mean),
      basis: Float64Array.from(basis),
      eigenvalues: Float64Array.from(eigenvalues),
      inverseScales: Float64Array.from(inverseScales),
    });
  }
  expectedTensorKeys.sort();
  if (
    tensors.keys.length !== expectedTensorKeys.length ||
    tensors.keys.some((key, index) => key !== expectedTensorKeys[index]) ||
    Object.keys(sidecar.ridge_per_layer).length !== layerMap.length
  ) {
    throw coreError(
      "CORE_PACK_WHITENER_INVALID",
      "The neutral whitener tensor closure does not exactly match the layer map",
    );
  }
  return result;
}

async function selectCoreWhiteners(
  tensorArtifacts: readonly BrowserModelArtifact[],
  sidecarArtifacts: readonly BrowserModelArtifact[],
  request: BrowserModelLoadRequest,
): Promise<ReadonlyMap<number, SerializedMahalanobisWhitener>> {
  if (tensorArtifacts.length === 0 && sidecarArtifacts.length === 0) {
    return new Map<number, SerializedMahalanobisWhitener>();
  }
  const tensors = keyedWhitenerArtifacts(tensorArtifacts, "tensor");
  const sidecars = keyedWhitenerArtifacts(sidecarArtifacts, "sidecar");
  if (
    tensors.size !== sidecars.size ||
    [...tensors.keys()].some((key) => !sidecars.has(key))
  ) {
    throw coreError(
      "CORE_PACK_WHITENER_INVALID",
      "Every context-specific neutral whitener must contain one tensor and one sidecar",
    );
  }
  const profile = request.variant.contextProfiles?.find(
    (candidate) => candidate.contextTokens === request.contextTokens,
  );
  if (!profile) {
    throw coreError(
      "CORE_PACK_WHITENER_INVALID",
      "The selected context is not present in the model's runtime identity",
    );
  }
  const allowedBindings = (request.variant.contextProfiles ?? []).flatMap((candidate) =>
    runtimeBindings(request, candidate.contextTokens)
  );
  const selectedBindings = runtimeBindings(request, request.contextTokens);
  const seenBindings = new Set<string>();
  let selected: { tensor: BrowserModelArtifact; sidecar: BrowserModelArtifact } | null = null;
  for (const [key, tensor] of tensors) {
    request.signal.throwIfAborted();
    const sidecar = sidecars.get(key)!;
    let metadata: unknown;
    try {
      metadata = JSON.parse(await sidecar.file.text());
    } catch {
      throw coreError("CORE_PACK_WHITENER_INVALID", "A neutral whitener sidecar is not valid JSON");
    }
    if (
      !isRecord(metadata) || metadata.format_version !== 1 ||
      typeof metadata.context_binding_sha256 !== "string" ||
      !allowedBindings.some((binding) =>
        metadata.runtime_identity_sha256 === binding.runtimeIdentitySha256 &&
        metadata.context_binding_sha256 === binding.contextBindingSha256
      ) ||
      metadata.tensors_sha256 !== tensor.manifest.sha256 ||
      seenBindings.has(metadata.context_binding_sha256)
    ) {
      throw coreError(
        "CORE_PACK_WHITENER_INVALID",
        "A context-specific neutral whitener is not bound to this model runtime",
      );
    }
    seenBindings.add(metadata.context_binding_sha256);
    if (selectedBindings.some((binding) =>
      metadata.runtime_identity_sha256 === binding.runtimeIdentitySha256 &&
      metadata.context_binding_sha256 === binding.contextBindingSha256
    )) {
      selected = { tensor, sidecar };
    }
  }
  if (selected === null) {
    throw coreError(
      "CORE_PACK_WHITENER_INVALID",
      `The required core pack has no neutral whitener for the ${request.contextTokens}-token context`,
    );
  }
  return loadCoreWhiteners(selected.tensor, selected.sidecar, request);
}

function keyedWhitenerArtifacts(
  artifacts: readonly BrowserModelArtifact[],
  label: string,
): ReadonlyMap<string, BrowserModelArtifact> {
  const result = new Map<string, BrowserModelArtifact>();
  for (const artifact of artifacts) {
    const path = artifact.manifest.path;
    const separator = path.lastIndexOf("/");
    const key = separator < 0 ? "" : path.slice(0, separator);
    if (result.has(key)) {
      throw coreError(
        "CORE_PACK_WHITENER_INVALID",
        `The required core pack repeats a neutral whitener ${label} directory`,
      );
    }
    result.set(key, artifact);
  }
  return result;
}

function sameLayerMap(value: unknown, expected: readonly number[]): boolean {
  return Array.isArray(value) && value.length === expected.length &&
    value.every((layer, index) => layer === expected[index]);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function matchingFittedArtifacts(
  verified: VerifiedDrowseArchive,
  request: BrowserModelLoadRequest,
  requireContextBinding = false,
): readonly VerifiedDrowseArchive["fittedArtifacts"][number][] {
  const compatibleBindings = runtimeBindings(request, request.contextTokens);
  return verified.fittedArtifacts.filter((artifact) =>
    (artifact.variant === "raw" && artifact.variantIdentity === null ||
      artifact.variant === "sae" && artifact.variantIdentity !== null) &&
    artifact.modelId === request.variant.runtimeIdentity.sourceModel &&
    compatibleBindings.some((binding) =>
      artifact.modelFingerprint === binding.runtimeIdentitySha256 &&
      (!requireContextBinding && artifact.contextBindingSha256 === null ||
        artifact.contextBindingSha256 === binding.contextBindingSha256)
    )
  );
}

/** The variant's runtime binding at `contextTokens` (none when the variant does not offer that
 *  context): what a core artifact's fingerprint and context binding must equal. */
function runtimeBindings(
  request: BrowserModelLoadRequest,
  contextTokens: number,
): readonly {
  runtimeIdentitySha256: string;
  contextBindingSha256: string;
}[] {
  const profile = request.variant.contextProfiles?.find(
    (candidate) => candidate.contextTokens === contextTokens,
  );
  if (!profile) return [];
  return [{
    runtimeIdentitySha256: request.variant.runtimeIdentitySha256,
    contextBindingSha256: profile.bindingSha256,
  }];
}

function fittedLayerNumbers(tensors: Fp32SafetensorsReader): number[] {
  const layers = new Set<number>();
  for (const key of tensors.keys) {
    const match = /^layer_(0|[1-9]\d*)\.basis$/u.exec(key);
    if (match) layers.add(Number(match[1]));
  }
  return [...layers].sort((left, right) => left - right);
}

function updateFp32Hash(
  hash: ReturnType<typeof sha256.create>,
  values: Float32Array,
): void {
  hash.update(new Uint8Array(values.buffer, values.byteOffset, values.byteLength));
}

function isCurvedManifold(manifold: CoreManifold): boolean {
  return fittedLayerNumbers(manifold.tensors).some((layer) =>
    manifold.tensors.keys.includes(`layer_${layer}.node_params`)
  );
}

function manifoldLabelNeedsPosition(manifold: CoreManifold): boolean {
  if (isCurvedManifold(manifold)) return true;
  return fittedLayerNumbers(manifold.tensors).some((layer) => {
    const shape = manifold.tensors.description.shapes.get(`layer_${layer}.basis`);
    return shape !== undefined && shape[0] !== 1;
  });
}

function isCurvedPositionTerm(term: ResolvedCoreTerm): boolean {
  return term.manifoldPosition !== null &&
    term.base.kind === "manifold" &&
    isCurvedManifold(term.base.manifold);
}

function groupAffineTermsByTrigger(
  terms: readonly ResolvedCoreTerm[],
): StructuredAffineTermGroup[] {
  const groups = new Map<string, StructuredAffineTermGroup>();
  for (const term of terms) {
    if (Math.abs(term.coefficient) < 1e-9) continue;
    const key = structuredTriggerKey(term.trigger);
    let group = groups.get(key);
    if (group === undefined) {
      group = { trigger: term.trigger, terms: [] };
      groups.set(key, group);
    }
    group.terms.push(term);
  }
  return [...groups.values()];
}

function termActsAt(term: ResolvedCoreTerm, layer: number): boolean {
  return term.actsAt === null || term.actsAt.has(layer);
}

function structuredTriggerKey(trigger: SteeringTrigger | null): string {
  const phase = trigger?.phase ?? { kind: "both" as const };
  const phaseKey = phase.kind === "first" || phase.kind === "after"
    ? `${phase.kind}:${phase.tokens}`
    : phase.kind;
  const gate = trigger?.gate;
  return gate === null || gate === undefined
    ? `${phaseKey}|none`
    : `${phaseKey}|${gate.probe}|${gate.operator}|${gate.threshold}`;
}

function symmetricEigen(
  matrix: Float64Array,
  dimensions: number,
): { values: Float64Array; vectors: Float64Array } {
  const diagonalized = matrix.slice();
  const vectors = new Float64Array(dimensions * dimensions);
  for (let index = 0; index < dimensions; index += 1) {
    vectors[index * dimensions + index] = 1;
  }
  const tolerance = Math.max(...diagonalized.map(Math.abs), 0) *
    Number.EPSILON * dimensions;
  for (
    let iteration = 0;
    iteration < Math.max(32, dimensions * dimensions * 64);
    iteration += 1
  ) {
    let left = 0;
    let right = 0;
    let largest = 0;
    for (let row = 0; row < dimensions; row += 1) {
      for (let column = row + 1; column < dimensions; column += 1) {
        const magnitude = Math.abs(diagonalized[row * dimensions + column]);
        if (magnitude > largest) {
          largest = magnitude;
          left = row;
          right = column;
        }
      }
    }
    if (largest <= tolerance) break;
    const leftValue = diagonalized[left * dimensions + left];
    const rightValue = diagonalized[right * dimensions + right];
    const cross = diagonalized[left * dimensions + right];
    const angle = 0.5 * Math.atan2(2 * cross, rightValue - leftValue);
    const cosine = Math.cos(angle);
    const sine = Math.sin(angle);
    for (let index = 0; index < dimensions; index += 1) {
      if (index === left || index === right) continue;
      const oldLeft = diagonalized[index * dimensions + left];
      const oldRight = diagonalized[index * dimensions + right];
      const nextLeft = cosine * oldLeft - sine * oldRight;
      const nextRight = sine * oldLeft + cosine * oldRight;
      diagonalized[index * dimensions + left] = nextLeft;
      diagonalized[left * dimensions + index] = nextLeft;
      diagonalized[index * dimensions + right] = nextRight;
      diagonalized[right * dimensions + index] = nextRight;
    }
    diagonalized[left * dimensions + left] = cosine * cosine * leftValue -
      2 * sine * cosine * cross + sine * sine * rightValue;
    diagonalized[right * dimensions + right] = sine * sine * leftValue +
      2 * sine * cosine * cross + cosine * cosine * rightValue;
    diagonalized[left * dimensions + right] = 0;
    diagonalized[right * dimensions + left] = 0;
    for (let index = 0; index < dimensions; index += 1) {
      const oldLeft = vectors[index * dimensions + left];
      const oldRight = vectors[index * dimensions + right];
      vectors[index * dimensions + left] = cosine * oldLeft - sine * oldRight;
      vectors[index * dimensions + right] = sine * oldLeft + cosine * oldRight;
    }
  }
  const order = Array.from({ length: dimensions }, (_, index) => index)
    .sort((left, right) =>
      diagonalized[left * dimensions + left] -
      diagonalized[right * dimensions + right]
    );
  const values = Float64Array.from(
    order,
    (index) => diagonalized[index * dimensions + index],
  );
  const orderedVectors = new Float64Array(dimensions * dimensions);
  order.forEach((source, column) => {
    for (let row = 0; row < dimensions; row += 1) {
      orderedVectors[row * dimensions + column] = vectors[row * dimensions + source];
    }
  });
  return { values, vectors: orderedVectors };
}

function centeredCoordinateRank(
  nodes: readonly ArrayLike<number>[],
  columns: number,
): number {
  const means = Float64Array.from({ length: columns }, (_, column) =>
    nodes.reduce((sum, node) => sum + node[column], 0) / nodes.length
  );
  const centeredColumns = Array.from({ length: columns }, (_, column) =>
    Float64Array.from(nodes, (node) => node[column] - means[column])
  );
  const maxNorm = centeredColumns.reduce(
    (maximum, column) => Math.max(maximum, euclideanNorm(column)),
    0,
  );
  if (maxNorm === 0) return 0;
  const tolerance = Math.max(nodes.length, columns) * 2 ** -23 * maxNorm;
  const orthonormal: Float64Array[] = [];
  for (const column of centeredColumns) {
    const residual = column.slice();
    for (let pass = 0; pass < 2; pass += 1) {
      for (const basis of orthonormal) {
        const projection = dot(residual, basis);
        for (let row = 0; row < residual.length; row += 1) {
          residual[row] -= projection * basis[row];
        }
      }
    }
    const norm = euclideanNorm(residual);
    if (norm <= tolerance) continue;
    orthonormal.push(Float64Array.from(residual, (value) => value / norm));
  }
  return orthonormal.length;
}

function resolveAffinePositionWeights(
  manifold: CoreManifold,
  position: readonly number[] | string,
): Float64Array {
  const nodeCount = manifold.labels.length;
  if (typeof position === "string") {
    const labelIndex = manifold.labels.indexOf(position);
    if (labelIndex < 0) {
      throw coreError(
        "CORE_PACK_SELECTOR_NOT_FOUND",
        `Core manifold ${manifold.key} has no node ${position}`,
      );
    }
    return Float64Array.from({ length: nodeCount }, (_, index) =>
      index === labelIndex ? 1 : 0
    );
  }

  const intrinsicDim = manifoldIntrinsicDimension(manifold);
  if (
    position.length !== intrinsicDim ||
    position.some((coordinate) => !Number.isFinite(coordinate))
  ) {
    throw coreError(
      "CORE_PACK_AFFINE_POSITION_INVALID",
      `Flat steering on ${manifold.key} requires ${intrinsicDim} target coordinate${intrinsicDim === 1 ? "" : "s"}`,
    );
  }
  const nodes = coreNodeCoordinates(manifold.tensors, nodeCount);
  if (nodes.some((node) => node.length !== intrinsicDim) || nodeCount < intrinsicDim + 1) {
    throw coreError(
      "CORE_PACK_AFFINE_POSITION_INVALID",
      `Manifold ${manifold.key} cannot interpolate a free coordinate; steer by node label instead`,
    );
  }

  const lower = Float64Array.from({ length: intrinsicDim }, (_, axis) =>
    Math.min(...nodes.map((node) => node[axis]))
  );
  const scale = Float64Array.from({ length: intrinsicDim }, (_, axis) =>
    Math.max(Math.max(...nodes.map((node) => node[axis])) - lower[axis], 1e-9)
  );
  const normalizedNodes = nodes.map((node) =>
    Float64Array.from(node, (coordinate, axis) => (coordinate - lower[axis]) / scale[axis])
  );
  const normalizedQuery = Float64Array.from(
    position,
    (coordinate, axis) => (coordinate - lower[axis]) / scale[axis],
  );
  if (centeredCoordinateRank(normalizedNodes, intrinsicDim) !== intrinsicDim) {
    throw coreError(
      "CORE_PACK_AFFINE_POSITION_INVALID",
      `Manifold ${manifold.key} cannot interpolate a free coordinate; steer by node label instead`,
    );
  }
  const size = nodeCount + intrinsicDim + 1;
  const matrix = new Float64Array(size * size);
  const right = new Float64Array(size);
  for (let row = 0; row < nodeCount; row += 1) {
    for (let column = 0; column < nodeCount; column += 1) {
      matrix[row * size + column] = rowDistance(
        normalizedNodes[row],
        normalizedNodes[column],
      ) ** 3;
    }
    matrix[row * size + nodeCount] = 1;
    matrix[nodeCount * size + row] = 1;
    for (let axis = 0; axis < intrinsicDim; axis += 1) {
      const value = normalizedNodes[row][axis];
      matrix[row * size + nodeCount + 1 + axis] = value;
      matrix[(nodeCount + 1 + axis) * size + row] = value;
    }
    right[row] = rowDistance(normalizedQuery, normalizedNodes[row]) ** 3;
  }
  right[nodeCount] = 1;
  for (let axis = 0; axis < intrinsicDim; axis += 1) {
    right[nodeCount + 1 + axis] = normalizedQuery[axis];
  }

  const solved = solveDenseSystem(matrix, right, size);
  let residual = 0;
  let scaleReference = 1;
  for (let row = 0; row < size; row += 1) {
    let actual = 0;
    for (let column = 0; column < size; column += 1) {
      actual += matrix[row * size + column] * solved[column];
    }
    residual = Math.max(residual, Math.abs(actual - right[row]));
    scaleReference = Math.max(scaleReference, Math.abs(actual), Math.abs(right[row]));
  }
  const weights = solved.slice(0, nodeCount);
  const weightSum = weights.reduce((sum, value) => sum + value, 0);
  if (
    weights.some((value) => !Number.isFinite(value)) ||
    residual > 1e-7 * scaleReference ||
    Math.abs(weightSum - 1) > 1e-6
  ) {
    throw coreError(
      "CORE_PACK_AFFINE_POSITION_INVALID",
      `Manifold ${manifold.key} cannot interpolate this free coordinate; steer by node label instead`,
    );
  }
  return weights;
}

function affineManifoldPositionDirection(
  atom: ResolvedManifoldAtom,
  weights: Float64Array,
  layer: number,
  hiddenSize: number,
): Float64Array | null {
  const manifold = atom.manifold;
  const prefix = `layer_${layer}`;
  const basisShape = manifold.tensors.description.shapes.get(`${prefix}.basis`);
  if (basisShape === undefined) return null;
  const nodeShape = manifold.tensors.description.shapes.get(`${prefix}.node_coords`);
  const nodeCount = manifold.labels.length;
  if (
    basisShape.length !== 2 || basisShape[1] !== hiddenSize || basisShape[0] < 1 ||
    !nodeShape || nodeShape.length !== 2 ||
    nodeShape[0] !== nodeCount || nodeShape[1] !== basisShape[0] ||
    weights.length !== nodeCount
  ) {
    throw coreError(
      "CORE_PACK_TENSOR_INVALID",
      `Flat manifold ${manifold.key} layer ${layer} has incompatible fitted tensors`,
    );
  }
  const rank = basisShape[0];
  const basis = manifold.tensors.tensor(`${prefix}.basis`, [rank, hiddenSize]).data;
  const nodeCoordinates = manifold.tensors.tensor(
    `${prefix}.node_coords`,
    [nodeCount, rank],
  ).data;
  const target = new Float64Array(rank);
  for (let node = 0; node < nodeCount; node += 1) {
    for (let axis = 0; axis < rank; axis += 1) {
      target[axis] += weights[node] * nodeCoordinates[node * rank + axis];
    }
  }
  const direction = new Float64Array(hiddenSize);
  for (let axis = 0; axis < rank; axis += 1) {
    for (let column = 0; column < hiddenSize; column += 1) {
      direction[column] += target[axis] * basis[axis * hiddenSize + column];
    }
  }
  return direction;
}

function encodeVariantIdentity(value: string): string {
  const alphabet = "abcdefghijklmnopqrstuvwxyz234567";
  let accumulator = 0;
  let bits = 0;
  let output = "";
  for (const byte of new TextEncoder().encode(value)) {
    accumulator = (accumulator << 8) | byte;
    bits += 8;
    while (bits >= 5) {
      bits -= 5;
      output += alphabet[(accumulator >>> bits) & 31];
      accumulator &= (1 << bits) - 1;
    }
  }
  if (bits > 0) output += alphabet[(accumulator << (5 - bits)) & 31];
  return `_z${output}`;
}

function disabledStructuredGroup(hiddenSize: number): StructuredAffineGroup {
  const basis = new Float32Array(hiddenSize);
  basis[0] = 1;
  return {
    active: false,
    basis: [basis],
    read: [basis],
    neutral: new Float32Array(hiddenSize),
    target: [0],
    along: 0,
    kappa: [0],
  };
}

function disabledStructuredProbe(hiddenSize: number): StructuredProbe {
  return { kind: "disabled", direction: new Float32Array(hiddenSize), bias: 0 };
}

/** What one gate key reads (`BrowserFeasibilityCorePackCompiler.resolveGate`). */
type GateResolution =
  | { readonly kind: "never" }
  | { readonly kind: "unit" | "slens" | "slot" | "geometry"; readonly probe: string };

type GateChannel =
  | { readonly kind: "axis"; readonly index: number }
  | { readonly kind: "fraction" | "membership" }
  | { readonly kind: "distance" | "assignment"; readonly label: string };

/** Native `_CHANNEL_WORDS`. */
const GATE_CHANNEL_WORDS: Readonly<Record<GateChannel["kind"], string>> = {
  axis: "coordinate axis",
  fraction: "subspace fraction",
  membership: "tube membership",
  distance: "label distance",
  assignment: "soft assignment",
};

/** Native `parse_gate_ref`: a verbatim gate scalar key into the probe it names and its channel
 *  (`name[i]`, `name:fraction`, `name:membership`, `name@label`, `name~label`, else axis 0). */
export function parseGateRef(key: string): { probe: string; channel: GateChannel } {
  if (key.endsWith("]")) {
    const open = key.lastIndexOf("[");
    const index = key.slice(open + 1, -1);
    if (open > 0 && /^\d+$/u.test(index)) return { probe: key.slice(0, open), channel: { kind: "axis", index: Number(index) } };
  }
  if (key.endsWith(":fraction")) return { probe: key.slice(0, -":fraction".length), channel: { kind: "fraction" } };
  if (key.endsWith(":membership")) return { probe: key.slice(0, -":membership".length), channel: { kind: "membership" } };
  for (const [separator, kind] of [["@", "distance"], ["~", "assignment"]] as const) {
    const at = key.indexOf(separator);
    if (at > 0 && at < key.length - 1) return { probe: key.slice(0, at), channel: { kind, label: key.slice(at + 1) } };
  }
  return { probe: key, channel: { kind: "axis", index: 0 } };
}

function hookControl(
  trigger: SteeringTrigger | null,
  gates: ReadonlyMap<string, GateResolution>,
  gateSlots: ReadonlyMap<string, readonly StructuredMeasurementSlot[]>,
): StructuredHookControl {
  const phase: SteeringPhase = trigger?.phase ?? { kind: "both" };
  const gate = trigger?.gate ?? null;
  if (gate === null) return { enabled: true, phase, gate: null };
  const resolved = gates.get(gate.probe) ?? { kind: "never" };
  // A gate that reads nothing never fires (native: no score, inactive).
  if (resolved.kind === "never") return { enabled: false, phase, gate: null };
  const read = resolved.kind === "slot"
    ? { slots: gateSlots.get(resolved.probe) ?? [] }
    : { slots: [], scoreKey: resolved.kind === "geometry" ? gate.probe : resolved.probe };
  return { enabled: true, phase, gate: { ...read, operator: gate.operator, threshold: gate.threshold } };
}

function structuredCurveDomain(
  domain: Record<string, unknown>,
  profile: StructuredHookCapacityProfile,
): {
  kind: "box" | "custom" | "sphere";
  axes: readonly StructuredCurveAxis[];
  embedDim: number;
} {
  if (domain.type === "box" && Array.isArray(domain.axes)) {
    if (domain.axes.length === 0 || domain.axes.length > profile.maxIntrinsicDim) {
      throw coreError(
        "CORE_PACK_CURVED_DIMENSION_UNAVAILABLE",
        `Browser curves support between one and ${profile.maxIntrinsicDim} authoring coordinates`,
      );
    }
    const axes = domain.axes.map((value): StructuredCurveAxis => {
      if (
        !isRecord(value) || typeof value.periodic !== "boolean" ||
        typeof value.period !== "number" || !Number.isFinite(value.period) || value.period <= 0 ||
        typeof value.lo !== "number" || !Number.isFinite(value.lo) ||
        typeof value.hi !== "number" || !Number.isFinite(value.hi) || value.lo >= value.hi
      ) {
        throw coreError("CORE_PACK_CURVED_TOPOLOGY_INVALID", "The manifold has an invalid box axis");
      }
      return {
        periodic: value.periodic,
        period: value.period,
        lowerBound: value.periodic ? 0 : value.lo,
        upperBound: value.periodic ? value.period : value.hi,
      };
    });
    const embedDim = axes.reduce((sum, axis) => sum + (axis.periodic ? 2 : 1), 0);
    if (embedDim > profile.maxEmbedDim) {
      throw coreError(
        "CORE_PACK_CURVED_DIMENSION_UNAVAILABLE",
        `Browser curve embeddings support at most ${profile.maxEmbedDim} coordinates`,
      );
    }
    return { kind: "box", axes, embedDim };
  }
  if (
    domain.type === "sphere" && Number.isSafeInteger(domain.dim) &&
    (domain.dim as number) >= 1 &&
    (domain.dim as number) <= profile.maxIntrinsicDim &&
    (domain.dim as number) + 1 <= profile.maxEmbedDim
  ) {
    const dim = domain.dim as number;
    const axes = Array.from({ length: dim }, (_, axis): StructuredCurveAxis => ({
      periodic: axis === dim - 1,
      period: 2 * Math.PI,
      lowerBound: 0,
      upperBound: axis === dim - 1 ? 2 * Math.PI : Math.PI,
    }));
    return { kind: "sphere", axes, embedDim: dim + 1 };
  }
  if (
    domain.type === "custom" && Number.isSafeInteger(domain.embed_dim) &&
    (domain.embed_dim as number) > 0 &&
    (domain.embed_dim as number) <= Math.min(
      profile.maxIntrinsicDim,
      Math.floor(profile.maxEmbedDim / 2),
    )
  ) {
    const embedDim = domain.embed_dim as number;
    if (domain.bounds !== null && domain.bounds !== undefined && !Array.isArray(domain.bounds)) {
      throw coreError("CORE_PACK_CURVED_TOPOLOGY_INVALID", "The custom domain bounds are invalid");
    }
    const rawBounds = domain.bounds as unknown[] | null | undefined;
    if (rawBounds && rawBounds.length !== embedDim) {
      throw coreError("CORE_PACK_CURVED_TOPOLOGY_INVALID", "The custom domain bounds are invalid");
    }
    const axes = Array.from({ length: embedDim }, (_, axis): StructuredCurveAxis => {
      const bound = rawBounds?.[axis];
      if (bound === undefined || bound === null) {
        return {
          periodic: false,
          period: 1,
          lowerBound: Number.NEGATIVE_INFINITY,
          upperBound: Number.POSITIVE_INFINITY,
        };
      }
      if (
        !Array.isArray(bound) || bound.length !== 2 ||
        bound.some((value) => typeof value !== "number" || !Number.isFinite(value)) ||
        (bound[0] as number) > (bound[1] as number)
      ) {
        throw coreError("CORE_PACK_CURVED_TOPOLOGY_INVALID", "The custom domain bounds are invalid");
      }
      return {
        periodic: false,
        period: 1,
        lowerBound: bound[0] as number,
        upperBound: bound[1] as number,
      };
    });
    return { kind: "custom", axes, embedDim };
  }
  throw coreError(
    "CORE_PACK_CURVED_TOPOLOGY_UNAVAILABLE",
    "The manifold domain exceeds the browser curve profile",
  );
}

function coreGeometryCurve(
  manifold: CoreManifold,
  layer: number,
  domain: {
    kind: "box" | "custom" | "sphere";
    axes: readonly StructuredCurveAxis[];
    embedDim: number;
  },
  hiddenSize: number,
): StructuredCurve {
  const prefix = `layer_${layer}`;
  const basisShape = manifold.tensors.description.shapes.get(`${prefix}.basis`);
  const nodeShape = manifold.tensors.description.shapes.get(`${prefix}.node_params`);
  if (!basisShape || !nodeShape || nodeShape[1] !== domain.embedDim) {
    throw coreError(
      "CORE_PACK_CURVED_SHAPE_INVALID",
      "The fitted RBF embedding does not match the manifold domain",
    );
  }
  const rank = basisShape[0];
  const nodeCount = nodeShape[0];
  const origin = manifold.originPerLayer.get(layer);
  if (origin === undefined || origin.length !== domain.axes.length) {
    throw coreError(
      "CORE_PACK_CURVED_SHAPE_INVALID",
      `Core manifold layer ${layer} has an incompatible origin`,
    );
  }
  const sigmaWeightKey = `${prefix}.sigma_rbf_weights`;
  const sigmaPolynomialKey = `${prefix}.sigma_poly_coeffs`;
  const hasSigma = manifold.tensors.keys.includes(sigmaWeightKey) &&
    manifold.tensors.keys.includes(sigmaPolynomialKey);
  return {
    active: true,
    domainKind: domain.kind,
    basis: rows(
      manifold.tensors.tensor(`${prefix}.basis`, [rank, hiddenSize]).data,
      rank,
      hiddenSize,
    ),
    read: rows(
      manifold.tensors.tensor(`${prefix}.read`, [rank, hiddenSize]).data,
      rank,
      hiddenSize,
    ),
    neutral: manifold.tensors.tensor(`${prefix}.mean`, [hiddenSize]).data,
    nodeParameters: rows(
      manifold.tensors.tensor(
        `${prefix}.node_params`,
        [nodeCount, domain.embedDim],
      ).data,
      nodeCount,
      domain.embedDim,
    ),
    rbfWeights: rows(
      manifold.tensors.tensor(`${prefix}.rbf_weights`, [nodeCount, rank]).data,
      nodeCount,
      rank,
    ),
    polynomial: rows(
      manifold.tensors.tensor(
        `${prefix}.poly_coeffs`,
        [domain.embedDim + 1, rank],
      ).data,
      domain.embedDim + 1,
      rank,
    ),
    coordinateOffset: manifold.tensors.tensor(
      `${prefix}.coord_offset`,
      [domain.embedDim],
    ).data,
    coordinateScale: manifold.tensors.tensor(
      `${prefix}.coord_scale`,
      [domain.embedDim],
    ).data,
    origin,
    target: origin,
    axes: domain.axes,
    along: 0,
    onto: 0,
    sigmaRbfWeights: hasSigma
      ? manifold.tensors.tensor(sigmaWeightKey, [nodeCount, 1]).data
      : null,
    sigmaPolynomial: hasSigma
      ? manifold.tensors.tensor(sigmaPolynomialKey, [domain.embedDim + 1, 1]).data
      : null,
    damping: 1e-3,
  };
}

function evaluateCoreCurve(
  curve: StructuredCurve,
  position: ArrayLike<number>,
): Float32Array {
  const embedded = embedCoreCurvePoint(curve, position);
  const normalized = Float64Array.from(embedded, (value, index) =>
    (value - curve.coordinateOffset[index]) / curve.coordinateScale[index]
  );
  const rank = curve.basis.length;
  const output = new Float32Array(rank);
  for (let axis = 0; axis < rank; axis += 1) {
    let value = curve.polynomial[0][axis];
    for (let embed = 0; embed < normalized.length; embed += 1) {
      value += normalized[embed] * curve.polynomial[embed + 1][axis];
    }
    for (let node = 0; node < curve.nodeParameters.length; node += 1) {
      let squared = 0;
      for (let embed = 0; embed < normalized.length; embed += 1) {
        squared += (normalized[embed] - curve.nodeParameters[node][embed]) ** 2;
      }
      value += Math.sqrt(squared) ** 3 * curve.rbfWeights[node][axis];
    }
    output[axis] = value;
  }
  return output;
}

function evaluateCoreSigma(
  curve: StructuredCurve,
  position: ArrayLike<number>,
): number | null {
  if (curve.sigmaRbfWeights == null || curve.sigmaPolynomial == null) return null;
  const embedded = embedCoreCurvePoint(curve, position);
  const normalized = Float64Array.from(embedded, (value, index) =>
    (value - curve.coordinateOffset[index]) / curve.coordinateScale[index]
  );
  let value = curve.sigmaPolynomial[0];
  for (let embed = 0; embed < normalized.length; embed += 1) {
    value += normalized[embed] * curve.sigmaPolynomial[embed + 1];
  }
  for (let node = 0; node < curve.nodeParameters.length; node += 1) {
    let squared = 0;
    for (let embed = 0; embed < normalized.length; embed += 1) {
      squared += (normalized[embed] - curve.nodeParameters[node][embed]) ** 2;
    }
    value += Math.sqrt(squared) ** 3 * curve.sigmaRbfWeights[node];
  }
  return Math.exp(value);
}

function embedCoreCurvePoint(
  curve: StructuredCurve,
  position: ArrayLike<number>,
): Float64Array {
  if (curve.domainKind === "sphere") {
    const output = new Float64Array(curve.axes.length + 1);
    let running = 1;
    curve.axes.forEach((_axis, index) => {
      output[index] = running * Math.cos(position[index]);
      running *= Math.sin(position[index]);
    });
    output[curve.axes.length] = running;
    return output;
  }
  const output: number[] = [];
  curve.axes.forEach((axis, index) => {
    if (axis.periodic) {
      const angle = 2 * Math.PI * position[index] / axis.period;
      output.push(Math.cos(angle), Math.sin(angle));
    } else {
      output.push(position[index]);
    }
  });
  return Float64Array.from(output);
}

function geometryLayerBandwidth(
  nodeWhite: readonly Float32Array[],
  neutralWhite: Float32Array,
  injectNeutral: boolean,
  curve: StructuredCurve | null,
  nodeCoordinates: readonly Float32Array[] | undefined,
  gram: Float64Array,
): number[] {
  let nodes: number[];
  if (
    curve !== null && nodeCoordinates !== undefined &&
    curve.sigmaRbfWeights != null && curve.sigmaPolynomial != null
  ) {
    let trace = 0;
    for (let axis = 0; axis < curve.basis.length; axis += 1) {
      trace += gram[axis * curve.basis.length + axis];
    }
    const scale = Math.sqrt(Math.max(trace / curve.basis.length, 1e-12));
    nodes = nodeCoordinates.map((coordinate) =>
      Math.max(evaluateCoreSigma(curve, coordinate) ?? 0, 0) * scale
    );
    return [...nodes, ...(injectNeutral ? [median(nodes)] : [])];
  }
  nodes = nodeWhite.map((row, rowIndex) => {
    if (nodeWhite.length < 2) return 1;
    let nearest = Number.POSITIVE_INFINITY;
    nodeWhite.forEach((other, otherIndex) => {
      if (otherIndex !== rowIndex) nearest = Math.min(nearest, rowDistance(row, other));
    });
    return nearest;
  });
  if (!injectNeutral) return nodes;
  return [
    ...nodes,
    Math.min(...nodeWhite.map((row) => rowDistance(row, neutralWhite))),
  ];
}

function affineCoordinateMap(
  reducedNodes: Float32Array,
  authorCoordinates: Float32Array,
  nodeCount: number,
  rank: number,
  intrinsicDim: number,
): { map: Float64Array[]; bias: Float64Array } {
  if (nodeCount === 1) {
    let denominator = 0;
    for (let axis = 0; axis < rank; axis += 1) denominator += reducedNodes[axis] ** 2;
    denominator = Math.max(denominator, 1e-8);
    return {
      map: Array.from({ length: intrinsicDim }, (_, coordinate) =>
        Float64Array.from(
          { length: rank },
          (_, axis) => authorCoordinates[coordinate] * reducedNodes[axis] / denominator,
        )
      ),
      bias: new Float64Array(intrinsicDim),
    };
  }
  const width = rank + 1;
  const normal = new Float64Array(width * width);
  const right = new Float64Array(width * intrinsicDim);
  for (let node = 0; node < nodeCount; node += 1) {
    for (let left = 0; left < width; left += 1) {
      const leftValue = left === rank ? 1 : reducedNodes[node * rank + left];
      for (let column = 0; column < width; column += 1) {
        const columnValue = column === rank ? 1 : reducedNodes[node * rank + column];
        normal[left * width + column] += leftValue * columnValue;
      }
      for (let coordinate = 0; coordinate < intrinsicDim; coordinate += 1) {
        right[left * intrinsicDim + coordinate] +=
          leftValue * authorCoordinates[node * intrinsicDim + coordinate];
      }
    }
  }
  const diagonalMean = Array.from({ length: width }, (_, index) =>
    normal[index * width + index]
  ).reduce((sum, value) => sum + value, 0) / width;
  for (let axis = 0; axis < width; axis += 1) {
    normal[axis * width + axis] += Math.max(1e-12, diagonalMean * 1e-12);
  }
  const solutions = Array.from({ length: intrinsicDim }, (_, coordinate) =>
    solveDenseSystem(
      normal,
      Float64Array.from({ length: width }, (_, row) =>
        right[row * intrinsicDim + coordinate]
      ),
      width,
    )
  );
  return {
    map: solutions.map((solution) => solution.slice(0, rank)),
    bias: Float64Array.from(solutions, (solution) => solution[rank]),
  };
}

function resolveCurvePosition(
  manifold: CoreManifold,
  position: readonly number[] | string,
  axes: readonly StructuredCurveAxis[],
): Float32Array {
  let value: Float32Array;
  if (typeof position === "string") {
    const labelIndex = manifold.labels.indexOf(position);
    if (labelIndex < 0) {
      throw coreError(
        "CORE_PACK_SELECTOR_NOT_FOUND",
        `Core manifold ${manifold.key} has no node ${position}`,
      );
    }
    value = manifold.tensors.tensor(
      "node_coords",
      [manifold.labels.length, axes.length],
    ).data.slice(labelIndex * axes.length, (labelIndex + 1) * axes.length);
  } else {
    if (position.length !== axes.length || position.some((coordinate) => !Number.isFinite(coordinate))) {
      throw coreError(
        "CORE_PACK_CURVED_DIMENSION_UNAVAILABLE",
        `Curved steering requires ${axes.length} target coordinates`,
      );
    }
    value = Float32Array.from(position);
  }
  return Float32Array.from(value, (coordinate, axis) => {
    const spec = axes[axis];
    if (spec.periodic) return ((coordinate % spec.period) + spec.period) % spec.period;
    return Math.max(spec.lowerBound, Math.min(spec.upperBound, coordinate));
  });
}

function measurementProbe(
  name: string | undefined,
  instruments: BrowserInstrumentRegistry,
): StructuredMeasurementProbe | null {
  if (name === undefined) return null;
  if (name.startsWith("jlens/")) {
    const token = instruments.jlensToken(name.slice("jlens/".length));
    return {
      name,
      family: "lens",
      tokenId: token.tokenId,
    };
  }
  if (name.startsWith("sae/")) {
    const featureId = Number(name.slice("sae/".length));
    const feature = instruments.saeFeature(String(featureId));
    return {
      name,
      family: "sae",
      featureId,
      label: feature.label,
      maxAct: feature.maxAct,
    };
  }
  return { name, family: "geometry" };
}

function rows(
  values: Float32Array,
  count: number,
  width: number,
): Float32Array[] {
  return Array.from({ length: count }, (_, index) =>
    values.slice(index * width, (index + 1) * width)
  );
}

function rows64(
  values: ArrayLike<number>,
  count: number,
  width: number,
): Float64Array[] {
  return Array.from({ length: count }, (_, index) =>
    Float64Array.from({ length: width }, (_, column) => values[index * width + column])
  );
}

function flattenRows(values: readonly ArrayLike<number>[], width: number): Float32Array {
  const output = new Float32Array(values.length * width);
  values.forEach((row, index) => {
    if (row.length !== width) throw new TypeError("Matrix rows have inconsistent widths");
    output.set(row, index * width);
  });
  return output;
}

function manifoldIntrinsicDimension(manifold: CoreManifold): number {
  return domainIntrinsicDimension(manifold.domain, manifold.key);
}

function restrictedInverseCovariance(
  basis: Float32Array,
  rank: number,
  whitener: SerializedMahalanobisWhitener,
): Float64Array {
  const gram = new Float64Array(rank * rank);
  const inverseRows = Array.from({ length: rank }, (_, row) =>
    applySerializedInverse(
      whitener,
      basis.subarray(row * whitener.columns, (row + 1) * whitener.columns),
    )
  );
  for (let left = 0; left < rank; left += 1) {
    const leftRow = basis.subarray(
      left * whitener.columns,
      (left + 1) * whitener.columns,
    );
    for (let right = 0; right < rank; right += 1) {
      gram[left * rank + right] = dot(leftRow, inverseRows[right]);
    }
  }
  return gram;
}

function lowerCholesky(matrix: Float64Array, size: number): Float64Array {
  const output = new Float64Array(size * size);
  for (let row = 0; row < size; row += 1) {
    for (let column = 0; column <= row; column += 1) {
      let value = matrix[row * size + column];
      for (let index = 0; index < column; index += 1) {
        value -= output[row * size + index] * output[column * size + index];
      }
      if (row === column) {
        if (!(value > 1e-12)) {
          throw coreError("CORE_PACK_METRIC_INVALID", "Probe subspace metric is not positive definite");
        }
        output[row * size + column] = Math.sqrt(value);
      } else {
        output[row * size + column] = value / output[column * size + column];
      }
    }
  }
  return output;
}

function inversePositiveDefinite(cholesky: Float64Array, size: number): Float64Array {
  const output = new Float64Array(size * size);
  for (let column = 0; column < size; column += 1) {
    const forward = new Float64Array(size);
    for (let row = 0; row < size; row += 1) {
      let value = row === column ? 1 : 0;
      for (let prior = 0; prior < row; prior += 1) {
        value -= cholesky[row * size + prior] * forward[prior];
      }
      forward[row] = value / cholesky[row * size + row];
    }
    const solved = new Float64Array(size);
    for (let row = size - 1; row >= 0; row -= 1) {
      let value = forward[row];
      for (let next = row + 1; next < size; next += 1) {
        value -= cholesky[next * size + row] * solved[next];
      }
      solved[row] = value / cholesky[row * size + row];
    }
    for (let row = 0; row < size; row += 1) output[row * size + column] = solved[row];
  }
  return output;
}

function solveDenseSystem(
  matrix: ArrayLike<number>,
  right: ArrayLike<number>,
  size: number,
): Float64Array {
  const rows = Array.from({ length: size }, (_, row) => [
    ...Array.from({ length: size }, (_, column) => matrix[row * size + column]),
    right[row],
  ]);
  for (let column = 0; column < size; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < size; row += 1) {
      if (Math.abs(rows[row][column]) > Math.abs(rows[pivot][column])) pivot = row;
    }
    [rows[column], rows[pivot]] = [rows[pivot], rows[column]];
    const divisor = rows[column][column];
    if (Math.abs(divisor) <= 1e-15) continue;
    for (let value = column; value <= size; value += 1) rows[column][value] /= divisor;
    for (let row = 0; row < size; row += 1) {
      if (row === column) continue;
      const factor = rows[row][column];
      for (let value = column; value <= size; value += 1) {
        rows[row][value] -= factor * rows[column][value];
      }
    }
  }
  return Float64Array.from(rows, (row) => row[size]);
}

function rowDistance(left: ArrayLike<number>, right: ArrayLike<number>): number {
  let squared = 0;
  for (let index = 0; index < left.length; index += 1) {
    squared += (left[index] - right[index]) ** 2;
  }
  return Math.sqrt(squared);
}

function median(values: readonly number[]): number {
  if (values.length === 0) return 1;
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.floor((sorted.length - 1) / 2)];
}

function multiplyRows(
  values: Float32Array,
  rowCount: number,
  width: number,
  matrix: Float64Array,
): Float32Array {
  const output = new Float32Array(rowCount * width);
  for (let row = 0; row < rowCount; row += 1) {
    for (let column = 0; column < width; column += 1) {
      let value = 0;
      for (let index = 0; index < width; index += 1) {
        value += values[row * width + index] * matrix[index * width + column];
      }
      output[row * width + column] = value;
    }
  }
  return output;
}

function maximumBasisOverlap(
  left: readonly ArrayLike<number>[],
  right: readonly ArrayLike<number>[],
): number {
  let maximum = 0;
  for (const leftRow of left) {
    for (const rightRow of right) {
      if (leftRow.length !== rightRow.length) {
        throw coreError("CORE_PACK_CURVED_SHAPE_INVALID", "Curved manifold bases have inconsistent widths");
      }
      maximum = Math.max(maximum, Math.abs(dot(leftRow, rightRow)));
    }
  }
  return maximum;
}

/** Native `_orthogonalize_affine_against`: the curved manifolds win the shared directions. The
 *  affine write rows and push displacement lose their curved-span component
 *  Mahalanobis-orthogonally (the curved frame's reads give the coefficients), and each new axis
 *  inherits its squared Mahalanobis projection onto the old axes' collapse coefficients. Null
 *  when the affine span lies inside the curved span. */
function orthogonalizeAffineGroup(
  group: StructuredAffineGroup,
  curvedFrame: CurvedFrame | undefined,
): StructuredAffineGroup | null {
  if (curvedFrame === undefined || curvedFrame.basis.length === 0) return group;
  const width = group.neutral.length;
  const oldBasis = group.basis.map((row) => Float64Array.from(row));
  const oldRead = group.read.map((row) => Float64Array.from(row));
  if (
    oldBasis.some((row) => row.length !== width) ||
    oldRead.length !== oldBasis.length || oldRead.some((row) => row.length !== width) ||
    curvedFrame.read.length !== curvedFrame.basis.length ||
    [...curvedFrame.basis, ...curvedFrame.read].some((row) => row.length !== width) ||
    group.target.length !== oldBasis.length || group.kappa.length !== oldBasis.length
  ) {
    throw coreError("CORE_PACK_CURVED_SHAPE_INVALID", "Affine and curved steering bases are incompatible");
  }
  const curve = orthoFrame(curvedFrame.basis, curvedFrame.read);
  // `x − (x · Cᵣᵀ) C_b`: the part of a vector (or, applied with the reads, a covector) that is
  // Mahalanobis-orthogonal to the curved span.
  const projectOut = (vector: Float64Array, pairs: readonly Float64Array[], coefficients: Float64Array) => {
    const output = Float64Array.from(vector);
    coefficients.forEach((coefficient, axis) => {
      for (let column = 0; column < width; column += 1) output[column] -= coefficient * pairs[axis][column];
    });
    return output;
  };
  const rowsOut: Float64Array[] = [];
  const readsOut: Float64Array[] = [];
  oldBasis.forEach((row, axis) => {
    const coefficients = Float64Array.from(curve.read, (read) => dot(row, read));
    rowsOut.push(projectOut(row, curve.basis, coefficients));
    readsOut.push(projectOut(oldRead[axis], curve.read, coefficients));
  });
  const { basis, read } = orthoFrame(rowsOut, readsOut);
  if (basis.length === 0) return null;
  const delta = new Float64Array(width);
  oldBasis.forEach((row, axis) => {
    for (let column = 0; column < width; column += 1) delta[column] += row[column] * group.target[axis];
  });
  const perpendicular = projectOut(delta, curve.basis, Float64Array.from(curve.read, (row) => dot(delta, row)));
  const target = Float64Array.from(read, (row) => dot(row, perpendicular));
  const kappa = Float64Array.from(read, (row) => {
    let inherited = 0;
    oldBasis.forEach((oldRow, axis) => {
      inherited += group.kappa[axis] * dot(row, oldRow) ** 2;
    });
    return inherited;
  });
  return { ...group, basis, read, target, kappa };
}

/** `Σ u` for the neutral whitener's regularized covariance `Σ = λI + V diag(e) Vᵀ` (native
 *  `LayerWhitener.apply`, the inverse of `applySerializedInverse`): the index-raising map from a
 *  read covector to the write vector that moves its reading furthest per unit of Mahalanobis
 *  displacement. */
export function raiseLensRead(
  whitener: SerializedMahalanobisWhitener,
  read: ArrayLike<number>,
): Float64Array {
  if (read.length !== whitener.columns) {
    throw new TypeError("Lens read does not match the whitener dimensions");
  }
  const output = Float64Array.from(read, (value) => value * whitener.ridge);
  for (let component = 0; component < whitener.rank; component += 1) {
    const offset = component * whitener.columns;
    let coordinate = 0;
    for (let column = 0; column < whitener.columns; column += 1) {
      coordinate += read[column] * whitener.basis[offset + column];
    }
    const scaled = coordinate * whitener.eigenvalues[component];
    for (let column = 0; column < whitener.columns; column += 1) {
      output[column] += scaled * whitener.basis[offset + column];
    }
  }
  return output;
}

/** Native `_ortho_frame`: ordered Gram-Schmidt of write `rows` in the metric their `reads`
 *  define (`⟨u, v⟩_M = u · read(v)`, the Mahalanobis product when each read is `Σ⁻¹` of its
 *  row). Every operation on a row is applied to its read, so the frame stays dual. Each row is
 *  unit-normalized first, and one whose residual falls below `eps` is dropped (collinear rows
 *  add no axis). Order matters: earlier rows keep their span. `reads = rows` is the Euclidean
 *  case. The reads are corrected to exact duality (`read · basisᵀ = I`, `dual_read`). */
function orthoFrame(
  rows: readonly ArrayLike<number>[],
  reads: readonly ArrayLike<number>[],
  eps = 1e-6,
): { basis: Float64Array[]; read: Float64Array[] } {
  const basis: Float64Array[] = [];
  const read: Float64Array[] = [];
  rows.forEach((source, index) => {
    const norm = dot(source, reads[index]);
    if (norm < 1e-24) return;
    const row = Float64Array.from(source, (value) => value / Math.sqrt(norm));
    const rowRead = Float64Array.from(reads[index], (value) => value / Math.sqrt(norm));
    basis.forEach((accepted, axis) => {
      const coefficient = dot(read[axis], row);
      for (let column = 0; column < row.length; column += 1) {
        row[column] -= coefficient * accepted[column];
        rowRead[column] -= coefficient * read[axis][column];
      }
    });
    const residual = dot(row, rowRead);
    if (residual < eps * eps) return;
    basis.push(row.map((value) => value / Math.sqrt(residual)));
    read.push(rowRead.map((value) => value / Math.sqrt(residual)));
  });
  return { basis, read: dualRead(basis, read) };
}

/** Native `dual_read`: `(read · basisᵀ)⁻¹ read`, the reads corrected to be exactly dual to the
 *  write rows, in their own span. */
function dualRead(basis: readonly Float64Array[], read: readonly Float64Array[]): Float64Array[] {
  const rank = basis.length;
  if (rank === 0) return [];
  const gram = new Float64Array(rank * rank);
  for (let left = 0; left < rank; left += 1) {
    for (let right = 0; right < rank; right += 1) gram[left * rank + right] = dot(read[left], basis[right]);
  }
  const inverse = Array.from({ length: rank }, (_, column) =>
    solveDenseSystem(gram, Float64Array.from({ length: rank }, (_, row) => row === column ? 1 : 0), rank)
  );
  return Array.from({ length: rank }, (_, row) => {
    const output = new Float64Array(read[0].length);
    for (let axis = 0; axis < rank; axis += 1) {
      const coefficient = inverse[axis][row];
      for (let column = 0; column < output.length; column += 1) output[column] += coefficient * read[axis][column];
    }
    return output;
  });
}

function dot(left: ArrayLike<number>, right: ArrayLike<number>): number {
  let value = 0;
  for (let index = 0; index < left.length; index += 1) value += left[index] * right[index];
  return value;
}

function roundSix(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}

function euclideanNorm(values: ArrayLike<number>): number {
  let squared = 0;
  for (let index = 0; index < values.length; index += 1) squared += values[index] ** 2;
  return Math.sqrt(squared);
}

function formatCoefficient(value: number): string {
  return Object.is(value, -0) ? "0" : String(value);
}

function formatManifoldPosition(position: readonly number[] | string): string {
  return typeof position === "string"
    ? position
    : position.map((coordinate) => formatCoefficient(coordinate)).join(",");
}

function formatAtom(atom: SteeringAtom): string {
  const qualified = atom.namespace === null ? atom.concept : `${atom.namespace}/${atom.concept}`;
  return atom.variant === "raw" ? qualified : `${qualified}:${atom.variant}`;
}

function coreError(code: string, message: string): Error & {
  code: string;
  status: number;
  recoverable: boolean;
} {
  return Object.assign(new Error(message), { code, status: 409, recoverable: true });
}
