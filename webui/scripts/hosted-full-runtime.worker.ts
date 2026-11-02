import {
  contextBindingSha256,
  runtimeIdentitySha256,
} from "../src/lib/runtime/catalog";
import { sha256 } from "@noble/hashes/sha2.js";
import { bytesToHex } from "@noble/hashes/utils.js";
import type {
  CatalogFile,
  CatalogInstrumentPack,
  ModelVariant,
  RuntimeIdentity,
  RuntimeProgressEvent,
} from "../../drowse/webui/src/lib/runtime/contracts";
import { BrowserDrowseArchiveService } from "../src/hosted/artifacts/service";
import { BrowserActivationSpool } from "../src/hosted/fitting/activationSpool";
import { BrowserCapabilityChecker } from "../src/hosted/runtime/capabilities";
import { BrowserFeasibilityCorePackCompiler } from "../src/hosted/runtime/browserCorePack";
import { BrowserInstrumentRuntime } from "../src/hosted/runtime/browserInstrumentRuntime";
import { BrowserManifoldFitting } from "../src/hosted/runtime/browserManifoldFitting";
import type {
  BrowserInstrumentPackArtifacts,
  BrowserModelArtifact,
  BrowserModelLoadRequest,
} from "../src/hosted/runtime/modelBackend";
import {
  DrowseWebLlmRuntime,
  type DrowseWebLlmModule,
} from "../src/hosted/runtime/webLlmEngine";
import type { StructuredHookProgramBuffers } from "../src/hosted/runtime/structuredHookProgram";
import {
  BLOCK_SITES_EXACT_READOUT_ABI,
  type BlockSpec,
} from "../src/hosted/runtime/structuredHookProfile";
import { packFileResidency } from "../src/hosted/runtime/optionalPackCompatibility";
import {
  rejectingBlockSitesBackend,
  UnsupportedBlockError,
  type BrowserBlockSitesBackend,
} from "../src/hosted/runtime/modelBackend";
import { webLlmBlockSitesBackend } from "../src/hosted/runtime/webLlmBlockSites";
import { geometryGateScores } from "../src/hosted/runtime/webLlmGeneration";
import {
  BrowserUnitsService,
  type UnitsNodeReplayPort,
} from "../src/hosted/runtime/units/unitsService";
import { sTokenResolver } from "../src/hosted/runtime/units/packSource";
import { UNIT_MEAN_FOOTPRINT_Z } from "../src/hosted/runtime/units/replayProbes";
import { validatePairedSteeringControls } from "./browser-full-runtime-contract.mjs";

type FileEntry = CatalogFile;

interface RunRequest {
  type: "run";
  modelId: string;
  runtimeIdentity: RuntimeIdentity;
  structuredHookProfile: ModelVariant["structuredHookProfile"];
  /** The runtime lock's block spec (standard-v4) and unit refusal (standard-v4-residual). */
  blockSpec: BlockSpec | null;
  unitRefusal: string | null;
  prefillChunkSize: number;
  thinkingProfile: ModelVariant["thinkingProfile"];
  quantization: "q4f16_1" | "q4f32_1" | "q0f32";
  requiredFeatures: GPUFeatureName[];
  contextTokens: number;
  contextProfiles: number[];
  modelFiles: FileEntry[];
  coreFiles: FileEntry[];
  /** Each optional pack's closure; empty when the run has no such pack. A standard-v4 library
   *  runs with its R (`jlens`), S (`slens`) and units packs, a residual one with no S or units
   *  pack; SAE is optional on both. */
  jlensFiles: FileEntry[];
  slensFiles: FileEntry[];
  unitsFiles: FileEntry[];
  saeFiles: FileEntry[];
  maxTokens: number;
  stopMaxTokens: number;
  fitLayer: number | null;
  jlensWord: string;
  slensWord: string;
  saeFeature: number;
}

type PackKind = "jlens" | "slens" | "units" | "sae";
const PACK_KINDS: readonly PackKind[] = ["jlens", "slens", "units", "sae"];
/** The prompt ordinary generation runs; the units check replays its first generated token. */
const ORDINARY_PROMPT = "The sky looks blue because";
/** The loom node the harness's replay port answers for (the ordinary generation). */
const HARNESS_NODE = "browser-e2e-ordinary";
/** The smallest next-token logprob change that counts as an effect (the paired controls'). */
const LOGPROB_EFFECT = 1e-4;

type CheckName =
  | "webgpu_adapter"
  | "load"
  | "ordinary_generation"
  | "forced_replay"
  | "explicit_capture"
  | "flat_gated_instruments"
  | "curved_fitting"
  | "curved_generation"
  | "units"
  | "unit_refusal"
  | "stop"
  | "unload_reload"
  | "offline_reuse";

const REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE = 10;
const REQUIRED_MAX_COMPUTE_WORKGROUP_SIZE_X = 256;
const REQUIRED_MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP = 256;
const nativeFetch = globalThis.fetch.bind(globalThis);
let offline = false;
const offlineNetworkAttempts: string[] = [];
const externalNetworkAttempts: string[] = [];
globalThis.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
  const url =
    typeof input === "string"
      ? input
      : input instanceof URL
        ? input.href
        : input.url;
  const resolved = new URL(url, globalThis.location.href);
  if (resolved.protocol === "http:" || resolved.protocol === "https:") {
    if (offline) {
      offlineNetworkAttempts.push(resolved.href);
      return Promise.reject(
        new TypeError(`offline E2E blocked network request ${resolved.href}`),
      );
    }
    if (resolved.origin !== globalThis.location.origin) {
      externalNetworkAttempts.push(resolved.href);
      return Promise.reject(
        new TypeError(
          `real-browser E2E blocked external request ${resolved.href}`,
        ),
      );
    }
  }
  return nativeFetch(input, init);
}) as typeof globalThis.fetch;

self.onmessage = async (event: MessageEvent<RunRequest>) => {
  if (event.data?.type !== "run") return;
  let currentCheck: CheckName | "start" = "start";
  let runtime: DrowseWebLlmRuntime | null = null;
  let artifacts: BrowserDrowseArchiveService | null = null;
  const harnessManifold = {
    namespace: "local",
    name: "browser_e2e_curve",
  } as const;
  try {
    if (
      event.data.structuredHookProfile !== "standard-v4" &&
      event.data.structuredHookProfile !== "standard-v4-residual"
    ) {
      throw new Error(
        `real-browser instruments require a v5 library (standard-v4 or standard-v4-residual, ${BLOCK_SITES_EXACT_READOUT_ABI})`,
      );
    }
    const webLlmUrl = "/webllm.js";
    const webllm = (await import(
      /* @vite-ignore */ webLlmUrl
    )) as unknown as DrowseWebLlmModule;
    const fingerprint = runtimeIdentitySha256(event.data.runtimeIdentity);
    const contextBinding = contextBindingSha256(
      fingerprint,
      event.data.contextTokens,
    );
    const checker = new BrowserCapabilityChecker();
    let capabilities: Awaited<ReturnType<BrowserCapabilityChecker["check"]>>;
    await check("webgpu_adapter", async () => {
      currentCheck = "webgpu_adapter";
      capabilities = await checker.check();
      if (!capabilities.supported) {
        const failures = capabilities.issues
          .filter((issue) => issue.severity === "hard")
          .map((issue) => `${issue.code}: ${issue.message}`);
        throw new Error(`browser compatibility failed: ${failures.join("; ")}`);
      }
      if (capabilities.webGpu.fallback !== "hardware") {
        throw new Error("browser did not confirm a hardware WebGPU adapter");
      }
      const storageBindings =
        capabilities.webGpu.limits.maxStorageBuffersPerShaderStage;
      if (
        typeof storageBindings !== "number" ||
        storageBindings < REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE
      ) {
        throw new Error(
          `browser exposes ${storageBindings ?? "unknown"} storage buffers per shader stage; ` +
            `${REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE} are required`,
        );
      }
      const workgroupSizeX =
        capabilities.webGpu.limits.maxComputeWorkgroupSizeX;
      const workgroupInvocations =
        capabilities.webGpu.limits.maxComputeInvocationsPerWorkgroup;
      if (
        typeof workgroupSizeX !== "number" ||
        workgroupSizeX < REQUIRED_MAX_COMPUTE_WORKGROUP_SIZE_X ||
        typeof workgroupInvocations !== "number" ||
        workgroupInvocations < REQUIRED_MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP
      ) {
        throw new Error(
          "browser cannot provide the exact readout workgroup limits " +
            `(sizeX=${workgroupSizeX ?? "unknown"}, ` +
            `invocations=${workgroupInvocations ?? "unknown"})`,
        );
      }
      return {
        fallback: capabilities.webGpu.fallback,
        featureCount: capabilities.webGpu.features.length,
        adapterFeatures: [...capabilities.webGpu.features].sort(),
        adapterLimits: Object.fromEntries(
          Object.entries(capabilities.webGpu.limits)
            .filter(([, value]) => typeof value === "number" && Number.isFinite(value))
            .sort(([left], [right]) => left.localeCompare(right)),
        ),
        maxStorageBufferBindingSize:
          capabilities.webGpu.limits.maxStorageBufferBindingSize ?? null,
        maxStorageBuffersPerShaderStage:
          capabilities.webGpu.limits.maxStorageBuffersPerShaderStage ?? null,
        requiredMaxStorageBuffersPerShaderStage:
          REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE,
        maxComputeWorkgroupSizeX: workgroupSizeX,
        maxComputeInvocationsPerWorkgroup: workgroupInvocations,
        requiredMaxComputeWorkgroupSizeX: REQUIRED_MAX_COMPUTE_WORKGROUP_SIZE_X,
        requiredMaxComputeInvocationsPerWorkgroup:
          REQUIRED_MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP,
        calibrationScore: capabilities.signals.calibrationScore,
        crossOriginIsolated: capabilities.crossOriginIsolated,
      };
    });
    currentCheck = "load";
    const unitsProfile = event.data.structuredHookProfile === "standard-v4";
    const packFiles: Record<PackKind, FileEntry[]> = {
      jlens: event.data.jlensFiles,
      slens: event.data.slensFiles,
      units: event.data.unitsFiles,
      sae: event.data.saeFiles,
    };
    const has = (kind: PackKind) => packFiles[kind].length > 0;
    if (unitsProfile ? !(has("jlens") && has("slens") && has("units")) : has("slens") || has("units")) {
      throw new Error(
        unitsProfile
          ? "a standard-v4 library runs with its R, S and units packs"
          : "a standard-v4-residual library has no S lens or units pack",
      );
    }
    const presentKinds = PACK_KINDS.filter(has);
    const modelArtifacts = await stage("model", event.data.modelFiles);
    const coreArtifacts = await stage("core", event.data.coreFiles);
    const packArtifacts = new Map<PackKind, BrowserModelArtifact[]>();
    for (const kind of presentKinds) packArtifacts.set(kind, await stage(kind, packFiles[kind]));
    const artifactVerification = {
      schemaVersion: 1,
      sets: {
        model: artifactSetReceipt("model", event.data.modelFiles),
        core: artifactSetReceipt("core", event.data.coreFiles),
        ...Object.fromEntries(presentKinds.map((kind) => [kind, artifactSetReceipt(kind, packFiles[kind])])),
      },
    };
    const variant: ModelVariant = {
      id: `${event.data.modelId}-${event.data.quantization}-e2e`,
      structuredHookProfile: event.data.structuredHookProfile,
      thinkingProfile: event.data.thinkingProfile,
      runtimeIdentity: event.data.runtimeIdentity,
      runtimeIdentitySha256: fingerprint,
      contextProfiles: event.data.contextProfiles.map((contextTokens) => ({
        contextTokens,
        bindingSha256: contextBindingSha256(fingerprint, contextTokens),
      })),
      requirements: {
        features: event.data.requiredFeatures,
        limits: {
          maxStorageBufferBindingSize: 134_217_728,
          maxComputeWorkgroupStorageSize: 32_768,
          maxComputeWorkgroupSizeX: REQUIRED_MAX_COMPUTE_WORKGROUP_SIZE_X,
          maxComputeInvocationsPerWorkgroup:
            REQUIRED_MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP,
          maxStorageBuffersPerShaderStage:
            REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE,
        },
      },
    } as unknown as ModelVariant;
    const requiredCorePack: CatalogInstrumentPack = {
      id: `${event.data.modelId}-core-e2e`,
      kind: "core",
      displayName: "Local E2E core pack",
      runtimeIdentitySha256: fingerprint,
      files: coreArtifacts.map(({ manifest }) => manifest),
    } as CatalogInstrumentPack;
    // The app's split (worker.ts): a pack installs its install files and fetches each on-demand
    // layer on the first read that needs it. Every file is staged and verified up front; the
    // on-demand ones reach the runtime only through `fetchFile`, counted here.
    const onDemandFetches = new Set<string>();
    const optionalPacks: BrowserInstrumentPackArtifacts[] = presentKinds.map((kind) =>
      optionalPack(kind, event.data.modelId, fingerprint, variant, packArtifacts.get(kind)!, onDemandFetches),
    );
    const loadRequest = async (): Promise<BrowserModelLoadRequest> => ({
      model: {
        id: event.data.modelId,
      } as unknown as BrowserModelLoadRequest["model"],
      variant,
      requiredCorePack,
      contextTokens: event.data.contextTokens,
      blockSpec: event.data.blockSpec,
      unitRefusal: event.data.unitRefusal,
      prefillChunkSize: event.data.prefillChunkSize,
      adapter: await checker.adapterForLoad(),
      artifacts: [...modelArtifacts, ...coreArtifacts],
      optionalPacks,
      activationSpool: new BrowserActivationSpool(),
      signal: new AbortController().signal,
      onDeviceLost(failure) {
        deviceLosses.push(failure);
      },
    });
    const deviceLosses: Array<{
      code: string;
      message: string;
      confirmedOom: boolean;
    }> = [];
    let request = await loadRequest();
    const instrumentReleaseAssertions = await verifyInstrumentRejections(request);
    runtime = new DrowseWebLlmRuntime(webllm);
    let compiler: BrowserFeasibilityCorePackCompiler;
    let instruments: BrowserInstrumentRuntime;
    let saeDictionary: ReturnType<
      BrowserFeasibilityCorePackCompiler["saeGpuDictionary"]
    > = null;
    let jlensDictionary: ReturnType<
      BrowserFeasibilityCorePackCompiler["jlensGpuDictionary"]
    > = null;
    // The units members over the loaded library, wired as the app's backend wires them
    // (drowseWebLlmBackend.ts `configureUnits`), with a replay port over the ordinary
    // generation in place of the loom's.
    const unitsAbort = new AbortController();
    let replayPort: UnitsNodeReplayPort | null = null;
    const configureUnits = async (
      target: BrowserInstrumentRuntime,
      loaded: DrowseWebLlmRuntime,
    ): Promise<void> => {
      const packs = target.unitsPackSource(event.data.modelId, unitsAbort.signal);
      await target.configureUnits(new BrowserUnitsService({
        modelId: event.data.modelId,
        blockSites: () => blockSitesFor(loaded),
        packs: () => packs,
        tokenTexts: (tokenIds) => loaded.tokenTexts(tokenIds),
        resolveToken: sTokenResolver(loaded, () => blockSitesFor(loaded).capabilities()?.vocabSize ?? 0),
        resolveSelector: (selector) => target.selectorDirections(selector),
        replay: () => replayPort,
        geometryGateScores,
      }));
    };
    const loadDictionaries = async (
      source: BrowserFeasibilityCorePackCompiler,
      loaded: DrowseWebLlmRuntime,
      lifecycle: string,
    ) => {
      const sae = source.saeGpuDictionary();
      const jlens = source.jlensGpuDictionary();
      if (has("sae") !== (sae !== null)) {
        throw new Error(`${lifecycle}: the SAE GPU dictionary does not follow the staged SAE pack`);
      }
      if (has("jlens") !== (jlens !== null)) {
        throw new Error(`${lifecycle}: the R-lens GPU dictionary does not follow the staged R pack`);
      }
      if (sae !== null) await loaded.setSaeDictionary(sae);
      if (jlens !== null) await loaded.setJlensDictionary(jlens);
      return { sae, jlens };
    };
    await check("load", async () => {
      currentCheck = "load";
      await runtime!.load(request);
      compiler = await BrowserFeasibilityCorePackCompiler.load(request);
      ({ sae: saeDictionary, jlens: jlensDictionary } = await loadDictionaries(compiler, runtime!, "load"));
      instruments = new BrowserInstrumentRuntime(
        compiler,
        runtime!,
        contextBinding,
      );
      await configureUnits(instruments, runtime!);
      const hooks = runtime!.hookCapabilities();
      for (const capability of [
        "curved_manifold",
        "phase_trigger",
        "probe_gate",
      ] as const) {
        if (!hooks[capability])
          throw new Error(`loaded runtime lacks ${capability}`);
      }
      for (const [capability, expected] of [
        ["sae", has("sae")],
        ["jlens", has("jlens")],
        ["units", unitsProfile],
        ["slens", unitsProfile],
      ] as const) {
        if (hooks[capability] !== expected) {
          throw new Error(`loaded runtime ${expected ? "lacks" : "claims"} ${capability}`);
        }
      }
      const runtimeCapabilities = runtime!.runtimeCapabilities();
      if (!runtimeCapabilities.tokenizer)
        throw new Error("loaded runtime lacks its exact tokenizer port");
      // The adapter attested the library's descriptor against the lock's profile and block spec
      // at load (a mismatch refuses with HOOK_ABI_MISMATCH); the surface it exposes is the lock's.
      const sites = runtime!.blockSitesCapabilities();
      if (sites === null) throw new Error("the loaded library attests no hook ABI v5 block sites");
      if (
        sites.profile.unitSites !== (unitsProfile ? 1 : 0) ||
        JSON.stringify(sites.blockSpec) !== JSON.stringify(unitsProfile ? event.data.blockSpec : null) ||
        sites.unitRefusal !== (unitsProfile ? null : event.data.unitRefusal) ||
        sites.layerCount !== event.data.runtimeIdentity.layerMap.length ||
        sites.hiddenSize !== event.data.runtimeIdentity.hiddenSize
      ) {
        throw new Error("the loaded library's block sites do not match the runtime lock");
      }
      assertNoDeviceLoss(deviceLosses);
      return {
        runtimeIdentitySha256: fingerprint,
        layers: event.data.runtimeIdentity.layerMap.length,
        hiddenSize: event.data.runtimeIdentity.hiddenSize,
        multiTerm: hooks.multi_term,
        curved: hooks.curved_manifold,
        sae: hooks.sae,
        jlens: hooks.jlens,
        units: hooks.units,
        slens: hooks.slens,
        tokenizer: runtimeCapabilities.tokenizer,
        exactReadoutProfile: event.data.structuredHookProfile,
        unitSites: sites.profile.unitSites,
        maxPatternLayers: sites.profile.maxPatternLayers,
        blockSpec: sites.blockSpec === null ? null : { ...sites.blockSpec },
        unitRefusal: sites.unitRefusal,
        vocabSize: sites.vocabSize,
        contextWindowTokens: sites.contextTokens,
        prefillChunkSize: sites.prefillChunkSize,
        packs: [...presentKinds],
        saeDictionaryFeatureCount: saeDictionary?.featureCount ?? null,
        saeDictionaryRuntimeLayerIndex: saeDictionary?.runtimeLayerIndex ?? null,
        stagedArtifactFiles:
          modelArtifacts.length +
          coreArtifacts.length +
          presentKinds.reduce((sum, kind) => sum + packFiles[kind].length, 0),
        stagedArtifactBytes: [
          ...event.data.modelFiles,
          ...event.data.coreFiles,
          ...presentKinds.flatMap((kind) => packFiles[kind]),
        ].reduce((sum, file) => sum + file.bytes, 0),
        allOpfsHashesVerified: true,
      };
    });

    let ordinaryRawTokens: number[] = [];
    await check("ordinary_generation", async () => {
      currentCheck = "ordinary_generation";
      const rawTokens: number[] = [];
      const result = await runtime!.streamGeneration(
        {
          input: { kind: "raw", prompt: "The sky looks blue because" },
          sampling: { ...deterministicSampling(event.data.maxTokens), return_top_k: 8 },
          thinking: false,
          steeringExpression: null,
          hookProgram: null,
          onRawToken(token) {
            if (token.tokenId !== null) rawTokens.push(token.tokenId);
          },
        },
        () => undefined,
      );
      if (result.tokens < 1 || rawTokens.length !== result.tokens) {
        throw new Error(
          "ordinary generation returned no exact raw-token stream",
        );
      }
      ordinaryRawTokens = [...rawTokens];
      if (result.measurements !== undefined) {
        throw new Error(
          "ordinary generation unexpectedly returned Drowse measurements",
        );
      }
      if (
        result.prefillTokensPerSecond === null ||
        result.prefillTokensPerSecond <= 0 ||
        result.decodeTokensPerSecond === null ||
        result.decodeTokensPerSecond <= 0
      ) {
        throw new Error(
          "ordinary generation returned no measured prefill/decode speed",
        );
      }
      const chatChecks: Array<{ prompt: string; response: string; passed: boolean; promptTokens: number; tokenIds: number[]; finishReason: string }> = [];
      if (runtime!.runtimeCapabilities().baseModel !== true) {
        for (const [prompt, expected] of [
          ["What is 2 + 2? Answer with only the number.", /^\s*4[.!]?\s*$/u],
          ["What is the capital of France? Answer with only the city.", /^\s*Paris[.!]?\s*$/iu],
        ] as const) {
          const tokenIds: number[] = [];
          const answer = await runtime!.streamGeneration({
            input: { kind: "chat", messages: [{ role: "user", content: prompt }] },
            sampling: { ...deterministicSampling(24), return_top_k: 8 },
            thinking: false,
            steeringExpression: null,
            hookProgram: null,
            onRawToken(token) { if (token.tokenId !== null) tokenIds.push(token.tokenId); },
          }, () => undefined);
          chatChecks.push({ prompt, response: answer.text, passed: expected.test(answer.text), promptTokens: answer.usage.promptTokens, tokenIds, finishReason: answer.finishReason });
        }
        if (chatChecks.some(({ passed }) => !passed)) {
          throw new Error(`unsteered chat sanity checks failed: ${JSON.stringify(chatChecks)}`);
        }
      }
      return {
        tokens: result.tokens,
        text: result.text,
        chatChecks,
        finishReason: result.finishReason,
        prefillTokensPerSecond: result.prefillTokensPerSecond,
        decodeTokensPerSecond: result.decodeTokensPerSecond,
      };
    });

    await check("forced_replay", async () => {
      currentCheck = "forced_replay";
      const runtimeCapabilities = runtime!.runtimeCapabilities();
      if (!runtimeCapabilities.forcedReplay || !runtimeCapabilities.replayScoring) {
        throw new Error(
          "loaded runtime lacks exact forced-token replay or compact replay scoring",
        );
      }
      const forcedPrefix = ordinaryRawTokens.slice(
        0,
        Math.min(2, ordinaryRawTokens.length),
      );
      if (forcedPrefix.length === 0) {
        throw new Error("ordinary generation left no token prefix to replay");
      }
      const requestedTokenIds = [...new Set(ordinaryRawTokens.slice(0, 4))];
      const replayed: Array<{
        tokenId: number | null;
        replayScore?: {
          emittedTokenId: number;
          sampledTokenId: number;
          forcedTokenId: number | null;
          requestedLogprobs: Array<{ tokenId: number; logprob: number }>;
          argmax: { tokenId: number; logprob: number };
          topLogprobs: Array<{ tokenId: number; logprob: number }>;
        };
      }> = [];
      const result = await runtime!.streamGeneration(
        {
          input: { kind: "raw", prompt: "The sky looks blue because" },
          sampling: {
            ...deterministicSampling(forcedPrefix.length),
            temperature: 1,
            return_top_k: 5,
          },
          thinking: false,
          replay: {
            forcedPrefixTokenIds: forcedPrefix,
            scoreTokenIds: requestedTokenIds,
          },
          steeringExpression: null,
          hookProgram: null,
          onRawToken(token) {
            replayed.push({
              tokenId: token.tokenId,
              ...(token.replayScore === undefined
                ? {}
                : { replayScore: token.replayScore }),
            });
          },
        },
        () => undefined,
      );
      if (result.tokens !== forcedPrefix.length || replayed.length !== forcedPrefix.length) {
        throw new Error("forced replay returned an unexpected token count");
      }
      for (let index = 0; index < replayed.length; index += 1) {
        const row = replayed[index];
        const score = row.replayScore;
        const missingRequested = score === undefined
          ? requestedTokenIds
          : requestedTokenIds.filter(
              (tokenId) =>
                !score.requestedLogprobs.some(
                  (candidate) =>
                    candidate.tokenId === tokenId &&
                    (Number.isFinite(candidate.logprob) ||
                      candidate.logprob === Number.NEGATIVE_INFINITY),
                ),
            );
        if (
          row.tokenId !== forcedPrefix[index] ||
          score === undefined ||
          score.emittedTokenId !== forcedPrefix[index] ||
          score.forcedTokenId !== forcedPrefix[index] ||
          score.topLogprobs.length < 2 ||
          missingRequested.length > 0
        ) {
          throw new Error(
            `forced replay row ${index} did not preserve its token and exact score envelope: ` +
              JSON.stringify({
                expectedTokenId: forcedPrefix[index],
                actualTokenId: row.tokenId,
                score,
                missingRequested,
              }),
          );
        }
      }
      const firstScore = replayed[0].replayScore!;
      const alternative = firstScore.topLogprobs.find(
        (candidate) => candidate.tokenId !== forcedPrefix[0],
      );
      if (alternative === undefined) {
        throw new Error("forced replay exposed no alternate token for a physical fork");
      }
      let forkedToken: (typeof replayed)[number] | null = null;
      const forked = await runtime!.streamGeneration(
        {
          input: { kind: "raw", prompt: "The sky looks blue because" },
          sampling: {
            ...deterministicSampling(1),
            temperature: 1,
            return_top_k: 5,
          },
          thinking: false,
          replay: {
            forcedPrefixTokenIds: [alternative.tokenId],
            scoreTokenIds: [forcedPrefix[0], alternative.tokenId],
          },
          steeringExpression: null,
          hookProgram: null,
          onRawToken(token) {
            forkedToken = {
              tokenId: token.tokenId,
              ...(token.replayScore === undefined
                ? {}
                : { replayScore: token.replayScore }),
            };
          },
        },
        () => undefined,
      );
      // Assigned in the callback, so TypeScript narrows the binding to its initializer.
      const forkedRow = forkedToken as (typeof replayed)[number] | null;
      if (
        forked.tokens !== 1 ||
        forkedRow?.tokenId !== alternative.tokenId ||
        forkedRow.replayScore?.forcedTokenId !== alternative.tokenId
      ) {
        throw new Error("alternate-token fork did not follow the exact forced token");
      }
      return {
        replayedTokens: replayed.length,
        requestedScoresPerToken: requestedTokenIds.length,
        topScoresPerToken: firstScore.topLogprobs.length,
        originalFirstTokenId: forcedPrefix[0],
        alternateFirstTokenId: alternative.tokenId,
        alternateTokenLogprob: alternative.logprob,
        forkedTokens: forked.tokens,
      };
    });

    await check("explicit_capture", async () => {
      currentCheck = "explicit_capture";
      const prepared = await runtime!.prepareCaptureRows([
        {
          system: "",
          messages: [
            { role: "user", content: "Describe the weather." },
            { role: "assistant", content: "The day is calm and clear." },
          ],
        },
      ]);
      if (prepared.length !== 1)
        throw new Error("explicit capture row preparation failed");
      const capture = await runtime!.capturePreparedRow(prepared[0]);
      const expectedValues =
        event.data.runtimeIdentity.layerMap.length *
        event.data.runtimeIdentity.hiddenSize;
      if (
        capture.layerCount !== event.data.runtimeIdentity.layerMap.length ||
        capture.hiddenSize !== event.data.runtimeIdentity.hiddenSize ||
        capture.positionCount !== 1 ||
        capture.values.length !== expectedValues ||
        capture.values.some((value) => !Number.isFinite(value))
      ) {
        throw new Error(
          "explicit post-block capture returned an invalid tensor envelope",
        );
      }
      let absoluteMax = 0;
      for (const value of capture.values)
        absoluteMax = Math.max(absoluteMax, Math.abs(value));
      if (!(absoluteMax > 0))
        throw new Error("explicit post-block capture was identically zero");
      return {
        layers: capture.layerCount,
        positions: capture.positionCount,
        hiddenSize: capture.hiddenSize,
        values: capture.values.length,
        absoluteMax,
      };
    });

    let flatSelector = "";
    let flatLabel = "";
    await check("flat_gated_instruments", async () => {
      currentCheck = "flat_gated_instruments";
      const flat = await firstGeometry(instruments!, true, "browser_e2e_flat");
      flatSelector = flat.selector;
      flatLabel = flat.label;
      const controlLogprobs: Record<string, number> = {};
      const controlOutcomes: Array<{ name: string; tokens: number; finishReason: string; logprob: string }> = [];
      for (const [name, coefficient] of [["baseline", null], ["zero", 0], ["positive", 0.5], ["negative", -0.5]] as const) {
        const controlExpression = coefficient === null
          ? null : `${coefficient} ${flat.selector}%${flat.label}@both`;
        const control = await runtime!.streamGeneration({
          input: { kind: "raw", prompt: "The sky looks blue because" },
          sampling: { ...deterministicSampling(1), temperature: 1 },
          thinking: false,
          replay: { forcedPrefixTokenIds: ordinaryRawTokens.slice(0, 1), scoreTokenIds: ordinaryRawTokens.slice(0, 1) },
          steeringExpression: controlExpression,
          hookProgram: controlExpression === null ? null : compiler!.compile(controlExpression),
          onRawToken(token) {
            const score = token.replayScore?.requestedLogprobs.find(row => row.tokenId === ordinaryRawTokens[0]);
            if (score) controlLogprobs[name] = score.logprob;
          },
        }, () => undefined);
        controlOutcomes.push({ name, tokens: control.tokens, finishReason: control.finishReason, logprob: String(controlLogprobs[name]) });
      }
      const controlMeasurements = validatePairedSteeringControls(controlLogprobs);
      // Geometry always; the R lens and the SAE when their packs are staged.
      const families = gatedFamiliesFor(has("jlens"), has("sae"));
      await attachInstrumentProbes(instruments!, event.data, families, "");
      for (const family of families) {
        await setLive(instruments!, family, true);
      }
      const expression = [
        `0.05 ${flat.selector}%${flat.label}@generated&when:browser_e2e_flat>-1000000`,
        ...(has("jlens") ? [`0.01 jlens/${event.data.jlensWord}@generated&when:browser_e2e_jlens>=0`] : []),
        ...(has("sae") ? [`0.01 sae/${event.data.saeFeature}@generated&when:browser_e2e_sae>-1000000`] : []),
      ].join(" + ");
      await compiler!.prepareJlensSelectors(expression, runtime!);
      const program = requireStructured(compiler!.compile(expression));
      if (!program.controls)
        throw new Error("flat steering gate exposed no runtime controls");
      const gatedSlots = program.controls.affine.flatMap((control, index) =>
        control?.gate ? [index] : [],
      );
      if (gatedSlots.length === 0)
        throw new Error("flat steering gate did not lower into GPU controls");
      const geometryScoreKeys = new Set(
        program.measurementSchema?.geometryProbes?.flatMap(
          (probe) => probe?.scoreKeys ?? [],
        ) ?? [],
      );
      const gatedFamilies = new Set(
        program.controls.affine.flatMap((control) => {
          const gate = control?.gate;
          if (!gate) return [];
          const families = gate.slots.flatMap((slot) => {
            const probe = program.measurementSchema?.probes[slot.probe];
            return probe ? [probe.family] : [];
          });
          if (gate.scoreKey && geometryScoreKeys.has(gate.scoreKey)) {
            families.push("geometry");
          }
          return families;
        }),
      );
      for (const family of families) {
        if (!gatedFamilies.has(family)) {
          throw new Error(
            `${family} measurements did not lower into a physical GPU gate`,
          );
        }
      }
      if (
        jlensDictionary !== null &&
        (program.jLensBindingId !== jlensDictionary.bindingId ||
          !program.jLensLayerIndices?.length)
      ) {
        throw new Error("J-lens steering did not lower into the GPU program");
      }
      const engine = runtime!.requireEngine();
      const originalUpdate = engine.updateDrowseStructuredControls.bind(engine);
      const gateTransitions: number[][] = [];
      engine.updateDrowseStructuredControls = async (
        affineActive,
        curveActive,
        modelId,
      ) => {
        gateTransitions.push(gatedSlots.map((slot) => affineActive[slot]));
        return originalUpdate(affineActive, curveActive, modelId);
      };
      let measurements: any = null;
      try {
        const result = await runtime!.streamGeneration(
          {
            input: { kind: "raw", prompt: "A welcoming person says" },
            sampling: deterministicSampling(event.data.maxTokens),
            thinking: false,
            steeringExpression: expression,
            hookProgram: program,
            onRawToken(token) {
              measurements = token.measurements ?? measurements;
            },
          },
          () => undefined,
        );
        if (result.tokens < 2 || result.measurements?.scope !== "aggregate") {
          throw new Error(
            "flat gated generation did not return aggregate measurements",
          );
        }
      } finally {
        engine.updateDrowseStructuredControls = originalUpdate;
      }
      const flatReading = measurements?.scores?.browser_e2e_flat;
      const lensReading = has("jlens")
        ? measurements?.instruments?.lens?.readings?.browser_e2e_jlens?.value
        : null;
      const saeReading = has("sae")
        ? measurements?.instruments?.sae?.readings?.browser_e2e_sae?.value
        : null;
      for (const [name, value] of Object.entries({
        flatReading,
        ...(has("jlens") ? { lensReading } : {}),
        ...(has("sae") ? { saeReading } : {}),
      })) {
        if (typeof value !== "number" || !Number.isFinite(value)) {
          throw new Error(`${name} was not a finite physical GPU measurement`);
        }
      }
      const exactReadouts = requireExactPrecomputedReadouts(
        measurements,
        compiler!.instrumentDescriptor(),
        "initial generation",
        { jlens: has("jlens"), sae: has("sae") },
      );
      const activatedGateSlots = gatedSlots.filter((_, slotIndex) =>
        gateTransitions.some((values) => values[slotIndex] === 1),
      );
      if (activatedGateSlots.length !== gatedSlots.length) {
        throw new Error(
          "not every physical geometry, J-lens, and SAE gate activated after a measured token",
        );
      }
      return {
        flatSelector,
        controlTokenId: ordinaryRawTokens[0],
        ...controlMeasurements,
        controlOutcomes,
        gateUpdates: gateTransitions.length,
        gatedTerms: gatedSlots.length,
        activatedGatedTerms: activatedGateSlots.length,
        gatedFamilies: [...gatedFamilies].sort(),
        flatReading,
        lensReading,
        saeReading,
        ...instrumentReleaseAssertions,
        ...exactReadouts,
        jLensBytes: jlensDictionary?.matrices.reduce(
          (bytes, matrix) => bytes + matrix.byteLength,
          0,
        ) ?? null,
      };
    });

    let curveSelector = "";
    let curveLabel = "";
    // Named roles (nearest-node role selection, role-prefixed headers) need a chat template; a
    // base model fits and steers its curve on raw text.
    const baseModel = runtime!.runtimeCapabilities().baseModel === true;
    await check("curved_fitting", async () => {
      currentCheck = "curved_fitting";
      for (const family of gatedFamiliesFor(has("jlens"), has("sae"))) {
        await instruments!.request(
          { service: "probes", method: "detach", args: [probeName(family, "")] },
          () => null,
        );
        if (family !== "geometry") await setLive(instruments!, family, false);
      }
      artifacts = new BrowserDrowseArchiveService(undefined, undefined, "0.0.0");
      await artifacts.request(
        {
          service: "manifolds",
          method: "delete",
          args: [harnessManifold.namespace, harnessManifold.name],
        },
        () => undefined,
      );
      await artifacts.request(
        {
          service: "manifolds",
          method: "createDiscover",
          args: [
            {
              namespace: harnessManifold.namespace,
              name: harnessManifold.name,
              description: "Real-browser E2E curved fitting fixture",
              fit_mode: "spectral",
              hyperparams: {
                max_dim: 1,
                min_dim: 1,
                k_nn: 2,
                max_subspace_dim: 2,
                smoothing: "auto",
              },
              nodes: [
                {
                  label: "distant",
                  role: "distant_voice",
                  statements: [
                    "I keep my distance from everyone.",
                    "I avoid inviting anyone closer.",
                  ],
                },
                {
                  label: "reserved",
                  role: "reserved_voice",
                  statements: [
                    "I remain quiet around new people.",
                    "I respond politely but keep to myself.",
                  ],
                },
                {
                  label: "open",
                  role: "open_voice",
                  statements: [
                    "I greet people and listen with interest.",
                    "I am comfortable sharing this moment.",
                  ],
                },
                {
                  label: "warm",
                  role: "warm_voice",
                  statements: [
                    "I make everyone feel welcome and at home.",
                    "I speak with generous warmth and care.",
                  ],
                },
                {
                  label: "exuberant",
                  role: "exuberant_voice",
                  statements: [
                    "I celebrate meeting everyone with delight.",
                    "I am overflowing with joyful enthusiasm.",
                  ],
                },
              // A base model has no chat template to name a role in: its nodes are plain voices.
              ].map(({ role, ...node }) => (baseModel ? node : { ...node, role })),
            },
          ],
        },
        () => undefined,
      );
      const fitLayer =
        event.data.fitLayer ??
        event.data.runtimeIdentity.layerMap[
          Math.floor(event.data.runtimeIdentity.layerMap.length / 2)
        ];
      if (!event.data.runtimeIdentity.layerMap.includes(fitLayer)) {
        throw new Error(
          `requested fit layer ${fitLayer} is outside the compiled layer map`,
        );
      }
      const progress: RuntimeProgressEvent[] = [];
      const fitting = new BrowserManifoldFitting(artifacts, "0.0.0");
      const fitted = (await fitting.request({
        request: {
          service: "manifolds",
          method: "fit",
          args: [
            harnessManifold.namespace,
            harnessManifold.name,
            {
              layers: [fitLayer],
              fit_mode: "spectral",
              hyperparams: {
                max_dim: 1,
                min_dim: 1,
                k_nn: 2,
                max_subspace_dim: 2,
                smoothing: "auto",
              },
              force: true,
            },
          ],
        },
        loadRequest: request,
        runtime: runtime!,
        onProgress(event) {
          progress.push(event);
        },
        signal: new AbortController().signal,
      })) as { fitted_models?: unknown };
      if (
        !Array.isArray(fitted.fitted_models) ||
        fitted.fitted_models.length === 0
      ) {
        throw new Error("browser curved fit did not persist a fitted artifact");
      }
      const stages = [
        ...new Set(
          progress.flatMap((event) => {
            const data = event.data as { stage?: unknown };
            return typeof data?.stage === "string" ? [data.stage] : [];
          }),
        ),
      ];
      for (const stage of [
        "capturing",
        "committing",
        "pooling",
        "fitting",
        "topology",
        "surface",
      ]) {
        if (!stages.includes(stage))
          throw new Error(`browser curved fit omitted ${stage} progress`);
      }
      const archives = await installedArchives(artifacts);
      compiler = await BrowserFeasibilityCorePackCompiler.load(request, {
        installedManifoldArchives: archives,
      });
      instruments = new BrowserInstrumentRuntime(
        compiler,
        runtime!,
        contextBinding,
      );
      // A new compiler (the fitted manifold installed) takes the units members again, as the
      // app's manifold refresh does.
      await configureUnits(instruments, runtime!);
      const curve = await exactGeometry(
        instruments,
        `${harnessManifold.namespace}/${harnessManifold.name}`,
        false,
        "browser_e2e_curve",
      );
      curveSelector = curve.selector;
      curveLabel = curve.label;
      if (!Object.keys(curve.geometry.layers).includes(String(fitLayer))) {
        throw new Error(
          "installed curved manifold did not retain the selected physical fit layer",
        );
      }
      return {
        selector: curveSelector,
        fitLayer,
        captureRows: 10,
        progressStages: stages,
        fittedLayers: Object.keys(curve.geometry.layers).length,
        installedArchives: archives.length,
        opfsSpoolingObserved:
          stages.includes("committing") && stages.includes("pooling"),
      };
    });

    await check("curved_generation", async () => {
      currentCheck = "curved_generation";
      await instruments!.request(
        {
          service: "instruments",
          method: "setLive",
          args: ["geometry", { enabled: true }],
        },
        () => null,
      );
      const expression = `0.05 ${curveSelector}%${curveLabel}@generated`;
      const program = requireStructured(compiler!.compile(expression));
      if (program.activeRole !== (baseModel ? null : "distant_voice")) {
        throw new Error(
          `curved manifold selected the wrong nearest-node role ${String(program.activeRole)}`,
        );
      }
      if (!baseModel && !runtime!.runtimeCapabilities().namedRoles) {
        throw new Error("the physical model chat template does not support named roles");
      }
      if (!program.curveRank.some((rank) => rank > 0)) {
        throw new Error(
          "curved manifold did not lower into a physical GPU curve program",
        );
      }
      let measurements: any = null;
      const engine = runtime!.requireEngine();
      const originalReadGeometry =
        engine.readDrowseGeometryMeasurements.bind(engine);
      const geometryStride = program.profile.geometryOutputStride;
      const activeGeometrySlots = Array.from(
        program.geometryActive ?? [],
      ).flatMap((active, slot) => (active === 1 ? [slot] : []));
      const geometryReads: number[] = [];
      const activeGeometryRows: number[][][] = [];
      engine.readDrowseGeometryMeasurements = async (...args) => {
        const values = await originalReadGeometry(...args);
        geometryReads.push(values?.length ?? 0);
        activeGeometryRows.push(
          activeGeometrySlots.map((slot) =>
            values === undefined
              ? []
              : Array.from(
                  values.subarray(
                    slot * geometryStride,
                    (slot + 1) * geometryStride,
                  ),
                ),
          ),
        );
        return values;
      };
      let result;
      try {
        result = await runtime!.streamGeneration(
          {
            input: baseModel
              ? { kind: "raw", prompt: "The feeling slowly changes as" }
              : {
                  kind: "chat",
                  messages: [{
                    role: "user",
                    content: "The feeling slowly changes as",
                  }],
                },
            sampling: deterministicSampling(event.data.maxTokens),
            thinking: false,
            generationSeat: baseModel ? null : "assistant",
            generationRoleName: program.activeRole,
            replay: { forcedPrefixTokenIds: ordinaryRawTokens.slice(0, 2) },
            steeringExpression: expression,
            hookProgram: program,
            onRawToken(token) {
              measurements = token.measurements ?? measurements;
            },
          },
          () => undefined,
        );
      } finally {
        engine.readDrowseGeometryMeasurements = originalReadGeometry;
      }
      const finiteGeometryRows = measurements?.instruments?.geometry?.readings
        ?.browser_e2e_curve
        ? 1
        : 0;
      const reading = measurements?.scores?.browser_e2e_curve;
      if (
        result.tokens < 1 ||
        typeof reading !== "number" ||
        !Number.isFinite(reading)
      ) {
        throw new Error(
          "curved physical GPU generation returned no finite geometry reading " +
            `(tokens=${result.tokens}, reads=${geometryReads.join(",") || "none"}, ` +
            `active=${activeGeometrySlots.length}, finite=${finiteGeometryRows}, ` +
            `row=${JSON.stringify(activeGeometryRows.at(-1)?.[0] ?? [])})`,
        );
      }
      return {
        tokens: result.tokens,
        finishReason: result.finishReason,
        reading,
        forcedPrefixTokenIds: ordinaryRawTokens.slice(0, 2),
        activeCurveSlots: program.curveRank.filter((rank) => rank > 0).length,
        activeGeometrySlots: activeGeometrySlots.length,
        finiteGeometryRows,
        geometryReadElements: geometryReads,
        activeRole: program.activeRole,
        namedRoleSupported: !baseModel,
      };
    });

    // The block sites' own members (B1 §2-§5): an ablation, the unit taps, one anatomy replay,
    // one static read and the S readout on standard-v4; on standard-v4-residual every unit member
    // answers the lock's refusal.
    let topUnit: string | null = null;
    if (unitsProfile) {
      await check("units", async () => {
        currentCheck = "units";
        const promptIds = await runtime!.renderPromptTokenIds({
          input: { kind: "raw", prompt: ORDINARY_PROMPT },
        });
        replayPort = harnessReplayPort(promptIds, ordinaryRawTokens);
        const target = ordinaryRawTokens[0];
        const anatomyResult = (await instruments!.request(
          {
            service: "instruments",
            method: "tokenAnatomy",
            args: [HARNESS_NODE, 0, { k: 8, units: 8, patterns: 2 }],
          },
          () => null,
        )) as { measurements?: { scope?: string; anatomy?: any } };
        const anatomy = anatomyResult.measurements?.anatomy;
        const anatomyUnits: Array<{ unit: string; kind: string; contribution: number; activation: number | null }> =
          Array.isArray(anatomy?.units) ? anatomy.units : [];
        if (
          anatomyResult.measurements?.scope !== "replay" ||
          anatomy?.token?.token_id !== target ||
          !(anatomy.token.probability > 0 && anatomy.token.probability <= 1) ||
          anatomyUnits.length === 0 ||
          anatomyUnits.some((row) => !Number.isFinite(row.contribution)) ||
          !Array.isArray(anatomy.layers) ||
          anatomy.layers.length !== event.data.runtimeIdentity.layerMap.length ||
          !Array.isArray(anatomy.patterns) ||
          anatomy.patterns.length !== 2
        ) {
          throw new Error(`token anatomy replay returned an invalid record: ${JSON.stringify(anatomy)?.slice(0, 2000)}`);
        }
        // The largest contributor to the token, mean-ablated at its own site, must move the token's
        // next-token logprob.
        const ablated = [...anatomyUnits].sort(
          (left, right) => Math.abs(right.contribution) - Math.abs(left.contribution),
        )[0];
        topUnit = ablated.unit;
        const ablation = `!${ablated.unit}`;
        const ablationProgram = await compileBound(compiler!, runtime!, ablation);
        if (ablationProgram.unitMeta === undefined) {
          throw new Error("unit ablation did not lower into the program's unit section");
        }
        const baselineLogprob = await forcedTokenLogprob(runtime!, target, null, null, null);
        const ablatedLogprob = await forcedTokenLogprob(
          runtime!,
          target,
          ablation,
          ablationProgram,
          instruments!.liveInstruments(),
        );
        if (
          !Number.isFinite(baselineLogprob) ||
          !Number.isFinite(ablatedLogprob) ||
          Math.abs(ablatedLogprob - baselineLogprob) < LOGPROB_EFFECT
        ) {
          throw new Error(
            `ablating ${ablated.unit} did not move the next-token logprob ` +
              `(${baselineLogprob} -> ${ablatedLogprob})`,
          );
        }
        // One static read: the first neuron among the contributors (any layer's neuron otherwise).
        const neuron = anatomyUnits.find((row) => row.kind === "neuron")?.unit ??
          `neuron/L${Math.floor(event.data.runtimeIdentity.layerMap.length / 2)}n0`;
        const read = (await instruments!.request(
          { service: "instruments", method: "unitRead", args: [neuron, { k: 8 }] },
          () => null,
        )) as { unit?: string; scalars?: Record<string, number | null>; poles?: Array<{ writes?: unknown; fires_on?: unknown }> };
        if (
          read.unit !== neuron ||
          !(typeof read.scalars?.w_norm === "number" && read.scalars.w_norm > 0) ||
          !Array.isArray(read.poles) ||
          read.poles.length !== 2 ||
          read.poles.some((pole) => pole.writes === undefined || pole.fires_on === undefined)
        ) {
          throw new Error(`static unit read returned an invalid record: ${JSON.stringify(read)?.slice(0, 2000)}`);
        }
        // The unit tap and the S readout ride one generation: a units probe on the ablated unit and
        // an `slens/` probe, both families live.
        await instruments!.request(
          { service: "probes", method: "attach", args: [{ selector: ablated.unit, name: "browser_e2e_unit" }] },
          () => null,
        );
        await instruments!.request(
          { service: "probes", method: "attach", args: [{ selector: `slens/${event.data.slensWord}`, name: "browser_e2e_slens" }] },
          () => null,
        );
        await setLive(instruments!, "lens", true);
        await setLive(instruments!, "units", true);
        let tokenMeasurements: any = null;
        let aggregate: any = null;
        try {
          const program = await compileBound(compiler!, runtime!, "");
          const result = await runtime!.streamGeneration(
            {
              input: { kind: "raw", prompt: ORDINARY_PROMPT },
              sampling: deterministicSampling(2),
              thinking: false,
              steeringExpression: null,
              hookProgram: program,
              liveInstruments: instruments!.liveInstruments(),
              onRawToken(token) {
                tokenMeasurements = token.measurements ?? tokenMeasurements;
              },
            },
            () => undefined,
          );
          aggregate = result.measurements ?? null;
        } finally {
          for (const name of ["browser_e2e_unit", "browser_e2e_slens"]) {
            await instruments!.request({ service: "probes", method: "detach", args: [name] }, () => null);
          }
          await setLive(instruments!, "lens", false);
          await setLive(instruments!, "units", false);
        }
        const unitTap = tokenMeasurements?.instruments?.units?.readings?.browser_e2e_unit;
        const slensReading = tokenMeasurements?.instruments?.lens?.readings?.browser_e2e_slens;
        const aggregateTap = aggregate?.instruments?.units?.readings?.browser_e2e_unit;
        const aggregateSlens = aggregate?.instruments?.lens?.readings?.browser_e2e_slens;
        for (const [name, reading] of Object.entries({ unitTap, slensReading, aggregateTap, aggregateSlens })) {
          if (typeof reading?.value !== "number" || !Number.isFinite(reading.value)) {
            throw new Error(`${name} was not a finite physical GPU reading: ${JSON.stringify(reading)}`);
          }
        }
        if (slensReading.unit !== UNIT_MEAN_FOOTPRINT_Z) {
          throw new Error(`the S readout is not a ${UNIT_MEAN_FOOTPRINT_Z} reading`);
        }
        assertNoDeviceLoss(deviceLosses);
        return {
          anatomyTokenId: target,
          anatomyProbability: anatomy.token.probability,
          anatomyUnits: anatomyUnits.length,
          anatomyPatterns: anatomy.patterns.length,
          anatomyLayers: anatomy.layers.length,
          ablatedUnit: ablated.unit,
          ablatedContribution: ablated.contribution,
          baselineLogprob,
          ablatedLogprob,
          ablationLogprobDelta: ablatedLogprob - baselineLogprob,
          logprobEffectThreshold: LOGPROB_EFFECT,
          staticReadUnit: neuron,
          staticReadWriteNorm: read.scalars!.w_norm!,
          unitTap: unitTap.value,
          aggregateUnitTap: aggregateTap.value,
          slensWord: event.data.slensWord,
          slensReading: slensReading.value,
          aggregateSlensReading: aggregateSlens.value,
          slensUnit: slensReading.unit,
          onDemandPackFiles: onDemandFetches.size,
        };
      });
    } else {
      await check("unit_refusal", async () => {
        currentCheck = "unit_refusal";
        const refusal = event.data.unitRefusal;
        if (typeof refusal !== "string" || !refusal) {
          throw new Error("a standard-v4-residual run needs the lock's unitRefusal");
        }
        const refused: string[] = [];
        for (const [method, args] of [
          ["tokenAnatomy", [HARNESS_NODE, 0, {}]],
          ["unitActivity", [HARNESS_NODE, {}]],
          ["unitRead", ["neuron/L0n0", {}]],
          ["unitTop", [event.data.slensWord, {}]],
          ["unitFind", [{ selector: "neuron/L0n0" }]],
          ["unitLabels", [["neuron/L0n0"]]],
        ] as const) {
          await requireRefusal(
            async () => instruments!.request({ service: "instruments", method, args: [...args] }, () => null),
            refusal,
            `instruments.${method}`,
          );
          refused.push(method);
        }
        // Native answers the catalog status from disk on every model.
        const catalog = (await instruments!.request(
          { service: "instruments", method: "unitCatalog", args: [] },
          () => null,
        )) as { bound?: unknown };
        if (catalog?.bound !== false) {
          throw new Error(`unitCatalog did not answer an unbound status: ${JSON.stringify(catalog)}`);
        }
        await requireRefusal(async () => compiler!.compile("!neuron/L0n0"), refusal, "a unit ablation");
        await requireRefusal(
          async () => instruments!.request(
            { service: "probes", method: "attach", args: [{ selector: "neuron/L0n0", name: "browser_e2e_unit" }] },
            () => null,
          ),
          refusal,
          "a unit probe",
        );
        // The bridge attested at load that the library exports no unit function (HOOK_ABI_MISMATCH
        // otherwise); its unit surface refuses.
        let bridgeRefusal = "";
        try {
          const engine = runtime!.requireEngine();
          if (engine.drowseBlockSites === undefined) throw new Error("UNIT_SITES_UNAVAILABLE: no block-sites surface");
          await engine.drowseBlockSites({ op: "headGrams" });
        } catch (error) {
          bridgeRefusal = error instanceof Error ? error.message : String(error);
        }
        if (!bridgeRefusal.startsWith("UNIT_SITES_UNAVAILABLE")) {
          throw new Error(`the bridge's unit surface did not refuse: ${bridgeRefusal || "it answered"}`);
        }
        const hooks = runtime!.hookCapabilities();
        assertNoDeviceLoss(deviceLosses);
        return {
          unitRefusal: refusal,
          refusedMembers: refused,
          refusedStatus: 422,
          unitCatalogBound: false,
          compileRefused: true,
          probeRefused: true,
          bridgeRefusal: bridgeRefusal.split(":")[0],
          unitsHook: hooks.units,
          slensHook: hooks.slens,
        };
      });
    }

    await check("stop", async () => {
      currentCheck = "stop";
      const engine = runtime!.requireEngine();
      const originalInterrupt = engine.interruptGenerate.bind(engine);
      let interruptCalls = 0;
      engine.interruptGenerate = async () => {
        interruptCalls += 1;
        return originalInterrupt();
      };
      let rawTokens = 0;
      let result;
      try {
        result = await runtime!.streamGeneration(
          {
            input: {
              kind: "raw",
              prompt: "Count upward without stopping: 1, 2, 3,",
            },
            sampling: deterministicSampling(event.data.stopMaxTokens),
            thinking: false,
            steeringExpression: null,
            hookProgram: null,
            async onRawToken() {
              rawTokens += 1;
              if (rawTokens === 1) await runtime!.stop();
            },
          },
          () => undefined,
        );
      } finally {
        engine.interruptGenerate = originalInterrupt;
      }
      if (interruptCalls !== 1 || rawTokens !== 1 || result.tokens !== 1) {
        throw new Error(
          "generation stop did not interrupt an in-flight physical decode",
        );
      }
      return {
        interruptCalls,
        tokensBeforeSettled: result.tokens,
        requestedMaxTokens: event.data.stopMaxTokens,
        finishReason: result.finishReason,
      };
    });

    await check("unload_reload", async () => {
      currentCheck = "unload_reload";
      await runtime!.unload();
      let unloadedRejected = false;
      try {
        runtime!.requireEngine();
      } catch {
        unloadedRejected = true;
      }
      if (!unloadedRejected)
        throw new Error("runtime remained usable after unload");
      request = await loadRequest();
      await runtime!.load(request);
      const restored = await loadDictionaries(compiler!, runtime!, "runtime reload");
      // The reloaded library takes the units members again, as the app's load does.
      await configureUnits(instruments!, runtime!);
      const families = gatedFamiliesFor(has("jlens"), has("sae"));
      const flat = await exactGeometry(instruments!, flatSelector, true, "browser_e2e_flat_reload");
      await attachInstrumentProbes(instruments!, event.data, families, "_reload");
      if (unitsProfile) {
        await instruments!.request(
          { service: "probes", method: "attach", args: [{ selector: `slens/${event.data.slensWord}`, name: "browser_e2e_slens_reload" }] },
          () => null,
        );
      }
      for (const family of families) await setLive(instruments!, family, true);
      if (unitsProfile) await setLive(instruments!, "lens", true);
      const reloadExpression = [
        `0.01 ${flat.selector}%${flatLabel}@generated`,
        ...(has("jlens") ? [`0.01 jlens/${event.data.jlensWord}@generated`] : []),
        ...(has("sae") ? [`0.01 sae/${event.data.saeFeature}@generated`] : []),
      ].join(" + ");
      const reloadProgram = await compileBound(compiler!, runtime!, reloadExpression);
      let reloadMeasurements: any = null;
      const result = await runtime!.streamGeneration(
        {
          input: { kind: "raw", prompt: "Reloaded model:" },
          sampling: deterministicSampling(1),
          thinking: false,
          steeringExpression: reloadExpression,
          hookProgram: reloadProgram,
          liveInstruments: instruments!.liveInstruments(),
          onRawToken(token) {
            reloadMeasurements = token.measurements ?? reloadMeasurements;
          },
        },
        () => undefined,
      );
      if (result.tokens !== 1)
        throw new Error("reloaded model did not complete a one-token decode");
      const reloadReadings = familyReadings(reloadMeasurements, families, "_reload");
      const reloadSlensReading = unitsProfile
        ? reloadMeasurements?.instruments?.lens?.readings?.browser_e2e_slens_reload?.value
        : null;
      for (const [name, value] of Object.entries({
        ...reloadReadings,
        ...(unitsProfile ? { reloadSlensReading } : {}),
      })) {
        if (typeof value !== "number" || !Number.isFinite(value)) {
          throw new Error(`${name} was not restored after runtime reload`);
        }
      }
      const exactReadouts = requireExactPrecomputedReadouts(
        reloadMeasurements,
        compiler!.instrumentDescriptor(),
        "runtime reload",
        { jlens: has("jlens"), sae: has("sae") },
      );
      for (const name of [
        "browser_e2e_flat_reload",
        ...families.filter((family) => family !== "geometry").map((family) => probeName(family, "_reload")),
        ...(unitsProfile ? ["browser_e2e_slens_reload"] : []),
      ]) {
        await instruments!.request({ service: "probes", method: "detach", args: [name] }, () => null);
      }
      assertNoDeviceLoss(deviceLosses);
      return {
        unloadInvalidatedEngine: unloadedRejected,
        reloadTokens: result.tokens,
        restoredSaeDictionaryFeatureCount: restored.sae?.featureCount ?? null,
        reloadGeometryReading: reloadReadings.geometry,
        reloadLensReading: reloadReadings.lens ?? null,
        reloadSaeReading: reloadReadings.sae ?? null,
        reloadSlensReading,
        ...exactReadouts,
        deviceLosses: deviceLosses.length,
      };
    });

    await check("offline_reuse", async () => {
      currentCheck = "offline_reuse";
      await runtime!.unload();
      runtime = null;
      await artifacts!.close();
      artifacts = null;
      const cutoff = await fetch("/offline-arm", { method: "POST" });
      if (!cutoff.ok)
        throw new Error("failed to arm offline E2E network cutoff");
      await cutoff.arrayBuffer();
      offline = true;
      const offlineModel = await reopen("model", event.data.modelFiles);
      const offlineCore = await reopen("core", event.data.coreFiles);
      const offlinePacks = new Map<PackKind, BrowserModelArtifact[]>();
      for (const kind of presentKinds) offlinePacks.set(kind, await reopen(kind, packFiles[kind]));
      artifacts = new BrowserDrowseArchiveService(undefined, undefined, "0.0.0");
      const archives = await installedArchives(artifacts);
      const offlineFetches = new Set<string>();
      const offlineOptional = presentKinds.map((kind) =>
        optionalPack(kind, event.data.modelId, fingerprint, variant, offlinePacks.get(kind)!, offlineFetches),
      );
      const offlineRequest: BrowserModelLoadRequest = {
        ...(await loadRequest()),
        artifacts: [...offlineModel, ...offlineCore],
        requiredCorePack: {
          ...requiredCorePack,
          files: offlineCore.map(({ manifest }) => manifest),
        },
        optionalPacks: offlineOptional,
      };
      runtime = new DrowseWebLlmRuntime(webllm);
      await runtime.load(offlineRequest);
      const offlineCompiler = await BrowserFeasibilityCorePackCompiler.load(
        offlineRequest,
        {
          installedManifoldArchives: archives,
        },
      );
      const offlineDictionaries = await loadDictionaries(offlineCompiler, runtime, "offline OPFS reuse");
      const offlineInstruments = new BrowserInstrumentRuntime(
        offlineCompiler,
        runtime,
        contextBinding,
      );
      await configureUnits(offlineInstruments, runtime);
      const curve = await exactGeometry(
        offlineInstruments,
        curveSelector,
        false,
        "browser_e2e_curve_offline",
      );
      const families = gatedFamiliesFor(has("jlens"), has("sae")).filter((family) => family !== "geometry");
      await attachInstrumentProbes(offlineInstruments, event.data, families, "_offline");
      if (unitsProfile) {
        await offlineInstruments.request(
          { service: "probes", method: "attach", args: [{ selector: `slens/${event.data.slensWord}`, name: "browser_e2e_slens_offline" }] },
          () => null,
        );
      }
      for (const family of ["geometry", ...families, ...(unitsProfile && !families.includes("lens") ? ["lens"] : [])]) {
        await setLive(offlineInstruments, family, true);
      }
      const expression = [
        `0.03 ${curve.selector}%${curve.label}@generated`,
        ...(has("jlens") ? [`0.01 jlens/${event.data.jlensWord}@generated`] : []),
        ...(has("sae") ? [`0.01 sae/${event.data.saeFeature}@generated`] : []),
      ].join(" + ");
      const program = await compileBound(offlineCompiler, runtime, expression);
      if (
        !program.curveRank.some((rank) => rank > 0) ||
        !program.geometryActive?.some((active) => active === 1)
      ) {
        throw new Error(
          "offline curved manifold did not lower into physical GPU programs",
        );
      }
      let offlineMeasurements: any = null;
      const result = await runtime.streamGeneration(
        {
          input: { kind: "raw", prompt: "Offline model:" },
          sampling: deterministicSampling(1),
          thinking: false,
          steeringExpression: expression,
          hookProgram: program,
          liveInstruments: offlineInstruments.liveInstruments(),
          onRawToken(token) {
            offlineMeasurements = token.measurements ?? offlineMeasurements;
          },
        },
        () => undefined,
      );
      const offlineReading =
        offlineMeasurements?.scores?.browser_e2e_curve_offline;
      const offlineReadings = familyReadings(offlineMeasurements, families, "_offline");
      const offlineSlensReading = unitsProfile
        ? offlineMeasurements?.instruments?.lens?.readings?.browser_e2e_slens_offline?.value
        : null;
      // On standard-v4 a static unit read offline: its lens and units layers come from OPFS.
      let offlineUnitRead: string | null = null;
      if (unitsProfile) {
        const unit = topUnit ?? `neuron/L${Math.floor(event.data.runtimeIdentity.layerMap.length / 2)}n0`;
        const read = (await offlineInstruments.request(
          { service: "instruments", method: "unitRead", args: [unit, { k: 8 }] },
          () => null,
        )) as { unit?: string; poles?: unknown[] };
        if (read.unit !== unit || !Array.isArray(read.poles) || read.poles.length === 0) {
          throw new Error(`offline unit read of ${unit} returned an invalid record`);
        }
        offlineUnitRead = unit;
      }
      if (
        result.tokens !== 1 ||
        typeof offlineReading !== "number" ||
        !Number.isFinite(offlineReading) ||
        Object.values({
          ...offlineReadings,
          ...(unitsProfile ? { offlineSlensReading } : {}),
        }).some((value) => typeof value !== "number" || !Number.isFinite(value)) ||
        offlineNetworkAttempts.length !== 0
      ) {
        throw new Error(
          "offline OPFS reuse lost a physical geometry/lens/SAE readout or attempted network access: " +
            JSON.stringify(offlineNetworkAttempts),
        );
      }
      const exactReadouts = requireExactPrecomputedReadouts(
        offlineMeasurements,
        offlineCompiler.instrumentDescriptor(),
        "offline reuse",
        { jlens: has("jlens"), sae: has("sae") },
      );
      await runtime.unload();
      runtime = null;
      await artifacts.request(
        {
          service: "manifolds",
          method: "delete",
          args: [harnessManifold.namespace, harnessManifold.name],
        },
        () => undefined,
      );
      await artifacts.close();
      artifacts = null;
      return {
        reopenedModelFiles: offlineModel.length,
        reopenedCoreFiles: offlineCore.length,
        ...Object.fromEntries(PACK_KINDS.map((kind) => [
          `reopened${kind[0].toUpperCase()}${kind.slice(1)}Files`,
          offlinePacks.get(kind)?.length ?? 0,
        ])),
        persistedManifoldArchives: archives.length,
        offlineTokens: result.tokens,
        offlineGeometryReading: offlineReading,
        offlineLensReading: offlineReadings.lens ?? null,
        offlineSaeReading: offlineReadings.sae ?? null,
        offlineSlensReading,
        offlineUnitRead,
        offlineOnDemandPackFiles: offlineFetches.size,
        restoredSaeDictionaryFeatureCount: offlineDictionaries.sae?.featureCount ?? null,
        ...exactReadouts,
        networkRequests: offlineNetworkAttempts.length,
      };
    });
    assertNoDeviceLoss(deviceLosses);
    if (externalNetworkAttempts.length !== 0) {
      throw new Error(
        `verified local runtime attempted external fetches: ${JSON.stringify(externalNetworkAttempts)}`,
      );
    }
    self.postMessage({
      type: "done",
      artifactVerification,
      environment: {
        modelId: event.data.modelId,
        contextTokens: event.data.contextTokens,
        userAgent: navigator.userAgent,
        language: navigator.language,
        runtimeIdentitySha256: fingerprint,
        adapterFallback: capabilities!.webGpu.fallback,
        adapterInfo: capabilities!.webGpu.adapterInfo,
        deviceMemoryGiB: capabilities!.signals.deviceMemoryGiB,
        logicalCpuCount: capabilities!.signals.logicalCpuCount,
        deviceLosses: deviceLosses.map((failure) => ({ ...failure })),
      },
    });
    setTimeout(() => self.close(), 0);
  } catch (error) {
    try {
      await runtime?.unload();
    } catch {}
    try {
      await cleanupArtifacts(artifacts, harnessManifold);
    } catch {}
    self.postMessage({
      type: "error",
      check: currentCheck,
      message:
        error instanceof Error ? (error.stack ?? error.message) : String(error),
    });
  }
};

async function check(
  name: CheckName,
  run: () => Promise<Record<string, unknown>>,
): Promise<void> {
  self.postMessage({ type: "check_started", check: name });
  const measurements = await run();
  if (Object.keys(measurements).length === 0)
    throw new Error(`check ${name} produced no measurements`);
  self.postMessage({ type: "check_passed", check: name, measurements });
}

async function stage(
  set: string,
  files: FileEntry[],
): Promise<BrowserModelArtifact[]> {
  const root = await navigator.storage.getDirectory();
  const harness = await root.getDirectoryHandle("drowse-real-browser-e2e-v1", {
    create: true,
  });
  const directory = await harness.getDirectoryHandle(set, { create: true });
  const revision = "a".repeat(40);
  const artifacts: BrowserModelArtifact[] = [];
  for (const entry of files) {
    const response = await fetch(localArtifactUrl(set, entry.path), {
      cache: "no-store",
    });
    if (!response.ok)
      throw new Error(`artifact fetch failed for ${set}/${entry.path}`);
    if (response.body === null)
      throw new Error(
        `artifact fetch returned no body for ${set}/${entry.path}`,
      );
    const handle = await fileHandle(directory, entry.path);
    const writable = await handle.createWritable();
    await response.body.pipeTo(writable);
    const file = await handle.getFile();
    await verifyPersistedFile(file, entry, set);
    artifacts.push({
      manifest: localManifest(entry, revision, set),
      file,
    });
  }
  return artifacts;
}

function localArtifactUrl(set: string, path: string): string {
  const url = new URL("/artifact", globalThis.location.origin);
  url.searchParams.set("set", set);
  url.searchParams.set("path", path);
  return url.href;
}

function artifactSetReceipt(set: string, files: FileEntry[]) {
  const tuples = files.map((file) => [set, file.path, file.bytes, file.sha256]);
  return {
    fileCount: files.length,
    totalBytes: files.reduce((sum, file) => sum + file.bytes, 0),
    entriesSha256: bytesToHex(
      sha256(new TextEncoder().encode(JSON.stringify(tuples))),
    ),
  };
}

async function reopen(
  set: string,
  files: FileEntry[],
): Promise<BrowserModelArtifact[]> {
  const root = await navigator.storage.getDirectory();
  const harness = await root.getDirectoryHandle("drowse-real-browser-e2e-v1");
  const directory = await harness.getDirectoryHandle(set);
  const revision = "a".repeat(40);
  const artifacts: BrowserModelArtifact[] = [];
  for (const entry of files) {
    const handle = await fileHandle(directory, entry.path, false);
    const file = await handle.getFile();
    await verifyPersistedFile(file, entry, set);
    artifacts.push({ manifest: localManifest(entry, revision, set), file });
  }
  return artifacts;
}

async function verifyPersistedFile(
  file: File,
  entry: FileEntry,
  set: string,
): Promise<void> {
  if (file.size !== entry.bytes) {
    throw new Error(`persisted artifact ${set}/${entry.path} changed size`);
  }
  const hasher = sha256.create();
  const reader = file.stream().getReader();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    hasher.update(value);
  }
  const actual = bytesToHex(hasher.digest());
  if (actual !== entry.sha256) {
    throw new Error(`persisted artifact ${set}/${entry.path} changed SHA-256`);
  }
}

function localManifest(
  entry: FileEntry,
  revision: string,
  set: string,
): CatalogFile {
  return {
    ...entry,
    revision,
    url: `https://huggingface.co/polythetic/local-${set}/resolve/${revision}/${entry.path}`,
  };
}

async function fileHandle(
  root: FileSystemDirectoryHandle,
  path: string,
  create = true,
): Promise<FileSystemFileHandle> {
  const parts = path.split("/");
  if (
    parts.length === 0 ||
    parts.some(
      (part) => !part || part === "." || part === ".." || part.includes("\\"),
    )
  )
    throw new Error(`artifact has an unsafe path: ${path}`);
  let directory = root;
  for (const part of parts.slice(0, -1)) {
    directory = await directory.getDirectoryHandle(part, { create });
  }
  return directory.getFileHandle(parts.at(-1)!, { create });
}

/** One optional pack as the app's worker hands it to a load (worker.ts): the catalog closure, the
 *  install files as artifacts, and each on-demand layer through `fetchFile` (from the staged,
 *  verified copy; `fetched` records which ones a read needed). */
function optionalPack(
  kind: PackKind,
  modelId: string,
  fingerprint: string,
  variant: ModelVariant,
  artifacts: BrowserModelArtifact[],
  fetched: Set<string>,
): BrowserInstrumentPackArtifacts {
  const pack = {
    id: `${modelId}-${kind}-e2e`,
    kind,
    displayName: `Local E2E ${kind} pack`,
    runtimeIdentitySha256: fingerprint,
    files: artifacts.map(({ manifest }) => manifest),
  } as CatalogInstrumentPack;
  const byPath = new Map(artifacts.map((artifact) => [artifact.manifest.path, artifact]));
  return {
    pack,
    artifacts: artifacts.filter(({ manifest }) => packFileResidency(pack, variant, manifest.path) === "install"),
    async fetchFile(manifest: CatalogFile) {
      const artifact = byPath.get(manifest.path);
      if (artifact === undefined || artifact.manifest.sha256 !== manifest.sha256) {
        throw new Error(`the harness staged no ${kind} file ${manifest.path}`);
      }
      fetched.add(`${kind}/${manifest.path}`);
      return artifact.file;
    },
  };
}

type GatedFamily = "geometry" | "lens" | "sae";

/** The instrument families a flat gate reads: geometry always, the R lens and the SAE with packs. */
function gatedFamiliesFor(jlens: boolean, sae: boolean): GatedFamily[] {
  return ["geometry", ...(jlens ? ["lens" as const] : []), ...(sae ? ["sae" as const] : [])];
}

function probeName(family: GatedFamily, suffix: string): string {
  return { geometry: "browser_e2e_flat", lens: "browser_e2e_jlens", sae: "browser_e2e_sae" }[family] + suffix;
}

/** The R-lens and SAE probes of `families` (a geometry probe attaches with its manifold). */
async function attachInstrumentProbes(
  instruments: BrowserInstrumentRuntime,
  data: Pick<RunRequest, "jlensWord" | "saeFeature">,
  families: readonly GatedFamily[],
  suffix: string,
): Promise<void> {
  for (const family of families) {
    if (family === "geometry") continue;
    const selector = family === "lens" ? `jlens/${data.jlensWord}` : `sae/${data.saeFeature}`;
    await instruments.request(
      { service: "probes", method: "attach", args: [{ selector, name: probeName(family, suffix) }] },
      () => null,
    );
  }
}

/** Each family's probe reading (`probeName(family, suffix)`) from one token envelope. */
function familyReadings(
  measurements: any,
  families: readonly GatedFamily[],
  suffix: string,
): Partial<Record<GatedFamily, unknown>> {
  const readings: Partial<Record<GatedFamily, unknown>> = {};
  for (const family of families) {
    const name = probeName(family, suffix);
    readings[family] = family === "geometry"
      ? measurements?.scores?.[name]
      : measurements?.instruments?.[family]?.readings?.[name]?.value;
  }
  return readings;
}

async function setLive(
  instruments: BrowserInstrumentRuntime,
  family: string,
  enabled: boolean,
): Promise<void> {
  await instruments.request(
    { service: "instruments", method: "setLive", args: [family, { enabled }] },
    () => null,
  );
}

/** The block sites over a loaded library, one surface per attested capabilities record, as the
 *  app's backend keeps it (drowseWebLlmBackend.ts `blockSites`). */
const blockSitesSurfaces = new WeakMap<object, BrowserBlockSitesBackend>();
const predatesBlockSites = rejectingBlockSitesBackend(
  () => new UnsupportedBlockError("the loaded model library predates hook ABI block-sites-v5"),
);
function blockSitesFor(loaded: DrowseWebLlmRuntime): BrowserBlockSitesBackend {
  const capabilities = loaded.blockSitesCapabilities();
  if (capabilities === null) return predatesBlockSites;
  let backend = blockSitesSurfaces.get(capabilities);
  if (backend === undefined) {
    backend = webLlmBlockSitesBackend(capabilities, loaded.blockSitesDevice());
    blockSitesSurfaces.set(capabilities, backend);
  }
  return backend;
}

/** A generation's program as the app compiles it (browserModelBackend.ts `compileSteering`): the
 *  device reads first (`jlens/` words, `neuron/` writes, `slens/` atoms), then the compile, then
 *  a unit section bound to the loaded library (its tapped heads' grams). */
async function compileBound(
  compiler: BrowserFeasibilityCorePackCompiler,
  loaded: DrowseWebLlmRuntime,
  expression: string,
): Promise<StructuredHookProgramBuffers> {
  await compiler.prepareJlensSelectors(expression, loaded);
  await compiler.prepareUnitSelectors(expression);
  const program = requireStructured(compiler.compile(expression));
  return program.unitMeta === undefined
    ? program
    : blockSitesFor(loaded).bindStructuredProgram(program);
}

/** The logprob of `tokenId` as the first token after the ordinary prompt, forced and scored. */
async function forcedTokenLogprob(
  loaded: DrowseWebLlmRuntime,
  tokenId: number,
  expression: string | null,
  program: StructuredHookProgramBuffers | null,
  liveInstruments: ReturnType<BrowserInstrumentRuntime["liveInstruments"]>,
): Promise<number> {
  let logprob: number | null = null;
  await loaded.streamGeneration(
    {
      input: { kind: "raw", prompt: ORDINARY_PROMPT },
      sampling: { ...deterministicSampling(1), temperature: 1 },
      thinking: false,
      replay: { forcedPrefixTokenIds: [tokenId], scoreTokenIds: [tokenId] },
      steeringExpression: expression,
      hookProgram: program,
      liveInstruments,
      onRawToken(token) {
        const score = token.replayScore?.requestedLogprobs.find((row) => row.tokenId === tokenId);
        if (score) logprob = score.logprob;
      },
    },
    () => undefined,
  );
  if (logprob === null) throw new Error(`the forced replay scored no logprob for token ${tokenId}`);
  return logprob;
}

/** The loom's node replay (browserLoom.ts `unitsNodeForward`) over one recorded generation: the
 *  forward producing raw row `rawIndex` is the prompt and the rows before it. */
function harnessReplayPort(promptIds: readonly number[], rawIds: readonly number[]): UnitsNodeReplayPort {
  return {
    async nodeForward(nodeId, rawIndex) {
      if (nodeId !== HARNESS_NODE) throw new Error(`the harness replays only ${HARNESS_NODE}, not ${nodeId}`);
      const anchor = rawIndex ?? rawIds.length - 1;
      if (!(Number.isSafeInteger(anchor) && anchor >= 0 && anchor < rawIds.length)) {
        throw new Error(`raw index ${anchor} is outside the harness node`);
      }
      return {
        forward: {
          ids: Int32Array.from([...promptIds, ...rawIds.slice(0, anchor)]),
          promptLength: promptIds.length,
          rawIds: Int32Array.from(rawIds),
          rawIndex: anchor,
          spans: [{ firstRow: 0, stopRow: rawIds.length, at: promptIds.length - 1 }],
          positionSpans: null,
        },
        program: null,
        steering: null,
        canThink: false,
      };
    },
  };
}

/** `run` must refuse with exactly the lock's unit refusal (native's 422 `UnsupportedBlockError`,
 *  no layer and no prefix). */
async function requireRefusal(
  run: () => Promise<unknown>,
  refusal: string,
  label: string,
): Promise<void> {
  try {
    await run();
  } catch (error) {
    const failure = error as { message?: unknown; status?: unknown; code?: unknown };
    if (
      failure?.code === "UnsupportedBlockError" &&
      failure.status === 422 &&
      failure.message === refusal
    ) return;
    throw new Error(
      `${label} did not answer the lock's unit refusal: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  throw new Error(`${label} answered on a library without unit sites`);
}

/** The pack loader refuses a pack bound to another runtime (the core pack, and each optional
 *  pack), and, with an SAE pack staged, a tensor that is not fp32. */
async function verifyInstrumentRejections(
  request: BrowserModelLoadRequest,
): Promise<{
  incompatibleBindingRejected: true;
  rejectedRuntimeBindings: string[];
  fp32TensorValidationPassed: boolean | null;
}> {
  // The core pack's manifolds bind the runtime they were fitted on: under another runtime
  // identity (and its context bindings) none of them loads.
  const foreign = "f".repeat(64);
  await requireCorePackRejection(
    {
      ...request,
      variant: {
        ...request.variant,
        runtimeIdentitySha256: foreign,
        contextProfiles: request.variant.contextProfiles.map((profile) => ({
          ...profile,
          bindingSha256: contextBindingSha256(foreign, profile.contextTokens),
        })),
      },
    },
    "CORE_PACK_RUNTIME_MISMATCH",
  );
  const rejectedRuntimeBindings = ["core"];
  for (let target = 0; target < request.optionalPacks.length; target += 1) {
    const mismatchedPacks = request.optionalPacks.map((entry, index) =>
      index === target
        ? {
            ...entry,
            pack: { ...entry.pack, runtimeIdentitySha256: "f".repeat(64) },
          }
        : entry,
    );
    await requireCorePackRejection(
      { ...request, optionalPacks: mismatchedPacks },
      "INSTRUMENT_PACK_RUNTIME_MISMATCH",
    );
    rejectedRuntimeBindings.push(request.optionalPacks[target].pack.kind);
  }

  const saeIndex = request.optionalPacks.findIndex((entry) => entry.pack.kind === "sae");
  if (saeIndex < 0) {
    return { incompatibleBindingRejected: true, rejectedRuntimeBindings, fp32TensorValidationPassed: null };
  }
  const original = request.optionalPacks[saeIndex];
  const manifestIndex = original.artifacts.findIndex((artifact) =>
    artifact.manifest.path.endsWith("/manifest.json"),
  );
  const tensorIndex = original.artifacts.findIndex((artifact) =>
    artifact.manifest.path.endsWith(".safetensors"),
  );
  if (manifestIndex < 0 || tensorIndex < 0)
    throw new Error("physical SAE pack has no manifest or tensor payload");
  const manifestArtifact = original.artifacts[manifestIndex];
  const tensorArtifact = original.artifacts[tensorIndex];
  const invalidTensorBytes = new Uint8Array(await tensorArtifact.file.arrayBuffer());
  const headerLength = Number(
    new DataView(
      invalidTensorBytes.buffer,
      invalidTensorBytes.byteOffset,
      8,
    ).getBigUint64(0, true),
  );
  const headerStart = 8;
  const headerEnd = headerStart + headerLength;
  const header = new TextDecoder().decode(
    invalidTensorBytes.subarray(headerStart, headerEnd),
  );
  const invalidHeader = header.replaceAll('"F32"', '"F16"');
  if (invalidHeader === header)
    throw new Error("physical SAE tensor has no fp32 header to reject");
  invalidTensorBytes.set(new TextEncoder().encode(invalidHeader), headerStart);
  const invalidTensorSha256 = bytesToHex(sha256(invalidTensorBytes));
  const invalidTensorCatalogFile = {
    ...tensorArtifact.manifest,
    bytes: invalidTensorBytes.byteLength,
    sha256: invalidTensorSha256,
  };
  const invalidManifest = JSON.parse(await manifestArtifact.file.text());
  invalidManifest.tensor_sha256 = invalidTensorSha256;
  const invalidBytes = new TextEncoder().encode(JSON.stringify(invalidManifest));
  const invalidSha256 = bytesToHex(sha256(invalidBytes));
  const invalidCatalogFile = {
    ...manifestArtifact.manifest,
    bytes: invalidBytes.byteLength,
    sha256: invalidSha256,
  };
  const invalidArtifacts = original.artifacts.map((artifact, index) =>
    index === manifestIndex
      ? {
          manifest: invalidCatalogFile,
          file: new File([invalidBytes], "manifest.json", {
            type: "application/json",
          }),
        }
      : index === tensorIndex
        ? {
            manifest: invalidTensorCatalogFile,
            file: new File([invalidTensorBytes], "invalid-fp16.safetensors", {
              type: "application/octet-stream",
            }),
          }
        : artifact,
  );
  const invalidPacks = request.optionalPacks.map((entry, index) =>
    index === saeIndex
      ? {
          ...entry,
          pack: {
            ...entry.pack,
            files: entry.pack.files.map((file) =>
              file.path === invalidCatalogFile.path
                ? invalidCatalogFile
                : file.path === invalidTensorCatalogFile.path
                  ? invalidTensorCatalogFile
                  : file,
            ),
          },
          artifacts: invalidArtifacts,
        }
      : entry,
  );
  await requireCorePackRejection(
    { ...request, optionalPacks: invalidPacks },
    "SAFETENSORS_INVALID",
  );
  return {
    incompatibleBindingRejected: true,
    rejectedRuntimeBindings,
    fp32TensorValidationPassed: true,
  };
}

async function requireCorePackRejection(
  request: BrowserModelLoadRequest,
  expectedCode: string,
): Promise<void> {
  try {
    await BrowserFeasibilityCorePackCompiler.load(request);
  } catch (error) {
    if ((error as { code?: unknown })?.code === expectedCode) return;
    throw error;
  }
  throw new Error(`physical instrument check did not reject ${expectedCode}`);
}

function requireExactPrecomputedReadouts(
  measurements: any,
  descriptor: ReturnType<
    BrowserFeasibilityCorePackCompiler["instrumentDescriptor"]
  >,
  lifecycle: string,
  expected: { jlens: boolean; sae: boolean },
): Record<string, unknown> {
  if ((descriptor.jlens !== null) !== expected.jlens || (descriptor.sae !== null) !== expected.sae) {
    throw new Error(
      `${lifecycle} lost its verified precomputed instrument descriptors`,
    );
  }
  const readouts: Record<string, unknown> = {};
  if (descriptor.jlens !== null) Object.assign(readouts, exactJlensReadout(measurements, descriptor.jlens, lifecycle));
  if (descriptor.sae !== null) Object.assign(readouts, exactSaeReadout(measurements, descriptor.sae, lifecycle));
  return readouts;
}

type InstrumentDescriptor = ReturnType<BrowserFeasibilityCorePackCompiler["instrumentDescriptor"]>;

function exactJlensReadout(
  measurements: any,
  jlens: NonNullable<InstrumentDescriptor["jlens"]>,
  lifecycle: string,
): Record<string, unknown> {
  const lensReadout = measurements?.instruments?.lens?.readout;
  const aggregate = lensReadout?.aggregate;
  const layers = lensReadout?.layers;
  if (
    !Array.isArray(aggregate) ||
    aggregate.length !== 8 ||
    aggregate.some(
      (row) =>
        typeof row?.token !== "string" ||
        typeof row?.strength !== "number" ||
        !Number.isFinite(row.strength) ||
        row.strength < 0 ||
        row.strength > 1 ||
        typeof row?.com !== "number" ||
        !Number.isFinite(row.com) ||
        typeof row?.spread !== "number" ||
        !Number.isFinite(row.spread) ||
        row.spread < 0,
    ) ||
    !Array.isArray(layers) ||
    layers.length !== jlens.layers.length ||
    layers.some(
      (row) =>
        !jlens.layers.includes(row?.layer) ||
        !Array.isArray(row?.tokens) ||
        row.tokens.length !== 8 ||
        row.tokens.some(
          (token: any) =>
            !Number.isSafeInteger(token?.id) ||
            typeof token?.token !== "string" ||
            typeof token?.logprob !== "number" ||
            !Number.isFinite(token.logprob),
        ),
    )
  ) {
    throw new Error(
      `${lifecycle} returned no valid exact full-vocabulary J-lens readout`,
    );
  }
  return {
    exactJlensAggregateTokens: aggregate.length,
    exactJlensLayerRows: layers.length,
    exactJlensPerLayerTokens: layers.reduce(
      (sum: number, row: { tokens: unknown[] }) => sum + row.tokens.length,
      0,
    ),
  };
}

function exactSaeReadout(
  measurements: any,
  sae: NonNullable<InstrumentDescriptor["sae"]>,
  lifecycle: string,
): Record<string, unknown> {
  const features = measurements?.instruments?.sae?.readout?.features;
  if (
    !Array.isArray(features) ||
    features.length < 1 ||
    features.length > 8 ||
    features.some(
      (feature) =>
        !Number.isSafeInteger(feature?.id) ||
        feature.id < 0 ||
        feature.id >= sae.features ||
        typeof feature?.activation !== "number" ||
        !Number.isFinite(feature.activation) ||
        feature.activation < 0,
    )
  ) {
    throw new Error(
      `${lifecycle} returned no valid exact full-dictionary SAE readout`,
    );
  }
  return {
    exactSaeDiscoveredFeatures: features.length,
    exactSaeDictionaryFeatureCount: sae.features,
    exactSaeLayer: sae.layer,
  };
}

async function firstGeometry(
  instruments: BrowserInstrumentRuntime,
  affine: boolean,
  probeName: string,
): Promise<{
  selector: string;
  label: string;
  geometry: { layers: Record<string, unknown> };
}> {
  const listed = (await instruments.request(
    { service: "manifolds", method: "list", args: [] },
    () => null,
  )) as {
    manifolds: Array<{
      namespace: string;
      name: string;
      node_labels: string[];
    }>;
  };
  const failures: string[] = [];
  for (const manifold of listed.manifolds) {
    const selector = `${manifold.namespace}/${manifold.name}`;
    try {
      return await exactGeometry(instruments, selector, affine, probeName);
    } catch (error) {
      failures.push(
        `${selector}: ${error instanceof Error ? error.message : String(error)}`,
      );
      await instruments.request(
        { service: "probes", method: "detach", args: [probeName] },
        () => null,
      );
    }
  }
  throw new Error(
    `no ${affine ? "affine" : "curved"} physical geometry was available: ${failures.join("; ")}`,
  );
}

async function exactGeometry(
  instruments: BrowserInstrumentRuntime,
  selector: string,
  affine: boolean,
  probeName: string,
): Promise<{
  selector: string;
  label: string;
  geometry: { layers: Record<string, unknown> };
}> {
  const parts = selector.split("/");
  if (parts.length !== 2)
    throw new Error(`geometry selector must be canonical: ${selector}`);
  const detail = (await instruments.request(
    { service: "manifolds", method: "get", args: parts },
    () => null,
  )) as { node_labels: string[] };
  if (!Array.isArray(detail.node_labels) || detail.node_labels.length === 0) {
    throw new Error(`${selector} has no addressable node label`);
  }
  await instruments.request(
    {
      service: "probes",
      method: "attach",
      args: [{ selector, name: probeName }],
    },
    () => null,
  );
  const geometry = (await instruments.request(
    { service: "probes", method: "geometry", args: [probeName] },
    () => null,
  )) as { is_affine?: boolean; layers?: Record<string, unknown> };
  if (
    geometry.is_affine !== affine ||
    !geometry.layers ||
    Object.keys(geometry.layers).length === 0
  ) {
    throw new Error(
      `${selector} is not a fitted ${affine ? "affine" : "curved"} geometry`,
    );
  }
  return {
    selector,
    label: detail.node_labels[0],
    geometry: { layers: geometry.layers },
  };
}

async function installedArchives(
  service: BrowserDrowseArchiveService,
): Promise<Blob[]> {
  const listed = (await service.request(
    { service: "manifolds", method: "drowseArchiveList", args: [] },
    () => undefined,
  )) as { packs?: Array<{ id?: unknown }> };
  if (!Array.isArray(listed.packs))
    throw new Error("local manifold repository returned no pack index");
  const archives: Blob[] = [];
  for (const pack of listed.packs) {
    if (typeof pack.id !== "string")
      throw new Error("local manifold repository returned an invalid pack ID");
    const archive = await service.request(
      { service: "manifolds", method: "drowseArchiveExport", args: [pack.id] },
      () => undefined,
    );
    if (!(archive instanceof Blob))
      throw new Error(`local manifold archive ${pack.id} is unreadable`);
    archives.push(archive);
  }
  return archives;
}

async function cleanupArtifacts(
  service: BrowserDrowseArchiveService | null,
  manifold: { namespace: string; name: string },
): Promise<void> {
  if (service === null) return;
  await service.request(
    {
      service: "manifolds",
      method: "delete",
      args: [manifold.namespace, manifold.name],
    },
    () => undefined,
  );
  await service.close();
}

function requireStructured(program: unknown): StructuredHookProgramBuffers {
  if (!program || typeof program !== "object" || !("affineActive" in program)) {
    throw new Error(
      "browser expression did not lower into the structured GPU program",
    );
  }
  return program as StructuredHookProgramBuffers;
}

function deterministicSampling(maxTokens: number) {
  return {
    max_tokens: maxTokens,
    seed: 1,
    temperature: 0,
    top_p: 1,
    top_k: 0,
    presence_penalty: 0,
    frequency_penalty: 0,
  };
}

function assertNoDeviceLoss(
  losses: Array<{ code: string; message: string; confirmedOom: boolean }>,
): void {
  if (losses.length > 0)
    throw new Error(
      `physical WebGPU device was lost: ${JSON.stringify(losses)}`,
    );
}
