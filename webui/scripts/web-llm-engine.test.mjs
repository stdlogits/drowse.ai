import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";
import {
  effectiveDrowseTopK,
  sampleDrowseTopKTopP,
} from "@drowse/web-llm";

assert.equal(typeof effectiveDrowseTopK, "function");
assert.equal(typeof sampleDrowseTopKTopP, "function");

const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({
  root,
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});

try {
  const {
    DrowseWebLlmRuntime,
    WEB_LLM_DECODED_TOKEN_CACHE_MAX,
    browserThinkingProfile,
  } = await server.ssrLoadModule(
    "/src/hosted/runtime/webLlmEngine.ts",
  );
  const {
    compileStructuredHookProgram,
  } = await server.ssrLoadModule(
    "/src/hosted/runtime/structuredHookProgram.ts",
  );
  const {
    WebLlmActivationCaptureSource,
    prepareWebLlmActivationCaptureRows,
  } = await server.ssrLoadModule(
    "/src/hosted/runtime/webLlmActivationCapture.ts",
  );
  const {
    STANDARD_STRUCTURED_HOOK_PROFILE_V4_RESIDUAL,
    expectedHookProfileDescriptor,
  } = await server.ssrLoadModule("/src/hosted/runtime/structuredHookProfile.ts");
  const { hookProfileDescriptorModel } = await server.ssrLoadModule(
    "/src/hosted/runtime/webLlmEngine.ts",
  );
  const revision = "a".repeat(40);
  const sha = "b".repeat(64);
  const base = `https://huggingface.co/a9lim/drowse-web-fixture/resolve/${revision}/`;
  const config = JSON.stringify({
    vocab_size: 16,
    tokenizer_files: ["tokenizer.json", "tokenizer_config.json"],
    drowse_capture_special_token_ids: [9],
    conv_template: {
      roles: {
        user: "<|im_start|>user",
        assistant: "<|im_start|>assistant",
      },
    },
  });
  const tensorCache = JSON.stringify({
    records: [{ dataPath: "params_shard_0.bin", nbytes: 1, records: [] }],
  });
  const artifact = (path, role, content) => ({
    manifest: {
      path,
      role,
      url: `${base}${path}`,
      revision,
      bytes: new Blob([content]).size,
      sha256: sha,
    },
    file: new File([content], path),
  });
  const artifacts = [
    artifact("mlc-chat-config.json", "configuration", config),
    artifact("tokenizer.json", "tokenizer", JSON.stringify({
      added_tokens: [{ id: 9, content: "<turn>", special: false }],
    })),
    artifact("tensor-cache.json", "converted_manifest", tensorCache),
    artifact("params_shard_0.bin", "weight", new Uint8Array([1])),
    artifact("tokenizer_config.json", "chat_template", JSON.stringify({
      chat_template: "{{ messages }}",
    })),
    artifact("model-lib.wasm", "model_library", new Uint8Array([0, 97, 115, 109])),
  ];
  const adapter = {
    features: new Set(["shader-f16"]),
    limits: {
      maxBufferSize: 268_435_456,
      maxStorageBufferBindingSize: 134_217_728,
      maxStorageBuffersPerShaderStage: 10,
      maxComputeWorkgroupSizeX: 256,
      maxComputeInvocationsPerWorkgroup: 256,
    },
    requestDevice() {},
  };
  // A hook ABI v5 library on the residual profile (no unit sites): every v5 library carries the
  // exact readout, so its variant declares the exact-readout WebGPU limits.
  const variant = {
    id: "fixture-q4",
    structuredHookProfile: "standard-v4-residual",
    thinkingProfile: null,
    contextProfiles: [{ contextTokens: 2048 }],
    runtimeIdentity: {
      hookAbi: "block-sites-v5",
      sourceModel: "HuggingFaceTB/SmolLM2-360M-Instruct",
      hiddenSize: 2,
      layerMap: [0, 1],
    },
    requirements: {
      features: ["shader-f16"],
      limits: {
        maxStorageBufferBindingSize: 134_217_728,
        maxStorageBuffersPerShaderStage: 10,
        maxComputeWorkgroupSizeX: 256,
        maxComputeInvocationsPerWorkgroup: 256,
      },
    },
  };
  const variantWith = (overrides = {}) => ({ ...structuredClone(variant), ...overrides });
  // The bridge's schema-4 record for `variant` (B1 §3.5): the residual descriptor the adapter
  // attests element by element.
  const compiledProfile = (overrides = {}, forVariant = variant) => ({
    schemaVersion: 4,
    id: "standard-v4-residual",
    hookAbi: "block-sites-v5",
    format: "drowse-structured-v4",
    unitSites: 0,
    descriptor: Array.from(expectedHookProfileDescriptor(
      STANDARD_STRUCTURED_HOOK_PROFILE_V4_RESIDUAL,
      hookProfileDescriptorModel({ variant: forVariant }),
    )),
    ...overrides,
  });
  const loadRequest = (overrides = {}) => ({
    model: { id: "smollm2-360m-instruct" },
    variant,
    contextTokens: 2048,
    adapter,
    artifacts,
    signal: new AbortController().signal,
    onDeviceLost() {},
    ...overrides,
  });

  class FixtureEngine {
    static instances = [];
    static structuredHooks = true;
    unloaded = 0;
    interrupted = 0;
    hookProgram = null;
    measurements = Float32Array.from([0.25, -0.5]);
    geometryMeasurements = Float32Array.from([1, 0.75, 0.1, 0.9]);
    decodedTokenCalls = [];
    jlensDictionarySets = 0;
    jlensDictionaryClears = 0;
    capture = true;
    constructor(config) {
      this.config = config;
      FixtureEngine.instances.push(this);
    }
    async reload(modelId) {
      this.modelId = modelId;
      this.config.initProgressCallback?.({
        progress: 0.5,
        timeElapsed: 1.25,
        text: "Loading model weights",
      });
    }
    async unload() { this.unloaded += 1; }
    async interruptGenerate() { this.interrupted += 1; }
    async resetChat() { this.chatResets = (this.chatResets ?? 0) + 1; }
    async getDrowseRuntimeCapabilities() {
      return {
        topK: true,
        forcedReplay: true,
        replayScoring: true,
        tokenizer: true,
        namedRoles: true,
        userSeatGeneration: true,
        sceneStitching: true,
      };
    }
    async getDrowseStructuredHookProfile() { return compiledProfile(); }
    async tokenizeDrowseText(text) {
      return [...text].map((character) => character.codePointAt(0));
    }
    async decodeDrowseTokens(tokenIds) {
      this.decodedTokenCalls.push([...tokenIds]);
      await Promise.resolve();
      return String.fromCodePoint(...tokenIds);
    }
    async supportsDrowseStructuredHooks() {
      return this.constructor.structuredHooks;
    }
    async supportsDrowseResidualCapture() { return this.capture; }
    async prepareDrowseCaptureRows(rows) {
      return rows.map((_row, index) => ({ inputIds: [index, 9], position: 1 }));
    }
    async captureDrowseResiduals(inputIds, positions) {
      if (positions.length > 1) {
        return {
          layerCount: 2,
          positionCount: positions.length,
          hiddenSize: 2,
          positions,
          values: Float32Array.from([
            0, 1, 2, 3,
            10, 11, 12, 13,
          ]),
        };
      }
      return {
        layerCount: 2,
        positionCount: positions.length,
        hiddenSize: 2,
        positions,
        values: Float32Array.from([inputIds[0], 1, inputIds[0] + 2, 3]),
      };
    }
    async setDrowseStructuredProgram(program) {
      this.structuredProgram = program;
      this.hookProgram = program;
    }
    async clearDrowseProgram() {
      this.structuredProgram = null;
      this.hookProgram = null;
    }
    async setDrowseJlensDictionary(dictionary) {
      this.jlensDictionarySets += 1;
      if (this.failNextJlensDictionarySet) {
        this.failNextJlensDictionarySet = false;
        throw new Error("J-lens upload failed");
      }
      this.jlensDictionary = dictionary;
    }
    async clearDrowseJlensDictionary() {
      this.jlensDictionary = null;
      this.jlensDictionaryClears += 1;
    }
    async readDrowseMeasurementBundle() {
      return {
        ...(this.measurements === undefined ? {} : { scalar: this.measurements }),
        ...(this.geometryMeasurements === undefined
          ? {}
          : { geometry: this.geometryMeasurements }),
        ...(this.hookProgram?.measurementSchema?.lensReadout !== true ||
        typeof this.readDrowseJlensTopTokens !== "function"
          ? {}
          : { jlensTopTokens: await this.readDrowseJlensTopTokens() }),
        ...(this.hookProgram?.measurementSchema?.saeReadout !== true ||
        typeof this.readDrowseSaeTopFeatures !== "function"
          ? {}
          : { saeTopFeatures: await this.readDrowseSaeTopFeatures() }),
      };
    }
    async readDrowseMeasurements() { return this.measurements; }
    async readDrowseGeometryMeasurements() { return this.geometryMeasurements; }
    async resolveDrowseJlensTokenDirections(bindingId, layerIndices, tokenIds) {
      assert.equal(bindingId, this.jlensDictionary.bindingId);
      return Float32Array.from(layerIndices.flatMap(() =>
        tokenIds.flatMap((tokenId) => [tokenId, -tokenId])
      ));
    }
  }
  assert.equal(browserThinkingProfile({
    model: { id: "qwen3-name-must-not-enable-thinking" },
    variant,
  }), null);
  const runtime = new DrowseWebLlmRuntime({
    DROWSE_HOOK_ABI: "block-sites-v5",
    MLCEngine: FixtureEngine,
  });
  const loadProgress = [];
  assert.throws(
    () => runtime.hookCapabilities(),
    (error) => error.code === "MODEL_NOT_LOADED",
  );
  assert.throws(
    () => runtime.vocabularySize(),
    (error) => error.code === "MODEL_NOT_LOADED",
  );
  assert.deepEqual(await runtime.load(loadRequest({
    onProgress: (event) => loadProgress.push(event),
  })), {
    prefillTokensPerSecond: null,
    decodeTokensPerSecond: null,
  });
  assert.deepEqual(loadProgress, [{
    event: "progress",
    data: {
      kind: "model_load",
      phase: "webllm_initialization",
      progress: 0.5,
      elapsedSeconds: 1.25,
      message: "Loading model weights",
    },
  }]);
  assert.deepEqual(runtime.runtimeCapabilities(), {
    topK: true,
    forcedReplay: true,
    replayScoring: true,
    tokenizer: true,
    namedRoles: true,
    userSeatGeneration: true,
    sceneStitching: true,
    baseModel: false,
    defaultUserRole: "user",
    defaultAssistantRole: "assistant",
  });
  assert.deepEqual(await runtime.tokenizeText(" lead"), [32, 108, 101, 97, 100]);
  assert.equal(runtime.vocabularySize(), 16);
  assert.equal(await runtime.decodeTokens([115, 117, 98]), "sub");
  assert.deepEqual(runtime.hookCapabilities(), {
    rank_one: true,
    multi_term: true,
    projection: true,
    ablation: true,
    curved_manifold: false,
    phase_trigger: true,
    probe_gate: true,
    sae: false,
    jlens: false,
    slens: false,
    units: false,
  });
  assert.doesNotThrow(() => runtime.assertSteeringSupported("0.5 deceptive"));
  assert.doesNotThrow(() => runtime.assertSteeringSupported("0.5 !deceptive"));
  assert.throws(
    () => runtime.assertSteeringSupported("deceptive + !sae/42@response"),
    (error) => error.code === "DROWSE_HOOK_FEATURE_UNAVAILABLE",
  );
  const engine = FixtureEngine.instances.at(-1);
  const appConfig = engine.config.appConfig;
  assert.ok(appConfig.artifactCache);
  assert.equal(Object.hasOwn(engine.config, "cacheBackend"), false);
  assert.equal(Object.hasOwn(engine.config, "opfsAccessMode"), false);
  assert.equal(Object.hasOwn(appConfig, "cacheBackend"), false);
  assert.equal(Object.hasOwn(appConfig, "opfsAccessMode"), false);
  assert.equal(
    await appConfig.artifactCache.hasAllKeys(artifacts.map(({ manifest }) => manifest.url)),
    true,
  );
  assert.equal(
    await appConfig.artifactCache.hasAllKeys([`${base}outside-install.bin`]),
    false,
  );
  await assert.rejects(
    appConfig.artifactCache.fetchWithCache(`${base}outside-install.bin`, "arraybuffer"),
    (error) => error.code === "UNVERIFIED_ARTIFACT_REQUESTED",
  );
  await assert.rejects(
    appConfig.artifactCache.addToCache(`${base}outside-install.bin`, "arraybuffer"),
    (error) => error.code === "UNVERIFIED_ARTIFACT_REQUESTED",
  );
  assert.equal(appConfig.gpuAdapter, adapter);
  assert.equal(appConfig.model_list[0].model, base);
  assert.equal(appConfig.model_list[0].model_lib, `${base}model-lib.wasm`);
  assert.equal(appConfig.model_list[0].overrides.context_window_size, 2048);
  assert.deepEqual(appConfig.model_list[0].required_features, ["shader-f16"]);
  const hookProgram = compileStructuredHookProgram(2, [{
    affineGroups: [{ active: true, basis: [[1, 0]], neutral: [0, 0], target: [1], along: 0.5, kappa: [0] }],
    probes: [{ kind: "linear", direction: [1, 0], bias: 0 }],
  }, {}], STANDARD_STRUCTURED_HOOK_PROFILE_V4_RESIDUAL);
  await runtime.installHookProgram(hookProgram);
  assert.equal(engine.hookProgram, hookProgram);
  const measurements = await runtime.readMeasurements();
  assert.deepEqual([...measurements], [0.25, -0.5]);
  assert.notEqual(measurements, engine.measurements);
  const geometryMeasurements = await runtime.readGeometryMeasurements();
  assert.deepEqual([...geometryMeasurements], [...engine.geometryMeasurements]);
  assert.notEqual(geometryMeasurements, engine.geometryMeasurements);
  await runtime.setJlensDictionary({
    bindingId: "c".repeat(64),
    hiddenSize: 2,
    layerIndices: Int32Array.of(0),
    matrices: [new Float32Array(4)],
  });
  assert.equal(engine.jlensDictionarySets, 0);
  assert.deepEqual(
    [...await runtime.resolveJlensTokenDirections("c".repeat(64), [0], [3, 7])],
    [3, -3, 7, -7],
  );
  assert.equal(engine.jlensDictionarySets, 1);
  const prepared = await runtime.prepareCaptureRows(
    [{ system: "brief", messages: [{ role: "assistant", content: "answer" }] }],
  );
  assert.deepEqual(prepared, [{ inputIds: [0, 9], position: 1 }]);
  const capture = await runtime.capturePreparedRow(prepared[0]);
  assert.deepEqual([...capture.values], [0, 1, 2, 3]);
  assert.notEqual(capture.values, (await engine.captureDrowseResiduals([0, 9], [1])).values);
  const multiCapture = await runtime.capturePreparedPositions(
    { inputIds: [0, 4, 9], position: 2 },
    [0, 2],
  );
  assert.deepEqual(multiCapture.positions, [0, 2]);
  assert.deepEqual([...multiCapture.values], [0, 1, 2, 3, 10, 11, 12, 13]);
  await assert.rejects(
    runtime.capturePreparedPositions({ inputIds: [0, 4, 9], position: 2 }, [2, 2]),
    (error) => error.code === "INVALID_CAPTURE_POSITIONS",
  );
  await runtime.clearHookProgram();
  assert.equal(engine.hookProgram, null);
  let generationRequest;
  engine.chat = {
    completions: {
      async create(request) {
        generationRequest = request;
        return successfulGenerationStream();
      },
    },
  };
  const generated = await runtime.streamGeneration(
    { input: { kind: "chat", messages: [{ role: "user", content: "hello" }] }, thinking: false },
    () => {},
  );
  assert.equal(generated.text, "answer");
  assert.equal(engine.chatResets, 1);
  assert.equal(generationRequest.max_tokens, loadRequest().contextTokens);
  assert.deepEqual(runtime.generationPerformance(), {
    prefillTokensPerSecond: 18,
    decodeTokensPerSecond: 7.5,
  });
  const followupMessages = [
    { role: "user", content: "hello" },
    { role: "assistant", content: "answer" },
    { role: "user", content: "and next?" },
  ];
  engine.chat.completions.create = async request => {
    assert.equal(engine.chatResets, 1, "ordinary chat lets the vendor verify and reuse an exact conversation prefix");
    assert.deepEqual(request.messages, followupMessages);
    return successfulGenerationStream();
  };
  assert.equal((await runtime.streamGeneration(
    { input: { kind: "chat", messages: followupMessages }, thinking: false,
      generationSeat: "assistant", generationRoleName: null, hookProgram: null }, () => {},
  )).text, "answer");
  engine.chat.completions.create = async () => {
    assert.equal(engine.chatResets, 2, "steering invalidates plain conversation prefix reuse");
    return successfulGenerationStream();
  };
  await runtime.streamGeneration({input: {kind: "chat", messages: followupMessages},
    hookProgram, steeringExpression: "0.5 local/fixture"}, () => {});
  engine.chat.completions.create = async () => {
    assert.equal(engine.chatResets, 3, "plain chat after steering starts from a clean prefix");
    return successfulGenerationStream();
  };
  await runtime.streamGeneration({input: {kind: "chat", messages: followupMessages}}, () => {});
  engine.chat.completions.create = async () => successfulGenerationStream("length");
  await runtime.streamGeneration({ input: { kind: "chat", messages: followupMessages }, thinking: false }, () => {});
  engine.chat.completions.create = async () => {
    assert.equal(engine.chatResets, 4, "a capped response cannot leave a reusable prefix");
    return successfulGenerationStream();
  };
  await runtime.streamGeneration({ input: { kind: "chat", messages: followupMessages }, thinking: false }, () => {});
  engine.chat.completions.create = async () => {
    throw new Error("fixture generation failed");
  };
  await assert.rejects(
    runtime.streamGeneration(
      { input: { kind: "chat", messages: [{ role: "user", content: "fail" }] } },
      () => {},
    ),
    /fixture generation failed/,
  );
  assert.deepEqual(runtime.generationPerformance(), {
    prefillTokensPerSecond: null,
    decodeTokensPerSecond: null,
  });
  await runtime.stop();
  assert.equal(engine.interrupted, 1);
  await runtime.unload();
  assert.deepEqual(runtime.generationPerformance(), {
    prefillTokensPerSecond: null,
    decodeTokensPerSecond: null,
  });

  const appleMobileRuntime = new DrowseWebLlmRuntime({
    DROWSE_HOOK_ABI: "block-sites-v5",
    MLCEngine: FixtureEngine,
  });
  await appleMobileRuntime.load(loadRequest({
    runtimeClass: "apple-mobile-webkit",
  }));
  const appleMobileEngine = FixtureEngine.instances.at(-1);
  let appleMobileGenerationRequest;
  appleMobileEngine.chat = {
    completions: {
      async create(request) {
        appleMobileGenerationRequest = request;
        return successfulGenerationStream();
      },
    },
  };
  await appleMobileRuntime.streamGeneration({
    input: { kind: "chat", messages: [{ role: "user", content: "hello" }] },
    sampling: { max_tokens: 8_192 },
  }, () => {});
  assert.equal(appleMobileGenerationRequest.max_tokens, loadRequest().contextTokens);
  await appleMobileRuntime.unload();

  class RetryableUnloadEngine extends FixtureEngine {
    unloadAttempts = 0;
    failUnload = true;
    async unload() {
      this.unloadAttempts += 1;
      if (this.failUnload) throw new Error("fixture physical unload failed");
      await super.unload();
    }
  }
  const retryableUnloadRuntime = new DrowseWebLlmRuntime({
    DROWSE_HOOK_ABI: "block-sites-v5",
    MLCEngine: RetryableUnloadEngine,
  });
  await retryableUnloadRuntime.load(loadRequest());
  const retryableUnloadEngine = RetryableUnloadEngine.instances.at(-1);
  await assert.rejects(
    retryableUnloadRuntime.unload(),
    /fixture physical unload failed/,
  );
  assert.equal(retryableUnloadRuntime.requireEngine(), retryableUnloadEngine);
  assert.equal(retryableUnloadRuntime.runtimeCapabilities().tokenizer, true);
  retryableUnloadEngine.failUnload = false;
  await retryableUnloadRuntime.unload();
  assert.equal(retryableUnloadEngine.unloadAttempts, 2);
  assert.throws(
    () => retryableUnloadRuntime.requireEngine(),
    (error) => error.code === "MODEL_NOT_LOADED",
  );

  class FailedLoadCleanupEngine extends FixtureEngine {
    async supportsDrowseStructuredHooks() { return false; }
    unloadAttempts = 0;
    failUnload = true;
    async unload() {
      this.unloadAttempts += 1;
      if (this.failUnload) throw new Error("fixture failed-load cleanup failed");
      await super.unload();
    }
  }
  const failedLoadCleanupRuntime = new DrowseWebLlmRuntime({
    DROWSE_HOOK_ABI: "block-sites-v5",
    MLCEngine: FailedLoadCleanupEngine,
  });
  await assert.rejects(
    failedLoadCleanupRuntime.load(loadRequest()),
    (error) => {
      assert.equal(error.code, "RUNTIME_CLEANUP_FAILED");
      assert.match(error.message, /structured Drowse hooks/);
      assert.match(error.message, /fixture failed-load cleanup failed/);
      return true;
    },
  );
  const failedLoadCleanupEngine = FailedLoadCleanupEngine.instances.at(-1);
  assert.equal(failedLoadCleanupRuntime.requireEngine(), failedLoadCleanupEngine);
  failedLoadCleanupEngine.failUnload = false;
  await failedLoadCleanupRuntime.unload();
  assert.equal(failedLoadCleanupEngine.unloadAttempts, 2);
  assert.throws(
    () => failedLoadCleanupRuntime.requireEngine(),
    (error) => error.code === "MODEL_NOT_LOADED",
  );

  await runtime.load(loadRequest());
  assert.deepEqual(runtime.generationPerformance(), {
    prefillTokensPerSecond: null,
    decodeTokensPerSecond: null,
  });
  await runtime.installHookProgram(hookProgram);
  assert.equal(FixtureEngine.instances.at(-1).structuredProgram, hookProgram);
  for (const format of ["drowse-structured-v2", "drowse-structured-v3"]) {
    await assert.rejects(
      runtime.installHookProgram({ ...hookProgram, format }),
      (error) => error.code === "HOOK_FORMAT_MISMATCH",
    );
  }
  await assert.rejects(
    runtime.installHookProgram({ ...hookProgram, hookAbi: "post-block-residual-v4" }),
    (error) => error.code === "HOOK_ABI_MISMATCH",
  );
  // Rank-one buffers have no v5 entry point.
  await assert.rejects(
    runtime.installHookProgram({
      hookAbi: "block-sites-v5", hiddenSize: 2, layerCount: 2, enabled: Uint32Array.of(1, 0),
    }),
    (error) => error.code === "HOOK_FORMAT_MISMATCH",
  );
  await runtime.unload();

  const thinkingProfile = {
    start: "<think>",
    end: "</think>",
    startsInThinking: false,
    startTokenIds: [..."<think>"].map((character) => character.codePointAt(0)),
    endTokenIds: [..."</think>"].map((character) => character.codePointAt(0)),
  };
  await runtime.load(loadRequest({
    variant: { ...variant, thinkingProfile },
  }));
  assert.deepEqual(
    browserThinkingProfile({ variant: { ...variant, thinkingProfile } }),
    thinkingProfile,
  );
  await runtime.unload();

  await assert.rejects(
    runtime.load(loadRequest({
      variant: {
        ...variant,
        thinkingProfile: { ...thinkingProfile, endTokenIds: [999] },
      },
    })),
    (error) => error.code === "THINKING_PROFILE_MISMATCH",
  );
  assert.equal(FixtureEngine.instances.at(-1).unloaded, 1);

  const captureWrites = [];
  const captureProgress = [];
  const source = new WebLlmActivationCaptureSource({
    async prepareCaptureRows(rows) {
      return rows.map((_row, index) => ({ inputIds: [index], position: 0 }));
    },
    async capturePreparedRow(row) {
      const baseValue = row.inputIds[0] * 100;
      return {
        layerCount: 4,
        positionCount: 1,
        hiddenSize: 2,
        positions: [0],
        values: Float32Array.from([
          baseValue, baseValue + 1,
          baseValue + 10, baseValue + 11,
          baseValue + 20, baseValue + 21,
          baseValue + 30, baseValue + 31,
        ]),
      };
    },
  }, {
    descriptor: {
      runtimeIdentitySha256: "1".repeat(64),
      contextBindingSha256: "2".repeat(64),
      captureSha256: "3".repeat(64),
      layers: [1, 3].map((layer) => ({
        layer,
        rows: 3,
        width: 2,
        expectedBytes: 24,
      })),
    },
    rows: Array.from({ length: 3 }, (_value, index) => ({
      system: "brief",
      messages: [{ role: "assistant", content: `row ${index}` }],
    })),
    layerMap: [0, 1, 2, 3],
    batchRows: 2,
  });
  await source.capture({
    async appendRows(layer, startRow, values) {
      captureWrites.push({ layer, startRow, values: [...values] });
    },
  }, new AbortController().signal, (completed, total) => {
    captureProgress.push([completed, total]);
  });
  assert.deepEqual(captureWrites, [
    { layer: 1, startRow: 0, values: [10, 11, 110, 111] },
    { layer: 3, startRow: 0, values: [30, 31, 130, 131] },
    { layer: 1, startRow: 2, values: [210, 211] },
    { layer: 3, startRow: 2, values: [230, 231] },
  ]);
  assert.deepEqual(captureProgress, [[1, 3], [2, 3], [3, 3]]);
  const preparedFixtureRuntime = {
    async prepareCaptureRows(rows) {
      return rows.map((_row, index) => ({ inputIds: [4, index + 5], position: 1 }));
    },
  };
  const rendered = await prepareWebLlmActivationCaptureRows(
    preparedFixtureRuntime,
    [
      { system: "", messages: [{ role: "assistant", content: "a" }] },
      { system: "", messages: [{ role: "assistant", content: "b" }] },
    ],
    new Uint32Array([0, 1, 2]),
  );
  assert.match(rendered.captureRenderSha256, /^[a-f0-9]{64}$/);
  assert.deepEqual(rendered.preparedRows, [
    { inputIds: [4, 5], position: 1 },
    { inputIds: [4, 6], position: 1 },
  ]);
  assert.equal(
    rendered.captureRenderSha256,
    (await prepareWebLlmActivationCaptureRows(
      preparedFixtureRuntime,
      [
        { system: "", messages: [{ role: "assistant", content: "a" }] },
        { system: "", messages: [{ role: "assistant", content: "b" }] },
      ],
      new Uint32Array([0, 1, 2]),
    )).captureRenderSha256,
  );
  assert.equal(engine.unloaded, 1);

  let deviceLoss = null;
  await runtime.load(loadRequest({ onDeviceLost: (failure) => { deviceLoss = failure; } }));
  FixtureEngine.instances.at(-1).config.appConfig.onDeviceLost({ reason: "unknown" });
  assert.equal(deviceLoss.code, "WEBGPU_DEVICE_LOST");
  assert.equal(deviceLoss.confirmedOom, false);
  assert.doesNotMatch(deviceLoss.message, /Last initialization phase/u);
  await runtime.unload();

  class LostDuringInitEngine extends FixtureEngine {
    async reload() {
      this.config.initProgressCallback({ progress: 0.4, timeElapsed: 2, text: "Loading GPU shader modules" });
      this.config.appConfig.onDeviceLost({ reason: "unknown" });
      throw new Error("initialization interrupted");
    }
  }
  const lostDuringInit = new DrowseWebLlmRuntime({
    DROWSE_HOOK_ABI: "block-sites-v5",
    MLCEngine: LostDuringInitEngine,
  });
  await assert.rejects(lostDuringInit.load(loadRequest({
    onDeviceLost: (failure) => { deviceLoss = failure; },
  })), /initialization interrupted/u);
  assert.match(deviceLoss.message, /Last initialization phase: Loading GPU shader modules/u);

  const badAbi = new DrowseWebLlmRuntime({
    DROWSE_HOOK_ABI: "different",
    MLCEngine: FixtureEngine,
  });
  await assert.rejects(
    badAbi.load(loadRequest()),
    (error) => error.code === "HOOK_ABI_MISMATCH",
  );

  // Schema-4 attestation: any descriptor element, the profile id or the unit-site flag that
  // differs from the lock fails the load (and unloads the engine); a variant on a pre-v5 profile
  // is refused before attestation.
  const changedDescriptor = compiledProfile().descriptor;
  changedDescriptor[5] += 1;
  for (const [record, code] of [
    [compiledProfile({ descriptor: changedDescriptor }), "HOOK_ABI_MISMATCH"],
    [compiledProfile({ id: "standard-v4" }), "HOOK_ABI_MISMATCH"],
    [compiledProfile({ unitSites: 1 }), "HOOK_ABI_MISMATCH"],
    [compiledProfile({ schemaVersion: 3 }), "HOOK_ABI_MISMATCH"],
  ]) {
    class MismatchedProfileEngine extends FixtureEngine {
      async getDrowseStructuredHookProfile() { return record; }
    }
    const mismatchedProfile = new DrowseWebLlmRuntime({
      DROWSE_HOOK_ABI: "block-sites-v5",
      MLCEngine: MismatchedProfileEngine,
    });
    await assert.rejects(mismatchedProfile.load(loadRequest()), (error) => error.code === code);
    assert.equal(MismatchedProfileEngine.instances.at(-1).unloaded, 1);
  }
  await assert.rejects(
    new DrowseWebLlmRuntime({ DROWSE_HOOK_ABI: "block-sites-v5", MLCEngine: FixtureEngine })
      .load(loadRequest({ variant: { ...variant, structuredHookProfile: "standard-v3" } })),
    (error) => error.code === "STRUCTURED_HOOK_PROFILE_MISMATCH",
  );

  class MissingProfileEngine extends FixtureEngine {}
  MissingProfileEngine.prototype.getDrowseStructuredHookProfile = undefined;
  const missingProfile = new DrowseWebLlmRuntime({
    DROWSE_HOOK_ABI: "block-sites-v5",
    MLCEngine: MissingProfileEngine,
  });
  await assert.rejects(
    missingProfile.load(loadRequest()),
    (error) => error.code === "STRUCTURED_HOOK_PROFILE_UNAVAILABLE",
  );

  await assert.rejects(
    runtime.load(loadRequest({
      adapter: { ...adapter, features: new Set() },
    })),
    (error) => error.code === "WEBGPU_FEATURE_UNAVAILABLE",
  );
  await assert.rejects(
    runtime.load(loadRequest({
      adapter: {
        ...adapter,
        limits: { maxStorageBufferBindingSize: 1024 },
      },
    })),
    (error) => error.code === "WEBGPU_LIMIT_TOO_LOW",
  );
  for (const name of [
    "maxComputeWorkgroupSizeX",
    "maxComputeInvocationsPerWorkgroup",
  ]) {
    const missingRequirements = variantWith().requirements;
    delete missingRequirements.limits[name];
    const engineCount = FixtureEngine.instances.length;
    await assert.rejects(
      runtime.load(loadRequest({
        variant: variantWith({ requirements: missingRequirements }),
        artifacts: [],
      })),
      (error) => error.code === "EXACT_READOUT_WEBGPU_REQUIREMENTS_INVALID",
    );
    assert.equal(
      FixtureEngine.instances.length,
      engineCount,
      "exact-readout requirements must fail before artifact preparation and engine creation",
    );

    const lowRequirements = variantWith().requirements;
    lowRequirements.limits[name] = 255;
    await assert.rejects(
      runtime.load(loadRequest({
        variant: variantWith({ requirements: lowRequirements }),
      })),
      (error) => error.code === "EXACT_READOUT_WEBGPU_REQUIREMENTS_INVALID",
    );

    await assert.rejects(
      runtime.load(loadRequest({
        variant: variantWith(),
        adapter: {
          ...adapter,
          limits: { ...adapter.limits, [name]: 255 },
        },
      })),
      (error) => error.code === "WEBGPU_LIMIT_TOO_LOW",
    );
  }
  await assert.rejects(
    runtime.load(loadRequest({
      adapter: {
        ...adapter,
        limits: {
          ...adapter.limits,
          maxStorageBuffersPerShaderStage: 9,
        },
      },
    })),
    (error) => error.code === "WEBGPU_LIMIT_TOO_LOW",
  );
  await assert.rejects(
    runtime.load(loadRequest({
      adapter: {
        ...adapter,
        limits: {
          ...adapter.limits,
          maxBufferSize: 8,
        },
      },
      optionalPacks: [{ pack: { kind: "jlens" } }],
    })),
    (error) => error.code === "JLENS_GPU_BUFFER_TOO_LARGE",
  );

  class MissingCaptureEngine extends FixtureEngine {
    capture = false;
  }
  const missingCapture = new DrowseWebLlmRuntime({
    DROWSE_HOOK_ABI: "block-sites-v5",
    MLCEngine: MissingCaptureEngine,
  });
  await assert.rejects(
    missingCapture.load(loadRequest()),
    (error) => error.code === "DROWSE_CAPTURE_UNAVAILABLE",
  );

  const controller = new AbortController();
  controller.abort(new DOMException("cancelled", "AbortError"));
  await assert.rejects(
    runtime.load(loadRequest({ signal: controller.signal })),
    (error) => error.name === "AbortError",
  );

  let loadingEngine = null;
  class InterruptibleLoadEngine extends FixtureEngine {
    constructor(config) {
      super(config);
      loadingEngine = this;
      this.reloadSettled = new Promise((resolve) => { this.resolveReload = resolve; });
    }
    async reload() { await this.reloadSettled; }
    async unload() {
      await super.unload();
      this.resolveReload();
    }
  }
  const loadingRuntime = new DrowseWebLlmRuntime({
    DROWSE_HOOK_ABI: "block-sites-v5",
    MLCEngine: InterruptibleLoadEngine,
  });
  const midLoadAbort = new AbortController();
  const pendingLoad = loadingRuntime.load(loadRequest({ signal: midLoadAbort.signal }));
  while (loadingEngine === null) await Promise.resolve();
  midLoadAbort.abort(new DOMException("cancelled during load", "AbortError"));
  await assert.rejects(pendingLoad, (error) => error.name === "AbortError");
  assert.equal(loadingEngine.unloaded, 1);
  assert.throws(
    () => loadingRuntime.requireEngine(),
    (error) => error.code === "MODEL_NOT_LOADED",
  );

  let checkingHooksEngine = null;
  class InterruptibleHookCheckEngine extends FixtureEngine {
    constructor(config) {
      super(config);
      checkingHooksEngine = this;
      this.hookCheckSettled = new Promise((resolve) => { this.resolveHookCheck = resolve; });
    }
    async supportsDrowseStructuredHooks() { return this.hookCheckSettled; }
    async unload() {
      await super.unload();
      this.resolveHookCheck(true);
    }
  }
  const hookCheckRuntime = new DrowseWebLlmRuntime({
    DROWSE_HOOK_ABI: "block-sites-v5",
    MLCEngine: InterruptibleHookCheckEngine,
  });
  const hookCheckAbort = new AbortController();
  const pendingHookCheck = hookCheckRuntime.load(
    loadRequest({ signal: hookCheckAbort.signal }),
  );
  while (checkingHooksEngine === null) await Promise.resolve();
  await Promise.resolve();
  hookCheckAbort.abort(new DOMException("cancelled during hook check", "AbortError"));
  await assert.rejects(pendingHookCheck, (error) => error.name === "AbortError");
  assert.equal(checkingHooksEngine.unloaded, 1);
  assert.throws(
    () => hookCheckRuntime.requireEngine(),
    (error) => error.code === "MODEL_NOT_LOADED",
  );

  class StructuredFixtureEngine extends FixtureEngine {
    async getDrowseStructuredHookProfile() { return compiledProfile(); }
    async supportsDrowseStructuredHooks() { return true; }
    async supportsDrowseCurvedHooks() { return true; }
    async setDrowseStructuredProgram(program) { this.hookProgram = program; }
    async updateDrowseStructuredControls() {}
    saeDictionarySets = 0;
    saeDictionaryClears = 0;
    jlensTopReads = 0;
    saeTopReads = 0;
    async setDrowseSaeDictionary(dictionary) {
      this.saeDictionarySets += 1;
      if (this.failNextSaeDictionarySet) {
        this.failNextSaeDictionarySet = false;
        throw new Error("SAE upload failed");
      }
      this.saeDictionary = dictionary;
    }
    async clearDrowseSaeDictionary() {
      this.saeDictionary = null;
      this.saeDictionaryClears += 1;
    }
    async readDrowseJlensTopTokens() {
      this.jlensTopReads += 1;
      return this.jlensTop;
    }
    async readDrowseSaeTopFeatures() {
      this.saeTopReads += 1;
      return this.saeTop;
    }
  }
  class MissingStructuredPackEngine extends FixtureEngine {
    async getDrowseStructuredHookProfile() { return compiledProfile(); }
  }
  class MissingBundleEngine extends StructuredFixtureEngine {
    readDrowseMeasurementBundle = undefined;
  }
  const missingBundleRuntime = new DrowseWebLlmRuntime({
    DROWSE_HOOK_ABI: "block-sites-v5",
    MLCEngine: MissingBundleEngine,
  });
  await assert.rejects(
    missingBundleRuntime.load(loadRequest({ variant: variantWith() })),
    (error) => error.code === "BUNDLED_MEASUREMENT_READBACK_UNAVAILABLE",
  );
  const missingStructuredPackRuntime = new DrowseWebLlmRuntime({
    DROWSE_HOOK_ABI: "block-sites-v5",
    MLCEngine: MissingStructuredPackEngine,
  });
  await assert.rejects(
    missingStructuredPackRuntime.load(loadRequest({
      variant: variantWith(),
      optionalPacks: [{ pack: { kind: "sae" }, artifacts: [] }],
    })),
    (error) => error.code === "EXACT_SAE_READOUT_UNAVAILABLE",
  );
  const instrumentRuntime = new DrowseWebLlmRuntime({
    DROWSE_HOOK_ABI: "block-sites-v5",
    MLCEngine: StructuredFixtureEngine,
  });
  await instrumentRuntime.load(loadRequest({
    variant: variantWith(),
    optionalPacks: [
      { pack: { kind: "sae" }, artifacts: [] },
      { pack: { kind: "jlens" }, artifacts: [] },
    ],
  }));
  const instrumentEngine = instrumentRuntime.requireEngine();

  assert.deepEqual(instrumentRuntime.hookCapabilities(), {
    rank_one: true,
    multi_term: true,
    projection: true,
    ablation: true,
    curved_manifold: true,
    phase_trigger: true,
    probe_gate: true,
    sae: true,
    jlens: true,
    slens: false,
    units: false,
  });
  assert.doesNotThrow(() => instrumentRuntime.assertSteeringSupported(
    "0.5 sae/1@response + 0.25 jlens/hello",
  ));
  await instrumentRuntime.setSaeDictionary({
    bindingId: "d".repeat(64),
    hiddenSize: 2,
    runtimeLayerIndex: 1,
    featureCount: 3,
    encoder: Float32Array.from([1, 0, 0, 0, 1, 0]),
    encoderBias: Float32Array.from([0, 0, 0]),
    decoderBias: Float32Array.from([0, 0]),
  });
  assert.equal(instrumentEngine.saeDictionarySets, 0);
  assert.equal(instrumentEngine.saeDictionaryClears, 0);
  assert.equal(instrumentEngine.saeDictionary, undefined);

  await instrumentRuntime.setJlensDictionary({
    bindingId: "e".repeat(64),
    hiddenSize: 2,
    layerIndices: Int32Array.from([0, 1]),
    matrices: [new Float32Array(4), new Float32Array(4)],
  });
  assert.equal(instrumentEngine.jlensDictionarySets, 0);
  assert.equal(instrumentEngine.jlensDictionaryClears, 0);
  instrumentEngine.failNextJlensDictionarySet = true;
  await assert.rejects(
    instrumentRuntime.resolveJlensTokenDirections("e".repeat(64), [0, 1], [7]),
    /J-lens upload failed/,
  );
  assert.equal(instrumentEngine.jlensDictionarySets, 1);
  assert.deepEqual(
    [...await instrumentRuntime.resolveJlensTokenDirections(
      "e".repeat(64),
      [0, 1],
      [7],
    )],
    [7, -7, 7, -7],
  );
  assert.equal(instrumentEngine.jlensDictionarySets, 2);

  const noFullSaeProgram = compileStructuredHookProgram(
    2,
    [{}, {}],
    STANDARD_STRUCTURED_HOOK_PROFILE_V4_RESIDUAL,
  );
  noFullSaeProgram.measurementSchema = {
    layerMap: [0, 1],
    modelLayerCount: 2,
    probes: new Array(8).fill(null),
    lensSource: null,
    saeSource: "fixture-sae",
    saeLayer: 1,
    saeFeatureCount: 3,
    saeReadout: false,
  };
  const fullSaeProgram = compileStructuredHookProgram(
    2,
    [{}, {}],
    STANDARD_STRUCTURED_HOOK_PROFILE_V4_RESIDUAL,
  );
  fullSaeProgram.measurementSchema = {
    ...noFullSaeProgram.measurementSchema,
    saeReadout: true,
  };
  fullSaeProgram.saeBindingId = "d".repeat(64);
  await instrumentRuntime.installHookProgram(noFullSaeProgram);
  assert.equal(instrumentEngine.saeDictionarySets, 0);
  assert.equal(instrumentEngine.saeDictionaryClears, 0);
  instrumentEngine.failNextSaeDictionarySet = true;
  await assert.rejects(
    instrumentRuntime.installHookProgram(fullSaeProgram),
    /SAE upload failed/,
  );
  assert.equal(instrumentEngine.saeDictionarySets, 1);
  await instrumentRuntime.installHookProgram(fullSaeProgram);
  assert.equal(instrumentEngine.saeDictionarySets, 2);
  assert.equal(instrumentEngine.saeDictionary.hookAbi, "block-sites-v5");
  await instrumentRuntime.installHookProgram(noFullSaeProgram);
  assert.equal(instrumentEngine.saeDictionaryClears, 0);
  assert.notEqual(instrumentEngine.saeDictionary, null);
  await instrumentRuntime.installHookProgram(fullSaeProgram);
  assert.equal(instrumentEngine.saeDictionarySets, 2);
  await instrumentRuntime.clearHookProgram();
  assert.equal(instrumentEngine.saeDictionaryClears, 0);
  assert.notEqual(instrumentEngine.saeDictionary, null);
  assert.equal(instrumentEngine.jlensDictionaryClears, 0);
  instrumentEngine.jlensTop = {
    tokenIds: Int32Array.from([65, 66, 67, 68, 69, 70, 71, 72]),
    strength: Float32Array.from([0.2, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13]),
    centerOfMass: Float32Array.from([0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.1]),
    spread: Float32Array.from([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
    fittedLayerCount: 3,
    layerIndices: Int32Array.from([0, 1, 2]),
    layerTokenIds: Int32Array.from([
      65, 66, 67, 68, 69, 70, 71, 72,
      73, 74, 75, 76, 77, 78, 79, 80,
      73, 74, 75, 76, 77, 78, 79, 80,
    ]),
    layerProbabilities: Float32Array.from([
      0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.06, 0.04,
      0.35, 0.25, 0.18, 0.14, 0.1, 0.07, 0.05, 0.03,
      0.32, 0.24, 0.17, 0.13, 0.09, 0.07, 0.05, 0.03,
    ]),
  };
  instrumentEngine.saeTop = {
    featureIds: Int32Array.from([2, 1]),
    activations: Float32Array.from([4, 1]),
    runtimeLayerIndex: 1,
    featureCount: 3,
  };
  const jlensTop3 = await instrumentRuntime.readJlensTopTokens(3);
  const saeTop1 = await instrumentRuntime.readSaeTopFeatures(1);
  assert.deepEqual(jlensTop3.tokens, ["A", "B", "C"]);
  assert.deepEqual(jlensTop3.layerTokens, ["A", "B", "C", "I", "J", "K", "I", "J", "K"]);
  assert.equal(instrumentEngine.decodedTokenCalls.length, 6);
  assert.deepEqual([...saeTop1.featureIds], [2]);
  const jlensTop = await instrumentRuntime.readJlensTopTokens(8);
  const saeTop = await instrumentRuntime.readSaeTopFeatures(8);
  assert.deepEqual(jlensTop.tokens, ["A", "B", "C", "D", "E", "F", "G", "H"]);
  assert.deepEqual(jlensTop.layerTokens.slice(0, 3), ["A", "B", "C"]);
  assert.equal(instrumentEngine.decodedTokenCalls.length, 16);
  assert.deepEqual([...saeTop.featureIds], [2, 1]);
  jlensTop.strength[0] = 0;
  saeTop.activations[0] = 0;
  assert.ok(Math.abs(instrumentEngine.jlensTop.strength[0] - 0.2) < 1e-6);
  assert.equal(instrumentEngine.saeTop.activations[0], 4);
  assert.equal(WEB_LLM_DECODED_TOKEN_CACHE_MAX % 8, 0);
  for (let frame = 0; frame < WEB_LLM_DECODED_TOKEN_CACHE_MAX / 8; frame += 1) {
    const ids = Int32Array.from(
      { length: 8 },
      (_, index) => 1_000 + frame * 8 + index,
    );
    instrumentEngine.jlensTop = {
      tokenIds: ids,
      strength: Float32Array.from({ length: 8 }, () => 0.1),
      centerOfMass: Float32Array.from({ length: 8 }, () => 0.5),
      spread: Float32Array.from({ length: 8 }, () => 0.25),
      fittedLayerCount: 1,
      layerIndices: Int32Array.from([0]),
      layerTokenIds: ids,
      layerProbabilities: Float32Array.from({ length: 8 }, () => 0.1),
    };
    await instrumentRuntime.readJlensTopTokens(8);
  }
  const decodeCallsBeforeEvictedRead = instrumentEngine.decodedTokenCalls.length;
  instrumentEngine.jlensTop = {
    tokenIds: Int32Array.from({ length: 8 }, () => 65),
    strength: Float32Array.from({ length: 8 }, () => 0.1),
    centerOfMass: Float32Array.from({ length: 8 }, () => 0.5),
    spread: Float32Array.from({ length: 8 }, () => 0.25),
    fittedLayerCount: 1,
    layerIndices: Int32Array.from([0]),
    layerTokenIds: Int32Array.from({ length: 8 }, () => 65),
    layerProbabilities: Float32Array.from({ length: 8 }, () => 0.1),
  };
  await instrumentRuntime.readJlensTopTokens(8);
  assert.equal(instrumentEngine.decodedTokenCalls.length, decodeCallsBeforeEvictedRead + 1);
  const jlensReadsBeforeCapacityFailure = instrumentEngine.jlensTopReads;
  const saeReadsBeforeCapacityFailure = instrumentEngine.saeTopReads;
  await assert.rejects(
    instrumentRuntime.readJlensTopTokens(9),
    (error) => error.code === "EXACT_READOUT_TOP_K_EXCEEDS_CAPACITY",
  );
  await assert.rejects(
    instrumentRuntime.readSaeTopFeatures(9),
    (error) => error.code === "EXACT_READOUT_TOP_K_EXCEEDS_CAPACITY",
  );
  assert.equal(instrumentEngine.jlensTopReads, jlensReadsBeforeCapacityFailure);
  assert.equal(instrumentEngine.saeTopReads, saeReadsBeforeCapacityFailure);
  instrumentEngine.saeTop = {
    ...instrumentEngine.saeTop,
    runtimeLayerIndex: 0,
  };
  await assert.rejects(
    instrumentRuntime.readSaeTopFeatures(),
    (error) => error.code === "INVALID_SAE_TOP_READOUT",
  );
  assert.equal(instrumentEngine.saeDictionarySets, 2);
  await instrumentRuntime.setSaeDictionary(null);
  assert.equal(instrumentEngine.saeDictionaryClears, 1);
  assert.equal(instrumentEngine.saeDictionary, null);
  await instrumentRuntime.setJlensDictionary(null);
  assert.equal(instrumentEngine.jlensDictionaryClears, 1);
  assert.equal(instrumentEngine.jlensDictionary, null);

  const aggregateProgram = compileStructuredHookProgram(2, [
    { probes: [{ kind: "sae", direction: [1, 0], bias: 0 }] },
    {},
  ], STANDARD_STRUCTURED_HOOK_PROFILE_V4_RESIDUAL);
  aggregateProgram.measurementSchema = {
    layerMap: [0, 1],
    modelLayerCount: 2,
    probes: [
      { name: "sae/0", family: "sae", featureId: 0 },
      null, null, null, null, null, null, null,
    ],
    lensSource: null,
    saeSource: "fixture-sae",
  };
  instrumentEngine.chat = {
    completions: {
      async create() {
        return (async function* () {
          instrumentEngine.measurements = Float32Array.from([
            0.25, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
          ]);
          yield {
            choices: [{
              index: 0,
              delta: { content: "answer" },
              finish_reason: null,
              logprobs: { content: [{
                token: "answer",
                token_id: 7,
                logprob: -0.1,
                drowse_sampler: {
                  entropy_nats: 0.2,
                  perplexity: Math.exp(0.2),
                },
              }] },
            }],
          };
          instrumentEngine.measurements = Float32Array.from([
            0.9, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
          ]);
          yield {
            choices: [{
              index: 0,
              delta: { content: "" },
              finish_reason: "stop",
              drowse_finish_reason: "eos",
              logprobs: { content: [{
                token: "",
                token_id: 9,
                logprob: -0.2,
                drowse_sampler: {
                  entropy_nats: 0.3,
                  perplexity: Math.exp(0.3),
                },
              }] },
            }],
          };
          yield {
            choices: [],
            usage: {
              prompt_tokens: 1,
              completion_tokens: 1,
              total_tokens: 2,
              extra: {
                prefill_tokens_per_s: 10,
                decode_tokens_per_s: 5,
              },
            },
          };
        })();
      },
    },
  };
  const verifiedAggregate = await instrumentRuntime.streamGeneration(
    {
      input: { kind: "chat", messages: [{ role: "user", content: "hello" }] },
      hookProgram: aggregateProgram,
      measurementSpecialTokenIds: [7],
    },
    () => {},
  );
  assert.equal(verifiedAggregate.measurements.scores["sae/0"], 0.25);
  await instrumentRuntime.unload();

  const productionLayerCount = 28;
  const productionLayerMap = Array.from(
    { length: productionLayerCount },
    (_, index) => index,
  );
  const productionVariant = variantWith({
    runtimeIdentity: { ...variant.runtimeIdentity, layerMap: productionLayerMap },
  });
  class ProductionJlensFixtureEngine extends StructuredFixtureEngine {
    events = [];
    async getDrowseStructuredHookProfile() {
      return compiledProfile({}, productionVariant);
    }
    async setDrowseJlensDictionary(dictionary) {
      this.events.push(`dictionary:${dictionary.layerIndices.length}`);
      await super.setDrowseJlensDictionary(dictionary);
    }
    async setDrowseStructuredProgram(program) {
      assert.equal(this.jlensDictionary?.bindingId, program.jLensBindingId);
      this.events.push(`program:${program.jLensLayerIndices?.length ?? 0}`);
      await super.setDrowseStructuredProgram(program);
    }
  }
  const productionJlensRuntime = new DrowseWebLlmRuntime({
    DROWSE_HOOK_ABI: "block-sites-v5",
    MLCEngine: ProductionJlensFixtureEngine,
  });
  await productionJlensRuntime.load(loadRequest({ variant: productionVariant }));
  const productionJlensBinding = "f".repeat(64);
  await productionJlensRuntime.setJlensDictionary({
    bindingId: productionJlensBinding,
    hiddenSize: 2,
    layerIndices: Int32Array.from(productionLayerMap),
    matrices: Array.from(
      { length: productionLayerCount },
      () => new Float32Array(4),
    ),
  });
  const productionJlensProgram = compileStructuredHookProgram(
    2,
    Array.from({ length: productionLayerCount }, () => ({
      probes: Array.from({ length: 8 }, () => ({
        kind: "jlens",
        direction: [0, 0],
        bias: 0,
      })),
    })),
    STANDARD_STRUCTURED_HOOK_PROFILE_V4_RESIDUAL,
    {
      bindingId: productionJlensBinding,
      layerIndices: Int32Array.from(productionLayerMap),
      tokenIds: new Int32Array(8),
    },
  );
  const productionJlensEngine = productionJlensRuntime.requireEngine();
  assert.deepEqual(productionJlensEngine.events, []);
  await productionJlensRuntime.warmJlensDictionary();
  assert.deepEqual(productionJlensEngine.events, ["dictionary:28"]);
  await productionJlensRuntime.installHookProgram(productionJlensProgram);
  assert.deepEqual(productionJlensEngine.events, [
    "dictionary:28",
    "program:28",
  ]);
  await productionJlensRuntime.unload();

  const warmSaeRuntime = new DrowseWebLlmRuntime({
    DROWSE_HOOK_ABI: "block-sites-v5",
    MLCEngine: StructuredFixtureEngine,
  });
  await warmSaeRuntime.load(loadRequest({
    variant: variantWith(),
    optionalPacks: [{ pack: { kind: "sae" }, artifacts: [] }],
  }));
  await warmSaeRuntime.setSaeDictionary({
    bindingId: "a".repeat(64),
    hiddenSize: 2,
    runtimeLayerIndex: 1,
    featureCount: 3,
    encoder: Float32Array.from([1, 0, 0, 0, 1, 0]),
    encoderBias: Float32Array.from([0, 0, 0]),
    decoderBias: Float32Array.from([0, 0]),
  });
  const warmSaeEngine = warmSaeRuntime.requireEngine();
  assert.equal(warmSaeEngine.saeDictionarySets, 0);
  await warmSaeRuntime.warmSaeDictionary();
  assert.equal(warmSaeEngine.saeDictionarySets, 1);
  await warmSaeRuntime.warmSaeDictionary();
  assert.equal(warmSaeEngine.saeDictionarySets, 1);
  await warmSaeRuntime.unload();

  console.log("WebLLM engine adapter checks passed");
} finally {
  await server.close();
}

async function* successfulGenerationStream(terminalReason = "eos") {
  yield {
    choices: [{
      index: 0,
      delta: { content: "answer" },
      finish_reason: null,
      logprobs: { content: [{
        token: "answer",
        token_id: 7,
        logprob: -0.25,
        top_logprobs: [],
        drowse_sampler: {
          entropy_nats: 0.5,
          perplexity: Math.exp(0.5),
        },
      }] },
    }],
  };
  yield {
    choices: [{
      index: 0,
      delta: {},
      finish_reason: terminalReason === "length" ? "length" : "stop",
      drowse_finish_reason: terminalReason,
      logprobs: null,
    }],
  };
  yield {
    choices: [],
    usage: {
      prompt_tokens: 2,
      completion_tokens: 1,
      total_tokens: 3,
      extra: {
        prefill_tokens_per_s: 18,
        decode_tokens_per_s: 7.5,
      },
    },
  };
}
