// Hook ABI v5 in the WebLLM adapter (webLlmEngine.ts, webLlmBlockSites.ts, webLlmGeneration.ts):
// the schema-4 descriptor attested element by element against the lock, the block-sites
// capabilities a load yields, the bridge surface (unit programs bound on a units library, refused
// on a residual one; the replay, residents and attention rows over a recording bridge), and live
// unit taps read as gate scores.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const goldens = new URL("../../drowse/tests/fixtures/browser/", import.meta.url);
const server = await createServer({
  root,
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});

try {
  const [engine, backendModule, generation, profiles, programs, modelBackend] = await Promise.all([
    server.ssrLoadModule("/src/hosted/runtime/webLlmEngine.ts"),
    server.ssrLoadModule("/src/hosted/runtime/webLlmBlockSites.ts"),
    server.ssrLoadModule("/src/hosted/runtime/webLlmGeneration.ts"),
    server.ssrLoadModule("/src/hosted/runtime/structuredHookProfile.ts"),
    server.ssrLoadModule("/src/hosted/runtime/structuredHookProgram.ts"),
    server.ssrLoadModule("/src/hosted/runtime/modelBackend.ts"),
  ]);
  const tiny = JSON.parse(readFileSync(new URL("units-tiny-models-v1.json", goldens), "utf8")).arches;
  const qwen = tiny.qwen3.config;
  const spec = profiles.validateBlockSpec({
    dFF: qwen.intermediate_size,
    nHeads: qwen.num_attention_heads,
    nKvHeads: qwen.num_key_value_heads,
    headDim: qwen.head_dim,
    sandwich: false,
    normOffset: 0,
    activation: "silu",
    embedScale: 1,
    rmsEps: qwen.rms_norm_eps,
    qkNorm: true,
    attnScaling: qwen.head_dim ** -0.5,
    attnLogitSoftcap: null,
    finalLogitSoftcap: null,
    ropeTheta: qwen.rope_parameters.rope_theta,
    ropeLocalTheta: null,
    ropeGlobalScale: 1,
    ropeInGraph: false,
    slidingWindow: null,
    slidingPattern: null,
  });
  const variant = (profile) => ({
    id: "tiny",
    structuredHookProfile: profile,
    contextProfiles: [{ contextTokens: 128 }, { contextTokens: 64 }],
    runtimeIdentity: { layerMap: [0, 1, 2, 3], hiddenSize: 64 },
  });
  const request = (profile, extra = {}) => ({ variant: variant(profile), ...extra });

  // ------------------------------------------------ descriptor model and attestation ---
  const model = engine.hookProfileDescriptorModel(request("standard-v4", { blockSpec: spec }));
  assert.deepEqual(model, {
    layerCount: 4, hiddenSize: 64, contextTokens: 128, prefillChunkSize: 64, blockSpec: spec,
  });
  const v4 = profiles.STANDARD_STRUCTURED_HOOK_PROFILE_V4;
  const residual = profiles.STANDARD_STRUCTURED_HOOK_PROFILE_V4_RESIDUAL;
  const descriptor = Array.from(profiles.expectedHookProfileDescriptor(v4, model));
  const bridgeProfile = (overrides = {}) => ({
    schemaVersion: 4,
    id: "standard-v4",
    unitSites: 1,
    descriptor: [...descriptor],
    hookAbi: "block-sites-v5",
    format: "drowse-structured-v4",
    ...overrides,
  });
  const attest = (value, extra = { blockSpec: spec }) =>
    engine.attestCompiledStructuredHookProfile(value, request("standard-v4", extra), "block-sites-v5");
  const attested = attest(bridgeProfile());
  assert.equal(attested.schemaVersion, 4);
  assert.equal(attested.unitSites, 1);
  assert.equal(attested.format, "drowse-structured-v4");
  assert.equal(attested.exactReadoutAbi, "exact-readout-v2");
  assert.deepEqual(Array.from(attested.descriptor), descriptor);
  const fieldIndex = (name) => profiles.HOOK_PROFILE_DESCRIPTOR_V4_FIELDS.indexOf(name);
  for (const field of ["ROPE_THETA_F32_BITS", "SANDWICH", "MAX_UNIT_TAPS", "CONTEXT_TOKENS"]) {
    const changed = [...descriptor];
    changed[fieldIndex(field)] += 1;
    assert.throws(
      () => attest(bridgeProfile({ descriptor: changed })),
      (error) => error.code === "HOOK_ABI_MISMATCH" && error.message.includes(field),
      field,
    );
  }
  assert.throws(
    () => attest(bridgeProfile({ descriptor: descriptor.slice(0, 31) })),
    (error) => error.code === "HOOK_ABI_MISMATCH" && /LENGTH/.test(error.message),
  );
  assert.throws(() => attest(bridgeProfile({ unitSites: 0 })), (error) => error.code === "HOOK_ABI_MISMATCH");
  assert.throws(
    () => engine.attestCompiledStructuredHookProfile(bridgeProfile(), request("standard-v4", { blockSpec: spec }), "post-block-residual-v4"),
    (error) => error.code === "HOOK_ABI_MISMATCH",
  );
  // A standard-v4 library cannot be attested without the lock's block spec.
  assert.throws(() => attest(bridgeProfile(), {}), (error) =>
    error.code === "HOOK_ABI_MISMATCH" && /block spec/.test(error.message));
  // A residual library attests the residual descriptor with no block spec.
  const residualModel = { ...model, blockSpec: null };
  const residualDescriptor = Array.from(profiles.expectedHookProfileDescriptor(residual, residualModel));
  const residualProfile = engine.attestCompiledStructuredHookProfile(
    bridgeProfile({ id: "standard-v4-residual", unitSites: 0, descriptor: residualDescriptor }),
    request("standard-v4-residual"),
    "block-sites-v5",
  );
  assert.equal(residualProfile.unitSites, 0);

  // ---------------------------------------------------------------- capabilities ---
  const capabilities = engine.blockSitesCapabilitiesFor(attested, request("standard-v4", { blockSpec: spec }), 259);
  assert.equal(capabilities.profile.id, "standard-v4");
  assert.equal(capabilities.vocabSize, 259);
  assert.equal(capabilities.contextTokens, 128);
  assert.equal(capabilities.prefillChunkSize, 64);
  assert.equal(capabilities.blockSpec, spec);
  assert.equal(capabilities.unitRefusal, null);
  const refusal = "drowse units need a dense gated MLP block; gpt2 fuses q/k/v";
  const residualCapabilities = engine.blockSitesCapabilitiesFor(
    residualProfile,
    request("standard-v4-residual", { unitRefusal: refusal }),
    259,
  );
  assert.equal(residualCapabilities.unitRefusal, refusal);
  assert.equal(engine.blockSitesCapabilitiesFor(residualProfile, request("standard-v4-residual"), 259), null);
  assert.equal(engine.blockSitesCapabilitiesFor(null, request("standard-v4"), 259), null);

  // --------------------------------------------------------------- bridge surface ---
  const surface = backendModule.webLlmBlockSitesBackend(capabilities);
  assert.equal(surface.capabilities(), capabilities);
  const plain = { format: "drowse-structured-v4", hookAbi: "block-sites-v5", affineActive: new Uint32Array(16) };
  assert.equal(await surface.bindStructuredProgram(plain), plain);
  const layout = programs.structuredUnitSectionLayout(v4, 4, spec.headDim);
  const unitProgram = () => ({
    ...plain,
    unitMeta: new Uint32Array(layout.metaLength),
    unitCoeff: new Float32Array(layout.coeffLength),
    unitActive: new Float32Array(layout.activeLength),
    tapGrams: new Float32Array(layout.tapGramsLength),
  });
  const ablation = unitProgram();
  assert.equal(await surface.bindStructuredProgram(ablation), ablation);
  // One tapped head (layer 2, head 3): its gram comes from the bridge's resident head grams, so
  // with no bridge the program cannot be bound.
  const tapped = unitProgram();
  const offsets = layout.meta.tapHeadOffsets.offset;
  for (let layer = 3; layer <= 4; layer += 1) tapped.unitMeta[offsets + layer] = 1;
  tapped.unitMeta[layout.meta.tapHeadIndex.offset] = 3;
  await assert.rejects(surface.bindStructuredProgram(tapped), (error) => error.code === "BLOCK_SITES_MEMBER_UNAVAILABLE");
  const short = unitProgram();
  short.unitCoeff = new Float32Array(layout.coeffLength - 1);
  await assert.rejects(surface.bindStructuredProgram(short), TypeError);
  await assert.rejects(surface.openReplay({}), (error) => error.code === "BLOCK_SITES_MEMBER_UNAVAILABLE");
  const residualSurface = backendModule.webLlmBlockSitesBackend(residualCapabilities);
  await assert.rejects(residualSurface.bindStructuredProgram(unitProgram()), (error) =>
    error instanceof modelBackend.UnsupportedBlockError && error.message === refusal);
  await assert.rejects(residualSurface.unitWeightRows("neuron_write", []), modelBackend.UnsupportedBlockError);
  await assert.rejects(residualSurface.logSumExp(modelBackend.f32Tensor(new Float32Array(2), [1, 2])),
    (error) => error.code === "BLOCK_SITES_MEMBER_UNAVAILABLE");
  assert.equal(await residualSurface.bindStructuredProgram(plain), plain);

  // ------------------------------------------------- replay over a recording bridge ---
  const sent = [];
  let installed = null;
  const L = 4;
  const H = spec.nHeads;
  const anatomyStride = 6 * qwen.hidden_size + spec.dFF + H * spec.headDim;
  const device = {
    installProgram: async (program) => { installed = program; },
    call: async (request) => {
      sent.push(request);
      switch (request.op) {
        case "headGrams": return { id: "head_grams:1", kind: "head_grams", shape: [L, H, spec.headDim, spec.headDim], bytes: 4 };
        case "uploadUnitScales": return { id: "unit_scales:2", kind: "unit_scales", shape: [L, 2 * spec.dFF + H], bytes: 4 };
        case "openReplay": return null;
        case "replayForward": return { measurements: new Float32Array(L * 8).fill(0.5), unitTaps: new Float32Array(64) };
        case "finishReplay": return {
          anatomyTaps: new Float32Array(L * anatomyStride),
          activityState: new Float32Array(L * (3 * spec.dFF + 3 * H)),
          traces: Float32Array.from({ length: 32 * 3 }, (_, i) => i),
          population: new Float32Array(L * (spec.dFF + H)),
          patterns: { id: "pattern_capture:3", kind: "pattern_capture", shape: [8, spec.nKvHeads, 10, spec.headDim], bytes: 4 },
        };
        case "attentionRows": return Float32Array.from({ length: 16 * 10 }, (_, i) => i);
        default: return null;
      }
    },
  };
  const live = backendModule.webLlmBlockSitesBackend(capabilities, device);
  const grams = await live.residentHeadGrams();
  assert.equal(grams.kind, "head_grams");
  await assert.rejects(live.release(grams), TypeError);
  const scales = await live.uploadUnitScales(modelBackend.f32Tensor(new Float32Array(L * (2 * spec.dFF + H)).fill(1), [L, 2 * spec.dFF + H]));
  await assert.rejects(live.uploadUnitScales(modelBackend.f32Tensor(new Float32Array(3), [3])), TypeError);
  const replay = await live.openReplay({
    program: null,
    anatomyPosition: 8,
    activity: {
      spans: [{ firstRow: 0, rows: 3, firstPosition: 6 }], rowCount: 3, scales,
      traces: [{ kind: "neuron", layer: 1, index: 5 }], populationRow: 1,
    },
    patternLayers: [1, 3],
    tokenCount: 10,
  });
  // An unsteered replay installs an empty program with an empty unit section (the units superset).
  assert.equal(installed.unitMeta.length, layout.metaLength);
  assert.ok(installed.unitMeta.every((value) => value === 0));
  const opened = sent.find((request) => request.op === "openReplay");
  assert.equal(opened.unitScales, "unit_scales:2");
  assert.equal(opened.rowCount, 3);
  assert.equal(opened.tokenCount, 10);
  const metaLayout = modelBackend.replayMetaLayout(v4);
  assert.equal(opened.replayMeta[metaLayout.anatomyPosition], 8);
  assert.equal(opened.replayMeta[metaLayout.flags], 7);
  await assert.rejects(live.openReplay({ program: null, anatomyPosition: null, activity: null, patternLayers: null }),
    (error) => error.code === "REPLAY_BUSY");
  const controls = { affineActive: new Uint32Array(16), curveActive: new Uint32Array(16), unitActive: Float32Array.of(1, 0, 0, 0) };
  const readback = await replay.forward({ tokenIds: Int32Array.of(1, 2, 3), controls });
  assert.deepEqual(readback.measurements.shape, [L, 8]);
  assert.deepEqual(readback.unitTaps.shape, [64]);
  assert.deepEqual(sent.at(-1).tokenIds, [1, 2, 3]);
  assert.deepEqual([...sent.at(-1).unitActive], [1, 0, 0, 0]);
  const result = await replay.finish();
  assert.deepEqual(result.anatomyTaps.shape, [L, anatomyStride]);
  assert.deepEqual(result.activity.traces.shape, [1, 3]);
  assert.deepEqual([...result.activity.traces.data], [0, 1, 2]);
  assert.deepEqual(result.activity.population.shape, [L, spec.dFF + H]);
  assert.equal(result.patterns.kind, "pattern_capture");
  const rows = await live.attentionRows(result.patterns, [{ layer: 3, head: 1 }, { layer: 1, head: 0 }]);
  assert.deepEqual(rows.shape, [2, 9]);
  assert.deepEqual([...rows.data.subarray(0, 3)], [0, 1, 2]);
  assert.deepEqual([...rows.data.subarray(9, 12)], [10, 11, 12]);
  const asked = sent.at(-1).rows;
  assert.deepEqual([...asked.subarray(0, 8)], [1, 3, 1, 8, 0, 1, 0, 8]);
  assert.ok(asked.subarray(8).every((value) => value === -1));
  await assert.rejects(live.attentionRows(result.patterns, [{ layer: 2, head: 0 }]), RangeError);
  await live.release(result.patterns);
  await assert.rejects(live.attentionRows(result.patterns, [{ layer: 1, head: 0 }]), TypeError);
  await assert.rejects(live.openReplay({ program: null, anatomyPosition: null, activity: null, patternLayers: [1] }), RangeError);

  // -------------------------------------------- static reads over a computing bridge ---
  await staticReads(backendModule, modelBackend, capabilities, residualCapabilities, spec, tapped);

  // ---------------------------------------------------------- unit taps as gate scores ---
  const tapSchema = [
    { name: "neuron/L1n5", unit: "neuron/L1n5", kind: "neuron", layer: 1, index: 5, slot: 0,
      scale: { source: "neutral", detail: "", top: [1, 1], value: [5.061468, 3.3126411] } },
    { name: "head/L2h1", unit: "head/L2h1", kind: "head", layer: 2, index: 1, slot: 1,
      scale: { source: "neutral", detail: "", top: 1, value: 7.324882 } },
    null,
  ];
  const program = {
    format: "drowse-structured-v4",
    affineActive: new Uint32Array(16),
    measurementSchema: { probes: [], unitTaps: tapSchema },
  };
  // unit-probes-gemma3-v1.json, unsteered forward 0: raw −0.97013634 → −0.29285887 (− pole),
  // write norm 4.0678535 → 0.5553473; the ablated neuron's 0.21354148 → 0.04218963 (+ pole).
  const scores = generation.unitTapReadings(program, Float32Array.of(-0.97013634, 4.0678535, 0));
  assert.ok(Math.abs(scores["neuron/L1n5"] - -0.29285887) < 1e-6);
  assert.ok(Math.abs(scores["head/L2h1"] - 0.5553473) < 1e-6);
  const positive = generation.unitTapReadings(program, Float32Array.of(0.21354148, 4.0678535, 0));
  assert.ok(Math.abs(positive["neuron/L1n5"] - 0.04218963) < 1e-6);
  assert.equal(generation.unitTapReadings(program, undefined), undefined);
  assert.throws(() => generation.unitTapReadings(program, Float32Array.of(Number.NaN, 1)), /finite tap/);

  // ------------------------------------------------- controls reach the bridge whole ---
  const runtime = new engine.DrowseWebLlmRuntime({});
  const calls = [];
  runtime.engine = { updateDrowseStructuredControls: async (...args) => { calls.push(args); } };
  await runtime.updateStructuredControls(new Uint32Array(4), new Uint32Array(4), Float32Array.of(1, 0, 0, 0));
  assert.deepEqual(calls[0].slice(2).map((value) => value === undefined ? value : [...value]), [undefined, [1, 0, 0, 0]]);
  console.log("WebLLM block-sites adapter: descriptor attestation, capabilities, surface and unit taps passed");
} finally {
  await server.close();
}


/** A bridge that runs the weight-free reads in JS and answers the weight reads with tagged
 *  values, so each member's request and its result assembly are checked. */
function computingBridge(log) {
  const LOWEST = -3.4028234663852886e38;
  const size = (shape) => shape.reduce((a, b) => a * b, 1);
  const run = (fn, a) => {
    const [x, y] = a;
    switch (fn) {
      case "drowse_neuron_rows": {
        const n = x.shape[0];
        const d = 64;
        return { shape: [n, 3, d], data: Float32Array.from({ length: n * 3 * d }, (_, i) =>
          1000 * Math.floor(i / (3 * d)) + 100 * (Math.floor(i / d) % 3) + (i % d)) };
      }
      case "drowse_head_rows": {
        const n = x.shape[0];
        return { shape: [n, 4, 16, 64], data: Float32Array.from({ length: n * 4 * 16 * 64 }, (_, i) => i) };
      }
      case "drowse_gather_head_grams":
        return { shape: [y.shape[0], 16, 16], data: Float32Array.from({ length: y.shape[0] * 256 }, (_, i) =>
          y.data[2 * Math.floor(i / 256)] * 10 + y.data[2 * Math.floor(i / 256) + 1]) };
      case "drowse_batched_matmul":
      case "drowse_batched_matmul_nt": {
        const [b, m, k] = x.shape;
        const nt = fn.endsWith("_nt");
        const n = nt ? y.shape[1] : y.shape[2];
        const out = new Float32Array(b * m * n);
        for (let i = 0; i < b; i += 1) for (let r = 0; r < m; r += 1) for (let c = 0; c < n; c += 1) {
          let acc = 0;
          const w = y.shape[0] === 1 ? 0 : i;
          for (let j = 0; j < k; j += 1) acc += x.data[(i * m + r) * k + j] * (nt ? y.data[(w * n + c) * k + j] : y.data[(w * k + j) * n + c]);
          out[(i * m + r) * n + c] = acc;
        }
        return { shape: [b, m, n], data: out };
      }
      case "drowse_log_sum_exp": {
        const [n, m] = x.shape;
        return { shape: [n], data: Float32Array.from({ length: n }, (_, r) => {
          const row = x.data.subarray(r * m, (r + 1) * m);
          const top = Math.max(...row);
          return top + Math.log(row.reduce((s, v) => s + Math.exp(v - top), 0));
        }) };
      }
      case "drowse_topk": {
        const [n, m] = x.shape;
        const k = y.shape[1];
        const values = new Float32Array(n * k);
        const ids = new Int32Array(n * k);
        for (let r = 0; r < n; r += 1) {
          const order = [...Array(m).keys()].sort((p, q) => x.data[r * m + q] - x.data[r * m + p] || p - q);
          for (let j = 0; j < k; j += 1) { ids[r * k + j] = order[j]; values[r * k + j] = x.data[r * m + order[j]]; }
        }
        return [{ shape: [n, k], data: values }, { shape: [n, k], data: ids }];
      }
      case "drowse_chan_merge": {
        const [count, mean, m2, rows] = a;
        const [r, w] = rows.shape;
        const outMean = new Float32Array(w);
        const outM2 = new Float32Array(w);
        for (let c = 0; c < w; c += 1) {
          let mb = 0;
          for (let i = 0; i < r; i += 1) mb += rows.data[i * w + c] / r;
          let qb = 0;
          for (let i = 0; i < r; i += 1) qb += (rows.data[i * w + c] - mb) ** 2;
          const delta = mb - mean.data[c];
          const total = count.data[0] + r;
          outMean[c] = mean.data[c] + delta * r / total;
          outM2[c] = m2.data[c] + qb + delta * delta * count.data[0] * r / total;
        }
        return [{ shape: [w], data: outMean }, { shape: [w], data: outM2 }];
      }
      case "drowse_footprint_scores":
        return { shape: [x.shape[0], 259], data: Float32Array.from({ length: x.shape[0] * 259 }, (_, i) =>
          i % 259 < 3 ? 5 - (i % 259) : LOWEST) };
      default:
        return { shape: [1], data: new Float32Array(1) };
    }
  };
  return {
    installProgram: async () => {},
    call: async (request) => {
      log.push(request);
      if (request.op === "headGrams") return { id: "head_grams:9", kind: "head_grams", shape: [4, 4, 16, 16], bytes: 4 };
      if (request.op === "uploadResident") {
        return { id: `${request.kind}:${log.length}`, kind: request.kind, shape: request.tensors[0].shape, bytes: 4 };
      }
      if (request.op === "release") return null;
      assert.equal(request.op, "read");
      const values = new Map();
      for (const step of request.steps) {
        const args = step.args.map((arg) => {
          if ("tensor" in arg) return arg.tensor;
          if ("ref" in arg) {
            const value = values.get(arg.ref);
            return arg.shape ? { ...value, shape: arg.shape } : value;
          }
          return { resident: arg.resident, shape: arg.shape ?? [1, 64, 64], data: Float32Array.from({ length: 64 * 64 }, (_, i) => (i % 65 === 0 ? 1 : 0)) };
        });
        const result = run(step.fn, args);
        const names = Array.isArray(step.out) ? step.out : [step.out];
        names.forEach((name, i) => values.set(name, Array.isArray(result) ? result[i] : result));
        assert.equal(size(values.get(names[0]).shape), values.get(names[0]).data.length);
      }
      return { tensors: Object.fromEntries(request.read.map((name) => [name, values.get(name)])) };
    },
  };
}

async function staticReads(backendModule, modelBackend, capabilities, residualCapabilities, spec, tapped) {
  const log = [];
  const live = backendModule.webLlmBlockSitesBackend(capabilities, computingBridge(log));
  const f32 = modelBackend.f32Tensor;
  // The tapped head's gram is gathered from the resident grams into tap slot 0.
  const bound = await live.bindStructuredProgram(tapped);
  assert.notEqual(bound, tapped);
  assert.ok(bound.tapGrams.subarray(0, 256).every((value) => value === 23));
  assert.ok(bound.tapGrams.subarray(256).every((value) => value === 0));
  const gather = log.at(-1).steps[0];
  assert.equal(gather.fn, "drowse_gather_head_grams");
  assert.equal(gather.args[0].resident, "head_grams:9");
  assert.deepEqual([...gather.args[1].tensor.data], [2, 3]);

  // Unit weight rows: slots and the head write block's transpose.
  const units = [{ kind: "neuron", layer: 1, index: 5 }, { kind: "neuron", layer: 3, index: 0 }];
  const write = await live.unitWeightRows("neuron_write", units);
  assert.deepEqual(write.shape, [2, 64]);
  assert.equal(write.data[64 + 7], 1000 + 7);
  const read = await live.unitWeightRows("neuron_read", units);
  assert.deepEqual(read.shape, [2, 2, 64]);
  assert.equal(read.data[64], 200);
  await assert.rejects(live.unitWeightRows("neuron_write", [{ kind: "neuron", layer: 4, index: 0 }]), RangeError);
  await assert.rejects(live.unitWeightRows("head_write", units), RangeError);
  const headWrite = await live.unitWeightRows("head_write", [{ kind: "head", layer: 0, index: 1 }]);
  assert.deepEqual(headWrite.shape, [1, 64, 16]);
  assert.equal(headWrite.data[5 * 16 + 2], 2 * 64 + 5);
  const headRead = await live.unitWeightRows("head_read", [{ kind: "head", layer: 0, index: 1 }]);
  assert.deepEqual(headRead.shape, [1, 3, 16, 64]);
  assert.equal(headRead.data[0], 16 * 64);

  // Residents, then the vocabulary reads composed over them.
  const lens = await live.uploadLensLayer("s", "out", 1, f32(new Float32Array(64 * 64), [64, 64]));
  assert.equal(lens.family, "s");
  assert.deepEqual(log.at(-1).tensors[0].shape, [1, 64, 64]);
  const stats = await live.uploadEmbeddingStatistics({
    meanE: f32(new Float32Array(64), [64]),
    sigmaE: f32(new Float32Array(64 * 64), [64, 64]),
    mask: modelBackend.u32Tensor(new Uint32Array(9), [9]),
  });
  await assert.rejects(live.uploadEmbeddingStatistics({ ...stats, meanE: f32(new Float32Array(3), [3]) }), TypeError);
  const footprint = await live.footprintTopK(f32(new Float32Array(2 * 64), [2, 64]), 5, stats);
  assert.deepEqual([...footprint.ids.data.subarray(0, 5)], [0, 1, 2, 3, 4]);
  assert.deepEqual([...footprint.z.data.subarray(0, 4)], [5, 4, 3, -Infinity]);
  await assert.rejects(live.footprintTopK(f32(new Float32Array(64), [1, 64]), 257, stats), RangeError);
  const directions = await live.footprintDirections([lens, lens], Int32Array.of(3, 4), stats);
  assert.deepEqual(directions.shape, [2, 2, 64]);
  assert.deepEqual(log.at(-1).steps.map((step) => step.fn),
    ["drowse_embedding_rows", "drowse_batched_matmul_nt", "drowse_batched_matmul_nt"]);
  await assert.rejects(live.footprintDirections([lens], Int32Array.of(259), stats), RangeError);
  const times = await live.vocabRowsTimes({ kind: "unembed", ids: Int32Array.of(1), factor: { lens, transpose: false } });
  assert.deepEqual(times.shape, [1, 64]);
  assert.equal(log.at(-1).steps[1].fn, "drowse_batched_matmul");

  // Utilities: the transport per layer, −∞ in and out, Chan's count.
  const rows = f32(Float32Array.from({ length: 2 * 3 * 64 }, (_, i) => i % 7), [2, 3, 64]);
  const moved = await live.lensTransport(rows, [lens, lens], true);
  assert.deepEqual([...moved.data], [...rows.data]);
  const x = f32(Float32Array.of(1, -Infinity, 3, 2), [1, 4]);
  const top = await live.topK(x, 4);
  assert.deepEqual([...top.indices.data], [2, 3, 0, 1]);
  assert.deepEqual([...top.values.data], [3, 2, 1, -Infinity]);
  await assert.rejects(live.topK(f32(Float32Array.of(Number.NaN), [1, 1]), 1), RangeError);
  await assert.rejects(live.topK(x, 5), RangeError);
  const merged = await live.chanMerge(
    { count: 2, mean: f32(Float32Array.of(1), [1]), m2: f32(Float32Array.of(2), [1]) },
    f32(Float32Array.of(3, 5), [2, 1]),
  );
  assert.equal(merged.count, 4);
  assert.equal(merged.mean.data[0], 2.5);
  await assert.rejects(live.layerWriteScores(f32(new Float32Array(64), [1, 1, 64]), Int32Array.of(4)), RangeError);

  // A residual library runs the readout reads and refuses the unit ones.
  const residual = backendModule.webLlmBlockSitesBackend(residualCapabilities, computingBridge([]));
  const lse = await residual.logSumExp(f32(Float32Array.of(0, 0), [1, 2]));
  assert.ok(Math.abs(lse.data[0] - Math.log(2)) < 1e-6);
  await assert.rejects(residual.headWriteGrams([{ layer: 0, head: 0 }]), modelBackend.UnsupportedBlockError);
  await assert.rejects(residual.unembedCotangent(f32(new Float32Array(64), [1, 64]), Int32Array.of(1)),
    modelBackend.UnsupportedBlockError);
  assert.equal(spec.headDim, 16);
}
