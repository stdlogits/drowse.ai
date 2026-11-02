// Hook ABI v5 device seam (modelBackend.ts): the surfaces every backend exposes before B4's
// bridge exists, the error drowse 6.0 answers unit calls with on an undescribed model, typed
// tensors, and the replay buffer layouts of B1-hook-abi-v5.md §2.3/§3.4.
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({
  root,
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});

try {
  const [backend, fake, profiles] = await Promise.all([
    server.ssrLoadModule("/src/hosted/runtime/modelBackend.ts"),
    server.ssrLoadModule("/src/hosted/runtime/fakeModelBackend.ts"),
    server.ssrLoadModule("/src/hosted/runtime/structuredHookProfile.ts"),
  ]);
  const v4 = profiles.STANDARD_STRUCTURED_HOOK_PROFILE_V4;
  const residual = profiles.STANDARD_STRUCTURED_HOOK_PROFILE_V4_RESIDUAL;
  const members = backend.BLOCK_SITES_BACKEND_MEMBERS;
  assert.equal(new Set(members).size, members.length);
  assert.ok(backend.BLOCK_SITES_UNIT_MEMBERS.every((member) => members.includes(member)));

  // ------------------------------------------- surfaces with no v5 library behind them ---
  const surfaceRejects = async (surface, check) => {
    assert.equal(surface.capabilities(), null);
    assert.deepEqual(Object.keys(surface).sort(), ["capabilities", ...members].sort());
    for (const member of members) {
      await assert.rejects(surface[member](), check, member);
    }
  };
  const fixture = new fake.DeterministicBrowserModelBackend();
  await surfaceRejects(fixture.blockSites(), (error) =>
    error instanceof backend.UnsupportedBlockError &&
    error.name === "UnsupportedBlockError" &&
    error.code === "UnsupportedBlockError" &&
    error.status === 422 &&
    error.message === fake.FIXTURE_UNIT_REFUSAL);
  await surfaceRejects(new backend.UnavailableBrowserModelBackend().blockSites(), (error) =>
    error.code === "CUSTOM_MLC_BACKEND_UNAVAILABLE" && error.status === 503);

  // Native's message shape: `reason`, or `layer N: reason` (B2's recorded refusals).
  const refusal = new backend.UnsupportedBlockError(
    "block has no softmax self-attention ('self_attn')",
    { layer: 0 },
  );
  assert.equal(refusal.message, "layer 0: block has no softmax self-attention ('self_attn')");
  assert.equal(refusal.layer, 0);
  assert.equal(new backend.UnsupportedBlockError("no rotary embedding").layer, null);

  // -------------------------------------------------------------------- capabilities ---
  const spec = profiles.validateBlockSpec({
    dFF: 9728, nHeads: 32, nKvHeads: 8, headDim: 128, sandwich: false, normOffset: 0,
    activation: "silu", embedScale: 1, rmsEps: 1e-6, qkNorm: true, attnScaling: 128 ** -0.5,
    attnLogitSoftcap: null, finalLogitSoftcap: null, ropeTheta: 1e6, ropeLocalTheta: null,
    ropeGlobalScale: 1, ropeInGraph: false, slidingWindow: null, slidingPattern: null,
  });
  const capabilities = {
    hookAbi: "block-sites-v5", runtimeAbi: "drowse-web-runtime-v2", exactReadoutAbi: "exact-readout-v2",
    profile: v4, layerCount: 36, hiddenSize: 2560, vocabSize: 151936, contextTokens: 4096,
    prefillChunkSize: 2048, blockSpec: spec, unitRefusal: null,
  };
  assert.equal(backend.validateBlockSitesCapabilities(capabilities), capabilities);
  backend.validateBlockSitesCapabilities({
    ...capabilities, profile: residual, blockSpec: null,
    unitRefusal: "layer 0: linear-attention layer ('linear_attn') in a hybrid stack",
  });
  assert.throws(() => backend.validateBlockSitesCapabilities({ ...capabilities, unitRefusal: "x" }), /reverse/);
  assert.throws(
    () => backend.validateBlockSitesCapabilities({ ...capabilities, profile: residual, blockSpec: null, unitRefusal: "" }),
    /name the reason/,
  );
  assert.throws(() => backend.validateBlockSitesCapabilities({ ...capabilities, hookAbi: "post-block-residual-v4" }), /wrong ABI/);
  assert.throws(
    () => backend.validateBlockSitesCapabilities({ ...capabilities, profile: { ...capabilities.profile, id: "unknown-profile" } }),
    /Unsupported structured hook capacity profile/,
  );

  // ------------------------------------------------------------------------ tensors ---
  const rows = backend.f32Tensor(new Float32Array(6), [2, 3]);
  assert.deepEqual(rows.shape, [2, 3]);
  assert.ok(Object.isFrozen(rows) && Object.isFrozen(rows.shape));
  assert.throws(() => backend.f32Tensor(new Float32Array(5), [2, 3]), /needs 6 values/);
  assert.equal(backend.requireTensorShape(rows, "f32", [2, null], "rows"), rows);
  assert.throws(() => backend.requireTensorShape(rows, "f32", [3, null], "rows"), /rows must be f32 \[3, \*\]/);
  assert.throws(() => backend.requireTensorShape(rows, "i32", [2, 3], "rows"), /must be i32/);
  assert.equal(backend.i32Tensor(new Int32Array(4), [4]).dtype, "i32");
  assert.equal(backend.u32Tensor(new Uint32Array(1), [1]).dtype, "u32");
  assert.equal(backend.f16Tensor(new Uint16Array(2), [1, 2]).dtype, "f16");

  // ----------------------------------------------------------- replay layouts (§2.3) ---
  const anatomy = backend.anatomyTapLayout(2560, spec);
  assert.deepEqual(anatomy, {
    stride: 29184, blockIn: 0, mid: 2560, outPre: 5120, out: 7680, attnO: 10240, mlpO: 12800,
    act: 15360, z: 25088,
  });
  assert.equal(anatomy.z + spec.nHeads * spec.headDim, anatomy.stride);
  const tinySpec = { ...spec, dFF: 3, nHeads: 2, nKvHeads: 1, headDim: 2 };
  assert.deepEqual(backend.activityStateLayout(tinySpec), {
    stride: 15, neuronValue: 0, neuronRow: 3, neuronRank: 6, headValue: 9, headRow: 11, headRank: 13,
  });
  const initialLayer = [0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1];
  assert.deepEqual(
    Array.from(backend.initialActivityState(2, tinySpec)),
    [...initialLayer, ...initialLayer],
  );
  assert.deepEqual(backend.unitScalesLayout(tinySpec), { stride: 8, positive: 0, negative: 3, head: 6 });

  const metaLayout = backend.replayMetaLayout(v4);
  assert.deepEqual(metaLayout, {
    length: 134, anatomyPosition: 0, spanCount: 1, spans: 2, populationRow: 18, traceCount: 19,
    traces: 20, patternLayerCount: 116, patternLayers: 117, flags: 133,
  });
  assert.throws(() => backend.replayMetaLayout(residual), /no replay entry point/);
  const qwenModel = { layerCount: 36, contextTokens: 4096, blockSpec: spec };
  const meta = backend.encodeReplayMeta(v4, qwenModel, {
    anatomyPosition: 41,
    activity: {
      spans: [{ firstRow: 0, rows: 3, firstPosition: -1 }, { firstRow: 3, rows: 5, firstPosition: 30 }],
      rowCount: 8,
      scales: { kind: "unit_scales", id: "s", bytes: 0, shape: [36, 19488] },
      traces: [{ kind: "neuron", layer: 8, index: 344 }, { kind: "head", layer: 20, index: 3 }],
      populationRow: 2,
    },
    patternLayers: [6, 10, 35],
  });
  assert.equal(meta.length, 134);
  assert.deepEqual(Array.from(meta.subarray(0, 18)), [
    41, 2, 0, 3, -1, 1, 3, 5, 30, 0, -1, -1, -1, -1, -1, -1, -1, -1,
  ]);
  assert.deepEqual(Array.from(meta.subarray(18, 27)), [2, 2, 8, 0, 344, 20, 1, 3, -1]);
  assert.deepEqual(Array.from(meta.subarray(116, 134)), [3, 6, 10, 35, ...Array(13).fill(-1), 7]);
  const none = backend.encodeReplayMeta(v4, qwenModel, { anatomyPosition: null, activity: null, patternLayers: null });
  assert.deepEqual([none[0], none[1], none[18], none[19], none[116], none[133]], [-1, 0, -1, 0, 0, 0]);
  assert.throws(
    () => backend.encodeReplayMeta(v4, qwenModel, { anatomyPosition: null, activity: null, patternLayers: [1] }),
    /query at the anatomy position/,
  );
  assert.throws(
    () => backend.encodeReplayMeta(v4, qwenModel, { anatomyPosition: 1, activity: null, patternLayers: [3, 2] }),
    /must ascend/,
  );
  assert.throws(
    () => backend.encodeReplayMeta(v4, qwenModel, {
      anatomyPosition: 1, activity: null, patternLayers: Array.from({ length: 17 }, (_, i) => i),
    }),
    /1 to 16 layers/,
  );
  // Nothing wraps in the int32 table: rows, positions and units are bounded by the model.
  const activityAt = (change) => ({
    anatomyPosition: null,
    patternLayers: null,
    activity: {
      spans: [{ firstRow: 0, rows: 1, firstPosition: 1 }], rowCount: 1,
      scales: { kind: "unit_scales", id: "s", bytes: 0, shape: [36, 19488] },
      traces: [], populationRow: null, ...change,
    },
  });
  assert.throws(
    () => backend.encodeReplayMeta(v4, qwenModel, activityAt({
      rowCount: 4294967297, spans: [{ firstRow: 4294967296, rows: 1, firstPosition: 1 }],
    })),
    /fit the compiled context/,
  );
  assert.throws(
    () => backend.encodeReplayMeta(v4, qwenModel, activityAt({ spans: [{ firstRow: 0, rows: 1, firstPosition: 4096 }] })),
    /outside the node's rows or the context/,
  );
  assert.throws(
    () => backend.encodeReplayMeta(v4, qwenModel, activityAt({ traces: [{ kind: "neuron", layer: 0, index: 9728 }] })),
    /trace unit 9728 is out of range/,
  );
  assert.throws(
    () => backend.encodeReplayMeta(v4, qwenModel, activityAt({ traces: [{ kind: "head", layer: 0, index: 32 }] })),
    /trace unit 32 is out of range/,
  );
  assert.throws(
    () => backend.encodeReplayMeta(v4, qwenModel, { anatomyPosition: 4096, activity: null, patternLayers: null }),
    /anatomy position 4096/,
  );
  // An authored first turn's unproduced first row (a one-row span at column −1) is valid.
  const unproduced = backend.encodeReplayMeta(v4, qwenModel, activityAt({ spans: [{ firstRow: 0, rows: 1, firstPosition: -1 }] }));
  assert.deepEqual(Array.from(unproduced.subarray(2, 6)), [0, 1, -1, 1]);

  // ------------------------------------------- buffer sizes against B1 §2.3, §3.1, §6 ---
  const MiB = 2 ** 20;
  const qwen4b = backend.blockSitesBufferElements(v4, {
    layerCount: 36, hiddenSize: 2560, contextTokens: 4096, prefillChunkSize: 2048, blockSpec: spec,
  });
  assert.equal(qwen4b.headGrams * 4 / MiB, 72);
  assert.equal(qwen4b.patternKeysSlot * 4 / MiB, 16);
  // Sixteen slots at the full context: 256 MiB, within the 1 GiB binding every units model needs.
  assert.equal(qwen4b.patternKeysSlot * v4.maxPatternLayers * 4 / MiB, 256);
  assert.equal(qwen4b.traces * 4 / MiB, 0.5);
  assert.ok(Math.abs(qwen4b.activityState * 4 / MiB - 4.0) < 0.05);
  assert.equal(qwen4b.anatomyTaps, 36 * 29184);
  assert.equal(qwen4b.unitMeta, 4 * 37 + 2 * (256 + 64 + 64 + 16));
  assert.equal(qwen4b.unitCoeff, 2 * 256 + 64 + 64 * 128);
  assert.equal(qwen4b.replayMeta, 134);
  const gemma270m = backend.blockSitesBufferElements(v4, {
    layerCount: 18, hiddenSize: 640, contextTokens: 4096, prefillChunkSize: 2048,
    blockSpec: { ...spec, dFF: 2048, nHeads: 4, nKvHeads: 1, headDim: 256 },
  });
  assert.equal(gemma270m.headGrams * 4 / MiB, 18);
  assert.equal(gemma270m.tapGrams * 4 / MiB, 4);
  assert.ok(gemma270m.unitMeta * 4 + gemma270m.unitCoeff * 4 < 100 * 1024, "unit tables stay under 100 KiB");

  console.log("block-sites backend seam: surfaces, errors, tensors and replay layouts pass");
} finally {
  await server.close();
}
