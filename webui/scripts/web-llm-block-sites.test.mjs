// Hook ABI v5 in the WebLLM adapter (webLlmEngine.ts, drowseWebLlmBackend.ts,
// webLlmGeneration.ts): the schema-4 descriptor attested element by element against the lock,
// the block-sites capabilities a load yields, the bridge surface (unit programs bound on a
// units library, refused on a residual one), and live unit taps read as gate scores.
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
    server.ssrLoadModule("/src/hosted/runtime/drowseWebLlmBackend.ts"),
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
  // One tapped head (layer 2): its gram must be present until the bridge builds grams itself.
  const tapped = unitProgram();
  const offsets = layout.meta.tapHeadOffsets.offset;
  for (let layer = 3; layer <= 4; layer += 1) tapped.unitMeta[offsets + layer] = 1;
  await assert.rejects(surface.bindStructuredProgram(tapped), (error) => error.code === "BLOCK_SITES_MEMBER_UNAVAILABLE");
  tapped.tapGrams[0] = 1;
  assert.equal(await surface.bindStructuredProgram(tapped), tapped);
  const short = unitProgram();
  short.unitCoeff = new Float32Array(layout.coeffLength - 1);
  await assert.rejects(surface.bindStructuredProgram(short), TypeError);
  await assert.rejects(surface.openReplay({}), (error) => error.code === "BLOCK_SITES_MEMBER_UNAVAILABLE");
  const residualSurface = backendModule.webLlmBlockSitesBackend(residualCapabilities);
  await assert.rejects(residualSurface.bindStructuredProgram(unitProgram()), (error) =>
    error instanceof modelBackend.UnsupportedBlockError && error.message === refusal);
  await assert.rejects(residualSurface.unitWeightRows("neuron_write", []), modelBackend.UnsupportedBlockError);
  assert.equal(await residualSurface.bindStructuredProgram(plain), plain);

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
