// Hook ABI v5 (block-sites-v5): the standard-v4 profiles, descriptor schema 4 and the
// drowse-structured-v4 unit section, pinned against B1-hook-abi-v5.md §3.1/§3.4/§3.5 and the
// drowse unit-ablation goldens (drowse/tests/fixtures/browser/unit-ablation-<arch>-v1.json).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const fixtures = fileURLToPath(new URL("../../drowse/tests/fixtures/browser/", import.meta.url));
const golden = (name) => JSON.parse(readFileSync(`${fixtures}${name}`, "utf8"));
const server = await createServer({
  root,
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});

function decodeF32(entry) {
  assert.equal(entry.encoding, "float32-le-base64");
  const bytes = Buffer.from(entry.data, "base64");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return Float32Array.from({ length: bytes.byteLength / 4 }, (_, i) => view.getFloat32(4 * i, true));
}

const typedKeys = (program) =>
  Object.keys(program).filter((key) => ArrayBuffer.isView(program[key])).sort();

try {
  const [profiles, hooks] = await Promise.all([
    server.ssrLoadModule("/src/hosted/runtime/structuredHookProfile.ts"),
    server.ssrLoadModule("/src/hosted/runtime/structuredHookProgram.ts"),
  ]);
  const v3 = profiles.STANDARD_STRUCTURED_HOOK_PROFILE_V3;
  const v4 = profiles.STANDARD_STRUCTURED_HOOK_PROFILE_V4;
  const residual = profiles.STANDARD_STRUCTURED_HOOK_PROFILE_V4_RESIDUAL;

  // ---------------------------------------------------------------- profiles (§3.5) ---
  for (const [key, value] of Object.entries(v3)) {
    if (key === "id") continue;
    assert.equal(v4[key], value, `standard-v4 keeps standard-v3's ${key}`);
    assert.equal(residual[key], value, `standard-v4-residual keeps standard-v3's ${key}`);
  }
  assert.deepEqual(
    Object.fromEntries(Object.entries(v4).filter(([key]) => !(key in v3))),
    {
      unitSites: 1, maxUnitGroups: 4, maxNeuronAblations: 256, maxHeadAblations: 64,
      maxUnitTaps: 64, maxHeadTaps: 16, maxActivityTraces: 32, maxActivitySpans: 4,
      maxPatternLayers: 8, maxPatternHeads: 16, maxLensReadTopK: 256, maxAnatomyUnits: 64,
    },
  );
  assert.deepEqual(
    Object.fromEntries(Object.entries(residual).filter(([key]) => !(key in v3))),
    {
      unitSites: 0, maxUnitGroups: 0, maxNeuronAblations: 0, maxHeadAblations: 0,
      maxUnitTaps: 0, maxHeadTaps: 0, maxActivityTraces: 0, maxActivitySpans: 0,
      maxPatternLayers: 0, maxPatternHeads: 0, maxLensReadTopK: 256, maxAnatomyUnits: 0,
    },
  );
  assert.equal(profiles.validateStructuredHookProfile({ ...v4 }), v4);
  assert.equal(profiles.resolveStructuredHookProfile("standard-v4-residual"), residual);
  assert.throws(() => profiles.validateStructuredHookProfile({ ...v4, maxUnitTaps: 65 }), /was modified/);
  const { maxPatternHeads: _dropped, ...missing } = v4;
  assert.throws(() => profiles.validateStructuredHookProfile(missing), /unsupported shape/);
  assert.throws(() => profiles.validateStructuredHookProfile({ ...v3, unitSites: 0 }), /unsupported shape/);
  assert.throws(() => profiles.validateBlockSitesProfile(v3), /predates hook ABI block-sites-v5/);
  assert.equal(profiles.BLOCK_SITES_HOOK_ABI, "block-sites-v5");
  assert.equal(profiles.BLOCK_SITES_RUNTIME_ABI, "drowse-web-runtime-v2");
  assert.equal(profiles.BLOCK_SITES_EXACT_READOUT_ABI, "exact-readout-v2");

  // ------------------------------------------------------------- BlockSpec (§8.2) ---
  const tiny = golden("units-tiny-models-v1.json").arches;
  const tinySpec = (arch) => {
    const config = tiny[arch].config;
    const gemma = arch === "gemma3";
    const rope = config.rope_parameters;
    return {
      dFF: config.intermediate_size,
      nHeads: config.num_attention_heads,
      nKvHeads: config.num_key_value_heads,
      headDim: config.head_dim,
      sandwich: gemma,
      normOffset: gemma ? 1 : 0,
      activation: gemma ? "gelu_tanh" : "silu",
      embedScale: gemma ? Math.sqrt(config.hidden_size) : 1,
      rmsEps: config.rms_norm_eps,
      qkNorm: true,
      attnScaling: (gemma ? config.query_pre_attn_scalar : config.head_dim) ** -0.5,
      attnLogitSoftcap: null,
      finalLogitSoftcap: null,
      ropeTheta: gemma ? rope.full_attention.rope_theta : rope.rope_theta,
      ropeLocalTheta: gemma ? rope.sliding_attention.rope_theta : null,
      ropeGlobalScale: 1,
      ropeInGraph: gemma,
      slidingWindow: gemma ? config.sliding_window : null,
      slidingPattern: gemma ? 2 : null,
    };
  };
  const specs = { gemma3: profiles.validateBlockSpec(tinySpec("gemma3")), qwen3: profiles.validateBlockSpec(tinySpec("qwen3")) };
  assert.deepEqual(Object.keys(specs.qwen3), [...profiles.BLOCK_SPEC_KEYS]);
  tiny.gemma3.config.layer_types.forEach((type, layer) => {
    assert.equal(profiles.isLocalAttentionLayer(specs.gemma3, layer), type === "sliding_attention");
  });
  assert.equal(profiles.isLocalAttentionLayer(specs.qwen3, 0), false);
  assert.throws(() => profiles.validateBlockSpec({ ...specs.qwen3, slidingWindow: 4 }), /set together/);
  assert.throws(() => profiles.validateBlockSpec({ ...specs.qwen3, nKvHeads: 3 }), /multiple of nKvHeads/);
  assert.throws(() => profiles.validateBlockSpec({ ...specs.qwen3, activation: "gelu" }), /silu or gelu_tanh/);
  assert.throws(() => profiles.validateBlockSpec({ ...specs.qwen3, extra: 1 }), /unsupported shape/);
  // B1-goldens: token-anatomy taps are `L × 544` on the 64-wide fixture models.
  assert.equal(profiles.anatomyStride(64, specs.qwen3), 544);
  assert.equal(profiles.anatomyStride(64, specs.gemma3), 544);
  // B1 §2.3's production strides.
  for (const [d, dFF, heads, hd, stride] of [
    [640, 2048, 4, 256, 6912], [1152, 6912, 4, 256, 14848], [2560, 10240, 8, 256, 27648],
    [2048, 6144, 16, 128, 20480], [2560, 9728, 32, 128, 29184],
  ]) {
    assert.equal(profiles.anatomyStride(d, { ...specs.qwen3, dFF, nHeads: heads, headDim: hd }), stride);
  }

  // ------------------------------------------------------- descriptor schema 4 (§3.5) ---
  const fields = profiles.HOOK_PROFILE_DESCRIPTOR_V4_FIELDS;
  assert.equal(fields.length, 67);
  assert.equal(profiles.HOOK_PROFILE_DESCRIPTOR_V4_LENGTH, 67);
  assert.equal(new Set(fields).size, 67);
  // v3's 31 fields, in the order the v4 bridge's `expected` array checks them.
  assert.deepEqual(fields.slice(0, 31), [
    "MAGIC", "SCHEMA_VERSION", "PROFILE", "HOOK_ABI_VERSION", "FORMAT_VERSION", "LAYER_COUNT",
    "HIDDEN_SIZE", "MAX_AFFINE_GROUPS", "MAX_RANK", "MAX_PROBES", "MAX_CURVES", "MAX_CURVE_NODES",
    "MAX_INTRINSIC_DIM", "MAX_EMBED_DIM", "CURVE_PARAMETER_STRIDE",
    "MIN_COMPUTE_WORKGROUP_STORAGE_SIZE", "MAX_GEOMETRY_PROBES", "MAX_WHITENER_RANK",
    "MAX_GEOMETRY_CANDIDATES", "GEOMETRY_OUTPUT_STRIDE", "GEOMETRY_FOOT_RESTARTS",
    "GEOMETRY_FOOT_ITERATIONS", "GEOMETRY_WARM_RESTARTS", "GEOMETRY_WARM_ITERATIONS",
    "REQUIRED_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE", "GEOMETRY_KERNEL_STORAGE_BINDINGS",
    "GEOMETRY_HEADER_STRIDE", "GEOMETRY_PAYLOAD_ELEMENTS", "EXACT_READOUT_ABI_VERSION",
    "READOUT_TOP_K", "MAX_SAE_FEATURES_PER_CHUNK",
  ]);
  // B1 §3.5's schema-4 additions, in its order.
  assert.deepEqual(fields.slice(31), [
    "UNIT_SITES", "D_FF", "N_Q_HEADS", "N_KV_HEADS", "HEAD_DIM", "SANDWICH", "NORM_OFFSET",
    "ACTIVATION", "EMBED_SCALE_F32_BITS", "RMS_EPS_F32_BITS", "QK_NORM", "ATTN_SCALING_F32_BITS",
    "ATTN_LOGIT_SOFTCAP_F32_BITS", "FINAL_LOGIT_SOFTCAP_F32_BITS", "ROPE_THETA_F32_BITS",
    "ROPE_LOCAL_THETA_F32_BITS", "ROPE_GLOBAL_SCALE_F32_BITS", "ROPE_IN_GRAPH", "SLIDING_WINDOW",
    "SLIDING_PATTERN", "PREFILL_CHUNK_SIZE", "MAX_UNIT_GROUPS", "MAX_NEURON_ABLATIONS",
    "MAX_HEAD_ABLATIONS", "MAX_UNIT_TAPS", "MAX_HEAD_TAPS", "MAX_ACTIVITY_TRACES",
    "MAX_ACTIVITY_SPANS", "MAX_PATTERN_LAYERS", "MAX_PATTERN_HEADS", "ANATOMY_STRIDE",
    "UNIT_POSITIONS_FROM", "MAX_LENS_READ_TOP_K", "MAX_ANATOMY_UNITS",
    "DROWSE_KERNEL_STORAGE_BINDINGS", "CONTEXT_TOKENS",
  ]);
  assert.equal(profiles.float32Bits(1), 0x3f800000);
  assert.equal(profiles.float32Bits(-2), -0x40000000);
  assert.equal(profiles.float32Bits(1e-6), 0x358637bd);

  // The v4 bridge's v3 geometry payload rule, restated independently of the profile module.
  const payload = (l, d) => l * 8 * (2 * d + 8 * d + 2 * 64 + 33 * 8 + 4 * 8 + 4 + 671 + 32 * 4 + 32 * 8);
  const model = (arch) => ({
    layerCount: 4, hiddenSize: 64, contextTokens: 4096, prefillChunkSize: 2048, blockSpec: specs[arch],
  });
  const named = (values) => Object.fromEntries(fields.map((name, i) => [name, values[i]]));
  const qwen = named(profiles.expectedHookProfileDescriptor(v4, model("qwen3")));
  assert.deepEqual(Object.values(qwen).slice(0, 31), [
    0x53414b4c, 4, 4, 5, 4, 4, 64, 4, 8, 8, 4, 32, 4, 8, 671, 32768, 8, 96, 33, 41, 3, 12, 2, 4,
    8, 8, 7, payload(4, 64), 2, 8, 16384,
  ]);
  assert.deepEqual(Object.values(qwen).slice(31), [
    1, 96, 4, 2, 16, 0, 0, 1, 0x3f800000, profiles.float32Bits(1e-6), 1,
    profiles.float32Bits(0.25), 0, 0, profiles.float32Bits(10000), 0, 0x3f800000, 0, 0, 0, 2048,
    4, 256, 64, 64, 16, 32, 4, 8, 16, 544, 1, 256, 64, 8, 4096,
  ]);
  const gemma = named(profiles.expectedHookProfileDescriptor(v4, model("gemma3")));
  assert.deepEqual(
    [gemma.SANDWICH, gemma.NORM_OFFSET, gemma.ACTIVATION, gemma.EMBED_SCALE_F32_BITS,
      gemma.ATTN_SCALING_F32_BITS, gemma.ROPE_THETA_F32_BITS, gemma.ROPE_LOCAL_THETA_F32_BITS,
      gemma.ROPE_IN_GRAPH, gemma.SLIDING_WINDOW, gemma.SLIDING_PATTERN],
    [1, 1, 2, profiles.float32Bits(8), profiles.float32Bits(0.25), profiles.float32Bits(1e6),
      profiles.float32Bits(1e4), 1, 4, 2],
  );
  const residualDescriptor = named(profiles.expectedHookProfileDescriptor(residual, {
    ...model("qwen3"), blockSpec: null,
  }));
  for (const name of fields.slice(31)) {
    const expected = {
      PREFILL_CHUNK_SIZE: 2048, MAX_LENS_READ_TOP_K: 256, DROWSE_KERNEL_STORAGE_BINDINGS: 8,
      CONTEXT_TOKENS: 4096,
    }[name] ?? 0;
    assert.equal(residualDescriptor[name], expected, `residual ${name}`);
  }
  assert.throws(
    () => profiles.expectedHookProfileDescriptor(v4, { ...model("qwen3"), blockSpec: null }),
    /requires the model's block spec/,
  );
  assert.throws(() => profiles.expectedHookProfileDescriptor(residual, model("qwen3")), /forbids a block spec/);
  assert.throws(() => profiles.expectedHookProfileDescriptor(v3, model("qwen3")), /predates/);
  assert.throws(
    () => profiles.expectedHookProfileDescriptor(v4, { ...model("qwen3"), prefillChunkSize: 8192 }),
    /exceeds the context window/,
  );
  const expected = profiles.expectedHookProfileDescriptor(v4, model("qwen3"));
  const drifted = Int32Array.from(expected);
  drifted[fields.indexOf("RMS_EPS_F32_BITS")] = profiles.float32Bits(1e-5);
  drifted[fields.indexOf("MAX_PATTERN_HEADS")] = 0;
  assert.deepEqual(profiles.hookProfileDescriptorMismatches(drifted, expected), [
    "RMS_EPS_F32_BITS", "MAX_PATTERN_HEADS",
  ]);
  assert.deepEqual(profiles.hookProfileDescriptorMismatches(expected, expected), []);
  assert.deepEqual(profiles.hookProfileDescriptorMismatches(expected.slice(0, 31), expected), ["LENGTH"]);
  assert.equal(profiles.hookProfileDescriptorProfileId(expected), "standard-v4");
  assert.equal(
    profiles.hookProfileDescriptorProfileId(
      profiles.expectedHookProfileDescriptor(residual, { ...model("qwen3"), blockSpec: null }),
    ),
    "standard-v4-residual",
  );
  assert.throws(() => profiles.hookProfileDescriptorProfileId(expected.slice(0, 31)), /not schema 4/);

  // ---------------------------------------------------------- unit section (§3.1) ---
  const shape = { dFF: 96, nHeads: 4, headDim: 16 };
  const layout = hooks.structuredUnitSectionLayout(v4, 4, 16);
  const field = (offset, length) => ({ offset, length });
  assert.deepEqual(layout.meta, {
    neuronOffsets: field(0, 5), neuronIndex: field(5, 256), neuronGroup: field(261, 256),
    headOffsets: field(517, 5), headIndex: field(522, 64), headGroup: field(586, 64),
    tapNeuronOffsets: field(650, 5), tapNeuronIndex: field(655, 64), tapNeuronSlot: field(719, 64),
    tapHeadOffsets: field(783, 5), tapHeadIndex: field(788, 16), tapHeadSlot: field(804, 16),
  });
  assert.deepEqual(layout.coeff, {
    neuronKeep: field(0, 256), neuronBias: field(256, 256), headKeep: field(512, 64),
    headBias: field(576, 1024),
  });
  assert.equal(layout.metaLength, 4 * 4 + 804);
  assert.equal(layout.coeffLength, 2 * 256 + 64 + 64 * 16);
  assert.equal(layout.activeLength, 4);
  assert.equal(layout.tapGramsLength, 16 * 16 * 16);
  assert.throws(() => hooks.structuredUnitSectionLayout(residual, 4, 16), /no unit sites/);
  assert.throws(() => hooks.structuredUnitSectionLayout(v3, 4, 16), /no unit sites/);

  const tap = (kind, layer, index, slot) => ({
    name: `probe-${slot}`, unit: hooks.structuredUnitName(kind, layer, index), kind, layer, index,
    slot, scale: { source: "neutral", detail: "fixture", top: kind === "neuron" ? [1, 1] : 1, value: kind === "neuron" ? [1, 1] : 1 },
  });
  const headBias = Float32Array.from({ length: 16 }, (_, i) => i / 8 - 1);
  const units = {
    ablations: [
      { kind: "neuron", layer: 2, index: 7, keep: 0.5, bias: 0.25, group: 0 },
      { kind: "head", layer: 0, index: 1, keep: 0, bias: headBias, group: 1 },
      { kind: "neuron", layer: 0, index: 3, keep: 0, bias: -1, group: 1 },
      { kind: "neuron", layer: 2, index: 2, keep: 2, bias: 0.5, group: 0 },
    ],
    taps: [tap("neuron", 3, 9, 2), tap("head", 2, 1, 1), tap("neuron", 1, 5, 0)],
    controls: [null, null],
  };
  const tables = hooks.encodeStructuredUnitTables(v4, 4, shape, units);
  const meta = (name) => Array.from(tables.unitMeta.subarray(layout.meta[name].offset, layout.meta[name].offset + layout.meta[name].length));
  const coeff = (name) => Array.from(tables.unitCoeff.subarray(layout.coeff[name].offset, layout.coeff[name].offset + layout.coeff[name].length));
  const padded = (values, length) => [...values, ...new Array(length - values.length).fill(0)];
  // CSR by layer, registration order within a layer (L0 n3, then L2 n7, L2 n2).
  assert.deepEqual(meta("neuronOffsets"), [0, 1, 1, 3, 3]);
  assert.deepEqual(meta("neuronIndex"), padded([3, 7, 2], 256));
  assert.deepEqual(meta("neuronGroup"), padded([1, 0, 0], 256));
  assert.deepEqual(meta("headOffsets"), [0, 1, 1, 1, 1]);
  assert.deepEqual(meta("headIndex"), padded([1], 64));
  assert.deepEqual(meta("headGroup"), padded([1], 64));
  assert.deepEqual(meta("tapNeuronOffsets"), [0, 0, 1, 1, 2]);
  assert.deepEqual(meta("tapNeuronIndex"), padded([5, 9], 64));
  assert.deepEqual(meta("tapNeuronSlot"), padded([0, 2], 64));
  assert.deepEqual(meta("tapHeadOffsets"), [0, 0, 0, 1, 1]);
  assert.deepEqual(meta("tapHeadIndex"), padded([1], 16));
  assert.deepEqual(meta("tapHeadSlot"), padded([1], 16));
  assert.deepEqual(coeff("neuronKeep"), padded([0, 0.5, 2], 256));
  assert.deepEqual(coeff("neuronBias"), padded([-1, 0.25, 0.5], 256));
  assert.deepEqual(coeff("headKeep"), padded([0], 64));
  assert.deepEqual(coeff("headBias"), padded(Array.from(headBias), 1024));
  assert.deepEqual(Array.from(tables.unitActive), [1, 1, 0, 0]);
  assert.ok(tables.tapGrams.every((value) => value === 0));
  // The upload is the arrays' bytes: u32 and f32 little-endian.
  const metaBytes = new DataView(tables.unitMeta.buffer);
  assert.equal(metaBytes.getUint32(4 * (layout.meta.neuronOffsets.offset + 3), true), 3);
  assert.equal(metaBytes.getUint32(4 * layout.meta.neuronIndex.offset, true), 3);
  assert.equal(new DataView(tables.unitCoeff.buffer).getFloat32(4 * (layout.coeff.neuronKeep.offset + 2), true), 2);

  // Fail closed (B1 §2.2, §3.1).
  const failsWith = (code, change, unitShape = shape) => {
    assert.throws(
      () => hooks.encodeStructuredUnitTables(v4, 4, unitShape, { ...units, ...change }),
      (error) => error instanceof hooks.StructuredUnitProgramError && error.code === code && error.status === 422,
      code,
    );
  };
  failsWith("UNIT_DUPLICATE", { ablations: [...units.ablations, { ...units.ablations[0], group: 1 }] });
  failsWith("UNIT_DUPLICATE", { taps: [...units.taps, tap("neuron", 0, 0, 2)] });
  failsWith("UNIT_DUPLICATE", { taps: [...units.taps, tap("neuron", 3, 9, 5)] });
  failsWith("UNIT_ABLATION_CAPACITY", {
    ablations: Array.from({ length: 257 }, (_, i) => ({ kind: "neuron", layer: i % 4, index: Math.floor(i / 4), keep: 0, bias: 0, group: 0 })),
  });
  failsWith(
    "UNIT_TAP_CAPACITY",
    { taps: Array.from({ length: 17 }, (_, i) => tap("head", i % 4, Math.floor(i / 4), i)) },
    { ...shape, nHeads: 8 },
  );
  failsWith("UNIT_TAP_CAPACITY", { taps: [tap("neuron", 0, 0, 64)] });
  failsWith("UNIT_GROUP_CAPACITY", { controls: [null, null, null, null, null] });
  failsWith("UNIT_GROUP_CAPACITY", { controls: [null] });
  failsWith("UNIT_OUT_OF_RANGE", { ablations: [{ kind: "neuron", layer: 0, index: 96, keep: 0, bias: 0, group: 0 }] });
  failsWith("UNIT_OUT_OF_RANGE", { ablations: [{ kind: "head", layer: 4, index: 0, keep: 0, bias: headBias, group: 0 }] });
  assert.throws(() => hooks.encodeStructuredUnitTables(residual, 4, shape, units), /no unit sites/);

  // ---------------------------------- a v4 program without units is v3's buffers (§3.4) ---
  const layers = [
    {
      affineGroups: [{ active: true, basis: [[1, 0], [0, 1]], neutral: [0, 0], target: [1, 0], along: 0.5, kappa: [0, 1] }],
      probes: [{ kind: "linear", direction: [1, 0], bias: -0.25 }],
    },
    { probes: [{ kind: "linear", direction: [0, 1], bias: 0 }] },
    {},
    {},
  ];
  const programV3 = hooks.compileStructuredHookProgram(2, layers, v3);
  const programV4 = hooks.compileStructuredHookProgram(2, layers, v4);
  assert.equal(programV3.format, "drowse-structured-v3");
  assert.equal(programV3.hookAbi, "post-block-residual-v4");
  assert.equal(programV4.format, "drowse-structured-v4");
  assert.equal(programV4.hookAbi, "block-sites-v5");
  assert.deepEqual(typedKeys(programV4), typedKeys(programV3));
  for (const key of typedKeys(programV3)) {
    assert.deepEqual(
      Buffer.from(programV4[key].buffer, programV4[key].byteOffset, programV4[key].byteLength),
      Buffer.from(programV3[key].buffer, programV3[key].byteOffset, programV3[key].byteLength),
      `${key} is byte-identical`,
    );
  }
  assert.equal(hooks.withStructuredUnitSection(programV4, shape, { ablations: [], taps: [], controls: [] }), programV4);
  assert.equal(hooks.hasStructuredUnitSection(programV4), false);
  assert.deepEqual(hooks.structuredTapGramHeads(programV4), []);
  assert.deepEqual(hooks.decodeStructuredUnitTables(programV4), []);
  const residualProgram = hooks.compileStructuredHookProgram(2, layers, residual);
  assert.equal(residualProgram.format, "drowse-structured-v4");
  assert.throws(
    () => hooks.withStructuredUnitSection(residualProgram, shape, units),
    (error) => error.code === "UNIT_SECTION_UNSUPPORTED" && error.status === 422,
  );
  assert.throws(
    () => hooks.validateStructuredHookProgram({ ...residualProgram, unitShape: shape, ...tables }),
    /rejects any unit section/,
  );
  assert.throws(() => hooks.validateStructuredHookProgram({ ...programV3, hookAbi: "block-sites-v5" }), /identity/);

  const withUnits = hooks.withStructuredUnitSection(programV4, shape, units);
  assert.equal(hooks.hasStructuredUnitSection(withUnits), true);
  assert.deepEqual(Array.from(withUnits.unitMeta), Array.from(tables.unitMeta));
  for (const key of typedKeys(programV4)) assert.equal(withUnits[key], programV4[key], `${key} is shared`);
  assert.deepEqual(hooks.structuredTapGramHeads(withUnits), [{ layer: 2, index: 1 }]);
  assert.equal(withUnits.controls.units.length, 4);
  assert.equal(withUnits.controls.affine.length, programV4.affineActive.length);
  assert.throws(() => hooks.withStructuredUnitSection(withUnits, shape, units), /already carries/);
  assert.throws(
    () => hooks.validateStructuredHookProgram({ ...withUnits, tapGrams: undefined }),
    /incomplete/,
  );
  const dirty = Uint32Array.from(withUnits.unitMeta);
  dirty[layout.meta.neuronIndex.offset + 200] = 1;
  assert.throws(() => hooks.validateStructuredHookProgram({ ...withUnits, unitMeta: dirty }), /zero past their last entry/);
  const decreasing = Uint32Array.from(withUnits.unitMeta);
  decreasing[layout.meta.neuronOffsets.offset + 2] = 0;
  assert.throws(() => hooks.validateStructuredHookProgram({ ...withUnits, unitMeta: decreasing }), /must not decrease/);

  // The measurement schema carries one entry per tap slot.
  const schema = {
    layerMap: [0, 1, 2, 3], modelLayerCount: 4, probes: Array(8).fill(null),
    lensSource: null, saeSource: null,
  };
  const measured = hooks.withStructuredUnitSection({ ...programV4, measurementSchema: schema }, shape, units);
  assert.equal(measured.measurementSchema.unitTaps.length, 64);
  assert.deepEqual(
    measured.measurementSchema.unitTaps.slice(0, 3).map((entry) => entry.unit),
    ["neuron/L1n5", "head/L2h1", "neuron/L3n9"],
  );
  assert.equal(measured.measurementSchema.unitTaps[3], null);

  // ---------------------------------------- triggers: `unit_active` per forward (§3.1) ---
  const triggered = hooks.withStructuredUnitSection(programV4, shape, {
    ...units,
    controls: [
      { enabled: true, phase: { kind: "prompt_only" }, gate: null },
      { enabled: true, phase: { kind: "first", tokens: 2 }, gate: null },
    ],
  });
  const controlsFor = hooks.createStructuredHookControlEvaluator(triggered);
  assert.deepEqual(Array.from(controlsFor({ prefill: true, thinking: false, generatedTokens: 0 }).unitActive), [1, 0, 0, 0]);
  assert.deepEqual(Array.from(controlsFor({ prefill: false, thinking: false, generatedTokens: 1 }).unitActive), [0, 1, 0, 0]);
  assert.deepEqual(Array.from(controlsFor({ prefill: false, thinking: false, generatedTokens: 2 }).unitActive), [0, 0, 0, 0]);
  const gated = hooks.withStructuredUnitSection(programV4, shape, {
    ablations: [{ kind: "head", layer: 2, index: 0, keep: 0, bias: headBias, group: 0 }],
    taps: [tap("neuron", 1, 5, 0)],
    controls: [{ enabled: true, phase: { kind: "generated_only" }, gate: { slots: [], scoreKey: "neuron/L1n5", operator: ">", threshold: -0.0427 } }],
  });
  const gate = (score) => Array.from(hooks.structuredHookControlsFor(gated, {
    prefill: false, thinking: false, generatedTokens: 1, priorScores: { "neuron/L1n5": score },
  }).unitActive);
  assert.deepEqual(gate(0), [1, 0, 0, 0]);
  assert.deepEqual(gate(-1), [0, 0, 0, 0]);
  assert.equal(hooks.structuredHookControlsFor(programV4, { prefill: true, thinking: false, generatedTokens: 0 }).unitActive, undefined);

  // ------------------------------------- the CPU reference of the ablation kernel (§3.1) ---
  const act = Float32Array.from({ length: 3 * 96 }, (_, i) => (i % 96) / 10 + Math.floor(i / 96));
  const ablated = hooks.runStructuredUnitSite(withUnits, 2, "act", act, [0, 1, 2]);
  assert.deepEqual(ablated.subarray(0, 96), act.subarray(0, 96), "the sink row keeps its own value");
  for (const row of [1, 2]) {
    const base = row * 96;
    assert.equal(ablated[base + 7], Math.fround(Math.fround(act[base + 7] * 0.5) + 0.25));
    assert.equal(ablated[base + 2], Math.fround(act[base + 2] * 2 + 0.5));
    assert.equal(ablated[base + 3], act[base + 3], "untouched coordinates are bit-identical");
  }
  const offGroup = hooks.runStructuredUnitSite(withUnits, 2, "act", act, [0, 1, 2], [0, 1, 0, 0]);
  assert.deepEqual(offGroup, act, "an inactive group (or a row outside a position span) is untouched");
  const z = Float32Array.from({ length: 2 * 64 }, (_, i) => i / 64);
  const zAblated = hooks.runStructuredUnitSite(withUnits, 0, "z", z, [3, 4]);
  for (const row of [0, 1]) {
    for (let i = 0; i < 64; i += 1) {
      const head = Math.floor(i / 16);
      assert.equal(zAblated[row * 64 + i], head === 1 ? headBias[i - 16] : z[row * 64 + i]);
    }
  }
  assert.deepEqual(hooks.runStructuredUnitSite(programV4, 2, "act", act, [0, 1, 2]), act);

  // ------------------- the unit-ablation goldens: native tables round-trip the v4 layout ---
  for (const arch of ["qwen3", "gemma3"]) {
    const ablation = golden(`unit-ablation-${arch}-v1.json`);
    assert.equal(ablation.hookAbi, "block-sites-v5");
    const config = tiny[arch].config;
    const archShape = { dFF: config.intermediate_size, nHeads: config.num_attention_heads, headDim: config.head_dim };
    const base = hooks.compileStructuredHookProgram(config.hidden_size, Array.from({ length: config.num_hidden_layers }, () => ({})), v4);
    let checked = 0;
    for (const testCase of ablation.cases) {
      if (testCase.tables === null) continue;
      const triggers = [];
      const nativeTables = testCase.tables.map((table) => {
        const key = JSON.stringify(table.trigger);
        if (!triggers.includes(key)) triggers.push(key);
        return { ...table, group: triggers.indexOf(key) };
      });
      const ablations = hooks.structuredUnitAblationsFromTables(nativeTables, archShape);
      const program = hooks.withStructuredUnitSection(base, archShape, {
        ablations, taps: [], controls: triggers.map(() => null),
      });
      const decoded = hooks.decodeStructuredUnitTables(program);
      const keyOf = (table) => `${table.layer}/${table.site}/${table.group}`;
      assert.deepEqual(decoded.map(keyOf).sort(), nativeTables.map(keyOf).sort(), `${arch} ${testCase.name}`);
      for (const table of nativeTables) {
        const mine = decoded.find((entry) => keyOf(entry) === keyOf(table));
        assert.equal(mine.width, table.width);
        assert.deepEqual(mine.entries, table.entries, `${arch} ${testCase.name} entries`);
        assert.deepEqual(mine.keep, table.keep.map(Math.fround), `${arch} ${testCase.name} keep`);
        assert.deepEqual(mine.bias, table.bias.map(Math.fround), `${arch} ${testCase.name} bias`);
      }
      checked += 1;
    }
    assert.ok(checked >= 7, `${arch}: every case with unit terms was checked`);

    // Native's fp32 lowering (keep = 1 − α, bias = α·mean) against the recorded tables and
    // the recorded neutral means.
    const stats = golden(`unit-stats-${arch}-v1.json`);
    const neuronMean = decodeF32(stats.neuron_mean);
    const dFF = stats.neuron_mean.shape[1];
    const coefficients = ablation.cases.find((entry) => entry.name === "coefficients");
    const act2 = coefficients.tables.find((table) => table.site === "act");
    const alphaOf = { 7: 0.5, 8: 2 };
    act2.entries.forEach((index, at) => {
      const term = hooks.structuredUnitAblationTerm("neuron", act2.layer, index, alphaOf[index], neuronMean[act2.layer * dFF + index], 0);
      assert.equal(term.keep, Math.fround(act2.keep[at]), `${arch} keep of L${act2.layer}n${index}`);
      assert.equal(term.bias, Math.fround(act2.bias[at]), `${arch} bias of L${act2.layer}n${index}`);
    });
    const head = coefficients.tables.find((table) => table.site === "z");
    const hd = archShape.headDim;
    const headIndex = head.entries[0] / hd;
    const [, heads, headWidth] = stats.head_mean.shape;
    const headStart = (head.layer * heads + headIndex) * headWidth;
    const headMean = stats.head_mean.values.slice(headStart, headStart + headWidth);
    const headTerm = hooks.structuredUnitAblationTerm("head", head.layer, headIndex, -1, headMean, 0);
    assert.equal(headTerm.keep, Math.fround(head.keep[0]));
    assert.deepEqual(Array.from(headTerm.bias), head.bias.map(Math.fround), `${arch} head bias`);
  }
  assert.throws(
    () => hooks.structuredUnitAblationsFromTables([{ layer: 0, site: "z", width: 64, group: 0, entries: [1, 2], keep: [0, 0], bias: [0, 0] }], shape),
    /whole heads/,
  );

  console.log("structured-hook units: profiles, descriptor schema 4, unit section and goldens pass");
} finally {
  await server.close();
}
