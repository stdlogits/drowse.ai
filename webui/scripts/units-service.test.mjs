// The worker's units members end to end: `instruments.<member>` through BrowserInstrumentRuntime
// and BrowserUnitsService over the reference backend and fixture packs, answering drowse's
// goldens; native's argument refusals; a residual library's 422 refusal on every member but the
// (disk-only) catalog; the operation policy offering every member.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { ARCHES, archGolden, compareRecord, viteServer } from "./support/units-goldens.mjs";
import { unitsFixture, unitsPackSource } from "./support/units-fixture.mjs";

const whiteners = JSON.parse(readFileSync(fileURLToPath(new URL("../../browser-runtime/fixtures/units-tiny-whitener-v1.json", import.meta.url)), "utf8"));
const server = await viteServer();
try {
  const [service, runtime, policy, profiles, cpu] = await Promise.all([
    server.ssrLoadModule("/src/hosted/runtime/units/unitsService.ts"),
    server.ssrLoadModule("/src/hosted/runtime/browserInstrumentRuntime.ts"),
    server.ssrLoadModule("/src/hosted/runtime/operationPolicy.ts"),
    server.ssrLoadModule("/src/hosted/runtime/structuredHookProfile.ts"),
    server.ssrLoadModule("/src/hosted/runtime/units/cpuBackend.ts"),
  ]);
  // Withheld from the hosted client until the model load wires this service (B4).
  for (const method of service.UNITS_SERVICE_METHODS) {
    assert.equal(policy.isOfferedRuntimeMethod("instruments", method), false, `${method} is withheld`);
  }
  const stubCompiler = {
    instrumentDescriptor: () => ({ jlens: null, sae: null }),
    setInstrumentLive() {},
  };
  const arch = "gemma3";
  const fx = await unitsFixture(server, arch);
  const packs = unitsPackSource(fx, arch, whiteners);
  const chat = fx.record.chat_prompt.ids;
  const anatomyGolden = archGolden("token-anatomy", arch);
  const activityGolden = archGolden("unit-activity", arch);
  const generatedRows = activityGolden.cases[0].result.tokens.map((t) => t.token_id);
  const replay = {
    async nodeForward(nodeId, rawIndex, options) {
      assert.equal(nodeId, "node-1");
      const anchor = rawIndex ?? generatedRows.length - 1;
      return {
        forward: {
          ids: Int32Array.from([...chat, ...generatedRows.slice(0, anchor)]),
          promptLength: chat.length,
          rawIds: Int32Array.from(generatedRows),
          rawIndex: anchor,
          spans: [{ firstRow: 0, stopRow: generatedRows.length, at: chat.length - 1 }],
        },
        program: null,
        steering: null,
        canThink: false,
        options,
      };
    },
  };
  const units = new service.BrowserUnitsService({
    modelId: "units-tiny",
    blockSites: () => fx.backend,
    packs: () => packs,
    decode: (id) => fx.tokenizer.decode([id]),
    resolveToken: (word) => (word.startsWith("#") ? Number(word.slice(1)) : fx.tokenizer.encode(word)[0]),
    replay: () => replay,
  });
  const instruments = new runtime.BrowserInstrumentRuntime(stubCompiler, null, null, false);
  const call = (method, ...args) => instruments.request({ service: "instruments", method, args }, () => null);

  // Before a service is configured: native's 422 for a library without unit sites.
  await assert.rejects(call("unitRead", "neuron/L1n5"), (error) => error.status === 422 && error.code === "UnsupportedBlockError");
  instruments.configureUnits(units);

  // ------------------------------------------------------------------ static reads ---
  const reads = archGolden("unit-reads", arch);
  const numberTol = () => 1e-4 + 2e-6;
  for (const [i, entry] of reads.reads.slice(0, 3).entries()) {
    const actual = await call("unitRead", entry.unit, { k: entry.k, hideHubs: entry.hide_hubs });
    assert.deepEqual(compareRecord(actual, entry.read, { root: `reads[${i}].read`, ties: reads.declared_ties, numberTol }), [], `unitRead ${entry.unit}`);
  }
  const top = reads.top[0];
  assert.deepEqual(compareRecord(await call("unitTop", top.token, { by: top.by, k: top.k, layers: "all" }), top.result, { root: "top[0].result", ties: reads.declared_ties, numberTol }), []);
  const find = reads.find[0];
  assert.deepEqual(compareRecord(await call("unitFind", { vector: find.vector, layer: find.layer, k: find.k }), find.result, { root: "find[0].result", ties: reads.declared_ties, numberTol }), []);
  assert.equal((await call("unitCatalog")).reason, "no neuron-lens catalog for this model (catalogued: Qwen/Qwen3-8B, google/gemma-4-12B)");
  assert.deepEqual((await call("unitLabels", ["neuron/L1n5", "head/L2h0"])).labels, []);

  // ----------------------------------------------------------- native's refusals ---
  const refuses = async (promise, text) => assert.rejects(promise, (error) => error.status === 400 && error.message === text);
  await refuses(call("unitFind", {}), "find takes exactly one of selector or vector");
  await refuses(call("unitFind", { vector: [1] }), "a vector needs the layer it lives at");
  await refuses(call("unitLabels", []), "units names at least one neuron/… or head/… unit");
  await refuses(call("unitLabels", Array.from({ length: 129 }, () => "neuron/L1n5")), "units names at most 128 units");
  await refuses(call("unitRead", "neuron/L1n5", { k: 257 }), "k must be in [1, 256]");
  await refuses(call("unitTop", "Q"), `'Q' is token ${fx.tokenizer.encode("Q")[0]}, which the token mask drops (special, added and byte-fallback pieces and tokens too rare in the counted text never rank; mask: corpus)`);
  await refuses(call("tokenAnatomy", "node-1", 3, { units: 65 }), "units must be in [1, 64]");
  await refuses(call("tokenAnatomy", "node-1", 3, { patterns: 17 }), "patterns must be in [0, 16]");
  await refuses(call("unitActivity", "node-1", { top: 0 }), "top must be in [1, 1024]");
  await refuses(call("unitActivity", "node-1", { layers: "0,2" }), "layers selects a population, which needs raw_index");
  await refuses(call("unitActivity", "node-1", { units: ["neuron/L1n5", "neuron/L1n5"] }), "units names a unit twice");
  // Layer selectors parse as native `parse_layer_selector` (modes any case, blank is the
  // default, written order, a layer named twice refused) and resolve as `resolve_layer_selector`.
  const topAll = await call("unitTop", top.token, { by: top.by, k: top.k, layers: "ALL" });
  assert.deepEqual(await call("unitTop", top.token, { by: top.by, k: top.k, layers: "  " }), topAll);
  assert.deepEqual(await call("unitTop", top.token, { by: top.by, k: top.k, layers: "3,0-2" }), topAll);
  assert.ok(await call("unitTop", top.token, { by: top.by, k: top.k, layers: "Workspace" }));
  await refuses(call("unitTop", "e", { layers: "1,1" }), "layer 1 is named twice in '1,1'");
  await refuses(call("unitTop", "e", { layers: "2-1" }), "reversed layer range '2-1' (write 1-2)");
  await refuses(call("unitTop", "e", { layers: "1,,2" }), "empty entry in layers '1,,2'");
  await refuses(call("unitTop", "e", { layers: "-1" }), "layer indices are non-negative; got '-1'");
  await refuses(call("unitTop", "e", { layers: "x" }), "malformed layers 'x' (want comma-separated indices or a-b ranges, or one of all, workspace, sample)");
  await refuses(call("unitTop", "e", { layers: "sample" }), "unknown layer mode 'sample' (want all, workspace, or an index list)");
  await refuses(call("unitTop", "e", { layers: "9" }), "layer 9 is outside the model's 4 layers");
  await refuses(call("unitFind", { vector: find.vector, layer: find.layer, layers: "0" }), "no overlap between 'vector''s layers and the requested layers");
  await refuses(call("unitActivity", "node-1", { rawIndex: 0, layers: "sample" }), "unknown layer mode 'sample' (want all, workspace, or an index list)");
  await refuses(call("unitActivity", "node-1", { rawIndex: 0, layers: "7" }), "layers must lie in [0, 4); got [7]");

  // ------------------------------------------------------------------ node replays ---
  const anatomyCase = anatomyGolden.cases[0];
  const envelope = await call("tokenAnatomy", "node-1", anatomyGolden.raw_index, anatomyCase.knobs);
  assert.equal(envelope.measurements.scope, "replay");
  assert.deepEqual(envelope.measurements.anatomy.token, anatomyCase.envelope.measurements.anatomy.token);
  assert.equal(envelope.measurements.anatomy.units.length, anatomyCase.knobs.units);
  const generated = activityGolden.cases[0];
  const activity = await call("unitActivity", "node-1", {
    top: generated.arguments.top, units: generated.arguments.units, rawIndex: generated.arguments.raw_index, layers: generated.arguments.layers,
  });
  assert.equal(activity.node_id, "node-1");
  const { node_id: _node, population: _population, ...rest } = activity;
  const { population: _expected, ...expected } = generated.result;
  assert.deepEqual(compareRecord(rest, expected, { root: "cases[0].result", ties: activityGolden.declared_ties, numberTol }), []);

  // ------------------------------------------------------- a residual library (422) ---
  const residualBackend = cpu.cpuBlockSitesBackend(fx.model, { profile: profiles.STANDARD_STRUCTURED_HOOK_PROFILE_V4_RESIDUAL, contextTokens: 64 });
  const residual = new service.BrowserUnitsService({
    modelId: "units-tiny",
    blockSites: () => residualBackend,
    packs: () => packs,
    decode: (id) => fx.tokenizer.decode([id]),
    resolveToken: () => 0,
    replay: () => replay,
  });
  const refusal = residualBackend.capabilities().unitRefusal;
  for (const [method, args] of [
    ["unitRead", ["neuron/L1n5"]], ["unitTop", ["e"]], ["unitFind", [{ vector: [1], layer: 0 }]],
    ["unitLabels", [["neuron/L1n5"]]], ["tokenAnatomy", ["node-1", 0]], ["unitActivity", ["node-1"]],
  ]) {
    await assert.rejects(residual.request(method, args), (error) => error.status === 422 && error.message === refusal, `${method} refuses`);
  }
  assert.equal((await residual.request("unitCatalog", [])).available, false);
  console.log("units service: every member answers like drowse; residual libraries refuse");
} finally {
  await server.close();
}
