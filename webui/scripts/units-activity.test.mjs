// Unit activity (native `NodeReplayer.unit_activity`) on the reference backend's replay against
// drowse's unit-activity golden (B1-goldens §2.5): every case's forward rebuilt from the model
// file's chat prompt and the case's rows, replay segments from the browser program.
import assert from "node:assert/strict";
import { ARCHES, archGolden, compareRecord, decode, viteServer } from "./support/units-goldens.mjs";
import { unitsFixture } from "./support/units-fixture.mjs";

const RAW_FIRST = { gemma3: "=", qwen3: "K" };

function f16(bits) {
  const s = bits & 0x8000 ? -1 : 1;
  const e = (bits >>> 10) & 0x1f;
  const m = bits & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}
const halfBits = (text) => {
  const bytes = Buffer.from(text, "base64");
  return Array.from({ length: bytes.length / 2 }, (_, i) => bytes[2 * i] | (bytes[2 * i + 1] << 8));
};
// f16 bit patterns on one ordered line (sign-magnitude to two's complement), so neighbours,
// zero's two signs included, are one apart: "within one f16 ulp" is |Δ| ≤ 1 there.
const ordered = (bits) => (bits & 0x8000 ? -(bits & 0x7fff) : bits);

const server = await viteServer();
try {
  const [replayModule, scalesModule, hooks, profiles, readerModule] = await Promise.all([
    server.ssrLoadModule("/src/hosted/runtime/units/nodeReplay.ts"),
    server.ssrLoadModule("/src/hosted/runtime/units/scales.ts"),
    server.ssrLoadModule("/src/hosted/runtime/structuredHookProgram.ts"),
    server.ssrLoadModule("/src/hosted/runtime/structuredHookProfile.ts"),
    server.ssrLoadModule("/src/hosted/runtime/units/unitReader.ts"),
  ]);
  for (const arch of ARCHES) {
    const fx = await unitsFixture(server, arch);
    const golden = archGolden("unit-activity", arch);
    const stats = fx.stats;
    const statistics = {
      documents: stats.n_documents,
      neuronMean: Float32Array.from(decode(stats.neuron_mean)),
      neuronScale: Float32Array.from(decode(stats.neuron_scale)),
      headMean: Float32Array.from(decode(stats.head_mean)),
      headScale: Float32Array.from(decode(stats.head_scale)),
    };
    const scales = scalesModule.readingScales(statistics, fx.model.spec, 4);
    const chat = Int32Array.from(fx.record.chat_prompt.ids);
    const shape = { dFF: 96, nHeads: 4, headDim: 16 };
    const emptyProgram = () => hooks.compileStructuredHookProgram(64, [{}, {}, {}, {}], profiles.STANDARD_STRUCTURED_HOOK_PROFILE_V4);
    const unitProgram = (ablation, phase) => hooks.withStructuredUnitSection(emptyProgram(), shape, {
      ablations: [ablation], taps: [], controls: [phase === "both" ? null : { enabled: true, phase: { kind: phase }, gate: null }],
    });
    const indexOf = (haystack, needle) => {
      for (let i = 0; i + needle.length <= haystack.length; i += 1) if (needle.every((x, j) => haystack[i + j] === x)) return i;
      return -1;
    };
    for (const [c, entry] of golden.cases.entries()) {
      const args = entry.arguments;
      const rows = entry.result.tokens.map((t) => t.token_id);
      let forward;
      let program = null;
      if (args.ids) {
        const runs = args.runs;
        forward = {
          ids: Int32Array.from(args.ids),
          promptLength: args.ids.length,
          rawIds: Int32Array.from(args.raw_ids),
          spans: runs.map(([first, position], i) => ({ firstRow: first, stopRow: i + 1 < runs.length ? runs[i + 1][0] : args.raw_ids.length, at: position - 1 })),
        };
        if (entry.name === "excluded-and-zero") program = unitProgram({ kind: "neuron", layer: 1, index: 5, keep: 0, bias: 0, group: 0 }, "both");
      } else if (entry.name.startsWith("sink")) {
        const first = fx.tokenizer.encode(RAW_FIRST[arch]);
        forward = { ids: Int32Array.from([...first, ...rows.slice(0, -1)]), promptLength: first.length, rawIds: Int32Array.from(rows), spans: [{ firstRow: 0, stopRow: rows.length, at: first.length - 1 }] };
      } else if (entry.name === "authored") {
        const position = indexOf([...chat], rows);
        assert.ok(position > 0, "the authored rows sit in the chat render");
        const ids = chat.slice(0, position + rows.length - 1);
        forward = { ids, promptLength: ids.length, rawIds: Int32Array.from(rows), spans: [{ firstRow: 0, stopRow: rows.length, at: position - 1 }] };
      } else {
        forward = { ids: Int32Array.from([...chat, ...rows.slice(0, -1)]), promptLength: chat.length, rawIds: Int32Array.from(rows), spans: [{ firstRow: 0, stopRow: rows.length, at: chat.length - 1 }] };
        if (entry.name === "segmented") {
          program = unitProgram(hooks.structuredUnitAblationTerm("neuron", 1, 5, 1, statistics.neuronMean[96 + 5], 0), "prompt_only");
        }
      }
      const units = (args.units ?? []).map((name) => readerModule.parseUnit(name, fx.model.spec, 4));
      const layers = args.layers ? String(args.layers).split(",").map(Number) : null;
      const actual = await replayModule.unitActivity(fx.backend, { nLayers: 4, spec: fx.model.spec }, scales, forward, program, {
        top: args.top ?? 32, units, rawIndex: args.raw_index ?? null, layers,
      }, (id) => fx.tokenizer.decode([id]));
      // unitActivity never names the steering (the service adds the loom node's); the program
      // this test lowered by hand stands for the case's recorded expression.
      assert.equal(entry.result.steering, entry.name === "segmented" ? "!neuron/L1n5@before" : null);
      assert.equal("steering" in actual, false);
      const expectedPopulation = entry.result.population;
      const actualPopulation = actual.population;
      const { steering: _steering, ...result } = { ...entry.result, population: null };
      actual.population = null;
      const tol = golden.tolerance.library;
      const problems = compareRecord(actual, result, {
        root: `cases[${c}].result`,
        ties: golden.declared_ties,
        numberTol: (path) => /(peak|traces\.[^.]+\[\d+\])$/.test(path) || /traces/.test(path) ? tol.z_atol + 2e-6 : 2e-6,
      });
      assert.deepEqual(problems, [], `${arch} unit activity ${entry.name}`);
      if (expectedPopulation === null) assert.equal(actualPopulation, null);
      else {
        assert.equal(actualPopulation.raw_index, expectedPopulation.raw_index);
        assert.equal(actualPopulation.encoding, expectedPopulation.encoding);
        assert.deepEqual(actualPopulation.layers.map((l) => l.layer), expectedPopulation.layers.map((l) => l.layer));
        assert.equal(actualPopulation.layers.length, expectedPopulation.layers.length);
        actualPopulation.layers.forEach((layer, i) => {
          for (const key of ["neurons", "heads"]) {
            const got = halfBits(layer[key]);
            const want = halfBits(expectedPopulation.layers[i][key]);
            assert.equal(got.length, want.length, `${arch} ${entry.name} population L${layer.layer} ${key} length`);
            got.forEach((bits, j) => assert.ok(
              Math.abs(ordered(bits) - ordered(want[j])) <= 1,
              `${arch} ${entry.name} population L${layer.layer} ${key}[${j}] ${f16(bits)} vs ${f16(want[j])}`,
            ));
          }
        });
      }
    }
  }
  console.log("units activity: every unit-activity case matches drowse");

  // Position-targeted steering (browser only, B1 §2.3/§3.1): the replay splits its segments at
  // the recorded spans and nothing fires outside them, so `!neuron/L1n5` over [p, p+2) changes
  // only rows produced at or after p, its own rows read the ablated mean, and a span over every
  // position replays as the unrestricted program.
  {
    const arch = "gemma3";
    const fx = await unitsFixture(server, arch);
    const golden = archGolden("unit-activity", arch);
    const stats = fx.stats;
    const statistics = {
      documents: stats.n_documents,
      neuronMean: Float32Array.from(decode(stats.neuron_mean)),
      neuronScale: Float32Array.from(decode(stats.neuron_scale)),
      headMean: Float32Array.from(decode(stats.head_mean)),
      headScale: Float32Array.from(decode(stats.head_scale)),
    };
    const scales = scalesModule.readingScales(statistics, fx.model.spec, 4);
    const chat = Int32Array.from(fx.record.chat_prompt.ids);
    const rows = golden.cases.find((entry) => entry.name === "generated").result.tokens.map((t) => t.token_id);
    const base = {
      ids: Int32Array.from([...chat, ...rows.slice(0, -1)]),
      promptLength: chat.length,
      rawIds: Int32Array.from(rows),
      spans: [{ firstRow: 0, stopRow: rows.length, at: chat.length - 1 }],
    };
    const ablate = () => hooks.withStructuredUnitSection(
      hooks.compileStructuredHookProgram(64, [{}, {}, {}, {}], profiles.STANDARD_STRUCTURED_HOOK_PROFILE_V4),
      { dFF: 96, nHeads: 4, headDim: 16 },
      { ablations: [hooks.structuredUnitAblationTerm("neuron", 1, 5, 1, statistics.neuronMean[96 + 5], 0)], taps: [], controls: [null] },
    );
    const units = [readerModule.parseUnit("neuron/L1n5", fx.model.spec, 4)];
    const run = (forward, program) => replayModule.unitActivity(fx.backend, { nLayers: 4, spec: fx.model.spec }, scales, forward, program, {
      top: 4, units, rawIndex: null, layers: null,
    }, (id) => fx.tokenizer.decode([id]));
    const p = chat.length + 1;
    assert.deepEqual(
      replayModule.replaySegments(ablate(), base.promptLength, base.ids.length, false, [[p, p + 2]]).map((seg) => [seg.start, seg.stop, seg.inSpan]),
      [[0, p, false], [p, p + 2, true], [p + 2, base.ids.length, false]],
    );
    const plain = (await run(base, null)).traces["neuron/L1n5"];
    const full = (await run(base, ablate())).traces["neuron/L1n5"];
    const everywhere = (await run({ ...base, positionSpans: [[0, base.ids.length]] }, ablate())).traces["neuron/L1n5"];
    assert.deepEqual(everywhere, full);
    const spanned = (await run({ ...base, positionSpans: [[p, p + 2]] }, ablate())).traces["neuron/L1n5"];
    // Row r is produced at column chat.length - 1 + r: rows 0 and 1 precede the span.
    assert.deepEqual(spanned.slice(0, 2), plain.slice(0, 2));
    assert.deepEqual(spanned.slice(2, 4), full.slice(2, 4));
    assert.notDeepEqual(spanned.slice(2, 4), plain.slice(2, 4));
    console.log("units activity: position-targeted steering replays with its spans");
  }
} finally {
  await server.close();
}
