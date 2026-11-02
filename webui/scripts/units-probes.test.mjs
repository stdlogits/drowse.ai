// Live unit taps (B1 §2.2) on the reference backend against drowse's unit-probes golden: each
// forward's tap values over their reading scales (the units channel's readings), under unit
// ablations and a unit gate evaluated from the previous forward's taps, as the decode loop does.
import assert from "node:assert/strict";
import { ARCHES, archGolden, compareRecord, decode, viteServer } from "./support/units-goldens.mjs";
import { unitsFixture } from "./support/units-fixture.mjs";

const server = await viteServer();
try {
  const [hooks, profiles, scalesModule, program, reader] = await Promise.all([
    server.ssrLoadModule("/src/hosted/runtime/structuredHookProgram.ts"),
    server.ssrLoadModule("/src/hosted/runtime/structuredHookProfile.ts"),
    server.ssrLoadModule("/src/hosted/runtime/units/scales.ts"),
    server.ssrLoadModule("/src/hosted/runtime/units/unitProgram.ts"),
    server.ssrLoadModule("/src/hosted/runtime/units/unitReader.ts"),
  ]);
  const profile = profiles.STANDARD_STRUCTURED_HOOK_PROFILE_V4;
  for (const arch of ARCHES) {
    const fx = await unitsFixture(server, arch);
    const golden = archGolden("unit-probes", arch);
    const stats = fx.stats;
    const statistics = {
      documents: stats.n_documents,
      neuronMean: Float32Array.from(decode(stats.neuron_mean)),
      neuronScale: Float32Array.from(decode(stats.neuron_scale)),
      headMean: Float32Array.from(decode(stats.head_mean)),
      headScale: Float32Array.from(decode(stats.head_scale)),
    };
    const source = { spec: fx.model.spec, refusal: null, layerCount: 4, statistics, scales: scalesModule.readingScales(statistics, fx.model.spec, 4), neuronWrite: () => null, slensAtom: () => null };
    const parse = (name) => reader.parseUnit(name, fx.model.spec, 4);
    for (const entry of golden.cases) {
      const tapped = [...golden.probes];
      const ablations = [];
      const controls = [];
      if (entry.steering) {
        const match = /^!(\S+?)(?:@when:(\S+) ([<>]=?) (\S+))?$/.exec(entry.steering);
        assert.ok(match, entry.steering);
        const unit = parse(match[1]);
        ablations.push(hooks.structuredUnitAblationTerm(unit.kind, unit.layer, unit.index, 1, program.unitMean(source, unit), 0));
        if (match[2]) {
          if (!tapped.includes(match[2])) tapped.push(match[2]);
          controls.push({ enabled: true, phase: { kind: "both" }, gate: { slots: [], scoreKey: match[2], operator: match[3], threshold: Number(match[4]) } });
        } else controls.push(null);
      }
      const taps = tapped.map((name, slot) => program.unitTap(source, name, parse(name), slot));
      const base = hooks.compileStructuredHookProgram(64, [{}, {}, {}, {}], profile);
      base.measurementSchema = {
        layerMap: [0, 1, 2, 3], modelLayerCount: 4, probes: Array.from({ length: profile.maxProbes }, () => null),
        lensSource: null, saeSource: null, saeLayer: null, saeFeatureCount: null, lensReadout: false, saeReadout: false,
      };
      const bound = await fx.backend.bindStructuredProgram(hooks.withStructuredUnitSection(base, { dFF: 96, nHeads: 4, headDim: 16 }, { ablations, taps, controls }));
      const evaluate = hooks.createStructuredHookControlEvaluator(bound);
      const replay = await fx.backend.openReplay({ program: bound, anatomyPosition: null, activity: null, patternLayers: null });
      const inputs = [golden.ids, ...entry.tokens.slice(0, entry.forwards.length - 1).map((token) => [token])];
      let readback = null;
      for (const [step, ids] of inputs.entries()) {
        const priorScores = readback === null ? null : program.unitGateScores(bound, readback);
        if (step > 0 && entry.gate_scalars.length) {
          const recorded = entry.gate_scalars[step - 1];
          for (const [name, value] of Object.entries(recorded)) {
            if (name in priorScores) assert.ok(Math.abs(priorScores[name] - value) <= 1e-4, `${arch} ${entry.name} gate scalar ${name} at step ${step}`);
          }
        }
        const controlState = evaluate({ prefill: step === 0, thinking: false, generatedTokens: step, priorScores });
        readback = await replay.forward({ tokenIds: Int32Array.from(ids), controls: controlState });
        const { readings } = program.unitTapReadings(bound.measurementSchema.unitTaps, readback.unitTaps.data, 4);
        const expected = entry.forwards[step].readings;
        const actual = Object.fromEntries(Object.keys(expected).map((name) => [name, readings[name]]));
        const problems = compareRecord(actual, expected, { numberTol: () => golden.tolerance.library.z_atol + 2e-6 });
        assert.deepEqual(problems, [], `${arch} ${entry.name} forward ${step}`);
      }
      await replay.cancel();
    }
  }
  console.log("units probes: live taps, readings and unit gates match drowse's unit-probes");
} finally {
  await server.close();
}
