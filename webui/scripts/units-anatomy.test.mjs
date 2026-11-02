// Token anatomy (native `NodeReplayer.token_anatomy`) on the reference backend against drowse's
// token-anatomy golden (B1-goldens §2.8): unsteered, under `0.3 neuron/L2n7 + !head/L1h3`, and
// with an R lens holding no `mid` layer.
import assert from "node:assert/strict";
import { ARCHES, archGolden, assertClose, compareRecord, decode, viteServer } from "./support/units-goldens.mjs";
import { unitsFixture } from "./support/units-fixture.mjs";

const server = await viteServer();
try {
  const [anatomyModule, scalesModule, hooks, profiles] = await Promise.all([
    server.ssrLoadModule("/src/hosted/runtime/units/anatomy.ts"),
    server.ssrLoadModule("/src/hosted/runtime/units/scales.ts"),
    server.ssrLoadModule("/src/hosted/runtime/structuredHookProgram.ts"),
    server.ssrLoadModule("/src/hosted/runtime/structuredHookProfile.ts"),
  ]);
  for (const arch of ARCHES) {
    const golden = archGolden("token-anatomy", arch);
    const push = archGolden("neuron-push", arch);
    const chatLength = (await unitsFixture(server, arch)).record.chat_prompt.ids.length;
    for (const [c, entry] of golden.cases.entries()) {
      const fx = await unitsFixture(server, arch, entry.name === "no-r-mid" ? { drop: ["R:mid"] } : {});
      const stats = fx.stats;
      const statistics = {
        documents: stats.n_documents,
        neuronMean: Float32Array.from(decode(stats.neuron_mean)),
        neuronScale: Float32Array.from(decode(stats.neuron_scale)),
        headMean: Float32Array.from(decode(stats.head_mean)),
        headScale: Float32Array.from(decode(stats.head_scale)),
      };
      const scales = scalesModule.readingScales(statistics, fx.model.spec, 4);
      const ids = Int32Array.from(entry.intermediates.ids);
      const rawIds = Int32Array.from([...ids.slice(chatLength), entry.intermediates.token_id]);
      const forward = { ids, promptLength: chatLength, rawIds, rawIndex: golden.raw_index, spans: [{ firstRow: 0, stopRow: rawIds.length, at: chatLength - 1 }] };
      let program = null;
      if (entry.steering) {
        const group = push.cases[0].layers[0];
        const layers = [{}, {}, {}, {}];
        layers[group.layer] = { affineGroups: [{ active: true, basis: [Float32Array.from(group.basis)], neutral: Float32Array.from(group.neutral), target: [group.target], along: group.along, kappa: [0] }] };
        const base = hooks.compileStructuredHookProgram(64, layers, profiles.STANDARD_STRUCTURED_HOOK_PROFILE_V4);
        program = hooks.withStructuredUnitSection(base, { dFF: 96, nHeads: 4, headDim: 16 }, {
          ablations: [hooks.structuredUnitAblationTerm("head", 1, 3, 1, statistics.headMean.subarray((1 * 4 + 3) * 16, (1 * 4 + 4) * 16), 0)],
          taps: [],
          controls: [null],
        });
      }
      const { anatomy } = await anatomyModule.tokenAnatomy(fx.context, { spec: fx.model.spec, scales, backend: fx.backend }, forward, program, entry.knobs, {
        sources: { R: entry.lens, S: "local:s" }, steering: entry.steering,
      });
      const envelope = { measurements: { version: 2, scope: "replay", provenance: "replayed", instruments: {}, anatomy } };
      const tol = golden.tolerance.library;
      const problems = compareRecord(envelope, entry.envelope, {
        root: `cases[${c}].envelope`,
        ties: golden.declared_ties,
        numberTol: (path) => /\.(value|contribution|activation|activation_over_scale)$/.test(path) ? tol.z_atol + 2e-6
          : /(probability|weights\[\d+\])$/.test(path) ? tol.probability_atol + 2e-6 : 2e-6,
      });
      assert.deepEqual(problems, [], `${arch} token anatomy ${entry.name}`);
    }
  }
  console.log("units anatomy: every token-anatomy case matches drowse");
} finally {
  await server.close();
}
