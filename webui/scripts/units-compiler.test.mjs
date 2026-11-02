// The structured compiler's unit programs against drowse's goldens, end to end: a core pack for
// the tiny fixture model (its neutral whitener), the units service's program source over the
// reference backend and B6-shaped packs (`prepare` reads each neuron's write off the device),
// then `compile`. neuron-push: each case's lowered affine group (basis, neutral, target, along)
// and trigger; unit-ablation: every case's unit tables (entries, keep, bias, width) and their
// triggers, as the encoder lays them out.
import assert from "node:assert/strict";
import { ARCHES, archGolden, viteServer } from "./support/units-goldens.mjs";
import { unitsFixture } from "./support/units-fixture.mjs";
import { unitsCompiler } from "./support/units-compiler.mjs";

const near = (actual, expected, tol, where) => {
  if (Array.isArray(expected)) {
    assert.equal(actual.length, expected.length, `${where} length`);
    expected.forEach((value, i) => near(actual[i], value, tol, `${where}[${i}]`));
    return;
  }
  assert.ok(Math.abs(actual - expected) <= tol.atol + tol.rtol * Math.abs(expected), `${where}: ${actual} != ${expected}`);
};

const server = await viteServer();
try {
  const [hooks, grammar, { BrowserInstrumentRuntime }] = await Promise.all([
    server.ssrLoadModule("/src/hosted/runtime/structuredHookProgram.ts"),
    server.ssrLoadModule("/src/hosted/runtime/steeringExpression.ts"),
    server.ssrLoadModule("/src/hosted/runtime/browserInstrumentRuntime.ts"),
  ]);
  // Native's trigger record (`Trigger` fields) of a unit or affine control.
  const trigger = (control) => {
    const fields = grammar.steeringTriggerFields(control === null ? null : { phase: control.phase, gate: null });
    return {
      prompt: fields.prompt, generated: fields.generated, thinking: fields.thinking, response: fields.response,
      first_n: fields.firstN, after_n: fields.afterN,
      gate: control?.gate ? { probe: control.gate.scoreKey, op: control.gate.operator, threshold: control.gate.threshold } : null,
    };
  };
  for (const arch of ARCHES) {
    const fx = await unitsFixture(server, arch);
    const d = fx.model.hiddenSize;
    const { compiler, compile } = await unitsCompiler(server, fx, arch);

    // A fresh slens/ probe attaches through the instrument runtime (its atom is prepared first)
    // and carries native's slens-footprint-z-v3 digest, so probeHashes (read on every
    // generation) never throws on it.
    const instruments = new BrowserInstrumentRuntime(compiler, null, null, false);
    const word = arch === "gemma3" ? "e" : "a";
    const attached = await instruments.request({ service: "probes", method: "attach", args: [{ selector: `slens/${word}` }] }, () => null);
    assert.deepEqual([attached.family, attached.namespace, attached.word], ["lens", "slens", word]);
    assert.match(instruments.probeHashes()[`slens/${word}`], /^[0-9a-f]{64}$/u);
    // A manifold refresh hands the runtime a new compiler: the unit source carries over and the
    // slens/ and unit probes re-attach through it, digests unchanged.
    await instruments.request({ service: "probes", method: "attach", args: [{ selector: "neuron/L1n5" }] }, () => null);
    const before = instruments.probeHashes();
    await instruments.replaceCompiler(await compiler.withInstalledManifolds([], new AbortController().signal));
    assert.deepEqual(instruments.probeHashes(), before);
    await instruments.request({ service: "probes", method: "detach", args: [`slens/${word}`] }, () => null);
    await instruments.request({ service: "probes", method: "detach", args: ["neuron/L1n5"] }, () => null);

    // ------------------------------------------------------------------ neuron pushes
    const push = archGolden("neuron-push", arch);
    const tol = push.tolerance.library;
    for (const entry of push.cases) {
      const program = await compile(entry.expression);
      assert.equal(program.format, "drowse-structured-v4");
      assert.equal(program.steeringExpression, entry.canonical, `${arch} ${entry.expression} canonical`);
      const G = program.profile.maxAffineGroups;
      const R = program.profile.maxRank;
      const groups = [];
      for (let layer = 0; layer < program.layerCount; layer += 1) {
        for (let g = 0; g < G; g += 1) {
          if (program.affineActive[layer * G + g] !== 1) continue;
          const at = layer * G + g;
          groups.push({
            layer,
            basis: Array.from(program.affineBasis.subarray(at * R * d, at * R * d + d)),
            neutral: Array.from(program.affineNeutral.subarray(at * d, (at + 1) * d)),
            target: program.affineTarget[at * R],
            along: program.affineAlong[at],
            control: program.controls.affine[at],
          });
        }
      }
      assert.equal(groups.length, entry.layers.length, `${arch} ${entry.expression} groups`);
      entry.layers.forEach((want, i) => {
        const where = `${arch} ${entry.expression} layer ${want.layer}`;
        // A group acts through `target − ⟨h, basis⟩` along `basis`, so (basis, target) and
        // (−basis, −target) are one push: drowse lowers a negative push with the basis flipped,
        // the browser with the target negated. Compare in drowse's orientation.
        const raw = groups[i];
        const flip = raw.basis.reduce((sum, value, j) => sum + value * want.basis[j], 0) < 0 ? -1 : 1;
        const got = { ...raw, basis: raw.basis.map((value) => flip * value), target: flip * raw.target };
        assert.equal(got.layer, want.layer, where);
        near(got.basis, want.basis, tol, `${where} basis`);
        near(got.neutral, want.neutral, tol, `${where} neutral`);
        near(got.target, want.target, tol, `${where} target`);
        near(got.along, want.along, tol, `${where} along`);
        // The push the group adds (`along · target · basis`, the golden's delta).
        near(got.basis.map((b) => got.along * got.target * b), want.delta, tol, `${where} delta`);
        assert.deepEqual(trigger(got.control), entry.trigger, `${where} trigger`);
      });
    }

    // ------------------------------------------------------------------ unit ablations
    const ablation = archGolden("unit-ablation", arch);
    for (const entry of ablation.cases) {
      const program = await compile(entry.steering);
      const tables = hooks.decodeStructuredUnitTables(program).map((table) => ({
        layer: table.layer,
        site: table.site,
        width: table.width,
        trigger: trigger(program.controls.units[table.group]),
        entries: table.entries,
        keep: table.keep,
        bias: table.bias,
      }));
      const order = (rows) => [...rows].sort((a, b) => a.layer - b.layer || a.site.localeCompare(b.site));
      const got = order(tables);
      const want = order(entry.tables);
      assert.equal(got.length, want.length, `${arch} ${entry.name} tables`);
      want.forEach((table, i) => {
        const where = `${arch} ${entry.name} L${table.layer} ${table.site}`;
        assert.deepEqual(
          { layer: got[i].layer, site: got[i].site, width: got[i].width, trigger: got[i].trigger, entries: got[i].entries },
          { layer: table.layer, site: table.site, width: table.width, trigger: table.trigger, entries: table.entries },
          where,
        );
        near(got[i].keep, table.keep, ablation.tolerance.library, `${where} keep`);
        near(got[i].bias, table.bias, ablation.tolerance.library, `${where} bias`);
      });
    }
  }
  console.log("units compiler: neuron pushes and unit ablations lower as drowse's goldens do");
} finally {
  await server.close();
}
