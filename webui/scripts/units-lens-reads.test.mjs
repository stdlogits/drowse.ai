// The four-lens read, the S readout and the reading scales on the reference backend against
// drowse's lens-reads, slens and unit-stats goldens (B1-goldens §2.2, §2.6, §2.7).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { ARCHES, archGolden, assertClose, compareRecord, decode, viteServer } from "./support/units-goldens.mjs";
import { unitsFixture } from "./support/units-fixture.mjs";

const whiteners = JSON.parse(readFileSync(fileURLToPath(new URL("../../browser-runtime/fixtures/units-tiny-whitener-v1.json", import.meta.url)), "utf8"));
const server = await viteServer();
try {
  const { raiseLensRead } = await server.ssrLoadModule("/src/hosted/runtime/browserCorePack.ts");
  const reads = await server.ssrLoadModule("/src/hosted/runtime/units/lensReads.ts");
  const slensPort = await server.ssrLoadModule("/src/hosted/runtime/units/slens.ts");
  const scalesPort = await server.ssrLoadModule("/src/hosted/runtime/units/scales.ts");
  const dense = await server.ssrLoadModule("/src/hosted/runtime/units/denseModel.ts");
  for (const arch of ARCHES) {
    const fx = await unitsFixture(server, arch);
    const slens = archGolden("slens", arch);
    // ------------------------------------------------- mask and statistics (§2.7) ---
    const kept = [...fx.keep.keys()].filter((v) => fx.keep[v]);
    assert.deepEqual(kept, slens.mask.kept, `${arch} token mask`);
    const embeddings = await fx.context.embeddings();
    assertClose(embeddings.statistics.meanE.data, decode(slens.mean_E), slens.tolerance.library, `${arch} mean_E`);
    assertClose(embeddings.statistics.sigmaE.data, decode(slens.sigma_E), slens.tolerance.library, `${arch} Σ_E`);

    // --------------------------------------------------------- lens reads (§2.6) ---
    const golden = archGolden("lens-reads", arch);
    const noS = await unitsFixture(server, arch, { families: ["R"] });
    const tol = golden.tolerance.library;
    for (const [i, request] of golden.requests.entries()) {
      const context = request.lens === "none prepared" ? noS.context : fx.context;
      const [actual] = await reads.readVectors(context, [{
        rows: Float32Array.from(decode(request.rows)),
        layer: request.layer,
        site: request.site,
        of: request.of,
      }], request.k);
      // The browser's missing-lens text names the pack to install, not native's CLI command.
      const expected = request.lens === "none prepared"
        ? request.reads.map((read) => ({ ...read, s: { ...read.s, absent: context.noLens.S } }))
        : request.reads;
      if (request.lens === "none prepared") assert.match(request.reads[0].s.absent, /^no S-lens available for <model>/);
      const problems = compareRecord(actual, expected, {
        root: `requests[${i}].reads`,
        ties: golden.declared_ties,
        numberTol: (path, expected) => path.endsWith(".value") && /\.s\.tokens/.test(path)
          ? tol.z_atol + 2e-6
          : path.endsWith(".value") ? tol.probability_atol + 2e-6 : 2e-6,
      });
      assert.deepEqual(problems, [], `${arch} lens read ${i} (${request.of} L${request.layer} ${request.site})`);
    }

    // ------------------------------------------- footprints, probes, atoms (§2.7) ---
    const zTol = { atol: slens.tolerance.library.z_atol, rtol: 0 };
    for (const entry of slens.footprints) {
      const u = Float32Array.from(decode(entry.u));
      const embeddingsResident = (await fx.context.embeddings()).resident;
      const top = await fx.backend.footprintTopK({ dtype: "f32", shape: [u.length / 64, 64], data: u }, entry.k, embeddingsResident);
      entry.top.forEach((row, r) => {
        assert.deepEqual(Array.from(top.ids.data.slice(r * entry.k, (r + 1) * entry.k)), row.map((t) => t.token_id), `${arch} footprint L${entry.layer} row ${r} ids`);
        assertClose(top.z.data.slice(r * entry.k, (r + 1) * entry.k), row.map((t) => t.z), zTol, `${arch} footprint z`);
      });
    }
    const sequence = new dense.DenseSequence(fx.model, 64);
    const outs = [];
    sequence.forward(fx.record.short_prompt, { capture: (layer, site) => { outs[layer] = site.out; } });
    const ids = Int32Array.from(slens.probe_word_ids);
    const positions = [2, 5, 11];
    const columns = [];
    for (const position of positions) {
      const hidden = new Float32Array(4 * 64);
      outs.forEach((out, layer) => hidden.set(out.subarray(position * 64, (position + 1) * 64), layer * 64));
      columns.push(await slensPort.slensProbeRows(fx.context, hidden, ids));
    }
    assertClose(columns[0].depths, decode(slens.probe_depths), { atol: 1e-7 }, `${arch} probe depths`);
    assertClose(columns[0].weights, decode(slens.probe_weights), { atol: 1e-7 }, `${arch} probe weights`);
    slens.probes.forEach((probe, p) => {
      const expected = decode(probe.rows);
      const rows = expected.length / positions.length;
      const actual = Float32Array.from({ length: expected.length }, (_, i) => columns[i % positions.length].rows[p][Math.floor(i / positions.length)]);
      assert.equal(rows, 3 + columns[0].layers.length);
      assertClose(actual, expected, zTol, `${arch} slens/${probe.word} probe rows`);
    });
    for (const atom of slens.atoms) {
      const word = atom.atom.slice("slens/".length);
      const id = word.startsWith("#") ? Number(word.slice(1)) : fx.tokenizer.encode(word)[0];
      const directions = await slensPort.slensAtomDirections(fx.context, word, id);
      assert.deepEqual([...directions.keys()].map(String), Object.keys(atom.layers));
      // The footprint is a read; drowse's profile is the vector it raises to (Σu, which the
      // compiler applies when it resolves the atom), under the fixture's neutral whitener.
      for (const [layer, values] of Object.entries(atom.layers)) {
        const entry = whiteners.arches[arch].find((row) => row.layer === Number(layer));
        const whitener = {
          columns: 64, rank: entry.eigenvalues.length, ridge: entry.ridge,
          basis: Float64Array.from(entry.basis.flat()), eigenvalues: Float64Array.from(entry.eigenvalues),
        };
        const raised = Float32Array.from(raiseLensRead(whitener, directions.get(Number(layer))));
        assertClose(raised, decode(values), slens.tolerance.library, `${arch} ${atom.atom} L${layer}`);
      }
    }
    await assert.rejects(
      slensPort.slensAtomDirections(fx.context, `#${slens.refusal.token_id}`, slens.refusal.token_id),
      (error) => error.code === slens.refusal.code && error.status === slens.refusal.status && error.message === slens.refusal.text,
    );

    // ------------------------------------------------------- reading scales (§2.2) ---
    const stats = fx.stats;
    const unitStats = {
      documents: stats.n_documents,
      neuronMean: Float32Array.from(decode(stats.neuron_mean)),
      neuronScale: Float32Array.from(decode(stats.neuron_scale)),
      headMean: Float32Array.from(decode(stats.head_mean)),
      headScale: Float32Array.from(decode(stats.head_scale)),
    };
    const scales = scalesPort.readingScales(unitStats, fx.model.spec, 4);
    assert.equal(scales.source, stats.scales.source);
    assert.equal(scales.detail, stats.scales.detail);
    assertClose(scales.neuron, decode(stats.scales.neuron), stats.tolerance.library, `${arch} neuron scales`);
    assertClose(scales.head, decode(stats.scales.head), stats.tolerance.library, `${arch} head scales`);
    assertClose(scales.typical.neuron, stats.scales.typical.neuron.map(Math.fround), { atol: 0 }, `${arch} typical neuron`);
    assertClose(scales.typical.head, stats.scales.typical.head.map(Math.fround), { atol: 0 }, `${arch} typical head`);
    assert.equal(Math.fround(scales.typical.modelNeuron), Math.fround(stats.scales.typical.model_neuron));
    assert.equal(Math.fround(scales.typical.modelHead), Math.fround(stats.scales.typical.model_head));
  }
  console.log("units lens reads: R and S halves, mask and Σ_E match drowse");
} finally {
  await server.close();
}
