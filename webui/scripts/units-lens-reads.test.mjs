// The four-lens read and the S readout on the reference backend against drowse's lens-reads
// and slens goldens (B1-goldens §2.6, §2.7).
import assert from "node:assert/strict";
import { ARCHES, archGolden, assertClose, compareRecord, decode, viteServer } from "./support/units-goldens.mjs";
import { unitsFixture } from "./support/units-fixture.mjs";

const server = await viteServer();
try {
  const reads = await server.ssrLoadModule("/src/hosted/runtime/units/lensReads.ts");
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
      const problems = compareRecord(actual, request.reads, {
        root: `requests[${i}].reads`,
        ties: golden.declared_ties,
        numberTol: (path, expected) => path.endsWith(".value") && /\.s\.tokens/.test(path)
          ? tol.z_atol + 2e-6
          : path.endsWith(".value") ? tol.probability_atol + 2e-6 : 2e-6,
      });
      assert.deepEqual(problems, [], `${arch} lens read ${i} (${request.of} L${request.layer} ${request.site})`);
    }
  }
  console.log("units lens reads: R and S halves, mask and Σ_E match drowse");
} finally {
  await server.close();
}
