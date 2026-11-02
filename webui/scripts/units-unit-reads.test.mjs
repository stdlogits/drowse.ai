// The UnitReader port (unitRead, unitTop, unitFind, catalog, labels) on the reference backend
// against drowse's unit-reads golden, with unit-artifacts as the offline packs and the fixture
// whitener (browser-runtime/fixtures/units-tiny-whitener-v1.json) as the core pack's.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { ARCHES, archGolden, compareRecord, decode, viteServer } from "./support/units-goldens.mjs";
import { unitsFixture } from "./support/units-fixture.mjs";

const whiteners = JSON.parse(readFileSync(fileURLToPath(new URL("../../browser-runtime/fixtures/units-tiny-whitener-v1.json", import.meta.url)), "utf8"));
const only = process.env.UNIT_READS_ONLY ?? null;

const server = await viteServer();
try {
  const [readerModule, scalesModule] = await Promise.all([
    server.ssrLoadModule("/src/hosted/runtime/units/unitReader.ts"),
    server.ssrLoadModule("/src/hosted/runtime/units/scales.ts"),
  ]);
  for (const arch of ARCHES) {
    const fx = await unitsFixture(server, arch, { modelId: "units-tiny" });
    const reads = archGolden("unit-reads", arch);
    const artifacts = archGolden("unit-artifacts", arch);
    const stats = fx.stats;
    const statistics = {
      documents: stats.n_documents,
      neuronMean: Float32Array.from(decode(stats.neuron_mean)),
      neuronScale: Float32Array.from(decode(stats.neuron_scale)),
      headMean: Float32Array.from(decode(stats.head_mean)),
      headScale: Float32Array.from(decode(stats.head_scale)),
    };
    const tables = new Map(artifacts.layers.map((entry) => [
      entry.layer,
      new Map(Object.entries(entry.tensors).map(([key, value]) => [key, Float32Array.from(decode(value))])),
    ]));
    const reader = new readerModule.UnitReader({
      context: fx.context,
      spec: fx.model.spec,
      artifacts: {
        statistics,
        statisticsAbsent: null,
        scales: scalesModule.readingScales(statistics, fx.model.spec, 4),
        layer: async (layer) => tables.get(layer),
        nullProjector: async () => Float32Array.from(decode(artifacts.null.projector)),
        headKeptIds: async () => Int32Array.from(artifacts.head_kept_ids),
        whitener: async (layer) => {
          const entry = whiteners.arches[arch].find((row) => row.layer === layer);
          return entry ? {
            columns: 64, rank: entry.eigenvalues.length, ridge: entry.ridge,
            mean: Float64Array.from(entry.mean), basis: Float64Array.from(entry.basis.flat()),
            eigenvalues: Float64Array.from(entry.eigenvalues), inverseScales: new Float64Array(entry.eigenvalues.length),
          } : null;
        },
      },
      resolveToken: (word) => (word.startsWith("#") ? Number(word.slice(1)) : fx.tokenizer.encode(word)[0]),
      resolveSelector: async (selector) => {
        // jlens/#<id>: native `token_direction`, `W_U[v] @ J_L` per R:out layer (raw W_U).
        const id = Number(selector.slice("jlens/#".length));
        const out = new Map();
        for (const [layer, matrix] of fx.lenses.get("R:out")) {
          const row = fx.model.embed.subarray(id * 64, (id + 1) * 64);
          out.set(layer, Float32Array.from({ length: 64 }, (_, j) => row.reduce((s, x, i) => s + x * matrix[i * 64 + j], 0)));
        }
        return out;
      },
    });
    const tol = reads.tolerance.library;
    const numberTol = (path) => {
      if (/\.(value|z|footprint_z|score|coeff|share|mean_share|copy_score\.[RS])$/.test(path)) return tol.z_atol + 2e-6;
      if (/probability$/.test(path)) return tol.probability_atol + 2e-6;
      return 2e-6;
    };
    for (const [i, entry] of reads.reads.entries()) {
      if (only && only !== "reads") continue;
      const actual = await reader.read(entry.unit, entry.k, entry.hide_hubs);
      const problems = compareRecord(actual, entry.read, { root: `reads[${i}].read`, ties: reads.declared_ties, numberTol });
      assert.deepEqual(problems, [], `${arch} unitRead ${entry.unit} hide_hubs=${entry.hide_hubs}`);
    }
    for (const [i, entry] of reads.top.entries()) {
      if (only && only !== "top") continue;
      const actual = await reader.top(entry.token, entry.by, null, entry.k);
      const problems = compareRecord(actual, entry.result, { root: `top[${i}].result`, ties: reads.declared_ties, numberTol });
      assert.deepEqual(problems, [], `${arch} unitTop ${entry.token} by ${entry.by}`);
    }
    for (const [i, entry] of reads.find.entries()) {
      if (only && only !== "find") continue;
      const actual = entry.target === "vector"
        ? await reader.find({ vector: Float32Array.from(entry.vector), layer: entry.layer }, entry.k)
        : await reader.find({ selector: entry.target }, entry.k);
      const problems = compareRecord(actual, entry.result, { root: `find[${i}].result`, ties: reads.declared_ties, numberTol });
      assert.deepEqual(problems, [], `${arch} unitFind ${entry.target}`);
    }
    // ---------------------------------------------------------- catalog and labels ---
    assert.deepEqual(reader.catalogStatus(), {
      model_id: "units-tiny", available: false, bound: false, slug: null, repo: "stdlogits/neuron-lens",
      revision: "b5bf11a2b9e35c9100a0fa0995ff7a4e57f5b777", credit: null,
      reason: "no neuron-lens catalog for this model (catalogued: Qwen/Qwen3-8B, google/gemma-4-12B)", binding: null,
    });
    assert.deepEqual(reader.labels(["neuron/L1n5", "head/L2h0"]).labels, []);
    assert.throws(() => reader.labels(["neuron/L9n5"]), /layer 9 is outside the model \(4 blocks, 0\.\.3\)/);
    assert.throws(() => reader.labels(["neuron/L1h5"]), /'neuron\/L1h5' is not a neuron reference/);
    await assert.rejects(reader.read("head/L1h9"), /head 9 is outside block 1 \(4 heads, 0\.\.3\)/);
    await assert.rejects(reader.read("neuron/L1n5", 0), /k must be in \[1, 256\]/);
  }
  console.log("units unit reads: unitRead, unitTop and unitFind match drowse");
} finally {
  await server.close();
}
