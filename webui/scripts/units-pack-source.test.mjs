// The units members' packs on a loaded model (units/packSource.ts) over B6's registry, on drowse's
// tiny fixture: build_hosted_packs.py builds the R, S, units and core-statistics packs from the
// fixture session (scripts/support/units-tiny-packs.py), the core whitener is fitted from its
// neutral rows as the core builder fits it, and the browser registry reads them back. The pack
// source must serve the unit-artifacts golden's tensors, native's S mask digest and lens
// identities, B5's fixture whitener and the unit-stats golden; and the units service over it
// (the CPU block-sites backend on the tiny model) answers every unit-reads read and top as drowse.
import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { promisify } from "node:util";
import { collectPackArtifacts } from "./assemble-hosted-pack.mjs";
import { fitNativeNeutralWhitener, neutralWhitenerSidecar } from "./neutral-whitener-node.mjs";
import { fixtureWhiteners } from "./support/units-compiler.mjs";
import { unitsFixture } from "./support/units-fixture.mjs";
import { ARCHES, archGolden, closeness, compareRecord, decode, viteServer, webuiRoot } from "./support/units-goldens.mjs";

const exec = promisify(execFile);
const python = process.env.PYTHON ?? "python3";
const revision = "a".repeat(40);
const CONTEXT = 64;

const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");
const artifact = (path, bytes, role) => ({
  manifest: {
    path, role, revision, bytes: bytes.byteLength, sha256: sha(bytes),
    url: `https://huggingface.co/drowse/local/resolve/${revision}/${path}`,
  },
  file: new File([bytes], path),
});

async function packEntry(directory, kind, fingerprint) {
  const { files } = await collectPackArtifacts(directory);
  const catalogFiles = files.map((file) => ({
    ...file, role: "instrument", revision, url: `https://huggingface.co/drowse/local/resolve/${revision}/${file.path}`,
  }));
  return {
    pack: { id: `units-tiny-${kind}`, kind, displayName: `units-tiny ${kind}`, runtimeIdentitySha256: fingerprint, files: catalogFiles },
    artifacts: await Promise.all(catalogFiles.map(async (entry) => ({
      manifest: entry, file: new File([await readFile(resolve(directory, entry.path))], entry.path),
    }))),
  };
}

/** The core pack's neutral statistics and one context's whitener, as the core builder writes them
 *  from `build_hosted_packs.py`'s staged `core/` (the whitener fitted in Node from its rows). */
async function coreArtifacts(server, directory, identity, fingerprint, binding) {
  const record = JSON.parse(await readFile(resolve(directory, "neutral-statistics.json"), "utf8"));
  const rows = await readFile(resolve(directory, record.neutral_rows?.file ?? "neutral-rows.safetensors"));
  assert.equal(sha(rows), record.neutral.rows_sha256, "the staged rows are the ones the statistics name");
  const modelLock = { layerMap: identity.layerMap, hiddenSize: identity.hiddenSize };
  const fitted = await fitNativeNeutralWhitener(server, rows, modelLock);
  const sidecar = neutralWhitenerSidecar({
    runtimeIdentitySha256: fingerprint, contextBindingSha256: binding, modelLock,
    tensorsSha256: sha(fitted.tensorBytes), ridgePerLayer: fitted.ridgePerLayer,
  });
  const statistics = new TextEncoder().encode(JSON.stringify({
    ...record,
    whitener: {
      rows_sha256: record.neutral.rows_sha256,
      tensors_sha256: { [CONTEXT]: sha(fitted.tensorBytes) },
      sidecars_sha256: { [CONTEXT]: sha(sidecar) },
    },
  }));
  return [
    artifact("core/neutral-statistics.json", statistics, "core_pack"),
    artifact(`core/${record.tensor_file}`, await readFile(resolve(directory, record.tensor_file)), "core_pack"),
    artifact(`core/${CONTEXT}/neutral-whitener.safetensors`, fitted.tensorBytes, "core_pack"),
    artifact(`core/${CONTEXT}/neutral-whitener.json`, sidecar, "core_pack"),
  ];
}

const server = await viteServer();
const scratch = await mkdtemp(join(tmpdir(), "units-pack-source-"));
try {
  const [packs, registryModule, core, catalog, reader, cpu, profiles, service] = await Promise.all([
    server.ssrLoadModule("/src/hosted/runtime/units/packSource.ts"),
    server.ssrLoadModule("/src/hosted/runtime/browserInstrumentPacks.ts"),
    server.ssrLoadModule("/src/hosted/runtime/browserCorePack.ts"),
    server.ssrLoadModule("/src/lib/runtime/catalog.ts"),
    server.ssrLoadModule("/src/hosted/runtime/units/unitReader.ts"),
    server.ssrLoadModule("/src/hosted/runtime/units/cpuBackend.ts"),
    server.ssrLoadModule("/src/hosted/runtime/structuredHookProfile.ts"),
    server.ssrLoadModule("/src/hosted/runtime/units/unitsService.ts"),
  ]);
  for (const arch of ARCHES) {
    const out = join(scratch, arch);
    const { stdout } = await exec(python, [resolve(webuiRoot, "scripts/support/units-tiny-packs.py"), arch, out], {
      maxBuffer: 1 << 24,
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    });
    const native = JSON.parse(stdout);
    const identity = native.runtime.runtimeIdentity;
    const fingerprint = native.runtime.runtimeIdentitySha256;
    const binding = catalog.contextBindingSha256(fingerprint, CONTEXT);
    const signal = new AbortController().signal;
    const request = {
      model: { id: "units-tiny" },
      variant: {
        id: "units-tiny-q0f32", runtimeIdentity: identity, runtimeIdentitySha256: fingerprint,
        contextProfiles: [{ contextTokens: CONTEXT, bindingSha256: binding }],
      },
      requiredCorePack: { files: [] },
      contextTokens: CONTEXT,
      adapter: {},
      artifacts: await coreArtifacts(server, join(out, "core"), identity, fingerprint, binding),
      optionalPacks: await Promise.all(["jlens", "slens", "units"].map((kind) => packEntry(join(out, kind), kind, fingerprint))),
      activationSpool: {},
      signal,
      onDeviceLost() {},
    };
    const registry = await registryModule.BrowserInstrumentRegistry.load(request);
    const coreFiles = request.artifacts;
    const whiteners = await core.loadCoreWhiteners(coreFiles[2], coreFiles[3], request);
    // The compiler surface the source reads (BrowserFeasibilityCorePackCompiler's).
    let neutralReads = 0;
    const compiler = {
      instrumentRegistry: () => registry,
      neutralStatistics: (abort) => {
        neutralReads += 1;
        return registryModule.readNeutralStatistics(coreFiles, { variant: request.variant, signal: abort });
      },
      neutralWhitener: (layer) => whiteners.get(layer) ?? null,
    };
    const source = packs.compilerUnitsPackSource(() => compiler, "units-tiny", signal);

    // ------------------------------------------------------------- lens identities ---
    for (const [family, kind] of [["R", "jlens"], ["S", "slens"]]) {
      const lens = source.lens(family);
      assert.equal(lens.identity, native.record.packs[kind].lens_identity, `${arch} ${family} lens identity`);
      assert.equal(lens.source, `units-tiny-${kind}`);
      for (const site of lens.sites) {
        for (const layer of lens.layers(site)) {
          assert.equal((await lens.fetch(site, layer)).length, identity.hiddenSize ** 2);
        }
      }
    }

    // --------------------------------------------- S statistics and the mask digest ---
    const fx = await unitsFixture(server, arch, { modelId: "units-tiny" });
    const slens = archGolden("slens", arch);
    const embedding = await source.embeddingStatistics();
    assert.equal(embedding.maskDigest, native.maskDigest, `${arch} native _mask_digest`);
    assert.ok(closeness(embedding.meanE, decode(slens.mean_E), 1e-6, 1e-6).worst <= 0, `${arch} mean_E`);
    assert.ok(closeness(embedding.sigmaE, decode(slens.sigma_E), 1e-6, 1e-6).worst <= 0, `${arch} Σ_E`);

    // ------------------------------------------------- unit statistics and means ---
    const { statistics, absent } = await source.statistics();
    assert.equal(absent, null);
    assert.equal(statistics.documents, fx.stats.n_documents);
    for (const [key, name] of [["neuronMean", "neuron_mean"], ["neuronScale", "neuron_scale"], ["headMean", "head_mean"], ["headScale", "head_scale"]]) {
      assert.ok(closeness(statistics[key], decode(fx.stats[name]), 1e-6, 1e-5).worst <= 0, `${arch} ${name}`);
    }
    const means = await source.means();
    assert.equal(means.out.length, identity.layerMap.length * identity.hiddenSize);
    assert.ok(closeness(means.out, fx.means.out, 1e-5, 1e-4).worst <= 0, `${arch} neutral out means`);

    // --------------------------------- units tensors: the unit-artifacts golden ---
    const artifacts = archGolden("unit-artifacts", arch);
    const tol = artifacts.tolerance.python;
    for (const entry of artifacts.layers) {
      const reads = await source.layerArtifacts(entry.layer, "reads");
      const contrast = await source.layerArtifacts(entry.layer, "contrast");
      assert.deepEqual([...reads.keys(), ...contrast.keys()].sort(), Object.keys(entry.tensors).sort(), `${arch} L${entry.layer} keys`);
      for (const [key, value] of Object.entries(entry.tensors)) {
        const actual = reads.get(key) ?? contrast.get(key);
        const { worst } = closeness(actual, decode(value), tol.atol, tol.rtol);
        assert.ok(worst <= 0, `${arch} L${entry.layer} ${key}`);
      }
    }
    assert.deepEqual([...await source.headKeptIds()], artifacts.head_kept_ids, `${arch} head_kept_ids`);
    const projector = Float32Array.from(decode(artifacts.null.projector));
    const d = identity.hiddenSize;
    for (let trial = 0; trial < 4; trial += 1) {
      const w = Float32Array.from({ length: d }, (_, i) => Math.sin(17 * trial + 3 * i + 1));
      let quad = 0;
      for (let i = 0; i < d; i += 1) for (let j = 0; j < d; j += 1) quad += w[i] * projector[i * d + j] * w[j];
      assert.ok(Math.abs(await source.nullProjection(w) - quad) <= 1e-4 * Math.max(1, quad), `${arch} null projection`);
    }

    // ------------------------------------------- the whitener: B5's fixture one ---
    for (const row of fixtureWhiteners.arches[arch]) {
      const ours = await source.whitener(row.layer);
      const theirs = {
        columns: d, rank: row.eigenvalues.length, ridge: row.ridge, mean: Float64Array.from(row.mean),
        basis: Float64Array.from(row.basis.flat()), eigenvalues: Float64Array.from(row.eigenvalues),
        inverseScales: new Float64Array(row.eigenvalues.length),
      };
      const x = Float32Array.from({ length: 3 * d }, (_, i) => Math.cos(0.37 * i + row.layer));
      const { worst } = closeness(reader.whitenRows(ours, x, 3), reader.whitenRows(theirs, x, 3), 1e-4, 1e-4);
      assert.ok(worst <= 0, `${arch} L${row.layer} whitener acts as the fixture's`);
    }

    // ---------------- the units service over the source: unit-reads as drowse ---
    const backend = cpu.cpuBlockSitesBackend(fx.model, { profile: profiles.STANDARD_STRUCTURED_HOOK_PROFILE_V4, contextTokens: CONTEXT });
    const units = new service.BrowserUnitsService({
      modelId: "units-tiny",
      blockSites: () => backend,
      // The fixture session reads with its own S→R band (a hosted session keeps native's default).
      packs: () => ({ ...source, band: { lo: fx.file.band[0], hi: fx.file.band[1] } }),
      tokenTexts: async (ids) => ids.map((id) => fx.tokenizer.decode([id])),
      resolveToken: packs.sTokenResolver({
        tokenizeText: async (text) => fx.tokenizer.encode(text),
        tokenTexts: async (ids) => ids.map((id) => fx.tokenizer.decode([id])),
      }, () => fx.model.vocabSize),
      replay: () => null,
    });
    const golden = archGolden("unit-reads", arch);
    const libraryTol = golden.tolerance.library;
    const numberTol = (path) => {
      if (/\.(value|z|footprint_z|score|coeff|share|mean_share|copy_score\.[RS])$/.test(path)) return libraryTol.z_atol + 2e-6;
      if (/probability$/.test(path)) return libraryTol.probability_atol + 2e-6;
      return 2e-6;
    };
    for (const [i, entry] of golden.reads.entries()) {
      const actual = await units.request("unitRead", [entry.unit, { k: entry.k, hideHubs: entry.hide_hubs }]);
      assert.deepEqual(compareRecord(actual, entry.read, { root: `reads[${i}].read`, ties: golden.declared_ties, numberTol }), [], `${arch} unitRead ${entry.unit}`);
    }
    for (const [i, entry] of golden.top.entries()) {
      const actual = await units.request("unitTop", [entry.token, { by: entry.by, k: entry.k }]);
      assert.deepEqual(compareRecord(actual, entry.result, { root: `top[${i}].result`, ties: golden.declared_ties, numberTol }), [], `${arch} unitTop ${entry.token}`);
    }
    for (const [i, entry] of golden.find.entries()) {
      if (entry.target !== "vector") continue;
      const actual = await units.request("unitFind", [{ vector: entry.vector, layer: entry.layer, k: entry.k }]);
      assert.deepEqual(compareRecord(actual, entry.result, { root: `find[${i}].result`, ties: golden.declared_ties, numberTol }), [], `${arch} unitFind vector`);
    }
    assert.ok(neutralReads >= 1);
  }
  console.log("units pack source: the registry's packs serve drowse's artifacts, mask digest, whitener and unit reads");
} finally {
  await server.close();
  await rm(scratch, { recursive: true, force: true });
}
