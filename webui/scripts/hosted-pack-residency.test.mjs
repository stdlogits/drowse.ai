// Full-depth lens and units packs (PLAN.md): which files install with a pack, which are fetched on
// first read, how a fetched layer joins its install, and how the registry reads a layer it does
// not hold yet.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({
  root,
  configFile: false,
  optimizeDeps: { noDiscovery: true, include: [] },
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});

const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");

function safetensors(tensors) {
  const header = {};
  let offset = 0;
  const parts = [];
  for (const [name, { shape, data }] of Object.entries(tensors)) {
    const bytes = new Uint8Array(new Float32Array(data).buffer);
    header[name] = { dtype: "F32", shape, data_offsets: [offset, offset + bytes.length] };
    offset += bytes.length;
    parts.push(bytes);
  }
  let json = JSON.stringify(header);
  while ((json.length + 8) % 8 !== 0) json += " ";
  const prefix = new Uint8Array(8);
  new DataView(prefix.buffer).setUint32(0, json.length, true);
  return new Uint8Array(Buffer.concat([prefix, Buffer.from(json), ...parts.map((part) => Buffer.from(part))]));
}

try {
  const {
    lensReadoutLayers,
    packFileResidency,
    packInstallCovers,
    packInstallFiles,
    packLayerShard,
  } = await server.ssrLoadModule("/src/hosted/runtime/optionalPackCompatibility.ts");
  const { BrowserInstrumentRegistry } = await server.ssrLoadModule(
    "/src/hosted/runtime/browserInstrumentPacks.ts",
  );

  // The live readout's layers: today's rule, evenly spaced over the blocks before the last.
  assert.deepEqual(lensReadoutLayers([...Array(18).keys()], 640), [0, 2, 5, 7, 9, 11, 14, 16]);
  assert.deepEqual(lensReadoutLayers([...Array(5).keys()], 64), [0, 1, 2, 3]);
  assert.deepEqual(lensReadoutLayers([...Array(36).keys()], 2560), [0, 5, 10, 15, 19, 24, 29, 34]);
  assert.throws(() => lensReadoutLayers([0], 64), /block before the last/u);

  const identity = { runtimeIdentity: { layerMap: [0, 1, 2, 3, 4], hiddenSize: 4 } };
  const jlens = { kind: "jlens" };
  assert.deepEqual(packLayerShard(jlens, "packs/jlens/mid.layer-3.safetensors"), { kind: "lens", site: "mid", layer: 3 });
  assert.equal(packLayerShard(jlens, "packs/jlens/layers.safetensors"), null);
  assert.equal(packLayerShard(jlens, "packs/slens/out.layer-3.safetensors"), null, "a shard sits in its kind's directory");
  assert.equal(packLayerShard(jlens, "packs/jlens/out.layer-03.safetensors"), null);
  assert.deepEqual(
    packLayerShard({ kind: "units" }, "packs/units/contrast.layer-12.safetensors"),
    { kind: "units", part: "contrast", layer: 12 },
  );
  assert.equal(packLayerShard({ kind: "sae" }, "packs/sae/out.layer-1.safetensors"), null);
  // R:out at the readout layers installs; every other layer and every S and units layer waits.
  assert.equal(packFileResidency(jlens, identity, "packs/jlens/out.layer-0.safetensors"), "install");
  assert.equal(packFileResidency(jlens, identity, "packs/jlens/mid.layer-0.safetensors"), "on_demand");
  assert.equal(packFileResidency({ kind: "slens" }, identity, "packs/slens/out.layer-0.safetensors"), "on_demand");
  assert.equal(packFileResidency({ kind: "slens" }, identity, "packs/slens/embedding-statistics.safetensors"), "install");
  assert.equal(packFileResidency({ kind: "units" }, identity, "packs/units/reads.layer-0.safetensors"), "on_demand");

  // A tiny full-depth R pack: d = 4, five blocks; out L0..3, mid L0..4.
  const d = 4;
  const layerMap = [0, 1, 2, 3, 4];
  const files = new Map();
  const site = (name, layers) => {
    const entry = { layers, tensor_files: {}, tensor_sha256: {} };
    for (const layer of layers) {
      const bytes = safetensors({
        [`layer_${layer}`]: { shape: [d, d], data: Array.from({ length: d * d }, (_, i) => layer * 100 + i + (name === "mid" ? 50 : 0)) },
      });
      const path = `packs/jlens/${name}.layer-${layer}.safetensors`;
      files.set(path, bytes);
      entry.tensor_files[String(layer)] = `${name}.layer-${layer}.safetensors`;
      entry.tensor_sha256[String(layer)] = sha(bytes);
    }
    return entry;
  };
  const variant = {
    id: "fixture-q0f32",
    runtimeIdentitySha256: "e".repeat(64),
    runtimeIdentity: { layerMap, hiddenSize: d, sourceRevision: "f".repeat(40), runtimeAbi: "drowse-web-runtime-v2" },
  };
  const manifest = {
    format_version: 7, kind: "R", sites: { out: site("out", [0, 1, 2, 3]), mid: site("mid", layerMap) },
    estimator_policy: {}, n_prompts: 64, d_model: d, dtype: "float32", corpus_spec: "fixture",
    corpus_sha256: "0".repeat(64), seq_len: 128, dim_batch: 256, corpus_hash_kind: "token_ids_v1",
    raw_corpus_sha256: null, raw_prompt_count: null, usable_prompt_count: 64, model_layer_count: 5,
    model_fingerprint: variant.runtimeIdentitySha256, model_source_fingerprint: variant.runtimeIdentity.sourceRevision,
    checkpoint: false, consumed_prefix_sha256: null,
  };
  files.set("packs/jlens/manifest.json", new TextEncoder().encode(JSON.stringify(manifest)));
  const vocabulary = safetensors({ unembedding: { shape: [1, d], data: [1, 0, 0, 0] } });
  files.set("packs/jlens/browser-vocabulary.safetensors", vocabulary);
  files.set("packs/jlens/browser-vocabulary.json", new TextEncoder().encode(JSON.stringify({
    format: "drowse-jlens-vocabulary-v2", hidden_size: d, tensor_file: "browser-vocabulary.safetensors",
    tensor_sha256: sha(vocabulary), words: [{ word: "yes", row: 0, token_id: 7 }],
  })));
  const catalogFiles = [...files].map(([path, bytes]) => ({
    path, role: "instrument", url: `https://huggingface.co/x/y/resolve/${"a".repeat(40)}/${path}`,
    revision: "a".repeat(40), bytes: bytes.byteLength, sha256: sha(bytes),
  }));
  const pack = {
    id: "fixture-jlens", kind: "jlens", displayName: "fixture R lens",
    runtimeIdentitySha256: variant.runtimeIdentitySha256, files: catalogFiles,
  };
  const installs = packInstallFiles(pack, variant).map((file) => file.path).sort();
  assert.deepEqual(installs, [
    "packs/jlens/browser-vocabulary.json", "packs/jlens/browser-vocabulary.safetensors",
    "packs/jlens/manifest.json", "packs/jlens/out.layer-0.safetensors", "packs/jlens/out.layer-1.safetensors",
    "packs/jlens/out.layer-2.safetensors", "packs/jlens/out.layer-3.safetensors",
  ]);
  const installHashes = packInstallFiles(pack, variant).map((file) => file.sha256);
  assert.equal(packInstallCovers(pack, variant, installHashes), true);
  assert.equal(packInstallCovers(pack, variant, installHashes.slice(1)), false, "an install holds every install file");
  assert.equal(packInstallCovers(pack, variant, [...installHashes, catalogFiles.at(-1).sha256]), true);
  assert.equal(packInstallCovers(pack, variant, [...installHashes, "9".repeat(64)]), false);

  const artifact = (file) => ({ manifest: file, file: new File([files.get(file.path)], file.path) });
  const fetched = [];
  const load = (fetchFile) => BrowserInstrumentRegistry.load({
    model: { id: "fixture" },
    variant,
    requiredCorePack: { files: [] },
    contextTokens: 2048,
    adapter: {},
    artifacts: [],
    optionalPacks: [{
      pack,
      artifacts: packInstallFiles(pack, variant).map(artifact),
      ...(fetchFile === undefined ? {} : { fetchFile }),
    }],
    activationSpool: {},
    signal: new AbortController().signal,
    onDeviceLost() {},
  });
  const registry = await load(async (file, signal) => {
    signal.throwIfAborted();
    fetched.push(file.path);
    return new File([files.get(file.path)], file.path);
  });
  const signal = new AbortController().signal;
  assert.deepEqual(registry.descriptor().jlens.layers, [0, 1, 2, 3], "the readout's layers stay resident");
  assert.deepEqual(registry.lensSiteLayers("R", "mid"), layerMap);
  const out2 = await registry.lensMatrix("R", "out", 2, signal);
  assert.equal(out2[0], 200);
  assert.deepEqual(fetched, [], "a readout layer is never fetched");
  const [mid4, again] = await Promise.all([
    registry.lensMatrix("R", "mid", 4, signal),
    registry.lensMatrix("R", "mid", 4, signal),
  ]);
  assert.equal(mid4[1], 451);
  assert.equal(again, mid4);
  assert.deepEqual(fetched, ["packs/jlens/mid.layer-4.safetensors"], "one fetch per layer, then the cache");
  assert.equal(await registry.lensMatrix("R", "out", 4, signal), null, "R:out holds no last block: an absent half");
  await assert.rejects(registry.lensMatrix("S", "out", 0, signal), { code: "SLENS_PACK_UNAVAILABLE" });
  const [source] = registry.jlensSources();
  assert.deepEqual(Object.keys(source.summary.coverage.sites), ["out", "mid"]);
  assert.deepEqual(source.summary.coverage.sites.mid.layers, layerMap);

  const offline = await load(undefined);
  await assert.rejects(offline.lensMatrix("R", "mid", 0, signal), { code: "INSTRUMENT_PACK_LAYER_UNAVAILABLE" });

  // A fetched file that is not the catalog's refuses.
  const lying = await load(async (file) => new File([new Uint8Array(file.bytes + 1)], file.path));
  await assert.rejects(lying.lensMatrix("R", "mid", 1, signal), { code: "INSTRUMENT_PACK_ARTIFACT_MISMATCH" });
  console.log("ok - hosted pack residency: install set, on-demand layers, fetch once, refusals");

  // The core pack's neutral statistics: stacked over layers, deeper tensors stored rank 2.
  const { isNeutralStatisticsPath, readNeutralStatistics } = await server.ssrLoadModule(
    "/src/hosted/runtime/browserInstrumentPacks.ts",
  );
  assert.equal(isNeutralStatisticsPath("neutral-statistics.json"), true);
  assert.equal(isNeutralStatisticsPath("manifolds/2048/default-deceptive.drowse"), false);
  const L = 5;
  const shapes = {
    head_mean: [L, 2, 3], head_scale: [L, 2], mid_mean: [L, d], neuron_mean: [L, 6],
    neuron_scale: [L, 2, 6], out_mean: [L, d],
  };
  const statisticsBytes = safetensors(Object.fromEntries(Object.entries(shapes).map(([name, shape]) => {
    const stored = shape.length > 2 ? [shape[0] * shape[1], shape[2]] : shape;
    return [name, { shape: stored, data: Array.from({ length: stored[0] * stored[1] }, (_, i) => i) }];
  })));
  const statisticsRecord = {
    format: "drowse-neutral-statistics-v1", runtime_identity_sha256: variant.runtimeIdentitySha256,
    model_layer_count: L, n_documents: 96, neutral: { sha256: "c".repeat(64) }, shapes,
    tensor_file: "neutral-statistics.safetensors", tensor_sha256: sha(statisticsBytes),
  };
  const coreFile = (path, bytes) => ({
    manifest: { path, role: "core_pack", bytes: bytes.byteLength, sha256: sha(bytes) },
    file: new File([bytes], path),
  });
  const statistics = await readNeutralStatistics([
    coreFile("neutral-statistics.json", new TextEncoder().encode(JSON.stringify(statisticsRecord))),
    coreFile("neutral-statistics.safetensors", statisticsBytes),
  ], { variant, signal });
  assert.equal(statistics.nDocuments, 96);
  assert.deepEqual(statistics.tensors.get("neuron_scale").shape, [L, 2, 6]);
  assert.equal(statistics.tensors.get("neuron_scale").data.length, L * 12);
  assert.equal(await readNeutralStatistics([], { variant, signal }), null, "a pack without units has none");
  await assert.rejects(
    readNeutralStatistics([
      coreFile("neutral-statistics.json", new TextEncoder().encode(JSON.stringify({
        ...statisticsRecord, runtime_identity_sha256: "0".repeat(64),
      }))),
      coreFile("neutral-statistics.safetensors", statisticsBytes),
    ], { variant, signal }),
    { code: "CORE_NEUTRAL_STATISTICS_INVALID" },
  );
  console.log("ok - core neutral statistics read back at their logical shapes and refuse another runtime");
} finally {
  await server.close();
}
