import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { cp, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { promisify } from "node:util";
import { createServer } from "vite";
import { readCoreManifoldSource } from "./hosted-core-manifold-source.mjs";
import { lockedModelExecutionProfiles } from "./hosted-model-profiles.mjs";

const exec = promisify(execFile);
const webuiRoot = resolve(import.meta.dirname, "..");
const repositoryRoot = resolve(webuiRoot, "..");
for (const script of ["build-hosted-whitener.mjs", "validate-hosted-core-pack.mjs"]) {
  await exec(process.execPath, ["--check", resolve(import.meta.dirname, script)]);
}
const runtimeLock = JSON.parse(await readFile(
  resolve(repositoryRoot, "browser-runtime/runtime-lock.json"),
  "utf8",
));
const qwen4b = runtimeLock.models.find((model) => model.id === "qwen3-4b");
assert.equal(qwen4b.quantization, "q4f32_1");
const executionProfiles = lockedModelExecutionProfiles(qwen4b, {
  structuredHookProfile: qwen4b.structuredHookProfile,
  thinkingProfile: qwen4b.thinkingProfile,
  quantization: qwen4b.quantization,
});
assert.deepEqual(executionProfiles, {
  structuredHookProfile: "standard-v5",
  thinkingProfile: qwen4b.thinkingProfile,
  quantization: "q4f32_1",
  requiredFeatures: [],
  blockSpec: qwen4b.blockSpec,
  unitRefusal: null,
  prefillChunkSize: 2048,
});
assert.ok(qwen4b.blockSpec, "the lock records qwen3-4b's block spec");
// The lock's prefill chunk travels with the profiles; a closure that says otherwise is refused.
assert.throws(
  () => lockedModelExecutionProfiles({ ...qwen4b, prefillChunkSize: undefined }, executionProfiles),
  /prefill chunk does not match/,
);
assert.throws(
  () => lockedModelExecutionProfiles(qwen4b, { ...executionProfiles, prefillChunkSize: 1024 }),
  /prefill chunk does not match/,
);
// A v5 library: the block spec travels with standard-v5, the refusal with -residual.
const blockSpec = { dFF: 9728, nHeads: 32 };
const { blockSpec: _lockedSpec, unitRefusal: _lockedRefusal, ...bareQwen4b } = qwen4b;
const v5 = (profile, extra) => lockedModelExecutionProfiles(
  { ...bareQwen4b, structuredHookProfile: profile, ...extra },
  { structuredHookProfile: profile, thinkingProfile: qwen4b.thinkingProfile, quantization: qwen4b.quantization },
);
assert.deepEqual(v5("standard-v5", { blockSpec }).blockSpec, blockSpec);
assert.equal(v5("standard-v5-residual", { unitRefusal: "fused qkv" }).unitRefusal, "fused qkv");
for (const [profile, extra] of [
  ["standard-v5", {}],
  ["standard-v5", { blockSpec, unitRefusal: "fused qkv" }],
  ["standard-v5-residual", {}],
  ["standard-v5-residual", { blockSpec, unitRefusal: "fused qkv" }],
]) {
  assert.throws(() => v5(profile, extra), /blockSpec and unitRefusal/, `${profile} ${JSON.stringify(extra)}`);
}
// Hook ABI v5 only: no other profile is a hosted profile.
assert.throws(() => v5("unknown-profile", { blockSpec }), /structuredHookProfile is not allow-listed/);
assert.throws(
  () => lockedModelExecutionProfiles(qwen4b, {
    structuredHookProfile: "standard-v5-residual",
    thinkingProfile: qwen4b.thinkingProfile,
    quantization: qwen4b.quantization,
  }),
  /runtime-lock execution profile/,
);
const seedSource = resolve(
  repositoryRoot,
  "browser-runtime/release-inputs/core-manifolds/welcoming.detached",
);
const seed = await readCoreManifoldSource(seedSource, repositoryRoot);
assert.equal(seed.name, "welcoming.detached");
assert.equal(seed.fitMode, "pca");
assert.deepEqual(seed.tags, ["starter", "tone"]);
assert.deepEqual(seed.source, { uri: "local", repository: null, revision: null });
assert.deepEqual(seed.nodes.map((node) => node.label), ["welcoming", "detached"]);
assert.deepEqual(seed.nodes.map((node) => node.statements.length), [48, 48]);

const invalidRoot = await mkdtemp(join(tmpdir(), "drowse-core-manifold-source-"));
const invalidSource = join(invalidRoot, "welcoming.detached");
try {
  await cp(seedSource, invalidSource, { recursive: true });
  await writeFile(join(invalidSource, "fitted.safetensors"), "not allowed");
  await assert.rejects(
    readCoreManifoldSource(invalidSource, repositoryRoot),
    /unexpected file closure/,
  );
  await rm(join(invalidSource, "fitted.safetensors"));
  const provenance = JSON.parse(await readFile(join(invalidSource, "provenance.json"), "utf8"));
  provenance.baseline_prompts_sha256 = "0".repeat(64);
  await writeFile(join(invalidSource, "provenance.json"), `${JSON.stringify(provenance, null, 2)}\n`);
  await assert.rejects(
    readCoreManifoldSource(invalidSource, repositoryRoot),
    /source provenance is invalid/,
  );
} finally {
  await rm(invalidRoot, { recursive: true, force: true });
}

const python = process.env.PYTHON ?? "python3";
const loaded = await exec(python, [
  "-c",
  [
    "import json, sys",
    "from pathlib import Path",
    "from drowse.store.manifold_folder import ManifoldFolder",
    "folder = ManifoldFolder.load(Path(sys.argv[1]))",
    "print(json.dumps({'name': folder.name, 'fit_mode': folder.fit_mode, 'nodes': folder.node_labels, 'groups': [len(rows) for _, rows in folder.node_groups()]}))",
  ].join("; "),
  seedSource,
], { cwd: repositoryRoot, env: { ...process.env, PYTHONPATH: resolve(repositoryRoot, "drowse") } });
assert.deepEqual(JSON.parse(loaded.stdout), {
  name: "welcoming.detached",
  fit_mode: "pca",
  nodes: ["welcoming", "detached"],
  groups: [48, 48],
});
assert.throws(
  () => lockedModelExecutionProfiles(qwen4b, {
    structuredHookProfile: qwen4b.structuredHookProfile,
    thinkingProfile: {
      start: "<think>",
      end: "</think>",
      startsInThinking: false,
      startTokenIds: [1],
      endTokenIds: [2],
    },
    quantization: qwen4b.quantization,
  }),
  /runtime-lock execution profile/,
);

const help = await exec(process.execPath, [
  resolve(import.meta.dirname, "build-hosted-whitener.mjs"),
  "--help",
]);
assert.match(help.stdout, /MODEL_ID MODEL_DIR MODEL\.wasm WEBLLM\.js OUTPUT_DIR/);
assert.match(help.stdout, /--manifold-source DIR/);
assert.match(help.stdout, /--runtime-lock PATH/);
assert.match(help.stdout, /--manual-browser/);
await assert.rejects(
  exec(process.execPath, [
    resolve(import.meta.dirname, "build-hosted-whitener.mjs"),
    "model", "directory", "model.wasm", "webllm.js", "output",
  ]),
  /--manifold-source is required/,
);

const vite = await createServer({
  root: webuiRoot,
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  publicDir: resolve(webuiRoot, "public-hosted"),
  server: { middlewareMode: true, watch: null },
});
try {
  const transformed = await vite.transformRequest("/scripts/hosted-whitener.worker.ts");
  assert.ok(transformed?.code.includes("module_or_path"));
  assert.ok(transformed.code.includes("BrowserFittingCoordinator"));
  assert.ok(transformed.code.includes("structuredHookProfile: event.data.structuredHookProfile"));
  assert.ok(transformed.code.includes("thinkingProfile: event.data.thinkingProfile"));
  assert.ok(transformed.code.includes("modelType: event.data.modelType"));
  assert.ok(transformed.code.includes("maxComputeWorkgroupSizeX"));
  assert.ok(transformed.code.includes("maxComputeInvocationsPerWorkgroup"));
  assert.ok(transformed.code.includes("contextBindingSha256"));
  assert.ok(transformed.code.includes("for (const context of contexts)"));
  assert.ok(transformed.code.includes("contextTokens: context.contextTokens"));
  assert.ok(transformed.code.includes("fittingWorker.dispose()"));
  assert.ok(transformed.code.includes("consensusGram: foundation.consensusGram"));
  assert.ok(transformed.code.includes("evaluatedLayers: foundation.layers"));
  // One-node concepts fold monopolar; the whitener comes from the native neutral rows when given.
  assert.ok(transformed.code.includes("manifold.nodes.length === 1"));
  assert.ok(transformed.code.includes("monopolarFoundation(centroids, whiteners)"));
  assert.ok(transformed.code.includes('fetch("/neutral-rows"'));
  assert.ok(transformed.code.includes("fitNeutralWhitener(fitting, native.matrices, native.rows, columns, layerMap)"));

  // The monopolar fold against drowse's own (`_fit_monopolar`, `fold_directions_to_subspace`):
  // the kernel's whitener over neutral rows, a concept centroid, and drowse's fp32 whitener over
  // the same rows give the same ray and share.
  const [{ monopolarFoundation }, { fitNeutralWhitener }, kernel] = await Promise.all([
    vite.ssrLoadModule("/src/hosted/fitting/monopolar.ts"),
    vite.ssrLoadModule("/src/hosted/fitting/neutralWhitener.ts"),
    import("../public-hosted/wasm/drowse_fitting_wasm.js"),
  ]);
  await kernel.default({
    module_or_path: await readFile(resolve(webuiRoot, "public-hosted/wasm/drowse_fitting_wasm_bg.wasm")),
  });
  const N = 12;
  const D = 6;
  const layerMap = [0, 1];
  let state = 7;
  const noise = () => ((state = (state * 1103515245 + 12345) % 2147483648) / 2147483648) - 0.5;
  const neutralRows = layerMap.map((layer) => Array.from({ length: N * D }, (_, i) => noise() * (1 + (i % D)) + layer));
  const concept = layerMap.map((layer) => Array.from({ length: D }, (_, i) => (i === layer ? 3 : 0.5) + noise()));
  const fitted = fitNeutralWhitener(kernel, neutralRows.map((rows) => Float64Array.from(rows)), N, D, layerMap);
  const foundation = monopolarFoundation({
    identity: {},
    captureRetained: false,
    layers: new Map(layerMap.map((layer, slot) =>
      [layer, { rows: 1, columns: D, values: Float64Array.from(concept[slot]) }])),
  }, fitted.whiteners);
  const scratch = await mkdtemp(join(tmpdir(), "drowse-monopolar-"));
  try {
    await writeFile(join(scratch, "inputs.json"), JSON.stringify({ neutralRows, concept, N, D }));
    const reference = JSON.parse((await exec(python, ["-c", [
      "import json, sys, torch",
      "from drowse.geometry.whitener import LayerWhitener",
      "from drowse.geometry.fold import fold_directions_to_subspace",
      "x = json.load(open(sys.argv[1]))",
      "rows = {L: torch.tensor(r, dtype=torch.float32).reshape(x['N'], x['D']) for L, r in enumerate(x['neutralRows'])}",
      "means = {L: r.mean(0) for L, r in rows.items()}",
      "w = LayerWhitener.from_neutral_activations(rows, means)",
      "d = {L: torch.tensor(c, dtype=torch.float32) - means[L] for L, c in enumerate(x['concept'])}",
      "m = fold_directions_to_subspace('c', d, means, whitener=w, label='c')",
      "print(json.dumps({str(L): {'basis': s.basis.reshape(-1).tolist(), 'read': s.read.reshape(-1).tolist(), 'mean': s.mean.tolist(), 'coord': float(s.node_coords.reshape(-1)[0]), 'share': m.mahalanobis_share[L]} for L, s in m.layers.items()}))",
    ].join("\n"), join(scratch, "inputs.json")], {
      cwd: repositoryRoot,
      env: { ...process.env, PYTHONPATH: resolve(repositoryRoot, "drowse") },
    })).stdout);
    const close = (actual, expected, label) => {
      for (let i = 0; i < expected.length; i += 1) {
        assert.ok(Math.abs(actual[i] - expected[i]) <= 1e-5 * (1 + Math.abs(expected[i])), `${label}[${i}]: ${actual[i]} vs ${expected[i]}`);
      }
    };
    for (const layer of layerMap) {
      const ours = foundation.finalAffineLayers.get(layer);
      const theirs = reference[String(layer)];
      close(ours.basis, theirs.basis, `L${layer} basis`);
      close(ours.read, theirs.read, `L${layer} read`);
      close(ours.mean, theirs.mean, `L${layer} mean`);
      close(ours.nodeCoordinates, [theirs.coord], `L${layer} pole`);
      close([ours.mahalanobisShare], [theirs.share], `L${layer} share`);
      assert.equal(ours.nodeCount, 1);
    }
  } finally {
    await rm(scratch, { recursive: true, force: true });
  }
  assert.throws(
    () => monopolarFoundation({
      identity: {},
      captureRetained: false,
      layers: new Map([[0, { rows: 1, columns: D, values: fitted.whiteners.get(0).mean.slice() }]]),
    }, fitted.whiteners),
    /indistinguishable from the neutral baseline/,
  );
  // A one-node core concept (6.0's deceptive) authors and validates as a monopolar archive.
  const deceptive = await readCoreManifoldSource(
    resolve(repositoryRoot, "browser-runtime/release-inputs/core-manifolds/deceptive"),
    repositoryRoot,
  );
  const { browserFittedFlatDiscoverPack: buildFlat } = await vite.ssrLoadModule("/src/hosted/artifacts/fittedAuthoring.ts");
  const { validateDrowseArchive: validateArchive } = await vite.ssrLoadModule("/src/hosted/artifacts/drowseArchive.ts");
  const monopolarArchive = await buildFlat({
    manifold: deceptive,
    closure: { source: deceptive.source, tags: deceptive.tags, template: null },
    modelId: "fixture/model",
    producerVersion: "browser-test",
    identity: {
      runtimeIdentitySha256: "1".repeat(64), contextBindingSha256: "2".repeat(64), modelSourceFingerprint: null,
      captureSha256: "3".repeat(64), captureVersion: 3, captureRenderSha256: "4".repeat(64),
      baselinePromptsSha256: "5".repeat(64), fitPolicyVersion: 1,
    },
    topology: foundation.topology,
    consensusGram: foundation.consensusGram,
    nodeCoordinates: foundation.anchoredNodeCoordinates,
    evaluatedLayers: foundation.layers,
    layers: foundation.finalAffineLayers,
  });
  const monopolar = await validateArchive(monopolarArchive);
  assert.deepEqual(monopolar.primaryIdentity, ["default", "deceptive"]);
  assert.equal(monopolar.fittedArtifacts.length, 1);
  const { browserFittedFlatDiscoverPack } = await vite.ssrLoadModule(
    "/src/hosted/artifacts/fittedAuthoring.ts",
  );
  const { validateDrowseArchive } = await vite.ssrLoadModule(
    "/src/hosted/artifacts/drowseArchive.ts",
  );
  const evaluated = {
    operation: "affine_fisher",
    nodeCount: 2,
    columns: 2,
    components: 1,
    centroidMean: new Float64Array(2),
    mean: new Float64Array(2),
    basis: new Float64Array([1, 0]),
    read: new Float64Array([1, 0]),
    nodeCoordinates: new Float64Array([-1, 1]),
    muCoordinates: new Float64Array([-1, 1]),
    whitenedGram: new Float64Array([1, -1, -1, 1]),
    neutralCrossGram: new Float64Array(2),
    explainedVariance: 1,
    mahalanobisShare: 2,
  };
  const archive = await browserFittedFlatDiscoverPack({
    manifold: seed,
    closure: { source: seed.source, tags: seed.tags, template: null },
    modelId: "fixture/model",
    producerVersion: "browser-test",
    identity: {
      runtimeIdentitySha256: "1".repeat(64),
      contextBindingSha256: "2".repeat(64),
      modelSourceFingerprint: null,
      captureSha256: "3".repeat(64),
      captureVersion: 3,
      captureRenderSha256: "4".repeat(64),
      baselinePromptsSha256: "5".repeat(64),
      fitPolicyVersion: 1,
    },
    topology: {
      operation: "topology",
      winnerName: "flat-pca",
      fitMode: "pca",
      intrinsicDimensions: 1,
      periodicDimensions: 0,
      persistentLoops: 0,
      usedFaintCycle: false,
      coordinates: new Float64Array([-1, 1]),
      embeddedCoordinates: new Float64Array([-1, 1]),
      candidates: [],
      diagnostics: {
        kind: "pca",
        perComponentVariance: new Float64Array([1]),
        cumulativeVariance: new Float64Array([1]),
        pickedDimensions: 1,
        threshold: 0.7,
      },
      winnerPlan: null,
    },
    consensusGram: new Float64Array([1, -1, -1, 1]),
    nodeCoordinates: new Float64Array([-1, 1]),
    evaluatedLayers: new Map([[0, evaluated]]),
    layers: new Map([[0, { ...evaluated, affineMap: null }]]),
  });
  const verified = await validateDrowseArchive(archive);
  assert.deepEqual(verified.primaryIdentity, ["default", "welcoming.detached"]);
  assert.equal(verified.fittedArtifacts.length, 1);
  assert.equal(verified.fittedArtifacts[0].variant, "raw");
  assert.equal(verified.fittedArtifacts[0].modelId, "fixture/model");
  assert.equal(verified.fittedArtifacts[0].contextBindingSha256, "2".repeat(64));
  assert.match(verified.fittedArtifacts[0].tensorPath, /\.safetensors$/);
  assert.equal(verified.manifold.source, "local");
  assert.deepEqual(verified.manifold.tags, ["starter", "tone"]);
} finally {
  await vite.close();
}

const builderSource = await readFile(resolve(import.meta.dirname, "build-hosted-whitener.mjs"), "utf8");
assert.match(builderSource, /\.\.\.executionProfiles/);
assert.match(builderSource, /readRuntimeLock\(runtimeLockPath\)/);
assert.match(builderSource, /options\.manualBrowser \? null : await launchReleaseToolContext/);
assert.match(builderSource, /message\.type !== "done"/);
assert.doesNotMatch(builderSource, /manifold === null|manifold !== null/);
const validatorSource = await readFile(resolve(import.meta.dirname, "validate-hosted-core-pack.mjs"), "utf8");
assert.match(validatorSource, /structuredHookProfile: lock\.structuredHookProfile/);
assert.match(validatorSource, /id: `\$\{modelId\}-\$\{lock\.quantization\}`/);
assert.match(validatorSource, /readRuntimeLock\(runtimeLockPath\)/);

// drowse 6.0's default concepts, as core sources: the checked-in copies equal the pinned drowse's
// bundled corpora, and every one reads as a core source (three of them one-node).
const { coreConceptSources, DEFAULT_PROBE_CATEGORIES } = await import("./sync-core-concepts.mjs");
const sessionSource = await readFile(resolve(repositoryRoot, "drowse/drowse/session/session.py"), "utf8");
assert.deepEqual(
  [...sessionSource.match(/PROBE_CATEGORIES = \[([^\]]*)\]/u)[1].matchAll(/"([a-z]+)"/gu)].map((match) => match[1]),
  [...DEFAULT_PROBE_CATEGORIES],
  "the core concepts follow drowse's default probe categories",
);
const concepts = await coreConceptSources();
assert.equal(Object.keys(concepts).length, 17);
for (const [name, files] of Object.entries(concepts)) {
  const directory = resolve(repositoryRoot, "browser-runtime/release-inputs/core-manifolds", name);
  for (const [path, text] of Object.entries(files)) {
    assert.equal(await readFile(resolve(directory, path), "utf8"), text, `${name}/${path} is drowse's (sync-core-concepts.mjs)`);
  }
  const concept = await readCoreManifoldSource(directory, repositoryRoot);
  assert.equal(`${concept.namespace}/${concept.name}`, `default/${name}`);
}
for (const name of ["deceptive", "refusing", "manipulative"]) {
  assert.equal(JSON.parse(concepts[name]["manifold.json"]).nodes.length, 1, `${name} is 6.0's one-node concept`);
}
const sources = (await import("node:fs/promises")).readdir;
assert.deepEqual(
  (await sources(resolve(repositoryRoot, "browser-runtime/release-inputs/core-manifolds"))).sort(),
  [...Object.keys(concepts), "welcoming.detached"].sort(),
  "the release inputs hold the default concepts and the hosted starter only",
);

console.log("Hosted whitener release-tool checks passed");
