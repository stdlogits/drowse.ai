#!/usr/bin/env node
// Half B end to end through the hosted runtime worker on WebGPU: B4's tiny v5 fixture library
// with its R, S, units and core packs, installed, loaded and driven as the app drives the worker
// (check, catalog, download, download_pack, load, probes, live state, generate, the units members).
//
//   node scripts/browser-units-worker.mjs --fixtures DIR --arch qwen3|gemma3 \
//     --quantization q0f32|q4f32_1 --work DIR [--chat | --goldens] [--port N] [--headed] [--python python3]
//
// DIR is what browser-core-build.mjs takes (support/units-webgpu-stage.mjs). The harness:
// 1. stages the library as a hosted base model and has drowse build its packs bound to that
//    runtime identity, with native references on the same model at the hosted lens band
//    (support/units-worker-reference.py);
// 2. fits the core concepts and the neutral whitener from drowse's neutral rows through the release
//    worker (hosted-whitener.worker.ts), as build-hosted-whitener.mjs does, and assembles the core
//    pack;
// 3. runs the hosted runtime worker (units-worker.worker.ts) on that catalog and compares with
//    drowse: the unit-probes golden (greedy raw generations with two unit probes attached and the
//    units family live: unsteered, a neuron and a head ablated, a head ablation gated on a unit no
//    probe attaches: tokens, every forward's units channel, the aggregate); a neuron push, a unit
//    ablation at @response and an S-lens-gated push with an `slens/` probe attached (tokens and the
//    aggregate's units and lens channels); generated nodes' token anatomy and unit activity
//    (unsteered, steered, S-lens-gated with slens/ attached, whose replay steps its gate, and
//    S-lens-gated with nothing attached, whose gate never fires; J-lens- and geometry-gated with
//    their probes attached: drowse loads this run's browser-fitted manifold, and each gate's
//    threshold is set from drowse's own per-step gate readings so it opens at the anatomy's
//    forward and shuts elsewhere); and unitRead, unitTop and unitFind. Tolerances are each golden's.
//
// `--chat` stages the library as the instruct model drowse's fixture chat template makes
// (support/units-webgpu-stage.mjs `chatConversation`) and replaces the raw cases with chat nodes,
// generated through the hosted loom and replayed through renderPromptTokenIds: the goldens'
// prompt (system and user) unsteered and steered, the user turn alone, a second assistant turn and
// a user-seat turn under the first (unsteered and steered); each node's tokens, token anatomy and
// unit activity against drowse's on the same model and template (the reference's `chat` mode).
//
// `--goldens` (chat staging) reads at the goldens' lens band (0.2/0.8, the worker's `lensBand`, as
// drowse's fixture sets `session.lens.band`) and compares with five golden files directly: the
// goldens' chat node (system and user, six distinct greedy tokens: presence penalty 1e4) unsteered,
// steered and segmented (`!neuron/L1n5@prompt`) and the raw first turn, generated through the
// hosted loom; token-anatomy's local:r cases, unit-activity's generated, sink, sink-population and
// segmented cases, every unit-reads read, top and find, and unit-ablation's cases as raw
// generations of the short prompt (each forward's greedy token against the golden logits' argmax;
// the prefill's top eight log-probabilities at temperature 1 against their log-softmax: a token
// row's logprob is the sampler's, native's too, one-hot at temperature 0).
// Not reachable through the worker: token-anatomy no-r-mid (a second R source), unit-activity
// authored (the browser measures no prompts) and its hand-built forwards, neuron-push's lowering
// and unit-ablation mixed (the fixture session's registered direction).
import { execFile } from "node:child_process";
import { createServer as createHttpServer } from "node:http";
import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { chromium } from "playwright";
import { createServer as createViteServer } from "vite";
import { collectPackArtifacts } from "./assemble-hosted-pack.mjs";
import { neutralWhitenerSidecar } from "./neutral-whitener-node.mjs";
import { releaseToolBrowserOptions, releaseToolGpuArgs } from "./release-tool-storage-quota.mjs";
import { compareRecord, decode } from "./support/units-goldens.mjs";
import { fixtureConcepts, goldensDirectory, sha256, stageUnitsFixture } from "./support/units-webgpu-stage.mjs";

const exec = promisify(execFile);
const options = parseArguments(process.argv.slice(2));
const webui = fileURLToPath(new URL("..", import.meta.url));
const repository = resolve(webui, "..");
const stage = await stageUnitsFixture({ ...options, chat: options.chat || options.goldens });
if (options.goldens) options.chat = true;
const modelType = options.chat ? "chat" : "base";
const { artifacts: modelFiles, blockSpec, contextTokens, runtimeIdentity, modelDirectory } = stage;
const modelId = `units-${options.arch}${options.chat ? "-chat" : ""}`;
const variantId = `${modelId}-${options.quantization}`;
const revision = "a".repeat(40);
const goldenFile = async (name) => JSON.parse(await readFile(resolve(goldensDirectory, `${name}-${options.arch}-v1.json`), "utf8"));

/** The unit-activity golden cases the worker reaches (generated nodes and the raw first turn). */
const GOLDEN_ACTIVITY = ["generated", "sink", "sink-population", "segmented"];

const PREVIEW_DEFINES = {
  __DROWSE_HOSTED_RELEASE__: false,
  __DROWSE_SOURCE_REVISION__: "0".repeat(40),
  __DROWSE_SOURCE_URL__: "https://github.com/stdlogits/drowse.ai",
  __DROWSE_VERSION__: "6.0.0",
  __DROWSE_ARTIFACT_SOURCE_REVISION__: "0".repeat(40),
  __DROWSE_RUNTIME_ABI__: "drowse-web-runtime-v2",
  __DROWSE_HOOK_ABI__: "block-sites-v5",
  __DROWSE_HOSTED_CHANNEL__: "preview",
  __DROWSE_FITTING_WASM__: { module: "/fitting.js", wasm: "/fitting.wasm" },
};
const vite = await createViteServer({
  root: webui, configFile: false, appType: "custom", logLevel: "silent",
  publicDir: resolve(webui, "public-hosted"),
  server: { middlewareMode: true, hmr: false, watch: null },
});
const catalogModule = await vite.ssrLoadModule("/src/lib/runtime/catalog.ts");
const fingerprint = catalogModule.runtimeIdentitySha256(runtimeIdentity);
const binding = catalogModule.contextBindingSha256(fingerprint, contextTokens);

// ------------------------------------------------- 1. drowse's packs and references ---
const nativeDirectory = resolve(options.work, `${stage.fixture}${options.chat ? "-chat" : ""}-native`);
await rm(nativeDirectory, { recursive: true, force: true });
await mkdir(nativeDirectory, { recursive: true });
await writeFile(resolve(nativeDirectory, "runtime.json"), JSON.stringify({
  modelId, runtimeIdentity, runtimeIdentitySha256: fingerprint,
}));
console.log(`drowse: packs and references for ${stage.fixture} (${fingerprint.slice(0, 12)})`);
await exec(options.python, [
  resolve(webui, "scripts/support/units-worker-reference.py"), options.arch, nativeDirectory,
  resolve(nativeDirectory, "runtime.json"), ...(options.chat ? ["chat"] : []),
], {
  maxBuffer: 1 << 26,
  env: { ...process.env, PYTHONUNBUFFERED: "1", PYTHONPATH: resolve(repository, "drowse") },
});
const reference = JSON.parse(await readFile(resolve(nativeDirectory, "reference.json"), "utf8"));
const packsDirectory = resolve(nativeDirectory, "packs");
const coreStatistics = JSON.parse(await readFile(resolve(packsDirectory, "core/neutral-statistics.json"), "utf8"));
const neutralRows = await readFile(resolve(packsDirectory, "core", coreStatistics.neutral_rows?.file ?? "neutral-rows.safetensors"));

// ---------------------------------------------------------------------- the server ---
const served = new Map();   // fixture path → absolute file
const outputs = { whitener: new Map(), manifolds: new Map() };
let harnessConfig = null;
const http = createHttpServer(async (request, response) => {
  for (const [name, value] of Object.entries({
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Resource-Policy": "same-origin",
  })) response.setHeader(name, value);
  const send = (body, type, status = 200) => {
    response.writeHead(status, { "content-type": type });
    response.end(body);
  };
  try {
    const url = new URL(request.url ?? "/", "http://127.0.0.1");
    if (url.pathname === "/") return send("<!doctype html><title>units worker</title>", "text/html");
    if (url.pathname === "/units-worker.js") {
      // What vite.hosted.config.ts injects into a preview build, set before the worker's modules
      // evaluate (the dev server has no build step to replace them).
      return send(`Object.assign(globalThis, ${JSON.stringify(PREVIEW_DEFINES)});\nawait import("/units-worker-main.js");\n`, "text/javascript");
    }
    for (const [path, source] of [["/whitener-worker.js", "/scripts/hosted-whitener.worker.ts"], ["/units-worker-main.js", "/scripts/units-worker.worker.ts"]]) {
      if (url.pathname === path) return send((await vite.transformRequest(source)).code, "text/javascript");
    }
    if (url.pathname === "/webllm.js") {
      return send(await readFile(resolve(repository, "node_modules/@drowse/web-llm/lib/index.js")), "text/javascript");
    }
    if (url.pathname === "/fitting.js") {
      return send(await readFile(resolve(webui, "public-hosted/wasm/drowse_fitting_wasm.js")), "text/javascript");
    }
    if (url.pathname === "/fitting.wasm") {
      return send(await readFile(resolve(webui, "public-hosted/wasm/drowse_fitting_wasm_bg.wasm")), "application/wasm");
    }
    if (url.pathname === "/model") {
      const path = url.searchParams.get("path");
      if (!(path in stage.roles)) return send("not found", "text/plain", 404);
      return send(await readFile(resolve(modelDirectory, path)), "application/octet-stream");
    }
    if (url.pathname === "/neutral-rows") return send(neutralRows, "application/octet-stream");
    if (url.pathname === "/harness/config.json" && harnessConfig !== null) {
      return send(JSON.stringify(harnessConfig), "application/json");
    }
    if (url.pathname.startsWith("/fixture/")) {
      const file = served.get(decodeURIComponent(url.pathname.slice("/fixture/".length)));
      if (file === undefined) return send("not found", "text/plain", 404);
      return send(await readFile(file), "application/octet-stream");
    }
    if (request.method === "POST" && (url.pathname === "/output" || url.pathname === "/manifold-output")) {
      const chunks = [];
      for await (const chunk of request) chunks.push(chunk);
      const key = url.pathname === "/output"
        ? url.searchParams.get("context")
        : `${url.searchParams.get("context")}:${url.searchParams.get("manifold")}`;
      outputs[url.pathname === "/output" ? "whitener" : "manifolds"].set(key, Buffer.concat(chunks));
      return send("ok", "text/plain");
    }
    vite.middlewares(request, response, () => send("not found", "text/plain", 404));
  } catch (error) {
    send(String(error?.stack ?? error), "text/plain", 500);
  }
});
await new Promise((ok) => http.listen(options.port, "127.0.0.1", ok));
const origin = `http://127.0.0.1:${http.address().port}`;
const browser = await chromium.launch(releaseToolBrowserOptions({
  headless: !options.headed, args: releaseToolGpuArgs({ headless: !options.headed }),
}));
let failures = 0;
const worst = new Map();
const check = (label, ok, detail = "") => {
  if (!ok) failures += 1;
  console.log(`${ok ? "ok" : "FAIL"} - ${label}${detail ? `: ${detail}` : ""}`);
};
try {
  const page = await browser.newPage();
  page.setDefaultTimeout(900_000);
  page.on("console", (message) => {
    if ((message.type() === "error" || message.type() === "warning") && !/vite|Vite server/.test(message.text())) {
      process.stderr.write(`browser console: ${message.text()}\n`);
    }
  });
  await page.goto(origin);

  // ------------------------- 2. the core pack: concepts and whitener on drowse's rows ---
  const manifolds = fixtureConcepts();
  const core = await page.evaluate(async (payload) => {
    const instance = new Worker("/whitener-worker.js", { type: "module" });
    return await new Promise((done) => {
      instance.onerror = (event) => done({ type: "error", message: event.message ?? "worker error" });
      instance.onmessage = (event) => {
        if (event.data.type === "progress" || event.data.type === "manifold-progress") return;
        instance.terminate();
        done(event.data);
      };
      instance.postMessage(payload);
    });
  }, {
    type: "run", modelId, modelType, artifacts: modelFiles,
    baselinePrompts: ["Hi.", "Go on.", "Say it."], contexts: [contextTokens], rows: [],
    nativeNeutralRows: true, runtimeIdentity, quantization: options.quantization, requiredFeatures: [],
    structuredHookProfile: "standard-v4", thinkingProfile: null, blockSpec, unitRefusal: null,
    artifactProducer: { drowseVersion: "0.0.0-fixture", producerVersion: "browser-units-worker" },
    manifolds,
  });
  check("core concepts and whitener through the release worker", core.type === "done", core.message ?? "");
  if (core.type !== "done") throw new Error("the core build failed");
  const [coreContext] = core.result.contexts;
  const coreDirectory = resolve(options.work, `${stage.fixture}${options.chat ? "-chat" : ""}-core`);
  await rm(coreDirectory, { recursive: true, force: true });
  await mkdir(resolve(coreDirectory, String(contextTokens)), { recursive: true });
  await mkdir(resolve(coreDirectory, "manifolds", String(contextTokens)), { recursive: true });
  const whitenerBytes = outputs.whitener.get(String(contextTokens));
  const sidecarBytes = neutralWhitenerSidecar({
    runtimeIdentitySha256: fingerprint, contextBindingSha256: binding,
    modelLock: { layerMap: runtimeIdentity.layerMap, hiddenSize: runtimeIdentity.hiddenSize },
    tensorsSha256: coreContext.tensorsSha256, ridgePerLayer: coreContext.ridgePerLayer,
  });
  await writeFile(resolve(coreDirectory, String(contextTokens), "neutral-whitener.safetensors"), whitenerBytes);
  await writeFile(resolve(coreDirectory, String(contextTokens), "neutral-whitener.json"), sidecarBytes);
  for (const [index, manifold] of manifolds.entries()) {
    await writeFile(resolve(coreDirectory, "manifolds", String(contextTokens), `${manifold.namespace}-${manifold.name}.drowse`),
      outputs.manifolds.get(`${contextTokens}:${index}`));
  }
  await writeFile(resolve(coreDirectory, "neutral-statistics.json"), JSON.stringify({
    ...coreStatistics,
    whitener: {
      rows_sha256: coreStatistics.neutral.rows_sha256,
      tensors_sha256: { [contextTokens]: coreContext.tensorsSha256 },
      sidecars_sha256: { [contextTokens]: sha256(sidecarBytes) },
    },
  }));
  await writeFile(resolve(coreDirectory, coreStatistics.tensor_file),
    await readFile(resolve(packsDirectory, "core", coreStatistics.tensor_file)));

  // Gated replays: drowse gates on the very manifold this core build fitted (its own loader on
  // the archive's fitted pair) and on the R pack's lens; its nodes join the reference's.
  if (!options.chat) {
    console.log("drowse: gated references on the browser's fitted manifold");
    await exec(options.python, [
      resolve(webui, "scripts/support/units-worker-reference.py"), options.arch, nativeDirectory,
      resolve(nativeDirectory, "runtime.json"), "gated",
      resolve(coreDirectory, "manifolds", String(contextTokens), "local-calm.alert.drowse"),
    ], { maxBuffer: 1 << 26, env: { ...process.env, PYTHONUNBUFFERED: "1", PYTHONPATH: resolve(repository, "drowse") } });
    const gated = JSON.parse(await readFile(resolve(nativeDirectory, "gated-reference.json"), "utf8"));
    Object.assign(reference.nodes, gated.nodes);
    reference.declared_ties.push(...gated.declared_ties);
  }

  // ------------------------------------------------------------- the catalog and lock ---
  const catalogFile = (prefix, file, role) => {
    // Hugging Face's layout, which WebLLM keys a model directory on.
    const path = `${prefix}/resolve/${revision}/${file.path}`;
    served.set(path, file.absolute);
    return { path: file.path, role, url: `https://fixture.invalid/${path}`, revision, bytes: file.bytes, sha256: file.sha256 };
  };
  const listed = async (directory) => (await collectPackArtifacts(directory)).files
    .map((file) => ({ ...file, absolute: resolve(directory, file.path) }));
  const pack = async (kind, directory, role, required) => {
    const files = (await listed(directory)).map((file) => catalogFile(kind, file, role));
    return {
      id: `${modelId}-${kind}`, kind, displayName: `${modelId} ${kind}`, license: "AGPL-3.0-or-later",
      sourceRepository: "drowse/units-tiny", sourceRevision: revision,
      bytes: files.reduce((sum, file) => sum + file.bytes, 0), required, runtimeIdentitySha256: fingerprint,
      compatibleContextBindingSha256: [binding], files,
    };
  };
  const packs = [
    await pack("core", coreDirectory, "core_pack", true),
    ...await Promise.all(["jlens", "slens", "units"].map((kind) => pack(kind, resolve(packsDirectory, kind), "instrument", false))),
  ];
  const files = modelFiles.map((file) => catalogFile("model", { ...file, absolute: resolve(modelDirectory, file.path) }, file.role));
  const limits = {
    maxBufferSize: 1 << 30, maxStorageBufferBindingSize: 1 << 30, maxStorageBuffersPerShaderStage: 10,
    maxComputeWorkgroupStorageSize: 32_768, maxComputeWorkgroupSizeX: 256, maxComputeInvocationsPerWorkgroup: 256,
  };
  const variant = {
    id: variantId, tier: "fastest", structuredHookProfile: "standard-v4", thinkingProfile: null,
    contextProfiles: [{
      contextTokens, bindingSha256: binding, minimumCalibrationScore: null, minimumDeviceMemoryGiB: null,
      expectedPrefillTokensPerSecond: [1, 2], expectedDecodeTokensPerSecond: [1, 2], measuredDevices: 1,
    }],
    downloadBytes: files.reduce((sum, file) => sum + file.bytes, 0),
    requiredCorePackBytes: packs[0].bytes,
    requirements: { features: [], limits: { maxStorageBuffersPerShaderStage: 8, maxComputeWorkgroupSizeX: 256, maxComputeInvocationsPerWorkgroup: 256 } },
    runtimeIdentity, runtimeIdentitySha256: fingerprint, files, packs,
  };
  harnessConfig = {
    document: {
      schemaVersion: 1, sequence: 1, issuedAt: "2026-01-01T00:00:00.000Z", expiresAt: "2099-01-01T00:00:00.000Z",
      runtimeAbi: runtimeIdentity.runtimeAbi,
      models: [{
        id: modelId, modelType, displayName: `Units tiny ${options.arch}`, description: "B4's tiny v5 fixture",
        sourceUrl: "https://huggingface.co/drowse/units-tiny", license: "AGPL-3.0-or-later", languages: ["en"],
        variants: [variant],
      }],
    },
    capabilities: {
      checkedAt: 0, supported: true, secureContext: true, crossOriginIsolated: true, dedicatedWorker: true,
      indexedDb: true, opfs: true, webLocks: true, broadcastChannel: true, webAssembly: true,
      webGpu: { available: true, fallback: "hardware", features: [], limits, adapterInfo: { vendor: "harness" } },
      storage: { quotaBytes: 8e9, usageBytes: 0, availableBytes: 8e9, persisted: true },
      signals: { deviceMemoryGiB: 16, logicalCpuCount: 8, mobile: false, language: "en-US", calibrationScore: 100 },
      deviceSignature: "units-worker-harness", runtimeVersion: runtimeIdentity.runtimeAbi, issues: [],
      operations: Object.fromEntries([
        "download", "generation", "fitting", "manifold_artifacts", "probe_subspace_trails", "jlens_fitting",
        "sae_training", "session_admin", "server_endpoints",
      ].map((name) => [name, { available: true, reasons: [] }])),
      limits: { manifoldFitMaxIntrinsicDim: 4 },
    },
    runtimeLockModels: [{
      modelType, runtimeIdentity, structuredHookProfile: "standard-v4", thinkingProfile: null,
      convertedRepository: "drowse/units-tiny", convertedRevision: revision, contextProfiles: [contextTokens],
      blockSpec, unitRefusal: null,
    }],
  };

  if (options.goldens) {
    const band = JSON.parse(await readFile(resolve(goldensDirectory, "units-tiny-models-v1.json"), "utf8")).band;
    harnessConfig.lensBand = { lo: band[0], hi: band[1] };
  }

  // ------------------------------------------------- 3. the hosted runtime worker ---
  const unitProbes = await goldenFile("unit-probes");
  const goldens = options.goldens ? {
    anatomy: await goldenFile("token-anatomy"), activity: await goldenFile("unit-activity"),
    reads: await goldenFile("unit-reads"), ablation: await goldenFile("unit-ablation"),
  } : null;
  const plan = {
    chat: options.chat, variantId, contextTokens, packIds: packs.slice(1).map((entry) => entry.id),
    text: reference.text, probes: reference.settings.probes, unitProbes: unitProbes.probes,
    probeCases: unitProbes.cases.map((entry) => ({ name: entry.name, steering: entry.steering })),
    generations: Object.entries(reference.generations).map(([name, entry]) => ({ name, steering: entry.steering })),
    goldens: goldens === null ? null : {
      ...reference.golden_inputs,
      anatomyIndex: goldens.anatomy.raw_index,
      anatomy: goldens.anatomy.cases.filter((entry) => entry.lens === "local:r")
        .map((entry) => ({ name: entry.name, steering: entry.steering, knobs: entry.knobs })),
      activity: goldens.activity.cases.filter((entry) => GOLDEN_ACTIVITY.includes(entry.name))
        .map((entry) => ({ name: entry.name, arguments: entry.arguments })),
      reads: goldens.reads.reads.map((entry) => ({ unit: entry.unit, k: entry.k, hideHubs: entry.hide_hubs })),
      top: goldens.reads.top.map((entry) => ({ token: entry.token, by: entry.by, k: entry.k })),
      find: goldens.reads.find.map((entry) => (entry.target === "vector"
        ? { vector: Array.from(decode(entry.vector)), layer: entry.layer, k: entry.k }
        : { selector: entry.target, k: entry.k })),
      ablation: goldens.ablation.cases.filter((entry) => entry.name !== "mixed")
        .map((entry) => ({ name: entry.name, steering: entry.steering, steps: entry.decode_steps })),
    },
    nodes: options.goldens ? [] : Object.entries(reference.nodes).map(([name, entry]) => ({
      name, steering: entry.steering, probes: entry.probes, input: entry.input, system: entry.system,
      parent: entry.parent, seat: entry.seat, raw: entry.raw,
      anatomyKnobs: entry.anatomy.knobs, activityKnobs: entry.activity?.knobs ?? null,
    })),
    settings: reference.settings,
    reads: options.goldens ? [] : reference.reads.map((entry) => ({ unit: entry.unit, k: entry.read.knobs.k, hideHubs: entry.hide_hubs })),
    top: options.goldens ? [] : reference.top.map((entry) => ({ token: entry.token, by: entry.by, k: entry.result.knobs.k })),
    find: options.goldens ? [] : reference.find.map((entry) => ({ vector: Array.from(decode(entry.vector)), layer: entry.layer, k: entry.result.knobs.k })),
  };
  const run = await page.evaluate(driveWorker, plan);
  if (run.error) {
    check("hosted runtime worker", false, run.error);
    throw new Error(run.error);
  }
  check("worker installs and loads the v5 library with its packs", true, run.loaded);
  if (!options.chat) {
    check("a plain greedy generation through the worker", JSON.stringify(run.plain) === JSON.stringify(unitProbes.cases[0].tokens.slice(0, 2)),
      `${JSON.stringify(run.plain)} vs ${JSON.stringify(unitProbes.cases[0].tokens.slice(0, 2))}`);
  }
  compareRun(run, unitProbes);
  if (goldens !== null) compareGoldens(run.goldens, goldens);
} finally {
  await browser.close();
  await new Promise((ok) => http.close(ok));
  await vite.close();
}
for (const [label, value] of worst) console.log(`worst |Δ| ${label}: ${value.toExponential(2)}`);
if (failures > 0) {
  console.error(`units worker on ${stage.fixture}: ${failures} check(s) failed`);
  process.exit(1);
}
console.log(`units worker on ${stage.fixture}: every check passed`);

// ---------------------------------------------------------------------- comparisons ---
function compareRun(run, unitProbes) {
  const tol = unitProbes.tolerance.library;
  const numberTol = (path) => {
    if (/probability$|\.p$/.test(path)) return tol.probability_atol + 2e-6;
    return tol.z_atol + 2e-6;
  };
  const record = (label, value, expected, options = {}) => {
    const deltas = [];
    const actual = withoutUnderflow(value);
    const problems = compareRecord(actual, expected, {
      numberTol: (path, e) => {
        return numberTol(path, e);
      },
      ...options,
    });
    measureWorst(label, actual, expected, deltas);
    check(label, problems.length === 0, problems.slice(0, 4).join("; "));
    return deltas;
  };
  // unit-probes: tokens, every forward's units channel, gate scalars through the gate case, aggregate.
  for (const [index, golden] of run.probeCases.length === 0 ? [] : unitProbes.cases.entries()) {
    const ours = run.probeCases[index];
    // Frames by raw index: a piece whose text completes in a later token (a split UTF-8 byte)
    // rides that token's frame, so a forward can have no frame of its own.
    const tokens = ours.tokens.filter(([raw]) => raw !== null);
    check(`unit-probes ${golden.name}: tokens`,
      tokens.length > 0 && tokens.every(([raw, id]) => golden.tokens[raw] === id) &&
        tokens.at(-1)[0] === golden.tokens.length - 1,
      `${JSON.stringify(ours.tokens)} vs ${JSON.stringify(golden.tokens)}`);
    for (const [raw, measurements] of ours.frames) {
      record(`unit-probes ${golden.name}: forward ${raw} units channel`,
        measurements?.instruments?.units?.readings ?? null, golden.forwards[raw]?.readings ?? null);
    }
    record(`unit-probes ${golden.name}: aggregate units`, ours.aggregate?.instruments?.units ?? null,
      golden.aggregate.instruments.units);
  }
  // generations with unit and slens/ probes: tokens and the aggregate's units and lens channels.
  for (const ours of run.generations) {
    const native = reference.generations[ours.name];
    check(`${ours.name} (${native.steering}): tokens`, JSON.stringify(ours.tokens) === JSON.stringify(native.tokens),
      `${JSON.stringify(ours.tokens)} vs ${JSON.stringify(native.tokens)}`);
    record(`${ours.name}: aggregate units`, ours.aggregate?.instruments?.units ?? null, native.aggregate.instruments.units);
    record(`${ours.name}: aggregate slens/ readings`, ours.aggregate?.instruments?.lens?.readings ?? null,
      native.aggregate.instruments.lens.readings);
    check(`${ours.name}: lens binding names the S source`,
      JSON.stringify(Object.keys(ours.aggregate?.instruments?.lens?.binding?.sources ?? {})) === JSON.stringify(["S"]),
      JSON.stringify(ours.aggregate?.instruments?.lens?.binding));
    check(`${ours.name}: units live rides the token envelopes`, ours.liveUnitFrames === ours.tokens.length,
      `${ours.liveUnitFrames} of ${ours.tokens.length}`);
  }
  // nodes: token anatomy and unit activity.
  for (const ours of run.nodes) {
    const native = reference.nodes[ours.name];
    check(`node ${ours.name}: tokens`, JSON.stringify(ours.tokens) === JSON.stringify(native.tokens),
      `${JSON.stringify(ours.tokens)} vs ${JSON.stringify(native.tokens)} (frames ${JSON.stringify(ours.frameTypes)})`);
    record(`node ${ours.name}: token anatomy`, withoutSources(ours.anatomy), withoutSources(native.anatomy.result),
      { root: `nodes.${ours.name}.anatomy.result`, ties: reference.declared_ties });
    if (native.activity) {
      // Node ids are each tree's own; the population is fp16, matched within one ulp.
      const population = populationProblems(ours.activity?.population ?? null, native.activity.result.population);
      check(`node ${ours.name}: unit activity population (fp16, one ulp)`, population.length === 0, population.slice(0, 3).join("; "));
      // A unit whose peak row has no margin (drowse's `_check_activity`) is compared by value.
      const tied = new Set(native.activity.tied_peak_rows ?? []);
      const rowless = (result) => ({
        ...result, node_id: null, population: null,
        top: (result?.top ?? []).map((entry) => (tied.has(entry.unit) ? { ...entry, raw_index: null } : entry)),
      });
      record(`node ${ours.name}: unit activity`, rowless(ours.activity), rowless(native.activity.result),
        { root: `nodes.${ours.name}.activity.result`, ties: reference.declared_ties });
    }
  }
  run.reads.forEach((ours, i) => record(`unitRead ${reference.reads[i].unit}`, ours, reference.reads[i].read.result,
    { root: `reads[${i}].result`, ties: reference.declared_ties }));
  run.top.forEach((ours, i) => record(`unitTop ${reference.top[i].token} by ${reference.top[i].by}`, ours,
    reference.top[i].result.result, { root: `top[${i}].result`, ties: reference.declared_ties }));
  run.find.forEach((ours, i) => record(`unitFind vector at L${reference.find[i].layer}`, ours,
    reference.find[i].result.result, { root: `find[${i}].result`, ties: reference.declared_ties }));
}

/** The worker's answers at the goldens' band against the golden files themselves. */
function compareGoldens(ours, goldens) {
  const tol = goldens.anatomy.tolerance.library;
  const numberTol = (path) => (/probability$|\.p$/.test(path) ? tol.probability_atol : tol.z_atol) + 2e-6;
  const record = (label, value, expected, options) => {
    const actual = withoutUnderflow(value);
    const want = withoutUnderflow(expected);
    const problems = compareRecord(actual, want, { numberTol, ...options });
    measureWorst(label, actual, want);
    check(label, problems.length === 0, problems.slice(0, 4).join("; "));
  };
  const caseOf = (name) => goldens.activity.cases.find((entry) => entry.name === name);
  for (const name of ["generated", "segmented", "sink"]) {
    const want = caseOf(name).result.tokens.map((entry) => entry.token_id);
    check(`golden node ${name}: tokens`, JSON.stringify(ours.tokens[name]) === JSON.stringify(want),
      `${JSON.stringify(ours.tokens[name])} vs ${JSON.stringify(want)}`);
  }
  goldens.anatomy.cases.filter((entry) => entry.lens === "local:r").forEach((entry, i) => {
    const index = goldens.anatomy.cases.indexOf(entry);
    record(`golden token-anatomy ${entry.name}`, withoutSources(ours.anatomy[i]), withoutSources(entry.envelope),
      { root: `cases[${index}].envelope`, ties: goldens.anatomy.declared_ties });
  });
  GOLDEN_ACTIVITY.forEach((name, i) => {
    const entry = caseOf(name);
    const index = goldens.activity.cases.indexOf(entry);
    const tied = new Set(entry.tied_peak_rows ?? []);
    const rowless = (result) => ({
      ...result, population: null,
      top: (result?.top ?? []).map((item) => (tied.has(item.unit) ? { ...item, raw_index: null } : item)),
    });
    const { node_id: _nodeId, ...mine } = ours.activity[i] ?? {};
    const population = populationProblems(mine.population ?? null, entry.result.population ?? null);
    check(`golden unit-activity ${name}: population (fp16, one ulp)`, population.length === 0, population.slice(0, 3).join("; "));
    record(`golden unit-activity ${name}`, rowless(mine), rowless(entry.result),
      { root: `cases[${index}].result`, ties: goldens.activity.declared_ties });
  });
  const reads = goldens.reads;
  reads.reads.forEach((entry, i) => record(`golden unit-reads read ${entry.unit}`, ours.reads[i], entry.read,
    { root: `reads[${i}].read`, ties: reads.declared_ties }));
  reads.top.forEach((entry, i) => record(`golden unit-reads top ${entry.token} by ${entry.by}`, ours.top[i], entry.result,
    { root: `top[${i}].result`, ties: reads.declared_ties }));
  reads.find.forEach((entry, i) => record(`golden unit-reads find ${entry.target}`, ours.find[i], entry.result,
    { root: `find[${i}].result`, ties: reads.declared_ties }));
  // unit-ablation: every forward's greedy token is the golden logits' argmax; the prefill's top
  // eight log-probabilities at temperature 1 are the golden logits' log-softmax (the library
  // tolerance on fp32 logits).
  goldens.ablation.cases.filter((entry) => entry.name !== "mixed").forEach((entry, i) => {
    const mine = ours.ablation[i] ?? { tokens: null, prefill: null };
    const problems = [];
    const logSoftmax = (encoded) => {
      const logits = Array.from(decode(encoded));
      const max = Math.max(...logits);
      const lse = max + Math.log(logits.reduce((sum, value) => sum + Math.exp(value - max), 0));
      return logits.map((value) => value - lse);
    };
    entry.forwards.forEach((forward, step) => {
      const logp = logSoftmax(forward.logits);
      const best = logp.reduce((at, value, index) => (value > logp[at] ? index : at), 0);
      if (mine.tokens?.[step] !== best) problems.push(`forward ${step}: token ${mine.tokens?.[step]} != ${best}`);
    });
    const logp = logSoftmax(entry.forwards[0].logits);
    const want = logp.map((value, id) => [id, value]).sort((a, b) => b[1] - a[1]).slice(0, 8);
    // A prefill whose every sample is a lone UTF-8 byte streams no frame of its own (native's
    // decode buffers it the same way): its argmax is checked above, its alternatives cannot be read.
    let note = "";
    if (mine.prefill === null) note = " (prefill alternatives unreadable: every sample is a lone UTF-8 byte)";
    else {
      let worstLogprob = 0;
      const alts = mine.prefill.alts ?? [];
      want.forEach(([id, value], rank) => {
        const alt = alts.find((item) => item.id === id);
        if (alt === undefined) return problems.push(`prefill: token ${id} (rank ${rank}) missing from the top alternatives`);
        const delta = Math.abs(alt.logprob - value);
        worstLogprob = Math.max(worstLogprob, delta);
        if (!(delta <= tol.atol + tol.rtol * Math.abs(value))) problems.push(`prefill: token ${id} logprob off by ${delta}`);
      });
      const chosen = Math.abs(mine.prefill.logprob - logp[mine.prefill.token]);
      if (!(chosen <= tol.atol + tol.rtol * Math.abs(logp[mine.prefill.token]))) problems.push(`prefill: sampled token logprob off by ${chosen}`);
      worst.set("golden unit-ablation prefill logprob", Math.max(worst.get("golden unit-ablation prefill logprob") ?? 0, worstLogprob, chosen));
    }
    check(`golden unit-ablation ${entry.name} (${entry.steering}): ${entry.forwards.length} forwards${note}`,
      problems.length === 0, problems.slice(0, 4).join("; "));
  });
}

/** An activity population's fp16 rows (base64, little-endian) against drowse's, one ulp apart at
 *  most (units-activity.test.mjs's rule). */
function populationProblems(actual, expected) {
  if (expected === null || actual === null) return actual === expected ? [] : ["population presence differs"];
  const halfBits = (text) => {
    const bytes = Buffer.from(text, "base64");
    return Array.from({ length: bytes.length / 2 }, (_, i) => bytes[2 * i] | (bytes[2 * i + 1] << 8));
  };
  const ordered = (bits) => (bits & 0x8000 ? -(bits & 0x7fff) : bits);
  const problems = [];
  if (actual.raw_index !== expected.raw_index || actual.encoding !== expected.encoding ||
    JSON.stringify(actual.layers.map((l) => l.layer)) !== JSON.stringify(expected.layers.map((l) => l.layer))) {
    return ["population header differs"];
  }
  actual.layers.forEach((layer, i) => {
    for (const key of ["neurons", "heads"]) {
      const got = halfBits(layer[key]);
      const want = halfBits(expected.layers[i][key]);
      if (got.length !== want.length) problems.push(`L${layer.layer} ${key} length`);
      got.forEach((bits, j) => {
        if (Math.abs(ordered(bits) - ordered(want[j])) > 1) problems.push(`L${layer.layer} ${key}[${j}]`);
      });
    }
  });
  return problems;
}

/** `value` without probability entries that underflowed to exactly 0 (the reference's
 *  `without_underflow`): which zero-probability tokens fill a list is arbitrary. */
function withoutUnderflow(value) {
  if (Array.isArray(value)) return value.map(withoutUnderflow);
  if (value === null || typeof value !== "object") return value;
  const probability = value.unit === "probability";
  return Object.fromEntries(Object.entries(value).map(([key, child]) => [key,
    probability && Array.isArray(child)
      ? child.filter((item) => !(item && typeof item === "object" && item.value === 0))
      : withoutUnderflow(child)]));
}

/** The lens sources' labels are each runtime's own (native `local:r`, a pack id here). */
function withoutSources(value) {
  return JSON.parse(JSON.stringify(value, (key, item) =>
    key === "sources" && item && typeof item === "object" ? Object.keys(item).sort() : item));
}

function measureWorst(label, actual, expected) {
  const family = label.replace(/^unit-probes \S+: forward \d+ /, "unit-probes ").replace(/^(node \S+|push|ablate-response|slens-gated): /, "");
  let max = worst.get(family) ?? 0;
  const walk = (a, e) => {
    if (typeof e === "number" && typeof a === "number") max = Math.max(max, Math.abs(a - e));
    else if (Array.isArray(e) && Array.isArray(a)) e.forEach((item, i) => walk(a[i], item));
    else if (e && typeof e === "object" && a && typeof a === "object") for (const key of Object.keys(e)) walk(a[key], e[key]);
  };
  walk(actual, expected);
  worst.set(family, max);
}

// ------------------------------------------------------------- the page-side driver ---
async function driveWorker(plan) {
  const worker = new Worker("/units-worker.js", { type: "module" });
  const pending = new Map();
  const generations = new Map();
  let ready;
  let broken;
  const readyPromise = new Promise((ok, fail) => { ready = ok; broken = fail; });
  let sequence = 0;
  const trail = [];
  let lastStatus = null;
  worker.onerror = (event) => {
    const error = new Error(`worker error: ${event.message ?? "unknown"} (${event.filename ?? ""}:${event.lineno ?? ""})`);
    broken(error);
    for (const entry of pending.values()) entry.fail(error);
    pending.clear();
  };
  worker.onmessage = (event) => {
    const message = event.data;
    if (message?.kind === "harness-ready") return ready();
    if (message?.kind === "response") {
      const entry = pending.get(message.requestId);
      if (entry) {
        pending.delete(message.requestId);
        if (message.ok) entry.ok(message.result);
        else entry.fail(new Error(`${message.error?.code ?? "ERROR"}: ${message.error?.message ?? "failed"}`));
      }
      return;
    }
    if (message?.kind === "event" && message.event === "status") lastStatus = message.payload;
    if (message?.kind === "event" && message.requestId !== null && generations.has(message.requestId)) {
      generations.get(message.requestId).push({ ...message.payload, type: message.payload?.type ?? message.event });
    }
  };
  const call = (command, payload) => {
    const requestId = `harness-${++sequence}`;
    trail.push(`${command}${command === "request" ? ` ${payload.service}.${payload.method}` : ""}`);
    return {
      requestId,
      promise: new Promise((ok, fail) => {
        const timer = setTimeout(() => {
          pending.delete(requestId);
          const frames = generations.get(requestId)?.map((frame) => frame.type) ?? [];
          fail(new Error(`${trail.at(-1)} timed out (frames ${JSON.stringify(frames)}; status ${JSON.stringify(lastStatus?.generation ?? lastStatus?.lifecycle ?? null)})`));
        }, command === "generate" ? 120_000 : 300_000);
        pending.set(requestId, {
          ok: (value) => { clearTimeout(timer); ok(value); },
          fail: (error) => { clearTimeout(timer); fail(error); },
        });
        worker.postMessage({ protocolVersion: 1, requestId, command, payload });
      }),
    };
  };
  const send = (command, payload) => call(command, payload).promise;
  const service = (serviceName, method, ...args) => send("request", { service: serviceName, method, args });
  const generate = async (request) => {
    const requestId = `harness-${sequence + 1}`;
    generations.set(requestId, []);
    await call("generate", { type: "generate", ...request }).promise;
    const frames = generations.get(requestId);
    generations.delete(requestId);
    const tokens = frames.filter((frame) => frame.type === "token" && frame.thinking !== true);
    const done = frames.find((frame) => frame.type === "done");
    return { frames, tokens, done };
  };
  const greedy = (steps) => ({ temperature: 0, max_tokens: steps, seed: 0 });
  const recordedTokens = async (nodeId) =>
    (await service("tree", "get")).nodes.find((item) => item.id === nodeId)?.raw_token_ids ?? null;
  // The goldens' nodes through the hosted loom, read at the goldens' band (the worker's lensBand).
  const driveGoldens = async (g, root) => {
    // drowse's goldens decode six distinct greedy tokens (presence penalty 1e4).
    const distinct = { temperature: 0, max_tokens: g.decode_steps, seed: 0, presence_penalty: 1e4 };
    const chatNode = async (steering) => {
      const result = await generate({
        input: g.user, raw: false, stateless: false, parent_node_id: root, steering, sampling: distinct,
        system_prompt: g.system, generate_seat: "assistant",
      });
      return result.done?.node_id ?? result.tokens.at(-1)?.node_id;
    };
    const nodes = {
      generated: await chatNode(null),
      steered: await chatNode(g.anatomy_steering),
      segmented: await chatNode(g.segmented_steering),
    };
    const first = await generate({
      input: g.raw_first, raw: true, stateless: false, parent_node_id: root, steering: null, sampling: greedy(3),
    });
    nodes.sink = first.done?.node_id ?? first.tokens.at(-1)?.node_id;
    nodes["sink-population"] = nodes.sink;
    const result = { tokens: {}, anatomy: [], activity: [], reads: [], top: [], find: [], ablation: [] };
    for (const [name, nodeId] of Object.entries(nodes)) result.tokens[name] = await recordedTokens(nodeId);
    for (const entry of g.anatomy) {
      result.anatomy.push(await service("instruments", "tokenAnatomy", entry.steering ? nodes.steered : nodes.generated,
        g.anatomyIndex, { ...entry.knobs, raw: false }));
    }
    for (const entry of g.activity) {
      const a = entry.arguments;
      result.activity.push(await service("instruments", "unitActivity", nodes[entry.name], {
        top: a.top, units: a.units, rawIndex: a.raw_index, layers: a.layers, raw: a.raw,
      }));
    }
    for (const entry of g.reads) result.reads.push(await service("instruments", "unitRead", entry.unit, { k: entry.k, hideHubs: entry.hideHubs }));
    for (const entry of g.top) result.top.push(await service("instruments", "unitTop", entry.token, { by: entry.by, k: entry.k }));
    for (const entry of g.find) result.find.push(await service("instruments", "unitFind", entry));
    for (const entry of g.ablation) {
      // The greedy decode under the case's program (its node's record), and the prefill once more
      // at temperature 1: a token row's logprobs are the sampler's (native's too), which only then
      // are the model's log-softmax, read through its top alternatives.
      const run = await generate({
        input: g.short_text, raw: true, stateless: false, parent_node_id: root, steering: entry.steering,
        sampling: greedy(entry.steps + 1),
      });
      const tokens = await recordedTokens(run.done?.node_id ?? run.tokens.at(-1)?.node_id);
      // A sampled lone UTF-8 byte streams no frame of its own: another seed then.
      let first;
      for (let seed = 0; first === undefined && seed < 8; seed += 1) {
        const sampled = await generate({
          input: g.short_text, raw: true, stateless: true, steering: entry.steering,
          sampling: { temperature: 1, top_p: 1, top_k: 0, max_tokens: 1, seed, return_top_k: 8 },
        });
        first = sampled.tokens[0];
      }
      result.ablation.push({ tokens, prefill: first === undefined ? null : { token: first.token_id, logprob: first.logprob, alts: first.top_alts } });
    }
    return result;
  };
  try {
    await readyPromise;
    await send("check", undefined);
    await send("catalog", {});
    await send("download", { modelVariantId: plan.variantId, contextTokens: plan.contextTokens });
    for (const packId of plan.packIds) await send("download_pack", { modelVariantId: plan.variantId, packId });
    await send("load", { modelVariantId: plan.variantId, contextTokens: plan.contextTokens, resetSession: true });
    await service("instruments", "setLive", "units", { enabled: true });
    const out = { loaded: `${plan.packIds.length} optional packs`, probeCases: [], generations: [], nodes: [] };
    // A plain greedy generation first: no program, no probes.
    if (!plan.chat) {
      const plain = await generate({ input: plan.text, raw: true, stateless: true, steering: null, sampling: greedy(2) });
      out.plain = plain.tokens.map((frame) => frame.token_id);
    }

    for (const unit of plan.chat ? [] : plan.unitProbes) await service("probes", "attach", { selector: unit });
    for (const entry of plan.chat ? [] : plan.probeCases) {
      const result = await generate({ input: plan.text, raw: true, stateless: true, steering: entry.steering, sampling: greedy(5) });
      out.probeCases.push({
        name: entry.name,
        tokens: result.tokens.map((frame) => [frame.raw_index, frame.token_id]),
        frames: result.tokens.map((frame) => [frame.raw_index, frame.measurements ?? null]),
        aggregate: result.done?.result?.measurements ?? null,
      });
    }
    if (!plan.chat) await service("probes", "attach", { selector: `slens/${plan.settings.slensWord}` });
    for (const entry of plan.generations) {
      const result = await generate({ input: plan.text, raw: true, stateless: true, steering: entry.steering, sampling: greedy(plan.settings.steps) });
      out.generations.push({
        name: entry.name,
        tokens: result.tokens.map((frame) => frame.token_id),
        liveUnitFrames: result.tokens.filter((frame) => frame.measurements?.instruments?.units !== undefined).length,
        aggregate: result.done?.result?.measurements ?? null,
      });
    }
    if (!plan.chat) {
      for (const name of [...plan.unitProbes, `slens/${plan.settings.slensWord}`]) await service("probes", "detach", name);
    }

    const root = (await service("tree", "get")).root_id;
    const nodeIds = new Map();
    for (const entry of plan.nodes) {
      // The node's probes stay attached through its generation and its replays, as drowse's.
      for (const selector of entry.probes) await service("probes", "attach", { selector });
      // From the root (or the named parent node), as drowse's reference generates it.
      const result = await generate({
        input: entry.input, raw: entry.raw, stateless: false,
        parent_node_id: entry.parent === null ? root : nodeIds.get(entry.parent), steering: entry.steering,
        sampling: greedy(plan.settings.decodeSteps),
        ...(entry.raw ? {} : { system_prompt: entry.system, generate_seat: entry.seat }),
      });
      const nodeId = result.done?.node_id ?? result.tokens.at(-1)?.node_id;
      nodeIds.set(entry.name, nodeId);
      // The node's raw token record (what its replays feed): a piece whose UTF-8 never completes
      // (a lone lead byte, repeated) streams no token frame of its own.
      const recorded = (await service("tree", "get")).nodes.find((item) => item.id === nodeId);
      const node = { name: entry.name, nodeId, tokens: recorded?.raw_token_ids ?? null,
        frameTypes: result.frames.map((frame) => `${frame.type}${frame.thinking ? ":thinking" : ""}`) };
      node.anatomy = await service("instruments", "tokenAnatomy", nodeId, plan.settings.anatomyIndex,
        { ...entry.anatomyKnobs, raw: entry.raw });
      if (entry.activityKnobs !== null) {
        const activity = plan.settings.activity;
        node.activity = await service("instruments", "unitActivity", nodeId, {
          top: entry.activityKnobs.top, units: activity.units, rawIndex: activity.raw_index, layers: activity.layers,
          raw: entry.raw,
        });
      }
      for (const selector of entry.probes) await service("probes", "detach", selector);
      out.nodes.push(node);
    }
    if (plan.goldens !== null) out.goldens = await driveGoldens(plan.goldens, root);
    out.reads = [];
    for (const entry of plan.reads) out.reads.push(await service("instruments", "unitRead", entry.unit, { k: entry.k, hideHubs: entry.hideHubs }));
    out.top = [];
    for (const entry of plan.top) out.top.push(await service("instruments", "unitTop", entry.token, { by: entry.by, k: entry.k }));
    out.find = [];
    for (const entry of plan.find) out.find.push(await service("instruments", "unitFind", { vector: entry.vector, layer: entry.layer, k: entry.k }));
    return out;
  } catch (error) {
    return { error: `${String(error?.stack ?? error)} (after ${trail.join(" → ")})` };
  } finally {
    worker.terminate();
  }
}

function parseArguments(args) {
  const out = { port: 0, headed: false, chat: false, goldens: false, python: process.env.PYTHON ?? "python3" };
  for (let i = 0; i < args.length; i += 1) {
    const flag = args[i];
    if (flag === "--headed") out.headed = true;
    else if (flag === "--chat") out.chat = true;
    else if (flag === "--goldens") out.goldens = true;
    else if (["--fixtures", "--arch", "--quantization", "--work", "--port", "--python"].includes(flag)) {
      out[flag.slice(2)] = flag === "--port" ? Number(args[++i]) : args[++i];
    } else throw new Error(`unknown argument ${flag}`);
  }
  for (const required of ["fixtures", "arch", "quantization", "work"]) {
    if (!out[required]) throw new Error(`--${required} is required`);
  }
  out.fixtures = resolve(out.fixtures);
  out.work = resolve(out.work);
  return out;
}
