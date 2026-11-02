#!/usr/bin/env node
// Half B end to end through the hosted runtime worker on WebGPU: B4's tiny v5 fixture library
// with its R, S, units and core packs, installed, loaded and driven as the app drives the worker
// (check, catalog, download, download_pack, load, probes, live state, generate, the units members).
//
//   node scripts/browser-units-worker.mjs --fixtures DIR --arch qwen3|gemma3 \
//     --quantization q0f32|q4f32_1 --work DIR [--port N] [--headed] [--python python3]
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
//    aggregate's units and lens channels); a generated node's token anatomy (unsteered and steered)
//    and unit activity; and unitRead, unitTop and unitFind. Tolerances are each golden's.
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
const stage = await stageUnitsFixture(options);
const { artifacts: modelFiles, blockSpec, contextTokens, runtimeIdentity, modelDirectory } = stage;
const modelId = `units-${options.arch}`;
const variantId = `${modelId}-${options.quantization}`;
const revision = "a".repeat(40);
const goldenFile = async (name) => JSON.parse(await readFile(resolve(goldensDirectory, `${name}-${options.arch}-v1.json`), "utf8"));

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
const nativeDirectory = resolve(options.work, `${stage.fixture}-native`);
await rm(nativeDirectory, { recursive: true, force: true });
await mkdir(nativeDirectory, { recursive: true });
await writeFile(resolve(nativeDirectory, "runtime.json"), JSON.stringify({
  modelId, runtimeIdentity, runtimeIdentitySha256: fingerprint,
}));
console.log(`drowse: packs and references for ${stage.fixture} (${fingerprint.slice(0, 12)})`);
await exec(options.python, [
  resolve(webui, "scripts/support/units-worker-reference.py"), options.arch, nativeDirectory,
  resolve(nativeDirectory, "runtime.json"),
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
    type: "run", modelId, modelType: "base", artifacts: modelFiles,
    baselinePrompts: ["Hi.", "Go on.", "Say it."], contexts: [contextTokens], rows: [],
    nativeNeutralRows: true, runtimeIdentity, quantization: options.quantization, requiredFeatures: [],
    structuredHookProfile: "standard-v4", thinkingProfile: null, blockSpec, unitRefusal: null,
    artifactProducer: { drowseVersion: "0.0.0-fixture", producerVersion: "browser-units-worker" },
    manifolds,
  });
  check("core concepts and whitener through the release worker", core.type === "done", core.message ?? "");
  if (core.type !== "done") throw new Error("the core build failed");
  const [coreContext] = core.result.contexts;
  const coreDirectory = resolve(options.work, `${stage.fixture}-core`);
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
        id: modelId, modelType: "base", displayName: `Units tiny ${options.arch}`, description: "B4's tiny v5 fixture",
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
      modelType: "base", runtimeIdentity, structuredHookProfile: "standard-v4", thinkingProfile: null,
      convertedRepository: "drowse/units-tiny", convertedRevision: revision, contextProfiles: [contextTokens],
      blockSpec, unitRefusal: null,
    }],
  };

  // ------------------------------------------------- 3. the hosted runtime worker ---
  const unitProbes = await goldenFile("unit-probes");
  const plan = {
    variantId, contextTokens, packIds: packs.slice(1).map((entry) => entry.id),
    text: reference.text, probes: reference.settings.probes, unitProbes: unitProbes.probes,
    probeCases: unitProbes.cases.map((entry) => ({ name: entry.name, steering: entry.steering })),
    generations: Object.entries(reference.generations).map(([name, entry]) => ({ name, steering: entry.steering })),
    nodes: Object.entries(reference.nodes).map(([name, entry]) => ({
      name, steering: entry.steering, anatomyKnobs: entry.anatomy.knobs, activityKnobs: entry.activity?.knobs ?? null,
    })),
    settings: reference.settings,
    reads: reference.reads.map((entry) => ({ unit: entry.unit, k: entry.read.knobs.k, hideHubs: entry.hide_hubs })),
    top: reference.top.map((entry) => ({ token: entry.token, by: entry.by, k: entry.result.knobs.k })),
    find: reference.find.map((entry) => ({ vector: Array.from(decode(entry.vector)), layer: entry.layer, k: entry.result.knobs.k })),
  };
  const run = await page.evaluate(driveWorker, plan);
  if (run.error) {
    check("hosted runtime worker", false, run.error);
    throw new Error(run.error);
  }
  check("worker installs and loads the v5 library with its packs", true, run.loaded);
  check("a plain greedy generation through the worker", JSON.stringify(run.plain) === JSON.stringify(unitProbes.cases[0].tokens.slice(0, 2)),
    `${JSON.stringify(run.plain)} vs ${JSON.stringify(unitProbes.cases[0].tokens.slice(0, 2))}`);
  compareRun(run, unitProbes);
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
  const record = (label, actual, expected, options = {}) => {
    const deltas = [];
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
  for (const [index, golden] of unitProbes.cases.entries()) {
    const ours = run.probeCases[index];
    check(`unit-probes ${golden.name}: tokens`, JSON.stringify(ours.tokens) === JSON.stringify(golden.tokens),
      `${JSON.stringify(ours.tokens)} vs ${JSON.stringify(golden.tokens)}`);
    golden.forwards.forEach((forward, step) => {
      record(`unit-probes ${golden.name}: forward ${step} units channel`,
        ours.frames[step]?.instruments?.units?.readings ?? null, forward.readings);
    });
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
      `${JSON.stringify(ours.tokens)} vs ${JSON.stringify(native.tokens)}`);
    record(`node ${ours.name}: token anatomy`, withoutSources(ours.anatomy), withoutSources(native.anatomy.result),
      { root: `nodes.${ours.name}.anatomy.result`, ties: reference.declared_ties });
    if (native.activity) {
      record(`node ${ours.name}: unit activity`, ours.activity, native.activity.result,
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
    const plain = await generate({ input: plan.text, raw: true, stateless: true, steering: null, sampling: greedy(2) });
    out.plain = plain.tokens.map((frame) => frame.token_id);

    for (const unit of plan.unitProbes) await service("probes", "attach", { selector: unit });
    for (const entry of plan.probeCases) {
      const result = await generate({ input: plan.text, raw: true, stateless: true, steering: entry.steering, sampling: greedy(5) });
      out.probeCases.push({
        name: entry.name,
        tokens: result.tokens.map((frame) => frame.token_id),
        frames: result.tokens.map((frame) => frame.measurements ?? null),
        aggregate: result.done?.measurements ?? null,
      });
    }
    await service("probes", "attach", { selector: `slens/${plan.settings.slensWord}` });
    for (const entry of plan.generations) {
      const result = await generate({ input: plan.text, raw: true, stateless: true, steering: entry.steering, sampling: greedy(plan.settings.steps) });
      out.generations.push({
        name: entry.name,
        tokens: result.tokens.map((frame) => frame.token_id),
        liveUnitFrames: result.tokens.filter((frame) => frame.measurements?.instruments?.units !== undefined).length,
        aggregate: result.done?.measurements ?? null,
      });
    }
    for (const name of [...plan.unitProbes, `slens/${plan.settings.slensWord}`]) await service("probes", "detach", name);

    for (const entry of plan.nodes) {
      const result = await generate({ input: plan.text, raw: true, stateless: false, steering: entry.steering, sampling: greedy(plan.settings.decodeSteps) });
      const nodeId = result.done?.node_id ?? result.tokens.at(-1)?.node_id;
      const node = { name: entry.name, nodeId, tokens: result.tokens.map((frame) => frame.token_id) };
      node.anatomy = await service("instruments", "tokenAnatomy", nodeId, plan.settings.anatomyIndex, entry.anatomyKnobs);
      if (entry.activityKnobs !== null) {
        const activity = plan.settings.activity;
        node.activity = await service("instruments", "unitActivity", nodeId, {
          top: entry.activityKnobs.top, units: activity.units, rawIndex: activity.raw_index, layers: activity.layers,
        });
      }
      out.nodes.push(node);
    }
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
  const out = { port: 0, headed: false, python: process.env.PYTHON ?? "python3" };
  for (let i = 0; i < args.length; i += 1) {
    const flag = args[i];
    if (flag === "--headed") out.headed = true;
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
