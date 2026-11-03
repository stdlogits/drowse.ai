#!/usr/bin/env node
// The in-browser core build on a hook ABI v5 library: B4's tiny fixture libraries through the
// release worker itself (hosted-whitener.worker.ts), on WebGPU.
//
//   node scripts/browser-core-build.mjs --webllm WEB_LLM/lib/index.js --fixtures DIR \
//     --arch qwen3|gemma3 --quantization q0f32|q4f32_1 --work DIR [--port N] [--headed]
//
// DIR holds build-units-fixtures.py's model directory `units-<arch>-<quantization>` and its
// library `libs/units-<arch>-<quantization>-v5.wasm`, staged under --work as a base model
// (support/units-webgpu-stage.mjs). The worker then runs as build-hosted-whitener.mjs runs it: the
// load attests the library's schema-4 descriptor against the fixture's block spec, the neutral
// rows are captured (prepareCaptureRows, capturePreparedRow) and a whitener fitted, and a
// two-node and a one-node core concept are captured through the activation source and fitted.
// Then, through the same hosted runtime, the plain residual capture (the call both captures make)
// of the block-sites golden's ids is checked against drowse's `out` residuals at every layer and
// position, at the golden's library tolerance.
import { createServer as createHttpServer } from "node:http";
import { readFile } from "node:fs/promises";
import { basename, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { createServer as createViteServer } from "vite";
import { releaseToolBrowserOptions, releaseToolGpuArgs } from "./release-tool-storage-quota.mjs";
import { fixtureConcepts, goldensDirectory, stageUnitsFixture } from "./support/units-webgpu-stage.mjs";

const options = parseArguments(process.argv.slice(2));
const webui = fileURLToPath(new URL("..", import.meta.url));
const goldens = goldensDirectory;
const {
  fixture, modelDirectory, roles, artifacts, blockSpec, layerCount, hiddenSize, contextTokens, prefillChunkSize,
  runtimeIdentity,
} = await stageUnitsFixture(options);
// Short rows: the fixture's byte-level vocabulary spends a token per character, in a
// 128-token context.
const buildRequest = {
  type: "run",
  modelId: `units-${options.arch}`,
  modelType: "base",
  artifacts,
  baselinePrompts: ["Hi.", "Go on.", "Say it."],
  contexts: [contextTokens],
  rows: ["The sky is blue.", "Water is wet.", "Cats can nap.", "Trees grow tall.", "Bread is warm.",
    "Rain may fall.", "Birds can sing.", "Snow is cold."].map((content) => ({
    system: "", messages: [{ role: "user", content: "Hi." }, { role: "assistant", content }],
  })),
  nativeNeutralRows: false,
  runtimeIdentity,
  quantization: options.quantization,
  requiredFeatures: [],
  structuredHookProfile: "standard-v4",
  thinkingProfile: null,
  blockSpec,
  unitRefusal: null,
  prefillChunkSize,
  artifactProducer: { drowseVersion: "0.0.0-fixture", producerVersion: "browser-core-build" },
  manifolds: fixtureConcepts(),
};

const blockSitesGolden = JSON.parse(await readFile(resolve(goldens, `block-sites-${options.arch}-v1.json`), "utf8"));
const goldenOut = decodeSite(blockSitesGolden.sites.out);
const goldenIds = blockSitesGolden.ids;
const tolerance = blockSitesGolden.tolerance.library;

// ---------------------------------------------------------------------- the server ---
const outputs = { whitener: new Map(), manifolds: new Map() };
const vite = await createViteServer({
  root: webui, configFile: false, appType: "custom", logLevel: "silent",
  // The fitting worker loads its kernel from /wasm/ (the hosted public directory).
  publicDir: resolve(webui, "public-hosted"),
  server: { middlewareMode: true, hmr: false, watch: null },
});
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
    if (url.pathname === "/") return send("<!doctype html><title>core build</title>", "text/html");
    if (url.pathname === "/worker.js") {
      const transformed = await vite.transformRequest("/scripts/hosted-whitener.worker.ts");
      return send(transformed.code, "text/javascript");
    }
    if (url.pathname === "/webllm.js") return send(await readFile(resolve(options.webllm)), "text/javascript");
    if (url.pathname === "/fitting.js") {
      return send(await readFile(resolve(webui, "public-hosted/wasm/drowse_fitting_wasm.js")), "text/javascript");
    }
    if (url.pathname === "/fitting.wasm") {
      return send(await readFile(resolve(webui, "public-hosted/wasm/drowse_fitting_wasm_bg.wasm")), "application/wasm");
    }
    if (url.pathname === "/model") {
      const path = url.searchParams.get("path");
      if (!(path in roles)) return send("not found", "text/plain", 404);
      return send(await readFile(resolve(modelDirectory, basename(path))), "application/octet-stream");
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
    console.error(error);
    send(error instanceof Error ? error.message : "request failed", "text/plain", 500);
  }
});
await new Promise((ok) => http.listen(options.port, "127.0.0.1", ok));
const origin = `http://127.0.0.1:${http.address().port}`;
const browser = await chromium.launch(releaseToolBrowserOptions({
  headless: !options.headed, args: releaseToolGpuArgs({ headless: !options.headed }),
}));
let failures = 0;
const check = (label, ok, detail = "") => {
  if (!ok) failures += 1;
  console.log(`${ok ? "ok" : "FAIL"} - ${label}${detail ? `: ${detail}` : ""}`);
};
try {
  const page = await browser.newPage();
  page.setDefaultTimeout(600_000);
  page.on("console", (message) => {
    // The Vite client's dev-server socket is not served here; its complaints are noise.
    if (message.type() === "error" && !/vite|Vite server/.test(message.text())) {
      process.stderr.write(`browser console: ${message.text()}\n`);
    }
  });
  await page.goto(origin);

  // 1. The release worker, unchanged, as build-hosted-whitener.mjs drives it.
  const worker = await page.evaluate(async (payload) => {
    const instance = new Worker("/worker.js", { type: "module" });
    const progress = [];
    return await new Promise((done) => {
      instance.onerror = (event) => done({ type: "error", message: event.message ?? "worker error" });
      instance.onmessage = (event) => {
        const message = event.data;
        if (message.type === "progress" || message.type === "manifold-progress") {
          progress.push(message.type);
          return;
        }
        instance.terminate();
        done({ ...message, progress: progress.length });
      };
      instance.postMessage(payload);
    });
  }, buildRequest);
  check("core build worker on the v5 library", worker.type === "done", worker.message ?? "");
  if (worker.type === "done") {
    const [context] = worker.result.contexts;
    check("neutral whitener fitted from browser-captured rows",
      outputs.whitener.get(String(contextTokens))?.length > 0 && context.ridgePerLayer !== undefined,
      `${outputs.whitener.get(String(contextTokens))?.length ?? 0} tensor bytes`);
    check("both core concepts fitted (two-node and one-node)",
      context.manifoldSha256.length === 2 && outputs.manifolds.size === 2,
      context.manifoldSha256.map((digest) => digest.slice(0, 12)).join(", "));
    const whitener = parseSafetensorsHeader(outputs.whitener.get(String(contextTokens)));
    check("whitener tensors cover every layer",
      runtimeIdentity.layerMap.every((layer) => `layer_${layer}.mean` in whitener),
      Object.keys(whitener).filter((key) => key !== "__metadata__").length + " tensors");
  }

  // 2. The capture call itself, through the hosted runtime on the same library, against drowse.
  const capture = await page.evaluate(async ({ artifacts, request, ids }) => {
    const { DrowseWebLlmRuntime } = await import("/src/hosted/runtime/webLlmEngine.ts");
    const runtime = new DrowseWebLlmRuntime(await import("/webllm.js"));
    const revision = "a".repeat(40);
    const files = [];
    for (const entry of artifacts) {
      const blob = await (await fetch(`/model?path=${encodeURIComponent(entry.path)}`)).blob();
      files.push({
        manifest: { ...entry, revision, url: `https://huggingface.co/drowse/local/resolve/${revision}/${entry.path}` },
        file: new File([blob], entry.path),
      });
    }
    const { runtimeIdentitySha256 } = await import("/src/lib/runtime/catalog.ts");
    const fingerprint = runtimeIdentitySha256(request.runtimeIdentity);
    await runtime.load({
      model: { id: request.modelId, modelType: request.modelType },
      variant: {
        id: `${request.modelId}-${request.quantization}`,
        structuredHookProfile: request.structuredHookProfile,
        thinkingProfile: null,
        runtimeIdentity: request.runtimeIdentity,
        runtimeIdentitySha256: fingerprint,
        contextProfiles: request.contexts.map((contextTokens) => ({ contextTokens, bindingSha256: "0".repeat(64) })),
        requirements: {
          features: [],
          limits: {
            maxStorageBufferBindingSize: 134217728,
            maxComputeWorkgroupSizeX: 256,
            maxComputeInvocationsPerWorkgroup: 256,
          },
        },
      },
      contextTokens: request.contexts[0],
      blockSpec: request.blockSpec,
      unitRefusal: null,
      prefillChunkSize: request.prefillChunkSize,
      adapter: await navigator.gpu.requestAdapter(),
      artifacts: files,
      signal: new AbortController().signal,
      onDeviceLost() {},
    });
    const positions = ids.map((_, index) => index);
    const row = { inputIds: ids, position: ids.length - 1 };
    const result = await runtime.capturePreparedPositions(row, positions);
    const blockSites = runtime.blockSitesCapabilities();
    // The q4 gate's steered capture (hosted-residual-capture.worker.ts): probes e_0, e_1 at every
    // layer, with and without a rank-1 group on e_0 at layer 1.
    const { compileStructuredHookProgram } = await import("/src/hosted/runtime/structuredHookProgram.ts");
    const d = request.runtimeIdentity.hiddenSize;
    const axis = (k) => Float32Array.from({ length: d }, (_, index) => (index === k ? 1 : 0));
    const probes = [0, 1].map((k) => ({ kind: "linear", direction: axis(k), bias: 0 }));
    const program = compileStructuredHookProgram(d, request.runtimeIdentity.layerMap.map((layer) => ({
      probes,
      ...(layer === 1
        ? { affineGroups: [{ active: true, basis: [axis(0)], neutral: new Float32Array(d), target: [1.25], along: 0.4, kappa: [0.3] }] }
        : {}),
    })), blockSites.profile);
    const control = await runtime.measurePreparedPositions(row, positions,
      { ...program, affineActive: new Uint32Array(program.affineActive.length) });
    const steered = await runtime.measurePreparedPositions(row, positions, program);
    // browser-app-runtime's injection-arithmetic oracle (browser-steering-positions.ts).
    const { verifySteeringPositions } = await import("/scripts/browser-steering-positions.ts");
    let arithmetic;
    try {
      arithmetic = await verifySteeringPositions(runtime, {
        hiddenSize: d, layerMap: request.runtimeIdentity.layerMap,
        structuredHookProfile: request.structuredHookProfile, quantization: request.quantization,
        source: request.runtimeIdentity.sourceModel,
      }, { arithmeticOnly: true });
    } catch (error) {
      arithmetic = { passed: false, error: String(error?.stack ?? error) };
    }
    await runtime.unload();
    return {
      layerCount: result.layerCount, hiddenSize: result.hiddenSize, positionCount: result.positionCount,
      values: Array.from(result.values),
      profile: blockSites?.profile.id ?? null,
      probeCount: control.probeCount,
      control: Array.from(control.values),
      steered: Array.from(steered.values),
      arithmetic,
    };
  }, { artifacts, request: buildRequest, ids: goldenIds });
  check("hosted runtime attests the library as standard-v4", capture.profile === "standard-v4", String(capture.profile));
  const shapeOk = capture.layerCount === layerCount && capture.hiddenSize === hiddenSize &&
    capture.positionCount === goldenIds.length;
  let worst = 0;
  let outside = 0;
  for (let index = 0; shapeOk && index < goldenOut.length; index += 1) {
    const error = Math.abs(capture.values[index] - goldenOut[index]);
    worst = Math.max(worst, error);
    if (!(error <= tolerance.atol + tolerance.rtol * Math.abs(goldenOut[index]))) outside += 1;
  }
  check("plain residual capture matches drowse's block-sites out [L, T, d]", shapeOk && outside === 0,
    `${goldenOut.length} values, worst |Δ| ${worst.toExponential(2)}, ${outside} outside atol ${tolerance.atol} + rtol ${tolerance.rtol}`);

  // 3. The structured measurement the q4 gate's steered capture reads: [position, layer, probe].
  const K = capture.probeCount;
  const probe = (values, p, l, k) => values[(p * layerCount + l) * K + k];
  let readback = 0;
  let preHook = 0;
  let hook = 0;
  for (let p = 0; p < goldenIds.length; p += 1) {
    for (let l = 0; l < layerCount; l += 1) {
      for (let k = 0; k < 2; k += 1) {
        const plain = capture.values[(l * goldenIds.length + p) * hiddenSize + k];
        readback = Math.max(readback, Math.abs(probe(capture.control, p, l, k) - plain));
        if (l < 1) preHook = Math.max(preHook, Math.abs(probe(capture.steered, p, l, k) - probe(capture.control, p, l, k)));
      }
      if (l === 1) {
        const before = probe(capture.control, p, l, 0);
        const expected = before + 0.4 * (1.25 - 0.3 * before);
        hook = Math.max(hook, Math.abs(probe(capture.steered, p, l, 0) - expected),
          Math.abs(probe(capture.steered, p, l, 1) - probe(capture.control, p, l, 1)));
      }
    }
  }
  check("structured probes read the plain capture's residual (control program)", readback <= 1e-5,
    `worst |Δ| ${readback.toExponential(2)}`);
  check("steered probes equal the control's before the hook layer", preHook === 0, `max |Δ| ${preHook}`);
  check("the hook layer's probes move by along·(target − kappa·h·u) on u only", hook <= 1e-5,
    `worst |Δ| ${hook.toExponential(2)}`);
  check("browser-steering-positions injection arithmetic (arithmeticOnly)", capture.arithmetic.passed === true,
    capture.arithmetic.passed
      ? capture.arithmetic.checks.map((entry) => `${entry.label}: ${entry.maxInjectionError.toExponential(2)}`).join("; ")
      : capture.arithmetic.error);
} finally {
  await browser.close();
  await new Promise((ok) => http.close(ok));
  await vite.close();
}
if (failures > 0) {
  console.error(`${failures} check(s) failed`);
  process.exit(1);
}
console.log(`core build on ${fixture}-v5: every check passed`);

function decodeSite(site) {
  if (site.encoding !== "float32-le-base64") throw new Error(`unexpected golden encoding ${site.encoding}`);
  const bytes = Buffer.from(site.data, "base64");
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

function parseSafetensorsHeader(bytes) {
  if (!bytes) return {};
  const length = Number(bytes.readBigUInt64LE(0));
  return JSON.parse(bytes.subarray(8, 8 + length).toString("utf8"));
}

function parseArguments(args) {
  const result = { headed: false, port: 0 };
  for (let i = 0; i < args.length; i += 1) {
    const key = args[i];
    if (key === "--headed") result.headed = true;
    else if (["--webllm", "--fixtures", "--arch", "--quantization", "--work", "--port"].includes(key)) result[key.slice(2)] = args[++i];
    else throw new Error(`unknown argument ${key}`);
  }
  for (const key of ["webllm", "fixtures", "arch", "quantization", "work"]) {
    if (!result[key]) throw new Error(`--${key} is required`);
  }
  if (!["qwen3", "gemma3"].includes(result.arch)) throw new Error("--arch is qwen3 or gemma3");
  if (!["q0f32", "q4f32_1"].includes(result.quantization)) throw new Error("--quantization is q0f32 or q4f32_1");
  result.port = Number(result.port);
  return result;
}
