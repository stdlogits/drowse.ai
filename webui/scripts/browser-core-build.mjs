#!/usr/bin/env node
// The in-browser core build on a hook ABI v5 library: B4's tiny fixture libraries through the
// release worker itself (hosted-whitener.worker.ts), on WebGPU.
//
//   node scripts/browser-core-build.mjs --webllm WEB_LLM/lib/index.js --fixtures DIR \
//     --arch qwen3|gemma3 --quantization q0f32|q4f32_1 --work DIR [--port N] [--headed]
//
// DIR holds build-units-fixtures.py's model directory `units-<arch>-<quantization>` and its
// library `libs/units-<arch>-<quantization>-v5.wasm`. The model is staged under --work as a
// base model (a tokenizer_config.json without a chat template, the capture prefix the hosted
// loader asks a base model for). The worker then runs as build-hosted-whitener.mjs runs it: the
// load attests the library's schema-4 descriptor against the fixture's block spec, the neutral
// rows are captured (prepareCaptureRows, capturePreparedRow) and a whitener fitted, and a
// two-node and a one-node core concept are captured through the activation source and fitted.
// Then, through the same hosted runtime, the plain residual capture (the call both captures make)
// of the block-sites golden's ids is checked against drowse's `out` residuals at every layer and
// position, at the golden's library tolerance.
import { createHash } from "node:crypto";
import { createServer as createHttpServer } from "node:http";
import { copyFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { basename, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { createServer as createViteServer } from "vite";
import { releaseToolBrowserOptions, releaseToolGpuArgs } from "./release-tool-storage-quota.mjs";

const options = parseArguments(process.argv.slice(2));
const webui = fileURLToPath(new URL("..", import.meta.url));
const goldens = resolve(webui, "../drowse/tests/fixtures/browser");
const fixture = `units-${options.arch}-${options.quantization}`;
const sourceDirectory = resolve(options.fixtures, fixture);
const libraryPath = resolve(options.fixtures, "libs", `${fixture}-v5.wasm`);
const modelDirectory = resolve(options.work, fixture);
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");

// ------------------------------------------------------------- the staged base model ---
await mkdir(modelDirectory, { recursive: true });
const chatConfig = JSON.parse(await readFile(resolve(sourceDirectory, "mlc-chat-config.json"), "utf8"));
chatConfig.tokenizer_files = ["tokenizer.json", "tokenizer_config.json"];
// A base model's capture prefix is its completion prefix (Gemma's BOS; nothing for Qwen).
chatConfig.conv_template = {
  ...chatConfig.conv_template,
  system_prefix_token_ids: chatConfig.drowse_completion_prefix_token_ids ?? [],
};
await writeFile(resolve(modelDirectory, "mlc-chat-config.json"), JSON.stringify(chatConfig, null, 2));
await writeFile(resolve(modelDirectory, "tokenizer_config.json"), JSON.stringify({ chat_template: null }));
const tensorCache = JSON.parse(await readFile(resolve(sourceDirectory, "tensor-cache.json"), "utf8"));
const shards = tensorCache.records.map((record) => record.dataPath);
for (const name of ["tensor-cache.json", "tokenizer.json", ...shards]) {
  await copyFile(resolve(sourceDirectory, name), resolve(modelDirectory, name));
}
await copyFile(libraryPath, resolve(modelDirectory, "model.wasm"));
const roles = {
  "mlc-chat-config.json": "configuration",
  "tensor-cache.json": "converted_manifest",
  "tokenizer.json": "tokenizer",
  "tokenizer_config.json": "chat_template",
  "model.wasm": "model_library",
  ...Object.fromEntries(shards.map((shard) => [shard, "weight"])),
};
const artifacts = [];
for (const [path, role] of Object.entries(roles)) {
  const bytes = await readFile(resolve(modelDirectory, path));
  artifacts.push({ path, role, bytes: bytes.length, sha256: sha256(bytes) });
}
const artifactSha = (path) => artifacts.find((artifact) => artifact.path === path).sha256;

// ------------------------------------------------------------------- fixture facts ---
const tiny = JSON.parse(await readFile(resolve(goldens, "units-tiny-models-v1.json"), "utf8")).arches[options.arch];
const config = tiny.config;
const gemma = options.arch === "gemma3";
const blockSpec = {
  dFF: config.intermediate_size, nHeads: config.num_attention_heads,
  nKvHeads: config.num_key_value_heads, headDim: config.head_dim, sandwich: gemma,
  normOffset: gemma ? 1 : 0, activation: gemma ? "gelu_tanh" : "silu",
  embedScale: gemma ? Math.sqrt(config.hidden_size) : 1, rmsEps: config.rms_norm_eps, qkNorm: true,
  attnScaling: (gemma ? config.query_pre_attn_scalar : config.head_dim) ** -0.5,
  attnLogitSoftcap: null, finalLogitSoftcap: null,
  ropeTheta: gemma ? config.rope_parameters.full_attention.rope_theta : config.rope_parameters.rope_theta,
  ropeLocalTheta: gemma ? config.rope_parameters.sliding_attention.rope_theta : null,
  ropeGlobalScale: 1, ropeInGraph: gemma,
  slidingWindow: gemma ? config.sliding_window : null, slidingPattern: gemma ? 2 : null,
};
const layerCount = config.num_hidden_layers;
const hiddenSize = config.hidden_size;
const contextTokens = chatConfig.context_window_size;
const runtimeIdentity = {
  sourceModel: `drowse/units-tiny-${options.arch}`,
  sourceRevision: "0".repeat(40),
  convertedManifestSha256: artifactSha("tensor-cache.json"),
  quantization: options.quantization,
  tokenizerSha256: artifactSha("tokenizer.json"),
  chatTemplateSha256: artifactSha("tokenizer_config.json"),
  modelLibrarySha256: artifactSha("model.wasm"),
  runtimeAbi: "drowse-web-runtime-v2",
  hookAbi: "block-sites-v5",
  hiddenSize,
  layerMap: Array.from({ length: layerCount }, (_, layer) => layer),
};
// Short rows: the fixture's byte-level vocabulary spends a token per character, in a
// 128-token context.
const statements = (word) => [`I feel ${word} now.`, `So ${word} today.`, `A ${word} mood.`, `Very ${word}.`];
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
  artifactProducer: { drowseVersion: "0.0.0-fixture", producerVersion: "browser-core-build" },
  manifolds: [
    concept("calm.alert", [["calm", statements("calm")], ["alert", statements("alert")]]),
    concept("deceptive", [["deceptive", statements("sly")]]),
  ],
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
  server: { middlewareMode: true, watch: null },
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
    send(String(error?.stack ?? error), "text/plain", 500);
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
    if (message.type() === "error") process.stderr.write(`browser console: ${message.text()}\n`);
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
      adapter: await navigator.gpu.requestAdapter(),
      artifacts: files,
      signal: new AbortController().signal,
      onDeviceLost() {},
    });
    const positions = ids.map((_, index) => index);
    const result = await runtime.capturePreparedPositions({ inputIds: ids, position: ids.length - 1 }, positions);
    const blockSites = runtime.blockSitesCapabilities();
    await runtime.unload();
    return {
      layerCount: result.layerCount, hiddenSize: result.hiddenSize, positionCount: result.positionCount,
      values: Array.from(result.values),
      profile: blockSites?.profile.id ?? null,
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

function concept(name, nodes) {
  return {
    namespace: "local",
    name,
    description: `fixture concept ${name}`,
    fitMode: "pca",
    hyperparams: { max_dim: 1, var_threshold: 0.7 },
    source: { uri: "local", repository: null, revision: null },
    tags: ["fixture"],
    nodes: nodes.map(([label, rows]) => ({ label, role: null, kind: null, statements: rows })),
  };
}

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
