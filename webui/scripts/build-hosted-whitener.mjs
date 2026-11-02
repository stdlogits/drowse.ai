#!/usr/bin/env node

import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { createServer as createHttpServer } from "node:http";
import {
  access,
  mkdir,
  mkdtemp,
  readFile,
  rename,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import { dirname, extname, join, resolve } from "node:path";
import { createServer as createViteServer } from "vite";
import { readCoreManifoldSource } from "./hosted-core-manifold-source.mjs";
import { nativeFitModel } from "./hosted-native-models.mjs";
import { fitNativeNeutralWhitener, neutralWhitenerSidecar } from "./neutral-whitener-node.mjs";
import { lockedModelExecutionProfiles } from "./hosted-model-profiles.mjs";
import {
  grantReleaseToolStorageQuota,
  launchReleaseToolContext,
  releaseToolGpuArgs,
} from "./release-tool-storage-quota.mjs";
import { readRuntimeLock } from "./runtime-lock-document.mjs";
import { verifyArtifactFile } from "./verify-artifact-file.mjs";
import { validatePublishDirectory } from "./publish-hosted-model.mjs";

const options = parseArguments(process.argv.slice(2));
if (options.help) {
  process.stdout.write(
    "Usage: npm run build:hosted:whitener -- MODEL_ID MODEL_DIR MODEL.wasm WEBLLM.js OUTPUT_DIR\n" +
    "       --manifold-source DIR [--manifold-source DIR]... [--neutral-statistics DIR]\n" +
    "       [--browser-channel chrome] [--timeout-ms 1800000] [--runtime-lock PATH] [--manual-browser]\n" +
    "  --manifold-source  one core concept source each (browser-runtime/release-inputs/core-manifolds/*)\n" +
    "  --neutral-statistics  the core/ directory build_hosted_packs.py wrote for this runtime (required for\n" +
    "                        Gemma 3 and Qwen 3; the whitener is fitted from its native neutral rows)\n",
  );
  process.exit(0);
}

const webuiRoot = resolve(import.meta.dirname, "..");
const repositoryRoot = resolve(webuiRoot, "..");
const runtimeLockPath = options.runtimeLock
  ? resolve(options.runtimeLock)
  : resolve(repositoryRoot, "browser-runtime/runtime-lock.json");
const { value: runtimeLock } = await readRuntimeLock(runtimeLockPath);
const runtimeLockSource = options.runtimeLock ? "override" : "repository";
const sourceRevision = execFileSync("git", ["rev-parse", "HEAD"], {
  cwd: repositoryRoot,
  encoding: "utf8",
}).trim();
if (!/^[0-9a-f]{40}$/.test(sourceRevision)) {
  throw new Error("core-pack authoring requires an exact Drowse source revision");
}
const drowseVersion = /__version__\s*=\s*"([^"]+)"/.exec(
  await readFile(resolve(repositoryRoot, "drowse/drowse/__init__.py"), "utf8"),
)?.[1];
if (!drowseVersion) throw new Error("core-pack authoring could not resolve the Drowse version");
const artifactProvenance = {
  schemaVersion: 1,
  drowseVersion,
  sourceRevision,
  runtimeAbi: runtimeLock.runtimeAbi,
  hookAbi: runtimeLock.hookAbi,
  channel: "candidate",
  verified: false,
};
const artifactProducer = {
  drowseVersion,
  producerVersion: JSON.stringify(artifactProvenance),
};
const lock = runtimeLock.models.find((entry) => entry.id === options.modelId);
if (!lock) throw new Error(`unknown runtime-lock model ${options.modelId}`);

const modelDirectory = resolve(options.modelDirectory);
const modelLibraryPath = resolve(options.modelLibrary);
const webLlmPath = resolve(options.webLlm);
const outputDirectory = resolve(options.outputDirectory);
await requireFile(modelLibraryPath, "model library");
await requireFile(webLlmPath, "WebLLM bundle");
await requireMissing(outputDirectory, "output directory");

const artifactsManifest = await validatePublishDirectory(modelDirectory, lock, runtimeLock);
const executionProfiles = lockedModelExecutionProfiles(lock, artifactsManifest);
if (
  artifactsManifest.runtimeAbi !== runtimeLock.runtimeAbi ||
  artifactsManifest.hookAbi !== runtimeLock.hookAbi ||
  artifactsManifest.quantization !== lock.quantization ||
  artifactsManifest.hiddenSize !== lock.hiddenSize ||
  JSON.stringify(artifactsManifest.layerMap) !== JSON.stringify(lock.layerMap)
) {
  throw new Error("hosted model artifacts do not match runtime-lock identity");
}
const artifacts = artifactsManifest.files.map((file) => ({ ...file }));
for (const artifact of artifacts) {
  const path = artifact.role === "model_library"
    ? modelLibraryPath
    : resolve(modelDirectory, artifact.path);
  await verifyArtifactFile(path, artifact, `model ${artifact.path}`);
}

const [baselinePrompts, neutralStatements] = await Promise.all([
  readJson(resolve(repositoryRoot, "drowse/drowse/data/baseline_prompts.json")),
  readJson(resolve(repositoryRoot, "drowse/drowse/data/neutral_statements.json")),
]);
if (
  !Array.isArray(baselinePrompts) || baselinePrompts.length === 0 ||
  !Array.isArray(neutralStatements) || neutralStatements.length === 0 ||
  neutralStatements.length % baselinePrompts.length !== 0 ||
  [...baselinePrompts, ...neutralStatements].some((value) => typeof value !== "string")
) {
  throw new Error("canonical neutral corpus is invalid");
}
const rows = neutralStatements.map((response, index) => ({
  system: "Answer in one short paragraph.",
  messages: [
    { role: "user", content: baselinePrompts[index % baselinePrompts.length] },
    { role: "assistant", content: response },
  ],
}));
const runtimeIdentity = {
  sourceModel: lock.sourceRepository,
  sourceRevision: lock.sourceRevision,
  convertedManifestSha256: lock.manifestSha256,
  quantization: lock.quantization,
  tokenizerSha256: lock.tokenizerSha256,
  chatTemplateSha256: lock.chatTemplateSha256,
  modelLibrarySha256: lock.librarySha256,
  runtimeAbi: runtimeLock.runtimeAbi,
  hookAbi: runtimeLock.hookAbi,
  hiddenSize: lock.hiddenSize,
  layerMap: [...lock.layerMap],
};
const manifolds = [];
for (const source of options.manifoldSources) {
  manifolds.push(await readCoreManifoldSource(resolve(source), repositoryRoot));
}
if (new Set(manifolds.map((manifold) => `${manifold.namespace}/${manifold.name}`)).size !== manifolds.length) {
  throw new Error("the core concept sources repeat a manifold identity");
}
// A model drowse fits natively carries its neutral statistics, and its whitener is fitted from
// the same native neutral rows the units pack's write norms were measured with.
if (nativeFitModel(lock) !== (options.neutralStatistics !== null)) {
  throw new Error(nativeFitModel(lock)
    ? `${options.modelId} is fitted natively: --neutral-statistics (build_hosted_packs.py core/) is required`
    : `${options.modelId} has no native neutral pass: --neutral-statistics does not apply`);
}
const neutralStatistics = options.neutralStatistics === null
  ? null
  : await readNeutralStatistics(resolve(options.neutralStatistics));
const captureRequest = {
  type: "run",
  modelId: options.modelId,
  modelType: lock.modelType ?? "chat",
  artifacts,
  contexts: lock.contextProfiles,
  baselinePrompts,
  rows,
  nativeNeutralRows: neutralStatistics !== null,
  runtimeIdentity,
  quantization: lock.quantization,
  requiredFeatures: lock.quantization === "q4f16_1" ? ["shader-f16"] : [],
  ...executionProfiles,
  manifolds,
  artifactProducer,
};

const tensorBytesByContext = new Map();
const manifoldBytesByContext = new Map();
let completeManualRun;
const manualRun = options.manualBrowser
  ? new Promise((resolvePromise) => { completeManualRun = resolvePromise; })
  : null;
const vite = await createViteServer({
  root: webuiRoot,
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  publicDir: resolve(webuiRoot, "public-hosted"),
  server: { middlewareMode: true, watch: null },
});
// The whitener the native rows give, refitted here with the same kernel: the browser's must be
// it, tensors and ridge, in every context.
const nativeRefit = neutralStatistics === null
  ? null
  : await fitNativeNeutralWhitener(vite, neutralStatistics.rowsBytes, lock);
const server = createHttpServer(async (request, response) => {
  for (const [name, value] of Object.entries({
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Resource-Policy": "same-origin",
  })) response.setHeader(name, value);
  try {
    const url = new URL(request.url ?? "/", "http://127.0.0.1");
    if (request.method === "GET" && url.pathname === "/") {
      return send(response, pageHtml(), "text/html; charset=utf-8");
    }
    if (request.method === "GET" && url.pathname === "/neutral-rows" && neutralStatistics !== null) {
      return send(response, neutralStatistics.rowsBytes, "application/octet-stream");
    }
    if (options.manualBrowser && request.method === "GET" && url.pathname === "/request") {
      return send(response, JSON.stringify(captureRequest), "application/json");
    }
    if (options.manualBrowser && request.method === "POST" && url.pathname === "/result") {
      const message = JSON.parse((await readBoundedBody(request, 1024 * 1024)).toString("utf8"));
      send(response, "ok", "text/plain");
      completeManualRun(message);
      return;
    }
    if (request.method === "GET" && url.pathname === "/worker.js") {
      const transformed = await vite.transformRequest("/scripts/hosted-whitener.worker.ts");
      if (!transformed) return send(response, "worker transform failed", "text/plain", 500);
      return send(response, transformed.code, "text/javascript; charset=utf-8");
    }
    if (request.method === "GET" && url.pathname === "/webllm.js") {
      return send(response, await readFile(webLlmPath), "text/javascript; charset=utf-8");
    }
    if (request.method === "GET" && url.pathname === "/fitting.js") {
      return send(
        response,
        await readFile(resolve(webuiRoot, "public-hosted/wasm/drowse_fitting_wasm.js")),
        "text/javascript; charset=utf-8",
      );
    }
    if (request.method === "GET" && url.pathname === "/fitting.wasm") {
      return send(
        response,
        await readFile(resolve(webuiRoot, "public-hosted/wasm/drowse_fitting_wasm_bg.wasm")),
        "application/wasm",
      );
    }
    if (request.method === "GET" && url.pathname === "/model") {
      const requested = url.searchParams.get("path");
      const artifact = artifacts.find((candidate) => candidate.path === requested);
      if (!artifact) return send(response, "not found", "text/plain", 404);
      const path = artifact.role === "model_library"
        ? modelLibraryPath
        : resolve(modelDirectory, artifact.path);
      return send(response, await readFile(path), contentType(path));
    }
    if (request.method === "POST" && url.pathname === "/output") {
      const context = Number(url.searchParams.get("context"));
      if (!lock.contextProfiles.includes(context) || tensorBytesByContext.has(context)) {
        return send(response, "invalid or repeated context", "text/plain", 400);
      }
      tensorBytesByContext.set(context, await readBoundedBody(request, 512 * 1024 * 1024));
      return send(response, "ok", "text/plain");
    }
    if (request.method === "POST" && url.pathname === "/manifold-output") {
      const context = Number(url.searchParams.get("context"));
      const index = Number(url.searchParams.get("manifold"));
      const key = `${context}:${index}`;
      if (
        !lock.contextProfiles.includes(context) || !Number.isSafeInteger(index) || index < 0 ||
        index >= manifolds.length || manifoldBytesByContext.has(key)
      ) {
        return send(response, "invalid or repeated context or manifold", "text/plain", 400);
      }
      manifoldBytesByContext.set(key, await readBoundedBody(request, 2 * 1024 * 1024 * 1024));
      return send(response, "ok", "text/plain");
    }
    vite.middlewares(request, response);
  } catch (error) {
    process.stderr.write(
      `whitener server error: ${error instanceof Error ? error.stack ?? error.message : String(error)}\n`,
    );
    send(
      response,
      "Internal server error",
      "text/plain",
      500,
    );
  }
});

await new Promise((resolvePromise, reject) => {
  server.once("error", reject);
  server.listen(0, "127.0.0.1", resolvePromise);
});
const address = server.address();
if (!address || typeof address === "string") throw new Error("failed to bind whitener server");
const origin = `http://127.0.0.1:${address.port}`;

const browser = options.manualBrowser ? null : await launchReleaseToolContext({
  headless: true,
  channel: options.browserChannel,
  args: [
    ...releaseToolGpuArgs(),
    "--unlimited-storage",
  ],
});
let result;
try {
  if (options.manualBrowser) {
    process.stdout.write(`Open browser core-pack builder: ${origin}\n`);
    const message = await manualRun;
    if (message.type !== "done") throw new Error(message.message ?? "browser core-pack build failed");
    result = message.result;
  } else {
  const page = browser.page;
  page.setDefaultTimeout(options.timeoutMs);
  page.on("console", (message) => {
    process.stderr.write(`browser ${message.type()}: ${message.text()}\n`);
  });
  page.on("requestfailed", (request) => {
    process.stderr.write(`browser request failed: ${request.url()} ${request.failure()?.errorText ?? ""}\n`);
  });
  page.on("response", (response) => {
    if (response.url().includes("hosted-whitener.worker") || response.url().endsWith("/worker.js")) {
      process.stderr.write(
        `browser worker response: ${response.status()} ${response.headers()["content-type"] ?? "unknown"}\n`,
      );
    }
  });
  await page.goto(origin, { waitUntil: "load" });
  await grantReleaseToolStorageQuota(page, origin, artifacts);
  result = await page.evaluate(async ({ timeoutMs, payload }) => {
    const worker = new Worker("/worker.js", { type: "module" });
    return await new Promise((resolvePromise, reject) => {
      const timeout = setTimeout(() => {
        worker.terminate();
        reject(new Error(`neutral capture timed out after ${timeoutMs} ms`));
      }, timeoutMs);
      worker.onerror = (event) => {
        clearTimeout(timeout);
        worker.terminate();
        reject(new Error(
          [
            event.message || "neutral capture worker failed",
            event.filename,
            event.lineno ? `line ${event.lineno}:${event.colno}` : "",
          ].filter(Boolean).join(" at "),
        ));
      };
      worker.onmessageerror = () => {
        clearTimeout(timeout);
        worker.terminate();
        reject(new Error("neutral capture worker returned an unreadable message"));
      };
      worker.onmessage = (event) => {
        const message = event.data;
        if (message.type === "progress" || message.type === "manifold-progress") {
          console.info(message.type === "progress"
            ? `neutral capture ${message.completed}/${message.total}`
            : `core manifold ${message.stage} ${message.completed}/${message.total}`);
          return;
        }
        clearTimeout(timeout);
        worker.terminate();
        if (message.type === "error") reject(new Error(message.message));
        else resolvePromise(message.result);
      };
      worker.postMessage(payload);
    });
  }, {
    timeoutMs: options.timeoutMs,
    payload: captureRequest,
  });
  }
} finally {
  await browser?.close();
  await vite.close();
  await new Promise((resolvePromise) => server.close(resolvePromise));
}

if (
  result.runtimeIdentitySha256 !== runtimeIdentityDigest(runtimeIdentity) ||
  !Array.isArray(result.contexts) || result.contexts.length !== lock.contextProfiles.length
) {
  throw new Error("browser whitener metadata does not match the runtime lock");
}

const sidecarSha256ByContext = {};
const stage = await mkdtemp(join(dirname(outputDirectory), ".drowse-whitener-"));
try {
  for (const contextTokens of lock.contextProfiles) {
    const context = result.contexts.find((candidate) => candidate.contextTokens === contextTokens);
    if (!context) throw new Error(`browser whitener omitted context ${contextTokens}`);
    const tensorBytes = tensorBytesByContext.get(contextTokens);
    const manifoldBytes = manifolds.map((_, index) => manifoldBytesByContext.get(`${contextTokens}:${index}`));
    if (!tensorBytes || sha256(tensorBytes) !== context.tensorsSha256 ||
        !Array.isArray(context.manifoldSha256) || context.manifoldSha256.length !== manifolds.length ||
        manifoldBytes.some((bytes, index) => !bytes || sha256(bytes) !== context.manifoldSha256[index]) ||
        context.bindingSha256 !== sha256(canonicalJson({
          runtimeIdentitySha256: result.runtimeIdentitySha256, contextTokens,
        })) ||
        JSON.stringify(Object.keys(context.ridgePerLayer).map(Number)) !== JSON.stringify(lock.layerMap)) {
      throw new Error(`browser core pack identity or payload differs for context ${contextTokens}`);
    }
    if (nativeRefit !== null && (
      sha256(tensorBytes) !== sha256(nativeRefit.tensorBytes) ||
      JSON.stringify(context.ridgePerLayer) !== JSON.stringify(nativeRefit.ridgePerLayer)
    )) throw new Error(`the ${contextTokens}-token whitener is not the one the native neutral rows give`);
    const directory = resolve(stage, String(contextTokens));
    await mkdir(directory, { recursive: true });
    await writeFile(resolve(directory, "neutral-whitener.safetensors"), tensorBytes);
    const sidecarBytes = neutralWhitenerSidecar({
      runtimeIdentitySha256: result.runtimeIdentitySha256,
      contextBindingSha256: context.bindingSha256,
      modelLock: lock,
      tensorsSha256: context.tensorsSha256,
      ridgePerLayer: context.ridgePerLayer,
    });
    sidecarSha256ByContext[String(contextTokens)] = sha256(sidecarBytes);
    await writeFile(resolve(directory, "neutral-whitener.json"), sidecarBytes);
    const manifoldDirectory = resolve(stage, "manifolds", String(contextTokens));
    await mkdir(manifoldDirectory, { recursive: true });
    for (const [index, manifold] of manifolds.entries()) {
      await writeFile(resolve(manifoldDirectory, `${manifold.namespace}-${manifold.name}.drowse`), manifoldBytes[index]);
    }
  }
  if (neutralStatistics !== null) {
    if (neutralStatistics.record.runtime_identity_sha256 !== result.runtimeIdentitySha256) {
      throw new Error("the neutral statistics were built for another runtime identity");
    }
    // The shipped record names the whitener the rows gave in each context, which the units
    // pack's binding reaches through it (browserInstrumentPacks.ts; validate-hosted-core-pack.mjs).
    const shipped = {
      ...neutralStatistics.record,
      whitener: {
        rows_sha256: neutralStatistics.record.neutral.rows_sha256,
        tensors_sha256: Object.fromEntries(result.contexts.map((context) =>
          [String(context.contextTokens), context.tensorsSha256])),
        sidecars_sha256: sidecarSha256ByContext,
      },
    };
    if (new Set(Object.values(shipped.whitener.tensors_sha256)).size !== 1) {
      throw new Error("the native neutral rows gave different whiteners in different contexts");
    }
    await writeFile(resolve(stage, "neutral-statistics.json"), `${JSON.stringify(shipped, null, 2)}\n`);
    await writeFile(resolve(stage, neutralStatistics.record.tensor_file), neutralStatistics.tensorBytes);
  }
  await rename(stage, outputDirectory);
} catch (error) {
  await rm(stage, { recursive: true, force: true });
  throw error;
}

process.stdout.write(`${JSON.stringify({
  modelId: options.modelId,
  runtimeLockSource,
  outputDirectory,
  neutralRows: neutralStatistics === null
    ? { source: "browser capture", rows: rows.length }
    : { source: "native", rows: neutralStatistics.record.neutral_rows.rows, sha256: neutralStatistics.record.neutral.rows_sha256 },
  hiddenSize: lock.hiddenSize,
  layers: lock.layerMap.length,
  runtimeIdentitySha256: result.runtimeIdentitySha256,
  contexts: result.contexts,
  manifolds: manifolds.map((manifold, index) => ({
    identity: `${manifold.namespace}/${manifold.name}`,
    sha256ByContext: Object.fromEntries(result.contexts.map(context => [context.contextTokens, context.manifoldSha256[index]])),
  })),
  neutralStatistics: neutralStatistics === null ? null : neutralStatistics.record.tensor_sha256,
}, null, 2)}\n`);

function parseArguments(args) {
  if (args.includes("--help")) return { help: true };
  const positional = [];
  const result = {
    browserChannel: "chrome",
    manifoldSources: [],
    neutralStatistics: null,
    runtimeLock: null,
    manualBrowser: false,
    timeoutMs: 1_800_000,
  };
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index];
    if (argument === "--manual-browser") {
      result.manualBrowser = true;
      continue;
    }
    if (!argument.startsWith("--")) {
      positional.push(argument);
      continue;
    }
    if (
      argument !== "--browser-channel" && argument !== "--manifold-source" &&
      argument !== "--timeout-ms" && argument !== "--runtime-lock" &&
      argument !== "--neutral-statistics"
    ) {
      throw new Error(`unexpected argument ${argument}`);
    }
    const value = args[index + 1];
    if (!value || value.startsWith("--")) throw new Error(`missing value for ${argument}`);
    if (argument === "--browser-channel") result.browserChannel = value;
    else if (argument === "--manifold-source") result.manifoldSources.push(value);
    else if (argument === "--neutral-statistics") result.neutralStatistics = value;
    else if (argument === "--runtime-lock") result.runtimeLock = value;
    else result.timeoutMs = Number(value);
    index += 1;
  }
  if (positional.length !== 5) {
    throw new Error("expected MODEL_ID MODEL_DIR MODEL.wasm WEBLLM.js OUTPUT_DIR");
  }
  if (result.manifoldSources.length === 0) {
    throw new Error("--manifold-source is required for a releasable core pack");
  }
  if (!Number.isSafeInteger(result.timeoutMs) || result.timeoutMs < 1_000) {
    throw new Error("timeout must be an integer of at least 1000 ms");
  }
  [result.modelId, result.modelDirectory, result.modelLibrary, result.webLlm, result.outputDirectory] = positional;
  return result;
}

async function readJson(path) {
  return JSON.parse(await readFile(path, "utf8"));
}

/** The core statistics `build_hosted_packs.py` wrote (`core/neutral-statistics.*`), checked
 *  against their own record; the record's runtime identity is checked against the build's. */
async function readNeutralStatistics(directory) {
  const recordBytes = await readFile(resolve(directory, "neutral-statistics.json"));
  const record = JSON.parse(recordBytes.toString("utf8"));
  if (
    record?.format !== "drowse-neutral-statistics-v1" || record.tensor_file !== "neutral-statistics.safetensors" ||
    !/^[0-9a-f]{64}$/u.test(record.tensor_sha256 ?? "") || !/^[0-9a-f]{64}$/u.test(record.runtime_identity_sha256 ?? "")
  ) {
    throw new Error("neutral-statistics.json is not a neutral statistics record");
  }
  const tensorBytes = await readFile(resolve(directory, record.tensor_file));
  if (sha256(tensorBytes) !== record.tensor_sha256) {
    throw new Error("the neutral statistics tensor does not match its record");
  }
  if (
    record.neutral_rows?.file !== "neutral-rows.safetensors" || "whitener" in record ||
    !/^[0-9a-f]{64}$/u.test(record.neutral?.rows_sha256 ?? "")
  ) throw new Error("neutral-statistics.json names no neutral rows");
  const rowsBytes = await readFile(resolve(directory, record.neutral_rows.file));
  if (sha256(rowsBytes) !== record.neutral.rows_sha256) {
    throw new Error("the neutral rows do not match their record");
  }
  return { record, tensorBytes, rowsBytes };
}

async function requireFile(path, label) {
  const info = await stat(path);
  if (!info.isFile()) throw new Error(`${label} is not a regular file: ${path}`);
}

async function requireMissing(path, label) {
  try {
    await access(path);
  } catch (error) {
    if (error?.code === "ENOENT") return;
    throw error;
  }
  throw new Error(`${label} already exists: ${path}`);
}

async function readBoundedBody(request, limit) {
  const chunks = [];
  let bytes = 0;
  for await (const chunk of request) {
    bytes += chunk.byteLength;
    if (bytes > limit) throw new Error("browser output exceeded the local release-tool limit");
    chunks.push(chunk);
  }
  return Buffer.concat(chunks, bytes);
}

function runtimeIdentityDigest(identity) {
  return sha256(Buffer.from(canonicalJson(identity)));
}

function canonicalJson(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  return `{${Object.entries(value).sort(([left], [right]) => left.localeCompare(right))
    .map(([key, entry]) => `${JSON.stringify(key)}:${canonicalJson(entry)}`).join(",")}}`;
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function pageHtml() {
  return `<!doctype html><meta charset=utf-8><link rel=icon href=data:,>
<title>Drowse core-pack builder</title><h1>Drowse core-pack builder</h1>
<p>Capture the canonical neutral corpus and fit the required geometry pack on this browser's GPU.</p>
<button id="run">Build core pack</button><pre id="status" role="status">Ready</pre>
<script type="module">
const button = document.querySelector('#run'), status = document.querySelector('#status');
button.onclick = async () => {
  button.disabled = true;
  let worker;
  async function finish(message) {
    worker?.terminate();
    status.textContent = message.type === 'done' ? 'Core pack built successfully' : message.message;
    await fetch('/result', {method: 'POST', body: JSON.stringify(message)});
  }
  try {
    const payload = await (await fetch('/request')).json();
    worker = new Worker('/worker.js', {type: 'module'});
    worker.onerror = event => finish({type: 'error', message: event.message});
    worker.onmessageerror = () => finish({type: 'error', message: 'Unreadable worker result'});
    worker.onmessage = event => {
      const message = event.data;
      if (message.type === 'progress' || message.type === 'manifold-progress') {
        status.textContent = (message.stage ?? 'Neutral capture') + ' ' + message.completed + '/' + message.total;
      } else if (message.type === 'done' || message.type === 'error') finish(message);
      else status.textContent = 'Initializing worker: ' + JSON.stringify(message);
    };
    status.textContent = 'Loading verified model files';
    worker.postMessage(payload);
  } catch (error) { await finish({type: 'error', message: error.message}); }
};
</script>`;
}

function contentType(path) {
  if (extname(path) === ".json") return "application/json";
  if (extname(path) === ".wasm") return "application/wasm";
  return "application/octet-stream";
}

function send(response, body, type, status = 200) {
  response.writeHead(status, { "Content-Type": type });
  response.end(body);
}
