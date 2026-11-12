#!/usr/bin/env node

import { createHash } from "node:crypto";
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
import {
  digestCanonical,
  runtimeLockIdentitySha256,
} from "./check-runtime-lock.mjs";
import { lockedModelExecutionProfiles } from "./hosted-model-profiles.mjs";
import { validatePublishDirectory } from "./publish-hosted-model.mjs";
import {
  grantReleaseToolStorageQuota,
  launchReleaseToolContext,
  releaseToolGpuArgs,
} from "./release-tool-storage-quota.mjs";
import { readRuntimeLock } from "./runtime-lock-document.mjs";
import { validateBrowserFullRuntimeModelClosure } from "./browser-full-runtime-contract.mjs";
import {
  CAPTURE_SCHEMA_VERSION,
  captureUnitsMember,
  unitsCaptureFailures,
} from "./hosted-residual-capture-units.mjs";

const options = parseArguments(process.argv.slice(2));
if (options.help) {
  process.stdout.write(
    "Usage: npm run capture:hosted:residuals -- MODEL_ID MODEL_DIR MODEL.wasm WEBLLM.js INPUT.json OUTPUT_DIR\n" +
    "       [--browser-channel chrome] [--timeout-ms 600000] [--shader-dump DIR]\n" +
    "       [--runtime-lock PATH]\n" +
    "INPUT.json contains inputIds, positions, greedyPrompt, and greedyMaxTokens.\n",
  );
  process.exit(0);
}
const webuiRoot = resolve(import.meta.dirname, "..");
const repositoryRoot = resolve(webuiRoot, "..");
const repositoryRuntimeLockPath = resolve(
  repositoryRoot,
  "browser-runtime/runtime-lock.json",
);
const runtimeLockPath = options.runtimeLock
  ? resolve(options.runtimeLock)
  : repositoryRuntimeLockPath;
const { bytes: runtimeLockBytes, value: runtimeLock } = await readRuntimeLock(
  runtimeLockPath,
);
const runtimeLockProvenance = {
  source: options.runtimeLock ? "override" : "repository",
  sha256: sha256(runtimeLockBytes),
  identity_sha256: runtimeLockIdentitySha256(runtimeLock),
};
const lock = runtimeLock.models.find((entry) => entry.id === options.modelId);
if (!lock) throw new Error(`unknown runtime-lock model ${options.modelId}`);
if (
  lock.manifestSha256 === null || lock.librarySha256 === null ||
  lock.tokenizerSha256 === null || lock.chatTemplateSha256 === null ||
  lock.hiddenSize === null || lock.layerMap === null
) throw new Error(`runtime-lock model ${options.modelId} has unresolved capture identity`);
const modelDirectory = resolve(options.modelDirectory);
const modelLibraryPath = resolve(options.modelLibrary);
const webLlmPath = resolve(options.webLlm);
const outputDirectory = resolve(options.outputDirectory);
const shaderDumpDirectory = options.shaderDump ? resolve(options.shaderDump) : null;
await Promise.all([
  requireFile(modelLibraryPath, "model library"),
  requireFile(webLlmPath, "WebLLM bundle"),
  requireMissing(outputDirectory, "output directory"),
  ...(shaderDumpDirectory ? [requireMissing(shaderDumpDirectory, "shader dump directory")] : []),
]);
const input = JSON.parse(await readFile(resolve(options.inputPath), "utf8"));
if (
  !Array.isArray(input.inputIds) || input.inputIds.length === 0 ||
  input.inputIds.some((value) => !Number.isSafeInteger(value) || value < 0) ||
  !Array.isArray(input.positions) || input.positions.length === 0 ||
  input.positions.some((value, index) =>
    !Number.isSafeInteger(value) || value < 0 || value >= input.inputIds.length ||
    (index > 0 && value <= input.positions[index - 1])
  ) || typeof input.greedyPrompt !== "string" || input.greedyPrompt.length === 0 ||
  !Number.isSafeInteger(input.greedyMaxTokens) || input.greedyMaxTokens < 1 ||
  input.greedyMaxTokens > 32 ||
  (input.namedRoleCheck !== undefined && !validNamedRoleCheck(input.namedRoleCheck))
) throw new Error("residual capture input IDs or positions are invalid");
const artifactsManifest = JSON.parse(
  await readFile(resolve(modelDirectory, "hosted-artifacts.json"), "utf8"),
);
validateBrowserFullRuntimeModelClosure({
  lock,
  runtimeLock,
  manifest: artifactsManifest,
  files: artifactsManifest.files,
  contextTokens: Math.min(...lock.contextProfiles),
});
await validatePublishDirectory(modelDirectory, lock, runtimeLock, {
  allowStaleToolchain: true,
});
const executionProfiles = lockedModelExecutionProfiles(lock, artifactsManifest);
if (
  artifactsManifest.hookAbi !== runtimeLock.hookAbi ||
  artifactsManifest.hiddenSize !== lock.hiddenSize ||
  JSON.stringify(artifactsManifest.layerMap) !== JSON.stringify(lock.layerMap)
) throw new Error("hosted model artifacts do not match runtime-lock identity");
const artifacts = artifactsManifest.files.map((file) => ({ ...file }));
for (const artifact of artifacts) {
  const path = artifact.role === "model_library"
    ? modelLibraryPath
    : resolve(modelDirectory, artifact.path);
  await requireFile(path, `model ${artifact.path}`);
  const bytes = await readFile(path);
  if (bytes.byteLength !== artifact.bytes || sha256(bytes) !== artifact.sha256) {
    throw new Error(`model ${artifact.path} differs from hosted-artifacts.json`);
  }
}
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
const expectedRuntimeIdentitySha256 = digestCanonical(runtimeIdentity);
const captureBodies = new Map();
const shaderSources = [];
const vite = await createViteServer({
  root: webuiRoot,
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  publicDir: resolve(webuiRoot, "public-hosted"),
  server: { middlewareMode: true, watch: null },
});
const server = createHttpServer(async (request, response) => {
  for (const [name, value] of Object.entries({
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Resource-Policy": "same-origin",
  })) response.setHeader(name, value);
  try {
    const url = new URL(request.url ?? "/", "http://127.0.0.1");
    if (request.method === "GET" && url.pathname === "/") {
      return send(response, "<!doctype html><meta charset=utf-8><link rel=icon href=data:,>", "text/html");
    }
    if (request.method === "GET" && url.pathname === "/worker.js") {
      const transformed = await vite.transformRequest("/scripts/hosted-residual-capture.worker.ts");
      return send(response, transformed?.code ?? "worker transform failed", "text/javascript", transformed ? 200 : 500);
    }
    if (request.method === "GET" && url.pathname === "/webllm.js") {
      return send(response, await readFile(webLlmPath), "text/javascript");
    }
    if (request.method === "GET" && url.pathname === "/model") {
      const requested = url.searchParams.get("path");
      const artifact = artifacts.find((candidate) => candidate.path === requested);
      if (!artifact) return send(response, "not found", "text/plain", 404);
      const path = artifact.role === "model_library" ? modelLibraryPath : resolve(modelDirectory, artifact.path);
      return send(response, await readFile(path), contentType(path));
    }
    if (request.method === "POST" && url.pathname === "/capture") {
      const kind = url.searchParams.get("kind");
      if (kind !== "unsteered" && kind !== "control-probes" && kind !== "steered-probes" && kind !== "sites") {
        return send(response, "unknown capture kind", "text/plain", 400);
      }
      if (captureBodies.has(kind)) {
        return send(response, `${kind} capture already received`, "text/plain", 409);
      }
      captureBodies.set(kind, await readBoundedBody(request, 512 * 1024 * 1024));
      return send(response, "ok", "text/plain");
    }
    if (request.method === "POST" && url.pathname === "/shader" && shaderDumpDirectory) {
      const entryPoint = url.searchParams.get("entry") ?? "unknown";
      shaderSources.push({ entryPoint, code: await readBoundedBody(request, 32 * 1024 * 1024) });
      return send(response, "ok", "text/plain");
    }
    vite.middlewares(request, response);
  } catch (error) {
    console.error(error);
    send(response, "Internal server error", "text/plain", 500);
  }
});
await new Promise((resolvePromise, reject) => {
  server.once("error", reject);
  server.listen(0, "127.0.0.1", resolvePromise);
});
const address = server.address();
if (!address || typeof address === "string") throw new Error("failed to bind residual capture server");
const origin = `http://127.0.0.1:${address.port}`;
const browser = await launchReleaseToolContext({
  headless: true,
  channel: options.browserChannel,
  args: releaseToolGpuArgs(),
});
let result;
try {
  const page = browser.page;
  page.setDefaultTimeout(options.timeoutMs);
  page.on("console", (message) => process.stderr.write(`browser ${message.type()}: ${message.text()}\n`));
  await page.goto(origin, { waitUntil: "load" });
  await grantReleaseToolStorageQuota(page, origin, artifacts);
  result = await page.evaluate(async ({ timeoutMs, payload }) => {
    const worker = new Worker("/worker.js", { type: "module" });
    return await new Promise((resolvePromise, reject) => {
      const timeout = setTimeout(() => {
        worker.terminate();
        reject(new Error(`residual capture timed out after ${timeoutMs} ms`));
      }, timeoutMs);
      worker.onerror = (event) => {
        clearTimeout(timeout);
        worker.terminate();
        reject(new Error(event.message || "residual capture worker failed"));
      };
      worker.onmessage = (event) => {
        clearTimeout(timeout);
        worker.terminate();
        if (event.data.type === "error") reject(new Error(event.data.message));
        else resolvePromise(event.data.result);
      };
      worker.postMessage(payload);
    });
  }, {
    timeoutMs: options.timeoutMs,
    payload: {
      type: "run",
      modelId: options.modelId,
      artifacts,
      runtimeIdentity,
      ...executionProfiles,
      inputIds: input.inputIds,
      positions: input.positions,
      greedyPrompt: input.greedyPrompt,
      greedyMaxTokens: input.greedyMaxTokens,
      ...(input.namedRoleCheck === undefined
        ? {}
        : { namedRoleCheck: { row: input.namedRoleCheck.row } }),
      contextTokens: Math.max(...lock.contextProfiles),
      contextProfiles: [...lock.contextProfiles],
      dumpShaders: shaderDumpDirectory !== null,
    },
  });
} finally {
  await browser.close();
  await vite.close();
  await new Promise((resolvePromise) => server.close(resolvePromise));
}
const residualBytes = captureBodies.get("unsteered");
const controlProbeBytes = captureBodies.get("control-probes");
const steeredProbeBytes = captureBodies.get("steered-probes");
const sitesBytes = captureBodies.get("sites");
const envelopeFailures = [
  ...captureEnvelopeFailures({
    result,
    residualBytes,
    controlProbeBytes,
    steeredProbeBytes,
    lock,
    input,
    expectedRuntimeIdentitySha256,
  }),
  ...unitsCaptureFailures({
    structuredHookProfile: executionProfiles.structuredHookProfile,
    units: result.units,
    sitesBytes,
    lock,
    positions: input.positions,
  }),
];
if (envelopeFailures.length > 0) {
  throw new Error(
    `browser residual capture returned an incompatible envelope:\n- ${envelopeFailures.join("\n- ")}`,
  );
}
if (shaderDumpDirectory) {
  await mkdir(shaderDumpDirectory, { recursive: true });
  for (const [index, shader] of shaderSources.entries()) {
    const name = shader.entryPoint.replace(/[^A-Za-z0-9_.-]+/g, "_").slice(0, 120) || "unknown";
    await writeFile(resolve(shaderDumpDirectory, `${String(index).padStart(4, "0")}-${name}.wgsl`), shader.code);
  }
}
const stage = await mkdtemp(join(dirname(outputDirectory), ".drowse-residual-"));
try {
  await mkdir(stage, { recursive: true });
  await writeFile(resolve(stage, "residuals.f32"), residualBytes);
  await writeFile(resolve(stage, "control-probes.f32"), controlProbeBytes);
  await writeFile(resolve(stage, "steered-probes.f32"), steeredProbeBytes);
  if (sitesBytes !== undefined) await writeFile(resolve(stage, "sites.f32"), sitesBytes);
  await writeFile(resolve(stage, "capture.json"), `${JSON.stringify({
    schema_version: CAPTURE_SCHEMA_VERSION,
    produced_at: new Date().toISOString(),
    model_id: options.modelId,
    runtime_identity_sha256: result.runtimeIdentitySha256,
    runtime_lock: runtimeLockProvenance,
    layer_map: lock.layerMap,
    hidden_size: result.hiddenSize,
    input_ids: input.inputIds,
    positions: result.positions,
    residual_sha256: result.residualSha256,
    control_probe_sha256: result.controlProbeSha256,
    steered_probe_sha256: result.steeredProbeSha256,
    steering_program: result.steeringProgram,
    greedy: {
      prompt: result.greedyPrompt,
      input_ids: result.greedyInputIds,
      token_ids: result.greedyTokenIds,
      max_tokens: input.greedyMaxTokens,
    },
    ...(result.namedRolePrepared === undefined
      ? {}
      : {
          named_role_capture: {
            input_ids: result.namedRolePrepared.inputIds,
            position: result.namedRolePrepared.position,
          },
        }),
    ...(result.units === undefined ? {} : { units: captureUnitsMember(result.units) }),
  }, null, 2)}\n`);
  await rename(stage, outputDirectory);
} catch (error) {
  await rm(stage, { recursive: true, force: true });
  throw error;
}
process.stdout.write(`${JSON.stringify({
  modelId: options.modelId,
  outputDirectory,
  layers: result.layerCount,
  positions: result.positions,
  hiddenSize: result.hiddenSize,
  residualSha256: result.residualSha256,
  controlProbeSha256: result.controlProbeSha256,
  steeredProbeSha256: result.steeredProbeSha256,
  greedyTokenIds: result.greedyTokenIds,
  sitesSha256: result.units?.sitesSha256 ?? null,
  staticReads: result.units?.staticReads.length ?? 0,
  runtimeLock: runtimeLockProvenance,
}, null, 2)}\n`);

function parseArguments(args) {
  if (args.includes("--help")) return { help: true };
  const positional = [];
  const result = {
    browserChannel: "chrome",
    timeoutMs: 600_000,
    shaderDump: null,
    runtimeLock: null,
  };
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index];
    if (!argument.startsWith("--")) {
      positional.push(argument);
      continue;
    }
    if (
      argument !== "--browser-channel" && argument !== "--timeout-ms" &&
      argument !== "--shader-dump" && argument !== "--runtime-lock"
    ) {
      throw new Error(`unexpected argument ${argument}`);
    }
    const value = args[++index];
    if (!value || value.startsWith("--")) throw new Error(`missing value for ${argument}`);
    if (argument === "--browser-channel") result.browserChannel = value;
    else if (argument === "--timeout-ms") result.timeoutMs = Number(value);
    else if (argument === "--shader-dump") result.shaderDump = value;
    else result.runtimeLock = value;
  }
  if (positional.length !== 6) {
    throw new Error("expected MODEL_ID MODEL_DIR MODEL.wasm WEBLLM.js INPUT.json OUTPUT_DIR");
  }
  if (!Number.isSafeInteger(result.timeoutMs) || result.timeoutMs < 10_000) {
    throw new Error("timeout must be an integer of at least 10000 ms");
  }
  [
    result.modelId,
    result.modelDirectory,
    result.modelLibrary,
    result.webLlm,
    result.inputPath,
    result.outputDirectory,
  ] = positional;
  return result;
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
    if (bytes > limit) throw new Error("residual capture exceeded the release-tool limit");
    chunks.push(chunk);
  }
  return Buffer.concat(chunks, bytes);
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

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function validNamedRoleCheck(value) {
  return value && typeof value === "object" && value.row &&
    typeof value.row.system === "string" && Array.isArray(value.row.messages) &&
    value.row.messages.length > 0 && value.row.messages.every((message) =>
      message && ["system", "user", "assistant"].includes(message.role) &&
      typeof message.content === "string" &&
      (message.roleName === undefined || typeof message.roleName === "string")
    ) && Array.isArray(value.expectedInputIds) && value.expectedInputIds.length > 0 &&
    value.expectedInputIds.every((tokenId) => Number.isSafeInteger(tokenId) && tokenId >= 0) &&
    Number.isSafeInteger(value.expectedPosition) && value.expectedPosition >= 0 &&
    value.expectedPosition < value.expectedInputIds.length;
}

function validSteeringProgram(result, layerCount, hiddenSize) {
  const program = result.steeringProgram;
  return program && program.format === "drowse-structured-v5" &&
    Number.isSafeInteger(program.enabledLayer) && program.enabledLayer >= 0 &&
    program.enabledLayer < layerCount && Array.isArray(program.probeDirections) &&
    program.probeDirections.length === result.probeCount && result.probeCount >= 1 &&
    program.probeDirections.every((direction) =>
      Array.isArray(direction) && direction.length === hiddenSize && direction.every(Number.isFinite)
    ) &&
    [program.target, program.along, program.kappa].every(Number.isFinite);
}

function captureEnvelopeFailures({
  result,
  residualBytes,
  controlProbeBytes,
  steeredProbeBytes,
  lock,
  input,
  expectedRuntimeIdentitySha256,
}) {
  const failures = [];
  if (residualBytes === undefined) failures.push("unsteered capture body is missing");
  else {
    if (residualBytes.length !== result.bytes) {
      failures.push(
        `unsteered byte count is ${residualBytes.length}, envelope declares ${String(result.bytes)}`,
      );
    }
    const digest = sha256(residualBytes);
    if (digest !== result.residualSha256) {
      failures.push(
        `unsteered SHA-256 is ${digest}, envelope declares ${String(result.residualSha256)}`,
      );
    }
  }
  for (const [label, body, digest] of [
    ["control probe", controlProbeBytes, result.controlProbeSha256],
    ["steered probe", steeredProbeBytes, result.steeredProbeSha256],
  ]) {
    if (body === undefined) {
      failures.push(`${label} capture body is missing`);
      continue;
    }
    const expectedBytes = input.positions.length * lock.layerMap.length * result.probeCount * 4;
    if (body.length !== result.probeBytes || body.length !== expectedBytes) {
      failures.push(
        `${label} byte count is ${body.length}, envelope declares ${String(result.probeBytes)} (expected ${expectedBytes})`,
      );
    }
    const actual = sha256(body);
    if (actual !== digest) failures.push(`${label} SHA-256 is ${actual}, envelope declares ${String(digest)}`);
  }
  if (result.layerCount !== lock.layerMap.length) {
    failures.push(
      `layer count is ${String(result.layerCount)}, expected ${lock.layerMap.length}`,
    );
  }
  if (result.hiddenSize !== lock.hiddenSize) {
    failures.push(`hidden size is ${String(result.hiddenSize)}, expected ${lock.hiddenSize}`);
  }
  if (result.positionCount !== input.positions.length) {
    failures.push(
      `position count is ${String(result.positionCount)}, expected ${input.positions.length}`,
    );
  }
  if (!validSteeringProgram(result, lock.layerMap.length, lock.hiddenSize)) {
    failures.push("the steering program or its probe directions are invalid");
  }
  if (result.runtimeIdentitySha256 !== expectedRuntimeIdentitySha256) {
    failures.push(
      `runtime identity is ${String(result.runtimeIdentitySha256)}, expected ${expectedRuntimeIdentitySha256}`,
    );
  }
  if (result.greedyPrompt !== input.greedyPrompt) {
    failures.push("greedy prompt differs from the requested prompt");
  }
  if (!validTokenIds(result.greedyInputIds, false)) {
    failures.push("greedy input token IDs are invalid or empty");
  }
  if (!validTokenIds(result.greedyTokenIds, true)) {
    failures.push("greedy output token IDs are invalid");
  } else if (result.greedyTokenIds.length > input.greedyMaxTokens) {
    failures.push(
      `greedy output has ${result.greedyTokenIds.length} tokens, maximum is ${input.greedyMaxTokens}`,
    );
  }
  if (input.namedRoleCheck !== undefined) {
    if (!sameNumbers(
      result.namedRolePrepared?.inputIds,
      input.namedRoleCheck.expectedInputIds,
    )) {
      failures.push(
        `named-role token IDs differ (observed ${summarizeTokenIds(result.namedRolePrepared?.inputIds)}, expected ${summarizeTokenIds(input.namedRoleCheck.expectedInputIds)})`,
      );
    }
    if (result.namedRolePrepared?.position !== input.namedRoleCheck.expectedPosition) {
      failures.push(
        `named-role position is ${String(result.namedRolePrepared?.position)}, expected ${input.namedRoleCheck.expectedPosition}`,
      );
    }
  }
  return failures;
}

function summarizeTokenIds(value) {
  if (!Array.isArray(value)) return String(value);
  const shown = value.slice(0, 32).join(",");
  return value.length > 32 ? `[${shown},…] (${value.length} total)` : `[${shown}]`;
}

function sameNumbers(actual, expected) {
  return Array.isArray(actual) && actual.length === expected.length &&
    actual.every((value, index) => value === expected[index]);
}

function validTokenIds(value, allowEmpty) {
  return Array.isArray(value) && (allowEmpty || value.length > 0) &&
    value.every((tokenId) => Number.isSafeInteger(tokenId) && tokenId >= 0);
}
