// The browser runtime lock and the distribution lock: their structure, each model's fields, the
// CSP's connect sources, and the vendored WebLLM archive against its digest and the installed
// package. Run directly (`node scripts/check-runtime-lock.mjs`); the other exports serve the
// release tools.

import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { fileURLToPath, pathToFileURL } from "node:url";
import { resolve } from "node:path";
import { thinkingProfile } from "./hosted-model-profiles.mjs";
import { validateRuntimeLock } from "./runtime-lock-document.mjs";
import { CURRENT_CATALOG_URLS } from "./hosted-catalog-location.mjs";

const repositoryRoot = fileURLToPath(new URL("../../", import.meta.url));
const COMMIT = /^[0-9a-f]{40}$/;
const SHA256 = /^[0-9a-f]{64}$/;

/** Every library-bound field still null in the lock. A shipped lock has none. */
export function browserRuntimeIdentityBlockers(runtime) {
  const models = runtime.models ?? [];
  if (models.length === 0) return ["models are empty"];
  return models.flatMap((model) => [
    "convertedRevision",
    "manifestSha256",
    "librarySha256",
    "tokenizerSha256",
    "chatTemplateSha256",
    "hiddenSize",
    "layerMap",
  ].filter((field) => model[field] == null).map((field) => `models.${model.id}.${field} is unresolved`));
}

async function main() {
  const read = (path) => readFile(new URL(path, import.meta.url), "utf8");
  const lock = JSON.parse(await read("../../browser-runtime/runtime-lock.json"));
  const distribution = JSON.parse(await read("../../browser-runtime/distribution-lock.json"));
  const packageManifest = JSON.parse(await read("../package.json"));
  const packageLock = JSON.parse(await read("../../package-lock.json"));
  const installed = JSON.parse(await read("../../node_modules/@drowse/web-llm/package.json"));

  validateRuntimeLock(lock);
  for (const model of lock.models) {
    const expectedThinking = model.architecture === "qwen3"
      ? { start: "<think>", end: "</think>", startsInThinking: false, startTokenIds: [151667], endTokenIds: [151668] }
      : null;
    if (JSON.stringify(thinkingProfile(model.thinkingProfile)) !== JSON.stringify(expectedThinking)) {
      throw new Error(`${model.id} has an unexpected thinking-token profile`);
    }
    if (model.layerMap?.some((layer, index) => index > 0 && layer <= model.layerMap[index - 1])) {
      throw new Error(`${model.id} layer map is invalid`);
    }
    if (model.contextProfiles.some((profile, index) => index > 0 && profile <= model.contextProfiles[index - 1])) {
      throw new Error(`${model.id} context profiles must be unique and sorted`);
    }
  }
  const unresolved = browserRuntimeIdentityBlockers(lock);
  if (lock.status === "verified" && unresolved.length > 0) {
    throw new Error(`verified browser runtime lock contains unresolved identities:\n- ${unresolved.join("\n- ")}`);
  }
  validateDistribution(distribution);
  if (distribution.status === "verified" && lock.status !== "verified") {
    throw new Error("verified browser distribution requires a verified runtime");
  }
  validateHostedConnectSources(await read("../public-hosted/_headers"), distribution);
  await validateWebLlmPackageArchive(lock.toolchain.webLlmPackage, lock.toolchain.webLlmFork);
  validateWebLlmDependency(packageManifest, packageLock, lock.toolchain.webLlmPackage);
  validateInstalledWebLlmDependency(packageManifest, packageLock, installed);
  console.log(`Browser runtime lock: ${lock.status}`);
  console.log(`Browser distribution lock: ${distribution.status}`);
}

function validateDistribution(value) {
  exactKeys(value, [
    "$schema",
    "schemaVersion",
    "status",
    "catalogUrl",
    "signatureUrl",
    "minimumAcceptedSequence",
    "publicKeys",
    "allowedCatalogRedirectOrigins",
    "allowedArtifactRedirectOrigins",
  ], "distribution lock");
  if (value.schemaVersion !== 1) throw new Error("unsupported distribution lock schema");
  if (!["feasibility-required", "verified"].includes(value.status)) {
    throw new Error("invalid browser distribution lock status");
  }
  // Each runtime generation reads its catalog at its own path (hosted-catalog-location.mjs).
  if (value.catalogUrl !== CURRENT_CATALOG_URLS.catalogUrl) throw new Error("unexpected browser catalog URL");
  if (value.signatureUrl !== CURRENT_CATALOG_URLS.signatureUrl) {
    throw new Error("unexpected browser catalog signature URL");
  }
  if (!Number.isSafeInteger(value.minimumAcceptedSequence) || value.minimumAcceptedSequence < 1) {
    throw new Error("invalid minimum accepted catalog sequence");
  }
  if (!Array.isArray(value.publicKeys) || value.publicKeys.length > 2) {
    throw new Error("browser catalog key ring must contain at most two keys");
  }
  const keyIds = new Set();
  for (const key of value.publicKeys) {
    exactKeys(key, ["keyId", "ed25519PublicKeyBase64"], "catalog public key");
    if (typeof key.keyId !== "string" || !key.keyId || keyIds.has(key.keyId)) {
      throw new Error("catalog public key IDs must be unique and non-empty");
    }
    keyIds.add(key.keyId);
    const decoded = Buffer.from(String(key.ed25519PublicKeyBase64), "base64");
    if (decoded.length !== 32 || decoded.toString("base64") !== key.ed25519PublicKeyBase64) {
      throw new Error(`catalog public key ${key.keyId} is invalid`);
    }
  }
  validateOrigins(value.allowedCatalogRedirectOrigins, "catalog redirect");
  validateOrigins(value.allowedArtifactRedirectOrigins, "artifact redirect");
  if (value.status === "verified") {
    if (value.publicKeys.length !== 2) throw new Error("verified distribution must embed current and next catalog keys");
    if (!value.allowedCatalogRedirectOrigins.some((origin) => origin === "https://huggingface.co")) {
      throw new Error("verified distribution must allow the catalog origin");
    }
    if (value.allowedArtifactRedirectOrigins.length === 0) {
      throw new Error("verified distribution must allow an artifact redirect origin");
    }
  }
}

function validateOrigins(values, label) {
  if (!Array.isArray(values) || new Set(values).size !== values.length) {
    throw new Error(`${label} origins must be a unique array`);
  }
  for (const value of values) {
    let url;
    try {
      url = new URL(value);
    } catch {
      throw new Error(`${label} origin is invalid`);
    }
    if (url.protocol !== "https:" || url.origin !== value) throw new Error(`${label} origin must be an HTTPS origin`);
  }
}

export function validateHostedConnectSources(headers, distributionLock) {
  const match = /(?:^|;)\s*connect-src\s+([^;\n]+)/m.exec(headers);
  if (!match) throw new Error("hosted CSP has no connect-src directive");
  const actual = new Set(match[1].trim().split(/\s+/));
  const expected = new Set([
    "'self'",
    "https://www.neuronpedia.org",
    ...distributionLock.allowedCatalogRedirectOrigins,
    ...distributionLock.allowedArtifactRedirectOrigins,
  ]);
  if (actual.size !== expected.size || [...actual].some((source) => !expected.has(source))) {
    throw new Error("hosted CSP connect-src does not match distribution and description sources");
  }
  if ([...actual].some((source) => source.includes("*"))) {
    throw new Error("hosted CSP connect-src must not contain wildcard origins");
  }
}

export function validateWebLlmDependency(manifest, packageLock, packagePin) {
  const declarations = { ...(manifest.dependencies ?? {}), ...(manifest.devDependencies ?? {}) };
  if (declarations["@drowse/web-llm"] !== `file:../${packagePin.path}`) {
    throw new Error("browser backend must use the runtime-locked Drowse WebLLM archive");
  }
  // The workspace lock at the repository root records the archive root-relative.
  const locked = packageLock.packages?.["node_modules/@drowse/web-llm"];
  if (!locked || locked.resolved !== `file:${packagePin.path}`) {
    throw new Error("package lock does not resolve the runtime-locked Drowse WebLLM archive");
  }
}

async function validateWebLlmPackageArchive(pin, forkPin) {
  exactKeys(pin, ["path", "sha256", "name", "version", "repository", "sourceCommit"], "WebLLM package pin");
  if (
    typeof pin.path !== "string" ||
    !/^browser-runtime\/vendor\/drowse-web-llm-[0-9A-Za-z.-]+\.tgz$/.test(pin.path) ||
    !SHA256.test(pin.sha256) || pin.name !== "@drowse/web-llm" ||
    /drowse-web-llm-(.+)\.tgz$/u.exec(pin.path)?.[1] !== pin.version
  ) {
    throw new Error("WebLLM package pin is invalid");
  }
  if (pin.sourceCommit !== null && (!COMMIT.test(pin.sourceCommit) || pin.sourceCommit !== forkPin.commit)) {
    throw new Error("WebLLM package source commit differs from the fork commit");
  }
  const archive = resolve(repositoryRoot, pin.path);
  if (createHash("sha256").update(await readFile(archive)).digest("hex") !== pin.sha256) {
    throw new Error("WebLLM package archive does not match its runtime-lock digest");
  }
  const packageManifest = JSON.parse(execFileSync("tar", ["-xOf", archive, "package/package.json"], {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  }));
  if (packageManifest.name !== pin.name || packageManifest.version !== pin.version) {
    throw new Error("WebLLM package manifest differs from its runtime-lock pin");
  }
}

export function validateInstalledWebLlmDependency(manifest, packageLock, installed) {
  const specifier = manifest.dependencies?.["@drowse/web-llm"] ?? manifest.devDependencies?.["@drowse/web-llm"];
  const locked = packageLock.packages?.["node_modules/@drowse/web-llm"];
  if (!locked || typeof locked.version !== "string") throw new Error("package lock has no installed Drowse WebLLM version");
  if (installed?.name !== "@drowse/web-llm" || installed.version !== locked.version) {
    throw new Error("installed Drowse WebLLM does not match the exact package lock");
  }
  if (typeof specifier === "string" && specifier.startsWith("file:")) {
    if (/drowse-web-llm-(.+)\.tgz$/u.exec(specifier)?.[1] !== installed.version) {
      throw new Error("installed Drowse WebLLM version does not match its local archive");
    }
  }
}

function exactKeys(value, keys, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} must be an object`);
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  if (actual.length !== expected.length || actual.some((key, index) => key !== expected[index])) {
    throw new Error(`${label} has unknown or missing fields`);
  }
}

export function lockedRuntimeIdentitySha256(runtime, model) {
  const identity = {
    sourceModel: model.sourceRepository,
    sourceRevision: model.sourceRevision,
    convertedManifestSha256: model.manifestSha256,
    quantization: model.quantization,
    tokenizerSha256: model.tokenizerSha256,
    chatTemplateSha256: model.chatTemplateSha256,
    modelLibrarySha256: model.librarySha256,
    runtimeAbi: runtime.runtimeAbi,
    hookAbi: runtime.hookAbi,
    hiddenSize: model.hiddenSize,
    layerMap: model.layerMap,
  };
  if (Object.values(identity).some((value) => value === null)) return null;
  return digestCanonical(identity);
}

export function runtimeLockIdentitySha256(runtime) {
  if (!runtime || typeof runtime !== "object" || Array.isArray(runtime)) {
    throw new Error("browser runtime lock identity must be an object");
  }
  const { status: _status, ...identity } = runtime;
  return digestCanonical(identity);
}

export function digestCanonical(value) {
  return createHash("sha256").update(canonicalJson(value)).digest("hex");
}

function canonicalJson(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  return `{${Object.entries(value)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, entry]) => `${JSON.stringify(key)}:${canonicalJson(entry)}`)
    .join(",")}}`;
}

if (process.argv[1] && pathToFileURL(process.argv[1]).href === import.meta.url) {
  await main();
}
