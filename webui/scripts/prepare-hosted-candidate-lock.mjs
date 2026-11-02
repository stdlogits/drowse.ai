#!/usr/bin/env node
// W4's lock tool on hook ABI v5. The repository lock carries every value a compile does not
// produce (ABIs, profiles, block specs, unit refusals, tokenizer and chat-template digests,
// shapes, toolchain); each library-bound field is null until a v5 library fills it:
//
//   node prepare-hosted-candidate-lock.mjs MODEL_ID PUBLISH_DIR OUTPUT.json \
//     [--base LOCK.json] [--converted-revision SHA]
//
// fills one model from an assembled publish directory (`assemble:hosted:model`): `librarySha256`
// (model.wasm), `manifestSha256` (tensor-cache.json) and, once the publisher returns it,
// `convertedRevision`; refreshes the toolchain digests of the checked-in fork manifest and
// compiler recipe; and refuses a directory whose build ABI, profile, architecture, shapes,
// tokenizer, chat template, VM function surface or block spec (from its mlc-chat-config.json, as
// the overlay derives the descriptor's) differ from the lock. `--base` chains models into one
// candidate; the output must not exist.
//
//   node prepare-hosted-candidate-lock.mjs --adopt CANDIDATE.json
//
// writes a complete candidate (no pending field) to browser-runtime/runtime-lock.json as
// `verified`, with runtime-lock-identity-v1.json beside it.

import { createHash, randomUUID } from "node:crypto";
import {
  access,
  link,
  mkdir,
  readFile,
  rename,
  rm,
  writeFile,
} from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
  browserRuntimeIdentityBlockers,
  runtimeLockIdentitySha256,
} from "./check-runtime-lock.mjs";
import { requireLibraryVmFunctions } from "./overlay-vm-functions.mjs";
import { validatePublishDirectory } from "./publish-hosted-model.mjs";
import { blockSpecFromMlcConfig, sameBlockSpec } from "./runtime-lock-block-spec.mjs";
import {
  readRuntimeLock,
  validateRuntimeLock,
} from "./runtime-lock-document.mjs";

const repositoryRoot = fileURLToPath(new URL("../../", import.meta.url));
const repositoryRuntimeLockPath = resolve(repositoryRoot, "browser-runtime/runtime-lock.json");
const identityFixturePath = resolve(repositoryRoot, "browser-runtime/runtime-lock-identity-v1.json");
const forkManifestPath = resolve(repositoryRoot, "browser-runtime/forks/manifest.json");
const compilerRecipePath = resolve(repositoryRoot, "browser-runtime/compiler/Dockerfile");

const USAGE =
  "usage: node prepare-hosted-candidate-lock.mjs MODEL_ID PUBLISH_DIR OUTPUT.json " +
  "[--base LOCK.json] [--converted-revision SHA] | --adopt CANDIDATE.json";

export function parseArguments(args) {
  if (args[0] === "--adopt") {
    if (args.length !== 2 || args[1].startsWith("--")) throw new Error(USAGE);
    return { adopt: resolve(args[1]) };
  }
  const positional = [];
  const options = { basePath: repositoryRuntimeLockPath, convertedRevision: null };
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index];
    if (argument === "--base" || argument === "--converted-revision") {
      const value = args[++index];
      if (!value || value.startsWith("--")) throw new Error(USAGE);
      if (argument === "--base") options.basePath = resolve(value);
      else options.convertedRevision = value;
    } else if (argument.startsWith("--")) {
      throw new Error(USAGE);
    } else {
      positional.push(argument);
    }
  }
  if (positional.length !== 3) throw new Error(USAGE);
  if (options.convertedRevision !== null && !/^[0-9a-f]{40}$/.test(options.convertedRevision)) {
    throw new Error("--converted-revision must be a 40-character commit");
  }
  return {
    modelId: positional[0],
    modelDirectory: resolve(positional[1]),
    outputPath: resolve(positional[2]),
    ...options,
  };
}

function unique(manifest, role, path) {
  const matches = manifest.files.filter((file) => file?.role === role);
  if (
    matches.length !== 1 || matches[0].path !== path ||
    typeof matches[0].sha256 !== "string" || !/^[0-9a-f]{64}$/.test(matches[0].sha256)
  ) {
    throw new Error(`hosted-artifacts.json has no unique ${path} (${role}) digest`);
  }
  return matches[0].sha256;
}

/** Fills `modelId`'s library-bound fields from an assembled v5 closure (`manifest` is its
 *  hosted-artifacts.json, `chatConfig` its mlc-chat-config.json) and refreshes the toolchain
 *  digests; every other field must already agree. Returns a new, validated lock. */
export function deriveCandidateRuntimeLock(runtimeLock, modelId, {
  manifest,
  chatConfig,
  convertedRevision = null,
  forkManifestSha256 = runtimeLock.toolchain?.forkManifestSha256,
  recipeSha256 = runtimeLock.toolchain?.compilerImage?.recipeSha256,
}) {
  validateRuntimeLock(runtimeLock);
  const modelIndex = runtimeLock.models.findIndex((model) => model.id === modelId);
  if (modelIndex < 0) throw new Error(`unknown runtime-lock model ${modelId}`);
  const model = runtimeLock.models[modelIndex];
  if (!manifest || typeof manifest !== "object" || !Array.isArray(manifest.files)) {
    throw new Error("hosted-artifacts.json is invalid");
  }
  const differs = (field, actual, expected) => {
    if (JSON.stringify(actual) !== JSON.stringify(expected)) {
      throw new Error(`${modelId} ${field} differs from the runtime lock: ${JSON.stringify(actual)}`);
    }
  };
  differs("hookAbi", manifest.hookAbi, runtimeLock.hookAbi);
  differs("runtimeAbi", manifest.runtimeAbi, runtimeLock.runtimeAbi);
  differs("architecture", manifest.architecture, model.architecture);
  differs("structuredHookProfile", manifest.structuredHookProfile, model.structuredHookProfile);
  differs("quantization", manifest.quantization, model.quantization);
  differs("modelType", manifest.modelType ?? "chat", model.modelType ?? "chat");
  differs("contextWindowSize", manifest.contextWindowSize, Math.max(...model.contextProfiles));
  // The descriptor attests the prefill chunk the browser expects: the smallest context profile.
  differs("prefillChunkSize", manifest.prefillChunkSize, Math.min(...model.contextProfiles));
  if (model.hiddenSize !== null) differs("hiddenSize", manifest.hiddenSize, model.hiddenSize);
  if (model.layerMap !== null) differs("layerMap", manifest.layerMap, model.layerMap);
  const tokenizer = unique(manifest, "tokenizer", "tokenizer.json");
  const chatTemplate = unique(manifest, "chat_template", "tokenizer_config.json");
  // The tokenizer and chat template come from the pinned source unchanged by a recompile; the S
  // statistics bind the tokenizer digest, so a drift here is a conversion error, not a refill.
  if (model.tokenizerSha256 !== null) differs("tokenizer digest", tokenizer, model.tokenizerSha256);
  if (model.chatTemplateSha256 !== null) differs("chat template digest", chatTemplate, model.chatTemplateSha256);
  if (model.structuredHookProfile === "standard-v4") {
    const compiled = blockSpecFromMlcConfig(model.architecture, chatConfig);
    if (!sameBlockSpec(compiled, model.blockSpec)) {
      throw new Error(
        `${modelId} block spec differs from its converted configuration: ${JSON.stringify(compiled)}`,
      );
    }
  }
  const candidate = structuredClone(runtimeLock);
  const pin = candidate.models[modelIndex];
  const librarySha256 = unique(manifest, "model_library", "model.wasm");
  const manifestSha256 = unique(manifest, "converted_manifest", "tensor-cache.json");
  // A new library or conversion is a new published closure: the old revision does not hold it.
  if (librarySha256 !== pin.librarySha256 || manifestSha256 !== pin.manifestSha256) {
    pin.convertedRevision = null;
  }
  pin.librarySha256 = librarySha256;
  pin.manifestSha256 = manifestSha256;
  pin.tokenizerSha256 = tokenizer;
  pin.chatTemplateSha256 = chatTemplate;
  pin.hiddenSize = manifest.hiddenSize;
  pin.layerMap = [...manifest.layerMap];
  if (convertedRevision !== null) pin.convertedRevision = convertedRevision;
  candidate.toolchain.forkManifestSha256 = forkManifestSha256;
  candidate.toolchain.compilerImage.recipeSha256 = recipeSha256;
  candidate.status = "feasibility-required";
  validateRuntimeLock(candidate);
  return candidate;
}

/** The lock `--adopt` writes: the candidate, complete, as `verified`. */
export function adoptedRuntimeLock(candidate) {
  validateRuntimeLock(candidate);
  const blockers = browserRuntimeIdentityBlockers(candidate);
  if (blockers.length > 0) {
    throw new Error(`candidate runtime lock is incomplete:\n- ${blockers.join("\n- ")}`);
  }
  const adopted = { ...structuredClone(candidate), status: "verified" };
  validateRuntimeLock(adopted);
  return adopted;
}

const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");

async function main() {
  const options = parseArguments(process.argv.slice(2));
  if (options.adopt) {
    const { value } = await readRuntimeLock(options.adopt);
    const adopted = adoptedRuntimeLock(value);
    await replaceFile(repositoryRuntimeLockPath, `${JSON.stringify(adopted, null, 2)}\n`);
    const identity = runtimeLockIdentitySha256(adopted);
    await replaceFile(
      identityFixturePath,
      `${JSON.stringify({ schemaVersion: 1, runtimeLockIdentitySha256: identity }, null, 2)}\n`,
    );
    console.log(JSON.stringify({ adopted: repositoryRuntimeLockPath, runtimeLockIdentitySha256: identity }));
    return;
  }
  await requireMissing(options.outputPath);
  const { value: base } = await readRuntimeLock(options.basePath);
  const [manifest, chatConfig, library, forkManifest, recipe] = await Promise.all([
    readFile(resolve(options.modelDirectory, "hosted-artifacts.json"), "utf8").then(JSON.parse),
    readFile(resolve(options.modelDirectory, "mlc-chat-config.json"), "utf8").then(JSON.parse),
    readFile(resolve(options.modelDirectory, "model.wasm")),
    readFile(forkManifestPath),
    readFile(compilerRecipePath),
  ]);
  const pin = base.models.find((entry) => entry.id === options.modelId);
  if (!pin) throw new Error(`unknown runtime-lock model ${options.modelId}`);
  requireLibraryVmFunctions(library, pin.structuredHookProfile);
  const candidate = deriveCandidateRuntimeLock(base, options.modelId, {
    manifest,
    chatConfig,
    convertedRevision: options.convertedRevision,
    forkManifestSha256: sha256(forkManifest),
    recipeSha256: sha256(recipe),
  });
  const model = candidate.models.find((entry) => entry.id === options.modelId);
  await validatePublishDirectory(options.modelDirectory, model, candidate);
  const bytes = `${JSON.stringify(candidate, null, 2)}\n`;
  const temporary = `${options.outputPath}.partial-${randomUUID()}`;
  await mkdir(dirname(options.outputPath), { recursive: true });
  try {
    await writeFile(temporary, bytes, { flag: "wx" });
    await link(temporary, options.outputPath);
    await rm(temporary);
  } catch (error) {
    await rm(temporary, { force: true });
    throw error;
  }
  console.log(JSON.stringify({
    modelId: options.modelId,
    outputPath: options.outputPath,
    librarySha256: model.librarySha256,
    manifestSha256: model.manifestSha256,
    convertedRevision: model.convertedRevision,
    pending: browserRuntimeIdentityBlockers(candidate),
  }));
}

async function replaceFile(path, text) {
  const temporary = `${path}.partial-${randomUUID()}`;
  await writeFile(temporary, text, { flag: "wx" });
  await rename(temporary, path);
}

async function requireMissing(path) {
  try {
    await access(path);
  } catch (error) {
    if (error?.code === "ENOENT") return;
    throw error;
  }
  throw new Error(`candidate runtime-lock output already exists: ${path}`);
}

if (
  process.argv[1] &&
  pathToFileURL(process.argv[1]).href === import.meta.url
) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
