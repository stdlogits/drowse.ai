import { readFile } from "node:fs/promises";
import {
  EXACT_READOUT_ABI,
  HOOK_ABI,
  RUNTIME_ABI,
  architectureStructuredHookProfile,
} from "./hosted-model-profiles.mjs";
import { validateLockBlockSpec } from "./runtime-lock-block-spec.mjs";

export const RUNTIME_LOCK_SCHEMA_VERSION = 2;
export const LOCK_ARCHITECTURES = Object.freeze(["gemma3_text", "qwen3", "gpt2", "gpt_neox", "qwen3_5"]);
const BASE_ARCHITECTURES = ["gpt2", "gpt_neox", "qwen3_5"];

export async function readRuntimeLock(path) {
  const bytes = await readFile(path);
  if (bytes.byteLength > 1024 * 1024) throw new Error("runtime lock exceeds 1048576 bytes");
  let value;
  try {
    value = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  } catch (error) {
    throw new Error("runtime lock is not valid UTF-8 JSON", { cause: error });
  }
  validateRuntimeLock(value);
  return { bytes, value };
}

export function validateRuntimeLock(value) {
  exactKeys(value, [
    "$schema",
    "schemaVersion",
    "status",
    "runtimeAbi",
    "hookAbi",
    "exactReadoutAbi",
    "toolchain",
    "models",
  ], "runtime lock");
  if (
    value.$schema !== "./runtime-lock.schema.json" ||
    value.schemaVersion !== RUNTIME_LOCK_SCHEMA_VERSION ||
    !["feasibility-required", "verified"].includes(value.status) ||
    value.runtimeAbi !== RUNTIME_ABI ||
    value.hookAbi !== HOOK_ABI ||
    value.exactReadoutAbi !== EXACT_READOUT_ABI
  ) throw new Error("runtime lock has an unsupported schema or ABI");
  validateRuntimeToolchain(value.toolchain);
  if (!Array.isArray(value.models) || value.models.length < 1) {
    throw new Error("runtime lock must contain at least one model");
  }
  const modelIds = new Set();
  for (const model of value.models) {
    validateRuntimeModel(model);
    if (modelIds.has(model.id)) throw new Error(`runtime lock repeats model ${model.id}`);
    modelIds.add(model.id);
  }
}

function validateRuntimeToolchain(value) {
  exactKeys(value, [
    "forkManifestSha256",
    "mlcLlmFork",
    "webLlmFork",
    "webLlmPackage",
    "tvmRepository",
    "tvmCommit",
    "tvmFfiCommit",
    "emsdkCommit",
    "emccVersion",
    "emccRevision",
    "emccPlatform",
    "llvmVersion",
    "llvmArtifacts",
    "llvmSourceBuild",
    "compilerImage",
  ], "runtime lock toolchain");
  validateRepositoryPin(value.mlcLlmFork, "MLC-LLM fork");
  validateRepositoryPin(value.webLlmFork, "WebLLM fork");
  exactKeys(value.webLlmPackage, [
    "path",
    "sha256",
    "name",
    "version",
    "repository",
    "sourceCommit",
  ], "runtime lock WebLLM package");
  if (
    !/^browser-runtime\/vendor\/drowse-web-llm-[0-9A-Za-z.-]+\.tgz$/.test(
      value.webLlmPackage.path,
    ) || !isSha256(value.webLlmPackage.sha256) ||
    value.webLlmPackage.name !== "@drowse/web-llm" ||
    !/^[0-9]+\.[0-9]+\.[0-9]+-drowse\.[0-9]+$/.test(value.webLlmPackage.version) ||
    value.webLlmPackage.repository !== "https://github.com/a9lim/web-llm-polythetic" ||
    !nullablePattern(value.webLlmPackage.sourceCommit, /^[0-9a-f]{40}$/)
  ) throw new Error("runtime lock WebLLM package is invalid");
  if (
    !isSha256(value.forkManifestSha256) ||
    value.tvmRepository !== "https://github.com/apache/tvm.git" ||
    !nullablePattern(value.tvmCommit, /^[0-9a-f]{40}$/) ||
    !/^[0-9a-f]{40}$/.test(value.tvmFfiCommit) ||
    !/^[0-9a-f]{40}$/.test(value.emsdkCommit) ||
    !/^[0-9]+\.[0-9]+\.[0-9]+$/.test(value.emccVersion) ||
    !/^[0-9a-f]{40}$/.test(value.emccRevision) ||
    value.emccPlatform !== "linux/amd64" ||
    !/^[0-9]+\.[0-9]+\.[0-9]+$/.test(value.llvmVersion)
  ) throw new Error("runtime lock toolchain is invalid");
  // One release archive per compiler platform (linux/amd64's is the source build's bootstrap).
  if (
    !Array.isArray(value.llvmArtifacts) || value.llvmArtifacts.length !== 2 ||
    value.llvmArtifacts.map((artifact) => artifact?.platform).sort().join(",") !== "linux/amd64,linux/arm64"
  ) {
    throw new Error("runtime lock LLVM artifacts must name one archive for each of linux/amd64 and linux/arm64");
  }
  const llvmArtifacts = new Set();
  for (const artifact of value.llvmArtifacts) {
    exactKeys(artifact, ["platform", "archive", "sha256"], "LLVM artifact");
    if (
      !["linux/amd64", "linux/arm64"].includes(artifact.platform) ||
      typeof artifact.archive !== "string" || artifact.archive.length === 0 ||
      !isSha256(artifact.sha256)
    ) throw new Error("runtime lock LLVM artifact is invalid");
    const identity = JSON.stringify(artifact);
    if (llvmArtifacts.has(identity)) throw new Error("runtime lock repeats an LLVM artifact");
    llvmArtifacts.add(identity);
  }
  validateLlvmSourceBuild(value.llvmSourceBuild, value);
  exactKeys(
    value.compilerImage,
    ["reference", "digest", "recipeSha256"],
    "runtime lock compiler image",
  );
  if (
    !nullablePattern(
      value.compilerImage.reference,
      /^[a-z0-9.-]+(?::[1-9][0-9]{0,4})?\/[a-z0-9]+(?:[._/-][a-z0-9]+)*(?::[A-Za-z0-9_][A-Za-z0-9._-]{0,127})?$/,
    ) ||
    !nullablePattern(value.compilerImage.digest, /^sha256:[0-9a-f]{64}$/) ||
    !isSha256(value.compilerImage.recipeSha256)
  ) throw new Error("runtime lock compiler image is invalid");
}

// linux/amd64 compiles TVM against LLVM built from source at the release tag, by the release
// archive's own clang (the archive's libraries carry GCC 7's std::optional ABI, which no C++20
// libstdc++ shares; parity handoff B40-toolchain.md §3); linux/arm64 links the archive directly.
export function validateLlvmSourceBuild(value, toolchain) {
  exactKeys(value, [
    "platform",
    "repository",
    "tag",
    "tagObject",
    "commit",
    "bootstrapArchive",
  ], "runtime lock LLVM source build");
  const bootstrap = toolchain.llvmArtifacts.find((artifact) => artifact.platform === value.platform);
  if (
    value.platform !== "linux/amd64" ||
    value.repository !== "https://github.com/llvm/llvm-project.git" ||
    value.tag !== `llvmorg-${toolchain.llvmVersion}` ||
    !/^[0-9a-f]{40}$/.test(value.tagObject) ||
    !/^[0-9a-f]{40}$/.test(value.commit) ||
    value.tagObject === value.commit ||
    bootstrap?.archive !== value.bootstrapArchive
  ) throw new Error("runtime lock LLVM source build is invalid");
}

function validateRepositoryPin(value, label) {
  exactKeys(value, ["repository", "commit"], label);
  if (
    typeof value.repository !== "string" || !validUrl(value.repository) ||
    !nullablePattern(value.commit, /^[0-9a-f]{40}$/)
  ) throw new Error(`${label} is invalid`);
}

function validateRuntimeModel(value) {
  const units = value?.structuredHookProfile === "standard-v5";
  exactKeys(value, [
    "id",
    "architecture",
    "structuredHookProfile",
    units ? "blockSpec" : "unitRefusal",
    "thinkingProfile",
    "sourceRepository",
    "sourceRevision",
    "convertedRepository",
    "convertedRevision",
    "manifestSha256",
    "librarySha256",
    "tokenizerSha256",
    "chatTemplateSha256",
    "hiddenSize",
    "layerMap",
    "quantization",
    "contextProfiles",
    "prefillChunkSize",
    ...("modelType" in value ? ["modelType"] : []),
  ], `runtime lock model ${String(value?.id)}`);
  if (
    typeof value.id !== "string" || !/^[a-z0-9][a-z0-9._-]{0,63}$/.test(value.id) ||
    !LOCK_ARCHITECTURES.includes(value.architecture) ||
    (value.modelType !== undefined && !["chat", "base"].includes(value.modelType)) ||
    (BASE_ARCHITECTURES.includes(value.architecture) && value.modelType !== "base") ||
    typeof value.sourceRepository !== "string" || !value.sourceRepository.includes("/") ||
    !/^[0-9a-f]{40}$/.test(value.sourceRevision) ||
    typeof value.convertedRepository !== "string" ||
    !value.convertedRepository.includes("/") ||
    !nullablePattern(value.convertedRevision, /^[0-9a-f]{40}$/) ||
    !nullablePattern(value.manifestSha256, /^[0-9a-f]{64}$/) ||
    !nullablePattern(value.librarySha256, /^[0-9a-f]{64}$/) ||
    !nullablePattern(value.tokenizerSha256, /^[0-9a-f]{64}$/) ||
    !nullablePattern(value.chatTemplateSha256, /^[0-9a-f]{64}$/) ||
    (value.hiddenSize !== null &&
      (!Number.isSafeInteger(value.hiddenSize) || value.hiddenSize < 1)) ||
    !["q4f16_1", "q4f32_1", "q0f32"].includes(value.quantization)
  ) throw new Error(`runtime lock model ${String(value.id)} is invalid`);
  // Hook ABI v5 (B1 §7, §8.2): the profile follows the architecture; a units library carries the
  // block description its descriptor attests, a residual one the refusal its unit calls answer.
  try {
    architectureStructuredHookProfile(value.architecture, value.structuredHookProfile);
  } catch (error) {
    throw new Error(`runtime lock model ${value.id}: ${error.message}`, { cause: error });
  }
  if (units) {
    validateLockBlockSpec(value.blockSpec, `runtime lock model ${value.id} block spec`);
  } else if (!boundedText(value.unitRefusal, 1, 512)) {
    throw new Error(`runtime lock model ${value.id} has no unit refusal text`);
  }
  if (value.thinkingProfile !== null) {
    exactKeys(value.thinkingProfile, [
      "start",
      "end",
      "startsInThinking",
      "startTokenIds",
      "endTokenIds",
    ], `runtime lock model ${value.id} thinking profile`);
    if (
      !boundedText(value.thinkingProfile.start, 1, 256) ||
      !boundedText(value.thinkingProfile.end, 1, 256) ||
      typeof value.thinkingProfile.startsInThinking !== "boolean" ||
      !tokenIds(value.thinkingProfile.startTokenIds) ||
      !tokenIds(value.thinkingProfile.endTokenIds)
    ) throw new Error(`runtime lock model ${value.id} thinking profile is invalid`);
  }
  if (
    value.layerMap !== null &&
    (!Array.isArray(value.layerMap) || value.layerMap.length === 0 ||
      value.layerMap.some((layer) => !Number.isSafeInteger(layer) || layer < 0) ||
      new Set(value.layerMap).size !== value.layerMap.length)
  ) throw new Error(`runtime lock model ${value.id} layer map is invalid`);
  if (
    !Array.isArray(value.contextProfiles) || value.contextProfiles.length < 1 ||
    value.contextProfiles.some((context) => ![1024, 2048, 4096].includes(context)) ||
    (value.architecture === "gpt2" && value.contextProfiles.some((context) => context > 1024)) ||
    new Set(value.contextProfiles).size !== value.contextProfiles.length
  ) throw new Error(`runtime lock model ${value.id} context profiles are invalid`);
  // The library's compiled prefill chunk, which its descriptor attests (HOOK_ABI.md §6): no forward
  // carries more rows, so it never exceeds the smallest context the model is offered at.
  if (
    !Number.isSafeInteger(value.prefillChunkSize) || value.prefillChunkSize < 1 ||
    value.prefillChunkSize > Math.min(...value.contextProfiles)
  ) throw new Error(`runtime lock model ${value.id} prefill chunk size is invalid`);
}

function exactKeys(value, expected, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  if (
    actual.length !== wanted.length ||
    actual.some((key, index) => key !== wanted[index])
  ) throw new Error(`${label} has unknown or missing fields`);
}

function nullablePattern(value, pattern) {
  return value === null || (typeof value === "string" && pattern.test(value));
}

function isSha256(value) {
  return typeof value === "string" && /^[0-9a-f]{64}$/.test(value);
}

function validUrl(value) {
  try {
    return new URL(value).href.length > 0;
  } catch {
    return false;
  }
}

function boundedText(value, minimum, maximum) {
  return typeof value === "string" &&
    value.length >= minimum && value.length <= maximum;
}

function tokenIds(value) {
  return Array.isArray(value) && value.length >= 1 && value.length <= 32 &&
    value.every((tokenId) => Number.isSafeInteger(tokenId) && tokenId >= 0);
}
