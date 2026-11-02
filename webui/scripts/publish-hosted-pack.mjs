#!/usr/bin/env node

import { execFile } from "node:child_process";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { validatePackDirectory } from "./build-hosted-catalog.mjs";
import { RELEASE_POLICY_KINDS } from "./hosted-instrument-release-policy.mjs";
import { uploadCommitRevision } from "./hugging-face-upload.mjs";

const exec = promisify(execFile);
const runtimeLockUrl = new URL("../../browser-runtime/runtime-lock.json", import.meta.url);
const coreValidatorUrl = new URL("./validate-hosted-core-pack.mjs", import.meta.url);
const instrumentValidatorUrl = new URL("./validate-hosted-instrument-pack.mjs", import.meta.url);

const USAGE =
  "usage: node publish-hosted-pack.mjs MODEL_ID core|sae|jlens|slens|units DIRECTORY " +
  "[--with PACK_DIR]... [--core-statistics DIR] [--upload]\n" +
  "  a units pack is validated through the R and S packs it binds (--with) and the core pack's " +
  "neutral statistics (--core-statistics)";

export function parseArguments(args) {
  const values = [];
  const companions = [];
  let coreStatistics = null;
  let upload = false;
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index];
    if (argument === "--upload") {
      upload = true;
    } else if (argument === "--with" || argument === "--core-statistics") {
      const value = args[++index];
      if (!value || value.startsWith("--")) throw new Error(USAGE);
      if (argument === "--with") companions.push(resolve(value));
      else coreStatistics = resolve(value);
    } else if (argument.startsWith("--")) {
      throw new Error(USAGE);
    } else {
      values.push(argument);
    }
  }
  if (values.length !== 3 || !["core", "sae", "jlens", "slens", "units"].includes(values[1])) {
    throw new Error(USAGE);
  }
  if ((values[1] === "units") !== (companions.length > 0 || coreStatistics !== null)) {
    throw new Error(`${USAGE}\n  --with and --core-statistics belong to a units pack`);
  }
  return {
    modelId: values[0],
    kind: values[1],
    directory: resolve(values[2]),
    companions,
    coreStatistics,
    upload,
  };
}

export function targetRepository(model, kind) {
  if (!model?.convertedRepository) throw new Error("runtime model has no converted repository");
  return kind === "core"
    ? model.convertedRepository
    : `${model.convertedRepository}-instruments`;
}

/** The validator a pack must pass before upload: the core validator, or the instrument
 *  validator, under the release policy (`--release`: full depth, the pinned corpus and its
 *  attribution) for the lens and units packs drowse fits natively or a provider's R lens. */
export function semanticValidationArguments(modelId, kind, directory, companions = [], coreStatistics = null) {
  return kind === "core"
    ? [fileURLToPath(coreValidatorUrl), modelId, directory]
    : [
        fileURLToPath(instrumentValidatorUrl),
        modelId,
        kind,
        directory,
        ...(RELEASE_POLICY_KINDS.includes(kind) ? ["--release"] : []),
        ...companions.flatMap((companion) => ["--with", companion]),
        ...(coreStatistics === null ? [] : ["--core-statistics", coreStatistics]),
      ];
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  const runtimeLock = JSON.parse(await readFile(runtimeLockUrl, "utf8"));
  const model = runtimeLock.models?.find((candidate) => candidate.id === options.modelId);
  if (!model) throw new Error(`unknown runtime model: ${options.modelId}`);
  const manifest = await validatePackDirectory(options.directory);
  await exec(process.execPath, semanticValidationArguments(
    options.modelId,
    options.kind,
    options.directory,
    options.companions,
    options.coreStatistics,
  ));
  const repository = targetRepository(model, options.kind);
  const info = await modelInfo(repository);
  if (info.private === true || info.gated === true || info.gated === "auto" || info.gated === "manual") {
    throw new Error(`hosted repository must be public and ungated: ${repository}`);
  }
  if (!options.upload) {
    console.log(JSON.stringify({
      checked: repository,
      kind: options.kind,
      files: manifest.files.length,
      bytes: manifest.files.reduce((sum, file) => sum + file.bytes, 0),
      currentRevision: info.sha,
    }));
    return;
  }
  const uploaded = await exec("hf", [
    "upload",
    repository,
    options.directory,
    ".",
    "--type",
    "model",
    "--commit-message",
    `publish ${options.modelId} ${options.kind} browser pack`,
    "--format",
    "json",
  ], { encoding: "utf8" });
  const revision = uploadCommitRevision(uploaded.stdout, repository);
  console.log(JSON.stringify({ repository, revision, kind: options.kind }));
}

async function modelInfo(repository) {
  const response = await exec("hf", [
    "models", "info", repository,
    "--expand", "sha,private,gated", "--format", "json",
  ], { encoding: "utf8" });
  return JSON.parse(response.stdout);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
