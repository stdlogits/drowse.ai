#!/usr/bin/env node
// The runtime identity a hosted catalog variant carries, read from a runtime lock: what every
// pack builder binds to (`browser-runtime/build_hosted_packs.py --runtime-identity`). W4's
// recompiled libraries change the lock, so packs are rebound by rerunning the builder with the
// new record.

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

export function lockedRuntimeIdentity(runtimeLock, modelId) {
  const lock = runtimeLock?.models?.find((entry) => entry.id === modelId);
  if (!lock) throw new Error(`unknown runtime-lock model ${String(modelId)}`);
  return {
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
}

/** The catalog's digest (`runtimeIdentitySha256` in `src/lib/runtime/catalog.ts`). */
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

export function runtimeIdentityRecord(runtimeLock, modelId) {
  const runtimeIdentity = lockedRuntimeIdentity(runtimeLock, modelId);
  return { modelId, runtimeIdentity, runtimeIdentitySha256: digestCanonical(runtimeIdentity) };
}

async function main(args) {
  const positional = [];
  let lockPath = fileURLToPath(new URL("../../browser-runtime/runtime-lock.json", import.meta.url));
  let output = null;
  for (let index = 0; index < args.length; index += 1) {
    if (args[index] === "--runtime-lock" || args[index] === "--output") {
      const value = args[index + 1];
      if (!value || value.startsWith("--")) throw new Error(`${args[index]} requires a path`);
      if (args[index] === "--runtime-lock") lockPath = resolve(value);
      else output = resolve(value);
      index += 1;
    } else {
      positional.push(args[index]);
    }
  }
  if (positional.length !== 1) {
    throw new Error("usage: hosted-runtime-identity.mjs MODEL_ID [--runtime-lock PATH] [--output FILE]");
  }
  const runtimeLock = JSON.parse(await readFile(lockPath, "utf8"));
  const text = `${JSON.stringify(runtimeIdentityRecord(runtimeLock, positional[0]), null, 2)}\n`;
  if (output === null) process.stdout.write(text);
  else await writeFile(output, text);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
  main(process.argv.slice(2)).catch((error) => {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  });
}
