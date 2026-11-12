import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { createHash } from "node:crypto";
import { promisify } from "node:util";
import {
  CAPTURE_SCHEMA_VERSION,
  MAXIMUM_STATIC_READS,
  captureUnitsMember,
  unitsCaptureFailures,
} from "./hosted-residual-capture-units.mjs";

const exec = promisify(execFile);
const repositoryRoot = resolve(import.meta.dirname, "../..");
const script = resolve(import.meta.dirname, "capture-hosted-residuals.mjs");
const runtimeLockPath = resolve(repositoryRoot, "browser-runtime/runtime-lock.json");

const help = await exec(process.execPath, [script, "--help"]);
assert.match(help.stdout, /--runtime-lock PATH/);

const root = await mkdtemp(join(tmpdir(), "drowse-capture-lock-"));
try {
  const runtimeLock = JSON.parse(await readFile(runtimeLockPath, "utf8"));
  const unexpectedPath = join(root, "unexpected.json");
  await writeFile(
    unexpectedPath,
    JSON.stringify({ ...runtimeLock, unexpected: true }),
  );
  await assert.rejects(
    runWithLock(unexpectedPath),
    /runtime lock has unknown or missing fields/,
  );

  const invalidModelPath = join(root, "invalid-model.json");
  runtimeLock.models[0].contextProfiles = [2048, 2048];
  await writeFile(invalidModelPath, JSON.stringify(runtimeLock));
  await assert.rejects(
    runWithLock(invalidModelPath),
    /context profiles are invalid/,
  );
} finally {
  await rm(root, { recursive: true, force: true });
}

// Capture schema 5: a standard-v5 capture carries its block sites and one static read per layer;
// a residual one carries neither.
{
  assert.equal(CAPTURE_SCHEMA_VERSION, 5);
  const lock = {
    hiddenSize: 4,
    layerMap: [0, 1],
    blockSpec: { dFF: 6, nHeads: 2, headDim: 2 },
  };
  const positions = [1, 3, 4];
  const width = 6 + 2 * 2 + 4;
  const sites = new Float32Array(lock.layerMap.length * positions.length * width).map((_, index) => index / 7);
  const sitesBytes = Buffer.from(sites.buffer);
  const read = (layer, neuron) => ({
    layer,
    neuron,
    ids: [5, 3, 9, 1, 0, 2, 8, 7],
    probabilities: [0.4, 0.2, 0.1, 0.1, 0.05, 0.05, 0.01, 0.01],
  });
  const units = {
    block: { d_ff: 6, n_heads: 2, head_dim: 2 },
    sitesSha256: createHash("sha256").update(sitesBytes).digest("hex"),
    sitesBytes: sitesBytes.length,
    staticReads: [read(0, 1), read(1, 4)],
  };
  const check = (overrides) => unitsCaptureFailures({
    structuredHookProfile: "standard-v5",
    units,
    sitesBytes,
    lock,
    positions,
    ...overrides,
  });
  assert.deepEqual(check({}), []);
  assert.deepEqual(captureUnitsMember(units), {
    block: { d_ff: 6, n_heads: 2, head_dim: 2 },
    sites_sha256: units.sitesSha256,
    static_reads: units.staticReads,
  });
  assert.match(check({ units: undefined }).join(), /units result is missing/);
  assert.match(check({ sitesBytes: undefined }).join(), /block sites capture body is missing/);
  assert.match(check({ sitesBytes: sitesBytes.subarray(4) }).join(), /byte count/);
  assert.match(check({ units: { ...units, sitesSha256: "0".repeat(64) } }).join(), /SHA-256/);
  const nonfinite = Buffer.from(sitesBytes);
  nonfinite.writeFloatLE(Number.NaN, 0);
  assert.match(
    check({ sitesBytes: nonfinite, units: { ...units, sitesSha256: createHash("sha256").update(nonfinite).digest("hex") } }).join(),
    /not finite/,
  );
  assert.match(check({ units: { ...units, block: { d_ff: 6, n_heads: 2, head_dim: 4 } } }).join(), /lock's block spec/);
  assert.match(check({ units: { ...units, staticReads: [read(0, 1)] } }).join(), /no static read at captured layers 1/);
  assert.match(check({ units: { ...units, staticReads: [read(0, 1), read(0, 1), read(1, 0)] } }).join(), /is invalid/);
  assert.match(check({ units: { ...units, staticReads: [read(0, 6), read(1, 0)] } }).join(), /is invalid/);
  assert.match(
    check({ units: { ...units, staticReads: [{ ...read(0, 1), probabilities: [0.1, 0.4, 0, 0, 0, 0, 0, 0] }, read(1, 0)] } }).join(),
    /is invalid/,
  );
  assert.match(check({ units: { ...units, staticReads: [{ ...read(0, 1), ids: [1, 1, 2, 3, 4, 5, 6, 7] }, read(1, 0)] } }).join(), /is invalid/);
  assert.match(
    check({ units: { ...units, staticReads: Array.from({ length: MAXIMUM_STATIC_READS + 1 }, (_, n) => read(n % 2, n % 6)) } }).join(),
    /1 to 64 static reads/,
  );
  assert.deepEqual(
    unitsCaptureFailures({ structuredHookProfile: "standard-v5-residual", units: undefined, sitesBytes: undefined, lock, positions }),
    [],
  );
  assert.match(
    unitsCaptureFailures({ structuredHookProfile: "standard-v5-residual", units, sitesBytes, lock, positions }).join(),
    /carries no block sites/,
  );
  assert.match(
    unitsCaptureFailures({ structuredHookProfile: "standard-v3", units, sitesBytes, lock, positions }).join(),
    /predates capture schema 5/,
  );
}

console.log("Hosted residual capture runtime-lock checks passed");

async function runWithLock(path) {
  return await exec(process.execPath, [
    script,
    "qwen3-1.7b",
    "missing-model-directory",
    "missing-model.wasm",
    "missing-webllm.js",
    "missing-input.json",
    "missing-output",
    "--runtime-lock",
    path,
  ]);
}
