#!/usr/bin/env node
// Sizes of a model's staged packs (`browser-runtime/build-hosted-packs.sh` output): per pack the
// total, what installs with it and what is fetched on first read, by the browser's own residency
// rule (`packFileResidency`).
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";
import { runtimeIdentityRecord } from "./hosted-runtime-identity.mjs";

const [input, modelId, ...rest] = process.argv.slice(2);
if (!input || !modelId || rest.length > 2 || (rest.length && rest[0] !== "--runtime-lock")) {
  throw new Error("usage: hosted-pack-sizes.mjs STAGED_DIR MODEL_ID [--runtime-lock PATH]");
}
const lockPath = rest[1] ?? fileURLToPath(new URL("../../browser-runtime/runtime-lock.json", import.meta.url));
const record = runtimeIdentityRecord(JSON.parse(await readFile(lockPath, "utf8")), modelId);
const server = await createServer({
  root: fileURLToPath(new URL("..", import.meta.url)),
  configFile: false,
  optimizeDeps: { noDiscovery: true, include: [] },
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
const MiB = 1024 * 1024;
try {
  const { packFileResidency, packLayerShard } = await server.ssrLoadModule("/src/hosted/runtime/optionalPackCompatibility.ts");
  const rows = {};
  for (const kind of ["jlens", "slens", "units"]) {
    let manifest;
    try {
      manifest = JSON.parse(await readFile(resolve(input, kind, "hosted-pack-artifacts.json"), "utf8"));
    } catch {
      continue;
    }
    const pack = { kind };
    const sum = (files) => files.reduce((total, file) => total + file.bytes, 0);
    const install = manifest.files.filter((file) => packFileResidency(pack, record, file.path) === "install");
    const layers = manifest.files.filter((file) => packLayerShard(pack, file.path) !== null);
    rows[kind] = {
      totalMiB: +(sum(manifest.files) / MiB).toFixed(1),
      installMiB: +(sum(install) / MiB).toFixed(1),
      onDemandMiB: +((sum(manifest.files) - sum(install)) / MiB).toFixed(1),
      layerFiles: layers.length,
      largestLayerMiB: +(Math.max(0, ...layers.map((file) => file.bytes)) / MiB).toFixed(2),
    };
  }
  process.stdout.write(`${JSON.stringify({ modelId, runtimeIdentitySha256: record.runtimeIdentitySha256, packs: rows }, null, 2)}\n`);
} finally {
  await server.close();
}
