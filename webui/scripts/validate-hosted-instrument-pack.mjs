#!/usr/bin/env node

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { createServer } from "vite";
import { validatePackDirectory } from "./build-hosted-catalog.mjs";
import { validateReleaseInstrumentPack } from "./hosted-instrument-release-policy.mjs";
import { lockedRuntimeIdentity } from "./hosted-runtime-identity.mjs";

const KINDS = ["sae", "jlens", "slens", "units"];

const arguments_ = process.argv.slice(2);
const values = [];
const companions = [];
let coreStatistics = null;
let release = false;
let runtimeLockInput = null;
for (let index = 0; index < arguments_.length; index += 1) {
  const argument = arguments_[index];
  if (argument === "--core-statistics") {
    const value = arguments_[++index];
    if (!value || value.startsWith("--")) throw new Error("--core-statistics requires a directory");
    coreStatistics = resolve(value);
    continue;
  }
  if (argument === "--with") {
    const value = arguments_[++index];
    if (!value || value.startsWith("--")) throw new Error("--with requires a pack directory");
    companions.push(resolve(value));
    continue;
  }
  if (argument === "--release") {
    if (release) throw new Error("--release may be supplied only once");
    release = true;
    continue;
  }
  if (argument === "--runtime-lock") {
    if (runtimeLockInput !== null) {
      throw new Error("--runtime-lock may be supplied only once");
    }
    const value = arguments_[++index];
    if (!value || value.startsWith("--")) {
      throw new Error("--runtime-lock requires a path");
    }
    runtimeLockInput = value;
    continue;
  }
  if (argument.startsWith("--")) throw new Error(`unexpected argument ${argument}`);
  values.push(argument);
}
const [modelId, kind, input] = values;
if (!modelId || !KINDS.includes(kind) || !input || values.length !== 3) {
  throw new Error(
    "usage: validate-hosted-instrument-pack.mjs MODEL_ID sae|jlens|slens|units PACK_DIR " +
    "[--with PACK_DIR]... [--core-statistics DIR] [--release] [--runtime-lock PATH]\n" +
    "  --with loads another pack beside it (a units pack is read through the R and S packs it binds)\n" +
    "  --core-statistics names the directory holding the core pack's neutral-statistics.* (units bind it)",
  );
}

const webuiRoot = resolve(import.meta.dirname, "..");
const repositoryRoot = resolve(webuiRoot, "..");
const directory = resolve(input);
const runtimeLockPath = runtimeLockInput === null
  ? resolve(repositoryRoot, "browser-runtime/runtime-lock.json")
  : resolve(runtimeLockInput);
const runtimeLock = JSON.parse(
  await readFile(runtimeLockPath, "utf8"),
);
const lock = runtimeLock.models.find((entry) => entry.id === modelId);
if (!lock) throw new Error(`unknown runtime-lock model ${modelId}`);
// Each kind's one record: a lens pack's manifest.json, the units pack's units.json.
const RECORD = { sae: "manifest.json", jlens: "manifest.json", slens: "manifest.json", units: "units.json" };
const revision = "a".repeat(40);

async function packEntry(packDirectory, packKind) {
  const packManifest = await validatePackDirectory(packDirectory);
  if (
    packManifest.files.some((file) => !file.path.includes("/") || !file.path.includes(`${packKind}/`)) ||
    packManifest.files.filter((file) => file.path.endsWith(`/${packKind}/${RECORD[packKind]}`)).length !== 1
  ) {
    throw new Error(`${packKind} pack files must share a nested ${packKind}/ closure with one ${RECORD[packKind]}`);
  }
  const packFiles = packManifest.files.map((file) => ({
    ...file,
    role: "instrument",
    revision,
    url: `https://huggingface.co/drowse/local/resolve/${revision}/${file.path}`,
  }));
  // Every file, on-demand layers included: the registry reads them as a fetch would.
  const packArtifacts = await Promise.all(packFiles.map(async (entry) => ({
    manifest: entry,
    file: new File([await readFile(resolve(packDirectory, entry.path))], entry.path),
  })));
  return { files: packFiles, artifacts: packArtifacts };
}

async function companionKind(packDirectory) {
  const packManifest = await validatePackDirectory(packDirectory);
  const kinds = KINDS.filter((candidate) =>
    packManifest.files.some((file) => file.path.endsWith(`/${candidate}/${RECORD[candidate]}`)));
  if (kinds.length !== 1) throw new Error(`${packDirectory} is not one instrument pack`);
  return kinds[0];
}

const manifest = await validatePackDirectory(directory);
const { files, artifacts } = await packEntry(directory, kind);
const runtimeIdentity = lockedRuntimeIdentity(runtimeLock, modelId);
let releaseQuality = null;
if (release) {
  const instrumentManifest = JSON.parse(await readFile(resolve(
    directory,
    manifest.files.find((file) => file.path.endsWith(`/${kind}/${RECORD[kind]}`)).path,
  ), "utf8"));
  const vocabularyEntry = kind === "jlens"
    ? manifest.files.find((file) => file.path.endsWith("/jlens/browser-vocabulary.json"))
    : null;
  releaseQuality = await validateReleaseInstrumentPack({
    directory,
    kind,
    instrumentManifest,
    vocabularyManifest: vocabularyEntry === null || vocabularyEntry === undefined
      ? null
      : JSON.parse(await readFile(resolve(directory, vocabularyEntry.path), "utf8")),
    runtimeLock,
    modelLock: lock,
  });
}
const vite = await createServer({
  root: webuiRoot,
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
try {
  const [{ BrowserInstrumentRegistry }, { runtimeIdentitySha256 }] = await Promise.all([
    vite.ssrLoadModule("/src/hosted/runtime/browserInstrumentPacks.ts"),
    vite.ssrLoadModule("/src/lib/runtime/catalog.ts"),
  ]);
  const fingerprint = runtimeIdentitySha256(runtimeIdentity);
  const registry = await BrowserInstrumentRegistry.load({
    model: { id: modelId },
    variant: {
      id: `${modelId}-${lock.quantization}`,
      runtimeIdentity,
      runtimeIdentitySha256: fingerprint,
    },
    requiredCorePack: { files: [] },
    contextTokens: lock.contextProfiles[0],
    adapter: {},
    artifacts: coreStatistics === null ? [] : await Promise.all(
      ["neutral-statistics.json", "neutral-statistics.safetensors"].map(async (path) => {
        const bytes = await readFile(resolve(coreStatistics, path));
        return {
          manifest: {
            path, role: "core_pack", revision, bytes: bytes.byteLength,
            sha256: createHash("sha256").update(bytes).digest("hex"),
            url: `https://huggingface.co/drowse/local/resolve/${revision}/${path}`,
          },
          file: new File([bytes], path),
        };
      }),
    ),
    optionalPacks: [
      {
        pack: {
          id: `${modelId}-${kind}`,
          kind,
          displayName: `${modelId} ${kind}`,
          runtimeIdentitySha256: fingerprint,
          files,
        },
        artifacts,
      },
      ...await Promise.all(companions.map(async (companion) => {
        const companionPackKind = await companionKind(companion);
        const entry = await packEntry(companion, companionPackKind);
        return {
          pack: {
            id: `${modelId}-${companionPackKind}`,
            kind: companionPackKind,
            displayName: `${modelId} ${companionPackKind}`,
            runtimeIdentitySha256: fingerprint,
            files: entry.files,
          },
          artifacts: entry.artifacts,
        };
      })),
    ],
    activationSpool: {},
    signal: new AbortController().signal,
    onDeviceLost() {},
  });
  const descriptor = registry.descriptor();
  const signal = new AbortController().signal;
  const square = runtimeIdentity.hiddenSize * runtimeIdentity.hiddenSize;
  let readBack = null;
  // Each family's layers read back through the loader: the first, last and middle layer of
  // every site, as a first read fetches them.
  const lensReadBack = async (family) => {
    const read = {};
    for (const site of ["out", "mid"]) {
      const layers = registry.lensSiteLayers(family, site);
      const probe = [...new Set([layers[0], layers[Math.floor(layers.length / 2)], layers.at(-1)])]
        .filter((layer) => layer !== undefined);
      for (const layer of probe) {
        const matrix = await registry.lensMatrix(family, site, layer, signal);
        assert.equal(matrix?.length, square, `${family}:${site} L${layer}`);
      }
      read[site] = { layers: layers.length, checked: probe };
    }
    return read;
  };
  if (kind === "sae") {
    assert.equal(registry.capabilities().sae, true);
    assert.ok(descriptor.sae?.features > 0);
    registry.saeFeature("0");
    registry.saeFeature(String(descriptor.sae.features - 1));
  } else if (kind === "jlens") {
    assert.equal(registry.capabilities().jlens, true);
    assert.ok(descriptor.jlens?.words.length > 0);
    registry.jlensToken(descriptor.jlens.words[0].word);
    readBack = { readout: descriptor.jlens.layers, lens: await lensReadBack("R"), identity: registry.lensIdentity("R") };
  } else if (kind === "slens") {
    assert.equal(registry.capabilities().slens, true);
    const statistics = await registry.embeddingStatistics(signal);
    assert.equal(statistics.meanE.length, runtimeIdentity.hiddenSize);
    assert.equal(statistics.sigmaE.length, square);
    readBack = {
      lens: await lensReadBack("S"),
      identity: registry.lensIdentity("S"),
      mask: statistics.maskDescription,
      tokenizerFingerprint: statistics.tokenizerFingerprint,
    };
  } else {
    const units = registry.unitsPack();
    assert.ok(units !== null, "the units pack did not load");
    // Read through the R source the units pack was built over.
    for (const row of registry.jlensSources()) {
      if (row.summary?.family !== "R") continue;
      await registry.activateJlensSource(row.source);
      if (registry.lensIdentity("R") === units.binding.rLensIdentity) break;
    }
    const nullBasis = await registry.unitsNullBasis(signal);
    assert.equal(nullBasis.length, units.dModel * units.nullRank);
    const layers = {};
    for (const layer of [...new Set([0, Math.floor(units.layers / 2), units.layers - 1])]) {
      const reads = await registry.unitsLayer("reads", layer, signal);
      const contrast = await registry.unitsLayer("contrast", layer, signal);
      for (const tensor of [...reads.values(), ...contrast.values()]) {
        assert.equal(tensor.data.length, tensor.shape.reduce((total, size) => total * size, 1));
      }
      layers[layer] = { reads: reads.size, contrast: contrast.size };
    }
    readBack = { units: { ...units, layers: units.layers }, checked: layers };
  }
  process.stdout.write(`${JSON.stringify({
    modelId,
    kind,
    runtimeLockSource: runtimeLockInput === null ? "repository" : "override",
    files: files.length,
    bytes: files.reduce((sum, file) => sum + file.bytes, 0),
    closureSha256: digest(files.map(({ path, bytes, sha256 }) => ({ path, bytes, sha256 }))),
    releaseQuality,
    descriptor: kind === "sae" ? descriptor.sae : kind === "jlens" ? descriptor.jlens : null,
    readBack,
  }, null, 2)}\n`);
} finally {
  await vite.close();
}

function digest(value) {
  return createHash("sha256").update(JSON.stringify(value)).digest("hex");
}
