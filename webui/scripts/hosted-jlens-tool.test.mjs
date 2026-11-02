#!/usr/bin/env node

import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { promisify } from "node:util";
import { createServer } from "vite";
import { selectHostedJlensLayers } from "./hosted-instrument-layers.mjs";

const exec = promisify(execFile);
const webuiRoot = resolve(import.meta.dirname, "..");
const repositoryRoot = resolve(webuiRoot, "..");
const builder = resolve(import.meta.dirname, "build-hosted-jlens.mjs");
const fitter = resolve(repositoryRoot, "browser-runtime/fit_hosted_jlens.py");

const help = await exec(process.execPath, [builder, "--help"]);
assert.match(help.stdout, /MODEL_ID MODEL_DIR MODEL\.wasm WEBLLM\.js CORPUS\.json WORDS\.json OUTPUT_DIR/);
assert.match(help.stdout, /--source-layers/);
assert.match(await readFile(builder, "utf8"), /\.\.\.executionProfiles/);
assert.match(await readFile(builder, "utf8"), /resolve\(options\.python\)/);

assert.deepEqual(
  selectHostedJlensLayers(Array.from({ length: 32 }, (_, index) => index), 960),
  [0, 4, 9, 13, 17, 21, 26, 30],
);
assert.deepEqual(
  selectHostedJlensLayers(Array.from({ length: 28 }, (_, index) => index), 2048),
  [0, 4, 7, 11, 15, 19, 22, 26],
);
const qwen4Layers = selectHostedJlensLayers(
  Array.from({ length: 36 }, (_, index) => index),
  2560,
);
assert.equal(qwen4Layers.length, 8);
assert.equal(qwen4Layers[0], 0);
assert.equal(qwen4Layers.at(-1), 34);
assert.equal(new Set(qwen4Layers).size, qwen4Layers.length);
assert.throws(
  () => selectHostedJlensLayers([0], 128),
  /source layer before the final residual/,
);

const vite = await createServer({
  root: webuiRoot,
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  optimizeDeps: { noDiscovery: true, include: [] },
  server: { middlewareMode: true, watch: null },
});
const python = process.env.PYTHON ?? "python3";
const pythonEnv = { ...process.env, PYTHONPATH: resolve(repositoryRoot, "drowse") };
try {
  const transformed = await vite.transformRequest("/scripts/hosted-jlens-capture.worker.ts");
  assert.ok(transformed?.code.includes("structuredHookProfile: event.data.structuredHookProfile"));
  assert.ok(transformed.code.includes("thinkingProfile: event.data.thinkingProfile"));
  assert.ok(transformed.code.includes("maxComputeWorkgroupSizeX"));
  assert.ok(transformed.code.includes("maxComputeInvocationsPerWorkgroup"));

  // The lens v7 reader rule and drowse's lens identity, against a manifest
  // drowse itself writes: an R lens at out and mid, with the fit's split-half
  // differences (floats Python and JavaScript spell differently).
  const { lensIdentity, lensManifestSites } = await vite.ssrLoadModule(
    "/src/hosted/runtime/browserInstrumentPacks.ts",
  );
  const stage = await mkdtemp(join(tmpdir(), "drowse-lens-v7-"));
  try {
    const written = await exec(python, ["-c", `
import json, sys, torch
from pathlib import Path
from drowse.lens.jlens import JacobianLens
from drowse.lens.key import LensKey
from drowse.lens.store import LensProvenance, lens_identity, save_lens_at
lens = JacobianLens(
    {0: torch.eye(2), 1: torch.eye(2)}, mid={1: torch.eye(2)},
    n_prompts=4, d_model=2, key=LensKey("R"),
)
root = Path(sys.argv[1])
save_lens_at(lens, root, LensProvenance(
    corpus_spec="fixture-\\u00e9\\u2028\\U0001f600\\x7f", corpus_sha256="1" * 64, seq_len=32,
    dim_batch=8,
), split_half={"out": {0: 1e-05, 1: 0.0}, "mid": {1: 1e16}})
manifest = json.loads((root / "manifest.json").read_text())
print(lens_identity(manifest))
# _validate_manifest accepts an int split_half, which json.dumps spells "0".
manifest["sites"]["out"]["split_half"]["1"] = 0
(root / "manifest-int.json").write_text(json.dumps(manifest, indent=2))
print(lens_identity(manifest))
`, stage], { cwd: repositoryRoot, env: pythonEnv });
    const [identity, intIdentity] = written.stdout.trim().split("\n");
    const manifestText = await readFile(join(stage, "manifest.json"), "utf8");
    const manifest = JSON.parse(manifestText);
    const sites = lensManifestSites(manifest);
    assert.deepEqual([...sites.keys()], ["out", "mid"]);
    assert.deepEqual(sites.get("out").layers, [0, 1]);
    assert.deepEqual(sites.get("out").splitHalf, { 0: 1e-5, 1: 0 });
    assert.deepEqual(sites.get("mid").layers, [1]);
    assert.deepEqual(sites.get("mid").splitHalf, { 1: 1e16 });
    assert.equal(lensIdentity(manifestText), identity);
    const intText = await readFile(join(stage, "manifest-int.json"), "utf8");
    assert.notEqual(intIdentity, identity);
    assert.equal(lensIdentity(intText), intIdentity);
    assert.equal(lensManifestSites(JSON.parse(intText)).get("out").splitHalf[1], 0);
    assert.throws(() => lensIdentity(`${manifestText}}`), /not JSON Python reads/u);

    const withSite = (site, entry) => ({
      ...manifest,
      sites: { ...manifest.sites, [site]: { ...manifest.sites[site], ...entry } },
    });
    const withoutSplitHalf = structuredClone(manifest);
    delete withoutSplitHalf.sites.out.split_half;
    delete withoutSplitHalf.sites.mid;
    assert.equal(lensManifestSites(withoutSplitHalf).get("out").splitHalf, null);
    const odd = {
      tensor_files: { 0: "out.odd.partial.layer-0.safetensors", 1: "out.odd.partial.layer-1.safetensors" },
      tensor_sha256: { 0: "a".repeat(64), 1: "b".repeat(64) },
    };
    for (const [label, malformed] of [
      ["odd shards outside a checkpoint", withSite("out", { odd })],
      ["split_half missing a layer", withSite("out", { split_half: { 0: 0.25 } })],
      ["negative split_half", withSite("out", { split_half: { 0: 0.25, 1: -1 } })],
      ["non-finite split_half", withSite("mid", { split_half: { 1: null } })],
      ["unknown site", { ...manifest, sites: { ...manifest.sites, in: manifest.sites.out } }],
      ["unknown site key", withSite("mid", { source_layers: [1] })],
      ["digest keys off the layers", withSite("out", { tensor_sha256: { 0: "a".repeat(64) } })],
      ["unsafe shard name", withSite("out", { tensor_files: { 0: "../x", 1: "y" } })],
      ["no sites", { ...manifest, sites: {} }],
    ]) {
      assert.equal(lensManifestSites(malformed), null, label);
    }
  } finally {
    await rm(stage, { recursive: true, force: true });
  }
} finally {
  await vite.close();
}

await assert.rejects(
  exec(process.execPath, [
    builder,
    "model", "directory", "model.wasm", "webllm.js", "corpus.json", "words.json", "output",
    "--source-layers", "1,nope",
  ]),
  /source layers must be a comma-separated integer list/,
);

const pythonHelp = await exec(python, [fitter, "--help"], {
  cwd: repositoryRoot,
  env: pythonEnv,
});
assert.match(pythonHelp.stdout, /R-family lens pack/);
assert.match(pythonHelp.stdout, /--kind \{R,J\}/);
assert.match(pythonHelp.stdout, /--dim-batch/);
assert.match(pythonHelp.stdout, /--source-layers/);
assert.match(pythonHelp.stdout, /--max-residual-error/);

console.log("Hosted lens production-tool and v7 reader checks passed");
