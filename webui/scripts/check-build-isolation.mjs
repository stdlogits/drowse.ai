import { readdir, readFile } from "node:fs/promises";
import { resolve } from "node:path";

// The dashboard bundle `drowse serve` mounts has its own isolation check in the
// drowse repository; this one requires what the hosted output must carry.
const [rootArg] = process.argv.slice(2);
if (!rootArg) {
  throw new Error("usage: node check-build-isolation.mjs OUTPUT_DIR");
}

const root = resolve(rootArg);
const paths = [];
async function walk(directory, prefix = "") {
  const entries = await readdir(directory, { withFileTypes: true });
  for (const entry of entries) {
    const relative = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.isDirectory()) await walk(resolve(directory, entry.name), relative);
    else paths.push(relative);
  }
}
await walk(root);

const has = (name) => paths.includes(name);
for (const required of [
  "index.html",
  "_headers",
  "manifest.webmanifest",
  "sw.js",
  "icons/drowse-192.png",
  "icons/drowse-512.png",
  "icons/drowse-maskable-512.png",
  "wasm/fitting-kernel.json",
  "wasm/drowse_fitting_wasm.js",
  "wasm/drowse_fitting_wasm_bg.wasm",
]) {
  if (!has(required)) throw new Error(`hosted build is missing ${required}`);
}
const headers = await readFile(resolve(root, "_headers"), "utf8");
for (const header of [
  "Cross-Origin-Opener-Policy: same-origin",
  "Cross-Origin-Embedder-Policy: require-corp",
  "Content-Security-Policy:",
]) {
  if (!headers.includes(header)) throw new Error(`hosted headers are missing ${header}`);
}
// Neuron Lens's image and sound packs are .bin files on purpose (scripts/neurons/media.py); any other
// .bin is a fixture's weights.
const mediaPack = /^neurons-data\/[^/]+\/media\/(?:image|audio)\/p\d+\.bin$/;
const fixtureAssets = paths.filter((path) => /fixture/i.test(path) || (/\.bin$/i.test(path) && !mediaPack.test(path)));
if (fixtureAssets.length) {
  throw new Error(`development fixture assets leaked into hosted build: ${fixtureAssets.join(", ")}`);
}
const scripts = paths.filter((path) => path.endsWith(".js"));
const source = (await Promise.all(
  scripts.map((path) => readFile(resolve(root, path), "utf8")),
)).join("\n");
for (const marker of [
  "development-fixture",
  "DeterministicFakeRuntime",
  "fixtureHostedRuntime",
  "fixture.invalid",
]) {
  if (source.includes(marker)) {
    throw new Error(`development fixture marker leaked into hosted build: ${marker}`);
  }
}
for (const marker of [
  "SAE training tokens",
  "J-lens corpus prompts",
  "Blocks generation; uses FineWeb-Edu",
  "generation paused during model fitting",
  "Create on this device",
  "drowse lens fit",
  "train_hosted_sae.py",
  "fit_hosted_jlens.py",
]) {
  if (source.includes(marker)) {
    throw new Error(`Python-only instrument authoring leaked into hosted build: ${marker}`);
  }
}

console.log(`hosted build isolation passed (${paths.length} files)`);
