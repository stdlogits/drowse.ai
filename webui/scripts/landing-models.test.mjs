import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createServer } from "vite";

// The landing roster, sizes, and requirement copy are hand-maintained in Landing.svelte so the
// landing chunk does not bundle the runtime lock. This test keeps them in step with the lock.
const lock = JSON.parse(await readFile(new URL("../../browser-runtime/runtime-lock.json", import.meta.url), "utf8"));
const landingSource = await readFile(new URL("../src/hosted/ui/Landing.svelte", import.meta.url), "utf8");
const recording = JSON.parse(await readFile(new URL("../src/hosted/data/neighbor-example.json", import.meta.url), "utf8"));

const server = await createServer({
  appType: "custom", logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
try {
  const { render } = await server.ssrLoadModule("svelte/server");
  const { default: Landing } = await server.ssrLoadModule("/src/hosted/ui/Landing.svelte");
  const html = render(Landing).body;

  // Every locked model is listed once, in the right group, linking to its catalog variant.
  const groups = Object.fromEntries([...html.matchAll(/<section class="model-group[^"]*" aria-labelledby="([^"]+)-title"[\s\S]*?<\/section>/g)]
    .map(([section, id]) => [id, [...section.matchAll(/href="\/app\?choose=1&amp;model=([^"]+)"/g)].map(match => decodeURIComponent(match[1]))]));
  const expected = { "chat-models": [], "base-models": [] };
  for (const model of lock.models) expected[model.modelType === "base" ? "base-models" : "chat-models"].push(`${model.id}-${model.quantization}`);
  for (const [group, ids] of Object.entries(expected)) {
    assert.deepEqual([...groups[group]].sort(), [...ids].sort(), `${group} matches browser-runtime/runtime-lock.json`);
  }
  assert.ok(html.includes("Gemma 3 270M"), "the smallest locked chat model is listed");

  // One storage range (signed catalog sequence 9; the lock has no sizes) is used everywhere it appears.
  const range = landingSource.match(/const storageRange = "(\d\.\d to \d\.\d GB)"/)?.[1];
  assert.ok(range, "the storage range is declared once");
  assert.ok(html.includes(`${range} per model`), `requirements state ${range}`);
  assert.ok(html.includes(`Each model is ${range}`), `FAQ states ${range}`);
  const modelsSection = html.match(/<section class="models[\s\S]*?<\/section>\s*<section class="guide/)[0];
  assert.doesNotMatch(modelsSection, /GB/, "the model roster stays free of download sizes");
  assert.match(html, /Gemma 3 270M runs at full precision, so its download is larger than Gemma 3 1B’s\.|Gemma 3 270M runs at full precision, so its download is larger than Gemma 3 1B&#39;s\.|Gemma 3 270M runs at full precision, so its download is larger than Gemma 3 1B's\./);

  // The prerendered requirement is neutral and visible; the WebGPU probe only replaces it.
  assert.match(html, /<span class="requirement[^"]*" data-webgpu="unknown">Needs a browser with WebGPU, such as current Chrome, Edge or Safari 26 or later\.<\/span>/);
  assert.doesNotMatch(html, /Chrome or Safari · compatible device required/);
  assert.match(landingSource, /gpu\.requestAdapter\(\)/);
  assert.doesNotMatch(landingSource, /import[^;]*runtime\/capabilities/, "the landing probe does not pull in the app's capability checker");

  // Visible guide sections, with native disclosure for questions.
  for (const heading of ["What you need.", "Words you&#39;ll see.", "Common questions.", "Need more models?"]) {
    assert.ok(html.includes(heading) || html.includes(heading.replace("&#39;", "'")), `${heading} section is visible`);
  }
  for (const term of ["Influence", "Strength (α)", "Word insights (R lens)", "S lens", "Neurons and heads", "Feature insights (SAE)", "Loom", "Base model"]) {
    assert.ok(new RegExp(`<dt[^>]*>${RegExp.escape(term)}</dt>`).test(html), `${term} is defined`);
  }
  assert.ok((html.match(/<details[\s>]/g) ?? []).length >= 8, "questions use native details elements");
  assert.doesNotMatch(html, /application\/ld\+json|FAQPage/, "structured data is not added by the landing component");
  assert.match(html, /href="\/privacy"[^>]*>What leaves your device/);
  assert.match(html, /href="https:\/\/github\.com\/[^"]+#installation"[^>]*>Install with pip/);
  assert.match(html, /<main id="main" tabindex="-1"[\s>]/);
  assert.doesNotMatch(landingSource, /scrollbar-width: none|::-webkit-scrollbar\) \{\s*display: none/);

  // The hero is static Ledger: type and one real reading, no shader, video or loop.
  assert.doesNotMatch(landingSource, /HeroShader|<canvas|<video|requestAnimationFrame\(\s*(?:render|loop|tick)/);
  assert.doesNotMatch(landingSource, /animation:[^;]*infinite/, "nothing on the landing loops");
  // Every number in the hero's reading is the recording's: the run steered toward welcoming at 0.10,
  // token 22, its five likeliest next tokens and the one it wrote.
  const run = recording.runs.find((candidate) => candidate.alpha === 0.1);
  const written = run.tokens[22];
  const context = run.tokens.slice(14, 22).map((token) => token.text).join("").replace("'", "’");
  assert.match(landingSource, new RegExp(`const heroContext = "${RegExp.escape(context)}";`));
  assert.match(landingSource, /const heroModel = "Gemma 3 4B";/);
  assert.equal(recording.modelId, "gemma3-4b-instruct");
  assert.match(landingSource, /const heroSteering = "steered toward welcoming, 0\.10";/);
  const heroNext = [...landingSource.matchAll(/\{ token: "([^"]+)", p: ([\d.]+)(, wrote: true)? \}/g)]
    .map(([, token, p, wrote]) => ({ token, p: Number(p), wrote: Boolean(wrote) }));
  assert.deepEqual(heroNext.map((next) => next.token), written.topAlts.map((alt) => alt.text));
  for (const [i, next] of heroNext.entries()) {
    assert.ok(Math.abs(next.p - Math.exp(written.topAlts[i].logprob)) < 5e-5, `${next.token} is ${Math.exp(written.topAlts[i].logprob)}`);
    assert.equal(next.wrote, next.token === written.text, `${next.token} ${next.wrote ? "is" : "is not"} the token written`);
  }
  assert.match(html, /<figure class="reading[^"]*"/);
  assert.match(html, /We’d love for you to feel <span class="slot[^"]*"[^>]*>[\s\S]*?at[\s\S]*?<\/span><\/p>/);
  assert.equal((html.match(/<button type="button" class="row[^"]*" aria-pressed="true"/g) ?? []).length, 1, "the written token starts chosen");
} finally {
  await server.close();
}
console.log("Landing: model roster matches the runtime lock, sizes and requirements agree, guide sections render, and the hero reading matches its recording");
