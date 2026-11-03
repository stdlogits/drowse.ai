import assert from "node:assert/strict";
import { parse } from "svelte/compiler";
import { readdir, readFile } from "node:fs/promises";
import { extname, join } from "node:path";

// The behaviour half of the old interface policy: what keeps the site's own surfaces usable with a
// keyboard, a screen reader, zoom and an unreliable runtime. The look (borders, shadows, colour,
// type, motion) is Ledger's (its DESIGN.md).
const root = new URL("..", import.meta.url).pathname;
const sourceRoots = ["src/hosted", "src/drawers", "src/lib", "hosted", "public-hosted"];
const textExtensions = new Set([".css", ".html", ".js", ".svelte", ".ts"]);

async function sourceFiles(directory) {
  let entries;
  try {
    entries = await readdir(join(root, directory), { withFileTypes: true });
  } catch (error) {
    if (error.code === "ENOENT") return [];
    throw error;
  }
  const files = [];
  for (const entry of entries) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) files.push(...await sourceFiles(path));
    else if (textExtensions.has(extname(entry.name))) files.push(path);
  }
  return files;
}

/** The violations in one file, as messages. */
export function violationsIn(file, source) {
  const found = [];
  if (extname(file) === ".svelte") {
    const visit = node => {
      if (!node || typeof node !== "object") return;
      // A component's own `title` prop is not a hover tooltip; only real elements are checked.
      if (node.type === "RegularElement" && node.attributes?.some(attribute => attribute.name === "title")) {
        found.push(`${file}: native hover title; use an accessible name, a description or a Ledger tip`);
      }
      if (node.type === "RegularElement" && node.name === "svg" && /<title[\s>]/.test(source.slice(node.start, node.end))) {
        found.push(`${file}: SVG hover title; use desc or aria-label`);
      }
      for (const value of Object.values(node)) {
        if (Array.isArray(value)) value.forEach(visit);
        else if (value && typeof value === "object") visit(value);
      }
    };
    visit(parse(source, { modern: true }).fragment);
    if (/(?:instanceof\s+Error\s*\?\s*\w+\.message|\w+\.error\?\.message)/u.test(source)) {
      found.push(`${file}: a raw runtime error can reach the interface`);
    }
  }
  if (/transition\s*:\s*all\b/i.test(source)) found.push(`${file}: transition: all`);
  if (/tabindex\s*=\s*["'{]?\s*[1-9]\d*/i.test(source)) found.push(`${file}: positive tabindex`);
  if (/user-scalable\s*=\s*no|maximum-scale\s*=\s*1(?:\.0+)?\b/i.test(source)) {
    found.push(`${file}: zoom-blocking viewport directive`);
  }
  return found;
}

// The rules catch what they claim to.
assert.equal(violationsIn("x.svelte", '<button title="Save">Save</button>').length, 1);
assert.equal(violationsIn("x.svelte", '<Card title="Save" />').length, 0, "component props are not tooltips");
assert.equal(violationsIn("x.svelte", "<svg><title>Logo</title></svg>").length, 1);
assert.equal(violationsIn("x.svelte", '<div tabindex="2"></div>').length, 1);
assert.equal(violationsIn("x.svelte", '<div tabindex="0"></div><div tabindex="-1"></div>').length, 0);
assert.equal(violationsIn("x.svelte", "<p>{error instanceof Error ? error.message : 'Failed'}</p>").length, 1);
assert.equal(violationsIn("x.css", ".a { transition: all 1s; }").length, 1);
assert.equal(violationsIn("x.html", '<meta name="viewport" content="width=device-width, maximum-scale=1">').length, 1);

const files = (await Promise.all(sourceRoots.map(sourceFiles))).flat();
assert.ok(files.length > 50, `expected the site's sources, found ${files.length} files`);
const violations = [];
for (const file of files) violations.push(...violationsIn(file, await readFile(join(root, file), "utf8")));
assert.deepEqual(violations, []);
console.log(`Interface accessibility: ${files.length} site files have no hover titles, positive tabindex, zoom blocking, transition: all or raw runtime errors`);
