// The Ledger ratchet for drowse.ai's own tree, held at zero: the hosted shell (src/hosted), the
// browser drawers (src/drawers) and the Vite root (hosted/) make no departure from Ledger (DESIGN.md
// in @stdlogits/ledger: flat and borderless, Wix Madefor only, no capitals, one curve, nothing
// looping, colours only from tokens and mixed in oklch, the depth accent only as a fill).
//
// The rules are drowse's (drowse/webui/scripts/check-ledger.mjs, which ratchets the shared tree):
// this script reads its categories and its per-file counter from the pinned submodule, so a pin
// bump brings the rules with it. If that script is reorganised and the counter can't be found,
// this check fails rather than passing on nothing.
//
//   node scripts/check-ledger.mjs
import { readdir, readFile } from "node:fs/promises";
import { extname, join, relative, resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const webui = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const ROOTS = ["src/hosted", "src/drawers", "hosted"];
const rulesPath = resolve(webui, "..", "drowse", "webui", "scripts", "check-ledger.mjs");

/** drowse's categories and countFile, cut from its script and loaded as a module of their own. */
async function loadRules() {
  const source = await readFile(rulesPath, "utf8");
  const start = source.indexOf("export const CATEGORIES");
  const end = source.indexOf("async function measure");
  if (start < 0 || end < start || !source.includes("function countFile(")) {
    throw new Error(`Ledger check: could not find the rules in ${relative(webui, rulesPath)}; update scripts/check-ledger.mjs to match it.`);
  }
  const module = [
    'import { readdir } from "node:fs/promises";',
    'import { extname, join } from "node:path";',
    source.slice(start, end),
    "export { countFile };",
  ].join("\n");
  return import(`data:text/javascript;base64,${Buffer.from(module).toString("base64")}`);
}

async function sourceFiles(directory) {
  const out = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) out.push(...await sourceFiles(path));
    else if ([".css", ".svelte", ".ts"].includes(extname(entry.name))) out.push(path);
  }
  return out.sort();
}

const { CATEGORIES, countFile } = await loadRules();
const departures = [];
let files = 0;
for (const root of ROOTS) {
  for (const path of await sourceFiles(resolve(webui, root))) {
    files += 1;
    const counts = countFile(await readFile(path, "utf8"), extname(path));
    for (const [key, n] of Object.entries(counts)) {
      if (n > 0) departures.push(`  ${relative(webui, path)}: ${key} ${n} (${CATEGORIES[key]})`);
    }
  }
}

if (departures.length) {
  console.error(`Ledger check failed: drowse.ai's tree departs from Ledger.\n${departures.join("\n")}\nWrite it in Ledger (DESIGN.md in @stdlogits/ledger) instead.`);
  process.exit(1);
}
console.log(`Ledger check passed: ${files} files in ${ROOTS.join(", ")}, no departures.`);
