import { readdir, readFile } from "node:fs/promises";
import { dirname, extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const webuiRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
// The hosted shell's own sources: src/ and the Vite root's hosted/ (main.ts, site.css).
const sourceRoots = [resolve(webuiRoot, "src"), resolve(webuiRoot, "hosted")];
// Ledger declares its tokens (and its pieces their own) in the package, installed at the workspace
// root; the shared dashboard declares its own (drowse.css). Both count as declared.
const ledgerRoot = resolve(webuiRoot, "..", "node_modules", "@stdlogits", "ledger", "src");
const dashboardRoot = resolve(webuiRoot, "..", "drowse", "webui", "src");
const sourceExtensions = new Set([".css", ".svelte", ".ts"]);

async function sourceFiles(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) {
      files.push(...await sourceFiles(path));
    } else if (sourceExtensions.has(extname(entry.name))) {
      files.push(path);
    }
  }
  return files;
}

const declarations = new Map();
const references = new Map();

for (const [root, prefix] of [[ledgerRoot, "@stdlogits/ledger/"], [dashboardRoot, "drowse:"]]) {
  for (const path of await sourceFiles(root)) {
    for (const match of (await readFile(path, "utf8")).matchAll(/(--[a-zA-Z0-9_-]+)\s*:/g)) {
      declarations.set(match[1], `${prefix}${path.slice(root.length + 1)}`);
    }
  }
}

for (const root of sourceRoots) {
  for (const path of await sourceFiles(root)) {
    const source = await readFile(path, "utf8");
    const relative = path.slice(webuiRoot.length + 1);

    for (const match of source.matchAll(/(--[a-zA-Z0-9_-]+)\s*:/g)) {
      declarations.set(match[1], relative);
    }
    for (const match of source.matchAll(/style:(--[a-zA-Z0-9_-]+)\s*=/g)) {
      declarations.set(match[1], relative);
    }
    for (const match of source.matchAll(/var\(\s*(--[a-zA-Z0-9_-]+)/g)) {
      const locations = references.get(match[1]) ?? new Set();
      locations.add(relative);
      references.set(match[1], locations);
    }
  }
}

const missing = [...references]
  .filter(([token]) => !declarations.has(token))
  .sort(([left], [right]) => left.localeCompare(right));

if (missing.length > 0) {
  console.error("Theme check failed: referenced CSS custom properties lack a source declaration.");
  for (const [token, locations] of missing) {
    console.error(`  ${token}: ${[...locations].sort().join(", ")}`);
  }
  process.exitCode = 1;
} else {
  console.log(
    `Theme check passed: ${references.size} referenced custom properties, `
      + `${declarations.size} declared.`,
  );
}
