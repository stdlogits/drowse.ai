#!/usr/bin/env node
// drowse 6.0's default concepts as hosted core sources: every bundled manifold tagged in a default
// probe category (drowse `session.py` PROBE_CATEGORIES: epistemic, alignment, register, cultural;
// the one-node deceptive, refusing and manipulative among them) copied from the submodule into
// browser-runtime/release-inputs/core-manifolds/<name>/, which the core pack builder fits in the
// browser runtime for each model (`build-hosted-whitener.mjs --manifold-source`). Rerun after a
// drowse pin bump; `--check` fails when the checked-in sources differ from the pinned drowse.

import { createHash } from "node:crypto";
import { mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const DEFAULT_PROBE_CATEGORIES = Object.freeze(["epistemic", "alignment", "register", "cultural"]);
const LICENSE = "AGPL-3.0-or-later";

const repositoryRoot = fileURLToPath(new URL("../..", import.meta.url));
const bundledRoot = resolve(repositoryRoot, "drowse/drowse/data/manifolds");
const targetRoot = resolve(repositoryRoot, "browser-runtime/release-inputs/core-manifolds");

/** `{name: {path: text}}` for every default concept, as the release inputs hold them. */
export async function coreConceptSources() {
  const baselineBytes = await readFile(resolve(repositoryRoot, "drowse/drowse/data/baseline_prompts.json"));
  const baselinePrompts = JSON.parse(baselineBytes.toString("utf8"));
  const sources = {};
  for (const name of (await readdir(bundledRoot)).sort()) {
    const manifest = JSON.parse(await readFile(resolve(bundledRoot, name, "manifold.json"), "utf8"));
    if (!manifest.tags?.some((tag) => DEFAULT_PROBE_CATEGORIES.includes(tag))) continue;
    if (manifest.source !== "bundled" || manifest.fit_mode !== "pca") {
      throw new Error(`bundled default concept ${name} is not a bundled PCA manifold`);
    }
    const files = {};
    files["manifold.json"] = `${JSON.stringify(manifest, null, 2)}\n`;
    for (const [index, node] of manifest.nodes.entries()) {
      const file = `nodes/${String(index).padStart(2, "0")}_${node.label}.json`;
      const statements = JSON.parse(await readFile(resolve(bundledRoot, name, file), "utf8"));
      if (!Array.isArray(statements) || statements.length !== baselinePrompts.length) {
        throw new Error(`${name} node ${node.label} does not answer every baseline prompt`);
      }
      files[file] = `${JSON.stringify(statements, null, 2)}\n`;
    }
    files["provenance.json"] = `${JSON.stringify({
      schema_version: 1,
      artifact: `default/${name}`,
      authorship: `drowse's bundled default concept (drowse/data/manifolds/${name}), copied unchanged from the pinned drowse submodule by webui/scripts/sync-core-concepts.mjs.`,
      license: LICENSE,
      source: "bundled",
      baseline_prompts_path: "drowse/data/baseline_prompts.json",
      baseline_prompts_sha256: createHash("sha256").update(baselineBytes).digest("hex"),
      responses_per_node: baselinePrompts.length,
    }, null, 2)}\n`;
    files["LICENSE.txt"] = `SPDX-License-Identifier: ${LICENSE}\n\nPart of drowse (https://github.com/stdlogits/drowse), licensed under the GNU Affero General Public License v3.0 or later.\n`;
    sources[name] = files;
  }
  return sources;
}

async function main(args) {
  const check = args.includes("--check");
  if (args.some((arg) => arg !== "--check")) throw new Error("usage: sync-core-concepts.mjs [--check]");
  const sources = await coreConceptSources();
  const drift = [];
  for (const [name, files] of Object.entries(sources)) {
    const directory = resolve(targetRoot, name);
    if (check) {
      for (const [path, text] of Object.entries(files)) {
        const current = await readFile(resolve(directory, path), "utf8").catch(() => null);
        if (current !== text) drift.push(`${name}/${path}`);
      }
      continue;
    }
    await rm(directory, { recursive: true, force: true });
    await mkdir(resolve(directory, "nodes"), { recursive: true });
    for (const [path, text] of Object.entries(files)) await writeFile(resolve(directory, path), text);
  }
  if (drift.length > 0) {
    throw new Error(`core concept sources differ from the pinned drowse: ${drift.join(", ")}; run sync-core-concepts.mjs`);
  }
  process.stdout.write(`${check ? "checked" : "wrote"} ${Object.keys(sources).length} default concepts: ${Object.keys(sources).join(", ")}\n`);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
  main(process.argv.slice(2)).catch((error) => {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  });
}
