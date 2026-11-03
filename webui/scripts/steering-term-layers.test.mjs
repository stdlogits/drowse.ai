// drowse's steering-term-layers golden (`POST .../steering/validate` over a model-free composer)
// replayed through the browser's `sessions.validateSteering`: the loom over the real structured
// compiler, loaded from a core pack holding the golden's whiteners, its profiles (one-node
// concepts), its flat manifold and its curved one. Each case's canonical expression, n_layers,
// every term's key, fitted layers, mean-1 share and selected layers, or the refusal text; for a
// curved % term, the hook's per-layer share and the lowered curved groups (along, onto).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { corePackRequest } from "./support/core-pack.mjs";
import { viteServer } from "./support/units-goldens.mjs";

const golden = JSON.parse(readFileSync(new URL("../../drowse/tests/fixtures/browser/steering-term-layers-v1.json", import.meta.url), "utf8"));

const close = (actual, expected, where) => {
  if (typeof expected === "number") {
    assert.ok(typeof actual === "number" && Math.abs(actual - expected) <= 1e-9 + 1e-6 * Math.abs(expected), `${where}: ${actual} != ${expected}`);
  } else if (Array.isArray(expected)) {
    assert.ok(Array.isArray(actual) && actual.length === expected.length, `${where}: ${JSON.stringify(actual)} != ${JSON.stringify(expected)}`);
    expected.forEach((value, i) => close(actual[i], value, `${where}[${i}]`));
  } else if (expected !== null && typeof expected === "object") {
    assert.ok(actual !== null && typeof actual === "object", `${where}: ${JSON.stringify(actual)} is not an object`);
    assert.deepEqual(Object.keys(actual).sort(), Object.keys(expected).sort(), `${where} keys`);
    for (const key of Object.keys(expected)) close(actual[key], expected[key], `${where}.${key}`);
  } else {
    assert.equal(actual, expected, where);
  }
};

const norm = (v) => Math.hypot(...v);
const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);

const server = await viteServer();
try {
  const { BrowserFeasibilityCorePackCompiler } = await server.ssrLoadModule("/src/hosted/runtime/browserCorePack.ts");
  const { BrowserLoomRuntime } = await server.ssrLoadModule("/src/hosted/runtime/browserLoom.ts");
  const d = golden.hiddenSize;
  const means = new Map(golden.whiteners.map((row) => [row.layer, row.mean]));
  // Neutral-anchored rows, as drowse's fold anchors them: the mean's component along the basis.
  const anchored = (layer, basis) => {
    const along = dot(means.get(layer), basis);
    return basis.map((value) => along * value);
  };
  // A profile is a 2-node `pca` concept in a browser core pack (the grammar's tier-2 bare name):
  // its push is the first node's coordinate along the unit basis and a projection reads the
  // basis times the baked share, so both carry |v| (0 with any unit basis for a zero layer).
  const profiles = Object.entries(golden.profiles).map(([name, rows]) => {
    const unit = (direction) => {
      const length = norm(direction);
      return length > 0 ? direction.map((value) => value / length) : direction.map((_, i) => (i === 0 ? 1 : 0));
    };
    return {
      name,
      labels: [`${name}_pole`, `${name}_anti`],
      nodeCoordinates: [1, -1],
      // A baked share must be positive; a zero layer's 1e-9 whitens below the projection's
      // near-zero cutoff, so `~` still drops it.
      shares: Object.fromEntries(rows.map(({ layer, direction }) => [layer, norm(direction) || 1e-9])),
      layers: rows.map(({ layer, direction }) => ({
        layer,
        mean: anchored(layer, unit(direction)),
        basis: [unit(direction)],
        // The push reads the first node; the second keeps the pair non-degenerate.
        nodeCoordinates: [[norm(direction)], [-(norm(direction) || 1)]],
      })),
    };
  });
  const manifolds = Object.entries(golden.manifolds).map(([name, spec]) => spec.curved
    ? { name, labels: spec.labels, curved: true, layers: spec.layers }
    : {
        name,
        labels: spec.labels,
        layers: spec.layers.map((row) => ({
          layer: row.layer,
          mean: anchored(row.layer, row.basis),
          basis: [row.basis],
          nodeCoordinates: row.nodeCoordinates.map((c) => [c]),
        })),
      });
  const request = await corePackRequest(server, {
    hiddenSize: d,
    layerCount: golden.nLayers,
    whiteners: golden.whiteners,
    manifolds: [...profiles, ...manifolds],
  });
  const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
  const loom = new BrowserLoomRuntime({
    session: {
      id: "default", model_id: "fixture-model", created: 0,
      config: { temperature: 0, top_p: 1, top_k: null, max_tokens: 64, system_prompt: null, thinking: false },
      profiles: [], probes: [], history_length: 0, supports_thinking: false, thinking_is_optional: false,
      is_base_model: false, jlens_fitted: false, instruments: [], default_steering: null,
      role_substitution_supported: true, user_role_supported: true, default_assistant_role: "assistant",
      default_user_role: "user", scene_mode: true, thinking_input_supported: false,
    },
    generation: {},
    compileSteering: (expression, probeRequests) => compiler.compileForGeneration(expression, probeRequests),
    steeringManifolds: () => compiler.steeringManifolds(),
  });
  for (const entry of golden.cases) {
    const response = await loom.request({ service: "sessions", method: "validateSteering", args: [entry.expression] });
    const { parts: _parts, ...rest } = response;
    close(asProfiles(rest), entry.response, entry.name);
    if (entry.curvedShares) {
      const program = compiler.compile(entry.expression);
      close(curvedShares(program, entry.curvedShares[0].manifold), entry.curvedShares, `${entry.name} curvedShares`);
      close(curvedGroups(program), entry.curvedGroups, `${entry.name} curvedGroups`);
    }
  }
  console.log(`steering term layers: ${golden.cases.length} validate cases match drowse's`);
} finally {
  await server.close();
}

// drowse registers the golden's profiles as session vectors, whose bare names win the atom
// ladder and print bare (`a`); the browser has no registered-profile store (its concepts are
// core-pack manifolds), so a bare profile name resolves through the label tier and prints as
// the label-form push `local/a%a_pole`, and the fold's conflicting-trigger refusal names it a
// manifold, not a concept. Everything else about the term is compared as is.
function asProfiles(value) {
  if (typeof value === "string") {
    return value
      .replace(/local\/([a-z]+)%\1_pole/gu, "$1")
      .replace(/^manifold '([a-z]+)' appears with conflicting triggers/u, "concept '$1' appears with conflicting triggers");
  }
  if (Array.isArray(value)) return value.map(asProfiles);
  if (value !== null && typeof value === "object") return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, asProfiles(item)]));
  return value;
}

// The lowered curved groups (`curveRank > 0` slots, per layer): along and onto as the kernel
// gets them.
function curvedGroups(program) {
  const slots = program.curveRank.length / program.layerCount;
  const groups = [];
  for (let layer = 0; layer < program.layerCount; layer += 1) {
    for (let slot = 0; slot < slots; slot += 1) {
      const at = layer * slots + slot;
      if (program.curveRank[at] > 0) groups.push({ layer, along: program.curveAlong[at], onto: program.curveOnto[at] });
    }
  }
  return groups;
}

// The hook's per-layer share (`_manifold_layer_shares`, mean 1 over the layers the term acts
// on) as the lowering applied it: each layer's along over their mean (one curved term here).
function curvedShares(program, manifold) {
  const groups = curvedGroups(program);
  const mean = groups.reduce((sum, group) => sum + group.along, 0) / groups.length;
  return [{ manifold, layers: groups.map((group) => group.layer), share: groups.map((group) => group.along / mean) }];
}
