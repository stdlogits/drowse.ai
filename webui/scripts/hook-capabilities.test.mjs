import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({
  root,
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});

try {
  const {
    assertHookFeaturesSupported,
    requiredHookFeatures,
    AFFINE_FEASIBILITY_CAPABILITIES,
  } = await server.ssrLoadModule("/src/hosted/runtime/hookCapabilities.ts");
  const features = (expression) => [...requiredHookFeatures(expression)].sort();

  assert.deepEqual(features("0.5 deceptive"), ["rank_one"]);
  assert.doesNotThrow(() =>
    assertHookFeaturesSupported("0.5 deceptive", AFFINE_FEASIBILITY_CAPABILITIES)
  );
  assert.doesNotThrow(() =>
    assertHookFeaturesSupported(
      "0.5 deceptive~sycophantic + 0.25 focused",
      AFFINE_FEASIBILITY_CAPABILITIES,
    )
  );
  assert.doesNotThrow(() =>
    assertHookFeaturesSupported("0.5 !deceptive", AFFINE_FEASIBILITY_CAPABILITIES)
  );

  assert.deepEqual(
    features("0.4 deceptive~sycophantic@response&when:mood[1]>0.2 + !sae/42 + persona%pirate + jlens/fake"),
    [
      "ablation",
      "curved_manifold",
      "jlens",
      "multi_term",
      "phase_trigger",
      "probe_gate",
      "projection",
      "rank_one",
      "sae",
    ],
  );
  assert.throws(
    () => assertHookFeaturesSupported(
      "deceptive + !sae/42@response",
      AFFINE_FEASIBILITY_CAPABILITIES,
    ),
    (error) => {
      assert.equal(error.code, "DROWSE_HOOK_FEATURE_UNAVAILABLE");
      assert.deepEqual(error.missingFeatures, [
        "phase_trigger",
        "sae",
      ]);
      return true;
    },
  );

  // A layer clause asks nothing of the hooks: every program is laid out per layer.
  assert.deepEqual(features("0.5 calm{12-20}"), ["rank_one"]);
  assert.deepEqual(features("0.5 calm{3} + 0.2 warm{all} + 0.1 focused{workspace}"), [
    "multi_term",
    "rank_one",
  ]);
  assert.deepEqual(features("0.3,0.5 demo%alert{0}"), ["curved_manifold", "rank_one"]);
  assert.deepEqual(features("!calm{4-8}@response"), ["ablation", "phase_trigger", "rank_one"]);
  assert.doesNotThrow(() =>
    assertHookFeaturesSupported("0.5 calm{12-20} - 0.2 warm{3}", AFFINE_FEASIBILITY_CAPABILITIES)
  );

  // Units: a neuron push, a unit ablation (inside the block, not the residual `ablation`) and a
  // unit gate all need the units hooks.
  assert.deepEqual(features("0.3 neuron/L8n344"), ["rank_one", "units"]);
  assert.deepEqual(features("!neuron/L8n344"), ["rank_one", "units"]);
  assert.deepEqual(features("!head/L20h17 + 0.2 calm"), ["multi_term", "rank_one", "units"]);
  assert.deepEqual(features("0.5 calm@when:neuron/L8n344>0.2"), [
    "phase_trigger",
    "probe_gate",
    "rank_one",
    "units",
  ]);
  assert.deepEqual(features("0.5 calm@after&when:head/L3h2<=0.5"), [
    "phase_trigger",
    "probe_gate",
    "rank_one",
    "units",
  ]);
  // The S lens: atoms (by word or vocab id), ablation and gates.
  assert.deepEqual(features("0.4 slens/Owl"), ["rank_one", "slens"]);
  assert.deepEqual(features("0.4 slens/#12"), ["rank_one", "slens"]);
  assert.deepEqual(features("!slens/Owl"), ["ablation", "rank_one", "slens"]);
  assert.deepEqual(features("0.5 alert@when:slens/Owl>3"), [
    "phase_trigger",
    "probe_gate",
    "rank_one",
    "slens",
  ]);
  // Lens and SAE gates install their own probe kinds.
  assert.deepEqual(features("0.5 alert@when:jlens/Owl>3"), [
    "jlens",
    "phase_trigger",
    "probe_gate",
    "rank_one",
  ]);
  assert.deepEqual(features("0.5 alert@when:sae/42>3"), [
    "phase_trigger",
    "probe_gate",
    "rank_one",
    "sae",
  ]);

  // Until hook ABI v5 no library runs units or the S lens. The refusal is the compiler's, word
  // for word (browser-core-pack pins the compiler side), naming the first atom it resolves and
  // then the first gate it installs.
  const structured = Object.fromEntries(
    Object.keys(AFFINE_FEASIBILITY_CAPABILITIES).map((feature) => [feature, true]),
  );
  structured.units = false;
  structured.slens = false;
  for (const [expression, namespace] of [
    ["0.3 slens/#12", "slens"],
    ["0.3 slens/Owl", "slens"],
    ["0.2 neuron/L8n344", "neuron"],
    ["!head/L3h2", "head"],
    ["0.5 alert@when:neuron/L8n344>2", "neuron"],
    ["0.5 alert@after&when:head/L3h2<=0.5", "head"],
    ["0.5 alert@when:slens/Owl>3", "slens"],
    ["0.5 alert@when:slens/Owl>3 + !head/L3h2", "head"],
    ["0.2 calm{3} + 0.3 slens/Owl + !neuron/L1n2", "slens"],
  ]) {
    assert.throws(
      () => assertHookFeaturesSupported(expression, structured),
      (error) => {
        assert.equal(error.code, "CORE_PACK_EXPRESSION_UNAVAILABLE", expression);
        assert.equal(
          error.message,
          `${namespace}/ instrument atoms are not available in the browser runtime`,
          expression,
        );
        assert.equal(error.status, 409);
        return true;
      },
    );
  }
  // A library that offers them (Half B) passes the same expressions.
  assert.doesNotThrow(() => assertHookFeaturesSupported(
    "0.2 neuron/L8n344 + !head/L3h2 + 0.3 slens/Owl@when:slens/Owl>3",
    { ...structured, units: true, slens: true },
  ));
  // The affine library also lacks the gate hooks; the units refusal still speaks first.
  assert.throws(
    () => assertHookFeaturesSupported("0.5 alert@when:neuron/L8n344>2", AFFINE_FEASIBILITY_CAPABILITIES),
    (error) => {
      assert.equal(error.message, "neuron/ instrument atoms are not available in the browser runtime");
      assert.deepEqual(error.missingFeatures, ["phase_trigger", "probe_gate", "units"]);
      return true;
    },
  );

  console.log("Hook capability checks passed");
} finally {
  await server.close();
}
