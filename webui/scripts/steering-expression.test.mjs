// The browser grammar against drowse's goldens: the parsed IR (steering_ir.json), the validate
// parts the recipe chips rebuild from (recipe_chips.json) and the accept/reject table with
// referenced selectors (browser_steering_expression.json). Every case in all three is replayed.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const fixture = async (name) => JSON.parse(await readFile(
  new URL(`../../drowse/tests/fixtures/${name}`, import.meta.url),
  "utf8",
));
const server = await createServer({
  root,
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});

try {
  const grammar = await server.ssrLoadModule("/src/hosted/runtime/steeringExpression.ts");
  const {
    formatSteering,
    formatSteeringTrigger,
    parseSteering,
    parseSteeringExpression,
    referencedSteeringSelectors,
    selectTermLayers,
    SteeringExpressionError,
    steeringParts,
    steeringTriggerFields,
  } = grammar;

  // steering_ir.json: every accepted expression's ordered terms and canonical text, against the
  // fixture's registry; every rejected one's error class.
  const ir = await fixture("steering_ir.json");
  assert.equal(ir.schemaVersion, 1);
  const manifolds = ir.registry.map((entry) => ({
    namespace: entry.namespace,
    name: entry.name,
    fitMode: entry.fitMode,
    labels: entry.labels,
  }));
  const resolution = (testCase) => ({ manifolds, profileNames: testCase.profileNames });
  const triggerRecord = (trigger) => trigger === null ? null : {
    text: formatSteeringTrigger(trigger),
    ...steeringTriggerFields(trigger),
    gate: trigger.gate === null
      ? null
      : { probe: trigger.gate.probe, op: trigger.gate.operator, threshold: trigger.gate.threshold },
  };
  const layersRecord = (layers) => Array.isArray(layers) ? [...layers] : layers;
  const termRecord = (term, part) => {
    const record = { key: term.key, text: part.text, kind: term.kind };
    if (term.kind === "manifold") {
      Object.assign(record, {
        manifold: term.manifold,
        position: typeof term.position === "string" ? term.position : [...term.position],
        along: term.along,
        onto: term.onto,
        layers: layersRecord(term.layers),
      });
    } else if (term.kind === "projection") {
      Object.assign(record, {
        base: term.base, operator: term.operator, onto: term.onto, coeff: term.coeff,
        layers: layersRecord(term.layers),
      });
    } else if (term.kind === "unitAblation") {
      Object.assign(record, { unit: term.unit, coeff: term.coeff });
    } else {
      Object.assign(record, { target: term.target, coeff: term.coeff, layers: layersRecord(term.layers) });
    }
    record.trigger = triggerRecord(term.trigger);
    return record;
  };
  for (const testCase of ir.accepted) {
    const parsed = parseSteering(testCase.expression, resolution(testCase));
    const parts = steeringParts(parsed);
    assert.deepStrictEqual(
      {
        canonical: formatSteering(parsed),
        terms: parsed.terms.map((term, index) => termRecord(term, parts[index])),
      },
      { canonical: testCase.canonical, terms: testCase.terms },
      testCase.expression,
    );
    assert.equal(
      formatSteering(parseSteering(testCase.canonical, resolution(testCase))),
      testCase.canonical,
      `${testCase.expression}: the canonical form is a fixed point`,
    );
  }
  for (const testCase of ir.rejected) {
    assert.throws(
      () => parseSteering(testCase.expression, resolution(testCase)),
      (error) => error.code === testCase.error,
      testCase.expression,
    );
  }

  // recipe_chips.json: the validate parts of every case, and the chips' canonical expression.
  const chips = await fixture("recipe_chips.json");
  for (const testCase of [...chips.cases, ...chips.readOnly]) {
    const parsed = parseSteering(testCase.expression);
    const parts = steeringParts(parsed);
    assert.deepStrictEqual(parts, testCase.parts, testCase.expression);
    assert.equal(
      parts.map((part) => part.text).join(" + ").replaceAll("+ -", "- "),
      formatSteering(parsed),
      testCase.expression,
    );
  }
  for (const testCase of chips.cases) {
    assert.equal(formatSteering(parseSteering(testCase.expression)), testCase.expression, testCase.name);
  }

  // browser_steering_expression.json: accepted with their referenced selectors, and rejected.
  const table = await fixture("browser_steering_expression.json");
  assert.equal(table.schemaVersion, 1);
  for (const testCase of table.accepted) {
    assert.doesNotThrow(() => parseSteering(testCase.expression), testCase.expression);
    assert.deepEqual(
      referencedSteeringSelectors(testCase.expression).map((item) => [
        item.namespace,
        item.concept,
        item.variant,
      ]),
      testCase.selectors,
      testCase.expression,
    );
  }
  for (const expression of table.rejected) {
    assert.throws(
      () => parseSteering(expression),
      (error) => error instanceof SteeringExpressionError,
      expression,
    );
  }

  // The syntax tree keeps written positions for the compiler.
  const tree = parseSteeringExpression(
    "-0.6,0.3 default/emotions:role%pirate{ 20, 4 - 6 }@after&when:personas[3]>=-0.4",
  );
  assert.deepEqual(tree.terms[0].coefficients, [-0.6, -0.3]);
  assert.deepEqual(tree.terms[0].layers, [4, 5, 6, 20]);
  assert.deepEqual(tree.terms[0].selector.base, {
    namespace: "default", concept: "emotions", variant: "role", column: 9,
  });
  assert.equal(tree.terms[0].selector.manifoldPosition, "pirate");
  assert.deepEqual(tree.terms[0].trigger, {
    phase: { kind: "after_thinking" },
    gate: { probe: "personas[3]", operator: ">=", threshold: -0.4 },
  });
  assert.equal(
    parseSteeringExpression("0.3 jlens/#1e3").terms[0].selector.base.concept,
    "#1000",
    "a token id is stored as its integer, like the Python grammar",
  );
  assert.equal(parseSteeringExpression("!neuron/L08N0344").terms[0].selector.base.concept, "L8n344");
  assert.equal(parseSteeringExpression("x{ALL}").terms[0].layers, "all");
  assert.throws(
    () => parseSteeringExpression("0.3 neuron/L8n344:raw"),
    /'neuron\/' atoms name instrument units and take no ':raw' variant/,
  );
  assert.throws(() => parseSteeringExpression("x{4-8,6}"), /layer 6 is named twice/);
  assert.throws(() => parseSteeringExpression("0.5 head/L20h17"), /silence it with !head\/L20h17/);

  // select_term_layers: a clause must name fitted layers; workspace intersects the band.
  assert.equal(selectTermLayers("x", [1, 2, 3], null, 12), null);
  assert.deepEqual(selectTermLayers("x{2-3}", [1, 2, 3], [2, 3], 12), [2, 3]);
  assert.throws(
    () => selectTermLayers("x{2-5}", [1, 2, 3], [2, 3, 4, 5], 12),
    (error) => error.code === "SteeringCompositionError" &&
      error.message === "steering term 'x{2-5}' selects layer(s) 4-5 it is not fitted at; fitted layers: 1-3",
  );
  assert.deepEqual(selectTermLayers("x{workspace}", [0, 5, 9, 11], "workspace", 12), [5, 9]);
  assert.throws(
    () => selectTermLayers("x{workspace}", [0, 1], "workspace", 12),
    /selects the workspace band \(layers 5-9\) but is fitted at layers 0-1/,
  );

  // The `inverted` regen, against drowse 6.0's Recipe.invert_steering where no bare label
  // resolves; a bare label keeps its written name (drowse resolves it, which changes only text).
  for (const [written, inverted] of [
    ["0.5,0.3 demo%alert", "-0.5,-0.3 demo%alert"],
    ["-0.6,0.3 demo%alert{1-2}", "0.6,-0.3 demo%alert{1-2}"],
    ["0 demo%alert", "0 demo%alert"],
    ["!x", "-!x"],
    ["-!x{0}", "!x{0}"],
    ["0.25 !neuron/L8n344@prompt", "-0.25 !neuron/L8n344@before"],
    ["0.3 a~b{1-2}@response", "-0.3 a~b{1-2}@response"],
    ["0.1 x|y - 0.2 !z{workspace}", "-0.1 x|y + 0.2 !z{workspace}"],
    ["0.5 sae/3@response&when:sae/2>0.5", "-0.5 sae/3@when:sae/2>0.5"],
    ["0.5 calm:sae", "-0.5 calm:sae"],
    ["0.5 calm{0}@response", "-0.5 calm{0}@response"],
    ["1e-7 calm@both", "-1e-07 calm"],
    ["0.5,0.3 alert{1}", "-0.5,-0.3 alert{1}"],
  ]) {
    assert.equal(grammar.invertSteeringExpression(written), inverted, written);
  }
  assert.equal(grammar.invertSteeringExpression("  "), "");

  // sessions.validateSteering answers as `POST .../steering/validate` does.
  const { BrowserLoomRuntime } = await server.ssrLoadModule("/src/hosted/runtime/browserLoom.ts");
  // drowse's tree golden, with the hosted tree's inline token fields.
  const loomTree = await fixture("loom_tree_v2.json");
  Object.assign(loomTree, { model_id: "fixture/model", session_id: "default" });
  for (const node of Object.values(loomTree.nodes)) {
    Object.assign(node, { prompt_capture: null, raw_token_ids: null, thinking_tokens: null, tokens: null });
  }
  const loomSession = {
    id: "default", model_id: "fixture/model", created: 1, config: { max_tokens: 64 },
    role_substitution_supported: true,
  };
  const compiles = [];
  const runtime = new BrowserLoomRuntime({
    session: loomSession,
    generation: {},
    initialTree: loomTree,
    compileSteering(expression) {
      compiles.push(expression);
      if (expression.includes("{9}")) {
        throw Object.assign(new Error(
          "steering term 'x{9}' selects layer(s) 9 it is not fitted at; fitted layers: 0-3",
        ), { code: "STEERING_COMPOSITION", status: 422 });
      }
      if (expression.startsWith("plain")) return { hookAbi: "fixture" };
      const parsed = parseSteering(expression, {
        manifolds: [{ namespace: "default", name: "personas", fitMode: "pca", labels: ["pirate", "robot"] }],
      });
      return {
        hookAbi: "fixture",
        steeringExpression: formatSteering(parsed),
        steeringParts: steeringParts(parsed),
        steeringTerms: parsed.terms.map((term) => ({
          key: term.key, fitted: [0, 1, 2, 3], share: [0.5, 1, 1.5, 1], layers: [1, 2],
        })),
        steeringDepth: 4,
      };
    },
  });
  const validate = (expression) => runtime.request({
    service: "sessions", method: "validateSteering", args: [expression],
  });
  const resolved = await validate("  0.5 pirate{1-2}  ");
  assert.deepStrictEqual(resolved, {
    valid: true,
    expression: "0.5 default/personas%pirate{1-2}",
    error: null,
    n_layers: 4,
    terms: [{ key: "default/personas%pirate{1-2}", fitted: [0, 1, 2, 3], share: [0.5, 1, 1.5, 1], layers: [1, 2] }],
    parts: steeringParts(parseSteering("0.5 default/personas%pirate{1-2}")),
  });
  assert.equal(compiles.at(-1), "0.5 pirate{1-2}", "the stripped expression is compiled");
  const plain = await validate("plain + 0.2 x");
  assert.equal(plain.valid, true);
  assert.equal(plain.expression, "0.5 plain + 0.2 x");
  assert.equal(plain.n_layers, undefined, "a program without placements leaves the strip out");
  assert.deepEqual(plain.parts.map((part) => part.target), ["plain", "x"]);
  assert.deepStrictEqual(await validate("   "), { valid: true, expression: "", error: null });
  const syntax = await validate("honest +");
  assert.equal(syntax.valid, false);
  assert.equal(syntax.expression, "honest +");
  const unfitted = await validate("x{9}");
  assert.deepStrictEqual(unfitted, {
    valid: false,
    expression: "x{9}",
    error: "steering term 'x{9}' selects layer(s) 9 it is not fitted at; fitted layers: 0-3",
  });
  runtime.generating = true;
  await assert.rejects(validate("0.5 x"), (error) => error.code === "ENGINE_BUSY" && error.status === 409);
  runtime.generating = false;

  console.log(
    `Steering grammar checks passed (${ir.accepted.length + ir.rejected.length} IR, ` +
      `${chips.cases.length + chips.readOnly.length} chip, ` +
      `${table.accepted.length + table.rejected.length} table cases)`,
  );
} finally {
  await server.close();
}
