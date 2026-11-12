// The browser grammar: the validate parts the recipe chips rebuild from (drowse's
// recipe_chips.json), positions, layer selection, rejection text, inversion and validate.
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
    parseSteering,
    parseSteeringExpression,
    selectTermLayers,
    steeringParts,
  } = grammar;

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

  // Rejection text is what validate shows; these are drowse 6.0's parse_expr messages verbatim
  // (Python reprs, token values, `(col N)`).
  for (const [written, message] of [
    ["0.٣ x", "unexpected character '٣' (col 2)"],
    ["a.b.c", "unexpected token DOT ('.') (col 3)"],
    ["x + ", "expected IDENT, got EOF ('') (col 4)"],
    ["x y", "unexpected identifier 'y' after a complete term; multi-word concept names use " +
      "underscores (e.g. 'artificial_intelligence', not 'artificial intelligence'), and bipolar " +
      "pairs join with '.' (e.g. 'human.artificial_intelligence') (col 2)"],
    ["\"x\"", "quoted identifiers are not supported; use underscores for multi-word concept names " +
      "(e.g. 'artificial_intelligence') and '.' for bipolar pairs (e.g. " +
      "'human.artificial_intelligence') (col 0)"],
    ["0.5 x@foo", "unknown trigger '@foo'; valid: after, before, both, generated, prompt, response, " +
      "thinking; 'first:<n>' / 'after:<n>' token windows; 'when:<probe><op><threshold>' probe " +
      "gates; or a compound '<preset>&when:<probe><op><threshold>'.  Note: '@' is for triggers " +
      "only; HF revisions are not accepted inside steering expressions. (col 6)"],
    ["foo@when:bar:fraction[0]>0",
      "expected comparison op (>, >=, <, <=) in '@when:' clause, got LBRACK ('[') (col 21)"],
    ["0.5 x@first:1.5", "trigger '@first:' takes a non-negative integer token count, got '1.5' (col 12)"],
  ]) {
    assert.throws(() => parseSteering(written), (error) => error.message === message, written);
  }

  // The `inverted` regen, against drowse 6.0's Recipe.invert_steering with this registry installed
  // (folders only): bare labels resolve to their `%` terms and repeats fold before the signs flip,
  // and a refusal is drowse's own class and words.
  const manifolds = [
    { namespace: "default", name: "verbose.concise", fitMode: "pca", labels: ["verbose", "concise"] },
    { namespace: "default", name: "personas", fitMode: "pca", labels: ["pirate", "hacker", "robot"] },
    { namespace: "local", name: "formal.casual", fitMode: "pca", labels: ["formal", "casual"] },
    { namespace: "local", name: "animals", fitMode: "pca", labels: ["deer", "owl"] },
    { namespace: "local", name: "forest", fitMode: "pca", labels: ["deer", "oak"] },
  ];
  for (const [written, expected] of [
    ["0.5,0.3 demo%alert", {"inverted": "-0.5,-0.3 demo%alert"}],
    ["-0.6,0.3 demo%alert{1-2}", {"inverted": "0.6,-0.3 demo%alert{1-2}"}],
    ["0 demo%alert", {"inverted": "0 demo%alert"}],
    ["!x", {"inverted": "-!x"}],
    ["-!x{0}", {"inverted": "!x{0}"}],
    ["0.25 !neuron/L8n344@prompt", {"inverted": "-0.25 !neuron/L8n344@before"}],
    ["0.3 a~b{1-2}@response", {"inverted": "-0.3 a~b{1-2}@response"}],
    ["0.1 x|y - 0.2 !z{workspace}", {"inverted": "-0.1 x|y + 0.2 !z{workspace}"}],
    ["0.5 sae/3@response&when:sae/2>0.5", {"inverted": "-0.5 sae/3@when:sae/2>0.5"}],
    ["0.5 calm:sae", {"inverted": "-0.5 calm:sae"}],
    ["1e-7 calm@both", {"inverted": "-1e-07 calm"}],
    ["0.5 pirate", {"inverted": "-0.5 default/personas%pirate"}],
    ["0.5 pirate{0}@response", {"inverted": "-0.5 default/personas%pirate{0}@response"}],
    ["0.5,0.3 pirate{1}", {"inverted": "-0.5,-0.3 default/personas%pirate{1}"}],
    ["0 pirate", {"inverted": "0 default/personas%pirate"}],
    ["0.5 pirate + 0.2 pirate", {"inverted": "-0.7 default/personas%pirate"}],
    ["0.5 calm + 0.2 calm", {"inverted": "-0.7 calm"}],
    ["0.5 calm{0}@response", {"inverted": "-0.5 calm{0}@response"}],
    ["0.5 verbose", {"inverted": "-0.5 default/verbose.concise%verbose"}],
    ["0.5 verbose.concise", {"inverted": "-0.5 default/verbose.concise%verbose"}],
    ["0.4 owl - 0.1 formal", {"inverted": "-0.4 local/animals%owl + 0.1 local/formal.casual%formal"}],
    ["-0.3 !pirate", {"inverted": "0.3 !pirate"}],
    ["0.5 concise~verbose", {"inverted": "-0.5 concise~verbose"}],
    ["0.5 x + 0.25 x@response", {"error": "SteeringExprError", "message": "concept 'x' appears with conflicting triggers; merge triggers explicitly or split into separate Steering entries"}],
    ["0.2 hacker + 0.3 calm{1}@after:3", {"inverted": "-0.2 default/personas%hacker - 0.3 calm{1}@after:3"}],
    ["0.5 deer", {"error": "AmbiguousSelectorError", "message": "ambiguous manifold label 'deer': matches local/animals (label 'deer'), local/forest (label 'deer'). Specify the manifold explicitly as <ns>/<name>%deer. (disambiguate with namespace/name)"}],
    ["0.2 personas%pirate + 0.1 hacker", {"inverted": "-0.2 personas%pirate - 0.1 default/personas%hacker"}],
    ["0.5,0.3 alert{1}", {"error": "SteeringExprError", "message": "comma-separated coefficients are only valid for `manifold % position` terms"}],
  ]) {
    if ("inverted" in expected) {
      assert.equal(grammar.invertSteeringExpression(written, { manifolds }), expected.inverted, written);
    } else {
      assert.throws(() => grammar.invertSteeringExpression(written, { manifolds }), (error) => {
        assert.equal(error.code, expected.error, written);
        assert.equal(error.message, expected.message, written);
        return true;
      });
    }
  }
  // Without the manifolds nothing resolves, as with an empty registry.
  assert.equal(grammar.invertSteeringExpression("0.5 pirate + 0.2 pirate"), "-0.7 pirate");
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
  const syntax = await validate("deceptive +");
  assert.equal(syntax.valid, false);
  assert.equal(syntax.expression, "deceptive +");
  const unfitted = await validate("x{9}");
  assert.deepStrictEqual(unfitted, {
    valid: false,
    expression: "x{9}",
    error: "steering term 'x{9}' selects layer(s) 9 it is not fitted at; fitted layers: 0-3",
  });
  // Busy only past the in-band answers: drowse returns the empty expression and a syntax error
  // before taking its lock.
  runtime.generating = true;
  const compilesBefore = compiles.length;
  await assert.rejects(validate("0.5 x"), (error) => error.code === "ENGINE_BUSY" && error.status === 409);
  assert.deepStrictEqual(await validate(""), { valid: true, expression: "", error: null });
  assert.equal((await validate("deceptive +")).valid, false);
  assert.equal(compiles.length, compilesBefore, "nothing compiles while busy");
  runtime.generating = false;

  // Columns count code points and whitespace is `str.isspace`, as Python's lexer sees a `str`: an
  // astral character inside a clause no longer shifts later columns, a lone surrogate never shows
  // in a message, NEL and the information separators are spaces and U+FEFF is not. Each input is
  // stripped as the validate route strips it. The outcomes are drowse 6.0's own
  // (`_Parser(_lex(text.strip())).parse()` on the pinned submodule, run from the venv).
  const { pythonStrip } = grammar;
  for (const [input, expected] of [
    ["0.5 calm{\ud83d\ude00} $", {"error": "unexpected character '$' (col 12)"}],
    ["0.5 calm{\ud83d\ude00\ud83d\ude00} + 0.2 warm{4} ?", {"error": "unexpected character '?' (col 27)"}],
    ["0.5 \ud83d\ude00", {"error": "unexpected character '\ud83d\ude00' (col 4)"}],
    ["0.5 cal\ud83d\ude00m", {"error": "unexpected character '\ud83d\ude00' (col 7)"}],
    ["0.5 calm{\ud83d\ude00}{", {"error": "unterminated layer clause; close it with '}' (col 11)"}],
    ["0.5 calm{ \ud83d\ude00 ,}", {"error": "layer clause: malformed layers ' \ud83d\ude00 ,' (want comma-separated indices or a-b ranges, or one of all, workspace) (col 10)"}],
    ["0.5 calm{4,\ud83d\ude00}", {"error": "layer clause: malformed layers '4,\ud83d\ude00' (want comma-separated indices or a-b ranges, or one of all, workspace) (col 11)"}],
    ["0.5 calm{\ud83d\ude00} + 0.2 warm{3-1}", {"error": "layer clause: malformed layers '\ud83d\ude00' (want comma-separated indices or a-b ranges, or one of all, workspace) (col 9)"}],
    ["0.5 calm{\ud83d\ude00}@when:x>\ud83d\ude00", {"error": "unexpected character '\ud83d\ude00' (col 19)"}],
    ["0.5\u0085calm", {"terms": [[0.5]]}],
    ["0.5\u001ccalm + 0.25\u001fwarm", {"terms": [[0.5], [0.25]]}],
    ["0.5\ufeffcalm", {"error": "unexpected character '\\ufeff' (col 3)"}],
    ["\u0085 0.5 calm \u001e", {"terms": [[0.5]]}],
    ["\ufeff0.5 calm", {"error": "unexpected character '\\ufeff' (col 0)"}],
    ["0.5 calm{\u00854\u0085}", {"terms": [[0.5]]}],
    ["0.5 calm{4\ufeff}", {"error": "layer clause: malformed layers '4\\ufeff' (want comma-separated indices or a-b ranges, or one of all, workspace) (col 9)"}],
    ["0.5 calm{'}", {"error": "layer clause: malformed layers \"'\" (want comma-separated indices or a-b ranges, or one of all, workspace) (col 9)"}],
    ["0.5 calm{\"'}", {"error": "layer clause: malformed layers '\"\\'' (want comma-separated indices or a-b ranges, or one of all, workspace) (col 9)"}],
    ["0.5 calm{4-6, \ud83d\ude00}", {"error": "layer clause: malformed layers '4-6, \ud83d\ude00' (want comma-separated indices or a-b ranges, or one of all, workspace) (col 14)"}],
    ["0.5 calm{ sample }", {"error": "layer clause: unknown layer mode 'sample' (want all, workspace, or an index list) (col 10)"}],
    ["0.5 calm{\u3000sample}", {"error": "layer clause: unknown layer mode 'sample' (want all, workspace, or an index list) (col 10)"}],
    ["0.5 calm{4,,5}", {"error": "layer clause: empty entry in layers '4,,5' (col 11)"}],
    ["0.5 calm{\ud83d\ude00,\ud83d\ude00}", {"error": "layer clause: malformed layers '\ud83d\ude00,\ud83d\ude00' (want comma-separated indices or a-b ranges, or one of all, workspace) (col 9)"}],
  ]) {
    const text = pythonStrip(input);
    if ("error" in expected) {
      assert.throws(() => parseSteeringExpression(text), (error) => {
        assert.equal(error.message, expected.error, JSON.stringify(input));
        return true;
      });
    } else {
      assert.deepEqual(
        parseSteeringExpression(text).terms.map((term) => [...term.coefficients]),
        expected.terms,
        JSON.stringify(input),
      );
    }
  }
  assert.equal(pythonStrip("\u0085\u001c 0.5 x\u3000\u001f"), "0.5 x");
  assert.equal(pythonStrip("\ufeff0.5 x"), "\ufeff0.5 x");
  assert.equal((await validate("\u0085 plain + 0.2 x\u001e")).expression, "0.5 plain + 0.2 x");

  console.log(`Steering grammar checks passed (${chips.cases.length + chips.readOnly.length} chip cases)`);
} finally {
  await server.close();
}
