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
  const { filterBrowserTree, parseBrowserTreeFilter } = await server.ssrLoadModule(
    "/src/hosted/runtime/browserTreeFilter.ts",
  );
  const rows = [
    node("root", {}, []),
    node("a", { confidence: 0.4, "default/personas": -0.2, "sae/12": 3 }, [
      token({ confidence: 0.1, "sae/12": 0.5 }),
      token({ confidence: 0.8 }),
    ]),
    node("b", { confidence: 0.7, "default/personas": 0.3 }, [
      token({ confidence: 0.6 }),
      token({ confidence: 0.5, "sae/12": 4 }),
    ]),
    node("c", { confidence: 0.9 }, [
      token({ confidence: 0.2 }),
      token({ confidence: 0.4 }),
    ]),
    node("d", { confidence: 0.1 }, [
      legacyToken({ confidence: 0.05 }),
    ]),
  ];

  assert.deepEqual(filterBrowserTree(rows, ""), []);
  assert.deepEqual(filterBrowserTree(rows, "confidence >= 0.7"), ["b", "c"]);
  assert.deepEqual(
    filterBrowserTree(rows, "any:confidence < 0.1"),
    [],
    "rows outside the measurement envelope carry no readings",
  );
  assert.deepEqual(filterBrowserTree(rows, "agg:confidence > 0.3,"), ["a", "b", "c"]);
  assert.deepEqual(filterBrowserTree(rows, "any:confidence > 0.7"), ["a"]);
  assert.deepEqual(filterBrowserTree(rows, "any:confidence < 0.3"), ["a", "c"]);
  assert.deepEqual(filterBrowserTree(rows, "last:confidence >= 0.5"), ["a", "b"]);
  assert.deepEqual(
    filterBrowserTree(rows, "agg:confidence > 0.5, last:confidence < 0.5"),
    ["c"],
  );
  assert.deepEqual(filterBrowserTree(rows, "default/personas >= 0"), ["b"]);
  assert.deepEqual(filterBrowserTree(rows, "default/personas[0] >= 0"), ["b"]);
  assert.deepEqual(parseBrowserTreeFilter("agg:sae/12[0] > 1").map((clause) => clause.probe), ["sae/12"]);
  assert.deepEqual(filterBrowserTree(rows, "agg:sae/12 > 1"), ["a"]);
  assert.deepEqual(filterBrowserTree(rows, "any:sae/12 > 1"), ["b"]);
  assert.deepEqual(filterBrowserTree(rows, "last:sae/12 < 1"), ["a"]);
  for (const [expression, reads] of [
    ["default/personas[1] >= 0", "coordinate axis 1"],
    ["x[01] > 0", "coordinate axis 1"],
    ["agg:emotions:fraction > 0.5", "the subspace fraction channel"],
    ["agg:emotions:membership > 0.5", "the tube membership channel"],
    ["emotions@happy > 0", "the label distance channel"],
    ["any:emotions~happy > 0.5", "the soft assignment channel"],
  ]) {
    assert.throws(
      () => parseBrowserTreeFilter(expression),
      (error) => error.message.includes(`reads ${reads}, but loom nodes persist axis-0 readings only`),
      expression,
    );
  }
  for (const expression of ["sae/1.5 > 0", "default/sae/1 > 0", "x[ 0] > 0", "jlens/#7 > 0"]) {
    assert.throws(() => parseBrowserTreeFilter(expression), /not a valid identifier/, expression);
  }
  assert.deepEqual(filterBrowserTree(rows, "agg:missing > 0"), []);

  assert.throws(() => parseBrowserTreeFilter("unknown:confidence > 0"), /unknown agg op/);
  assert.throws(() => parseBrowserTreeFilter("agg:confidence"), /missing comparison op/);
  assert.throws(() => parseBrowserTreeFilter("agg:9confidence > 0"), /not a valid identifier/);
  assert.throws(() => parseBrowserTreeFilter("agg:confidence > nope"), /is not a number/);
  assert.throws(() => parseBrowserTreeFilter("agg:confidence > 0x10"), /is not a number/);
  assert.throws(() => parseBrowserTreeFilter("  "), /empty filter expression/);
} finally {
  await server.close();
}

console.log("browser tree filter checks passed");

function node(id, aggregate_readings, tokens) {
  return { id, aggregate_readings, tokens, thinking_tokens: null };
}

function token(scores) {
  return {
    text: "x",
    logprob: null,
    perplexity: null,
    measurements: {
      version: 1,
      scope: "token",
      provenance: "captured",
      instruments: {},
      scores,
    },
  };
}

function legacyToken(probes) {
  return { text: "x", logprob: null, perplexity: null, probes };
}
