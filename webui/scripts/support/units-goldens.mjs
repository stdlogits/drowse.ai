// Shared helpers for the hook ABI v5 script tests: load drowse's units goldens
// (drowse/tests/fixtures/browser/<name>-<arch>-v1.json), decode their arrays and compare within
// each file's tolerance block, matching declared ties by identity (lanes/B3.md).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

export const webuiRoot = fileURLToPath(new URL("../..", import.meta.url));
const fixtures = fileURLToPath(new URL("../../../drowse/tests/fixtures/browser/", import.meta.url));

export const golden = (name) => JSON.parse(readFileSync(`${fixtures}${name}`, "utf8"));
export const archGolden = (name, arch) => golden(`${name}-${arch}-v1.json`);
export const ARCHES = ["gemma3", "qwen3"];

export async function viteServer() {
  return createServer({
    root: webuiRoot,
    configFile: false,
    appType: "custom",
    logLevel: "silent",
    server: { middlewareMode: true, watch: null },
  });
}

/** `encode`'s inverse: a list, an inline `{shape, values}` or a base64 record → typed array. */
export function decode(value) {
  if (Array.isArray(value)) return Float64Array.from(value);
  if (value && value.encoding) {
    const bytes = Buffer.from(value.data, "base64");
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    if (value.dtype === "int32") return Int32Array.from({ length: bytes.length / 4 }, (_, i) => view.getInt32(4 * i, true));
    if (value.dtype === "float16") return Uint16Array.from({ length: bytes.length / 2 }, (_, i) => view.getUint16(2 * i, true));
    return Float32Array.from({ length: bytes.length / 4 }, (_, i) => view.getFloat32(4 * i, true));
  }
  if (value && Array.isArray(value.values)) return Float64Array.from(value.values);
  throw new TypeError("not an encoded array");
}

export const shapeOf = (value) => (Array.isArray(value) ? [value.length] : value.shape);

/** Max violation of `|a − b| ≤ atol + rtol·|b|` (≤ 0 passes). */
export function closeness(actual, expected, atol, rtol = 0) {
  assert.equal(actual.length, expected.length, "length");
  let worst = -Infinity;
  let at = -1;
  for (let i = 0; i < expected.length; i += 1) {
    const excess = Math.abs(actual[i] - expected[i]) - (atol + rtol * Math.abs(expected[i]));
    if (!(excess <= worst) || Number.isNaN(excess)) {
      worst = Number.isNaN(excess) ? Infinity : excess;
      at = i;
    }
  }
  return { worst, at };
}

export function assertClose(actual, expected, tol, label) {
  const { worst, at } = closeness(actual, expected, tol.atol, tol.rtol ?? 0);
  assert.ok(
    worst <= 0,
    `${label}: element ${at} is ${actual[at]}, expected ${expected[at]} (excess ${worst})`,
  );
}

/** Deep comparison of a JSON result against the golden's: numbers within `atol` (rounded record
 *  fields, 2e-6) or `numberTol(path)`, strings and structure exact, declared tie runs matched by
 *  identity (B3: `{path, key, groups}`). */
export function compareRecord(actual, expected, options = {}) {
  const ties = new Map((options.ties ?? []).map((tie) => [tie.path, tie]));
  const problems = [];
  const tolFor = options.numberTol ?? (() => 2e-6);
  const walk = (a, e, path) => {
    if (problems.length > 20) return;
    if (typeof e === "number") {
      if (typeof a !== "number" || !(Math.abs(a - e) <= tolFor(path, e))) problems.push(`${path}: ${a} != ${e}`);
      return;
    }
    if (e === null || typeof e !== "object") {
      if (a !== e) problems.push(`${path}: ${JSON.stringify(a)} != ${JSON.stringify(e)}`);
      return;
    }
    if (Array.isArray(e)) {
      if (!Array.isArray(a) || a.length !== e.length) {
        problems.push(`${path}: length ${Array.isArray(a) ? a.length : typeof a} != ${e.length}`);
        return;
      }
      const tie = ties.get(path.replace(/^\./, ""));
      let order = e.map((_, i) => i);
      if (tie) {
        const ident = (x) => JSON.stringify(tie.key.map((k) => x[k]));
        for (const group of tie.groups) {
          const want = group.map((i) => ident(e[i])).sort();
          const got = group.map((i) => ident(a[i])).sort();
          if (JSON.stringify(want) !== JSON.stringify(got)) problems.push(`${path}: tie run ${group} differs`);
          for (const i of group) {
            const j = group.find((g) => ident(a[g]) === ident(e[i]));
            if (j !== undefined) order[i] = j;
          }
        }
      }
      e.forEach((item, i) => walk(a[order[i]], item, `${path}[${i}]`));
      return;
    }
    if (a === null || typeof a !== "object") {
      problems.push(`${path}: ${JSON.stringify(a)} is not an object`);
      return;
    }
    const keys = new Set([...Object.keys(e), ...Object.keys(a)]);
    for (const key of keys) {
      if (!(key in e)) problems.push(`${path}.${key}: unexpected`);
      else if (!(key in a)) problems.push(`${path}.${key}: missing`);
      else walk(a[key], e[key], `${path}.${key}`);
    }
  };
  walk(actual, expected, options.root ?? "");
  return problems;
}
