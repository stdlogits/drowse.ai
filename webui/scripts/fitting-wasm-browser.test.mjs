import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import init, {
  detectPeriodicTopology,
  selectTopologyFromTargets,
} from "../public-hosted/wasm/drowse_fitting_wasm.js";

const wasm = await readFile(new URL(
  "../public-hosted/wasm/drowse_fitting_wasm_bg.wasm",
  import.meta.url,
));
assert.doesNotMatch(wasm.toString("latin1"), /\/(?:Users|home)\/[^/\x00]+\//,
  "shipped WebAssembly must not embed personal home directories");
await init({ module_or_path: wasm });

const automaticCurveTargets = new Float64Array(40 * 3);
for (let index = 0; index < 40; index += 1) {
  const t = index / 39;
  automaticCurveTargets[3 * index] = Math.sin(5 * t);
  automaticCurveTargets[3 * index + 1] = Math.cos(7 * t);
  automaticCurveTargets[3 * index + 2] = Math.sin(11 * t);
}
const automaticCurve = selectTopologyFromTargets(
  centeredGram(automaticCurveTargets, 40, 3),
  40,
  automaticCurveTargets,
  new Uint32Array([0, automaticCurveTargets.length]),
  6,
  0.7,
  "auto",
  undefined,
  undefined,
  undefined,
  0.5,
);
try {
  assert.equal(automaticCurve.winnerName, "spectral");
} finally {
  automaticCurve.free();
}

const circle = new Float64Array(40 * 2);
for (let index = 0; index < 40; index += 1) {
  const angle = 2 * Math.PI * index / 40;
  circle[2 * index] = Math.cos(angle);
  circle[2 * index + 1] = Math.sin(angle);
}
const periodic = selectTopologyFromTargets(
  centeredGram(circle, 40, 2),
  40,
  circle,
  new Uint32Array([0, circle.length]),
  6,
  0.7,
  "auto",
  undefined,
  undefined,
  undefined,
  0.5,
);
try {
  assert.equal(periodic.winnerName, "torus-T1");
  assert.equal(periodic.periodicDimensions, 1);
} finally {
  periodic.free();
}

const thinTorus = [];
const mobius = [];
const sphere = [];
for (let i = 0; i < 16; i++) {
  for (let j = 0; j < 8; j++) {
    const u = 2 * Math.PI * i / 16, v = 2 * Math.PI * j / 8;
    thinTorus.push([Math.cos(u), Math.sin(u), 0.3 * Math.cos(v), 0.3 * Math.sin(v)]);
  }
}
for (let i = 0; i < 24; i++) {
  for (let j = 0; j < 5; j++) {
    const u = 2 * Math.PI * i / 24, v = 0.1 * (j / 2 - 1);
    mobius.push([(1 + v * Math.cos(u / 2)) * Math.cos(u),
      (1 + v * Math.cos(u / 2)) * Math.sin(u), v * Math.sin(u / 2)]);
  }
}
for (let i = 0; i < 80; i++) {
  const z = 1 - 2 * (i + 0.5) / 80, theta = i * Math.PI * (3 - Math.sqrt(5));
  sphere.push([Math.sqrt(1 - z * z) * Math.cos(theta), Math.sqrt(1 - z * z) * Math.sin(theta), z]);
}
for (const [name, points, expected] of [["thin torus", thinTorus, 2], ["Mobius strip", mobius, 0], ["sphere", sphere, 0]]) {
  for (const rows of [points, [...points].reverse()]) {
    const chart = detectPeriodicTopology(centeredGram(new Float64Array(rows.flat()), rows.length, rows[0].length),
      rows.length, 6, undefined, undefined, 0.5);
    try {
      assert.equal(chart?.dimensions ?? 0, expected, `${name}: wrong periodic chart`);
    } finally {
      chart?.free();
    }
  }
}

const unrelatedTargets = Float64Array.from({ length: 40 }, (_, i) => i % 2 ? -1 : 1);
const unsupportedPeriodicFit = selectTopologyFromTargets(
  centeredGram(circle, 40, 2), 40, unrelatedTargets, new Uint32Array([0, 40]),
  6, 0.7, "auto", undefined, undefined, undefined, 0.5,
);
try {
  assert.notEqual(unsupportedPeriodicFit.winnerName, "torus-T1", "poor periodic predictions must not override the baseline");
} finally {
  unsupportedPeriodicFit.free();
}

console.log("browser fitting WASM topology checks passed");

function centeredGram(values, rows, columns) {
  const means = new Float64Array(columns);
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      means[column] += values[row * columns + column] / rows;
    }
  }
  const gram = new Float64Array(rows * rows);
  for (let left = 0; left < rows; left += 1) {
    for (let right = 0; right < rows; right += 1) {
      for (let column = 0; column < columns; column += 1) {
        gram[left * rows + right] +=
          (values[left * columns + column] - means[column]) *
          (values[right * columns + column] - means[column]);
      }
    }
  }
  return gram;
}
