// Unit and `slens/` probe digests: the Python reprs and sha256s drowse computes
// (`UnitsInstrument.probe_hash`, `LensInstrument.probe_hash` and `_centre_digest`). The
// expected strings and digests were printed by CPython 3.13 from the same tuples.
import assert from "node:assert/strict";
import { viteServer } from "./support/units-goldens.mjs";

const server = await viteServer();
try {
  const digest = await server.ssrLoadModule("/src/hosted/runtime/units/probeDigest.ts");
  const unitTuple = [
    "units-readout-v2", "m/x", "neuron/L2n5", "300 neutral statements",
    { list: [0.0009765625, 1.2345678912345, 2.0, 1e-7, 1.5e16, 123456.789].map((v) => ({ float: digest.pythonRound(v, 9) })) },
  ];
  assert.equal(
    digest.pythonRepr(unitTuple),
    "('units-readout-v2', 'm/x', 'neuron/L2n5', '300 neutral statements', [0.000976562, 1.234567891, 2.0, 1e-07, 1.5e+16, 123456.789])",
  );
  assert.equal(
    digest.unitProbeDigest("m/x", "neuron/L2n5", "300 neutral statements", [0.0009765625, 1.2345678912345, 2.0, 1e-7, 1.5e16, 123456.789]),
    "ba4c5c723f9dde1e021668eb474961c73b9241f9ba63749d997af4a0dc43b1c7",
  );
  // Python rounds the exact binary value half to even (0.0009765625 is a tie at 9 decimals).
  assert.equal(digest.pythonRound(0.0009765625, 9), 0.000976562);
  assert.equal(digest.pythonRound(-2.5e-9, 9), -3e-9); // -2.5e-9 is just past the tie
  assert.equal(digest.pythonFloatRepr(1e16), "1e+16");
  assert.equal(digest.pythonFloatRepr(0.0001), "0.0001");
  assert.equal(digest.pythonFloatRepr(0.00001), "1e-05");
  assert.equal(digest.pythonFloatRepr(-3), "-3.0");

  assert.equal(
    digest.pythonRepr(["slens-footprint-z-v3", "m/x", "it's é\n​", 5, [0], ["abc", "def", "tokenizer"]]),
    "('slens-footprint-z-v3', 'm/x', \"it's é\\n\\u200b\", 5, (0,), ('abc', 'def', 'tokenizer'))",
  );
  assert.equal(
    digest.slensProbeDigest("m/x", "it's é\n​", 5, [0], { lens: "abc", centre: "def", mask: "tokenizer" }),
    "c7db2f6034af84c20de67b384f08196a44ce3d6142de2bc582eb83057402bf22",
  );
  const means = Float32Array.from([0.5, -1.25, 3.0, 1e-3, 2.5, 0.0, -0.75, 7.0, 1.0, 2.0, 3.0, 4.0]);
  assert.equal(digest.centreDigest(means, 4, [0, 2]), "8676100171c2570bb14965176b66f8109046a31914149cef956fdc35f9f420f9");
  console.log("units probe digests match drowse's");
} finally {
  await server.close();
}
