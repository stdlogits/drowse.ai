import assert from "node:assert/strict";
import { join } from "node:path";
import {
  parseArguments,
  semanticValidationArguments,
  targetRepository,
} from "./publish-hosted-pack.mjs";

assert.deepEqual(parseArguments(["qwen3-0.6b", "core", "./core"]), {
  modelId: "qwen3-0.6b",
  kind: "core",
  directory: join(process.cwd(), "core"),
  companions: [],
  coreStatistics: null,
  runtimeLock: null,
  upload: false,
});
assert.deepEqual(
  parseArguments(["qwen3-0.6b", "units", "./units", "--with", "./jlens", "--with", "./slens", "--core-statistics", "./core"]),
  {
    modelId: "qwen3-0.6b",
    kind: "units",
    directory: join(process.cwd(), "units"),
    companions: [join(process.cwd(), "jlens"), join(process.cwd(), "slens")],
    coreStatistics: join(process.cwd(), "core"),
    runtimeLock: null,
    upload: false,
  },
);
assert.throws(() => parseArguments(["qwen3-0.6b", "slens", "./slens", "--with", "./jlens"]), /belong to a units pack/);
assert.throws(() => parseArguments(["qwen3-0.6b", "units", "./units"]), /belong to a units pack/);
assert.equal(
  parseArguments(["qwen3-0.6b", "sae", "./sae", "--upload"]).upload,
  true,
);
assert.throws(() => parseArguments(["qwen3-0.6b", "unknown", "./pack"]), /usage/);
assert.throws(() => parseArguments(["qwen3-0.6b", "core", "./pack", "--force"]), /usage/);

const model = { convertedRepository: "stdlogits/drowse-web-qwen3-0.6b" };
assert.equal(targetRepository(model, "core"), model.convertedRepository);
assert.equal(
  targetRepository(model, "sae"),
  "stdlogits/drowse-web-qwen3-0.6b-instruments",
);
assert.equal(
  targetRepository(model, "jlens"),
  "stdlogits/drowse-web-qwen3-0.6b-instruments",
);
assert.throws(() => targetRepository({}, "core"), /converted repository/);
assert.deepEqual(
  semanticValidationArguments("fixture", "core", "/tmp/core").slice(1),
  ["fixture", "/tmp/core"],
);
// A candidate lock reaches the validators of every kind.
assert.equal(parseArguments(["qwen3-0.6b", "core", "./core", "--runtime-lock", "c.json"]).runtimeLock, join(process.cwd(), "c.json"));
assert.deepEqual(
  semanticValidationArguments("fixture", "core", "/tmp/core", [], null, "/tmp/c.json").slice(1),
  ["fixture", "/tmp/core", "--runtime-lock", "/tmp/c.json"],
);
assert.deepEqual(
  semanticValidationArguments("fixture", "slens", "/tmp/slens", [], null, "/tmp/c.json").slice(1),
  ["fixture", "slens", "/tmp/slens", "--release", "--runtime-lock", "/tmp/c.json"],
);
// Native lens and units packs (and a provider R lens) publish under the release policy.
assert.deepEqual(
  semanticValidationArguments("fixture", "jlens", "/tmp/jlens").slice(1),
  ["fixture", "jlens", "/tmp/jlens", "--release"],
);
assert.deepEqual(
  semanticValidationArguments("fixture", "sae", "/tmp/sae").slice(1),
  ["fixture", "sae", "/tmp/sae"],
);
assert.deepEqual(
  semanticValidationArguments("fixture", "units", "/tmp/units", ["/tmp/jlens", "/tmp/slens"], "/tmp/core").slice(1),
  ["fixture", "units", "/tmp/units", "--release", "--with", "/tmp/jlens", "--with", "/tmp/slens", "--core-statistics", "/tmp/core"],
);

console.log("hosted pack publisher checks passed");
