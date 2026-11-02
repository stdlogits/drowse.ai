// The checked-in release evidence configs (browser-runtime/evidence-configs): each one is valid
// under evidence-config.schema.json, names an observer in its producer's source closure, and the
// hook ABI v5 runtime records cover the lock: one productionLibrary config per lock model, one
// productionQ4Parity config per q4f32_1 model, and the C VM and WebGPU goldens.
import assert from "node:assert/strict";
import { readdir, readFile } from "node:fs/promises";
import { resolve } from "node:path";
import test from "node:test";
import {
  parseBrowserFullRuntimeArguments,
  validateBrowserFullRuntimePackSets,
} from "./browser-full-runtime-contract.mjs";
import { requireReleaseEvidenceProducer } from "./release-evidence-producers.mjs";

const repositoryRoot = resolve(import.meta.dirname, "../..");
const configRoot = resolve(repositoryRoot, "browser-runtime/evidence-configs");
const schema = JSON.parse(await readFile(resolve(repositoryRoot, "browser-runtime/evidence-config.schema.json"), "utf8"));
const runtimeLock = JSON.parse(await readFile(resolve(repositoryRoot, "browser-runtime/runtime-lock.json"), "utf8"));
const configs = new Map();
for (const name of (await readdir(configRoot)).filter((entry) => entry.endsWith(".json")).sort()) {
  configs.set(name, JSON.parse(await readFile(resolve(configRoot, name), "utf8")));
}
const BRIDGE = "node_modules/@drowse/web-llm/lib/index.js";

/** evidence-config.schema.json's constraints, read from the schema itself. */
function schemaFailures(config) {
  const failures = [];
  const properties = schema.properties;
  const keys = Object.keys(config);
  for (const key of schema.required) if (!keys.includes(key)) failures.push(`missing ${key}`);
  for (const key of keys) if (!Object.hasOwn(properties, key)) failures.push(`unknown ${key}`);
  if (config.$schema !== properties.$schema.const) failures.push("$schema");
  if (config.schemaVersion !== properties.schemaVersion.const) failures.push("schemaVersion");
  if (!properties.evidenceType.enum.includes(config.evidenceType)) failures.push("evidenceType");
  if (typeof config.observer !== "string" || !new RegExp(properties.observer.pattern, "u").test(config.observer)) {
    failures.push("observer");
  }
  const set = schema.$defs.argumentSet;
  const validSet = (value) =>
    Array.isArray(value) && value.length <= set.maxItems &&
    value.every((item) => typeof item === "string" && item.length >= set.items.minLength && item.length <= set.items.maxLength);
  const [single, many] = properties.arguments.oneOf;
  assert.equal(single.$ref, "#/$defs/argumentSet");
  const asSingle = validSet(config.arguments);
  const asMany = Array.isArray(config.arguments) && config.arguments.length >= many.minItems &&
    config.arguments.length <= many.maxItems && config.arguments.every(validSet);
  if (asSingle === asMany) failures.push("arguments match neither or both argument forms");
  return failures;
}

function argumentValue(args, flag) {
  const index = args.indexOf(flag);
  return index < 0 ? null : args[index + 1];
}

test("every config validates against evidence-config.schema.json and names an observer in its closure", () => {
  assert.ok(configs.size > 0);
  for (const [name, config] of configs) {
    assert.deepEqual(schemaFailures(config), [], name);
    const producer = requireReleaseEvidenceProducer(config.evidenceType);
    assert.ok(producer.closure.includes(config.observer), `${name}: ${config.observer} is outside its closure`);
    assert.notEqual(config.observer, producer.executable, name);
  }
  // The schema check refuses what it should.
  const base = configs.get("cvm-goldens.json");
  assert.match(schemaFailures({ ...base, extra: 1 }).join(), /unknown extra/);
  assert.match(schemaFailures({ ...base, observer: "/abs/observer.py" }).join(), /observer/);
  assert.match(schemaFailures({ ...base, arguments: [[]].concat(Array(8).fill([])) }).join(), /neither or both/);
  assert.match(schemaFailures({ ...base, arguments: [""] }).join(), /neither or both/);
});

test("the C VM goldens run on jobe's source-built toolchain over the four fixture libraries", () => {
  const config = configs.get("cvm-goldens.json");
  assert.equal(config.evidenceType, "cVmGoldens");
  assert.equal(config.observer, "browser-runtime/record-cvm-goldens.py");
  for (const flag of ["--python", "--mlc-repository", "--tvm-repository", "--units-fixtures"]) {
    assert.match(argumentValue(config.arguments, flag) ?? "", /^\/home\/jobe\/drowse-toolchain\//u, flag);
  }
});

test("the WebGPU goldens run the vendored bridge once per fixture library", () => {
  const config = configs.get("webgpu-goldens.json");
  assert.equal(config.evidenceType, "webGpuGoldens");
  assert.equal(config.observer, "webui/scripts/browser-units-goldens.mjs");
  const libraries = config.arguments.map((args) => {
    assert.equal(argumentValue(args, "--webllm"), BRIDGE);
    assert.ok(args.includes("--report"));
    assert.ok(argumentValue(args, "--fixtures"));
    return `${argumentValue(args, "--arch")}-${argumentValue(args, "--quantization")}`;
  });
  assert.deepEqual(libraries.sort(), ["gemma3-q0f32", "gemma3-q4f32_1", "qwen3-q0f32", "qwen3-q4f32_1"]);
});

test("one productionLibrary config per lock model, on the repository lock, with its profile's packs", () => {
  const ids = [];
  for (const [name, config] of configs) {
    if (config.evidenceType !== "productionLibrary") continue;
    const options = parseBrowserFullRuntimeArguments(config.arguments);
    const model = runtimeLock.models.find((entry) => entry.id === options.modelId);
    assert.ok(model, `${name} names a lock model`);
    assert.equal(name, `production-library-${model.id}.json`);
    assert.equal(options.webLlm, BRIDGE, name);
    assert.equal(options.runtimeLock, null, `${name}: records bind the repository lock`);
    assert.equal(options.browserFlags, "normal", name);
    assert.equal(options.verbose, false, `${name}: an observer writes only its report`);
    assert.ok(model.contextProfiles.includes(options.contextTokens), `${name}: an advertised context`);
    assert.equal(options.saeDirectory, null, `${name}: the library record runs without SAE`);
    const packs = validateBrowserFullRuntimePackSets(model.structuredHookProfile, {
      jlens: options.jlensDirectory,
      slens: options.slensDirectory,
      units: options.unitsDirectory,
      sae: options.saeDirectory,
    });
    assert.deepEqual(packs, model.structuredHookProfile === "standard-v4" ? ["jlens", "slens", "units"] : [], name);
    ids.push(model.id);
  }
  assert.deepEqual(ids.sort(), runtimeLock.models.map((entry) => entry.id).sort());
});

test("one productionQ4Parity config per q4f32_1 lock model, over its capture", () => {
  const ids = [];
  for (const [name, config] of configs) {
    if (config.evidenceType !== "productionQ4Parity") continue;
    assert.equal(config.observer, "browser-runtime/validate_mlc_q4_parity.py");
    const [modelDirectory, captureDirectory] = config.arguments;
    const id = name.replace(/^production-q4-parity-/u, "").replace(/\.json$/u, "");
    const model = runtimeLock.models.find((entry) => entry.id === id);
    assert.ok(model, name);
    assert.equal(model.quantization, "q4f32_1", name);
    assert.ok(modelDirectory.endsWith(`/${id}/model`) && captureDirectory.endsWith(`/${id}/q4-capture`), name);
    assert.equal(argumentValue(config.arguments, "--evidence-type"), "productionQ4Parity");
    assert.equal(
      argumentValue(config.arguments, "--fixture-path"),
      requireReleaseEvidenceProducer("productionQ4Parity").fixturePath,
    );
    ids.push(id);
  }
  assert.deepEqual(
    ids.sort(),
    runtimeLock.models.filter((entry) => entry.quantization === "q4f32_1").map((entry) => entry.id).sort(),
  );
});
