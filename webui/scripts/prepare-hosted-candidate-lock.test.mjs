import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import {
  adoptedRuntimeLock,
  deriveCandidateRuntimeLock,
  parseArguments,
} from "./prepare-hosted-candidate-lock.mjs";
import { validateRuntimeLock } from "./runtime-lock-document.mjs";
import { lockedModelExecutionProfiles } from "./hosted-model-profiles.mjs";

const repositoryRoot = resolve(import.meta.dirname, "../..");
const runtimeLock = JSON.parse(
  await readFile(resolve(repositoryRoot, "browser-runtime/runtime-lock.json"), "utf8"),
);

// The converted closure W4 assembles for one model, consistent with the lock.
function closure(model, { library = "b".repeat(64), converted = "c".repeat(64), ...fields } = {}) {
  return {
    schemaVersion: 1,
    runtimeAbi: runtimeLock.runtimeAbi,
    hookAbi: runtimeLock.hookAbi,
    structuredHookProfile: model.structuredHookProfile,
    thinkingProfile: model.thinkingProfile,
    architecture: model.architecture,
    ...(model.modelType ? { modelType: model.modelType } : {}),
    quantization: model.quantization,
    contextWindowSize: Math.max(...model.contextProfiles),
    prefillChunkSize: Math.min(...model.contextProfiles),
    hiddenSize: model.hiddenSize,
    layerMap: model.layerMap,
    files: [
      { path: "model.wasm", role: "model_library", sha256: library },
      { path: "tensor-cache.json", role: "converted_manifest", sha256: converted },
      { path: "tokenizer.json", role: "tokenizer", sha256: model.tokenizerSha256 },
      { path: "tokenizer_config.json", role: "chat_template", sha256: model.chatTemplateSha256 },
    ],
    ...fields,
  };
}

// The mlc-chat-config.json the overlay reads the descriptor's block constants from.
function chatConfig(model) {
  const spec = model.blockSpec;
  if (!spec) return { model_config: {} };
  if (model.architecture === "qwen3") {
    return { model_config: {
      intermediate_size: spec.dFF, num_attention_heads: spec.nHeads,
      num_key_value_heads: spec.nKvHeads, head_dim: spec.headDim, rms_norm_eps: spec.rmsEps,
      rope_theta: spec.ropeTheta,
    } };
  }
  return { model_config: { text_config: {
    hidden_size: Math.round(spec.embedScale ** 2), intermediate_size: spec.dFF,
    num_attention_heads: spec.nHeads, num_key_value_heads: spec.nKvHeads, head_dim: spec.headDim,
    rms_norm_eps: spec.rmsEps, query_pre_attn_scalar: Math.round(spec.attnScaling ** -2),
    position_embedding_base: spec.ropeTheta, rope_local_base_freq: spec.ropeLocalTheta,
    rope_scaling: spec.ropeGlobalScale === 1 ? null : { factor: 1 / spec.ropeGlobalScale, rope_type: "linear" },
    sliding_window_size: spec.slidingWindow, sliding_window_pattern: spec.slidingPattern,
  } } };
}

const model = (id) => runtimeLock.models.find((entry) => entry.id === id);
const derive = (lock, id, options = {}) => deriveCandidateRuntimeLock(lock, id, {
  manifest: closure(model(id), options.closure),
  chatConfig: options.chatConfig ?? chatConfig(model(id)),
  convertedRevision: options.convertedRevision ?? null,
});

// One model filled: its library and converted manifest digests, nothing else.
const qwen = "qwen3-1.7b";
const candidate = derive(runtimeLock, qwen);
const filled = candidate.models.find((entry) => entry.id === qwen);
assert.equal(filled.librarySha256, "b".repeat(64));
assert.equal(filled.manifestSha256, "c".repeat(64));
assert.equal(filled.convertedRevision, null);
assert.equal(candidate.status, "feasibility-required");
const normalized = structuredClone(candidate);
Object.assign(normalized.models.find((entry) => entry.id === qwen), {
  librarySha256: model(qwen).librarySha256,
  manifestSha256: model(qwen).manifestSha256,
});
assert.deepEqual(normalized, runtimeLock);

// Every refusal: ABI, profile, architecture, tokenizer drift, a block spec the converted
// configuration would not attest, a closure without its library.
const refusals = [
  [{ closure: { hookAbi: "post-block-residual-v4" } }, /hookAbi differs/],
  [{ closure: { structuredHookProfile: "standard-v4-residual" } }, /structuredHookProfile differs/],
  [{ closure: { architecture: "llama" } }, /architecture differs/],
  [{ closure: { contextWindowSize: 2048 } }, /contextWindowSize differs/],
  [{ closure: { layerMap: [0, 1] } }, /layerMap differs/],
  [{ closure: { prefillChunkSize: 4096 } }, /prefillChunkSize differs/],
  [{ closure: { files: closure(model(qwen)).files.map((file) => file.role === "tokenizer" ? { ...file, sha256: "d".repeat(64) } : file) } }, /tokenizer digest differs/],
  [{ closure: { files: closure(model(qwen)).files.filter((file) => file.role !== "model_library") } }, /no unique model\.wasm/],
  [{ chatConfig: { model_config: { ...chatConfig(model(qwen)).model_config, rope_theta: 10_000 } } }, /block spec differs from its converted configuration/],
  [{ chatConfig: { model_config: { ...chatConfig(model(qwen)).model_config, rope_scaling: { factor: 2 } } } }, /without scaling or a window/],
];
for (const [options, pattern] of refusals) {
  assert.throws(() => derive(runtimeLock, qwen, options), pattern);
}
// A Gemma 3 block spec differs in float32 only when the configuration does.
const gemma = "gemma3-4b-instruct";
assert.doesNotThrow(() => derive(runtimeLock, gemma));
const scaled = chatConfig(model(gemma));
scaled.model_config.text_config.rope_scaling = null;
assert.throws(() => derive(runtimeLock, gemma, { chatConfig: scaled }), /block spec differs/);
assert.throws(
  () => deriveCandidateRuntimeLock(runtimeLock, "missing", { manifest: closure(model(qwen)), chatConfig: {} }),
  /unknown runtime-lock model/,
);

// Chained over all nine (residual libraries carry no block spec), then adopted as verified.
assert.throws(() => adoptedRuntimeLock(candidate), /candidate runtime lock is incomplete/);
let chained = runtimeLock;
for (const [index, entry] of runtimeLock.models.entries()) {
  chained = derive(chained, entry.id, {
    convertedRevision: String(index + 1).repeat(40),
    closure: { library: `${index}`.repeat(64), converted: "e".repeat(64) },
  });
}
const adopted = adoptedRuntimeLock(chained);
// Recompiling a library after publication clears its revision until the new one is published.
const recompiled = derive(adopted, qwen, { closure: { library: "f".repeat(64), converted: "e".repeat(64) } });
assert.equal(recompiled.models.find((entry) => entry.id === qwen).convertedRevision, null);
assert.throws(() => adoptedRuntimeLock(recompiled), /incomplete/);
const unchanged = derive(adopted, qwen, {
  closure: { library: adopted.models.find((entry) => entry.id === qwen).librarySha256, converted: "e".repeat(64) },
});
assert.equal(
  unchanged.models.find((entry) => entry.id === qwen).convertedRevision,
  adopted.models.find((entry) => entry.id === qwen).convertedRevision,
);
// One LLVM archive per compiler platform.
assert.throws(
  () => adoptedRuntimeLock({ ...adopted, toolchain: { ...adopted.toolchain, llvmArtifacts: adopted.toolchain.llvmArtifacts.slice(0, 1) } }),
  /one archive for each of linux\/amd64 and linux\/arm64/,
);
assert.equal(adopted.status, "verified");
assert.equal(adopted.models.every((entry) => entry.librarySha256 && entry.convertedRevision), true);
assert.doesNotThrow(() => validateRuntimeLock(adopted));

// The lock still describes base-model architectures and their execution profiles.
for (const id of ["gpt2-base", "pythia-70m-base", "qwen35-2b-base"]) {
  const pin = model(id);
  assert.equal(lockedModelExecutionProfiles(pin, pin).structuredHookProfile, "standard-v4-residual");
  const base = structuredClone(runtimeLock);
  delete base.models.find((entry) => entry.id === id).modelType;
  assert.throws(() => validateRuntimeLock(base), /invalid|unknown or missing/);
}
const gpt2 = structuredClone(runtimeLock);
gpt2.models.find((entry) => entry.id === "gpt2-base").contextProfiles = [2048];
assert.throws(() => validateRuntimeLock(gpt2), /context profiles/);

assert.throws(() => parseArguments([qwen, "model"]), /MODEL_ID PUBLISH_DIR OUTPUT\.json/);
assert.throws(() => parseArguments([qwen, "model", "out.json", "--converted-revision", "abc"]), /40-character/);
assert.deepEqual(
  parseArguments([qwen, "model", "candidate.json", "--base", "base.json", "--converted-revision", "f".repeat(40)]),
  {
    modelId: qwen,
    modelDirectory: resolve("model"),
    outputPath: resolve("candidate.json"),
    basePath: resolve("base.json"),
    convertedRevision: "f".repeat(40),
  },
);
assert.deepEqual(parseArguments(["--adopt", "candidate.json"]), { adopt: resolve("candidate.json") });

console.log("Hosted candidate runtime-lock checks passed");
