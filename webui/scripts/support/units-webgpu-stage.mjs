// B4's tiny hook ABI v5 fixture library staged as a hosted base model: what the WebGPU harnesses
// that load it through the hosted runtime share (browser-core-build.mjs, browser-units-worker.mjs).
//
// FIXTURES holds build-units-fixtures.py's model directory `units-<arch>-<quantization>` and its
// library `libs/units-<arch>-<quantization>-v5.wasm`. The model is staged under WORK as a base
// model (a tokenizer_config.json without a chat template, the capture prefix the hosted loader
// asks a base model for); the block spec comes from drowse's fixture config, the runtime identity
// from the staged files' digests.
import { createHash } from "node:crypto";
import { copyFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const goldensDirectory = fileURLToPath(new URL("../../../drowse/tests/fixtures/browser", import.meta.url));
export const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");

export async function stageUnitsFixture({ fixtures, arch, quantization, work }) {
  const fixture = `units-${arch}-${quantization}`;
  const sourceDirectory = resolve(fixtures, fixture);
  const libraryPath = resolve(fixtures, "libs", `${fixture}-v5.wasm`);
  const modelDirectory = resolve(work, fixture);
  await mkdir(modelDirectory, { recursive: true });
  const chatConfig = JSON.parse(await readFile(resolve(sourceDirectory, "mlc-chat-config.json"), "utf8"));
  chatConfig.tokenizer_files = ["tokenizer.json", "tokenizer_config.json"];
  // A base model's capture prefix is its completion prefix (Gemma's BOS; nothing for Qwen).
  chatConfig.conv_template = {
    ...chatConfig.conv_template,
    system_prefix_token_ids: chatConfig.drowse_completion_prefix_token_ids ?? [],
  };
  await writeFile(resolve(modelDirectory, "mlc-chat-config.json"), JSON.stringify(chatConfig, null, 2));
  await writeFile(resolve(modelDirectory, "tokenizer_config.json"), JSON.stringify({ chat_template: null }));
  const tensorCache = JSON.parse(await readFile(resolve(sourceDirectory, "tensor-cache.json"), "utf8"));
  const shards = tensorCache.records.map((record) => record.dataPath);
  for (const name of ["tensor-cache.json", "tokenizer.json", ...shards]) {
    await copyFile(resolve(sourceDirectory, name), resolve(modelDirectory, name));
  }
  await copyFile(libraryPath, resolve(modelDirectory, "model.wasm"));
  const roles = {
    "mlc-chat-config.json": "configuration",
    "tensor-cache.json": "converted_manifest",
    "tokenizer.json": "tokenizer",
    "tokenizer_config.json": "chat_template",
    "model.wasm": "model_library",
    ...Object.fromEntries(shards.map((shard) => [shard, "weight"])),
  };
  const artifacts = [];
  for (const [path, role] of Object.entries(roles)) {
    const bytes = await readFile(resolve(modelDirectory, path));
    artifacts.push({ path, role, bytes: bytes.length, sha256: sha256(bytes) });
  }
  const artifactSha = (path) => artifacts.find((artifact) => artifact.path === path).sha256;

  const tiny = JSON.parse(await readFile(resolve(goldensDirectory, "units-tiny-models-v1.json"), "utf8")).arches[arch];
  const config = tiny.config;
  const gemma = arch === "gemma3";
  const blockSpec = {
    dFF: config.intermediate_size, nHeads: config.num_attention_heads,
    nKvHeads: config.num_key_value_heads, headDim: config.head_dim, sandwich: gemma,
    normOffset: gemma ? 1 : 0, activation: gemma ? "gelu_tanh" : "silu",
    embedScale: gemma ? Math.sqrt(config.hidden_size) : 1, rmsEps: config.rms_norm_eps, qkNorm: true,
    attnScaling: (gemma ? config.query_pre_attn_scalar : config.head_dim) ** -0.5,
    attnLogitSoftcap: null, finalLogitSoftcap: null,
    ropeTheta: gemma ? config.rope_parameters.full_attention.rope_theta : config.rope_parameters.rope_theta,
    ropeLocalTheta: gemma ? config.rope_parameters.sliding_attention.rope_theta : null,
    ropeGlobalScale: 1, ropeInGraph: gemma,
    slidingWindow: gemma ? config.sliding_window : null, slidingPattern: gemma ? 2 : null,
  };
  const layerCount = config.num_hidden_layers;
  const hiddenSize = config.hidden_size;
  const runtimeIdentity = {
    sourceModel: `drowse/units-tiny-${arch}`,
    sourceRevision: "0".repeat(40),
    convertedManifestSha256: artifactSha("tensor-cache.json"),
    quantization,
    tokenizerSha256: artifactSha("tokenizer.json"),
    chatTemplateSha256: artifactSha("tokenizer_config.json"),
    modelLibrarySha256: artifactSha("model.wasm"),
    runtimeAbi: "drowse-web-runtime-v2",
    hookAbi: "block-sites-v5",
    hiddenSize,
    layerMap: Array.from({ length: layerCount }, (_, layer) => layer),
  };
  return {
    fixture, modelDirectory, roles, artifacts, blockSpec, layerCount, hiddenSize,
    contextTokens: chatConfig.context_window_size, runtimeIdentity,
  };
}

/** The two core concepts the harnesses fit through the release worker (a two-node and a
 *  one-node concept): short rows, since the fixture's byte-level vocabulary spends a token per
 *  character in a 128-token context. */
export function fixtureConcepts() {
  const statements = (word) => [`I feel ${word} now.`, `So ${word} today.`, `A ${word} mood.`, `Very ${word}.`];
  const concept = (name, nodes) => ({
    namespace: "local",
    name,
    description: `fixture concept ${name}`,
    fitMode: "pca",
    hyperparams: { max_dim: 1, var_threshold: 0.7 },
    source: { uri: "local", repository: null, revision: null },
    tags: ["fixture"],
    nodes: nodes.map(([label, rows]) => ({ label, role: null, kind: null, statements: rows })),
  });
  return [
    concept("calm.alert", [["calm", statements("calm")], ["alert", statements("alert")]]),
    concept("deceptive", [["deceptive", statements("sly")]]),
  ];
}
