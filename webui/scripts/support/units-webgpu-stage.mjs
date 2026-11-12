// B4's tiny hook ABI v5 fixture library staged as a hosted base model: what the WebGPU harnesses
// that load it through the hosted runtime share (browser-core-build.mjs, browser-units-worker.mjs).
//
// FIXTURES holds build-units-fixtures.py's model directory `units-<arch>-<quantization>` and its
// library `libs/units-<arch>-<quantization>-v5.wasm`. The model is staged under WORK as a base
// model (a tokenizer_config.json without a chat template, the capture prefix the hosted loader
// asks a base model for) or, with `chat`, as an instruct model carries one: drowse's fixture
// chat template (tests/support/tiny_models.py: ChatML for qwen3, Gemma's for gemma3) in
// tokenizer_config.json and the conversation template its production family ships
// (`chatConversation`); the block spec comes from drowse's fixture config, the runtime identity
// from the staged files' digests.
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { copyFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const goldensDirectory = fileURLToPath(new URL("../../../drowse/tests/fixtures/browser", import.meta.url));
export const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");

export async function stageUnitsFixture({ fixtures, arch, quantization, work, chat = false, python = "python3" }) {
  const fixture = `units-${arch}-${quantization}`;
  const sourceDirectory = resolve(fixtures, fixture);
  const libraryPath = resolve(fixtures, "libs", `${fixture}-v5.wasm`);
  const modelDirectory = resolve(work, chat ? `${fixture}-chat` : fixture);
  await mkdir(modelDirectory, { recursive: true });
  const chatConfig = JSON.parse(await readFile(resolve(sourceDirectory, "mlc-chat-config.json"), "utf8"));
  chatConfig.tokenizer_files = ["tokenizer.json", "tokenizer_config.json"];
  let tokenizerConfig = { chat_template: null };
  if (chat) {
    // An instruct model has no completion prefix; its conversation template renders the turns.
    delete chatConfig.drowse_completion_prefix_token_ids;
    chatConfig.conv_template = chatConversation(arch, chatConfig);
    tokenizerConfig = { chat_template: fixtureChatTemplate(arch, python) };
  } else {
    // A base model's capture prefix is its completion prefix (Gemma's BOS; nothing for Qwen).
    chatConfig.conv_template = {
      ...chatConfig.conv_template,
      system_prefix_token_ids: chatConfig.drowse_completion_prefix_token_ids ?? [],
    };
  }
  await writeFile(resolve(modelDirectory, "mlc-chat-config.json"), JSON.stringify(chatConfig, null, 2));
  await writeFile(resolve(modelDirectory, "tokenizer_config.json"), JSON.stringify(tokenizerConfig));
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
    runtimeAbi: "drowse-web-runtime-v3",
    hookAbi: "block-sites-v6",
    hiddenSize,
    layerMap: Array.from({ length: layerCount }, (_, layer) => layer),
  };
  return {
    fixture, modelDirectory, roles, artifacts, blockSpec, layerCount, hiddenSize,
    contextTokens: chatConfig.context_window_size, prefillChunkSize: chatConfig.prefill_chunk_size,
    runtimeIdentity,
  };
}

/** drowse's chat template for the fixture's family, as its fixture tokenizer carries it. */
function fixtureChatTemplate(arch, python) {
  const family = { qwen3: "qwen3", gemma3: "gemma3_text" }[arch];
  return JSON.parse(execFileSync(python, ["-c",
    "import json, sys; from tests.support.tiny_models import FAMILIES; print(json.dumps(FAMILIES[sys.argv[1]].chat_template))",
    family,
  ], { cwd: fileURLToPath(new URL("../../../drowse", import.meta.url)), encoding: "utf8" }));
}

/** The conversation template an instruct model of the fixture's family ships with, its special
 *  tokens the fixture's: qwen3's ChatML (MLC `qwen3`; no system block without a system message)
 *  and Gemma 3's turns (MLC `gemma3_instruction`, BOS as the prefix, the system message folded
 *  into the first turn). The fixture's one end-of-turn token is its only stop token, as drowse's
 *  fixture generation config has it. */
export function chatConversation(arch, config) {
  const turnTemplates = { user: "{user_message}", assistant: "{assistant_message}", tool: "{tool_message}" };
  const common = {
    add_role_after_system_message: true, role_templates: turnTemplates, messages: [],
    role_content_sep: "\n", role_empty_sep: "\n", function_string: "", use_function_calling: false,
    stop_token_ids: [config.eos_token_id],
  };
  if (arch === "qwen3") {
    return {
      ...common, name: "qwen3", system_template: "<|im_start|>system\n{system_message}<|im_end|>\n", system_message: "",
      system_prefix_token_ids: null, roles: { user: "<|im_start|>user", assistant: "<|im_start|>assistant" },
      seps: ["<|im_end|>\n"], stop_str: ["<|im_end|>"],
    };
  }
  return {
    ...common, name: "gemma3_instruction", system_template: "{system_message}", system_message: "",
    system_prefix_token_ids: [config.bos_token_id], drowse_system_fold_sep: "\n\n",
    roles: { user: "<start_of_turn>user", assistant: "<start_of_turn>model" },
    seps: ["<end_of_turn>\n"], stop_str: ["<end_of_turn>"],
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
