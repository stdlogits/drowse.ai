import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import {
  validateReleaseCorpusInputs,
  validateReleaseInstrumentPack,
} from "./hosted-instrument-release-policy.mjs";
import { selectHostedJlensLayers } from "./hosted-instrument-layers.mjs";

const repositoryRoot = resolve(import.meta.dirname, "../..");
const runtimeLock = JSON.parse(await readFile(
  resolve(repositoryRoot, "browser-runtime/runtime-lock.json"),
  "utf8",
));
const modelLock = runtimeLock.models.find((model) => model.id === "qwen3-1.7b");
const source = JSON.parse(await readFile(
  resolve(repositoryRoot, "browser-runtime/release-corpus/wikitext-103-raw-test.candidate.json"),
  "utf8",
));
const lockedTokenizerClosure = [...runtimeLock.models]
  .sort((left, right) => left.id.localeCompare(right.id))
  .map((model) => ({
    id: model.id,
    sourceRepository: model.sourceRepository,
    sourceRevision: model.sourceRevision,
    tokenizerSha256: model.tokenizerSha256,
    tokenizerConfigSha256: model.chatTemplateSha256,
  }));
assert.ok(source.tokenizers.length > 0);
for (const tokenizer of source.tokenizers) {
  assert.deepEqual(tokenizer, lockedTokenizerClosure.find(model => model.id === tokenizer.id));
}
for (const manifestName of [
  "fineweb-sample-10bt-train.approved.json",
  "fineweb-sample-10bt-train.candidate.json",
]) {
  const manifest = JSON.parse(await readFile(
    resolve(repositoryRoot, "browser-runtime/release-corpus", manifestName),
    "utf8",
  ));
  assert.ok(manifest.tokenizers.length > 0);
  for (const tokenizer of manifest.tokenizers) {
    assert.deepEqual(tokenizer, lockedTokenizerClosure.find(model => model.id === tokenizer.id));
  }
}
source.tokenizers = lockedTokenizerClosure;
source.selection.algorithm = "drowse-release-corpus-v1";
source.status = "approved";
source.blockers = [];
source.source.split = "train";
source.source.path = source.source.path.replace("test-", "train-");
source.source.url = source.source.url.replace("/test-", "/train-");
source.license.reviewStatus = "approved";
source.license.reviewedBy = "release-policy fixture";
source.license.reviewedAt = "2026-08-30T00:00:00.000Z";
source.license.redistributionNotice = "Release-policy test fixture; not a distribution grant.";
const generatorBytes = await readFile(
  resolve(repositoryRoot, "webui/scripts/prepare_hosted_release_corpus.py"),
);
const licenseEvidenceBytes = Buffer.from("release-policy license evidence fixture\n");
source.license.evidence.sha256 = sha256(licenseEvidenceBytes);
source.license.evidence.bytes = licenseEvidenceBytes.byteLength;
const updatedSourceBytes = Buffer.from(`${JSON.stringify(source, null, 2)}\n`);
const seedWords = (await readFile(
  resolve(repositoryRoot, "browser-runtime/release-corpus/jlens-word-seeds.txt"),
  "utf8",
)).trim().split("\n").slice(0, 80);
const saeBytes = Buffer.from(JSON.stringify(
  Array.from({ length: source.selection.saeDocuments }, (_, index) => `sae document ${index}`),
));
const jlensBytes = Buffer.from(JSON.stringify(
  Array.from({ length: source.selection.jlensDocuments }, (_, index) => `jlens document ${index}`),
));
const wordBytes = Buffer.from(JSON.stringify(seedWords));
const release = {
  $schema: "https://polythetic.ai/schemas/release-corpus-v1.json",
  schemaVersion: 1,
  algorithm: "drowse-release-corpus-v1",
  generatorSha256: sha256(generatorBytes),
  sourceManifestSha256: sha256(updatedSourceBytes),
  source: source.source,
  license: source.license,
  wordSeeds: source.wordSeeds,
  selection: source.selection,
  outputs: {
    sae: output("sae-corpus.json", saeBytes, source.selection.saeDocuments),
    jlens: output("jlens-corpus.json", jlensBytes, source.selection.jlensDocuments),
    words: {
      path: "jlens-words.json",
      sha256: sha256(wordBytes),
      bytes: wordBytes.byteLength,
      count: seedWords.length,
      candidateCount: 123,
      tokenIds: Object.fromEntries(runtimeLock.models.map((model) => [
        model.id,
        seedWords.map((_, index) => index + 1),
      ])),
    },
  },
};
const releaseBytes = Buffer.from(`${JSON.stringify(release, null, 2)}\n`);
const root = await mkdtemp(join(tmpdir(), "drowse-instrument-release-policy-"));
const releaseManifestPath = join(root, "release-corpus.json");
const sourceManifestPath = join(root, "approved-source.json");
await Promise.all([
  writeFile(releaseManifestPath, releaseBytes),
  writeFile(sourceManifestPath, updatedSourceBytes),
  writeFile(join(root, "license-evidence"), licenseEvidenceBytes),
]);

const inputClosure = await validateReleaseCorpusInputs({
  releaseManifestPath,
  sourceManifestPath,
  licenseEvidencePath: join(root, "license-evidence"),
  kind: "sae",
  corpusBytes: saeBytes,
  runtimeLock,
  modelLock,
});
assert.deepEqual(inputClosure, {
  releaseBytes,
  sourceBytes: updatedSourceBytes,
  licenseEvidenceBytes,
});
await assert.rejects(
  validateReleaseCorpusInputs({
    releaseManifestPath,
    sourceManifestPath,
    licenseEvidencePath: join(root, "license-evidence"),
    kind: "jlens",
    corpusBytes: jlensBytes,
    wordsBytes: Buffer.from("[]"),
    runtimeLock,
    modelLock,
  }),
  /word bytes differ/,
);

const saeDirectory = join(root, "sae");
await Promise.all([
  mkdir(join(saeDirectory, "packs/sae"), { recursive: true }),
]);
for (const directory of [join(saeDirectory, "packs/sae")]) {
  await Promise.all([
    writeFile(join(directory, "release-corpus.json"), releaseBytes),
    writeFile(join(directory, "release-corpus-source.json"), updatedSourceBytes),
    writeFile(join(directory, "release-license-evidence"), licenseEvidenceBytes),
  ]);
}
const saeManifest = {
  corpus_sha256: release.outputs.sae.sha256,
  d_model: modelLock.hiddenSize,
  d_sae: Math.min(modelLock.hiddenSize * 4, 8192),
  tokens_trained: 50_000,
  seq_len: 128,
};
assert.equal((await validateReleaseInstrumentPack({
  directory: saeDirectory,
  kind: "sae",
  instrumentManifest: saeManifest,
  runtimeLock,
  modelLock,
})).tokensTrained, 50_000);
await assert.rejects(
  validateReleaseInstrumentPack({
    directory: saeDirectory,
    kind: "sae",
    instrumentManifest: { ...saeManifest, tokens_trained: 49_999 },
    runtimeLock,
    modelLock,
  }),
  /release SAE requires/,
);

// Native R and S lenses: full depth at both sites, split-half at every layer, the pinned corpus.
const nativeCorpus = JSON.parse(await readFile(
  new URL("../../browser-runtime/release-corpus/native-lens-corpus.json", import.meta.url),
  "utf8",
));
const depth = modelLock.layerMap.length;
const every = (count) => Array.from({ length: count }, (_, layer) => layer);
const site = (layers) => ({
  layers,
  split_half: Object.fromEntries(layers.map((layer) => [String(layer), 0.1])),
});
const nativeLens = (kind) => ({
  format_version: 7, kind, dtype: "float32", checkpoint: false,
  corpus_spec: nativeCorpus.corpusSpec, corpus_hash_kind: "token_ids_v1",
  d_model: modelLock.hiddenSize, model_layer_count: depth,
  estimator_policy: { direction: kind === "R" ? "output" : "input" },
  sites: { out: site(every(kind === "R" ? depth - 1 : depth)), mid: site(every(depth)) },
  n_prompts: 64, usable_prompt_count: 64, seq_len: 128,
});
assert.deepEqual(await validateReleaseInstrumentPack({
  directory: join(root, "native-jlens"),
  kind: "jlens",
  instrumentManifest: nativeLens("R"),
  vocabularyManifest: { words: [{ word: "yes", row: 0, token_id: 1 }] },
  runtimeLock,
  modelLock,
}), {
  source: "native", family: "R", corpusSpec: nativeCorpus.corpusSpec, attributionApproval: null,
  prompts: 64, sequenceLength: 128, layers: { out: depth - 1, mid: depth }, words: 1,
});
assert.equal((await validateReleaseInstrumentPack({
  directory: join(root, "native-slens"), kind: "slens", instrumentManifest: nativeLens("S"), runtimeLock, modelLock,
})).layers.out, depth);
for (const [label, change] of [
  ["an eight-layer pack", (manifest) => { manifest.sites.out = site([0, 1, 2, 3, 4, 5, 6, 7]); }],
  ["no mid site", (manifest) => { delete manifest.sites.mid; }],
  ["no split-half", (manifest) => { delete manifest.sites.mid.split_half; }],
  ["another corpus", (manifest) => { manifest.corpus_spec = "hf:other"; }],
  ["the S kind in an R pack", (manifest) => { manifest.kind = "S"; }],
]) {
  const manifest = nativeLens("R");
  change(manifest);
  await assert.rejects(
    validateReleaseInstrumentPack({
      directory: join(root, "native-jlens"), kind: "jlens", instrumentManifest: manifest,
      vocabularyManifest: { words: [{ word: "yes", row: 0, token_id: 1 }] }, runtimeLock, modelLock,
    }),
    /full-depth native lens/,
    label,
  );
}
await assert.rejects(
  validateReleaseInstrumentPack({
    directory: join(root, "native-jlens"), kind: "jlens",
    instrumentManifest: { ...nativeLens("R"), n_prompts: 32, usable_prompt_count: 32 },
    vocabularyManifest: { words: [{ word: "yes", row: 0, token_id: 1 }] }, runtimeLock, modelLock,
  }),
  /at least 64 documents/,
);
const unitsRecord = {
  format: "drowse-units-pack-v1", model_layer_count: depth, d_model: modelLock.hiddenSize,
  layers: Object.fromEntries(every(depth).map((layer) => [String(layer), {}])), null: { rank: 6 },
  binding: {
    r_lens_identity: "a".repeat(64), s_lens_identity: "b".repeat(64),
    embedding_statistics_sha256: "c".repeat(64), neutral: { sha256: "d".repeat(64) },
  },
};
assert.deepEqual(await validateReleaseInstrumentPack({
  directory: join(root, "native-units"), kind: "units", instrumentManifest: unitsRecord, runtimeLock, modelLock,
}), { source: "native", layers: depth, nullRank: 6 });
await assert.rejects(
  validateReleaseInstrumentPack({
    directory: join(root, "native-units"), kind: "units",
    instrumentManifest: { ...unitsRecord, binding: { ...unitsRecord.binding, neutral: {} } }, runtimeLock, modelLock,
  }),
  /bind its lenses/,
);

assert.deepEqual(await validateReleaseInstrumentPack({
  directory: join(root, "provider-sae"),
  kind: "sae",
  instrumentManifest: {
    activation: "jump_relu",
    corpus_spec: "provider:fixture/sae",
    d_model: modelLock.hiddenSize,
    d_sae: 16_384,
    layer: modelLock.layerMap[1],
  },
  runtimeLock,
  modelLock,
}), {
  source: "provider",
  layer: modelLock.layerMap[1],
  features: 16_384,
  activation: "jump_relu",
});
assert.deepEqual(await validateReleaseInstrumentPack({
  directory: join(root, "provider-jlens"),
  kind: "jlens",
  instrumentManifest: {
    d_model: modelLock.hiddenSize,
    estimator_policy: { method: "provider_jacobian_lens" },
    format_version: 7,
    model_layer_count: modelLock.layerMap.length,
    n_prompts: 278,
    seq_len: 128,
    sites: { out: { layers: selectHostedJlensLayers(modelLock.layerMap, modelLock.hiddenSize) } },
  },
  vocabularyManifest: { words: [{ word: "yes", row: 0, token_id: 1 }] },
  runtimeLock,
  modelLock,
}), {
  source: "provider",
  prompts: 278,
  sequenceLength: 128,
  layers: selectHostedJlensLayers(modelLock.layerMap, modelLock.hiddenSize).length,
  words: 1,
});

for (const scenario of [
  "scoped", "missing-target", "unknown-model", "duplicate-model", "wrong-tokenizer",
  "missing-token-model", "short-token-list", "negative-token", "duplicate-token", "fractional-token",
]) {
  const scopedSource = structuredClone(source);
  scopedSource.tokenizers = source.tokenizers.filter(tokenizer => tokenizer.id === modelLock.id);
  const scopedRelease = structuredClone(release);
  scopedRelease.outputs.words.tokenIds = { [modelLock.id]: [...release.outputs.words.tokenIds[modelLock.id]] };
  if (scenario === "missing-target") scopedSource.tokenizers = source.tokenizers.filter(tokenizer => tokenizer.id !== modelLock.id);
  if (scenario === "unknown-model") scopedSource.tokenizers.push({ ...scopedSource.tokenizers[0], id: "unlocked-model" });
  if (scenario === "duplicate-model") scopedSource.tokenizers.push({ ...scopedSource.tokenizers[0] });
  if (scenario === "wrong-tokenizer") scopedSource.tokenizers[0].tokenizerSha256 = "0".repeat(64);
  if (scenario === "missing-token-model") scopedRelease.outputs.words.tokenIds = {};
  const tokenIds = scopedRelease.outputs.words.tokenIds[modelLock.id];
  if (scenario === "short-token-list") tokenIds.pop();
  if (scenario === "negative-token") tokenIds[0] = -1;
  if (scenario === "duplicate-token") tokenIds[0] = tokenIds[1];
  if (scenario === "fractional-token") tokenIds[0] = 0.5;
  const sourceBytes = Buffer.from(JSON.stringify(scopedSource));
  scopedRelease.sourceManifestSha256 = sha256(sourceBytes);
  await writeFile(sourceManifestPath, sourceBytes);
  await writeFile(releaseManifestPath, JSON.stringify(scopedRelease));
  const validate = () => validateReleaseCorpusInputs({
    releaseManifestPath, sourceManifestPath, licenseEvidencePath: join(root, "license-evidence"),
    kind: "sae", corpusBytes: saeBytes, runtimeLock, modelLock,
  });
  if (scenario === "scoped") await validate();
  else await assert.rejects(validate(), /token.*closure/);
}

console.log("Hosted instrument release-policy checks passed: scoped model identities, exact word token IDs, malformed closures, native full-depth lenses and units, and provider packs");

function output(path, bytes, documents) {
  return {
    path,
    sha256: sha256(bytes),
    bytes: bytes.byteLength,
    documents,
    characters: bytes.toString("utf8").length,
  };
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}
