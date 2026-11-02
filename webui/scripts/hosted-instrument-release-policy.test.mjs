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
// The release corpus feeds only the SAE builder (schema 2): no lens corpus, words or tokenizers.
for (const manifestName of [
  "fineweb-sample-10bt-train.approved.json",
  "fineweb-sample-10bt-train.candidate.json",
  "wikitext-103-raw-test.candidate.json",
]) {
  const manifest = JSON.parse(await readFile(
    resolve(repositoryRoot, "browser-runtime/release-corpus", manifestName),
    "utf8",
  ));
  assert.equal(manifest.schemaVersion, 2);
  for (const retired of ["wordSeeds", "tokenizers"]) assert.equal(retired in manifest, false, `${manifestName} ${retired}`);
  for (const retired of ["jlensDocuments", "minimumSharedWords"]) assert.equal(retired in manifest.selection, false);
}
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
const saeBytes = Buffer.from(JSON.stringify(
  Array.from({ length: source.selection.saeDocuments }, (_, index) => `sae document ${index}`),
));
const release = {
  $schema: "https://polythetic.ai/schemas/release-corpus-v2.json",
  schemaVersion: 2,
  algorithm: "drowse-release-corpus-v1",
  generatorSha256: sha256(generatorBytes),
  sourceManifestSha256: sha256(updatedSourceBytes),
  source: source.source,
  license: source.license,
  selection: source.selection,
  outputs: {
    sae: output("sae-corpus.json", saeBytes, source.selection.saeDocuments),
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
  corpusBytes: saeBytes,
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
    corpusBytes: Buffer.from("[]"),
  }),
  /sae corpus bytes differ/,
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
// The release closure is the pinned corpus and its attribution review: none ships while the
// checked-in review is pending (attributionApproval null), and an approved review passes.
const approval = {
  reviewStatus: "approved", reviewedBy: "a9lim", reviewedAt: "2026-11-02T00:00:00Z",
  attribution: "FineWeb-Edu (HuggingFaceFW), ODC-By 1.0",
};
const approvedCorpus = { ...nativeCorpus, attributionApproval: approval };
// Each native pack carries the approved notice beside its record (build_hosted_packs.py).
const attributionText = (corpus) => [
  "# Corpus attribution", corpus.attributionApproval.attribution,
  `- Dataset: ${corpus.dataset.url}`, `- License: ${corpus.dataset.license}`, `- Corpus: \`${corpus.corpusSpec}\``,
].join("\n\n");
for (const kind of ["jlens", "slens", "units"]) {
  await mkdir(join(root, `native-${kind}`, "packs", kind), { recursive: true });
  await writeFile(join(root, `native-${kind}`, "packs", kind, "ATTRIBUTION.md"), attributionText(approvedCorpus));
}
for (const kind of ["jlens", "slens", "units"]) {
  await assert.rejects(
    validateReleaseInstrumentPack({
      directory: join(root, `native-${kind}`), kind,
      instrumentManifest: nativeLens(kind === "slens" ? "S" : "R"),
      vocabularyManifest: { words: [{ word: "yes", row: 0, token_id: 1 }] }, runtimeLock, modelLock,
      ...(nativeCorpus.attributionApproval === null ? {} : { nativeLensCorpus: { ...nativeCorpus, attributionApproval: null } }),
    }),
    /attribution review is not approved/,
    `${kind} waits for the corpus review`,
  );
}
for (const [label, corpus] of [
  ["a pending review", { ...approvedCorpus, attributionApproval: { ...approval, reviewStatus: "pending" } }],
  ["no reviewer", { ...approvedCorpus, attributionApproval: { ...approval, reviewedBy: "" } }],
  ["an unpinned dataset", { ...approvedCorpus, dataset: { ...nativeCorpus.dataset, revision: "main" } }],
  ["a spec off the dataset", { ...approvedCorpus, corpusSpec: "hf:other/x@" + "0".repeat(40) + ";preprocess=1" }],
]) {
  await assert.rejects(
    validateReleaseInstrumentPack({
      directory: join(root, "native-jlens"), kind: "jlens", instrumentManifest: nativeLens("R"),
      vocabularyManifest: { words: [{ word: "yes", row: 0, token_id: 1 }] }, runtimeLock, modelLock,
      nativeLensCorpus: corpus,
    }),
    /native lens corpus/,
    label,
  );
}
assert.deepEqual(await validateReleaseInstrumentPack({
  directory: join(root, "native-jlens"),
  kind: "jlens",
  instrumentManifest: nativeLens("R"),
  vocabularyManifest: { words: [{ word: "yes", row: 0, token_id: 1 }] },
  runtimeLock,
  modelLock,
  nativeLensCorpus: approvedCorpus,
}), {
  source: "native", family: "R", corpusSpec: nativeCorpus.corpusSpec, attributionApproval: approval,
  prompts: 64, sequenceLength: 128, layers: { out: depth - 1, mid: depth }, words: 1,
});
assert.equal((await validateReleaseInstrumentPack({
  directory: join(root, "native-slens"), kind: "slens", instrumentManifest: nativeLens("S"), runtimeLock, modelLock,
  nativeLensCorpus: approvedCorpus,
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
      nativeLensCorpus: approvedCorpus,
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
    nativeLensCorpus: approvedCorpus,
  }),
  /at least 64 documents/,
);
// A pack without the notice, or with another one, does not ship.
await writeFile(join(root, "native-slens", "packs", "slens", "ATTRIBUTION.md"), "FineWeb-Edu");
await assert.rejects(
  validateReleaseInstrumentPack({
    directory: join(root, "native-slens"), kind: "slens", instrumentManifest: nativeLens("S"), runtimeLock, modelLock,
    nativeLensCorpus: approvedCorpus,
  }),
  /ATTRIBUTION\.md does not carry the approved corpus attribution/,
);
await assert.rejects(
  validateReleaseInstrumentPack({
    directory: join(root, "native-missing"), kind: "slens", instrumentManifest: nativeLens("S"), runtimeLock, modelLock,
    nativeLensCorpus: approvedCorpus,
  }),
  /slens pack corpus attribution is missing/,
);
const unitsRecord = {
  format: "drowse-units-pack-v1", model_layer_count: depth, d_model: modelLock.hiddenSize,
  layers: Object.fromEntries(every(depth).map((layer) => [String(layer), {}])), null: { rank: 6 },
  binding: {
    r_lens_identity: "a".repeat(64), s_lens_identity: "b".repeat(64),
    embedding_statistics_sha256: "c".repeat(64), neutral: { sha256: "d".repeat(64), rows_sha256: "e".repeat(64) },
  },
};
assert.deepEqual(await validateReleaseInstrumentPack({
  directory: join(root, "native-units"), kind: "units", instrumentManifest: unitsRecord, runtimeLock, modelLock,
  nativeLensCorpus: approvedCorpus,
}), { source: "native", layers: depth, nullRank: 6, attributionApproval: approval });
for (const neutral of [{}, { sha256: "d".repeat(64) }]) {
  await assert.rejects(
    validateReleaseInstrumentPack({
      directory: join(root, "native-units"), kind: "units",
      instrumentManifest: { ...unitsRecord, binding: { ...unitsRecord.binding, neutral } }, runtimeLock, modelLock,
      nativeLensCorpus: approvedCorpus,
    }),
    /bind its lenses/,
  );
}

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
// A provider R lens is only for a model drowse cannot fit (GPT-2, Pythia); a Gemma 3 or Qwen 3
// pack calling itself one would skip the native corpus and depth gates.
const providerLock = runtimeLock.models.find((model) => model.id === "gpt2-base");
const providerLens = (lock) => ({
  d_model: lock.hiddenSize,
  estimator_policy: { method: "provider_jacobian_lens" },
  format_version: 7,
  model_layer_count: lock.layerMap.length,
  n_prompts: 278,
  seq_len: 128,
  sites: { out: { layers: selectHostedJlensLayers(lock.layerMap, lock.hiddenSize) } },
});
assert.deepEqual(await validateReleaseInstrumentPack({
  directory: join(root, "provider-jlens"),
  kind: "jlens",
  instrumentManifest: providerLens(providerLock),
  vocabularyManifest: { words: [{ word: "yes", row: 0, token_id: 1 }] },
  runtimeLock,
  modelLock: providerLock,
}), {
  source: "provider",
  prompts: 278,
  sequenceLength: 128,
  layers: selectHostedJlensLayers(providerLock.layerMap, providerLock.hiddenSize).length,
  words: 1,
});
await assert.rejects(
  validateReleaseInstrumentPack({
    directory: join(root, "provider-jlens"), kind: "jlens",
    instrumentManifest: { ...providerLens(modelLock), corpus_spec: nativeCorpus.corpusSpec },
    vocabularyManifest: { words: [{ word: "yes", row: 0, token_id: 1 }] }, runtimeLock, modelLock,
  }),
  /native fit, not a provider lens/,
  "a natively fitted model's R lens cannot pass as a provider's",
);
for (const kind of ["slens", "units"]) {
  await assert.rejects(
    validateReleaseInstrumentPack({
      directory: join(root, `native-${kind}`), kind, instrumentManifest: nativeLens("S"),
      runtimeLock, modelLock: providerLock, nativeLensCorpus: approvedCorpus,
    }),
    /drowse cannot fit it/,
    `${kind} only for a natively fitted model`,
  );
}

// A release corpus that still names the retired lens outputs or words is refused.
for (const [label, edit] of [
  ["a lens output", (value) => { value.outputs.jlens = value.outputs.sae; }],
  ["word seeds", (value) => { value.wordSeeds = {}; }],
  ["schema 1", (value) => { value.schemaVersion = 1; }],
]) {
  const retired = structuredClone(release);
  edit(retired);
  await writeFile(releaseManifestPath, JSON.stringify(retired));
  await assert.rejects(
    validateReleaseCorpusInputs({
      releaseManifestPath, sourceManifestPath, licenseEvidencePath: join(root, "license-evidence"),
      corpusBytes: saeBytes,
    }),
    /release corpus (manifest has|manifest identity|outputs)/,
    label,
  );
}

console.log("Hosted instrument release-policy checks passed: the SAE release corpus, malformed closures, native full-depth lenses and units with their attribution, and provider packs");

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
