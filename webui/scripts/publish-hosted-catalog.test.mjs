import assert from "node:assert/strict";
import { generateKeyPairSync, sign } from "node:crypto";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  lockedCatalogModels,
  parseArguments,
  uploadCommitRevision,
  validateCatalogBundle,
  validateCatalogAdvance,
  validatePublishedAdvance,
} from "./publish-hosted-catalog.mjs";
import {
  CATALOG_DIRECTORIES,
  CURRENT_CATALOG_URLS,
  catalogLocation,
  catalogUrls,
  previousCatalogDirectory,
} from "./hosted-catalog-location.mjs";
import { readFile } from "node:fs/promises";
import {
  catalogFixture,
  catalogRuntimeLockModels,
} from "./catalog-fixture.mjs";

assert.deepEqual(parseArguments(["catalog.json", "catalog.sig.json"]), {
  catalogPath: join(process.cwd(), "catalog.json"),
  signaturePath: join(process.cwd(), "catalog.sig.json"),
  previousLockPath: null,
  upload: false,
});
assert.equal(parseArguments(["catalog.json", "catalog.sig.json", "--upload"]).upload, true);
assert.equal(
  parseArguments(["--previous-lock", "old.json", "catalog.json", "catalog.sig.json"]).previousLockPath,
  join(process.cwd(), "old.json"),
);
assert.throws(() => parseArguments(["catalog.json"]), /usage/);
assert.throws(() => parseArguments(["catalog.json", "catalog.sig.json", "--previous-lock"]), /usage/);
assert.throws(
  () => parseArguments(["catalog.json", "catalog.sig.json", "--previous-lock", "--upload"]),
  /usage/,
);
assert.throws(
  () => parseArguments(["catalog.json", "catalog.sig.json", "--upload", "--upload"]),
  /usage/,
);

const lockedModels = lockedCatalogModels({
  runtimeAbi: "drowse-web-runtime-v1",
  hookAbi: "post-block-residual-v4",
  models: [{
    id: "fixture",
    sourceRepository: "fixture/source",
    sourceRevision: "a".repeat(40),
    manifestSha256: "b".repeat(64),
    quantization: "q4f16_1",
    tokenizerSha256: "c".repeat(64),
    chatTemplateSha256: "d".repeat(64),
    librarySha256: "e".repeat(64),
    hiddenSize: 4,
    layerMap: [0, 1],
    structuredHookProfile: "standard-v3",
    thinkingProfile: null,
    convertedRepository: "fixture/converted",
    convertedRevision: "f".repeat(40),
    contextProfiles: [2048],
  }],
});
assert.equal(lockedModels[0].structuredHookProfile, "standard-v3");
assert.equal(lockedModels[0].thinkingProfile, null);

const repository = "stdlogits/drowse-web-catalog";
const distribution = {
  ...catalogUrls("v2/"),
  minimumAcceptedSequence: 7,
  publicKeys: [],
};
// Runtime v2 publishes at its own directory, which the distribution lock names; the root keeps
// what deployed v1 builds read.
assert.deepEqual(CATALOG_DIRECTORIES, ["", "v2/"]);
const lockFile = JSON.parse(await readFile(new URL("../../browser-runtime/distribution-lock.json", import.meta.url), "utf8"));
assert.equal(lockFile.catalogUrl, CURRENT_CATALOG_URLS.catalogUrl);
assert.equal(lockFile.signatureUrl, CURRENT_CATALOG_URLS.signatureUrl);
assert.equal(CURRENT_CATALOG_URLS.catalogUrl, `https://huggingface.co/${repository}/resolve/main/v2/catalog.json`);
assert.deepEqual(catalogLocation(distribution), { repository, directory: "v2/" });
assert.deepEqual(catalogLocation(catalogUrls("")), { repository, directory: "" });
assert.equal(previousCatalogDirectory("v2/"), "");
assert.equal(previousCatalogDirectory(""), null);
assert.throws(
  () => catalogLocation({
    ...distribution,
    signatureUrl: "https://huggingface.co/other/repo/resolve/main/v2/catalog.sig.json",
  }),
  /one directory of one repository/,
);
assert.throws(
  () => catalogLocation({ ...distribution, signatureUrl: catalogUrls("").signatureUrl }),
  /one directory of one repository/,
);
assert.throws(() => catalogLocation(catalogUrls("v3/")), /not a published catalog path/);
assert.throws(
  () => catalogLocation({ ...distribution, catalogUrl: `https://huggingface.co/${repository}/resolve/dev/v2/catalog.json` }),
  /main-branch URL/,
);
assert.throws(
  () => catalogLocation({ ...distribution, catalogUrl: `https://huggingface.co/${repository}/resolve/main/v2/../catalog.json` }),
  /catalog and signature|main-branch URL/,
);
const uploadedRevision = "b".repeat(40);
assert.equal(
  uploadCommitRevision(
    JSON.stringify({ url: `https://huggingface.co/${repository}/commit/${uploadedRevision}` }),
    repository,
  ),
  uploadedRevision,
);
for (const output of [
  "not-json",
  JSON.stringify({ url: `https://huggingface.co/${repository}/commit/${uploadedRevision}`, extra: true }),
  JSON.stringify({ url: `https://huggingface.co/other/repo/commit/${uploadedRevision}` }),
  JSON.stringify({ url: `https://huggingface.co/${repository}/resolve/main/catalog.json` }),
  JSON.stringify({ url: `https://huggingface.co/${repository}/commit/${uploadedRevision}?mutable=1` }),
]) {
  assert.throws(() => uploadCommitRevision(output, repository), /Hugging Face upload/);
}

const root = await mkdtemp(join(tmpdir(), "drowse-catalog-publisher-"));
try {
  const catalogPath = join(root, "catalog.json");
  const signaturePath = join(root, "catalog.sig.json");
  const document = catalogFixture();
  const exactBytes = Buffer.from(`${JSON.stringify(document)}\n`);
  const { privateKey, publicKey } = generateKeyPairSync("ed25519");
  const rawPublicKey = publicKey.export({ type: "spki", format: "der" }).subarray(-32);
  distribution.publicKeys = [{
    keyId: "fixture-key",
    ed25519PublicKeyBase64: rawPublicKey.toString("base64"),
  }];
  const envelope = {
    schemaVersion: 1,
    algorithm: "Ed25519",
    keyId: "fixture-key",
    signature: sign(null, exactBytes, privateKey).toString("base64"),
  };
  const publishedBytes = Buffer.from(JSON.stringify({ ...document, sequence: 6 }));
  const publishedSignature = Buffer.from(JSON.stringify({
    ...envelope,
    signature: sign(null, publishedBytes, privateKey).toString("base64"),
  }));
  assert.equal(validateCatalogAdvance(7, publishedBytes, publishedSignature, distribution.publicKeys), 6);
  for (const sequence of [3, 6, NaN]) {
    assert.throws(
      () => validateCatalogAdvance(sequence, publishedBytes, publishedSignature, distribution.publicKeys),
      /must advance the published sequence/,
    );
  }
  assert.equal(validateCatalogAdvance(7, null, null, distribution.publicKeys), null);
  assert.throws(
    () => validateCatalogAdvance(7, publishedBytes, null, distribution.publicKeys),
    /must both exist/,
  );
  assert.throws(
    () => validateCatalogAdvance(7, publishedBytes, publishedSignature, []),
    /not trusted/,
  );
  assert.throws(
    () => validateCatalogAdvance(7, Buffer.from("{}"), publishedSignature, distribution.publicKeys),
    /signature is invalid/,
  );
  const malformedBytes = Buffer.from(JSON.stringify({ sequence: "6" }));
  assert.throws(
    () => validateCatalogAdvance(7, malformedBytes, Buffer.from(JSON.stringify({
      ...envelope,
      signature: sign(null, malformedBytes, privateKey).toString("base64"),
    })), distribution.publicKeys),
    /sequence is invalid/,
  );
  // The first catalog at a new directory advances the sequence the previous directory published
  // (deployed builds still read it), verified with the published keys; later ones advance their
  // own directory's.
  const published = new Map([["", [publishedBytes, publishedSignature]]]);
  const fetchPair = async (directory) => published.get(directory) ?? [null, null];
  assert.deepEqual(
    await validatePublishedAdvance(7, "v2/", fetchPair, distribution.publicKeys),
    { directory: "", previousSequence: 6 },
  );
  for (const sequence of [5, 6]) {
    await assert.rejects(
      () => validatePublishedAdvance(sequence, "v2/", fetchPair, distribution.publicKeys),
      /must advance the published sequence 6/,
    );
  }
  const { publicKey: strangerKey } = generateKeyPairSync("ed25519");
  const strangerKeys = [{
    keyId: "fixture-key",
    ed25519PublicKeyBase64: strangerKey.export({ type: "spki", format: "der" }).subarray(-32).toString("base64"),
  }];
  await assert.rejects(
    () => validatePublishedAdvance(7, "v2/", fetchPair, strangerKeys),
    /signature is invalid/,
  );
  await assert.rejects(
    () => validatePublishedAdvance(7, "v2/", fetchPair, []),
    /not trusted/,
  );
  published.set("", [publishedBytes, null]);
  await assert.rejects(
    () => validatePublishedAdvance(7, "v2/", fetchPair, distribution.publicKeys),
    /must both exist/,
  );
  published.set("", [publishedBytes, publishedSignature]);
  const v2Bytes = Buffer.from(JSON.stringify({ ...document, sequence: 11 }));
  published.set("v2/", [v2Bytes, Buffer.from(JSON.stringify({
    ...envelope,
    signature: sign(null, v2Bytes, privateKey).toString("base64"),
  }))]);
  assert.deepEqual(
    await validatePublishedAdvance(12, "v2/", fetchPair, distribution.publicKeys),
    { directory: "v2/", previousSequence: 11 },
  );
  await assert.rejects(
    () => validatePublishedAdvance(11, "v2/", fetchPair, distribution.publicKeys),
    /must advance the published sequence 11/,
  );
  // Nothing published before it: only the root directory may start a lineage; a new directory
  // with no verified predecessor is refused.
  assert.deepEqual(
    await validatePublishedAdvance(1, "", async () => [null, null], distribution.publicKeys),
    { directory: null, previousSequence: null },
  );
  await assert.rejects(
    () => validatePublishedAdvance(11, "v2/", async () => [null, null], distribution.publicKeys),
    /first catalog at v2\/ must advance a published catalog at the root/,
  );
  await assert.rejects(
    () => validatePublishedAdvance(1, "v3/", fetchPair, distribution.publicKeys),
    /not a published catalog path/,
  );

  await Promise.all([
    writeFile(catalogPath, exactBytes),
    writeFile(signaturePath, `${JSON.stringify(envelope)}\n`),
  ]);
  const checked = await validateCatalogBundle(
    catalogPath,
    signaturePath,
    distribution,
    catalogRuntimeLockModels(document),
  );
  assert.equal(checked.sequence, 7);
  assert.equal(checked.keyId, "fixture-key");

  const missingModel = structuredClone(catalogRuntimeLockModels(document)[0]);
  missingModel.runtimeIdentity.sourceModel = "fixture/missing-chat-model";
  await assert.rejects(
    () => validateCatalogBundle(catalogPath, signaturePath, distribution, [
      ...catalogRuntimeLockModels(document), missingModel,
    ]),
    /catalog is missing locked models: fixture\/missing-chat-model/,
  );
  missingModel.modelType = "base";
  missingModel.runtimeIdentity.sourceModel = "fixture/missing-base-model";
  await assert.rejects(
    () => validateCatalogBundle(catalogPath, signaturePath, distribution, [
      ...catalogRuntimeLockModels(document), missingModel,
    ]),
    /catalog is missing locked models: fixture\/missing-base-model/,
  );

  await writeFile(catalogPath, Buffer.from(`${JSON.stringify({ ...document, sequence: 8 })}\n`));
  await assert.rejects(
    () => validateCatalogBundle(
      catalogPath,
      signaturePath,
      distribution,
      catalogRuntimeLockModels(document),
    ),
    /signature is invalid/,
  );
  const nextBytes = Buffer.from(`${JSON.stringify({ ...document, sequence: 8 })}\n`);
  await Promise.all([
    writeFile(catalogPath, nextBytes),
    writeFile(signaturePath, `${JSON.stringify({
      ...envelope,
      signature: sign(null, nextBytes, privateKey).toString("base64"),
    })}\n`),
  ]);
  const advanced = await validateCatalogBundle(
    catalogPath,
    signaturePath,
    distribution,
    catalogRuntimeLockModels(document),
  );
  assert.equal(advanced.sequence, 8);

} finally {
  await rm(root, { recursive: true, force: true });
}

console.log("hosted catalog publisher checks passed");
