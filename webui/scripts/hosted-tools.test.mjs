import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { generateKeyPairSync } from "node:crypto";
import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import {
  browserRuntimeIdentityBlockers,
  validateInstalledWebLlmDependency,
  validateWebLlmDependency,
} from "./check-runtime-lock.mjs";
import { validateRuntimeLock } from "./runtime-lock-document.mjs";

const exec = promisify(execFile);
const script = (name) => new URL(`./${name}`, import.meta.url).pathname;
const dir = await mkdtemp(join(tmpdir(), "drowse-catalog-test-"));

// A signed catalog verifies, and the same signature refuses an edited catalog.
{
  const catalog = join(dir, "catalog.json");
  const signature = join(dir, "catalog.sig.json");
  const privateKeyPath = join(dir, "private.pem");
  const publicKeyPath = join(dir, "public.pem");
  const { privateKey, publicKey } = generateKeyPairSync("ed25519");
  await Promise.all([
    writeFile(catalog, '{"schemaVersion":1,"sequence":7}\n'),
    writeFile(privateKeyPath, privateKey.export({ type: "pkcs8", format: "pem" })),
    writeFile(publicKeyPath, publicKey.export({ type: "spki", format: "pem" })),
  ]);
  await exec(process.execPath, [script("sign-catalog.mjs"), catalog, privateKeyPath, "test-key", signature]);
  const verified = await exec(process.execPath, [script("verify-catalog.mjs"), catalog, signature, publicKeyPath]);
  assert.match(verified.stdout, /Verified catalog sequence 7/);
  await writeFile(catalog, '{"schemaVersion":1,"sequence":8}\n');
  await assert.rejects(
    exec(process.execPath, [script("verify-catalog.mjs"), catalog, signature, publicKeyPath]),
    /catalog signature verification failed/,
  );
  const envelope = JSON.parse(await readFile(signature, "utf8"));
  assert.equal(envelope.algorithm, "Ed25519");
  assert.equal(envelope.keyId, "test-key");
}

// The repository's locks pass the checker.
{
  const { stdout } = await exec(process.execPath, [script("check-runtime-lock.mjs")]);
  assert.match(stdout, /Browser runtime lock: /);
  assert.match(stdout, /Browser distribution lock: /);
}

// The lock's structure: profiles follow the architecture, and block specs and refusals are exact.
{
  const lock = JSON.parse(await readFile(new URL("../../browser-runtime/runtime-lock.json", import.meta.url), "utf8"));
  assert.doesNotThrow(() => validateRuntimeLock(lock));
  assert.deepEqual(browserRuntimeIdentityBlockers(lock), []);
  const withModel = (id, edit) => {
    const candidate = structuredClone(lock);
    edit(candidate.models.find((model) => model.id === id));
    return candidate;
  };
  assert.throws(() => validateRuntimeLock({ ...lock, hookAbi: "post-block-residual-v4" }), /unsupported schema or ABI/);
  assert.throws(
    () => validateRuntimeLock(withModel("qwen3-1.7b", (model) => {
      model.structuredHookProfile = "standard-v4-residual";
      delete model.blockSpec;
      model.unitRefusal = "refused";
    })),
    /qwen3 libraries attest standard-v4, not standard-v4-residual/,
  );
  assert.throws(
    () => validateRuntimeLock(withModel("pythia-70m-base", (model) => { model.unitRefusal = ""; })),
    /no unit refusal text/,
  );
  assert.throws(
    () => validateRuntimeLock(withModel("gemma3-1b-instruct", (model) => { model.blockSpec.slidingPattern = null; })),
    /block spec is invalid/,
  );
  assert.deepEqual(
    browserRuntimeIdentityBlockers(withModel("gpt2-base", (model) => { model.librarySha256 = null; })),
    ["models.gpt2-base.librarySha256 is unresolved"],
  );
}

// The installed WebLLM is the lock's archive.
{
  const manifest = { dependencies: { "@drowse/web-llm": "file:../browser-runtime/vendor/drowse-web-llm-0.2.84-drowse.3.tgz" } };
  const packageLock = {
    packages: {
      "node_modules/@drowse/web-llm": {
        resolved: "file:browser-runtime/vendor/drowse-web-llm-0.2.84-drowse.3.tgz",
        version: "0.2.84-drowse.3",
      },
    },
  };
  const pin = { path: "browser-runtime/vendor/drowse-web-llm-0.2.84-drowse.3.tgz" };
  validateWebLlmDependency(manifest, packageLock, pin);
  validateInstalledWebLlmDependency(manifest, packageLock, { name: "@drowse/web-llm", version: "0.2.84-drowse.3" });
  assert.throws(
    () => validateInstalledWebLlmDependency(manifest, packageLock, { name: "@drowse/web-llm", version: "0.2.84-drowse.2" }),
    /does not match the exact package lock/,
  );
  assert.throws(
    () => validateWebLlmDependency(manifest, packageLock, { path: "browser-runtime/vendor/drowse-web-llm-0.2.84-drowse.4.tgz" }),
    /runtime-locked Drowse WebLLM archive/,
  );
}

// An artifact manifest lists every file, sorted, with its size and digest.
{
  const artifacts = join(dir, "artifacts");
  await mkdir(join(artifacts, "nested"), { recursive: true });
  await Promise.all([
    writeFile(join(artifacts, "b.bin"), Buffer.from([2, 3])),
    writeFile(join(artifacts, "nested", "a.bin"), Buffer.from([1])),
  ]);
  const { stdout } = await exec(process.execPath, [
    script("manifest-artifacts.mjs"),
    artifacts,
    "a9lim/drowse-web-fixture",
    "0123456789abcdef0123456789abcdef01234567",
  ]);
  const manifest = JSON.parse(stdout);
  assert.deepEqual(manifest.files.map((file) => file.path), ["b.bin", "nested/a.bin"]);
  assert.deepEqual(manifest.files.map((file) => file.bytes), [2, 1]);
  assert.match(manifest.files[0].sha256, /^[0-9a-f]{64}$/);
}

console.log("Hosted catalog tools passed");
