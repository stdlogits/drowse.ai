import { createHash } from "node:crypto";
import { execFile } from "node:child_process";
import { lstat, readFile, realpath } from "node:fs/promises";
import { dirname, relative, resolve, sep } from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);
const locallyExecutedReceiptCandidates = new WeakSet();

const PRODUCER_EXECUTABLE = "webui/scripts/hosted-release-evidence-producer.mjs";
// A producer config is a small JSON document; closure files include the vendored WebLLM archive
// (about 4.6 MB for 0.2.84-drowse.46), so they get a larger bound.
const MAX_CONFIG_BYTES = 4 * 1024 * 1024;
const MAX_CLOSURE_FILE_BYTES = 64 * 1024 * 1024;
const COMMON_CLOSURE = Object.freeze([
  PRODUCER_EXECUTABLE,
  "webui/scripts/release-evidence-producers.mjs",
  "webui/scripts/record-hosted-evidence.mjs",
  "webui/scripts/check-runtime-lock.mjs",
  "browser-runtime/evidence-config.schema.json",
  "browser-runtime/release-evidence-record.schema.json",
]);

// Hook ABI v5's runtime evidence (browser-runtime/README.md, "Release evidence"): drowse's units,
// replay and read goldens over the four fixture libraries on the C TVM VM and on WebGPU, then each
// production library through the physical runtime harness, and each q4 library's parity gate.
const UNITS_GOLDENS = Object.freeze([
  "drowse/tests/fixtures/browser/units-tiny-models-v1.json",
  ...["qwen3", "gemma3"].flatMap((arch) => [
    "block-sites", "unit-ablation", "unit-probes", "attention-patterns", "token-anatomy",
    "unit-activity", "unit-reads", "lens-reads", "slens", "unit-artifacts", "neuron-push",
  ].map((name) => `drowse/tests/fixtures/browser/${name}-${arch}-v1.json`)),
  "browser-runtime/fixtures/unit-ablation-mixed-affine-v1.json",
]);

const cVmGoldensClosure = Object.freeze([
  ...COMMON_CLOSURE,
  "browser-runtime/record-cvm-goldens.py",
  "browser-runtime/forks/verify-mlc-hook.py",
  "browser-runtime/forks/units_goldens.py",
  "browser-runtime/forks/static_reads.py",
  "browser-runtime/forks/overlay_checks.py",
  "browser-runtime/forks/compile-tiny-webgpu.py",
  "browser-runtime/forks/build-units-fixtures.py",
  "browser-runtime/forks/manifest.json",
  "browser-runtime/forks/mlc-llm-drowse.patch",
  "browser-runtime/forks/tvm-webgpu-readonly.patch",
  ...UNITS_GOLDENS,
]);

const webGpuGoldensClosure = Object.freeze([
  ...COMMON_CLOSURE,
  "webui/scripts/browser-units-goldens.mjs",
  "webui/scripts/browser-units-reads.mjs",
  "webui/scripts/release-tool-storage-quota.mjs",
  "webui/src/hosted/runtime/structuredHookProfile.ts",
  "webui/src/hosted/runtime/structuredHookProgram.ts",
  "webui/src/hosted/runtime/webLlmEngine.ts",
  "webui/src/hosted/runtime/webLlmBlockSites.ts",
  "webui/src/hosted/runtime/webLlmStaticReads.ts",
  "browser-runtime/forks/manifest.json",
  "browser-runtime/forks/web-llm-drowse.patch",
  "browser-runtime/vendor/drowse-web-llm-0.2.84-drowse.46.tgz",
  ...UNITS_GOLDENS,
]);

const productionLibraryClosure = Object.freeze([
  ...COMMON_CLOSURE,
  "webui/scripts/browser-full-runtime.mjs",
  "webui/scripts/browser-full-runtime-contract.mjs",
  "webui/scripts/browser-full-runtime-artifact-audit.mjs",
  "webui/scripts/hosted-full-runtime.worker.ts",
  "webui/scripts/hosted-model-profiles.mjs",
  "webui/scripts/overlay-vm-functions.mjs",
  "webui/scripts/runtime-lock-block-spec.mjs",
  "webui/scripts/runtime-lock-document.mjs",
  "webui/scripts/publish-hosted-model.mjs",
  "webui/scripts/hosted-model-closure.mjs",
  "webui/src/hosted/runtime/structuredHookProfile.ts",
  "webui/src/hosted/runtime/webLlmEngine.ts",
  "browser-runtime/forks/overlay_checks.py",
  "browser-runtime/forks/manifest.json",
  "browser-runtime/runtime-lock-identity-v1.json",
]);

const q4ParityClosure = Object.freeze([
  ...COMMON_CLOSURE,
  "webui/scripts/hosted-model-profiles.mjs",
  "webui/scripts/runtime-lock-document.mjs",
  "webui/scripts/publish-hosted-model.mjs",
  "webui/scripts/hosted-model-closure.mjs",
  "webui/scripts/capture-hosted-residuals.mjs",
  "webui/scripts/hosted-residual-capture.worker.ts",
  "browser-runtime/fixtures/production-q4-parity-input-v1.json",
  "browser-runtime/mlc_q4_torch.py",
  "browser-runtime/q4_unit_parity.py",
  "browser-runtime/forks/overlay_checks.py",
  "browser-runtime/validate_mlc_q4_parity.py",
]);

const authoringClosure = Object.freeze([
  ...COMMON_CLOSURE,
  "webui/scripts/browser-full-runtime.mjs",
  "webui/scripts/browser-full-runtime-contract.mjs",
  "webui/scripts/browser-full-runtime-artifact-audit.mjs",
  "webui/scripts/hosted-model-profiles.mjs",
  "webui/scripts/hosted-full-runtime.worker.ts",
  "webui/scripts/browser-manifold-fitting.test.mjs",
  "webui/scripts/fitting-wasm-browser.test.mjs",
  "webui/scripts/hosted-release-local-observer.mjs",
  "webui/scripts/drowse-archive-browser.test.mjs",
  "webui/scripts/browser-core-pack.test.mjs",
  "webui/scripts/steering-expression.test.mjs",
  "webui/scripts/browser-loom.test.mjs",
  "webui/scripts/worker-runtime.test.mjs",
  "webui/scripts/structured-hook-parity.test.mjs",
  "webui/scripts/sampling-parity.test.mjs",
  "webui/scripts/web-llm-generation.test.mjs",
  "webui/scripts/build-fitting-wasm.mjs",
  "webui/src/hosted/runtime/browserCorePack.ts",
  "webui/src/hosted/runtime/browserInstrumentPacks.ts",
  "webui/src/hosted/runtime/browserLoom.ts",
  "drowse/webui/src/lib/roleSlug.ts",
  "webui/src/hosted/runtime/steeringExpression.ts",
  "webui/src/hosted/runtime/structuredHookProgram.ts",
  "webui/src/hosted/runtime/webLlmGeneration.ts",
  "webui/src/hosted/artifacts/drowseArchive.ts",
  "webui/src/hosted/artifacts/safetensors.ts",
  "webui/src/hosted/artifacts/zip.ts",
  "webui/src/hosted/fitting/coordinator.ts",
  "webui/public-hosted/wasm/fitting-kernel.json",
  "webui/public-hosted/wasm/drowse_fitting_wasm.js",
  "webui/public-hosted/wasm/drowse_fitting_wasm_bg.wasm",
  "browser-runtime/fitting-wasm/Cargo.lock",
  "browser-runtime/fitting-wasm/Cargo.toml",
  "browser-runtime/fitting-wasm/src/error.rs",
  "browser-runtime/fitting-wasm/src/kernel.rs",
  "browser-runtime/fitting-wasm/src/lib.rs",
  "browser-runtime/fitting-wasm/src/linalg.rs",
  "browser-runtime/fitting-wasm/tests/golden.rs",
  "drowse/tests/fixtures/browser/affine-fisher-fit-v1.json",
  "drowse/tests/fixtures/browser/post-block-rank-one-v2.json",
  "drowse/tests/fixtures/browser/structured-program-parity-v1.json",
  "drowse/tests/fixtures/browser/measurement-envelope-v2.json",
  "drowse/tests/fixtures/browser/loom-operation-transcript-v1.json",
  "drowse/tests/fixtures/browser/recipe-compose-v1.json",
  "drowse/tests/fixtures/browser/topology-rbf-parity-v1.json",
  "drowse/tests/fixtures/browser/sampling-parity-v1.json",
  "browser-runtime/fixtures/drowse-interchange-v1.json",
  "browser-runtime/fixtures/shared-parity-suite-v1.json",
  "drowse/tests/browser_drowse_archive_bridge.py",
  "drowse/tests/browser/test_browser_parity.py",
  "drowse/tests/store/test_drowse_archive.py",
  "drowse/drowse/generation/decode.py",
  "drowse/drowse/generation/sampling.py",
  "drowse/drowse/geometry/domains.py",
  "drowse/drowse/geometry/inject.py",
  "drowse/drowse/geometry/manifold.py",
  "drowse/drowse/geometry/rbf.py",
  "drowse/drowse/geometry/subspace.py",
  "drowse/drowse/geometry/topology.py",
  "drowse/drowse/lens/jlens.py",
  "drowse/drowse/loom/recipe.py",
  "drowse/drowse/loom/tree.py",
  "drowse/drowse/readout/measurements.py",
  "drowse/drowse/sae/backend.py",
  "drowse/drowse/steering/grammar.py",
  "drowse/drowse/store/archive.py",
]);

const benchmarkClosure = Object.freeze([
  ...COMMON_CLOSURE,
  "webui/scripts/browser-full-runtime.mjs",
  "webui/scripts/browser-full-runtime-contract.mjs",
  "webui/scripts/browser-full-runtime-artifact-audit.mjs",
  "webui/scripts/hosted-model-profiles.mjs",
  "webui/scripts/hosted-full-runtime.worker.ts",
]);

function descriptor(id, kind, closure, fixturePath) {
  return Object.freeze({
    id,
    kind,
    executable: PRODUCER_EXECUTABLE,
    closure,
    fixturePath,
  });
}

const RELEASE_EVIDENCE_PRODUCERS = new Map([
  [
    "cVmGoldens",
    descriptor(
      "browser-cvm-units-goldens-v1",
      "gate",
      cVmGoldensClosure,
      "drowse/tests/fixtures/browser/units-tiny-models-v1.json",
    ),
  ],
  [
    "webGpuGoldens",
    descriptor(
      "browser-webgpu-units-goldens-v1",
      "gate",
      webGpuGoldensClosure,
      "drowse/tests/fixtures/browser/units-tiny-models-v1.json",
    ),
  ],
  [
    "productionLibrary",
    descriptor(
      "browser-production-library-v1",
      "gate",
      productionLibraryClosure,
      "browser-runtime/runtime-lock-identity-v1.json",
    ),
  ],
  [
    "productionQ4Parity",
    descriptor(
      "browser-production-q4-parity-v2",
      "gate",
      q4ParityClosure,
      "browser-runtime/fixtures/production-q4-parity-input-v1.json",
    ),
  ],
  [
    "activationCapture",
    descriptor(
      "browser-activation-capture-v1",
      "gate",
      authoringClosure,
      "browser-runtime/runtime-lock-identity-v1.json",
    ),
  ],
  [
    "fittingKernelParity",
    descriptor(
      "browser-fitting-kernel-parity-v1",
      "gate",
      authoringClosure,
      "drowse/tests/fixtures/browser/affine-fisher-fit-v1.json",
    ),
  ],
  [
    "topologyOrchestration",
    descriptor(
      "browser-topology-orchestration-v1",
      "gate",
      authoringClosure,
      "drowse/tests/fixtures/browser/topology-rbf-parity-v1.json",
    ),
  ],
  [
    "manifoldSerialization",
    descriptor(
      "browser-manifold-serialization-v1",
      "gate",
      authoringClosure,
      "browser-runtime/fixtures/drowse-interchange-v1.json",
    ),
  ],
  [
    "instrumentIntegration",
    descriptor(
      "browser-instrument-integration-v1",
      "gate",
      authoringClosure,
      "browser-runtime/runtime-lock-identity-v1.json",
    ),
  ],
  [
    "sharedGoldenResults",
    descriptor(
      "browser-shared-goldens-v1",
      "gate",
      authoringClosure,
      "browser-runtime/fixtures/shared-parity-suite-v1.json",
    ),
  ],
  [
    "physicalBrowserFitting",
    descriptor(
      "physical-browser-fitting-v1",
      "gate",
      authoringClosure,
      "browser-runtime/runtime-lock-identity-v1.json",
    ),
  ],
  [
    "physicalBrowserBenchmark",
    descriptor(
      "physical-browser-benchmark-v1",
      "benchmark",
      benchmarkClosure,
      null,
    ),
  ],
]);

export function requireReleaseEvidenceProducer(evidenceType) {
  const producer = RELEASE_EVIDENCE_PRODUCERS.get(evidenceType);
  if (!producer) {
    throw new Error(
      `${evidenceType} has no registered executable release evidence producer`,
    );
  }
  return producer;
}

export function releaseEvidenceProducerTypes() {
  return [...RELEASE_EVIDENCE_PRODUCERS.keys()];
}

export async function releaseEvidenceProducerToolVersion(
  evidenceType,
  repositoryRoot,
  configPath,
) {
  return (await releaseEvidenceProducerBinding(
    evidenceType,
    repositoryRoot,
    configPath,
  )).toolVersion;
}

export async function releaseEvidenceProducerBinding(
  evidenceType,
  repositoryRoot,
  configPath,
) {
  const producer = requireReleaseEvidenceProducer(evidenceType);
  const relativeConfig = repositoryRelativePath(
    repositoryRoot,
    configPath,
    "producer config",
  );
  const configBytes = await readRegularContainedFile(
    resolve(repositoryRoot, ...relativeConfig.split("/")),
    repositoryRoot,
    "producer config",
    MAX_CONFIG_BYTES,
  );
  const digest = createHash("sha256");
  for (const path of producer.closure) {
    const bytes = await readRegularContainedFile(
      resolve(repositoryRoot, ...path.split("/")),
      repositoryRoot,
      "release evidence producer source",
      MAX_CLOSURE_FILE_BYTES,
    );
    digest.update(`${path}\0${bytes.byteLength}\0`);
    digest.update(bytes);
    digest.update("\0");
  }
  digest.update(`${relativeConfig}\0${configBytes.byteLength}\0`);
  digest.update(configBytes);
  digest.update("\0");
  return Object.freeze({
    producerId: producer.id,
    configPath: relativeConfig,
    configSha256: createHash("sha256").update(configBytes).digest("hex"),
    toolVersion: `${producer.id}+sha256.${digest.digest("hex")}`,
  });
}

export async function runReleaseEvidenceProducer({
  evidenceType,
  configPath,
  repositoryRoot,
  nonce,
  execute = execFileAsync,
}) {
  const producer = requireReleaseEvidenceProducer(evidenceType);
  if (typeof nonce !== "string" || !/^[0-9a-f]{64}$/.test(nonce)) {
    throw new Error("release evidence producer nonce is invalid");
  }
  const relativeConfig = repositoryRelativePath(
    repositoryRoot,
    configPath,
    "producer config",
  );
  const binding = await releaseEvidenceProducerBinding(
    evidenceType,
    repositoryRoot,
    configPath,
  );
  const executable = resolve(
    repositoryRoot,
    ...producer.executable.split("/"),
  );
  const { stdout, stderr } = await execute(
    process.execPath,
    [
      executable,
      "--evidence-type",
      evidenceType,
      "--config",
      relativeConfig,
      "--nonce",
      nonce,
    ],
    {
      cwd: repositoryRoot,
      encoding: "utf8",
      maxBuffer: 4 * 1024 * 1024,
      timeout: 2 * 60 * 60 * 1000,
      windowsHide: true,
    },
  );
  if (typeof stderr === "string" && stderr.trim()) {
    throw new Error("release evidence producer wrote to stderr");
  }
  let receipt;
  try {
    receipt = JSON.parse(stdout);
  } catch (error) {
    throw new Error("release evidence producer did not return one JSON receipt", {
      cause: error,
    });
  }
  validateProducerReceipt(receipt, producer, evidenceType, nonce);
  const boundReceipt = {
    ...receipt,
    producerConfigPath: binding.configPath,
    producerConfigSha256: binding.configSha256,
    producerToolVersion: binding.toolVersion,
  };
  deepFreeze(boundReceipt);
  locallyExecutedReceiptCandidates.add(boundReceipt);
  return boundReceipt;
}

export function assertExecutedProducerReceipt(receipt) {
  if (
    !receipt ||
    typeof receipt !== "object" ||
    !locallyExecutedReceiptCandidates.has(receipt)
  ) {
    throw new Error(
      "release evidence must come from an executed registered producer",
    );
  }
}

export function validateProducerReceipt(receipt, producer, evidenceType, nonce) {
  if (!receipt || typeof receipt !== "object" || Array.isArray(receipt)) {
    throw new Error("release evidence producer receipt must be an object");
  }
  const expected = producer.kind === "benchmark"
    ? [
        "schemaVersion",
        "evidenceType",
        "producerId",
        "nonce",
        "producedAt",
        "passed",
        "run",
      ]
    : [
        "schemaVersion",
        "evidenceType",
        "producerId",
        "nonce",
        "producedAt",
        "passed",
        "fixturePath",
        "results",
      ];
  exactKeys(receipt, expected, "release evidence producer receipt");
  if (
    receipt.schemaVersion !== 1 ||
    receipt.evidenceType !== evidenceType ||
    receipt.producerId !== producer.id ||
    receipt.nonce !== nonce ||
    receipt.passed !== true
  ) {
    throw new Error("release evidence producer receipt identity is invalid");
  }
  const producedAt = Date.parse(receipt.producedAt);
  if (!Number.isFinite(producedAt)) {
    throw new Error("release evidence producer receipt timestamp is invalid");
  }
  if (producer.kind === "gate" && receipt.fixturePath !== producer.fixturePath) {
    throw new Error("release evidence producer receipt fixture is not pinned");
  }
}

function repositoryRelativePath(repositoryRoot, path, label) {
  const value = relative(repositoryRoot, resolve(repositoryRoot, path));
  if (
    !value ||
    value === ".." ||
    value.startsWith(`..${sep}`) ||
    value.includes("\\") ||
    value.split(sep).some((part) => !part || part === "." || part === "..")
  ) {
    throw new Error(`${label} must be inside the Drowse repository`);
  }
  return value.split(sep).join("/");
}

async function readRegularContainedFile(path, root, label, maxBytes) {
  const [rootRealPath, parentRealPath, info] = await Promise.all([
    realpath(root),
    realpath(dirname(path)),
    lstat(path),
  ]);
  const parentRelative = relative(rootRealPath, parentRealPath);
  if (
    parentRelative === ".." ||
    parentRelative.startsWith(`..${sep}`) ||
    info.isSymbolicLink() ||
    !info.isFile() ||
    info.size > maxBytes
  ) {
    throw new Error(
      `${label} must be a regular file inside the Drowse repository, at most ${maxBytes} bytes`,
    );
  }
  return readFile(path);
}

function exactKeys(value, keys, label) {
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  if (
    actual.length !== expected.length ||
    actual.some((key, index) => key !== expected[index])
  ) {
    throw new Error(`${label} has unknown or missing fields`);
  }
}

function deepFreeze(value) {
  if (!value || typeof value !== "object" || Object.isFrozen(value)) return value;
  for (const child of Object.values(value)) deepFreeze(child);
  return Object.freeze(value);
}
