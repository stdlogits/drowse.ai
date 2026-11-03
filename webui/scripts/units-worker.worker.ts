// The hosted runtime worker as the app runs it (HostedRuntimeWorker over the WebLLM model backend,
// the verified downloader, the content store and the archive service), on a catalog and runtime
// lock browser-units-worker.mjs stages for B4's tiny v5 fixture library and its packs. Only the
// catalog's signature check and the compatibility probe are stood in for: the harness serves its
// own catalog (fixture.invalid URLs, fetched from the harness server) and the device facts.
import type {
  CatalogDocument,
  RuntimeCapabilities,
  VerifiedCatalog,
} from "../../drowse/webui/src/lib/runtime/contracts";
import type { CatalogRuntimeLockModel } from "../src/lib/runtime/catalog";
import { BrowserDrowseArchiveService } from "../src/hosted/artifacts/service";
import { createLazyBrowserModelBackend } from "../src/hosted/runtime/browserModelBackend";
import { BrowserContentStore } from "../src/hosted/runtime/contentStore";
import { VerifiedArtifactDownloader } from "../src/hosted/runtime/downloader";
import type { DrowseWebLlmModule } from "../src/hosted/runtime/webLlmEngine";
import { HostedRuntimeWorker, type WorkerScopeLike } from "../src/hosted/runtime/worker";

interface HarnessConfig {
  document: CatalogDocument;
  capabilities: RuntimeCapabilities;
  runtimeLockModels: CatalogRuntimeLockModel[];
  /** The units reads' S→R band, when the harness reads at a golden's (drowse sets
   *  `session.lens.band` the same way); the default otherwise. */
  lensBand?: { lo: number; hi: number };
}

const FIXTURE_ORIGIN = "https://fixture.invalid/";
const WEB_LLM_URL = "/webllm.js";
const config = await (await fetch("/harness/config.json", { cache: "no-store" })).json() as HarnessConfig;
const catalog: VerifiedCatalog = {
  document: config.document,
  exactBytes: new TextEncoder().encode(JSON.stringify(config.document)),
  signature: { schemaVersion: 1, algorithm: "Ed25519", keyId: "units-worker-harness", signature: "harness" },
  allowDownloads: true,
  stale: false,
};

async function harnessFetch(url: string, init: RequestInit): Promise<Response> {
  if (!url.startsWith(FIXTURE_ORIGIN)) throw new TypeError(`Unknown harness artifact ${url}`);
  const response = await fetch(`/fixture/${url.slice(FIXTURE_ORIGIN.length)}`, init);
  Object.defineProperty(response, "url", { value: url });
  return response;
}

const artifacts = new BrowserDrowseArchiveService();
const contentStore = new BrowserContentStore();
new HostedRuntimeWorker(globalThis as unknown as WorkerScopeLike, {
  capabilityChecker: {
    async check() {
      return { ...config.capabilities, checkedAt: Date.now() };
    },
    clearAdapter() {},
    async adapterForLoad() {
      const adapter = await (navigator as unknown as { gpu: { requestAdapter(): Promise<unknown> } }).gpu.requestAdapter();
      if (adapter === null) throw new Error("the harness browser offers no WebGPU adapter");
      return adapter as never;
    },
  },
  catalogRepository: {
    async get() {
      return catalog;
    },
    async withDownloadAuthorization(_catalog, operation) {
      return operation();
    },
    async clear() {},
  },
  contentStore,
  downloader: new VerifiedArtifactDownloader(contentStore, { fetch: harnessFetch }),
  artifactService: artifacts,
  // The vendored bridge as the harness serves it (node_modules/@drowse/web-llm/lib/index.js),
  // outside Vite's dependency optimizer.
  modelBackend: createLazyBrowserModelBackend(
    () => import(/* @vite-ignore */ WEB_LLM_URL) as Promise<DrowseWebLlmModule>,
    artifacts,
    config.lensBand === undefined ? {} : { lensBand: config.lensBand },
  ),
  runtimeLockModels: config.runtimeLockModels,
});
(globalThis as unknown as { postMessage(message: unknown): void }).postMessage({ kind: "harness-ready" });
