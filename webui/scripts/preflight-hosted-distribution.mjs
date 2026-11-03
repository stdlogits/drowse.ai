// The signed catalog at the distribution lock's URL, verified as the browser verifies it
// (signature, sequence floor, the lock's models), so a build never ships reading a catalog path
// that is not published. Every deploy runs it; it downloads no artifacts (the browser checks each
// download's digest).

import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const browserRuntime = fileURLToPath(new URL("../../browser-runtime/", import.meta.url));
const PREFLIGHT_ORIGIN = "https://drowse.invalid";

export async function main() {
  const distribution = JSON.parse(await readFile(resolve(browserRuntime, "distribution-lock.json"), "utf8"));
  if (distribution.status !== "verified") {
    throw new Error("Catalog preflight requires a verified distribution lock");
  }
  const catalogOrigins = new Set(distribution.allowedCatalogRedirectOrigins);
  const server = await createServer({
    root,
    configFile: false,
    appType: "custom",
    logLevel: "silent",
    server: { middlewareMode: true, watch: null },
  });
  try {
    const [{ verifyCatalog, WebCryptoEd25519Verifier }, { HOSTED_DISTRIBUTION_CONFIG }] = await Promise.all([
      server.ssrLoadModule("/src/lib/runtime/catalog.ts"),
      server.ssrLoadModule("/src/hosted/runtime/distributionConfig.ts"),
    ]);
    const [catalogResponse, signatureResponse] = await Promise.all([
      fetchChecked(distribution.catalogUrl, catalogOrigins, 60_000),
      fetchChecked(distribution.signatureUrl, catalogOrigins, 60_000),
    ]);
    const catalogBytes = new Uint8Array(await catalogResponse.arrayBuffer());
    const signature = JSON.parse(await signatureResponse.text());
    const catalog = await verifyCatalog(
      catalogBytes,
      signature,
      new WebCryptoEd25519Verifier(HOSTED_DISTRIBUTION_CONFIG.publicKeys),
      {
        now: Date.now(),
        online: true,
        minimumSequence: HOSTED_DISTRIBUTION_CONFIG.minimumAcceptedSequence,
        lastAcceptedSequence: null,
        expectedRuntimeAbi: HOSTED_DISTRIBUTION_CONFIG.expectedRuntimeAbi,
        runtimeLockModels: HOSTED_DISTRIBUTION_CONFIG.runtimeLockModels,
      },
    );
    assertCatalogSequenceFloor(catalog.document.sequence, HOSTED_DISTRIBUTION_CONFIG.minimumAcceptedSequence);
    console.log(`Hosted catalog preflight passed: sequence ${catalog.document.sequence} at ${distribution.catalogUrl}`);
  } finally {
    await server.close();
  }
}

export function assertCatalogSequenceFloor(sequence, minimumAcceptedSequence) {
  if (sequence < minimumAcceptedSequence) {
    throw new Error(`Catalog sequence ${sequence} is below the embedded rollback floor ${minimumAcceptedSequence}`);
  }
}

/** Fetches as a browser would, following at most five redirects and only to approved HTTPS origins. */
export async function fetchChecked(url, allowedOrigins, timeoutMs, fetcher = fetch) {
  const approvedOrigins = new Set([new URL(url).origin, ...allowedOrigins]);
  let currentUrl = url;
  for (let redirects = 0; redirects <= 5; redirects += 1) {
    const current = validatedApprovedUrl(currentUrl, approvedOrigins, url);
    const response = await fetcher(current, {
      cache: "no-store",
      credentials: "omit",
      redirect: "manual",
      referrerPolicy: "no-referrer",
      headers: { Origin: PREFLIGHT_ORIGIN },
      signal: AbortSignal.timeout(timeoutMs),
    });
    const allowedOrigin = response.headers.get("Access-Control-Allow-Origin");
    if (allowedOrigin !== "*" && allowedOrigin !== PREFLIGHT_ORIGIN) {
      await response.body?.cancel().catch(() => undefined);
      throw new Error(`${current} did not allow the requested browser origin`);
    }
    if ([301, 302, 303, 307, 308].includes(response.status)) {
      const location = response.headers.get("Location");
      await response.body?.cancel().catch(() => undefined);
      if (!location) throw new Error(`${current} returned a redirect without Location`);
      if (redirects === 5) throw new Error(`${url} exceeded five approved redirects`);
      currentUrl = new URL(location, current).href;
      continue;
    }
    if (!response.ok) {
      await response.body?.cancel().catch(() => undefined);
      throw new Error(`${current} returned HTTP ${response.status}`);
    }
    return response;
  }
  throw new Error(`${url} exceeded five approved redirects`);
}

export function validatedApprovedUrl(value, approvedOrigins, signedUrl) {
  let url;
  try {
    url = new URL(value);
  } catch {
    throw new Error(`${signedUrl} resolved to an invalid URL`);
  }
  if (url.protocol !== "https:" || url.username || url.password || !approvedOrigins.has(url.origin)) {
    throw new Error(`${signedUrl} redirected to unapproved URL ${url.href}`);
  }
  return url.href;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
