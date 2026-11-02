import { expect, type Page, type Worker } from "@playwright/test";
import { test } from "./origin-outage-fixture";
import { readdir, readFile } from "node:fs/promises";
import { resolve } from "node:path";
import ts from "typescript";

test("first-visit model setup caches the engine for a fresh worker during an origin outage", async ({ page, context, outageOrigin, browserName }) => {
  test.setTimeout(180_000);
  const assets = await readdir(resolve("dist-hosted/assets"));
  const workerName = assets.find(name => /^browser\.worker-.*\.js$/u.test(name))!;
  const engineName = assets.find(name => /^drowse-web-llm-.*\.js$/u.test(name))!;
  expect(workerName).toBeTruthy();
  expect(engineName).toBeTruthy();
  const workerUrl = new URL(`/assets/${workerName}`, outageOrigin.url).href;
  const engineUrl = new URL(`/assets/${engineName}`, outageOrigin.url).href;
  await page.goto(`${outageOrigin.url}/app`);
  const firstWorker = await startWorker(page);
  await expect.poll(() => page.evaluate(() => Boolean(navigator.serviceWorker.controller)), {
    timeout: 120_000,
  }).toBe(true);
  expect(await page.evaluate(async url => Boolean(await caches.match(url)), engineUrl)).toBe(false);
  const { outputText } = ts.transpileModule(
    await readFile(resolve("src/hosted/runtime/offlineRuntimeAssets.ts"), "utf8"),
    { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } },
  );
  await page.evaluate(async source => {
    const exports: { cacheOfflineRuntimeAssets?: () => Promise<void> } = {};
    new Function("exports", source)(exports);
    await exports.cacheOfflineRuntimeAssets!();
  }, outputText);

  async function startWorker(target: Page) {
    const started = target.waitForEvent("worker", { timeout: 15_000 });
    await target.evaluate(url => {
      Object.assign(window, { offlineRuntimeTestWorker: new Worker(url, { type: "module" }) });
    }, workerUrl);
    return started;
  }

  async function importEngine(worker: Worker) {
    expect(await worker.evaluate(async url => {
      const module = await import(url);
      return Object.values(module).some(value =>
        typeof (value as { MLCEngine?: unknown })?.MLCEngine === "function"
      );
    }, engineUrl)).toBe(true);
  }

  await importEngine(firstWorker);
  await expect.poll(() => page.evaluate(async url => Boolean(await caches.match(url, { ignoreVary: true })), engineUrl)).toBe(true);
  await outageOrigin.disconnect();
  const offlinePage = await context.newPage();
  await offlinePage.goto(`${outageOrigin.url}/app?outage=1`, { waitUntil: "domcontentloaded" });
  // The app starts its worker on model load, after the page has loaded. In WebKit a module worker
  // made in the instant after DOMContentLoaded sometimes misses the service worker (its fetches go
  // to the dead origin although the page reports a controller; 3 of 6 runs), so start it from a
  // loaded page, as the app does.
  await offlinePage.waitForLoadState("load");
  expect(await offlinePage.evaluate(() => Boolean(navigator.serviceWorker.controller))).toBe(true);
  await importEngine(await startWorker(offlinePage));
});
