import { expect, test, type Locator, type Page } from "@playwright/test";
import { composer as composerOf, devUrl } from "./workbench-navigation";
const desktopViewports = [
  { width: 1024, height: 768 },
  { width: 1280, height: 800 },
  { width: 1440, height: 900 },
] as const;

test.beforeEach(async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
});

function collectPageErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  return errors;
}

async function expectNoHorizontalOverflow(page: Page): Promise<void> {
  await expect.poll(async () => page.evaluate(() => {
    const root = document.documentElement;
    return Math.max(root.scrollWidth, document.body.scrollWidth) - root.clientWidth <= 1;
  })).toBe(true);
}

async function expectInsideViewport(locator: Locator): Promise<void> {
  await expect.poll(async () => locator.evaluate((element) => {
    const box = element.getBoundingClientRect();
    return (
      box.left >= -1 &&
      box.right <= document.documentElement.clientWidth + 1 &&
      box.top >= -1 &&
      box.bottom <= window.innerHeight + 1
    );
  })).toBe(true);
}

async function expectStorageAndCoordinationCanaries(
  page: Page,
  exerciseOpfs: boolean,
): Promise<void> {
  expect(await page.evaluate(async (shouldExerciseOpfs) => {
    const suffix = crypto.randomUUID();
    const lockName = `drowse-browser-test-${suffix}`;
    const lockWorked = await navigator.locks.request(lockName, (lock) => lock?.name === lockName);

    const workerUrl = URL.createObjectURL(new Blob(["postMessage('ready')"], {
      type: "text/javascript",
    }));
    let workerWorked = false;
    try {
      workerWorked = await new Promise<boolean>((resolve, reject) => {
        const worker = new Worker(workerUrl, { type: "module" });
        const timeout = window.setTimeout(() => {
          worker.terminate();
          reject(new Error("Dedicated worker canary timed out"));
        }, 2_000);
        worker.onmessage = (event) => {
          window.clearTimeout(timeout);
          worker.terminate();
          resolve(event.data === "ready");
        };
        worker.onerror = (event) => {
          window.clearTimeout(timeout);
          worker.terminate();
          reject(new Error(event.message));
        };
      });
    } finally {
      URL.revokeObjectURL(workerUrl);
    }

    let opfs: boolean | null = null;
    if (shouldExerciseOpfs) {
      const directoryName = `.drowse-browser-test-${suffix}`;
      const root = await navigator.storage.getDirectory();
      let created = false;
      try {
        const directory = await root.getDirectoryHandle(directoryName, { create: true });
        created = true;
        const handle = await directory.getFileHandle("canary.txt", { create: true });
        const initial = await handle.createWritable();
        await initial.write("dro");
        await initial.close();
        const resumed = await handle.createWritable({ keepExistingData: true });
        await resumed.seek(3);
        await resumed.write("wse-extra");
        await resumed.truncate(6);
        await resumed.close();
        const contents = await (await handle.getFile()).text();
        const entries = [];
        for await (const [name] of directory.entries()) entries.push(name);
        opfs = contents === "drowse" && entries.includes("canary.txt");
      } finally {
        if (created) await root.removeEntry(directoryName, { recursive: true });
      }
    }

    return { opfs, webLock: lockWorked, worker: workerWorked };
  }, exerciseOpfs)).toEqual({
    opfs: exerciseOpfs ? true : null,
    webLock: true,
    worker: true,
  });
}

test("desktop landing documents supported browsers and supplies runtime prerequisites", async ({
  page,
  browserName,
}, testInfo) => {
  const pageErrors = collectPageErrors(page);
  const response = await page.goto("/");

  expect(response).not.toBeNull();
  expect(response!.headers()["cross-origin-opener-policy"]).toBe("same-origin");
  expect(response!.headers()["cross-origin-embedder-policy"]).toBe("require-corp");
  if (testInfo.project.use.userAgent) {
    expect(await page.evaluate(() => navigator.userAgent)).toBe(testInfo.project.use.userAgent);
  }

  const { opfs, ...prerequisites } = await page.evaluate(() => ({
    secureContext: globalThis.isSecureContext,
    crossOriginIsolated: globalThis.crossOriginIsolated,
    worker: typeof Worker === "function",
    indexedDb: "indexedDB" in globalThis,
    broadcastChannel: typeof BroadcastChannel === "function",
    webAssembly: typeof WebAssembly === "object",
    webLocks: typeof navigator.locks?.request === "function",
    opfs: typeof navigator.storage?.getDirectory === "function",
  }));
  expect(prerequisites).toEqual({
    secureContext: true,
    crossOriginIsolated: true,
    worker: true,
    indexedDb: true,
    broadcastChannel: true,
    webAssembly: true,
    webLocks: true,
  });
  if (browserName !== "webkit") expect(opfs).toBe(true);
  testInfo.annotations.push({ type: "native-opfs-api", description: String(opfs) });
  // Playwright WebKit either omits OPFS on Linux or rejects getDirectory() in
  // ephemeral contexts; the app's live canary still enforces real Safari storage.
  await expectStorageAndCoordinationCanaries(page, browserName !== "webkit");

  for (const viewport of desktopViewports) {
    await page.setViewportSize(viewport);
    await expect(page.getByRole("main").getByRole("link", { name: "Open Drowse", exact: true }).first()).toBeVisible();
    await expectNoHorizontalOverflow(page);
  }

  const browserSupport = page.locator("main [data-webgpu]");
  await expect(browserSupport).toHaveAttribute("data-webgpu", /^(available|unavailable)$/);
  await expect(browserSupport).toHaveText(await browserSupport.getAttribute("data-webgpu") === "available"
    ? "WebGPU is available in this browser. Drowse checks your device's memory next."
    : "This browser can't run Drowse yet. Try a current version of Chrome, Edge or Safari 26 or later on a computer with a graphics processor.");
  expect(pageErrors).toEqual([]);
});

test("fixture workbench stays usable across desktop widths and keyboard navigation", async ({
  page,
}) => {
  const pageErrors = collectPageErrors(page);
  await page.goto(`${devUrl}/app?layoutFixture=1`);
  const composer = composerOf(page);
  const send = page.getByRole("button", { name: "Send", exact: true });
  const thread = page.getByRole("complementary", { name: "Thread", exact: true });
  const inspector = page.getByRole("complementary", { name: "Inspector", exact: true });
  const rooms = page.getByRole("navigation", { name: "Rooms", exact: true });
  await expect(composer).toBeVisible();

  for (const viewport of desktopViewports) {
    await page.setViewportSize(viewport);
    await composer.focus();
    await expect(send).toBeVisible();
    await expect(inspector).toBeVisible();
    await expectInsideViewport(rooms);
    await expectInsideViewport(composer);
    await expectInsideViewport(send);
    if (await thread.isVisible()) await expectInsideViewport(thread);
    await expectNoHorizontalOverflow(page);
  }

  await composer.fill("Cross-browser keyboard check");
  const palette = page.getByRole("dialog", { name: "Search everything", exact: true });
  await page.getByRole("button", { name: "Search everything (⌘K)", exact: true }).click();
  await expect(palette).toBeVisible();
  await expect(palette.getByRole("combobox", { name: "Search everything" })).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(palette).toHaveCount(0);
  await expect(composer).toHaveValue("Cross-browser keyboard check");

  const lens = rooms.getByRole("button", { name: "Lens", exact: true });
  await lens.focus();
  await lens.press("Enter");
  await expect(page.locator("main h1")).toHaveText("Lens");
  await expect(lens).toHaveAttribute("aria-pressed", "true");
  expect(pageErrors).toEqual([]);
});
