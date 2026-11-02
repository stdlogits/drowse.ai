import { expect, test, type Page } from "@playwright/test";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

// Chats saved before Drowse 6.0: a v7 saved chat in the library and a 5.x autosave the worker
// stashed. Your chats counts them in one notice, downloads them as one JSON file and can dismiss the
// notice on this device; neither the download nor the dismissal deletes anything.
const modules = {
  home: `/@fs${resolve("src/hosted/ui/HostedHome.svelte")}`,
  stash: `/@fs${resolve("src/lib/runtime/legacyAutosaves.ts")}`,
};

const oldChat = {
  id: "old-chat", schemaVersion: 1, name: "Before the upgrade", avatarSeed: "old", modelId: "test-model",
  updatedAt: 1, createdAt: 1, snapshot: { version: 7, tree: { tree_format: 2 } },
};
const oldAutosave = { key: "test-model@500", modelVariantId: "test-model", stashedAt: 500, record: { modelVariantId: "test-model", updatedAt: 400 } };

async function seed(page: Page, autosaves = [oldAutosave]) {
  await page.evaluate(async ({ chat, autosaves, stashUrl }) => {
    await new Promise<void>((resolve, reject) => {
      const request = indexedDB.open("drowse-saved-conversations", 1);
      request.onupgradeneeded = () => {
        if (!request.result.objectStoreNames.contains("conversations")) request.result.createObjectStore("conversations", { keyPath: "id" });
      };
      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const database = request.result;
        const transaction = database.transaction("conversations", "readwrite");
        transaction.objectStore("conversations").put(chat);
        transaction.oncomplete = () => { database.close(); resolve(); };
        transaction.onerror = () => reject(transaction.error);
      };
    });
    const { BrowserHostedAutosaveStash } = await import(stashUrl);
    const stash = new BrowserHostedAutosaveStash();
    for (const entry of autosaves) await stash.put(entry);
    stash.close();
  }, { chat: oldChat, autosaves, stashUrl: modules.stash });
}

async function mountHome(page: Page) {
  await page.evaluate(async modules => {
    const [{ default: Home }, { mount }] = await Promise.all([import(modules.home), import("/e2e/svelte-runtime.ts")]);
    document.body.replaceChildren();
    mount(Home, { target: document.body, props: {
      controller: { capabilities: () => ({ signals: { appleMobile: false } }), open: async () => {}, check: async () => {} },
      snapshot: { phase: "supported", models: [{ id: "test-model", modelId: "test-model", name: "Gemma 3 1B", setupComplete: true, fit: "recommended" }],
        download: { available: true, phase: "idle", reason: "Ready" }, runtime: { available: true }, storage: { persisted: true } },
      onChooseModels() {},
    } });
  }, modules);
}

const stashed = (page: Page) => page.evaluate(async stashUrl => {
  const { BrowserHostedAutosaveStash } = await import(stashUrl);
  const stash = new BrowserHostedAutosaveStash();
  try { return (await stash.list()).map((entry: { key: string }) => entry.key); } finally { stash.close(); }
}, modules.stash);

test.beforeEach(async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("http://127.0.0.1:4176/outside-the-workbench");
});

test("old chats get one notice, download as one file and stay stored", async ({ page }) => {
  await seed(page);
  await mountHome(page);
  const notice = page.locator("section.legacy-chats");
  await expect(notice.getByRole("heading", { level: 2 })).toHaveText("2 chats from before Drowse 6.0 can’t be opened here");
  await expect(page.getByText("This chat could not be read")).toHaveCount(0);
  await expect(page.getByText("Before the upgrade")).toHaveCount(0);

  const [download] = await Promise.all([page.waitForEvent("download"), notice.getByRole("button", { name: "Download" }).click()]);
  expect(download.suggestedFilename()).toMatch(/^drowse-chats-before-6\.0-\d{4}-\d{2}-\d{2}\.json$/);
  const file = JSON.parse(await readFile(await download.path(), "utf8"));
  expect(file.format).toBe("drowse-legacy-conversations");
  expect(file.conversations).toEqual([{ id: "old-chat", record: oldChat }]);
  expect(file.autosaves.map((entry: { model_variant_id: string }) => entry.model_variant_id)).toEqual(["test-model"]);
  await expect(notice.getByRole("status")).toHaveText(`Downloaded ${download.suggestedFilename()}.`);
  expect(await stashed(page)).toEqual(["test-model@500"]);
});

test("dismissing hides the notice on this device until another old chat appears", async ({ page }) => {
  await seed(page);
  await mountHome(page);
  const notice = page.locator("section.legacy-chats");
  await expect(notice).toBeVisible();
  await notice.getByRole("button", { name: "Dismiss" }).click();
  await expect(notice).toHaveCount(0);
  expect(await stashed(page)).toEqual(["test-model@500"]);

  await mountHome(page);
  await expect(page.getByRole("heading", { name: "Saved chats" })).toBeVisible();
  await expect(page.locator("section.legacy-chats")).toHaveCount(0);
  await expect(page.getByText("This chat could not be read")).toHaveCount(0);

  await seed(page, [{ ...oldAutosave, key: "test-model@900", stashedAt: 900 }]);
  await mountHome(page);
  await expect(page.locator("section.legacy-chats").getByRole("heading", { level: 2 })).toHaveText("3 chats from before Drowse 6.0 can’t be opened here");
});

test("the notice fits a phone and a wide screen in both themes", async ({ page }) => {
  await seed(page);
  await mountHome(page);
  const notice = page.locator("section.legacy-chats");
  for (const theme of ["light", "dark"]) {
    await page.evaluate(theme => document.documentElement.dataset.theme = theme, theme);
    for (const width of [375, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      await expect(notice).toBeVisible();
      const layout = await notice.evaluate(element => ({
        overflow: element.scrollWidth > element.clientWidth || document.documentElement.scrollWidth > innerWidth,
        border: getComputedStyle(element).borderTopWidth,
      }));
      expect(layout).toEqual({ overflow: false, border: "0px" });
    }
  }
});
