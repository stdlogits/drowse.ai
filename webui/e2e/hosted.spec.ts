import {
  chooseFixtureModel, composer, inspectToken, inspector, openMenuItem, returnToChats, selectRoom,
  sendMessage, setAppearance, settle, toolDialog,
} from "./workbench-navigation";
import { expect, test, type Locator, type Page } from "@playwright/test";
import { readdir, readFile } from "node:fs/promises";
import { text as readStreamText } from "node:stream/consumers";
import { test as pwaTest } from "./pwa-update-fixture";
import { resolve } from "node:path";
import { siteTitle } from "../scripts/site-metadata.mjs";
import { waitForReply } from "./workbench-navigation";

const devUrl = "http://127.0.0.1:4176";
const fixtureResponse = "This is a deterministic local Drowse runtime fixture.";
const toastModuleUrl = `/@fs/${resolve("../drowse/webui/src/lib/stores.svelte.ts")}`;
const fixtureModelHashes = [
  "f85e180e7e8d61d1e225062424360dcc7b1997ce4c28233188af886d8511156f",
  "89ff00a0903817d42e9fddb23babfbc8d2956d5e9b044729074062d1dc41bb93",
] as const;
const fixturePackHashes = [
  "fb79e067d57e7def0b65653f5f6329db5db98144c88d61c59ff84c77f1696770",
  "e6386fbf77c663ac05ac0b296e441476ad8aa8f2cdc0c905b75f84a41e0bdb0c",
] as const;
const fixtureHashes = [...fixtureModelHashes, ...fixturePackHashes] as const;

const sendButton = (page: Page) => page.getByRole("button", { name: "Send", exact: true });

interface FixtureTokenReadoutRequest {
  service: string;
  method: string;
  args: unknown[];
}

async function installFixtureRpcProbe(page: Page): Promise<void> {
  await page.addInitScript(() => {
    const root = globalThis as typeof globalThis & {
      __drowseFixtureRpc?: {
        requests: Array<{ service: string; method: string; args: unknown[] }>;
        failNextFamily: "lens" | "sae" | null;
      };
    };
    root.__drowseFixtureRpc = { requests: [], failNextFamily: null };
    const originalPostMessage = Worker.prototype.postMessage;
    Object.defineProperty(Worker.prototype, "postMessage", {
      configurable: true,
      value: function(this: Worker, message: unknown, transfer?: unknown): void {
        const request = message && typeof message === "object"
          ? message as {
              protocolVersion?: unknown;
              requestId?: unknown;
              command?: unknown;
              payload?: unknown;
            }
          : null;
        const payload = request?.payload && typeof request.payload === "object"
          ? request.payload as { service?: unknown; method?: unknown; args?: unknown }
          : null;
        if (
          request?.command === "request" && payload?.service === "instruments" &&
          payload.method === "tokenReadout" && Array.isArray(payload.args)
        ) {
          const state = root.__drowseFixtureRpc!;
          const recorded = structuredClone(payload) as {
            service: string;
            method: string;
            args: unknown[];
          };
          state.requests.push(recorded);
          const family = recorded.args[0];
          if (family === state.failNextFamily) {
            state.failNextFamily = null;
            queueMicrotask(() => {
              this.dispatchEvent(new MessageEvent("message", {
                data: {
                  protocolVersion: request.protocolVersion,
                  kind: "response",
                  requestId: request.requestId,
                  ok: false,
                  error: {
                    code: "FIXTURE_REPLAY_FAILED",
                    message: "Fixture token replay failed. Reopen the model and try again.",
                    recoverable: true,
                    status: 409,
                  },
                },
              }));
            });
            return;
          }
        }
        Reflect.apply(
          originalPostMessage,
          this,
          transfer === undefined ? [message] : [message, transfer],
        );
      },
    });
  });
}

async function installFixtureTreeDeleteFailure(page: Page): Promise<void> {
  await page.addInitScript(() => {
    const root = globalThis as typeof globalThis & {
      __drowseFailNextTreeDelete?: boolean;
    };
    root.__drowseFailNextTreeDelete = false;
    const originalPostMessage = Worker.prototype.postMessage;
    Object.defineProperty(Worker.prototype, "postMessage", {
      configurable: true,
      value: function(this: Worker, message: unknown, transfer?: unknown): void {
        const request = message && typeof message === "object"
          ? message as {
              protocolVersion?: unknown;
              requestId?: unknown;
              command?: unknown;
              payload?: unknown;
            }
          : null;
        const payload = request?.payload && typeof request.payload === "object"
          ? request.payload as { service?: unknown; method?: unknown }
          : null;
        if (
          root.__drowseFailNextTreeDelete && request?.command === "request" &&
          payload?.service === "tree" && payload.method === "delete"
        ) {
          root.__drowseFailNextTreeDelete = false;
          queueMicrotask(() => {
            this.dispatchEvent(new MessageEvent("message", {
              data: {
                protocolVersion: request.protocolVersion,
                kind: "response",
                requestId: request.requestId,
                ok: false,
                error: {
                  code: "MUTATION_DURING_GENERATION",
                  message: "cannot delete_subtree on a node inside an in-flight generation's reservation",
                  recoverable: true,
                  status: 409,
                },
              },
            }));
          });
          return;
        }
        Reflect.apply(
          originalPostMessage,
          this,
          transfer === undefined ? [message] : [message, transfer],
        );
      },
    });
  });
}

async function openModelAndStorage(page: Page): Promise<Locator> {
  await selectRoom(page, "Model");
  const modelControls = page.getByRole("region", { name: "Model controls", exact: true });
  await expect(modelControls).toBeVisible();
  await modelControls.getByRole("button", { name: "Storage and downloads" }).click();
  return modelControls;
}

function downloadedPackRow(drawer: Locator, packId: string): Locator {
  const section = drawer.getByRole("heading", { name: "Downloaded additions" })
    .locator("..");
  return section.locator(`li[data-pack-id="${packId}"]`);
}

async function attemptPackDeletion(row: Locator): Promise<void> {
  await row.getByRole("button", { name: "Delete", exact: true }).click();
  await expect(row.getByRole("button", { name: "Cancel", exact: true })).toBeVisible();
  await row.getByRole("button", { name: "Delete tool", exact: true }).click();
}

/** Opens the Loom page's Map (the Thread's Map entry), or returns to the conversation. */
async function openWorkspace(page: Page, name: "Conversation" | "Branches") {
  const loom = page.getByRole("region", { name: "Loom", exact: true });
  if (name === "Conversation") {
    if (await loom.isVisible()) await page.getByRole("button", { name: "Back to the conversation", exact: true }).click();
    await expect(composer(page)).toBeVisible();
    return;
  }
  const map = page.getByRole("complementary", { name: "Thread", exact: true }).getByRole("button", { name: "Map", exact: true });
  if (!await map.isVisible()) await page.getByRole("button", { name: "Thread", exact: true }).click();
  await map.click();
  await expect(loom).toBeVisible();
}

async function openTranscriptDrawer(page: Page): Promise<Locator> {
  await openMenuItem(page, "Transcript");
  const drawer = toolDialog(page, "Transcript");
  await expect(drawer).toBeVisible();
  return drawer;
}

async function openFixtureWorkbench(page: Page, extraQuery = ""): Promise<void> {
  await page.goto(`${devUrl}/app?fixture=1${extraQuery}`);
  await chooseFixtureModel(page);
  await expect(
    page.getByRole("heading", { name: "Choose your first model" }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: "Qwen3 1.7B fixture source" }),
  ).toHaveAttribute(
    "href",
    "https://huggingface.co/Qwen/Qwen3-1.7B",
  );
  await expect(page.getByText("Apache-2.0", { exact: true })).toBeVisible();

  const toolPicker = page.locator(".tool-picker");
  await expect(toolPicker.getByText("Response controls", { exact: true })).toBeVisible();
  await expect(toolPicker.locator(".required-tool", { hasText: "Word insights" }).last()).toBeVisible();
  await expect(toolPicker.getByRole("checkbox", { name: /Word insights/ })).toHaveCount(0);
  await expect(toolPicker.getByRole("checkbox", { name: /R-lens readouts/ })).not.toBeChecked();
  await expect(toolPicker.locator(".required-tool", { hasText: "Feature insights" })).toContainText("SAE");
  await page.getByRole("button", { name: "Download and open", exact: true }).click();
  await expect(composer(page)).toBeVisible();
}

async function openInstalledFixtureWorkbench(page: Page): Promise<void> {
  await page.goto(`${devUrl}/app?fixture=1&reopen=1`);
  await expect(page.locator(".shell")).toBeVisible();
  await expect(page.getByRole("heading", { name: "This device is ready" })).toHaveCount(0);
}

async function setFixtureTokenDelay(page: Page, delay: number): Promise<void> {
  const worker = page.workers().find(worker => worker.url().includes("fixtureBrowser.worker"));
  expect(worker).toBeDefined();
  await worker!.evaluate(delay => {
    const scope = globalThis as typeof globalThis & {
      __drowseFixtureSetTimeout?: typeof setTimeout;
    };
    const original = scope.__drowseFixtureSetTimeout ?? globalThis.setTimeout;
    scope.__drowseFixtureSetTimeout = original;
    globalThis.setTimeout = ((handler, timeout, ...args) => Reflect.apply(
      original,
      globalThis,
      [handler, timeout === 35 ? delay : timeout, ...args],
    )) as typeof setTimeout;
  }, delay);
}

async function continueFromChatHome(page: Page): Promise<void> {
  await expect(page.getByRole("heading", { name: "Your chats", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Continue with Qwen3 1.7B fixture", exact: true }).click();
}

async function savedConversationAvatarSeed(page: Page, name: string): Promise<string | null> {
  return page.evaluate((conversationName) => new Promise<string | null>((resolveSeed, rejectSeed) => {
    const open = indexedDB.open("drowse-saved-conversations", 1);
    open.onerror = () => rejectSeed(open.error);
    open.onsuccess = () => {
      const database = open.result;
      const request = database.transaction("conversations", "readonly")
        .objectStore("conversations")
        .getAll();
      request.onerror = () => {
        database.close();
        rejectSeed(request.error);
      };
      request.onsuccess = () => {
        database.close();
        const record = (request.result ?? []).find(
          (candidate: { name?: string }) => candidate.name === conversationName,
        ) as { avatarSeed?: string } | undefined;
        resolveSeed(record?.avatarSeed ?? null);
      };
    };
  }), name);
}

test("chats autosave, rename on Your chats, and survive Loom cuts and clearing", async ({ page }) => {
  await openFixtureWorkbench(page);
  const libraryUrl = `/@fs/${resolve("../drowse/webui/src/lib/stores/savedConversations.svelte.ts")}`;
  const records = () => page.evaluate(async url => (await import(url)).conversationLibrary.list(), libraryUrl);
  await sendMessage(page, "Automatic marmot notes");
  await expect.poll(async () => (await records()).conversations.length).toBe(1);
  const original = (await records()).conversations[0];
  expect(original.name).toMatch(/^New Chat - [A-Z][a-z]{2} \d{1,2} - \d{1,2}:\d{2} \S+$/);
  for (const width of [320, 390, 1024, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  }
  await returnToChats(page);
  const card = page.locator(`[data-saved-conversation="${original.id}"]`);
  await card.locator(".chat-name").click();
  await card.getByRole("textbox", { name: "Chat name", exact: true }).fill("My automatic chat");
  await card.getByRole("textbox", { name: "Chat name", exact: true }).press("Enter");
  await card.getByRole("button", { name: "Open chat", exact: true }).click();
  await expect(composer(page)).toBeVisible();
  await expect(page.locator(".chat[aria-label='Chat']").getByText("Automatic marmot notes", { exact: true })).toBeVisible();
  await sendMessage(page, "Add a second turn");
  await expect.poll(async () => (await records()).conversations[0].snapshot.tree.nodes.length).toBeGreaterThan(original.snapshot.tree.nodes.length);
  const updated = (await records()).conversations[0];
  expect(updated.id).toBe(original.id);
  expect(updated.name).toBe("My automatic chat");

  await openWorkspace(page, "Branches");
  await page.getByRole("button", { name: "Cut branch…", exact: true }).click();
  const cutDialog = page.getByRole("dialog", { name: "Delete branch", exact: true });
  await expect(cutDialog).toContainText("Earlier turns and sibling branches stay");
  await cutDialog.getByRole("button", { name: /^cancel$/i }).click();
  expect((await records()).conversations[0].snapshot.tree.nodes.length).toBe(updated.snapshot.tree.nodes.length);
  await page.getByRole("button", { name: "Cut branch…", exact: true }).click();
  await cutDialog.getByRole("button", { name: "Delete branch", exact: true }).click();
  await expect(cutDialog).not.toBeVisible();
  await expect.poll(async () => (await records()).conversations[0].snapshot.tree.nodes.length).toBe(updated.snapshot.tree.nodes.length - 1);
  await page.getByRole("button", { name: "Clear loom…", exact: true }).click();
  const clearDialog = page.getByRole("dialog", { name: "Clear loom", exact: true });
  await clearDialog.getByRole("button", { name: /^cancel$/i }).click();
  await page.getByRole("button", { name: "Clear loom…", exact: true }).click();
  await clearDialog.getByRole("button", { name: "Save and clear loom", exact: true }).click();
  await expect(clearDialog).not.toBeVisible();
  expect((await records()).conversations).toHaveLength(1);
  await openWorkspace(page, "Conversation");
  await sendMessage(page, "A separate chat");
  await expect.poll(async () => (await records()).conversations.length).toBe(2);
  await returnToChats(page);
  await page.reload();
  await expect(page.locator("[data-saved-conversation]")).toHaveCount(2);
  await expect(page.locator(`[data-saved-conversation="${original.id}"]`)).toContainText("My automatic chat");
});

test("autosave failures are visible and retry preserves the same chat", async ({ page }) => {
  await openFixtureWorkbench(page);
  const libraryUrl = `/@fs/${resolve("../drowse/webui/src/lib/stores/savedConversations.svelte.ts")}`;
  const records = () => page.evaluate(async url => {
    const { conversationLibrary } = await import(url);
    const { conversations } = await conversationLibrary.list();
    return Promise.all(conversations.map((record: { id: string }) => conversationLibrary.get(record.id)));
  }, libraryUrl);
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Keep this safe");
  await sendButton(page).click();
  await expect(page.getByRole("status").filter({ hasText: "Response complete." })).toBeVisible();
  await expect.poll(async () => (await records())[0]?.snapshot.tree.nodes.some((node: { role: string; text: string }) => node.role === "assistant" && node.text === fixtureResponse)).toBe(true);
  await expect(page.locator("header.dw-top")).not.toContainText(/Autosaved|Saving chat/);
  const original = await page.evaluate(async url => {
    const { conversationLibrary } = await import(url);
    const record = await conversationLibrary.get((await conversationLibrary.list()).conversations[0].id);
    const originalPut = IDBObjectStore.prototype.put;
    (window as any).__restoreAutosave = () => { IDBObjectStore.prototype.put = originalPut; };
    IDBObjectStore.prototype.put = function(...args) {
      if (this.name === "conversations") throw new DOMException("Storage is full", "QuotaExceededError");
      return originalPut.apply(this, args);
    };
    return record;
  }, libraryUrl);
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Save this next reply too");
  await sendButton(page).click();
  const failure = page.getByRole("alert").filter({ has: page.getByRole("button", { name: "Retry", exact: true }) });
  await expect(failure).toBeVisible();
  const beforeRetry = await records();
  expect(beforeRetry).toHaveLength(1);
  expect(beforeRetry[0].snapshot.tree.nodes.length).toBe(original.snapshot.tree.nodes.length);
  await page.evaluate(() => (window as any).__restoreAutosave());
  await failure.getByRole("button", { name: "Retry", exact: true }).click();
  await expect(failure).toHaveCount(0);
  await expect.poll(async () => (await records())[0].snapshot.tree.nodes.length).toBeGreaterThan(original.snapshot.tree.nodes.length);
  const recovered = await records();
  expect(recovered).toHaveLength(1);
  expect(recovered[0].id).toBe(original.id);
  expect(recovered[0].snapshot.tree.nodes.length).toBeGreaterThan(original.snapshot.tree.nodes.length);
});

test("saved chats persist, rename in the library, and delete only after confirmation", async ({ page }) => {
  await openFixtureWorkbench(page);
  await sendMessage(page, "What do marmots do in winter?");
  await openMenuItem(page, "Save chat");
  const saveDrawer = toolDialog(page, "Save chat");
  await saveDrawer.getByRole("textbox", { name: "Name" }).fill("Marmot field notes");
  await saveDrawer.getByRole("button", { name: /^(Save|Update)$/ }).click();
  await expect(saveDrawer).not.toBeVisible();

  await page.goto(`${devUrl}/app?fixture=1`);
  await expect(page.getByRole("heading", { name: "Your chats", exact: true })).toBeVisible();
  const savedHomeCard = page.locator("[data-saved-conversation]").filter({ hasText: "Marmot field notes" });
  await savedHomeCard.getByRole("button", { name: "Open chat", exact: true }).click();
  await expect(composer(page)).toBeVisible();
  await expect(page.locator(".chat[aria-label='Chat']").getByText("What do marmots do in winter?", { exact: true })).toBeVisible();

  await openMenuItem(page, "Saved chats");
  let libraryDrawer = toolDialog(page, "Saved chats");
  let card = libraryDrawer.getByRole("list", { name: "Saved conversations" }).locator("[data-saved-conversation]").first();
  await expect(card).toBeVisible();
  await card.getByRole("button", { name: "Rename" }).click();
  await card.getByRole("textbox", { name: "Conversation name" }).fill("Hibernation notes");
  await card.getByRole("button", { name: "Save name" }).click();
  await expect(card.getByText("Hibernation notes", { exact: true })).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(libraryDrawer).not.toBeVisible();

  await openMenuItem(page, "Saved chats");
  libraryDrawer = toolDialog(page, "Saved chats");
  card = libraryDrawer.locator("[data-saved-conversation]").first();
  await expect(card.getByText("Open now", { exact: true })).toBeVisible();
  await card.getByRole("button", { name: "Delete", exact: true }).click();
  await expect(card.getByText(/This cannot be undone/)).toBeVisible();
  await card.getByRole("button", { name: "Cancel", exact: true }).click();
  await expect(card.getByText("Hibernation notes", { exact: true })).toBeVisible();
  await card.getByRole("button", { name: "Delete", exact: true }).click();
  await card.getByRole("button", { name: "Delete", exact: true }).click();
  await expect(card).toHaveCount(0);
});

test("saved chat cards keep summaries and open or export the latest stored transcript", async ({ page }) => {
  await openFixtureWorkbench(page);
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Original saved prompt");
  await sendButton(page).click();
  await expect(page.locator('.chat[aria-label="Chat"]')).toContainText(fixtureResponse);
  await expect.poll(() => page.evaluate(async (url) => (await import(url)).genStatus.active, toastModuleUrl))
    .toBe(false);
  const recordId = await page.evaluate(async ({ storesUrl, workspaceUrl }) => {
    const { conversationLibrary } = await import(storesUrl);
    const { captureConversationSnapshot } = await import(workspaceUrl);
    const snapshot = structuredClone(captureConversationSnapshot());
    snapshot.tree.root_id = `saved-${snapshot.tree.root_id}`;
    snapshot.tree.active_node_id = `saved-${snapshot.tree.active_node_id}`;
    snapshot.tree.nodes = snapshot.tree.nodes.map((node: any) => ({ ...node, id: `saved-${node.id}`, parent_id: node.parent_id === null ? null : `saved-${node.parent_id}` }));
    snapshot.tree.children_of = Object.fromEntries(Object.entries(snapshot.tree.children_of).map(([id, children]) => [`saved-${id}`, (children as string[]).map(child => `saved-${child}`)]));
    const record = await conversationLibrary.create({ name: "Fresh transcript", snapshot });
    const originalGetAll = IDBObjectStore.prototype.getAll;
    IDBObjectStore.prototype.getAll = function(...args) {
      if (this.name === "conversations") throw new Error("Saved chat browsing must not bulk-load transcripts");
      return Reflect.apply(originalGetAll, this, args);
    };
    try {
      const summaries = await conversationLibrary.listSummaries();
      if (summaries.issues.length || !summaries.conversations.some((summary: any) => summary.id === record.id) ||
        summaries.conversations.some((summary: any) => "snapshot" in summary) || !await conversationLibrary.hasAny()) {
        throw new Error("Invalid saved chat summary");
      }
    } finally {
      IDBObjectStore.prototype.getAll = originalGetAll;
    }
    return record.id as string;
  }, { storesUrl: toastModuleUrl, workspaceUrl: `/@fs/${resolve("../drowse/webui/src/lib/conversationWorkspace.ts")}` });

  const changeStoredPrompt = async (text: string) => {
    await page.evaluate(async ({ storesUrl, id, text }) => {
      const { conversationLibrary } = await import(storesUrl);
      const record = await conversationLibrary.get(id);
      record.snapshot.tree.nodes.find((node: { role: string }) => node.role === "user").text = text;
      await conversationLibrary.update(id, { snapshot: record.snapshot });
    }, { storesUrl: toastModuleUrl, id: recordId, text });
  };
  await openMenuItem(page, "Saved chats");
  const drawer = toolDialog(page, "Saved chats");
  const card = drawer.locator(`[data-saved-conversation="${recordId}"]`);
  await expect(card).toBeVisible();
  await changeStoredPrompt("Updated after opening the library");
  const downloading = page.waitForEvent("download");
  await card.getByRole("button", { name: "Download", exact: true }).click();
  const download = await downloading;
  const exported = JSON.parse(await readStreamText(await download.createReadStream()));
  expect(exported.conversation.snapshot.tree.nodes.find((node: { role: string }) => node.role === "user").text)
    .toBe("Updated after opening the library");

  await changeStoredPrompt("Latest prompt when opening the chat");
  await card.getByRole("button", { name: "Open", exact: true }).click();
  await expect(drawer).not.toBeVisible();
  await expect(page.locator('.chat[aria-label="Chat"]')).toContainText("Latest prompt when opening the chat");
});

test("Loom defers hidden edge work and preserves labels through navigation and bookmarks", async ({ page }) => {
  await page.addInitScript(() => {
    const root = globalThis as typeof globalThis & { __edgeRequests: number };
    root.__edgeRequests = 0;
    const originalPostMessage = Worker.prototype.postMessage;
    Worker.prototype.postMessage = function(message, ...rest) {
      if (message?.command === "request" && message.payload?.service === "tree" &&
        message.payload.method === "edgeLabel") root.__edgeRequests += 1;
      return Reflect.apply(originalPostMessage, this, [message, ...rest]);
    };
  });
  await openFixtureWorkbench(page);
  await page.getByRole("textbox", { name: /^Compose as / }).fill("A map performance check");
  await sendButton(page).click();
  await expect(page.locator(".chat[aria-label='Chat']")).toContainText(fixtureResponse);
  await expect.poll(() => page.evaluate(async (url) => (await import(url)).genStatus.active, toastModuleUrl))
    .toBe(false);
  const requestCount = () => page.evaluate(() =>
    (globalThis as typeof globalThis & { __edgeRequests: number }).__edgeRequests);
  expect(await requestCount()).toBe(0);
  await openWorkspace(page, "Branches");
  await expect.poll(requestCount).toBeGreaterThan(0);
  const firstCount = await requestCount();
  await expect.poll(() => page.evaluate(async (url) => (await import(url)).edgeLabelCache.size, toastModuleUrl))
    .toBe(firstCount);
  await page.evaluate(async (url) => {
    const { loomTree, loomNavigate, loomStar } = await import(url);
    const id = loomTree.active_node_id;
    await loomStar(id, true);
    await loomNavigate(id);
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  }, toastModuleUrl);
  expect(await requestCount()).toBe(firstCount);
});

async function fixtureInstallIds(page: Page): Promise<string[]> {
  return page.evaluate(() => new Promise<string[]>((resolveInstalls, rejectInstalls) => {
    const open = indexedDB.open("drowse-hosted-runtime", 5);
    open.onerror = () => rejectInstalls(open.error);
    open.onsuccess = () => {
      const database = open.result;
      const request = database.transaction("installs", "readonly")
        .objectStore("installs")
        .getAll();
      request.onerror = () => {
        database.close();
        rejectInstalls(request.error);
      };
      request.onsuccess = () => {
        database.close();
        resolveInstalls(
          (request.result ?? []).map((install: { id: string }) => install.id).sort(),
        );
      };
    };
  }));
}

async function fixtureStorageSnapshot(page: Page) {
  return page.evaluate(async (hashes) => {
    const installs = await new Promise<Array<{
      id: string;
      kind: string;
      objectHashes: string[];
    }>>((resolveInstalls, rejectInstalls) => {
      const open = indexedDB.open("drowse-hosted-runtime", 5);
      open.onerror = () => rejectInstalls(open.error);
      open.onsuccess = () => {
        const database = open.result;
        const request = database.transaction("installs", "readonly")
          .objectStore("installs")
          .getAll();
        request.onerror = () => {
          database.close();
          rejectInstalls(request.error);
        };
        request.onsuccess = () => {
          database.close();
          resolveInstalls(request.result ?? []);
        };
      };
    });
    const storage = navigator.storage as StorageManager & {
      getDirectory(): Promise<FileSystemDirectoryHandle>;
    };
    const root = await storage.getDirectory();
    const objects = await (await root.getDirectoryHandle("drowse"))
      .getDirectoryHandle("objects");
    const sizes: Record<string, number> = {};
    for (const hash of hashes) {
      const file = await (await objects.getFileHandle(`${hash}.data`)).getFile();
      sizes[hash] = file.size;
    }
    const names: string[] = [];
    const iterable = objects as FileSystemDirectoryHandle & {
      entries(): AsyncIterable<[string, FileSystemHandle]>;
    };
    for await (const [name, handle] of iterable.entries()) {
      if (handle.kind === "file") names.push(name);
    }
    names.sort();
    return {
      install: installs.find((candidate) => candidate.id === "qwen3-1.7b-fixture") ?? null,
      installs,
      sizes,
      names,
    };
  }, fixtureHashes);
}

async function fixtureDeletionSnapshot(page: Page) {
  return page.evaluate(async (hashes) => {
    const read = <T>(databaseName: string, version: number, storeName: string, key: string) =>
      new Promise<T | null>((resolveRecord, rejectRecord) => {
        const open = indexedDB.open(databaseName, version);
        open.onerror = () => rejectRecord(open.error);
        open.onsuccess = () => {
          const database = open.result;
          const request = database.transaction(storeName, "readonly")
            .objectStore(storeName)
            .get(key);
          request.onerror = () => {
            database.close();
            rejectRecord(request.error);
          };
          request.onsuccess = () => {
            database.close();
            resolveRecord(request.result ?? null);
          };
        };
      });
    const readAll = <T>(databaseName: string, version: number, storeName: string) =>
      new Promise<T[]>((resolveRecords, rejectRecords) => {
        const open = indexedDB.open(databaseName, version);
        open.onerror = () => rejectRecords(open.error);
        open.onsuccess = () => {
          const database = open.result;
          const request = database.transaction(storeName, "readonly")
            .objectStore(storeName)
            .getAll();
          request.onerror = () => {
            database.close();
            rejectRecords(request.error);
          };
          request.onsuccess = () => {
            database.close();
            resolveRecords(request.result ?? []);
          };
        };
      });
    const [install, session, installs] = await Promise.all([
      read("drowse-hosted-runtime", 5, "installs", "qwen3-1.7b-fixture"),
      read("drowse-hosted-sessions", 2, "sessions", "qwen3-1.7b-fixture"),
      readAll<{ id: string; kind: string }>("drowse-hosted-runtime", 5, "installs"),
    ]);
    const storage = navigator.storage as StorageManager & {
      getDirectory(): Promise<FileSystemDirectoryHandle>;
    };
    const root = await storage.getDirectory();
    const objects = await (await root.getDirectoryHandle("drowse"))
      .getDirectoryHandle("objects");
    const remaining: string[] = [];
    for (const hash of hashes) {
      try {
        await objects.getFileHandle(`${hash}.data`);
        remaining.push(hash);
      } catch (error) {
        if (!(error instanceof DOMException) || error.name !== "NotFoundError") throw error;
      }
    }
    return {
      install,
      session,
      packIds: installs.filter((candidate) => candidate.kind === "pack")
        .map((candidate) => candidate.id)
        .sort(),
      remaining,
    };
  }, fixtureHashes);
}

test("landing page routes into the hosted app", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));

  await page.goto("/");

  await expect(page).toHaveTitle(siteTitle);
  await expect(
    page.getByRole("heading", { name: /See inside your model/ }),
  ).toBeVisible();
  await expect(page.locator(".hero-action-row").getByRole("link", { name: "Open Drowse" })).toHaveAttribute(
    "href",
    "/app",
  );
  await expect(page.locator(".trust-line")).toContainText("Open source. Inference and saved work stay on your device.");
  await expect(page.locator(".trust-line").getByRole("link", { name: "What leaves your device" })).toHaveAttribute("href", "/privacy");
  const requirement = page.locator(".hero-action-row > span");
  await expect(requirement).toHaveAttribute("data-webgpu", /^(available|unavailable)$/);
  await expect(requirement).toHaveText(await requirement.getAttribute("data-webgpu") === "available"
    ? "WebGPU is available in this browser. Drowse checks your device's memory next."
    : "This browser can't run Drowse yet. Try a current version of Chrome, Edge or Safari 26 or later on a computer with a graphics processor.");
  await expect(page.getByRole("button", { name: "Pause background animation", exact: true })).toBeVisible();
  await page.locator(".hero-action-row").getByRole("link", { name: "Open Drowse" }).click();
  await expect(page).toHaveURL(/\/app$/);
  expect(pageErrors).toEqual([]);
});

test("model storage notice retries protection without repeating instructions or reinstalling", async ({ page }, testInfo) => {
  await page.addInitScript(() => {
    let calls = 0;
    Object.defineProperty(navigator.storage, "persist", { configurable: true, value: async () => ++calls >= 2 });
    Object.defineProperty(navigator.storage, "persisted", { configurable: true, value: async () => false });
  });
  await page.route("**/fixtureBrowser.worker.ts*", async route => {
    const response = await route.fetch();
    await route.fulfill({ response, body: (await response.text()).replace(/persisted:\s*true/g, "persisted: false") });
  });
  await openFixtureWorkbench(page);
  await page.goto(`${devUrl}/app?fixture=1&choose=1`);
  await chooseFixtureModel(page);
  const notice = page.getByRole("status", { name: "Storage protection", exact: true });
  await expect(notice).toHaveCount(1);
  await notice.getByRole("button", { name: "Protect storage", exact: true }).click();
  await expect(notice.getByRole("button", { name: "Protect storage", exact: true })).toBeEnabled();
  await expect(notice).toContainText("Downloads and chats still save on this device.");
  await expect(notice.locator("p")).toHaveCount(1);
  await expect(notice).not.toContainText("Select Protect storage");
  for (const width of [1440, 320]) {
    await page.setViewportSize({ width, height: 900 });
    expect(await notice.evaluate(element => element.scrollWidth <= element.clientWidth)).toBe(true);
  }
  await notice.getByRole("button", { name: "Protect storage", exact: true }).click();
  await expect(notice).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Open workbench", exact: true })).toBeEnabled();
});

test("a returning user sees their chat home instead of onboarding", async ({ page }) => {
  await openFixtureWorkbench(page);
  await page.goto(`${devUrl}/`);

  const open = page.locator(".hero-action-row").getByRole("link", { name: "Open Drowse", exact: true });
  await expect(open).toBeVisible();
  await expect(open).toHaveAttribute("href", "/app");
  await page.goto(`${devUrl}/app?fixture=1`);

  const logo = page.getByRole("link", { name: "Drowse home", exact: true });
  await expect(logo).toHaveAttribute("href", "/");
  await logo.click();
  await expect(page).toHaveURL(`${devUrl}/`);
  await expect(page.locator("#hero-title")).toBeVisible();
  await page.goto(`${devUrl}/app?fixture=1`);

  await continueFromChatHome(page);
  await expect(page.locator(".shell")).toBeVisible();
  await expect(page.getByRole("heading", { name: "This device is ready" })).toHaveCount(0);
  await expect(logo).toHaveAttribute("href", "/");
  await logo.click();
  await expect(page).toHaveURL(`${devUrl}/`);
  await expect(page.locator("#hero-title")).toBeVisible();
});

test("a stale workbench module reloads once and preserves the installed model", async ({ page }) => {
  let failedImports = 0;
  await page.route("**/src/App.svelte*", async (route) => {
    if (failedImports === 0) {
      failedImports += 1;
      await route.abort("failed");
      return;
    }
    await route.continue();
  });

  await openFixtureWorkbench(page);

  expect(failedImports).toBe(1);
  await expect(page).toHaveURL(`${devUrl}/app?fixture=1&reopen=1`);
  await expect(page.locator(".shell")).toBeVisible();
  expect(await fixtureInstallIds(page)).toContain("qwen3-1.7b-fixture");
});

test("a stale hosted bootstrap module reloads once instead of leaving a broken page", async ({ page }) => {
  let failedImports = 0;
  await page.route("**/src/hosted/runtime/workerFixtureRuntime.ts*", async (route) => {
    if (failedImports === 0) {
      failedImports += 1;
      await route.abort("failed");
      return;
    }
    await route.continue();
  });

  await page.goto(`${devUrl}/app?fixture=1`);

  expect(failedImports).toBe(1);
  await expect(page).toHaveURL(`${devUrl}/app?fixture=1`);
  await expect(page.getByRole("heading", { name: "Choose your first model" })).toBeVisible();
});

test("persistent missing workbench assets stop after one automatic reload and allow manual recovery", async ({ page }) => {
  let failedImports = 0;
  let missing = true;
  await page.route("**/src/App.svelte*", async (route) => {
    if (!missing) return route.continue();
    failedImports += 1;
    await route.abort("failed");
  });

  await page.goto(`${devUrl}/app?fixture=1`);
  await expect(page.getByRole("heading", { name: "Choose your first model" })).toBeVisible();
  await chooseFixtureModel(page);
  await page.getByRole("button", { name: "Download and open", exact: true }).click();

  await expect(page.getByText(
    "Some app files could not load. This can happen after an update or a connection problem. Reload Drowse to refresh its app files.",
  )).toBeVisible();
  await expect(page.getByRole("button", { name: "Reload Drowse" })).toBeVisible();
  expect(failedImports).toBe(2);
  await page.waitForTimeout(750);
  expect(failedImports).toBe(2);

  missing = false;
  await page.getByRole("button", { name: "Reload Drowse", exact: true }).click();
  await expect(page.locator(".shell")).toBeVisible();
  expect(failedImports).toBe(2);
  expect(await fixtureInstallIds(page)).toContain("qwen3-1.7b-fixture");
  expect(new URL(page.url()).searchParams.has("app-recovery")).toBe(false);
});

test("hosted navigation is cross-origin isolated", async ({ page }) => {
  await page.goto("/");
  expect(await page.evaluate(() => globalThis.crossOriginIsolated)).toBe(true);
});

test("unknown paths render the hosted 404", async ({ page }) => {
  await page.goto("/outside-the-workbench");

  await expect(page).toHaveTitle("Page not found · Drowse");
  await expect(
    page.getByRole("region", { name: "Page not found", exact: true }),
  ).toBeVisible();
  await expect(page.getByRole("region", { name: "Page not found", exact: true }).getByRole("link", { name: "Open Drowse", exact: true })).toHaveAttribute(
    "href",
    "/app",
  );
});

test("landing and onboarding fit a mobile viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");

  await expect(page.locator(".hero-action-row").getByRole("link", { name: "Open Drowse" })).toBeVisible();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1),
  ).toBe(true);

  await page.goto(`${devUrl}/app?fixture=1`);
  await expect(
    page.getByRole("heading", { name: "Choose your first model" }),
  ).toBeVisible();
  await chooseFixtureModel(page);
  await expect(page.getByRole("button", { name: "Download and open", exact: true })).toBeVisible();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1),
  ).toBe(true);
});

test("first-run offline notice does not cover the mobile primary action and restores focus", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await page.evaluate(async () => {
    if (!("serviceWorker" in navigator)) throw new Error("Service workers unavailable");
    await navigator.serviceWorker.ready;
  });

  const notice = page.getByRole("status").filter({ hasText: "The interface is ready offline." });
  await expect(notice).toBeVisible({ timeout: 15_000 });
  const primary = page.locator(".hero-action-row").getByRole("link", { name: "Open Drowse" });
  for (const width of [390, 320]) {
    await page.setViewportSize({ width, height: 844 });
    await expect.poll(async () => {
      const [noticeBox, primaryBox] = await Promise.all([notice.boundingBox(), primary.boundingBox()]);
      if (!noticeBox || !primaryBox) return false;
      const overlaps = noticeBox.x < primaryBox.x + primaryBox.width &&
        noticeBox.x + noticeBox.width > primaryBox.x &&
        noticeBox.y < primaryBox.y + primaryBox.height &&
        noticeBox.y + noticeBox.height > primaryBox.y;
      return noticeBox.y >= 0 && noticeBox.y + noticeBox.height <= 844 && !overlaps;
    }).toBe(true);
  }

  await primary.focus();
  const dismiss = notice.getByRole("button", { name: "Dismiss" });
  await dismiss.focus();
  await page.keyboard.press("Enter");
  await expect(notice).toHaveCount(0);
  await expect(primary).toBeFocused();
});

test("dismissing the landing notice does not make the hero a selectable focus container", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(async () => { await navigator.serviceWorker.ready; });
  const notice = page.getByRole("status").filter({ hasText: "The interface is ready offline." });
  await expect(notice).toBeVisible({ timeout: 15_000 });
  await notice.getByRole("button", { name: "Dismiss" }).click();
  await expect(notice).toHaveCount(0);
  await expect(page.locator(".hero .primary-action")).toBeFocused();
  await expect(page.locator("main")).toHaveAttribute("tabindex", "-1");
  await expect(page.locator("main")).toHaveCSS("outline-style", "none");
});

test("a controlled offline launch resumes PWA update monitoring after reconnect", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(async () => {
    if (!("serviceWorker" in navigator)) throw new Error("Service workers unavailable");
    await navigator.serviceWorker.ready;
  });
  await page.reload();
  await expect.poll(() => page.evaluate(() => Boolean(navigator.serviceWorker.controller))).toBe(true);

  await page.addInitScript(() => {
    const root = globalThis as typeof globalThis & {
      __drowsePwaReconnect?: { online: boolean; registrations: number };
    };
    const state = { online: false, registrations: 0 };
    root.__drowsePwaReconnect = state;
    Object.defineProperty(navigator, "onLine", {
      configurable: true,
      get: () => state.online,
    });
    const serviceWorkers = navigator.serviceWorker;
    const register = serviceWorkers.register.bind(serviceWorkers);
    Object.defineProperty(serviceWorkers, "register", {
      configurable: true,
      value: (...args: Parameters<ServiceWorkerContainer["register"]>) => {
        state.registrations += 1;
        return register(...args);
      },
    });
  });
  await page.reload();

  const notice = page.getByRole("status").filter({ hasText: "The interface is ready offline." });
  await expect(notice).toBeVisible();
  expect(await page.evaluate(() => {
    const root = globalThis as typeof globalThis & {
      __drowsePwaReconnect?: { online: boolean; registrations: number };
    };
    return root.__drowsePwaReconnect?.registrations;
  })).toBe(0);

  await page.evaluate(() => {
    const root = globalThis as typeof globalThis & {
      __drowsePwaReconnect?: { online: boolean; registrations: number };
    };
    const state = root.__drowsePwaReconnect;
    if (!state) throw new Error("Missing PWA reconnect fixture");
    state.online = true;
    window.dispatchEvent(new Event("online"));
    window.dispatchEvent(new Event("online"));
  });
  await expect.poll(
    () => page.evaluate(() => {
      const root = globalThis as typeof globalThis & {
        __drowsePwaReconnect?: { online: boolean; registrations: number };
      };
      return root.__drowsePwaReconnect?.registrations;
    }),
  ).toBe(1);
});

test("fixture installs, opens the shared workbench, stops, and generates", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await openFixtureWorkbench(page);

  const composer = page.getByRole("textbox", { name: /^Compose as / });
  const submit = sendButton(page);
  const stop = page.locator(".chat").getByRole("button", { name: /^Stop$/i, includeHidden: true });
  await page.getByRole("button", { name: "Sampling settings" }).click();
  const samplingDrawer = page.getByRole("dialog", { name: "Sampling settings" });
  await expect(samplingDrawer.getByText(
    "Set how the model samples tokens and what it saves.",
    { exact: true },
  )).toHaveCount(0);
  await expect(samplingDrawer.getByText(
    "Limit the token choices and change how repetition is scored.",
    { exact: true },
  )).toHaveCount(0);
  const alternatives = page.getByRole("spinbutton", {
    name: "Alternatives per token",
  });
  await expect(alternatives).toBeEnabled();
  await expect(alternatives).toHaveAttribute("max", String(Number.MAX_SAFE_INTEGER));
  await expect(alternatives).toHaveValue("8");
  await page.keyboard.press("Escape");
  await openWorkspace(page, "Conversation");

  await setFixtureTokenDelay(page, 2_000);
  await composer.fill("Stop this response early");
  await submit.click();
  await expect(stop).toBeEnabled();
  await stop.dispatchEvent("click");
  await waitForReply(page);
  await expect(page.locator(".msg .dw-said").last()).not.toHaveText(fixtureResponse);

  await setFixtureTokenDelay(page, 35);
  await composer.fill("Finish this response");
  await submit.click();
  await expect(page.locator(".msg .dw-said").last()).toContainText(fixtureResponse);
  await waitForReply(page);
  const storage = await fixtureStorageSnapshot(page);
  expect(storage.install).toMatchObject({
    id: "qwen3-1.7b-fixture",
    kind: "model",
    objectHashes: expect.arrayContaining([...fixtureModelHashes]),
  });
  expect(storage.installs.map((install) => install.id).sort()).toEqual([
    "fixture-jlens",
    "fixture-sae",
    "qwen3-1.7b-fixture",
  ]);
  expect(storage.sizes).toEqual({
    [fixtureHashes[0]]: 45,
    [fixtureHashes[1]]: 53,
    [fixtureHashes[2]]: 57,
    [fixtureHashes[3]]: 54,
  });
  expect(pageErrors).toEqual([]);
});

test("bottom conversation action clears the active path without deleting its branches", async ({ page }) => {
  await openFixtureWorkbench(page);

  const composer = page.getByRole("textbox", { name: /^Compose as / });
  await composer.fill("Keep this branch available");
  await sendButton(page).click();
  await expect(page.locator(".msg .dw-said").last()).toContainText(fixtureResponse);

  const clear = page.getByRole("button", { name: "Clear conversation" });
  await expect(clear).toHaveClass(/clear-conversation/);
  const [clearBox, headerBox] = await Promise.all([
    clear.boundingBox(),
    page.locator("header.dw-top").boundingBox(),
  ]);
  expect(clearBox).not.toBeNull();
  expect(headerBox).not.toBeNull();
  expect(clearBox!.y).toBeGreaterThan(headerBox!.y + headerBox!.height);
  await clear.click();
  const confirm = page.getByRole("button", { name: "Confirm clear conversation" });
  await expect(confirm).toBeVisible();
  await confirm.click();

  await expect(page.getByText("Write the first message below", { exact: true })).toBeVisible();
  await expect(page.locator(".conversation-empty h2, .empty-mark, .starter-prompts")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Clear conversation" })).toHaveCount(0);

  await openWorkspace(page, "Branches");
  await page.getByRole("button", { name: "Fit whole loom" }).click();
  await expect(page.locator(".tree-node-wrap").filter({ hasText: "Keep this branch available" })).toBeVisible();
});

test("first run includes J-lens and SAE while additional model tools remain manageable", async ({ page }) => {
  await page.goto(`${devUrl}/app?fixture=1`);
  await expect(page.getByRole("heading", { name: "Choose your first model" })).toBeVisible();
  await chooseFixtureModel(page);
  const toolPicker = page.locator(".tool-picker");
  const includedSae = toolPicker.locator(".required-tool", { hasText: "Feature insights" });
  const includedJlens = toolPicker.locator(".required-tool", { hasText: "Word insights" });
  await expect(includedJlens).toContainText("J-lens");
  await expect(includedJlens.locator('input[type="checkbox"]')).toHaveCount(0);
  await expect(includedSae).toContainText("SAE");
  await expect(includedSae.locator('input[type="checkbox"]')).toHaveCount(0);
  await expect(toolPicker.locator(".tool-total")).toContainText("209 B");

  await page.getByRole("button", { name: "Download and open", exact: true }).click();
  await expect(page.locator(".shell")).toBeVisible();
  await expect.poll(() => fixtureInstallIds(page)).toEqual([
    "fixture-jlens",
    "fixture-sae",
    "qwen3-1.7b-fixture",
  ]);

  const drawer = await openModelAndStorage(page);
  const optionalSection = drawer.locator("section.panel", {
    hasText: "Included files and additions",
  });
  const jlensOption = optionalSection.locator('li[data-pack-id="fixture-jlens"]');
  await expect(jlensOption.getByText("Ready", { exact: true })).toBeVisible();
  await expect(jlensOption.getByRole("button")).toHaveCount(0);

  const saeOption = optionalSection.locator('li[data-pack-id="fixture-sae"]');
  await expect(saeOption.getByRole("button", { name: "Download" })).toHaveCount(0);
  await expect(saeOption.getByText("Ready", { exact: true })).toBeVisible();
  await expect.poll(() => fixtureInstallIds(page)).toEqual([
    "fixture-jlens",
    "fixture-sae",
    "qwen3-1.7b-fixture",
  ]);

  const rLensOption = optionalSection.locator('li[data-pack-id="fixture-rlens"]');
  const downloadRLens = rLensOption.getByRole("button", { name: "Download" });
  await expect(downloadRLens).toBeEnabled();
  await downloadRLens.click();
  await expect(rLensOption.getByText("Ready", { exact: true })).toBeVisible();
  await expect.poll(() => fixtureInstallIds(page)).toEqual([
    "fixture-jlens",
    "fixture-rlens",
    "fixture-sae",
    "qwen3-1.7b-fixture",
  ]);

  const installedSae = downloadedPackRow(drawer, "fixture-sae");
  await attemptPackDeletion(installedSae);
  await page.waitForURL(`${devUrl}/app?fixture=1&reopen=1`);
  await expect(page.getByRole("heading", { name: "Models", exact: true })).toBeVisible();
  await expect.poll(() => fixtureInstallIds(page)).toEqual([
    "fixture-jlens",
    "fixture-rlens",
    "qwen3-1.7b-fixture",
  ]);

  await chooseFixtureModel(page);
  await page.getByRole("button", { name: "Add tools and open", exact: true }).click();
  await expect(page.locator(".shell")).toBeVisible();
  await expect.poll(() => fixtureInstallIds(page)).toEqual([
    "fixture-jlens",
    "fixture-rlens",
    "fixture-sae",
    "qwen3-1.7b-fixture",
  ]);
});

test("installed lens and SAE packs are usable without preparation errors", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await openFixtureWorkbench(page);
  // Artifacts: the installed packs are in use, and the hosted build offers no fitting or training.
  await selectRoom(page, "Artifacts");
  const sections = page.getByRole("group", { name: "Artifacts sections", exact: true });
  const lenses = page.getByRole("region", { name: "Lenses", exact: true });
  await expect(lenses.getByRole("button", { name: "Compatible data pack" })).toContainText("fixture-jlens");
  await expect(lenses.getByRole("button", { name: "In use", exact: true })).toBeDisabled();
  await expect(lenses.getByRole("button", { name: /^(confirm )?fit\b/i })).toHaveCount(0);
  await sections.getByRole("button", { name: "SAEs", exact: true }).click();
  const saes = page.getByRole("region", { name: "SAE", exact: true });
  await expect(saes.getByRole("button", { name: "Compatible data pack" })).toContainText("fixture-sae");
  await expect(saes.getByRole("button", { name: "In use", exact: true })).toBeDisabled();
  await expect(saes.getByRole("button", { name: /^(confirm )?train\b/i })).toHaveCount(0);
  await expect(sections.getByRole("button", { name: "Build", exact: true })).toHaveCount(0);

  // The Inspector's instruments: readouts switch, and a watched word and feature are listed.
  await selectRoom(page, "Workbench");
  const lens = inspector(page).getByRole("region", { name: "Lens", exact: true });
  const readout = lens.getByRole("switch", { name: "Lens readout", exact: true });
  await expect(readout).toBeChecked();
  await readout.click();
  await expect(readout).not.toBeChecked();
  await readout.click();
  await expect(readout).toBeChecked();
  await lens.getByRole("textbox", { name: "A word", exact: true }).fill("fixture");
  await lens.getByRole("button", { name: "Watch", exact: true }).click();
  const features = inspector(page).getByRole("region", { name: "Features", exact: true });
  await expect(features.getByRole("switch", { name: "Feature readout", exact: true })).toBeChecked();
  await features.getByRole("textbox", { name: "A feature number", exact: true }).fill("7");
  await features.getByRole("button", { name: "Watch", exact: true }).click();

  await sendMessage(page, "Measure installed instruments");
  await expect(lens.getByRole("list", { name: "Lens probe tokens" })).toBeVisible();
  await page.getByRole("button", { name: "Reroll assistant message" }).click();
  await waitForReply(page);
  await expect.poll(async () => page.evaluate(async (moduleUrl) => {
    const stores = await import(moduleUrl);
    const user = [...stores.loomTree.nodes.values()].find(
      (node: { text: string }) => node.text === "Measure installed instruments",
    );
    if (!user) return [];
    const childIds = stores.loomTree.children_of.get(user.id) ?? [];
    return childIds.map((id: string) => {
      const node = stores.loomTree.nodes.get(id);
      return {
        role: node?.role,
        hasLens: node?.tokens?.some((token: { measurements?: { instruments?: object } }) =>
          token.measurements?.instruments && "lens" in token.measurements.instruments
        ) ?? false,
        hasSae: node?.tokens?.some((token: { measurements?: { instruments?: object } }) =>
          token.measurements?.instruments && "sae" in token.measurements.instruments
        ) ?? false,
      };
    });
  }, toastModuleUrl)).toEqual([
    { role: "assistant", hasLens: true, hasSae: true },
    { role: "assistant", hasLens: true, hasSae: true },
  ]);
  await expect(page.getByText(/Fake runtime method .* is not implemented/)).toHaveCount(0);
  expect(pageErrors).toEqual([]);
});

test("models without an SAE say that no SAE is available", async ({ page }) => {
  await page.goto(`${devUrl}/app?fixture=1&fixtureSae=0`);
  await expect(page.getByRole("heading", { name: "Choose your first model" })).toBeVisible();
  await expect(page.getByRole("checkbox", { name: /Feature explorer/ })).toHaveCount(0);
  await chooseFixtureModel(page);
  await page.getByRole("button", { name: "Download and open", exact: true }).click();
  await expect(composer(page)).toBeVisible();
  await expect(inspector(page).getByRole("region", { name: "Features", exact: true })).toHaveCount(0);
  await sendMessage(page, "Inspect without an SAE");
  const token = await inspectToken(page, "deterministic");
  await expect(token.getByRole("region", { name: "Features", exact: true })).toHaveCount(0);
  await selectRoom(page, "Artifacts");
  await page.getByRole("group", { name: "Artifacts sections", exact: true }).getByRole("button", { name: "SAEs", exact: true }).click();
  await expect(page.getByRole("main")).toContainText(/no (compatible )?SAE/i);
});

test("changing models closes the runtime and opens installed model choices", async ({ page }) => {
  await openFixtureWorkbench(page);
  await selectRoom(page, "Model");
  const modelControls = page.getByRole("region", { name: "Model controls", exact: true });
  await expect(modelControls).toBeVisible();
  await modelControls.getByRole("button", { name: "Change model", exact: true }).click();
  await expect.poll(() => page.url()).toBe(`${devUrl}/app?fixture=1&choose=1`);
  await expect(page.getByRole("heading", { name: "Models" }))
    .toBeVisible();
  await expect(page.getByText("Installed and verified", { exact: true })).toBeVisible();
});

test("installed standard and R-lens packs swap live without fitting", async ({ page }) => {
  await page.goto(`${devUrl}/app?fixture=1`);
  await expect(page.getByRole("heading", { name: "Choose your first model" })).toBeVisible();
  await chooseFixtureModel(page);
  const toolPicker = page.locator(".tool-picker");
  const rLensChoice = toolPicker.getByRole("checkbox", { name: /R-lens readouts/ });
  await expect(rLensChoice).not.toBeChecked();
  await rLensChoice.check();
  await page.getByRole("button", { name: "Download and open", exact: true }).click();
  await expect(composer(page)).toBeVisible();
  await expect.poll(() => fixtureInstallIds(page)).toEqual([
    "fixture-jlens",
    "fixture-rlens",
    "fixture-sae",
    "qwen3-1.7b-fixture",
  ]);
  await selectRoom(page, "Artifacts");
  const lenses = page.getByRole("region", { name: "Lenses", exact: true });
  const source = lenses.getByRole("button", { name: "Compatible data pack" });
  await source.click();
  await page.getByRole("option", { name: "fixture-rlens", exact: true }).click();
  await lenses.getByRole("button", { name: "Use pack", exact: true }).click();
  await expect(source).toContainText("fixture-rlens");
  await expect(lenses.getByRole("button", { name: "In use", exact: true })).toBeDisabled();
  await expect(lenses.getByRole("button", { name: /^(confirm )?fit\b/i })).toHaveCount(0);

  await selectRoom(page, "Workbench");
  await sendMessage(page, "Use the active R-lens");
  const token = await inspectToken(page, "deterministic");
  await expect(token.getByRole("region", { name: "Across layers", exact: true })).toBeVisible();
});

test("token replay routes the exact token request and exposes a recoverable failure", async ({ page }) => {
  await installFixtureRpcProbe(page);
  await openFixtureWorkbench(page);
  // With the live readout off, a token's lens reading has to be replayed.
  const readout = inspector(page).getByRole("region", { name: "Lens", exact: true })
    .getByRole("switch", { name: "Lens readout", exact: true });
  await readout.click();
  await expect(readout).not.toBeChecked();
  await sendMessage(page, "Create a replayable token");
  await page.evaluate(() => {
    const state = (globalThis as typeof globalThis & {
      __drowseFixtureRpc: { requests: FixtureTokenReadoutRequest[]; failNextFamily: "lens" | "sae" | null };
    }).__drowseFixtureRpc;
    state.requests = [];
    state.failNextFamily = "lens";
  });
  const expected = await page.evaluate(async (moduleUrl) => {
    const stores = await import(moduleUrl);
    const turn = [...stores.chatLog.turns].reverse().find(
      (candidate: { role: string; nodeId?: string | null; tokens?: unknown[] }) =>
        candidate.role === "assistant" && candidate.nodeId && (candidate.tokens?.length ?? 0) > 0,
    );
    const token = turn?.tokens?.[0] as { rawIndex?: number | null } | undefined;
    if (!turn?.nodeId || token?.rawIndex == null) throw new Error("Fixture assistant token is missing its replay identity");
    return { nodeId: turn.nodeId, rawIndex: token.rawIndex };
  }, toastModuleUrl);
  const token = await inspectToken(page, "This");
  await expect.poll(() => page.evaluate(() =>
    (globalThis as typeof globalThis & { __drowseFixtureRpc: { requests: FixtureTokenReadoutRequest[] } })
      .__drowseFixtureRpc.requests.filter(request => request.args[0] === "lens").length,
  )).toBeGreaterThan(0);
  const request = (await page.evaluate(() =>
    (globalThis as typeof globalThis & { __drowseFixtureRpc: { requests: FixtureTokenReadoutRequest[] } })
      .__drowseFixtureRpc.requests,
  )).find(candidate => candidate.args[0] === "lens")!;
  expect(request.service).toBe("instruments");
  expect(request.method).toBe("tokenReadout");
  expect(request.args.slice(0, 3)).toEqual(["lens", expected.nodeId, expected.rawIndex]);
  await expect(token.getByRole("alert").first()).toBeVisible();
  // The failure is recoverable: the next token replays normally.
  await token.getByRole("button", { name: "Next token", exact: true }).click();
  await expect(token.getByRole("heading", { name: "is" })).toBeVisible();
  await expect(token.getByRole("region", { name: "Across layers", exact: true })).toBeVisible();
});

test("optional SAE packs cannot be deleted while steering or readings use them", async ({ page }) => {
  await openFixtureWorkbench(page);
  const features = inspector(page).getByRole("region", { name: "Features", exact: true });
  await features.getByRole("textbox", { name: "A feature number", exact: true }).fill("7");
  await features.getByRole("button", { name: "Watch", exact: true }).click();
  await page.evaluate(async (moduleUrl) => (await import(moduleUrl)).applyRecipe("0.3 sae/7"), toastModuleUrl);
  const drawer = await openModelAndStorage(page);
  await expect(downloadedPackRow(drawer, "fixture-jlens")).toHaveCount(0);
  const saeRow = downloadedPackRow(drawer, "fixture-sae");
  await attemptPackDeletion(saeRow);
  await expect(drawer.getByRole("alert")).toContainText("sae/7");
  await expect(saeRow).toBeVisible();
});

test("optional SAE packs cannot be deleted while a condition depends on them", async ({ page }) => {
  await openFixtureWorkbench(page);
  await page.evaluate(async (moduleUrl) => {
    (await import(moduleUrl)).applyRecipe("0.5 fixture/calm.focused@when:sae/7>0.1");
  }, toastModuleUrl);
  const drawer = await openModelAndStorage(page);
  const saeRow = downloadedPackRow(drawer, "fixture-sae");
  await attemptPackDeletion(saeRow);
  await expect(drawer.getByRole("alert")).toContainText(/SAE/);
  await expect(saeRow).toBeVisible();
});

test("geometry measurements populate and completed timing stays frozen", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await openFixtureWorkbench(page);
  await page.evaluate(async (moduleUrl) => (await import(moduleUrl)).attachProbe("fixture/calm.focused"), toastModuleUrl);
  const probes = inspector(page).getByRole("region", { name: "Probes", exact: true });
  await expect(probes).toContainText("calm.focused");
  await sendMessage(page, "Measure geometry");
  await expect.poll(() => page.evaluate(async (moduleUrl) => {
    const stores = await import(moduleUrl);
    const reply = [...stores.loomTree.nodes.values()].find((node: { role: string; tokens?: unknown[] | null }) =>
      node.role === "assistant" && (node.tokens?.length ?? 0) > 0) as { tokens: Array<{ measurements?: { instruments?: Record<string, unknown> } }> } | undefined;
    return Object.keys(reply?.tokens[0]?.measurements?.instruments ?? {}).sort();
  }, toastModuleUrl)).toContain("geometry");
  const status = page.getByLabel("Generation status", { exact: true });
  const settledStatus = await status.locator(".text").allInnerTexts();
  await page.waitForTimeout(750);
  expect(await status.locator(".text").allInnerTexts()).toEqual(settledStatus);
  expect(pageErrors).toEqual([]);
});

test("comparison controls appear only when usable and auto-compare completes", async ({ page }) => {
  const pageErrors: string[] = [];
  const lifecycleWarnings: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "warning" && message.text().includes("[svelte]")) lifecycleWarnings.push(message.text());
  });
  await openFixtureWorkbench(page);
  const colors = page.getByRole("button", { name: "Colors", exact: true });
  await colors.click();
  await expect(colors).toHaveAttribute("aria-expanded", "true");
  const colorPicker = page.getByRole("button", { name: /^Color generated words by/ });
  await colorPicker.click();
  await expect(page.getByRole("option", { name: "No color", exact: true })).toBeVisible();
  await expect(page.getByRole("option", { name: "Token surprisal", exact: true })).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("button", { name: "Compare two colors", exact: true })).toHaveCount(0);

  await page.evaluate(async (moduleUrl) => (await import(moduleUrl)).attachProbe("fixture/calm.focused"), toastModuleUrl);
  await sendMessage(page, "Create colors to compare");
  const colorComparison = page.getByRole("button", { name: "Compare two colors", exact: true });
  await expect(colorComparison).toBeVisible();
  await colorComparison.click();
  const second = page.getByRole("button", { name: /^Second word color/ });
  await second.click();
  await page.getByRole("option", { name: "calm.focused", exact: true }).click();
  await expect(second).toContainText("calm.focused");

  const autoCompare = page.getByRole("switch", { name: "Compare replies" });
  await autoCompare.click();
  await expect(autoCompare).toBeChecked();
  await composer(page).fill("Compare this response");
  await sendButton(page).click();
  const shadow = page.locator(".chat .msg.shadow");
  await expect(shadow.locator(".dw-said").last()).toHaveText(fixtureResponse);
  await expect.poll(() => page.evaluate(async (moduleUrl) => {
    const stores = await import(moduleUrl);
    return { active: stores.genStatus.active, processing: stores.abState.processingAb, pendingTurn: stores.abState.pendingTurnIdx };
  }, toastModuleUrl)).toEqual({ active: false, processing: false, pendingTurn: null });
  expect(pageErrors).toEqual([]);
  expect(lifecycleWarnings).toEqual([]);
});

test("fixture survives repeated generation, stop, and reload cycles", async ({ page }) => {
  test.setTimeout(180_000);
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await openFixtureWorkbench(page);

  const runGeneration = async (prompt: string, stopEarly: boolean) => {
    await setFixtureTokenDelay(page, stopEarly ? 2_000 : 35);
    const composer = page.getByRole("textbox", { name: /^Compose as / });
    const stop = page.locator(".chat").getByRole("button", { name: /^Stop$/i, includeHidden: true });
    const responses = page.locator(".msg:not(.user) .dw-said");
    const previousResponses = await responses.count();
    const response = responses.last();
    await composer.fill(prompt);
    if (stopEarly) await page.evaluate(previousResponses => {
      const observer = new MutationObserver(() => {
        const responses = document.querySelectorAll(".msg:not(.user) .dw-said");
        const stop = document.querySelector<HTMLButtonElement>('button[aria-description="Escape · stop the current reply"]');
        if (responses.length === previousResponses + 1 && responses[responses.length - 1].textContent?.includes("This") && stop && !stop.disabled) {
          observer.disconnect();
          stop.click();
        }
      });
      observer.observe(document.body, { childList: true, subtree: true, characterData: true, attributes: true, attributeFilter: ["disabled"] });
    }, previousResponses);
    await sendButton(page).click();
    await expect(responses).toHaveCount(previousResponses + 1);
    if (stopEarly) {
      await expect(response).toContainText("This");
      await expect(response).not.toHaveText(fixtureResponse);
    } else {
      await expect(response).toContainText(fixtureResponse);
    }
    await waitForReply(page);
    await expect.poll(() => page.evaluate(async (moduleUrl) => {
      const stores = await import(moduleUrl);
      return {
        active: stores.genStatus.active,
        queued: stores.pendingActions.queue.length,
      };
    }, toastModuleUrl)).toEqual({ active: false, queued: 0 });
  };

  for (let index = 0; index < 20; index += 1) {
    await runGeneration(`Lifecycle generation ${index}`, index % 4 === 0);
  }
  for (let index = 0; index < 6; index += 1) {
    await openInstalledFixtureWorkbench(page);
    await runGeneration(`Reload generation ${index}`, false);
  }

  const storage = await fixtureStorageSnapshot(page);
  expect(storage.names).toEqual(fixtureHashes.map((hash) => `${hash}.data`).sort());
  expect(pageErrors).toEqual([]);
});

test("reload during generation restores the stable user turn without a partial assistant", async ({ page }) => {
  await openFixtureWorkbench(page);
  await setFixtureTokenDelay(page, 2_000);
  const prompt = "Persist before interrupted decode";
  await page.getByRole("textbox", { name: /^Compose as / }).fill(prompt);
  await sendButton(page).click();
  await expect(page.getByRole("button", { name: /^Stop$/i })).toBeEnabled();
  await expect(page.locator(".msg .dw-said").last()).toContainText("This");
  await expect(page.locator(".msg .dw-said").last()).not.toHaveText(fixtureResponse);

  await openInstalledFixtureWorkbench(page);
  await expect(
    page.locator(".chat .log").getByText(prompt, { exact: true }),
  ).toBeVisible();
  await expect(page.locator('.role-chip[title="assistant"]')).toHaveCount(0);

  await page.getByRole("textbox", { name: /^Compose as / }).fill("Continue after recovery");
  await sendButton(page).click();
  await expect(page.locator(".msg .dw-said").last()).toContainText(fixtureResponse);
});

test("reroll creates an assistant sibling under the original user turn", async ({ page }) => {
  await openFixtureWorkbench(page);
  const prompt = "Create a reroll branch";
  await page.getByRole("textbox", { name: /^Compose as / }).fill(prompt);
  await sendButton(page).click();
  await expect(page.locator(".msg .dw-said").last()).toContainText(fixtureResponse);

  await page.getByRole("button", { name: "Reroll assistant message" }).click();
  await expect(page.getByRole("button", { name: /^Stop$/i })).toBeEnabled();
  await waitForReply(page);

  const branch = await page.evaluate(async (moduleUrl) => {
    const stores = await import(moduleUrl);
    const nodes = [...stores.loomTree.nodes.values()];
    const user = nodes.find((node: { text: string }) => node.text === "Create a reroll branch");
    if (!user) throw new Error("Reroll fixture user node is missing");
    const children = stores.loomTree.children_of.get(user.id) ?? [];
    return {
      childIds: [...children],
      childParents: children.map((id: string) => stores.loomTree.nodes.get(id)?.parent_id),
      childRoles: children.map((id: string) => stores.loomTree.nodes.get(id)?.role),
    };
  }, toastModuleUrl);
  expect(branch.childIds).toHaveLength(2);
  expect(new Set(branch.childParents).size).toBe(1);
  expect(branch.childRoles).toEqual(["assistant", "assistant"]);

  await openWorkspace(page, "Branches");
  const loom = page.locator(".loom-canvas");
  await expect(loom).toHaveAttribute("data-loom-nodes", "4");
  await expect(loom).toHaveAttribute("data-loom-edges", "3");
  await expect(loom.locator("[data-loom-junction]")).toHaveCount(1);
  const parentCard = page.locator(`[data-node-id="${branch.childParents[0]}"]`);
  const firstReply = page.locator(`[data-node-id="${branch.childIds[0]}"]`);
  const secondReply = page.locator(`[data-node-id="${branch.childIds[1]}"]`);
  const [parentBox, firstBox, secondBox] = await Promise.all([
    parentCard.boundingBox(),
    firstReply.boundingBox(),
    secondReply.boundingBox(),
  ]);
  expect(parentBox).not.toBeNull();
  expect(firstBox).not.toBeNull();
  expect(secondBox).not.toBeNull();
  expect(parentBox!.x + parentBox!.width).toBeLessThan(firstBox!.x);
  expect(firstBox!.x).toBeCloseTo(secondBox!.x, 0);
  expect(Math.abs(firstBox!.y - secondBox!.y)).toBeGreaterThan(firstBox!.height);

  const initialZoom = Number(await loom.getAttribute("data-loom-zoom"));
  await page.getByRole("button", { name: "Zoom in" }).click();
  await expect.poll(async () => Number(await loom.getAttribute("data-loom-zoom")))
    .toBeGreaterThan(initialZoom);
  await page.getByRole("button", { name: "Fit whole loom" }).click();

  await parentCard.getByRole("button", { name: "Hide child paths" }).click();
  await expect(loom).toHaveAttribute("data-loom-nodes", "1");
  await expect(loom).toHaveAttribute("data-loom-edges", "0");
  await parentCard.getByRole("button", { name: "Show child paths" }).click();
  await expect(loom).toHaveAttribute("data-loom-nodes", "4");
  await expect(loom).toHaveAttribute("data-loom-edges", "3");

  await page.evaluate(async ({ moduleUrl, childIds }) => {
    const stores = await import(moduleUrl);
    stores.openDrawer("node_compare", { node_ids: childIds });
  }, { moduleUrl: toastModuleUrl, childIds: branch.childIds });
  const comparison = page.getByRole("dialog", { name: "Compare branches" });
  await expect(comparison).toBeVisible();
  await expect(comparison.getByText(/rank-1 changes/)).toBeVisible();
  await expect(comparison.getByRole("alert")).toHaveCount(0);
});

test("a saved token alternative creates an exact sibling loom branch", async ({ page }) => {
  await openFixtureWorkbench(page);
  const prompt = "Fork this response from one token";
  await page.getByRole("textbox", { name: /^Compose as / }).fill(prompt);
  await sendButton(page).click();

  const response = page.locator(".msg .dw-said").last();
  await expect(response).toContainText(fixtureResponse);
  await waitForReply(page);
  const drilldown = await inspectToken(page, "This");
  const startBranch = drilldown.getByRole("list", { name: "Alternatives at this position" })
    .getByRole("button", { name: /^Branch with / }).first();
  await expect(startBranch).toBeEnabled();

  const source = await page.evaluate(async ({ moduleUrl, prompt }) => {
    const stores = await import(moduleUrl);
    const user = [...stores.loomTree.nodes.values()].find(
      (node: { text: string }) => node.text === prompt,
    );
    if (!user) throw new Error("Token-fork fixture user node is missing");
    const sourceId = (stores.loomTree.children_of.get(user.id) ?? [])[0];
    const sourceNode = stores.loomTree.nodes.get(sourceId);
    return {
      parentId: user.id,
      sourceId,
      sourceFirstToken: sourceNode?.raw_token_ids?.[0],
    };
  }, { moduleUrl: toastModuleUrl, prompt });

  await startBranch.click();
  await expect(page.locator(".msg .dw-said").last()).toContainText("Alternative-0 ");
  await waitForReply(page);

  const fork = await page.evaluate(async ({ moduleUrl, parentId, sourceId }) => {
    const stores = await import(moduleUrl);
    const siblings = stores.loomTree.children_of.get(parentId) ?? [];
    const forkId = siblings.find((id: string) => id !== sourceId);
    const node = forkId ? stores.loomTree.nodes.get(forkId) : null;
    return {
      activeId: stores.loomTree.active_node_id,
      siblingIds: [...siblings],
      forkId: forkId ?? null,
      forkParentId: node?.parent_id ?? null,
      forkFirstToken: node?.raw_token_ids?.[0] ?? null,
      forkText: node?.text ?? null,
    };
  }, { moduleUrl: toastModuleUrl, parentId: source.parentId, sourceId: source.sourceId });

  expect(fork.siblingIds).toHaveLength(2);
  expect(fork.forkParentId).toBe(source.parentId);
  expect(fork.activeId).toBe(fork.forkId);
  expect(fork.forkFirstToken).not.toBe(source.sourceFirstToken);
  expect(fork.forkText).toMatch(/^Alternative-0 /);

  await openWorkspace(page, "Branches");
  await expect(page.getByRole("treeitem")).toHaveCount(3);
});

test("typed text replaces a token in a new sibling loom branch", async ({ page }) => {
  await openFixtureWorkbench(page);
  const prompt = "Replace this response token with authored text";
  await page.getByRole("textbox", { name: /^Compose as / }).fill(prompt);
  await sendButton(page).click();

  const response = page.locator(".msg .dw-said").last();
  await expect(response).toContainText(fixtureResponse);
  await waitForReply(page);
  const drilldown = await inspectToken(page, "This");
  const act = drilldown.getByRole("region", { name: "Act on this token", exact: true });
  await act.getByRole("button", { name: "Branch here", exact: true }).click();
  const replacement = act.getByRole("textbox", { name: "Branch with", exact: true });
  await expect(replacement).toBeFocused();
  await replacement.fill("Authored ");
  await act.getByRole("button", { name: "Branch", exact: true }).click();
  await expect(page.locator(".msg .dw-said").last()).toContainText("Authored ");
  await waitForReply(page);
  const siblings = await page.evaluate(async ({ moduleUrl, prompt }) => {
    const stores = await import(moduleUrl);
    const user = [...stores.loomTree.nodes.values()].find(
      (node: { text: string }) => node.text === prompt,
    );
    return user ? (stores.loomTree.children_of.get(user.id) ?? []).length : 0;
  }, { moduleUrl: toastModuleUrl, prompt });
  expect(siblings).toBe(2);
});

test("a sentence in the loom starts an exact sibling continuation", async ({ page }) => {
  await openFixtureWorkbench(page);
  const prompt = "Continue from this sentence";
  await page.getByRole("textbox", { name: /^Compose as / }).fill(prompt);
  await sendButton(page).click();
  await expect(page.locator(".msg .dw-said").last()).toContainText(fixtureResponse);
  await expect.poll(() => page.evaluate(async (moduleUrl) => {
    const stores = await import(moduleUrl);
    return stores.genStatus.active;
  }, toastModuleUrl)).toBe(false);

  const source = await page.evaluate(async ({ moduleUrl, prompt }) => {
    const stores = await import(moduleUrl);
    const user = [...stores.loomTree.nodes.values()].find(
      (node: { text: string }) => node.text === prompt,
    );
    if (!user) throw new Error("Sentence-branch fixture user node is missing");
    const sourceId = (stores.loomTree.children_of.get(user.id) ?? [])[0];
    const node = stores.loomTree.nodes.get(sourceId);
    return {
      parentId: user.id,
      sourceId,
      rawTokenIds: [...(node?.raw_token_ids ?? [])],
    };
  }, { moduleUrl: toastModuleUrl, prompt });

  await openWorkspace(page, "Branches");
  const sourceCard = page.locator(`[data-loom-node-id="${source.sourceId}"]`);
  const sentence = sourceCard.locator("[data-loom-sentence]").first();
  await expect(sentence).toContainText("This is a deterministic local Drowse runtime fixture.");
  const branchButton = sentence.getByRole("button", { name: /^Branch after sentence 1:/ });
  const branchLayout = await Promise.all([
    sourceCard.boundingBox(),
    branchButton.boundingBox(),
  ]);
  expect(branchLayout[0]).not.toBeNull();
  expect(branchLayout[1]).not.toBeNull();
  expect(branchLayout[1]!.y + branchLayout[1]!.height)
    .toBeLessThanOrEqual(branchLayout[0]!.y + branchLayout[0]!.height + 1);
  await branchButton.click();
  await expect.poll(() => page.evaluate(async ({ moduleUrl, parentId }) => {
    const stores = await import(moduleUrl);
    return (stores.loomTree.children_of.get(parentId) ?? []).length;
  }, { moduleUrl: toastModuleUrl, parentId: source.parentId })).toBe(2);
  await expect.poll(() => page.evaluate(async (moduleUrl) => {
    const stores = await import(moduleUrl);
    return stores.genStatus.active;
  }, toastModuleUrl)).toBe(false);

  const fork = await page.evaluate(async ({ moduleUrl, parentId, sourceId }) => {
    const stores = await import(moduleUrl);
    const siblings = stores.loomTree.children_of.get(parentId) ?? [];
    const forkId = siblings.find((id: string) => id !== sourceId);
    const node = forkId ? stores.loomTree.nodes.get(forkId) : null;
    return {
      activeId: stores.loomTree.active_node_id,
      forkId: forkId ?? null,
      parentId: node?.parent_id ?? null,
      rawTokenIds: [...(node?.raw_token_ids ?? [])],
    };
  }, { moduleUrl: toastModuleUrl, parentId: source.parentId, sourceId: source.sourceId });

  expect(fork.forkId).not.toBeNull();
  expect(fork.activeId).toBe(fork.forkId);
  expect(fork.parentId).toBe(source.parentId);
  expect(fork.rawTokenIds).toEqual(source.rawTokenIds);
  await expect(page.locator(".loom-canvas")).toHaveAttribute("data-loom-edges", "3");
});

test("loom projections stay synchronized across path, options, and starred points", async ({ page }) => {
  await openFixtureWorkbench(page);
  const prompt = "Show the synchronized loom views";
  await page.getByRole("textbox", { name: /^Compose as / }).fill(prompt);
  await sendButton(page).click();
  await expect(page.locator(".msg .dw-said").last()).toContainText(fixtureResponse);
  await page.getByRole("button", { name: "Reroll assistant message" }).click();
  await waitForReply(page);

  await openWorkspace(page, "Branches");
  await expect(page.getByRole("group", { name: "Loom views" })).toBeVisible();

  await page.getByRole("button", { name: /Current path/ }).click();
  const path = page.getByRole("list", { name: "Current conversation path" });
  await expect(path.locator("[data-path-snippet]")).toHaveCount(2);
  await expect(path.getByText(prompt, { exact: true })).toBeVisible();
  await expect(path.getByRole("button", { name: "Branch after sentence" })).toBeEnabled();

  await page.getByRole("button", { name: /Next options/ }).click();
  const options = page.getByRole("list", { name: "Alternative paths" });
  await expect(options.locator("[data-option-node]")).toHaveCount(2);
  const firstOption = options.locator("[data-option-node]").first();
  await firstOption.getByRole("button", { name: "Save for later" }).click();
  await expect(firstOption.getByText("saved", { exact: true })).toBeVisible();

  await page.getByRole("button", { name: /^Starred/ }).click();
  const saved = page.getByRole("list", { name: "Starred conversation points" });
  await expect(saved.locator("[data-saved-node]")).toHaveCount(1);
  await expect(saved).toContainText(fixtureResponse);
  await saved.getByRole("button", { name: "Remove" }).click();
  await expect(saved).toHaveCount(0);
  const emptySaved = page.locator(".saved-projection .projection-empty");
  await expect(emptySaved).toBeVisible();
  await expect(emptySaved).toContainText("Nothing saved yet");

  await page.getByRole("button", { name: /^Map/ }).click();
  await expect(page.locator(".loom-canvas")).toHaveAttribute("data-loom-nodes", "4");
});

test("the loom exposes every generated token as an exact branch point", async ({ page }) => {
  await openFixtureWorkbench(page);
  const prompt = "Show every token in the loom";
  await page.getByRole("textbox", { name: /^Compose as / }).fill(prompt);
  await sendButton(page).click();
  await expect(page.locator(".msg .dw-said").last()).toContainText(fixtureResponse);

  await expect.poll(() => page.evaluate(async ({ moduleUrl, prompt }) => {
    const stores = await import(moduleUrl);
    const user = [...stores.loomTree.nodes.values()].find(
      (node: { text: string }) => node.text === prompt,
    );
    if (!user) return 0;
    const nodeId = (stores.loomTree.children_of.get(user.id) ?? [])[0];
    return stores.loomTree.nodes.get(nodeId)?.tokens?.length ?? 0;
  }, { moduleUrl: toastModuleUrl, prompt })).toBeGreaterThan(0);

  const generated = await page.evaluate(async ({ moduleUrl, prompt }) => {
    const stores = await import(moduleUrl);
    const user = [...stores.loomTree.nodes.values()].find(
      (node: { text: string }) => node.text === prompt,
    );
    if (!user) throw new Error("Token-loom fixture user node is missing");
    const nodeId = (stores.loomTree.children_of.get(user.id) ?? [])[0];
    const node = stores.loomTree.nodes.get(nodeId);
    return {
      userId: user.id,
      nodeId,
      tokens: node?.tokens?.map((token: { text: string }) => token.text) ?? [],
      rawTokenIds: [...(node?.raw_token_ids ?? [])],
    };
  }, { moduleUrl: toastModuleUrl, prompt });

  await openWorkspace(page, "Branches");
  await page.getByRole("button", { name: "Fit whole loom" }).click();
  const card = page.locator(`[data-loom-node-id="${generated.nodeId}"]`);
  const tokenNodes = card.locator("[data-loom-token-node]");
  await expect(tokenNodes).toHaveCount(generated.tokens.length);
  await expect(card).toContainText("deterministic");

  const targetIndex = Math.min(2, generated.tokens.length - 1);
  const siblingCountBefore = await page.evaluate(async ({ moduleUrl, parentId }) => {
    const stores = await import(moduleUrl);
    return (stores.loomTree.children_of.get(parentId) ?? []).length;
  }, { moduleUrl: toastModuleUrl, parentId: generated.userId });
  await tokenNodes.nth(targetIndex).click();
  const tokenTools = inspector(page).getByRole("region", { name: "Token", exact: true });
  await expect(tokenTools.getByRole("heading", { name: generated.tokens[targetIndex].trim() })).toBeVisible();
  await tokenTools.getByRole("list", { name: "Alternatives at this position" })
    .getByRole("button", { name: /^Branch with / }).first().click();
  await expect.poll(() => page.evaluate(async (moduleUrl) => {
    const stores = await import(moduleUrl);
    return stores.loomTree.active_node_id;
  }, toastModuleUrl)).not.toBe(generated.nodeId);
  await expect.poll(() => page.evaluate(async ({ moduleUrl, parentId }) => {
    const stores = await import(moduleUrl);
    return (stores.loomTree.children_of.get(parentId) ?? []).length;
  }, { moduleUrl: toastModuleUrl, parentId: generated.userId })).toBe(siblingCountBefore + 1);
  const fork = await page.evaluate(async ({ moduleUrl, parentId, sourceId }) => {
    const stores = await import(moduleUrl);
    const siblings = stores.loomTree.children_of.get(parentId) ?? [];
    const forkId = siblings.find((id: string) => id !== sourceId);
    const source = stores.loomTree.nodes.get(sourceId);
    const node = forkId ? stores.loomTree.nodes.get(forkId) : null;
    return {
      sourceRawTokenIds: [...(source?.raw_token_ids ?? [])],
      forkRawTokenIds: [...(node?.raw_token_ids ?? [])],
    };
  }, {
    moduleUrl: toastModuleUrl,
    parentId: generated.userId,
    sourceId: generated.nodeId,
  });
  expect(fork.sourceRawTokenIds).toEqual(generated.rawTokenIds);
  // The fork keeps every token before the branch point and replaces the one at it.
  expect(fork.forkRawTokenIds.slice(0, targetIndex)).toEqual(generated.rawTokenIds.slice(0, targetIndex));
  expect(fork.forkRawTokenIds[targetIndex]).not.toBe(generated.rawTokenIds[targetIndex]);
});

test("arbitrary role labels and structural role swaps survive generation", async ({ page }) => {
  await openFixtureWorkbench(page);
  await page.getByRole("button", { name: /^Roles / }).click();
  await page.evaluate(async (moduleUrl) => {
    const stores = await import(moduleUrl);
    stores.samplingState.user_role = "critic";
    stores.samplingState.assistant_role = "guide";
  }, toastModuleUrl);

  await expect(page.getByRole("combobox", { name: "You write as" })).toHaveValue("critic");
  await expect(page.getByRole("combobox", { name: "Model writes as" })).toHaveValue("guide");
  await page.getByRole("button", { name: "Swap writer roles" }).click();
  await expect(page.getByRole("combobox", { name: "You write as" })).toHaveValue("guide");
  await expect(page.getByRole("combobox", { name: "Model writes as" })).toHaveValue("critic");

  const prompt = "Continue this role-swapped scene";
  await page.getByRole("textbox", { name: "Compose as guide" }).fill(prompt);
  await sendButton(page).click();
  await expect(page.locator(".msg .dw-said").last()).toContainText(fixtureResponse);

  const roles = await page.evaluate(async ({ moduleUrl, prompt }) => {
    const stores = await import(moduleUrl);
    const authored = [...stores.loomTree.nodes.values()].find(
      (node: { text: string }) => node.text === prompt,
    );
    if (!authored) throw new Error("Role-swapped authored node is missing");
    const generatedId = (stores.loomTree.children_of.get(authored.id) ?? [])[0];
    const generated = stores.loomTree.nodes.get(generatedId);
    return {
      authored: [authored.role, authored.role_label],
      generated: [generated?.role, generated?.role_label],
    };
  }, { moduleUrl: toastModuleUrl, prompt });
  expect(roles).toEqual({
    authored: ["assistant", "guide"],
    generated: ["user", "critic"],
  });
});

test("custom roles can be entered directly, configured, and used for generation", async ({ page }) => {
  await openFixtureWorkbench(page);
  await page.getByRole("button", { name: /^Roles / }).click();

  const generatedRole = page.getByRole("combobox", { name: "Model writes as" });
  await generatedRole.fill("critic");
  await expect(generatedRole).toHaveValue("critic");

  await page.getByRole("button", { name: "Role settings" }).click();
  const drawer = page.getByRole("dialog", { name: "Role settings" });
  await expect(drawer.getByText(
    "Set reusable behavior for roles already used in this conversation.",
    { exact: true },
  )).toHaveCount(0);
  await expect(drawer.getByText(
    "Choose a role to configure. Change role names directly beside the message box.",
    { exact: true },
  )).toHaveCount(0);
  await drawer.getByRole("button", { name: /critic/i }).click();
  const note = drawer.getByRole("textbox", { name: "Private note" });
  await note.fill("Reviews the response carefully");
  await drawer.getByRole("button", { name: /^assistant\b/i }).click();
  await drawer.getByRole("button", { name: /^critic\b/i }).click();
  await expect(note).toHaveValue("Reviews the response carefully");
  await expect(drawer.getByRole("status").filter({ hasText: /^Saved$/ })).toHaveText("Saved");
  await page.keyboard.press("Escape");
  await expect(generatedRole).toHaveValue("critic");

  const prompt = "Review this as a critic";
  await page.getByRole("textbox", { name: /^Compose as / }).fill(prompt);
  await sendButton(page).click();
  await expect(page.locator(".msg .dw-said").last()).toContainText(fixtureResponse);

  const generated = await page.evaluate(async ({ moduleUrl, prompt }) => {
    const stores = await import(moduleUrl);
    const authored = [...stores.loomTree.nodes.values()].find(
      (node: { text: string }) => node.text === prompt,
    );
    if (!authored) throw new Error("Custom-role authored node is missing");
    const generatedId = (stores.loomTree.children_of.get(authored.id) ?? [])[0];
    const node = stores.loomTree.nodes.get(generatedId);
    return [node?.role, node?.role_label];
  }, { moduleUrl: toastModuleUrl, prompt });
  expect(generated).toEqual(["assistant", "critic"]);
});

test("raw completion mode submits and preserves a flat base-model buffer", async ({ page }) => {
  await openFixtureWorkbench(page);
  await page.getByRole("button", { name: "Sampling settings" }).click();
  const settings = page.getByRole("dialog", { name: "Sampling settings" });
  await settings.getByRole("button", { name: "Additional parameters" }).click();
  await settings.getByRole("button", { name: "Raw completion", exact: true }).click();
  await settings.getByRole("button", { name: "Close drawer" }).click();
  await openWorkspace(page, "Conversation");
  const buffer = page.getByLabel("Completion buffer", { exact: true });
  await expect(buffer).toBeVisible();
  const editor = buffer.getByRole("textbox", { name: "Editable completion buffer" });
  await editor.fill("Raw prefix: ");
  await buffer.getByRole("button", { name: "Continue text", exact: true }).click();
  await expect(editor).toHaveValue(`Raw prefix: ${fixtureResponse}`);
  await expect.poll(() => page.evaluate(async (moduleUrl) => {
    const stores = await import(moduleUrl);
    return stores.genStatus.active;
  }, toastModuleUrl)).toBe(false);

  const rawNodes = await page.evaluate(async (moduleUrl) => {
    const stores = await import(moduleUrl);
    return [...stores.loomTree.nodes.values()].map(
      (node: { text: string; role: string }) => ({ text: node.text, role: node.role }),
    );
  }, toastModuleUrl);
  expect(rawNodes.some((node) => node.text === "Raw prefix: " && node.role === "user")).toBe(true);
  expect(rawNodes.some((node) => node.text === fixtureResponse && node.role === "assistant")).toBe(true);
});

test("online local generation never sends prompts or activations", async ({ page }) => {
  await openFixtureWorkbench(page);
  const promptSentinel = "DROWSE_PRIVATE_PROMPT_7f6fbb8b";
  const activationSentinel = "DROWSE_PRIVATE_ACTIVATION_2384d3d1";
  const requests: Array<{ url: string; body: string }> = [];
  page.on("request", (request) => {
    requests.push({ url: request.url(), body: request.postData() ?? "" });
  });

  const composer = page.getByRole("textbox", { name: /^Compose as / });
  await composer.fill(`${promptSentinel} ${activationSentinel}`);
  await sendButton(page).click();
  await expect(page.locator(".msg .dw-said").last()).toContainText(fixtureResponse);

  for (const request of requests) {
    expect(decodeURIComponent(request.url)).not.toContain(promptSentinel);
    expect(decodeURIComponent(request.url)).not.toContain(activationSentinel);
    expect(request.body).not.toContain(promptSentinel);
    expect(request.body).not.toContain(activationSentinel);
  }
  expect(
    requests.filter((request) => new URL(request.url).origin !== new URL(devUrl).origin),
  ).toEqual([]);
});

test("browser transcript YAML exports and imports through the worker runtime", async ({ page }) => {
  await openFixtureWorkbench(page);
  const prompt = "Round-trip this local transcript";
  await page.getByRole("textbox", { name: /^Compose as / }).fill(prompt);
  await sendButton(page).click();
  await expect(page.locator(".msg .dw-said").last()).toContainText(fixtureResponse);

  const drawer = await openTranscriptDrawer(page);
  await drawer.getByRole("button", { name: "preview YAML" }).click();
  const rendered = drawer.getByRole("textbox", { name: "Rendered transcript YAML" });
  await expect(rendered).toHaveValue(/drowse_transcript: 2/);
  await expect(rendered).toHaveValue(new RegExp(prompt));
  const yaml = await rendered.inputValue();

  await drawer.getByRole("tab", { name: "import" }).click();
  await drawer.getByRole("textbox", { name: "Transcript YAML to import" }).fill(yaml);
  await drawer.getByRole("button", { name: "import", exact: true }).click();
  await expect(drawer.getByText(/imported · leaf/)).toBeVisible();
  await drawer.getByRole("button", { name: "close", exact: true }).click();
  await expect(
    page.locator(".chat .log").getByText(prompt, { exact: true }),
  ).toBeVisible();
});

test("worker-backed fixture restores its loom and generates offline without redownloading", async ({
  context,
  page,
}) => {
  await openFixtureWorkbench(page);
  const composer = page.getByRole("textbox", { name: /^Compose as / });
  await composer.fill("Persist this conversation across reload");
  await sendButton(page).click();
  await expect(page.locator(".msg .dw-said").last()).toContainText(fixtureResponse);
  await waitForReply(page);

  const artifactRequests: string[] = [];
  page.on("request", (request) => {
    if (request.url().includes(".fixture.bin")) artifactRequests.push(request.url());
  });
  await openInstalledFixtureWorkbench(page);
  expect(artifactRequests).toEqual([]);
  await expect(
    page.locator(".chat .log")
      .getByText("Persist this conversation across reload", { exact: true }),
  ).toBeVisible();
  await expect(page.locator(".msg .dw-said").last()).toContainText(fixtureResponse);

  const offlineRequests: string[] = [];
  page.on("request", (request) => offlineRequests.push(request.url()));
  await context.setOffline(true);
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Generate entirely offline");
  await sendButton(page).click();
  await expect(page.locator(".msg .dw-said").last()).toContainText(fixtureResponse);
  expect(offlineRequests).toEqual([]);
});

test("deleting the active model unloads it and removes its files while preserving chats and packs", async ({
  page,
}) => {
  await openFixtureWorkbench(page);
  const composer = page.getByRole("textbox", { name: /^Compose as / });
  await composer.fill("Delete this model-specific conversation");
  await sendButton(page).click();
  await expect(page.getByRole("button", { name: /^Stop$/i })).toBeEnabled();

  const modelControls = await openModelAndStorage(page);
  const installedModels = modelControls.getByRole("region", { name: "Downloaded models" });
  await installedModels.getByRole("button", { name: "Delete", exact: true }).click();
  await installedModels.getByRole("button", { name: "Delete model", exact: true }).click();

  await page.waitForURL(`${devUrl}/app?fixture=1&choose=1`);
  await expect.poll(() => page.evaluate(() => localStorage.getItem("drowse.entry.v1"))).toBeNull();
  await expect(page.getByRole("heading", { name: "Choose your first model", exact: true })).toBeVisible();
  await expect.poll(async () => fixtureDeletionSnapshot(page)).toEqual({
    install: null,
    session: expect.objectContaining({
      modelVariantId: "qwen3-1.7b-fixture",
      tree: expect.objectContaining({
        nodes: expect.arrayContaining([
          expect.objectContaining({ role: "user", text: "Delete this model-specific conversation" }),
        ]),
      }),
    }),
    packIds: ["fixture-jlens", "fixture-sae"],
    remaining: [...fixturePackHashes],
  });
});

test("an idle tab cooperatively transfers the exclusive GPU lock", async ({ context, page }) => {
  await openFixtureWorkbench(page);
  await expect.poll(() => page.evaluate(async () =>
    (await navigator.locks.query()).held?.some((lock) => lock.name === "drowse-webgpu-runtime") ?? false
  )).toBe(true);

  const requester = await context.newPage();
  await requester.goto(`${devUrl}/app?fixture=1`);
  await continueFromChatHome(requester);

  await expect(requester.locator(".shell")).toBeVisible({ timeout: 15_000 });
  await expect(page.locator(".shell")).toHaveCount(0);
  await expect.poll(() => requester.evaluate(async () =>
    (await navigator.locks.query()).held?.some((lock) => lock.name === "drowse-webgpu-runtime") ?? false
  )).toBe(true);
});

test("exclusive GPU ownership survives repeated cooperative tab transfers", async ({
  context,
  page,
}) => {
  const pageErrors: string[] = [];
  context.on("page", (candidate) => {
    candidate.on("pageerror", (error) => pageErrors.push(error.message));
  });
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await openFixtureWorkbench(page);

  let owner = page;
  for (let index = 0; index < 6; index += 1) {
    const requester = await context.newPage();
    await requester.goto(`${devUrl}/app?fixture=1`);
    await continueFromChatHome(requester);

    await expect(requester.locator(".shell")).toBeVisible({ timeout: 15_000 });
    await expect(owner.locator(".shell")).toHaveCount(0);
    await expect.poll(() => requester.evaluate(async () =>
      (await navigator.locks.query()).held?.filter(
        (lock) => lock.name === "drowse-webgpu-runtime",
      ).length ?? 0
    )).toBe(1);
    owner = requester;
  }

  expect(pageErrors).toEqual([]);
});

test("a busy owner explicitly approves takeover before generation is stopped", async ({
  context,
  page,
}) => {
  await openFixtureWorkbench(page, "&fixtureSlow=1");
  const requester = await context.newPage();
  await requester.goto(`${devUrl}/app?fixture=1&choose=1`);
  await chooseFixtureModel(requester);
  await expect(
    requester.getByRole("heading", { name: "Models" }),
  ).toBeVisible();
  await expect(requester.getByText("Installed and verified")).toBeVisible();
  await requester.getByRole("link", { name: "Chats", exact: true }).click();
  await expect(requester.getByRole("heading", { name: "Your chats", exact: true })).toBeVisible();

  await page.getByRole("textbox", { name: /^Compose as / }).fill("Keep the runtime busy");
  await sendButton(page).click();
  await expect(page.getByRole("button", { name: /^Stop$/i })).toBeEnabled();
  await continueFromChatHome(requester);

  const dialog = page.getByRole("alertdialog", { name: "Another tab wants to use the GPU" });
  await expect(dialog).toBeVisible();
  const keep = dialog.getByRole("button", { name: "Keep working here" });
  const release = dialog.getByRole("button", { name: "Release runtime" });
  await expect(keep).toBeFocused();
  await page.keyboard.press("Tab");
  await expect(release).toBeFocused();
  await release.click();

  await expect(requester.locator(".shell")).toBeVisible({ timeout: 15_000 });
  await expect(page.locator(".shell")).toHaveCount(0);
});

test("loom tree, node menu, and modal are fully keyboard scoped", async ({ page }) => {
  await openFixtureWorkbench(page);
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Create loom nodes");
  await sendButton(page).click();
  await waitForReply(page);

  await openWorkspace(page, "Branches");
  const tree = page.getByRole("tree", { name: /Conversation (loom|threads)/ });
  const items = tree.getByRole("treeitem");
  await expect(items).toHaveCount(2);
  await items.first().focus();
  await page.keyboard.press("End");
  const activeItem = items.last();
  await expect(activeItem).toBeFocused();
  await page.keyboard.press("Shift+F10");

  const menu = page.getByRole("menu", { name: "Node actions" });
  await expect(menu).toBeVisible();
  const menuItems = menu.getByRole("menuitem");
  await expect(menuItems.first()).toBeFocused();
  await page.keyboard.press("ArrowDown");
  await expect(menuItems.nth(1)).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(activeItem).toBeFocused();

  const visibleActions = activeItem.getByRole("button", { name: /^Actions for / });
  await visibleActions.focus();
  await page.keyboard.press("Enter");
  await expect(menu).toBeVisible();
  await expect.poll(async () => {
    const [nodeBox, menuBox] = await Promise.all([
      activeItem.boundingBox(),
      menu.boundingBox(),
    ]);
    if (!nodeBox || !menuBox) return Number.POSITIVE_INFINITY;
    const horizontalGap = Math.max(
      0,
      nodeBox.x - (menuBox.x + menuBox.width),
      menuBox.x - (nodeBox.x + nodeBox.width),
    );
    const verticalGap = Math.max(
      0,
      nodeBox.y - (menuBox.y + menuBox.height),
      menuBox.y - (nodeBox.y + nodeBox.height),
    );
    return Math.hypot(horizontalGap, verticalGap);
  }).toBeLessThanOrEqual(32);
  const menuAttachment = await Promise.all([
    activeItem.boundingBox(),
    menu.boundingBox(),
  ]);
  expect(menuAttachment[0]).not.toBeNull();
  expect(menuAttachment[1]).not.toBeNull();
  const [nodeBox, menuBox] = menuAttachment as [
    NonNullable<(typeof menuAttachment)[0]>,
    NonNullable<(typeof menuAttachment)[1]>,
  ];
  const viewport = page.viewportSize();
  expect(viewport).not.toBeNull();
  expect(menuBox.x).toBeGreaterThanOrEqual(8);
  expect(menuBox.x + menuBox.width).toBeLessThanOrEqual(viewport!.width - 8);
  expect(menuBox.y).toBeGreaterThanOrEqual(8);
  expect(menuBox.y + menuBox.height).toBeLessThanOrEqual(viewport!.height - 8);
  await page.keyboard.press("Escape");
  await expect(visibleActions).toBeFocused();

  await activeItem.focus();
  await page.keyboard.press("Shift+F10");
  await menu.getByRole("menuitem", { name: "Edit this turn…" }).click();
  const modal = page.getByRole("dialog", { name: "Edit turn" });
  await expect(modal).toBeVisible();
  await expect(modal.getByRole("textbox", { name: "Node text" })).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(activeItem).toBeFocused();
});

test("a reserved branch cannot be deleted and a rejected mutation keeps the loom visible", async ({ page }) => {
  await installFixtureTreeDeleteFailure(page);
  await openFixtureWorkbench(page);
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Create a branch to protect");
  await sendButton(page).click();
  await expect(page.locator(".msg .dw-said").last()).toContainText(fixtureResponse);
  await waitForReply(page);

  const result = await page.evaluate(async (moduleUrl) => {
    const stores = await import(moduleUrl);
    const nodeId = stores.loomTree.active_node_id;
    if (!nodeId) throw new Error("Fixture conversation has no active Loom node");
    const before = stores.loomTree.nodes.size;

    stores.genStatus.active = true;
    stores.loomTree.pendingNodeId = nodeId;
    await stores.loomDelete(nodeId);
    const guardedToast = stores.toasts.entries.at(-1)?.message ?? null;

    stores.genStatus.active = false;
    (globalThis as typeof globalThis & { __drowseFailNextTreeDelete?: boolean })
      .__drowseFailNextTreeDelete = true;
    await stores.loomDelete(nodeId);
    stores.genStatus.active = true;
    stores.loomTree.pendingNodeId = nodeId;
    return {
      nodeId,
      before,
      after: stores.loomTree.nodes.size,
      treeError: stores.loomTree.error,
      guardedToast,
      rejectedToast: stores.toasts.entries.at(-1)?.message ?? null,
    };
  }, toastModuleUrl);
  expect(result.after).toBe(result.before);
  expect(result.treeError).toBeNull();
  expect(result.guardedToast).toBe(
    "Finish or stop the current reply before deleting this branch.",
  );
  expect(result.rejectedToast).toBe(
    "delete: Finish or stop the current reply before changing that part of the conversation.",
  );

  await openWorkspace(page, "Branches");
  await expect(page.getByText("tree unavailable", { exact: true })).toHaveCount(0);
  const pendingNode = page.locator(`[data-node-id="${result.nodeId}"]`);
  await pendingNode.getByRole("button", { name: /^Actions for / }).click();
  await expect(
    page.getByRole("menu", { name: "Node actions" })
      .getByRole("menuitem", { name: "Delete this branch…" }),
  ).toBeDisabled();

  await page.evaluate(async (moduleUrl) => {
    const stores = await import(moduleUrl);
    stores.genStatus.active = false;
    stores.loomTree.pendingNodeId = null;
  }, toastModuleUrl);
});

test("unsupported authoring commands are filtered before runtime requests", async ({ page }) => {
  await openFixtureWorkbench(page);
  await page.getByRole("button", { name: "Search everything (⌘K)", exact: true }).click();
  const palette = page.getByRole("dialog", { name: "Search everything", exact: true });
  const search = palette.getByRole("combobox", { name: "Search everything" });
  await search.fill("templates");
  await expect(palette.getByRole("option", { name: /template/i })).toHaveCount(0);
  await search.fill("build a probe");
  await expect(palette.getByRole("option", { name: /build a probe/i })).toHaveCount(0);
  // A supported command is still found.
  await search.fill("transcript");
  await expect(palette.getByRole("option", { name: /transcript/i }).first()).toBeVisible();
});

test("error notifications are sticky and timed notifications pause while inspected", async ({
  page,
}) => {
  await openFixtureWorkbench(page);
  const errorTtl = await page.evaluate(async (moduleUrl) => {
    const store = await import(moduleUrl);
    const id = store.pushToast("Persistent fixture error", { kind: "error" });
    return store.toasts.entries.find((toast: { id: number }) => toast.id === id)?.ttlMs;
  }, toastModuleUrl);
  expect(errorTtl).toBeNull();
  const error = page.getByRole("alert").filter({ hasText: "Persistent fixture error" });
  await expect(error).toBeVisible();
  for (const width of [1440, 390, 320]) {
    await page.setViewportSize({ width, height: 900 });
    await expect.poll(async () => {
      const toast = (await error.boundingBox())!;
      const header = (await page.locator("header.dw-top").boundingBox())!;
      return toast.y >= header.y + header.height;
    }).toBe(true);
    await page.getByRole("button", { name: "Menu", exact: true }).click();
    await expect(page.getByRole("menu", { name: "Menu", exact: true })).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(error).toBeVisible();
  }
  await error.getByRole("button", { name: /Dismiss notification/ }).click();
  await expect(error).toHaveCount(0);

  await page.evaluate(async (moduleUrl) => {
    const store = await import(moduleUrl);
    store.pushToast("Pause this notification", { ttlMs: 800 });
  }, toastModuleUrl);
  const notice = page.getByRole("status").filter({ hasText: "Pause this notification" });
  await expect(notice).toBeVisible();
  await notice.hover();
  await page.waitForTimeout(1_000);
  await expect(notice).toBeVisible();
  await page.mouse.move(1, 1);
  await expect(notice).toHaveCount(0);
});

test("portable pack management remains available when browser fitting is disabled", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openFixtureWorkbench(page);
  await selectRoom(page, "Artifacts");
  const sections = page.getByRole("group", { name: "Artifacts sections", exact: true });
  await expect(sections.getByRole("button", { name: "Build", exact: true })).toHaveCount(0);
  await sections.getByRole("button", { name: "Packs", exact: true }).click();
  await expect(sections.getByRole("button", { name: "Packs", exact: true })).toHaveAttribute("aria-pressed", "true");
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true);
});

test("production build excludes the deterministic fixture", async () => {
  const assetsDirectory = resolve("dist-hosted", "assets");
  const scripts = (await readdir(assetsDirectory)).filter((name) => name.endsWith(".js"));
  const sources = await Promise.all(
    scripts.map((name) => readFile(resolve(assetsDirectory, name), "utf8")),
  );
  const productionJavaScript = sources.join("\n");

  expect(productionJavaScript).not.toContain("development-fixture");
  expect(productionJavaScript).not.toContain("DeterministicFakeRuntime");
  expect(productionJavaScript).not.toContain("fixtureHostedRuntime");
  expect(productionJavaScript).not.toContain("fixture.invalid");
  expect((await readdir(assetsDirectory)).filter((name) => name.endsWith(".bin"))).toEqual([]);
});

test("installed PWA shell remains available offline without preloading the model engine", async ({ context, page }) => {
  const assetNames = await readdir(resolve("dist-hosted", "assets"));
  const webLlmChunks = assetNames
    .filter((name) => /^drowse-web-llm-[A-Za-z0-9_-]+\.js$/u.test(name));
  expect(webLlmChunks).toHaveLength(1);
  const webLlmUrl = `/assets/${webLlmChunks[0]}`;
  const shellRegistries = assetNames.filter((name) => /^registry-[A-Za-z0-9_-]+\.js$/u.test(name));
  expect(shellRegistries.length).toBeGreaterThan(0);

  await page.goto("/");
  await page.evaluate(async () => {
    if (!("serviceWorker" in navigator)) throw new Error("Service workers unavailable");
    await navigator.serviceWorker.ready;
  });
  await page.reload();
  await expect
    .poll(() => page.evaluate(() => Boolean(navigator.serviceWorker.controller)))
    .toBe(true);

  expect(await page.evaluate(async (url) => Boolean(await caches.match(url)), webLlmUrl)).toBe(false);
  for (const name of shellRegistries) {
    expect(await page.evaluate(async (url) => Boolean(await caches.match(url)), `/assets/${name}`)).toBe(true);
  }

  await context.setOffline(true);
  await page.goto("/app", { waitUntil: "domcontentloaded" });

  await expect(page.getByRole("link", { name: "Drowse home" })).toBeVisible();
  await expect(page.locator("#device-check")).toBeVisible();
});

pwaTest("an installed PWA update waits for consent and reloads without clearing local data", async ({
  page,
}, testInfo) => {
  pwaTest.setTimeout(90_000);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.clock.install();

  await page.goto("/");
  await page.evaluate(async () => {
    localStorage.setItem("drowse-pwa-upgrade-test", "retained");
    await navigator.serviceWorker.ready;
  });
  await page.reload();
  await expect.poll(() => page.evaluate(() => Boolean(navigator.serviceWorker.controller))).toBe(true);

  expect((await page.request.post("/__drowse_test__/update")).status()).toBe(204);
  await page.evaluate(async () => {
    const registration = await navigator.serviceWorker.ready;
    await registration.update();
  });

  const notice = page.getByRole("status").filter({ hasText: "A Drowse update is ready." });
  await expect(notice).toBeVisible();
  await expect(notice.getByRole("button", { name: "Update and reload" })).toBeVisible();
  await expect(notice).toHaveCSS("position", "fixed");
  await expect(notice).toHaveCSS("transform", "none");
  for (const width of [1440, 390, 320]) {
    await page.setViewportSize({ width, height: 900 });
    await expect.poll(() => notice.evaluate((element, width) => {
      const bounds = element.getBoundingClientRect();
      return bounds.width <= 400 && bounds.x >= 15 && bounds.right <= width - 15 && bounds.bottom <= 885;
    }, width)).toBe(true);
    const bounds = (await notice.boundingBox())!;
    expect(bounds.width).toBeLessThanOrEqual(400);
    expect(bounds.x).toBeGreaterThanOrEqual(15);
    expect(bounds.x + bounds.width).toBeLessThanOrEqual(width - 15);
    expect(bounds.y + bounds.height).toBeLessThanOrEqual(885);
    expect(await notice.evaluate(element => element.scrollWidth <= element.clientWidth)).toBe(true);
    await expect(notice.getByRole("button", { name: "Update later" })).toBeVisible();
  }
  await notice.getByRole("button", { name: "Update later" }).click();
  await expect(notice).toHaveCount(0);
  expect(await page.evaluate(async () => Boolean((await navigator.serviceWorker.getRegistration())?.waiting))).toBe(true);
  await page.reload();
  await expect(notice).toHaveCount(0);
  await page.clock.fastForward(3_600_000);
  await expect(notice).toBeVisible();
  await Promise.all([
    page.waitForEvent("domcontentloaded"),
    notice.getByRole("button", { name: "Update and reload" }).click(),
  ]);

  await expect(page.locator(".hero-action-row").getByRole("link", { name: "Open Drowse" })).toBeVisible();
  expect(await page.evaluate(() => localStorage.getItem("drowse-pwa-upgrade-test"))).toBe("retained");
  await expect.poll(() => page.evaluate(async () => {
    const registration = await navigator.serviceWorker.getRegistration();
    return {
      controlled: Boolean(navigator.serviceWorker.controller),
      active: registration?.active?.state ?? null,
      waiting: Boolean(registration?.waiting),
      installing: Boolean(registration?.installing),
    };
  })).toEqual({ controlled: true, active: "activated", waiting: false, installing: false });
});
