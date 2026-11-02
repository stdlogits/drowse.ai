import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { resolve } from "node:path";
import { buffer as readStreamBuffer, text as readStreamText } from "node:stream/consumers";
import {
  chooseFixtureModel, composer, devUrl, inspector, openFixtureWorkbench, openMenuItem, returnToChats,
  selectRoom, sendMessage, settle, sourceUrl, storesUrl, toolDialog, waitForReply,
} from "./workbench-navigation";

// Saved chats through the hosted shell and the 6.0 workbench: backups, import, Your chats,
// Download chat, and the model's settings (reset, version).

const servicesUrl = sourceUrl("lib/runtime/services.ts");
const themeUrl = sourceUrl("lib/theme.ts");

async function workbench(page: Page) {
  await page.goto(`${devUrl}/app?layoutFixture=instruments`);
  await expect(composer(page)).toBeVisible();
}

function expectCopiedConversation(copy: any, original: any) {
  expect(copy.id).not.toBe(original.id);
  expect(copy.name).toBe(`${original.name} (copy)`);
  expect(copy.snapshot.tree.nodes).toHaveLength(original.snapshot.tree.nodes.length);
  const sourceIds = new Map(copy.snapshot.tree.nodes.map((node: any, index: number) => [node.id, original.snapshot.tree.nodes[index].id]));
  for (const node of copy.snapshot.tree.nodes) expect(original.snapshot.tree.nodes.some((source: any) => source.id === node.id)).toBe(false);
  const tree = copy.snapshot.tree;
  expect({
    ...copy,
    id: original.id, name: original.name, createdAt: original.createdAt, updatedAt: original.updatedAt,
    snapshot: { ...copy.snapshot, tree: {
      ...tree,
      root_id: sourceIds.get(tree.root_id), active_node_id: sourceIds.get(tree.active_node_id),
      nodes: tree.nodes.map((node: any) => ({ ...node, id: sourceIds.get(node.id), parent_id: node.parent_id === null ? null : sourceIds.get(node.parent_id) })),
      children_of: Object.fromEntries(Object.entries(tree.children_of).map(([parent, children]) => [sourceIds.get(parent), (children as string[]).map(id => sourceIds.get(id))])),
    } },
  }).toEqual(original);
}

test("chat backups download, import separately, and reopen the entire Loom", async ({ page, browserName }) => {
  test.skip(browserName === "webkit", "The worker requires OPFS; WebKit import/download is covered by the chat card test.");
  test.setTimeout(120_000);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(`${devUrl}/app?fixture=1`);
  await chooseFixtureModel(page);
  await page.getByRole("button", { name: "Download and open", exact: true }).click();
  await expect(composer(page)).toBeVisible();
  await page.getByRole("textbox", { name: /^Compose as / }).fill("A complete local backup");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(page.locator('.chat[aria-label="Chat"]')).toContainText("This is a deterministic local Drowse runtime fixture.");
  await waitForReply(page);
  const libraryUrl = `/@fs${resolve("../drowse/webui/src/lib/stores/savedConversations.svelte.ts")}`;
  const original = await page.evaluate(async ({ libraryUrl, workspaceUrl }) => {
    const { conversationLibrary, flushConversationAutosave } = await import(libraryUrl);
    await flushConversationAutosave();
    const { captureConversationSnapshot } = await import(workspaceUrl);
    const snapshot = structuredClone(captureConversationSnapshot());
    const answer = snapshot.tree.nodes.find((node: any) => node.role === "assistant");
    if (!answer?.tokens.length) throw new Error("Generation did not capture token data");
    const branch = { ...structuredClone(answer), id: crypto.randomUUID(), notes: "Keep this alternate branch", starred: true };
    snapshot.tree.nodes.push(branch);
    snapshot.tree.children_of[branch.parent_id].push(branch.id);
    snapshot.tree.children_of[branch.id] = [];
    return conversationLibrary.create({ name: "Backup experiment", avatarSeed: "backup-avatar", snapshot });
  }, { libraryUrl, workspaceUrl: `/@fs${resolve("../drowse/webui/src/lib/conversationWorkspace.ts")}` });
  await page.goto(`${devUrl}/app?fixture=1`);
  const source = page.locator(`[data-saved-conversation="${original.id}"]`);
  await expect(source).toBeVisible();
  const downloading = page.waitForEvent("download");
  await source.getByRole("button", { name: "Download backup of Backup experiment", exact: true }).click();
  const download = await downloading;
  expect(download.suggestedFilename()).toBe("Backup-experiment.drowsechat");
  const text = await readStreamText(await download.createReadStream());
  const backup = JSON.parse(text);
  expect(backup.conversation).toEqual(original);
  const records = () => page.evaluate(async url => (await (await import(url)).conversationLibrary.list()).conversations, libraryUrl);
  const before = await records();
  const fileInput = page.getByLabel("Import chat backup files", { exact: true });
  await fileInput.setInputFiles({ name: download.suggestedFilename(), mimeType: "application/json", buffer: Buffer.from(text) });
  await expect(page.locator(".import-duplicate")).toContainText("Backup experiment” is already imported.");
  expect(await records()).toEqual(before);
  await page.getByRole("button", { name: "Import as a copy", exact: true }).click();
  await expect(page.locator(".backup-result")).toContainText("Backup experiment (copy)");
  const after = await records();
  expect(after).toHaveLength(before.length + 1);
  const imported = after.find((record: any) => record.name === `${original.name} (copy)` && record.id !== original.id)!;
  expectCopiedConversation(imported, original);
  expect(after.find((record: any) => record.id === original.id)).toEqual(original);
  backup.conversation.name = "Damaged copy";
  await fileInput.setInputFiles({ name: "damaged.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify(backup)) });
  await expect(page.getByRole("alert")).toContainText("integrity check");
  expect(await records()).toEqual(after);
  await page.reload();
  const card = page.locator(`[data-saved-conversation="${imported.id}"]`);
  for (const theme of ["light", "dark"]) {
    await page.evaluate(async ({ url, theme }) => (await import(url)).setTheme(theme), { url: themeUrl, theme });
    for (const width of [320, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      await card.scrollIntoViewIfNeeded();
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    }
    await settle(page);
    const accessibility = await new AxeBuilder({ page }).include("#saved-chats").analyze();
    expect(accessibility.violations).toEqual([]);
  }
  await card.getByRole("button", { name: "Open chat", exact: true }).click();
  await expect(composer(page)).toBeVisible();
  await expect(page.locator("main#main")).not.toHaveAttribute("aria-busy", "true");
  const restored = await page.evaluate(async url => (await import(url)).apiTree.get(), servicesUrl);
  expect({ ...restored, session_id: imported.snapshot.tree.session_id, rev: imported.snapshot.tree.rev }).toEqual(imported.snapshot.tree);
  await expect(page.locator('.chat[aria-label="Chat"]')).toContainText("A complete local backup");
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Continue only the imported copy");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect.poll(() => page.evaluate(async url => (await (await import(url)).apiTree.get()).nodes.length, servicesUrl)).toBe(original.snapshot.tree.nodes.length + 2);
  await waitForReply(page);
  await page.evaluate(async url => (await import(url)).flushConversationAutosave(), libraryUrl);
  const continued = await records();
  expect(continued.find((record: any) => record.id === original.id)).toEqual(original);
  expect(continued.find((record: any) => record.id === imported.id).snapshot.tree.nodes).toHaveLength(original.snapshot.tree.nodes.length + 2);
});

test("Your chats from the Menu keeps the conversation, and Continue restores it", async ({ page, browserName }) => {
  test.skip(browserName === "webkit", "Headless WebKit rejects OPFS; chat restoration requires persistent worker storage.");
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await page.goto(`${devUrl}/app?fixture=1`);
  await chooseFixtureModel(page);
  await page.getByRole("button", { name: "Download and open", exact: true }).click();
  await expect(composer(page)).toBeVisible();
  await sendMessage(page, "Remember this path when I go home.");
  const reply = await page.locator(".msg .dw-said").last().innerText();
  await returnToChats(page);
  await expect(page.getByRole("heading", { name: "Your chats", exact: true })).toBeVisible();
  await expect(page.locator(".shell")).toHaveCount(0);
  await expect(page.getByRole("heading", { name: "Your chats", exact: true })).toBeFocused();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.getByRole("button", { name: "Continue with Qwen3 1.7B fixture", exact: true }).click();
  await expect(composer(page)).toBeVisible();
  await expect(page.locator('.chat[aria-label="Chat"]')).toContainText("Remember this path when I go home.");
  await expect(page.locator(".msg .dw-said").last()).toHaveText(reply);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await returnToChats(page);
  await expect(page.getByRole("heading", { name: "Your chats", exact: true })).toBeVisible();
  await expect(page.locator(".shell")).toHaveCount(0);
});

test("header download confirms filename and exact backup size without changing the chat", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await workbench(page);
  await sendMessage(page, "Preserve this chat in the backup.");
  const before = await page.evaluate(async url => (await import(url)).captureConversationSnapshot(), sourceUrl("lib/conversationWorkspace.ts"));
  await openMenuItem(page, "Download chat");
  const dialog = page.getByRole("dialog", { name: "Download chat", exact: true });
  await expect(dialog.getByLabel("File name", { exact: true })).toBeEnabled();
  await dialog.getByRole("button", { name: "Cancel", exact: true }).click();
  await expect(dialog).toHaveCount(0);
  await openMenuItem(page, "Download chat");
  const filename = dialog.getByLabel("File name", { exact: true });
  await filename.fill("   ");
  await expect(dialog.getByRole("button", { name: "Download", exact: true })).toBeDisabled();
  await filename.fill("My backup.drowse-chat.json");
  const size = Number(await dialog.getByLabel("Backup size").getAttribute("data-bytes"));
  expect(size).toBeGreaterThan(0);
  for (const theme of ["light", "dark"]) {
    await page.evaluate(async ({ url, theme }) => (await import(url)).setTheme(theme), { url: themeUrl, theme });
    for (const width of [320, 1280]) {
      await page.setViewportSize({ width, height: 900 });
      expect(await dialog.evaluate(el => el.scrollWidth <= el.clientWidth)).toBe(true);
    }
  }
  const pendingDownload = page.waitForEvent("download");
  await dialog.getByRole("button", { name: "Download", exact: true }).click();
  const download = await pendingDownload;
  expect(download.suggestedFilename()).toBe("My-backup.drowsechat");
  const bytes = await readStreamBuffer(await download.createReadStream());
  expect(bytes.byteLength).toBe(size);
  const backup = JSON.parse(bytes.toString());
  expect(backup.format).toBe("drowse-chat-backup");
  expect(backup.conversation.snapshot.tree).toEqual(before.tree);
  expect(backup.conversation.snapshot.samplingState).toEqual(before.samplingState);
  expect(backup.conversation.name).not.toBe("My backup.drowse-chat.json");
  await expect(dialog).toHaveCount(0);
});

test("model reset preserves chats and restores defaults after reopening", async ({ page, browserName }) => {
  test.skip(browserName === "webkit", "Persistent worker storage requires OPFS.");
  await openFixtureWorkbench(page);
  await sendMessage(page, "Keep this conversation after reset.");
  const slider = inspector(page).getByRole("slider", { name: "Temperature", exact: true });
  const original = await slider.inputValue();
  await slider.focus();
  await page.keyboard.press("End");
  const changed = await slider.inputValue();
  expect(changed).not.toBe(original);
  await openMenuItem(page, "System prompt");
  const prompt = toolDialog(page, "System prompt");
  await prompt.locator("textarea").fill("Use a changed system prompt.");
  await prompt.getByRole("button", { name: /^Save/ }).click();
  await page.reload();
  await expect(composer(page)).toBeVisible();
  await expect(slider).toHaveValue(changed);
  await selectRoom(page, "Model");
  const reset = page.locator("main .reset-settings.full");
  await reset.getByRole("button", { name: "Reset settings", exact: true }).click();
  await reset.getByRole("button", { name: "Cancel", exact: true }).click();
  await reset.getByRole("button", { name: "Reset settings", exact: true }).click();
  await reset.getByRole("button", { name: "Reset", exact: true }).click();
  await expect(reset.getByRole("button", { name: "Cancel", exact: true })).toHaveCount(0);
  await selectRoom(page, "Workbench");
  await expect(slider).toHaveValue(original);
  await openMenuItem(page, "System prompt");
  await expect(toolDialog(page, "System prompt").locator("textarea")).toHaveValue("");
  await page.keyboard.press("Escape");
  await expect(page.locator('.chat[aria-label="Chat"]')).toContainText("Keep this conversation after reset.");
});

test("model settings end with the versioned Drowse footer", async ({ page }) => {
  test.slow();
  await page.emulateMedia({ reducedMotion: "reduce" });
  await workbench(page);
  await selectRoom(page, "Model");
  const footer = page.getByRole("contentinfo", { name: "About Drowse" }).or(page.locator(".product-footer"));
  await expect(footer.locator(".wordmark")).toHaveText("Drowse");
  await expect(footer.locator(".version")).toHaveText("v 0.1");
  await expect(footer).toContainText("© 2026 Drowse Contributors");
  await expect(footer).toContainText("GNU AGPL v3 or later");
  await expect(footer.locator('a[href="/LICENSE"]')).toHaveCount(0);
  await expect(footer.getByRole("link", { name: "Contribute" })).toHaveAttribute("href", "https://drowse.ai/contribute");
  await expect(footer.getByRole("link", { name: "Credits" })).toHaveAttribute("href", "/credits");
  for (const theme of ["light", "dark"]) {
    await page.evaluate(async ({ url, theme }) => (await import(url)).setTheme(theme), { url: themeUrl, theme });
    for (const width of [320, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      await footer.scrollIntoViewIfNeeded();
      await expect(footer).toBeInViewport();
      expect(await footer.evaluate(el => el.scrollWidth <= el.clientWidth)).toBe(true);
      await settle(page);
      expect((await new AxeBuilder({ page }).include(".product-footer").analyze()).violations).toEqual([]);
    }
  }
});

test.describe("saved chat card interactions", () => {
  test.use({ hasTouch: false, isMobile: false });
  test("chat names edit in place and backups round-trip from the card", async ({ page }) => {
    const mountHome = () => page.evaluate(async ({ home, saved, stores, workspace }) => {
      const [{ default: HostedHome }, { mount }, { conversationLibrary, registerConversationAutosave }, { sessionState }, { captureConversationSnapshot }] = await Promise.all([
        import(home), import("/e2e/svelte-runtime.ts"), import(saved), import(stores), import(workspace),
      ]);
      if (!(await conversationLibrary.hasAny())) {
        await conversationLibrary.create({ name: "Chat 1", snapshot: captureConversationSnapshot() });
      }
      registerConversationAutosave(async () => {});
      const modelId = sessionState.info.model_id;
      document.body.replaceChildren();
      const target = document.createElement("div");
      document.body.append(target);
      mount(HostedHome, { target, props: {
        controller: { capabilities: () => ({ signals: { appleMobile: false } }), open: async () => {}, check: async () => {} },
        snapshot: { phase: "supported", models: [{ id: modelId, modelId, name: "Qwen3 1.7B", setupComplete: true, fit: "recommended" }], download: { available: true, phase: "idle", reason: "Ready" }, runtime: { available: true }, storage: { persisted: true } },
        onChooseModels() {},
      } });
    }, { home: `/@fs${resolve("src/hosted/ui/HostedHome.svelte")}`, saved: `/@fs${resolve("../drowse/webui/src/lib/stores/savedConversations.svelte.ts")}`, stores: `/@fs${resolve("../drowse/webui/src/lib/stores.svelte.ts")}`, workspace: `/@fs${resolve("../drowse/webui/src/lib/conversationWorkspace.ts")}` });
    await workbench(page);
    await page.getByRole("textbox", { name: /^Compose as / }).fill("Hello, I love marmots!");
    await page.getByRole("button", { name: /^(Send|Generate reply)$/ }).click();
    await expect(page.getByRole("status").filter({ hasText: "Response complete." })).toBeVisible();
    await waitForReply(page);
    await mountHome();
    const card = page.locator("[data-saved-conversation]").first();
    await expect(card).toBeVisible();
    await expect(card.locator("time .morph-source")).toHaveText("just now");
    await expect(card.locator(".chat-counts .morph-source")).toHaveText("2 messages · 1 Loom thread");
    await card.locator(".chat-name").click();
    const input = card.getByRole("textbox", { name: "Chat name" });
    await expect(input).toBeFocused();
    await input.fill("Discard this");
    await input.press("Escape");
    await expect(card.locator(".chat-name")).not.toContainText("Discard this");
    await card.locator(".chat-name").focus();
    await page.keyboard.press("Enter");
    await input.fill(" ");
    await expect(card.getByRole("button", { name: "Save name" })).toBeDisabled();
    await input.fill("Marmot notes");
    await input.press("Enter");
    await expect(card.locator(".chat-name .morph-source")).toHaveText("Marmot notes");
    await card.locator(".chat-name").click();
    await input.fill("Discard this too");
    await card.getByRole("button", { name: "Cancel", exact: true }).click();
    await expect(card.locator(".chat-name .morph-source")).toHaveText("Marmot notes");
    await card.locator(".chat-name").click();
    await input.fill(" ");
    await card.locator(".chat-counts").click();
    await expect(card.locator(".chat-name .morph-source")).toHaveText("Marmot notes");
    await card.locator(".chat-name").click();
    await input.fill("Marmot research");
    await card.locator(".chat-counts").click();
    await expect(card.locator(".chat-name .morph-source")).toHaveText("Marmot research");

    const timestamp = await card.locator("time").getAttribute("datetime");
    await expect(card.getByRole("button", { name: "Download backup of Marmot research", exact: true }).locator(".morph-source")).toHaveText("Download");
    await card.getByRole("button", { name: "More options for Marmot research", exact: true }).click();
    await expect(card.getByRole("menuitem", { name: "Delete", exact: true })).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(card.locator("time")).toHaveAttribute("datetime", timestamp!);

    for (const width of [320, 1000]) {
      await page.setViewportSize({ width, height: 900 });
      expect(await card.evaluate(el => el.scrollWidth <= el.clientWidth)).toBe(true);
    }
    await page.reload();
    await expect(page.locator(".shell")).toBeVisible();
    await mountHome();
    await expect(page.locator("[data-saved-conversation]").first().locator(".chat-name .morph-source")).toHaveText("Marmot research");
    const downloading = page.waitForEvent("download");
    await page.getByRole("button", { name: "Download backup of Marmot research", exact: true }).click();
    const download = await downloading;
    const text = await readStreamText(await download.createReadStream());
    const backup = JSON.parse(text);
    expect(backup.conversation.name).toBe("Marmot research");
    await page.getByLabel("Import chat backup files", { exact: true }).setInputFiles({ name: "backup.json", mimeType: "application/json", buffer: Buffer.from(text) });
    await expect(page.locator(".import-duplicate")).toContainText("Marmot research” is already imported.");
    await page.getByRole("button", { name: "Import as a copy", exact: true }).click();
    await expect(page.locator(".backup-result")).toContainText("Marmot research (copy)");
    const imported = await page.evaluate(async ({ url, id }) => {
      const { conversations } = await (await import(url)).conversationLibrary.list();
      return conversations.find((record: any) => record.name === "Marmot research (copy)" && record.id !== id);
    }, { url: `/@fs${resolve("../drowse/webui/src/lib/stores/savedConversations.svelte.ts")}`, id: backup.conversation.id });
    expectCopiedConversation(imported, backup.conversation);
  });
});
