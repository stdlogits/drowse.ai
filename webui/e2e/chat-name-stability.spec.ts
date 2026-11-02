import { expect, test, type Page } from "@playwright/test";
import { resolve } from "node:path";
import { waitForReply } from "./workbench-navigation";

const modules = {
  stores: `/@fs${resolve("../drowse/webui/src/lib/stores.svelte.ts")}`,
  saved: `/@fs${resolve("../drowse/webui/src/lib/stores/savedConversations.svelte.ts")}`,
  workspace: `/@fs${resolve("../drowse/webui/src/lib/conversationWorkspace.ts")}`,
  home: `/@fs${resolve("src/hosted/ui/HostedHome.svelte")}`,
};
async function seed(page: Page) {
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=base");
  await expect(page.getByRole("textbox", { name: "Editable completion buffer" })).toBeVisible();
  return page.evaluate(async modules => {
    const { conversationLibrary } = await import(modules.saved);
    if (!(await conversationLibrary.hasAny())) {
      const snapshot = (await import(modules.workspace)).captureConversationSnapshot();
      for (const [name, timestamp] of [["Newest", 5_000], ["Middle", 4_000], ["Oldest", 4_000]] as const) {
        await conversationLibrary.create({ name, snapshot, createdAt: timestamp, updatedAt: timestamp });
      }
    }
    return (await conversationLibrary.listSummaries()).conversations;
  }, modules);
}
async function mountHome(page: Page) {
  await page.evaluate(async modules => {
    const [{ default: HostedHome }, { mount }, { registerConversationAutosave }, { sessionState }] = await Promise.all([
      import(modules.home), import("/e2e/svelte-runtime.ts"), import(modules.saved), import(modules.stores),
    ]);
    registerConversationAutosave(async () => {});
    const modelId = sessionState.info.model_id;
    document.body.replaceChildren();
    const target = document.createElement("div");
    document.body.append(target);
    mount(HostedHome, { target, props: {
      controller: { capabilities: () => ({ signals: { appleMobile: false } }), open: async () => {}, check: async () => {} },
      snapshot: { phase: "supported", models: [{ id: modelId, modelId, name: "Pythia 70M", setupComplete: true, fit: "recommended" }], download: { available: true, phase: "idle", reason: "Ready" }, runtime: { available: true }, storage: { persisted: true } },
      onChooseModels() {},
    } });
  }, modules);
}
const order = (page: Page) => page.locator("[data-saved-conversation]").evaluateAll(cards => cards.map(card => card.getAttribute("data-saved-conversation")));

for (const [zone, instant, expected] of [
  ["America/New_York", "2026-09-07T07:05:00Z", "New Chat - Sep 7 - 3:05 EDT"],
  ["America/New_York", "2026-01-07T20:05:00Z", "New Chat - Jan 7 - 15:05 EST"],
  ["America/Los_Angeles", "2026-09-07T06:05:00Z", "New Chat - Sep 6 - 23:05 PDT"],
  ["Asia/Kolkata", "2026-09-07T07:05:00Z", "New Chat - Sep 7 - 12:35 GMT+5:30"],
]) {
  test.describe(`${zone} at ${instant}`, () => {
    test.use({ timezoneId: zone });
    test("default chat names use the browser clock and survive messages and reload", async ({ page }) => {
      await page.clock.setFixedTime(new Date(instant));
      await page.goto("http://127.0.0.1:4176/app?layoutFixture=base");
      await page.getByRole("textbox", { name: "Editable completion buffer" }).fill("My first message is not the title");
      await page.getByRole("button", { name: "Continue text", exact: true }).click();
      await expect.poll(() => page.evaluate(async url =>
        (await import(url)).currentLoomTreeSnapshot()?.nodes.some((node: { tokens?: unknown[] }) => node.tokens?.length), modules.stores)).toBe(true);
      await waitForReply(page);
      const original = await page.evaluate(async modules => {
        const { conversationLibrary, flushConversationAutosave } = await import(modules.saved);
        await flushConversationAutosave();
        return (await conversationLibrary.list()).conversations[0];
      }, modules);
      expect(original.name).toBe(expected);
      expect(original.createdAt).toBe(Date.parse(instant));
      await page.clock.setFixedTime(new Date(Date.parse(instant) + 86_400_000));
      const updated = await page.evaluate(async ({ modules, id }) => {
        const { conversationLibrary } = await import(modules.saved);
        const snapshot = (await conversationLibrary.get(id)).snapshot;
        snapshot.tree.nodes.find((node: { parent_id: string | null }) => node.parent_id !== null).text = "This message must not become the chat name";
        snapshot.tree.rev += 1;
        return conversationLibrary.autosave(snapshot, id);
      }, { modules, id: original.id });
      expect(updated.name).toBe(expected);
      await mountHome(page);
      await expect(page.locator(`[data-saved-conversation="${original.id}"] .chat-name .morph-source`)).toHaveText(expected);
      await page.reload();
      await expect(page.getByRole("textbox", { name: "Editable completion buffer" })).toBeVisible();
      await mountHome(page);
      await expect(page.locator(`[data-saved-conversation="${original.id}"] .chat-name .morph-source`)).toHaveText(expected);
    });
  });
}

test("renaming a chat on Your chats preserves order and activity time through reload", async ({ page }) => {
  const records = await seed(page);
  await mountHome(page);
  await expect(page.locator("[data-saved-conversation]")).toHaveCount(3);
  const before = await order(page);
  const target = records.at(-1);
  const card = page.locator(`[data-saved-conversation="${target.id}"]`);
  const timestamp = await card.locator("time").getAttribute("datetime");
  await card.locator(".chat-name").click();
  const input = card.getByRole("textbox", { name: "Chat name", exact: true });
  await input.fill("Alpha renamed");
  await input.press("Enter");
  await expect(card.locator(".chat-name")).toContainText("Alpha renamed");
  expect(await order(page)).toEqual(before);
  await expect(card.locator("time")).toHaveAttribute("datetime", timestamp!);
  await seed(page);
  await mountHome(page);
  await expect(page.locator("[data-saved-conversation]")).toHaveCount(3);
  expect(await order(page)).toEqual(before);
  await expect(card.locator(".chat-name")).toContainText("Alpha renamed");
});

test("renaming in the saved-chat drawer preserves position and activity time", async ({ page }) => {
  const records = await seed(page);
  const open = () => page.evaluate(async url => (await import(url)).openDrawer("load_conversation"), modules.stores);
  await open();
  await expect(page.locator("[data-saved-conversation]")).toHaveCount(3);
  const before = await order(page);
  const target = records.at(-1);
  const card = page.locator(`[data-saved-conversation="${target.id}"]`);
  const timestamp = await card.locator("time").getAttribute("datetime");
  await card.getByRole("button", { name: "Rename", exact: true }).click();
  const input = card.getByRole("textbox", { name: "Conversation name", exact: true });
  await input.fill("Alpha renamed");
  await input.press("Enter");
  await expect(card.locator(".card-copy strong")).toHaveText("Alpha renamed");
  expect(await order(page)).toEqual(before);
  await expect(card.locator("time")).toHaveAttribute("datetime", timestamp!);
  await page.evaluate(async url => (await import(url)).closeDrawer(), modules.stores);
  await expect(card).toHaveCount(0);
  await open();
  await expect(page.locator("[data-saved-conversation]")).toHaveCount(3);
  expect(await order(page)).toEqual(before);
});
