import { resolve } from "node:path";
import { expect, type Locator, type Page } from "@playwright/test";

// Shared steps for drowse 6.0's workbench under the hosted shell. The dashboard is drowse's
// (drowse/webui/src): a header with the rooms as tabs (in the Menu on a phone), the workbench
// (Thread | Conversation | Inspector), and tools that open as one dialog over the page.

export const devUrl = "http://127.0.0.1:4176";

/** A module of the shared dashboard, importable from a dev-server page. */
export const sourceUrl = (path: string) => `/@fs${resolve("../drowse/webui/src", path)}`;
export const storesUrl = sourceUrl("lib/stores.svelte.ts");

export const composer = (page: Page): Locator => page.getByRole("textbox", { name: /^Compose as / });

/** The chat's live region: it names lifecycle changes only (Chat.svelte's streamAnnouncement). */
export const chatStatus = (page: Page): Locator => page.locator('.chat > [role="status"]');

export async function chooseFixtureModel(page: Page): Promise<void> {
  await expect(page.locator(".model-grid > button").first()).toBeVisible();
  const available = page.locator('.model-grid > button:not(:disabled):not([aria-disabled="true"])');
  if (await page.locator('.model-grid > button[aria-pressed="true"]').count() === 0
    && await available.count() > 0) {
    await available.first().click();
  }
}

/** Opens the worker fixture runtime's chooser, installs the fixture model and waits for the
 *  workbench's composer. `query` is appended to `/app?fixture=1`. */
export async function openFixtureWorkbench(page: Page, query = ""): Promise<void> {
  await page.goto(`${devUrl}/app?fixture=1${query}`);
  await expect(page.getByRole("heading", { name: "Choose your first model" })).toBeVisible();
  await chooseFixtureModel(page);
  await page.getByRole("button", { name: "Download and open", exact: true }).click();
  await expect(composer(page)).toBeVisible({ timeout: 30_000 });
}

/** Waits until no reply is being written: the live region names an ending and Stop is gone (a
 *  held Stop slot is aria-hidden, so the role query skips it). */
export async function waitForReply(page: Page): Promise<void> {
  await expect(chatStatus(page))
    .toHaveText(/^(Response complete|Generation stopped|Generation ended|Completion finished|Token limit reached|Edit saved)\.$/);
  await expect(page.getByRole("button", { name: "Stop", exact: true })).toHaveCount(0);
}

/** Sends a chat message and waits for the reply to end. */
export async function sendMessage(page: Page, text: string): Promise<void> {
  const turns = page.locator(".chat .msg");
  const before = await turns.count();
  await composer(page).fill(text);
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(turns).toHaveCount(before + 2);
  await waitForReply(page);
}

/** Opens a room: the header's room tabs on a wide screen, the Menu's Rooms group on a phone. */
export async function selectRoom(page: Page, name: string): Promise<void> {
  const tabs = page.getByRole("navigation", { name: "Rooms", exact: true });
  if (await tabs.isVisible()) {
    await tabs.getByRole("button", { name, exact: true }).click();
  } else {
    await openMenuItem(page, name);
  }
  await expect(page.locator("main h1")).toHaveText(name);
}

/** Runs one item of the header's Menu (Save chat, Saved chats, Transcript, System prompt, …). */
export async function openMenuItem(page: Page, name: string): Promise<void> {
  const menu = page.getByRole("menu", { name: "Menu", exact: true });
  if (!await menu.isVisible()) await page.getByRole("button", { name: "Menu", exact: true }).click();
  await menu.getByRole("menuitem", { name, exact: true }).click();
}

/** Leaves the workbench for the hosted shell's Your chats (the Menu's Settings group). */
export async function returnToChats(page: Page): Promise<void> {
  await openMenuItem(page, "Your chats");
  await expect(page.getByRole("heading", { name: "Your chats", exact: true })).toBeVisible();
}

/** The one dialog tools open in, by its label (App.svelte's DIALOG_LABELS). */
export const toolDialog = (page: Page, name: string): Locator =>
  page.getByRole("dialog", { name, exact: true });

/** The Inspector: a column beside the page, or a sheet (a region) on a phone. */
export const inspector = (page: Page): Locator => page.getByRole("complementary", { name: "Inspector", exact: true })
  .or(page.getByRole("region", { name: "Inspector", exact: true }));

/** Selects a generated token and returns the Inspector's Token region. */
export async function inspectToken(page: Page, text: string): Promise<Locator> {
  await page.getByRole("button", { name: `Inspect token ${text}`, exact: true }).first().click();
  const token = inspector(page).getByRole("region", { name: "Token", exact: true });
  await expect(token).toBeVisible();
  return token;
}

/** Sets an explicit light or dark theme through Ledger's ThemeButton, which steps auto, the
 *  system's opposite, then its own look (theme-default.spec covers that order). Where no button
 *  is on screen (a phone's workbench keeps it in the header menu) the dev server's theme module
 *  sets it directly. */
export async function setAppearance(page: Page, name: string): Promise<void> {
  const choice = name.toLowerCase();
  const html = page.locator("html");
  const button = page.getByRole("button", { name: /^(Use the (light|dark) theme|Follow the system's theme)$/ }).first();
  for (let step = 0; step < 3 && await html.getAttribute("data-theme") !== choice; step += 1) {
    if (await button.isVisible()) await button.click();
    else await page.evaluate(async ({ url, choice }) => (await import(url)).setTheme(choice),
      { url: sourceUrl("lib/theme.ts"), choice });
  }
  await expect(html).toHaveAttribute("data-theme", choice);
  // Leave nothing transient behind: a focused button keeps its tip up.
  await page.evaluate(() => (document.activeElement as HTMLElement | null)?.blur());
  await page.mouse.move(0, 0);
  await expect(page.locator("#tip.shown")).toHaveCount(0);
}
