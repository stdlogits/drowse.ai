import { expect, test, type Locator, type Page } from "@playwright/test";
import {
  chooseFixtureModel, composer, devUrl, inspectToken, openMenuItem, sendMessage, toolDialog,
} from "./workbench-navigation";

// Phone layouts of the hosted setup and the 6.0 workbench: what a person reaches with a thumb in
// portrait and landscape, and what a short (keyboard-up) visual viewport keeps on screen.

test.use({ hasTouch: true });
const portraitViewports = [
  { width: 320, height: 568 },
  { width: 375, height: 667 },
  { width: 390, height: 844 },
  { width: 430, height: 932 },
] as const;
const landscapeViewports = [
  { width: 667, height: 375 },
  { width: 844, height: 390 },
  { width: 932, height: 430 },
] as const;

test.beforeEach(async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
});

async function expectNoPageOverflow(page: Page): Promise<void> {
  await expect.poll(async () => page.evaluate(() => (
    document.documentElement.scrollWidth - document.documentElement.clientWidth <= 1 &&
    document.body.scrollWidth - document.documentElement.clientWidth <= 1
  ))).toBe(true);
}

async function expectNoHorizontalOverflow(locator: Locator): Promise<void> {
  await expect.poll(async () => locator.evaluate((element) =>
    element.scrollWidth - element.clientWidth <= 1
  )).toBe(true);
}

async function expectInsideVisualViewport(locator: Locator): Promise<void> {
  await expect.poll(async () => locator.evaluate((element) => {
    const box = element.getBoundingClientRect();
    const viewport = window.visualViewport;
    const left = viewport?.offsetLeft ?? 0;
    const top = viewport?.offsetTop ?? 0;
    const right = left + (viewport?.width ?? window.innerWidth);
    const bottom = top + (viewport?.height ?? window.innerHeight);
    return box.left >= left - 1 && box.right <= right + 1 && box.top >= top - 1 && box.bottom <= bottom + 1;
  })).toBe(true);
}

/** WCAG 2.2's minimum target (2.5.8): 24 CSS px. */
async function expectTarget(locator: Locator): Promise<void> {
  const box = (await locator.boundingBox())!;
  expect(Math.min(box.width, box.height)).toBeGreaterThanOrEqual(24);
}

async function openWorkbench(page: Page): Promise<void> {
  await page.goto(`${devUrl}/app?layoutFixture=1`);
  await expect(composer(page)).toBeVisible();
}

async function expectConversationUsable(page: Page): Promise<void> {
  const send = page.getByRole("button", { name: "Send", exact: true });
  await composer(page).focus();
  await composer(page).scrollIntoViewIfNeeded();
  await expectInsideVisualViewport(composer(page));
  await send.scrollIntoViewIfNeeded();
  await expectInsideVisualViewport(send);
  await expectTarget(send);
  // iOS zooms the page into a focused field under 16px.
  expect(await composer(page).evaluate(el => Number.parseFloat(getComputedStyle(el).fontSize))).toBeGreaterThanOrEqual(16);
  await expectNoPageOverflow(page);
}

test("hosted setup reflows across representative iPhone viewports", async ({ page }) => {
  for (const viewport of [...portraitViewports, ...landscapeViewports]) {
    await page.setViewportSize(viewport);
    await page.goto(`${devUrl}/app?layoutFixture=setup`);
    await chooseFixtureModel(page);
    const action = page.getByRole("button", { name: "Download and open", exact: true });
    await action.scrollIntoViewIfNeeded();
    await expectInsideVisualViewport(action);
    await expectTarget(action);
    await expectNoPageOverflow(page);
  }
});

test("conversation composer remains reachable in portrait and landscape", async ({ page }) => {
  const resizeErrors: string[] = [];
  page.on("pageerror", error => { if (error.message.includes("ResizeObserver")) resizeErrors.push(error.message); });
  await page.setViewportSize(portraitViewports[2]);
  await openWorkbench(page);
  await sendMessage(page, "A phone conversation.");
  for (const viewport of [...portraitViewports, ...landscapeViewports]) {
    await page.setViewportSize(viewport);
    await expectConversationUsable(page);
  }
  expect(resizeErrors).toEqual([]);
});

test("keyboard-short visual viewports keep the composer actions reachable", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openWorkbench(page);
  await composer(page).focus();
  for (const viewport of [{ width: 390, height: 430 }, { width: 320, height: 360 }]) {
    await page.setViewportSize(viewport);
    const send = page.getByRole("button", { name: "Send", exact: true });
    await send.scrollIntoViewIfNeeded();
    await expectInsideVisualViewport(composer(page));
    await expectInsideVisualViewport(send);
    await expectNoPageOverflow(page);
  }
});

async function savedChatsAndToken(page: Page) {
  await openWorkbench(page);
  await openMenuItem(page, "Saved chats");
  const library = toolDialog(page, "Saved chats");
  await expect(library).toBeVisible();
  await expectInsideVisualViewport(library);
  await expectNoHorizontalOverflow(library);
  await page.keyboard.press("Escape");
  await expect(library).toHaveCount(0);

  await sendMessage(page, "Inspect this reply on a phone.");
  const token = await inspectToken(page, "deterministic");
  await expect(token.getByRole("heading", { name: "deterministic" })).toBeVisible();
  await expectNoHorizontalOverflow(token);
  await expectNoPageOverflow(page);
}

test("saved chats and token details stay contained on a portrait phone", async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 568 });
  await savedChatsAndToken(page);
});

test("saved chats and token details stay contained on a landscape phone", async ({ page }) => {
  await page.setViewportSize({ width: 667, height: 375 });
  await savedChatsAndToken(page);
});

test("short screens take the phone layout, the log beside the dock from 480 px wide", async ({ page }) => {
  // drowse PR 71: a phone is a screen up to 720 px wide or 520 px tall (a phone on its side, or a
  // short desktop window). A short one puts the conversation log beside the dock from 480 px wide;
  // narrower, it keeps one column with the dock under the log.
  const geometry = () => page.evaluate(() => {
    const box = (selector: string) => document.querySelector(selector)!.getBoundingClientRect().toJSON() as DOMRect;
    return { phone: document.querySelector(".shell.phone") !== null, log: box(".chat .log"), dock: box(".chat .dock") };
  });
  await page.setViewportSize({ width: 667, height: 375 });
  await openWorkbench(page);
  await sendMessage(page, "A short screen.");
  for (const viewport of [{ width: 480, height: 320 }, { width: 568, height: 320 }, ...landscapeViewports, { width: 1280, height: 500 }]) {
    await page.setViewportSize(viewport);
    await expect.poll(async () => (await geometry()).phone, `${viewport.width}x${viewport.height}`).toBe(true);
    const { log, dock } = await geometry();
    expect(dock.left, `${viewport.width}x${viewport.height}: the dock sits beside the log`).toBeGreaterThanOrEqual(log.right);
    expect(log.height, `${viewport.width}x${viewport.height}: the log keeps room`).toBeGreaterThanOrEqual(140);
    expect(log.bottom).toBeLessThanOrEqual(viewport.height + 1);
    await expectConversationUsable(page);
  }
  await page.setViewportSize({ width: 320, height: 480 });
  await expect.poll(async () => (await geometry()).phone).toBe(true);
  const narrow = await geometry();
  expect(narrow.dock.top, "320x480 keeps one column").toBeGreaterThanOrEqual(narrow.log.bottom - 1);
  expect(narrow.log.height).toBeGreaterThanOrEqual(140);
  await page.setViewportSize({ width: 1280, height: 900 });
  await expect.poll(async () => (await geometry()).phone, "a tall desktop window is not a phone").toBe(false);
});

test("every Menu dialog stays operable at 320 px", async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 568 });
  await openWorkbench(page);
  await sendMessage(page, "A narrow conversation.");
  for (const [item, label] of [
    ["Save chat", "Save chat"], ["Saved chats", "Saved chats"], ["Download chat", "Download chat"],
    ["Transcript", "Transcript"], ["System prompt", "System prompt"], ["Sampling", "Sampling settings"],
    ["Help and shortcuts", "Help and shortcuts"], ["Feedback", "Feedback"],
  ]) {
    await openMenuItem(page, item);
    const dialog = toolDialog(page, label);
    await expect(dialog, item).toBeVisible();
    await expectInsideVisualViewport(dialog);
    await expectNoHorizontalOverflow(dialog);
    const close = dialog.getByRole("button", { name: /^Close/ }).first();
    await expectInsideVisualViewport(close);
    await expectTarget(close);
    await page.keyboard.press("Escape");
    await expect(dialog).toHaveCount(0);
  }
  await expectNoPageOverflow(page);
});

/** Weave three continuations from a long starting text at 390x844; returns the Weave view. */
async function weaveThreeChoices(page: Page): Promise<Locator> {
  await page.setViewportSize({ width: 390, height: 844 });
  await openWorkbench(page);
  const weaveEntry = page.getByRole("complementary", { name: "Thread", exact: true }).getByRole("button", { name: "Weave", exact: true });
  if (!await weaveEntry.isVisible()) await page.getByRole("button", { name: "Thread", exact: true }).click();
  await weaveEntry.click();
  const weave = page.locator(".weave");
  await weave.getByRole("textbox", { name: "Starting text" }).fill("Explore the possibilities. ".repeat(12));
  await weave.getByRole("button", { name: "Generate 3", exact: true }).click();
  await expect(weave.locator("[data-weave-choice]")).toHaveCount(3);
  return weave;
}

test("Weave keeps branch controls reachable on narrow phones and in landscape", async ({ page }) => {
  const weave = await weaveThreeChoices(page);
  for (const viewport of [{ width: 320, height: 568 }, { width: 390, height: 844 }, { width: 844, height: 390 }]) {
    await page.setViewportSize(viewport);
    await expectNoPageOverflow(page);
    const choose = weave.locator("[data-weave-choice]").first().getByRole("button", { name: /^Cho/ });
    await choose.scrollIntoViewIfNeeded();
    await expectInsideVisualViewport(choose);
    await expectTarget(choose);
  }
});

test("Weave scrolls only vertically on a phone", async ({ page }) => {
  // drowse bug (lanes/N3.md): on a phone LoomWeave's .weave scrolls (overflow-y: auto, so overflow-x
  // computes to auto) and its sections keep their desktop focus gutter (margin 0 -4px), so the view
  // pans sideways by 4 px at every phone width. Fix in drowse: overflow-x: hidden on the phone .weave.
  test.fixme(true, "the phone Weave pans sideways by its sections' 4 px focus gutter; see lanes/N3.md");
  const weave = await weaveThreeChoices(page);
  for (const viewport of [{ width: 320, height: 568 }, { width: 390, height: 844 }, { width: 844, height: 390 }]) {
    await page.setViewportSize(viewport);
    await expectNoHorizontalOverflow(weave);
  }
});
