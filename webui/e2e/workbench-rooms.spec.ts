import { expect, test } from "@playwright/test";
import {
  composer, devUrl, openFixtureWorkbench, openLayoutWorkbench, openMenuItem, returnToChats, selectRoom, sendMessage,
  toolDialog,
} from "./workbench-navigation";

// drowse 6.0's rooms under the hosted shell: the header's tabs (the Menu on a phone), the model
// pill, and the address bar (rooms are hash routes, #/lens).

test.beforeEach(async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
});

test("room tabs, the model pill and the drowse mark move between rooms", async ({ page, isMobile }) => {
  test.skip(isMobile, "A phone keeps the rooms in the Menu (the next test).");
  await openLayoutWorkbench(page);
  const tabs = page.getByRole("navigation", { name: "Rooms", exact: true });
  // The fixture runtimes read no units, so the Units room is hidden.
  await expect(tabs.getByRole("button")).toHaveText(["Workbench", "Lens", "Probes", "Artifacts", "Model"]);
  for (const room of ["Lens", "Probes", "Artifacts", "Workbench"]) {
    await selectRoom(page, room);
    await expect(tabs.getByRole("button", { name: room, exact: true })).toHaveAttribute("aria-pressed", "true");
  }
  await page.getByRole("button", { name: /: model room$/ }).click();
  await expect(page.locator("main h1")).toHaveText("Model");
  await expect(page).toHaveURL(/#\/model$/);
  await page.getByRole("banner").getByRole("button", { name: "drowse", exact: true }).click();
  await expect(page.locator("main h1")).toHaveText("Workbench");
  await expect(composer(page)).toBeVisible();
});

test("a phone reaches every room from the Menu", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openLayoutWorkbench(page);
  await expect(page.getByRole("navigation", { name: "Rooms", exact: true })).toHaveCount(0);
  for (const room of ["Lens", "Artifacts", "Workbench"]) await selectRoom(page, room);
  await expect(composer(page)).toBeVisible();
});

test("a tool dialog has one close action and also closes from the scrim and Escape", async ({ page, isMobile }) => {
  await openLayoutWorkbench(page);
  const help = toolDialog(page, "Help and shortcuts");
  await openMenuItem(page, "Help and shortcuts");
  await expect(help).toBeVisible();
  const close = help.getByRole("button", { name: /^Close/ });
  await expect(close).toHaveCount(1);
  await close.click();
  await expect(help).toHaveCount(0);
  if (!isMobile) {
    // A phone's dialog covers the page, so only a wider screen leaves scrim to click.
    await openMenuItem(page, "Help and shortcuts");
    await expect(help).toBeVisible();
    await page.mouse.click(4, 400);
    await expect(help).toHaveCount(0);
  }
  await openMenuItem(page, "Help and shortcuts");
  await expect(help).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(help).toHaveCount(0);
});

test("the landing and workbench mirror without clipping in RTL", async ({ page }) => {
  const rtl = () => page.evaluate(() => document.documentElement.setAttribute("dir", "rtl"));
  const contained = () => expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  for (const width of [390, 1280]) {
    await page.setViewportSize({ width, height: 900 });
    await page.goto(devUrl);
    await rtl();
    await contained();
  }
  await openLayoutWorkbench(page);
  await rtl();
  const tabs = page.getByRole("navigation", { name: "Rooms", exact: true }).getByRole("button");
  const xs = await tabs.evaluateAll(buttons => buttons.map(button => button.getBoundingClientRect().x));
  expect(xs[0]).toBeGreaterThan(xs[1]);
  await sendMessage(page, "Mirror this message");
  await contained();
});

test("the workbench's tab title names the app, not the last shell view", async ({ page, browserName }) => {
  test.skip(browserName === "webkit", "The worker fixture needs OPFS, which headless WebKit rejects.");
  // The chooser's "Models · Drowse" and Continue's "Opening your chat… · Drowse" give way to the app.
  await openFixtureWorkbench(page);
  await expect(page).toHaveTitle("Drowse");
  await page.reload();
  await expect(composer(page)).toBeVisible();
  await expect(page).toHaveTitle("Drowse");
  await returnToChats(page);
  await expect(page).toHaveTitle("Your chats · Drowse");
  await page.getByRole("button", { name: /^Continue with / }).click();
  await expect(composer(page)).toBeVisible();
  await expect(page).toHaveTitle("Drowse");
});

test("Back and Forward move between rooms without leaving the workbench", async ({ page, browserName }) => {
  test.skip(browserName === "webkit", "The worker fixture needs OPFS, which headless WebKit rejects.");
  // Rooms are hash routes: a history step that keeps the path and query stays in the workbench.
  await openFixtureWorkbench(page);
  await selectRoom(page, "Lens");
  await page.goBack();
  await expect(page.locator("main h1")).toHaveText("Workbench");
  await expect(composer(page)).toBeVisible();
  await page.goForward();
  await expect(page.locator("main h1")).toHaveText("Lens");
});

test("Your chats from a room leaves the room behind, and Continue opens on the Workbench", async ({ page, browserName }) => {
  test.skip(browserName === "webkit", "The worker fixture needs OPFS, which headless WebKit rejects.");
  // The shell's views carry no room hash, so the next open starts where drowse starts.
  await openFixtureWorkbench(page);
  await selectRoom(page, "Lens");
  await expect(page).toHaveURL(/#\/lens$/);
  await returnToChats(page);
  expect(new URL(page.url()).hash).toBe("");
  await page.getByRole("button", { name: /^Continue with / }).click();
  await expect(composer(page)).toBeVisible();
  await expect(page.locator("main h1")).toHaveText("Workbench");
  expect(new URL(page.url()).hash).toBe("");
});

test("Back from the workbench's first room returns to the shell view before it", async ({ page, browserName }) => {
  test.skip(browserName === "webkit", "The worker fixture needs OPFS, which headless WebKit rejects.");
  await openFixtureWorkbench(page);
  await returnToChats(page);
  // Your chats > Models adds a history entry; opening the workbench from it replaces that one.
  await page.getByRole("navigation", { name: "Primary navigation" }).getByRole("link", { name: "Models", exact: true }).click();
  await page.getByRole("button", { name: "Open workbench", exact: true }).click();
  await expect(composer(page)).toBeVisible();
  await selectRoom(page, "Lens");
  await page.goBack();
  await expect(page.locator("main h1")).toHaveText("Workbench");
  await page.goBack();
  await expect(page.getByRole("heading", { name: "Your chats", exact: true })).toBeVisible();
  await expect(composer(page)).toHaveCount(0);
});
