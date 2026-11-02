import { expect, test } from "@playwright/test";
import {
  composer, devUrl, openFixtureWorkbench, openMenuItem, selectRoom, sendMessage, toolDialog,
} from "./workbench-navigation";

// drowse 6.0's rooms under the hosted shell: the header's tabs (the Menu on a phone), the model
// pill, and the address bar (rooms are hash routes, #/lens).

test.beforeEach(async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
});

test("room tabs, the model pill and the drowse mark move between rooms", async ({ page }) => {
  await openFixtureWorkbench(page);
  const tabs = page.getByRole("navigation", { name: "Rooms", exact: true });
  // The fixture runtime reads no units, so the Units room is hidden.
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
  await openFixtureWorkbench(page);
  await expect(page.getByRole("navigation", { name: "Rooms", exact: true })).toHaveCount(0);
  for (const room of ["Lens", "Artifacts", "Workbench"]) await selectRoom(page, room);
  await expect(composer(page)).toBeVisible();
});

test("a tool dialog has one close action and also closes from the scrim and Escape", async ({ page }) => {
  await openFixtureWorkbench(page);
  const help = toolDialog(page, "Help and shortcuts");
  await openMenuItem(page, "Help and shortcuts");
  await expect(help).toBeVisible();
  const close = help.getByRole("button", { name: /^Close/ });
  await expect(close).toHaveCount(1);
  await close.click();
  await expect(help).toHaveCount(0);
  await openMenuItem(page, "Help and shortcuts");
  await expect(help).toBeVisible();
  await page.mouse.click(4, 400);
  await expect(help).toHaveCount(0);
  await openMenuItem(page, "Help and shortcuts");
  await expect(help).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(help).toHaveCount(0);
});

test("the landing, chooser and workbench mirror without clipping in RTL", async ({ page }) => {
  const rtl = () => page.evaluate(() => document.documentElement.setAttribute("dir", "rtl"));
  const contained = () => expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  for (const width of [390, 1280]) {
    await page.setViewportSize({ width, height: 900 });
    await page.goto(devUrl);
    await rtl();
    await contained();
  }
  await openFixtureWorkbench(page);
  await rtl();
  const tabs = page.getByRole("navigation", { name: "Rooms", exact: true }).getByRole("button");
  const xs = await tabs.evaluateAll(buttons => buttons.map(button => button.getBoundingClientRect().x));
  expect(xs[0]).toBeGreaterThan(xs[1]);
  await sendMessage(page, "Mirror this message");
  await contained();
});

test("the workbench's tab title names the app, not the last shell view", async ({ page }) => {
  // Product bug (lanes/A5.md, "workbench tab title"): the title stays "Models · Drowse" after the
  // chooser opens the workbench, and "Opening your chat… · Drowse" after Continue.
  test.fixme(true, "the workbench keeps the hosted shell's last title; see lanes/A5.md");
  await openFixtureWorkbench(page);
  await expect(page).toHaveTitle("Drowse");
  await page.reload();
  await expect(composer(page)).toBeVisible();
  await expect(page).toHaveTitle("Drowse");
});

test("Back and Forward move between rooms without leaving the workbench", async ({ page }) => {
  // Product bug (lanes/A5.md, "Back leaves the workbench"): HostedRoot's popstate handler returns
  // to Your chats on any history step while the workbench is open, including a room's hash step.
  test.fixme(true, "HostedRoot treats a room's hash popstate as leaving the app; see lanes/A5.md");
  await openFixtureWorkbench(page);
  await selectRoom(page, "Lens");
  await page.goBack();
  await expect(page.locator("main h1")).toHaveText("Workbench");
  await expect(composer(page)).toBeVisible();
  await page.goForward();
  await expect(page.locator("main h1")).toHaveText("Lens");
});
