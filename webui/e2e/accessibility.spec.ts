import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import {
  chooseFixtureModel, composer, devUrl, openFixtureWorkbench, openMenuItem, selectRoom, sendMessage,
  setAppearance, settle, storesUrl, toolDialog,
} from "./workbench-navigation";

// Automated WCAG 2.2 AA checks (axe) over the hosted pages, the 6.0 workbench and its rooms, and
// every tool dialog, after animations settle.

test.beforeEach(async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
});

async function expectAccessible(page: Page, include?: string): Promise<void> {
  await settle(page);
  let builder = new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21aa", "wcag22aa"]);
  if (include) builder = builder.include(include);
  const results = await builder.analyze();
  expect(results.violations.map(({ id, impact, nodes }) => ({
    id, impact, nodes: nodes.map(node => ({ target: node.target, html: node.html, failureSummary: node.failureSummary })),
  }))).toEqual([]);
}

test("landing, onboarding, and workbench meet automated WCAG checks", async ({ page }) => {
  await page.goto(devUrl);
  await expect(page.getByRole("main").getByRole("link", { name: "Open Drowse", exact: true }).first()).toBeVisible();
  await expectAccessible(page);

  await page.goto(`${devUrl}/app?fixture=1`);
  await expect(page.getByRole("heading", { name: "Choose your first model" })).toBeVisible();
  await expectAccessible(page);

  await chooseFixtureModel(page);
  await page.getByRole("button", { name: "Download and open", exact: true }).click();
  await expect(composer(page)).toBeVisible();
  await expectAccessible(page);
  for (const theme of ["Dark", "Light"]) {
    await setAppearance(page, theme);
    await expectAccessible(page);
  }
  await setAppearance(page, "Dark");

  await page.goto(devUrl);
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await expectAccessible(page);
  await page.goto(`${devUrl}/app?fixture=1&choose=1`);
  await expect(page.getByRole("heading", { name: "Models" })).toBeVisible();
  await expectAccessible(page);
});

test("every room of the populated workbench meets automated WCAG checks", async ({ page }) => {
  await openFixtureWorkbench(page);
  await sendMessage(page, "Create inspectable context");
  for (const theme of ["Light", "Dark"]) {
    await setAppearance(page, theme);
    for (const room of ["Workbench", "Lens", "Probes", "Artifacts", "Model"]) {
      await selectRoom(page, room);
      await expectAccessible(page, "main");
    }
  }
});

test("every Menu dialog meets automated WCAG checks", async ({ page }) => {
  await openFixtureWorkbench(page);
  await sendMessage(page, "Create inspectable context");
  const dialogs: Array<[item: string, label: string]> = [
    ["Save chat", "Save chat"], ["Saved chats", "Saved chats"], ["Download chat", "Download chat"],
    ["Transcript", "Transcript"], ["System prompt", "System prompt"], ["Sampling", "Sampling settings"],
    ["Help and shortcuts", "Help and shortcuts"], ["Feedback", "Feedback"],
  ];
  for (const [item, label] of dialogs) {
    await openMenuItem(page, item);
    await expect(toolDialog(page, label), item).toBeVisible();
    await expect(toolDialog(page, label).getByText("checking…", { exact: true })).toHaveCount(0);
    await expectAccessible(page, '[role="dialog"]');
    await page.keyboard.press("Escape");
    await expect(toolDialog(page, label)).toHaveCount(0);
  }
  await page.getByRole("button", { name: "Search everything (⌘K)", exact: true }).click();
  await expect(page.getByRole("dialog", { name: "Search everything", exact: true })).toBeVisible();
  await expectAccessible(page, '[role="dialog"]');
});

test("context-launched tool dialogs meet automated WCAG checks", async ({ page }) => {
  await openFixtureWorkbench(page);
  await sendMessage(page, "Create inspectable context");
  const drawers: Array<{ name: string; params?: unknown }> = [
    { name: "subspace" }, { name: "manifolds" }, { name: "node_compare", params: { node_ids: [] } },
  ];
  for (const drawer of drawers) {
    await page.evaluate(async ({ moduleUrl, name, params }) => {
      (await import(moduleUrl)).openDrawer(name, params);
    }, { moduleUrl: storesUrl, ...drawer });
    const dialog = page.getByRole("dialog").last();
    await expect(dialog, drawer.name).toBeVisible();
    await expectAccessible(page, '[role="dialog"]');
    await page.keyboard.press("Escape");
  }
});

test("a populated saved-chat library meets automated WCAG checks", async ({ page }) => {
  await openFixtureWorkbench(page);
  await sendMessage(page, "A saved accessibility check");
  await openMenuItem(page, "Save chat");
  const save = toolDialog(page, "Save chat");
  await save.getByRole("textbox", { name: "Name" }).fill("Accessible saved chat");
  await expectAccessible(page, '[role="dialog"]');
  await save.getByRole("button", { name: /^(Save|Update)$/ }).click();
  await expect(save).toHaveCount(0);
  await openMenuItem(page, "Saved chats");
  const library = toolDialog(page, "Saved chats");
  await expect(library.getByText("Accessible saved chat", { exact: true })).toBeVisible();
  await expectAccessible(page, '[role="dialog"]');
  await library.getByRole("button", { name: "Delete", exact: true }).first().click();
  await expectAccessible(page, '[role="dialog"]');
});

test("closing a tool dialog returns focus to the control that opened it", async ({ page }) => {
  // drowse PR 71: focus goes back to the opener while it is still a tab stop, else to Menu. A Menu
  // item is gone once its dialog opens, so Menu takes it.
  await openFixtureWorkbench(page);
  await openMenuItem(page, "Feedback");
  await expect(toolDialog(page, "Feedback")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(toolDialog(page, "Feedback")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Menu", exact: true })).toBeFocused();
});
