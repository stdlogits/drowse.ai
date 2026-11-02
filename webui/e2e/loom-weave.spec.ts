import { expect, test, type Page } from "@playwright/test";
import { openLoom as openLoomView, sendMessage, storesUrl } from "./workbench-navigation";

const loomViews = (page: Page) => page.getByRole("group", { name: "Loom views", exact: true });

/** Opens the Loom page from the Thread's Map or Weave entry. */
async function openLoom(page: Page, view: "Map" | "Weave") {
  await openLoomView(page, view);
  await expect(loomViews(page).getByRole("button", { name: view, exact: true })).toHaveAttribute("aria-pressed", "true");
}

async function showView(page: Page, view: "Map" | "Weave") {
  await loomViews(page).getByRole("button", { name: view, exact: true }).click();
  await expect(loomViews(page).getByRole("button", { name: view, exact: true })).toHaveAttribute("aria-pressed", "true");
}

/** Generation started from the Loom has ended. */
async function loomIdle(page: Page) {
  await expect.poll(() => page.evaluate(async url => (await import(url)).genStatus.active, storesUrl)).toBe(false);
}

test("Weave creates alternatives, backtracks, extends exact tokens, and preserves originals", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=1");
  await expect(page.locator(".shell")).toBeVisible();
  await openLoom(page, "Map");
  await showView(page, "Weave");
  await page.getByRole("textbox", { name: "Starting text" }).fill("A path with several futures.");
  await showView(page, "Map");
  await showView(page, "Weave");
  await expect(page.getByRole("textbox", { name: "Starting text" })).toHaveValue("A path with several futures.");
  await page.getByRole("button", { name: "Generate 3", exact: true }).click();
  const options = page.locator("[data-weave-choice]");
  await expect(options).toHaveCount(3);
  const weave = page.locator(".weave");
  await loomIdle(page);
  const snapshot = () => page.evaluate(async url => {
    const { loomTree } = await import(url);
    return [...loomTree.nodes.values()].map((node: any) => ({
      id: node.id, parent: node.parent_id, text: node.text, tokens: node.raw_token_ids,
    }));
  }, storesUrl);
  const original = await snapshot();
  const sourceId = await options.first().getAttribute("data-weave-choice");
  await options.first().getByRole("button", { name: "Choose", exact: true }).click();
  await expect(options.first().getByRole("button", { name: "Chosen", exact: true })).toBeDisabled();
  await expect(options.first()).toHaveClass(/chosen/);
  await expect(options).toHaveCount(3);
  await options.first().getByRole("button", { name: "Continue", exact: true }).click();
  await expect(options).toHaveCount(4);
  await loomIdle(page);
  const extended = await snapshot();
  for (const old of original) expect(extended.find((node: any) => node.id === old.id)).toEqual(old);
  const source = original.find((node: any) => node.id === sourceId)!;
  const fork = extended.find((node: any) => !original.some((old: any) => old.id === node.id))!;
  expect(fork.parent).toBe(source.parent);
  expect(fork.tokens.slice(0, source.tokens.length)).toEqual(source.tokens);

  await options.first().getByRole("button", { name: "Explore next", exact: true }).click();
  await expect(options).toHaveCount(0);
  await expect(weave.getByRole("status").filter({ hasText: /^0 available$/ })).toBeVisible();
  await weave.getByRole("button", { name: "Alternatives", exact: true }).click();
  await page.getByRole("option", { name: "1", exact: true }).click();
  await weave.getByRole("button", { name: "Generate 1", exact: true }).click();
  await expect(options).toHaveCount(1);
  await loomIdle(page);
  const nextTurnId = await options.first().getAttribute("data-weave-choice");
  expect((await snapshot()).find((node: any) => node.id === nextTurnId)?.parent).toBe(sourceId);
  const beforeReroll = await snapshot();
  await page.evaluate(async ({ url, id }) => {
    const { loomRegenerateNode } = await import(url);
    await loomRegenerateNode(id);
  }, { url: storesUrl, id: nextTurnId });
  await expect(options).toHaveCount(2);
  await loomIdle(page);
  const afterReroll = await snapshot();
  for (const old of beforeReroll) expect(afterReroll.find((node: any) => node.id === old.id)).toEqual(old);
  await weave.getByRole("button", { name: "Back one point", exact: true }).click();
  await expect(options).toHaveCount(4);
  await weave.getByRole("button", { name: "Back one point", exact: true }).click();
  await expect(options).toHaveCount(1);
  await expect(options.first()).toContainText("A path with several futures.");
  await options.first().getByRole("button", { name: "Explore next", exact: true }).click();
  await expect(options).toHaveCount(4);
  await weave.getByRole("button", { name: "Alternatives", exact: true }).click();
  await page.getByRole("option", { name: "2", exact: true }).click();
  await weave.getByRole("button", { name: "Generate 2", exact: true }).click();
  await expect(options).toHaveCount(6);
  await loomIdle(page);
  await options.first().getByRole("button", { name: "Write another", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "Start a branch" });
  await expect(dialog).toBeVisible();
  await dialog.getByRole("textbox").fill("An authored possibility.\nWith whitespace intact.");
  await dialog.locator(".dw-button").click();
  await expect(options).toHaveCount(7);
  const afterWrite = await snapshot();
  for (const old of original) expect(afterWrite.find((node: any) => node.id === old.id)).toEqual(old);
  await expect(options.last()).toContainText("An authored possibility.");
  await options.first().getByRole("button", { name: "Star option 1", exact: true }).click();
  await expect(options.first().getByRole("button", { name: "Starred option 1", exact: true })).toHaveAttribute("aria-pressed", "true");
  await showView(page, "Map");
  await page.getByRole("button", { name: "Fit whole loom" }).click();
  await expect(page.locator(".tree-node-wrap:not([data-loom-shared])")).toHaveCount(10);
  await expect(page.locator("[data-loom-shared]")).toHaveCount(2);
  expect(errors).toEqual([]);
});

test("Weave opens at a chat reply's branch point without hiding its alternatives", async ({ page }) => {
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=1");
  await sendMessage(page, "Start in chat, then explore.");
  await openLoom(page, "Map");
  await expect(page.locator('.node[aria-current="true"]')).toBeVisible();
  await showView(page, "Weave");
  await expect(page.locator("[data-weave-choice]")).toHaveCount(1);
  await expect(page.locator("[data-weave-choice]").first()).toContainText("deterministic local Drowse");
  await expect(page.locator(".weave .branch-point")).toContainText("Start in chat, then explore.");
});
