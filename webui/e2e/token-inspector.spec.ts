import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import {
  composer, inspectToken, inspector, openLayoutWorkbench, sendMessage, settle, sourceUrl, storesUrl,
} from "./workbench-navigation";

const registryUrl = sourceUrl("lib/runtime/registry.ts");

// The 6.0 Inspector's Token view on a desktop: a selected token docks beside the conversation
// (this token, alternatives, the lens by depth, features, actions), follows the cursor and the
// selection, and leaves the conversation usable.

async function open(page: Page) {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 1440, height: 900 });
  await openLayoutWorkbench(page);
  await sendMessage(page, "Inspect the reply.");
}

test("a selected token docks in the Inspector, follows the cursor and selection, and leaves the conversation usable", async ({ page }) => {
  await open(page);
  const token = await inspectToken(page, "deterministic");
  await expect(token.getByRole("heading", { name: "deterministic" })).toBeVisible();
  await token.getByRole("button", { name: "Next token", exact: true }).click();
  await expect(token.getByRole("heading", { name: "local" })).toBeVisible();
  await page.getByRole("button", { name: "Inspect token This", exact: true }).click();
  await expect(token.getByRole("heading", { name: "This" })).toBeVisible();
  await composer(page).fill("Still typing while a token is open");
  await expect(composer(page)).toHaveValue("Still typing while a token is open");
  await inspector(page).getByRole("button", { name: "Close", exact: true }).click();
  await expect(token).toHaveCount(0);
  await expect(inspector(page).getByRole("region", { name: "Lens", exact: true })).toBeVisible();
});

test("a token opens from the keyboard and Escape closes it before anything else", async ({ page }) => {
  await open(page);
  const word = page.getByRole("button", { name: "Inspect token deterministic", exact: true });
  await word.focus();
  await page.keyboard.press("Enter");
  const token = inspector(page).getByRole("region", { name: "Token", exact: true });
  await expect(token.getByRole("heading", { name: "deterministic" })).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(token).toHaveCount(0);
  await expect(composer(page)).toBeVisible();
});

test("the token view passes automated WCAG checks", async ({ page }) => {
  await open(page);
  await inspectToken(page, "deterministic");
  await settle(page);
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21aa", "wcag22aa"])
    .include('[aria-label="Inspector"]')
    .analyze();
  expect(results.violations.map(({ id, nodes }) => ({ id, targets: nodes.map(node => node.target) }))).toEqual([]);
});

test("a steered reply's token reads steered or replayed without the steering", async ({ page }) => {
  // The Token view's Readings switch (6.0's captured/replayed boundary): a token generated under a
  // recipe offers Steered (the recipe as its hint) and Unsteered ("Replayed without the steering");
  // Unsteered asks the runtime to replay every family without the steering, and Steered asks again
  // with it. A reply sent without steering offers no switch.
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 1440, height: 900 });
  await openLayoutWorkbench(page, "instruments");
  await page.evaluate(async ({ storesUrl, registryUrl }) => {
    const stores = await import(storesUrl);
    const instruments = (await import(registryUrl)).getRuntimeClient().instruments;
    const original = instruments.tokenReadout.bind(instruments);
    (window as any).readouts = [];
    instruments.tokenReadout = (family: string, nodeId: string, index: number, options: { steered?: boolean }) => {
      (window as any).readouts.push({ family, steered: options?.steered ?? true });
      return original(family, nodeId, index, options);
    };
    stores.applyRecipe("0.5 fixture/calm.focused");
  }, { storesUrl, registryUrl });
  await sendMessage(page, "Read this reply steered and unsteered.");
  const reads = () => page.evaluate(() => (window as any).readouts as { family: string; steered: boolean }[]);

  const token = await inspectToken(page, "deterministic");
  const readings = token.getByRole("group", { name: "Readings", exact: true });
  await expect(readings.getByRole("button", { name: "Steered", exact: true })).toHaveAttribute("aria-pressed", "true");
  await expect(token.getByText("0.5 fixture/calm.focused").first()).toBeVisible();
  await expect.poll(async () => (await reads()).filter(read => read.steered).length).toBeGreaterThan(0);
  expect((await reads()).filter(read => !read.steered)).toEqual([]);

  await readings.getByRole("button", { name: "Unsteered", exact: true }).click();
  await expect(readings.getByRole("button", { name: "Unsteered", exact: true })).toHaveAttribute("aria-pressed", "true");
  await expect.poll(async () => [...new Set((await reads()).filter(read => !read.steered).map(read => read.family))].sort())
    .toEqual(["lens", "sae"]);
  await expect(token.getByRole("alert")).toHaveCount(0);

  await readings.getByRole("button", { name: "Steered", exact: true }).click();
  await expect(readings.getByRole("button", { name: "Steered", exact: true })).toHaveAttribute("aria-pressed", "true");

  await inspector(page).getByRole("button", { name: "Close", exact: true }).click();
  await page.evaluate(async url => (await import(url)).applyRecipe(""), storesUrl);
  await sendMessage(page, "Now without steering.");
  const plain = page.locator(".chat .msg:not(.user)").last().getByRole("button", { name: "Inspect token deterministic", exact: true });
  await plain.click();
  await expect(token.getByRole("heading", { name: "deterministic" })).toBeVisible();
  await expect(token.getByRole("group", { name: "Readings", exact: true })).toHaveCount(0);
  expect(errors).toEqual([]);
});
