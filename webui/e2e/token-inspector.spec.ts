import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { composer, inspectToken, inspector, openFixtureWorkbench, sendMessage, settle } from "./workbench-navigation";

// The 6.0 Inspector's Token view on a desktop: a selected token docks beside the conversation
// (this token, alternatives, the lens by depth, features, actions), follows the cursor and the
// selection, and leaves the conversation usable.

async function open(page: Page) {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 1440, height: 900 });
  await openFixtureWorkbench(page);
  await sendMessage(page, "Inspect the reply.");
}

test("a selected token docks in the Inspector, follows the cursor and selection, and leaves the conversation usable", async ({ page }) => {
  await open(page);
  const token = await inspectToken(page, "deterministic");
  await expect(token.getByRole("heading", { name: "deterministic" })).toBeVisible();
  await expect(token.getByRole("list", { name: "Alternatives at this position" })).toBeVisible();
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
