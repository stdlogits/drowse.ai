import { expect, test, type Page } from "@playwright/test";
import { composer, devUrl, sendMessage } from "./workbench-navigation";

// drowse 6.0's phone Inspector: a sheet that rises when a token (or unit) is selected, with a grab
// handle that expands and collapses it, over a workspace that stays usable behind it.

const sheet = (page: Page) => page.getByRole("region", { name: "Inspector", exact: true });
const handle = (page: Page) => sheet(page).getByRole("button", { name: /^(Expand|Collapse) the inspector$/ });
const sheetHeight = (page: Page) => sheet(page).evaluate(el => el.getBoundingClientRect().height);

test.beforeEach(async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(`${devUrl}/app?layoutFixture=1`);
  await expect(composer(page)).toBeVisible();
  await sendMessage(page, "Show the phone inspector.");
});

test("a token raises the sheet; the handle expands and collapses it; Escape and Close dismiss it", async ({ page }) => {
  await expect(sheet(page)).toHaveCount(0);
  await page.getByRole("button", { name: "Inspect token deterministic", exact: true }).click();
  await expect(sheet(page).getByRole("heading", { name: "deterministic" })).toBeVisible();
  await expect(handle(page)).toHaveAccessibleName("Expand the inspector");
  const peek = await sheetHeight(page);
  expect(peek).toBeLessThan(844);
  await handle(page).click();
  await expect(handle(page)).toHaveAccessibleName("Collapse the inspector");
  await expect(handle(page)).toHaveAttribute("aria-expanded", "true");
  await expect.poll(() => sheetHeight(page)).toBeGreaterThan(peek);
  await handle(page).click();
  await expect(handle(page)).toHaveAccessibleName("Expand the inspector");
  await expect.poll(() => sheetHeight(page)).toBeLessThanOrEqual(peek + 1);

  await page.keyboard.press("Escape");
  await expect(sheet(page)).toHaveCount(0);
  await page.getByRole("button", { name: "Inspect token deterministic", exact: true }).click();
  await expect(sheet(page)).toBeVisible();
  await sheet(page).getByRole("button", { name: "Close", exact: true }).click();
  await expect(sheet(page)).toHaveCount(0);
  await expect(composer(page)).toBeVisible();
});

test("the handle works from the keyboard and the token cursor steps through the reply", async ({ page }) => {
  await page.getByRole("button", { name: "Inspect token deterministic", exact: true }).click();
  await handle(page).focus();
  await page.keyboard.press("Enter");
  await expect(handle(page)).toHaveAccessibleName("Collapse the inspector");
  await page.keyboard.press("Space");
  await expect(handle(page)).toHaveAccessibleName("Expand the inspector");
  await sheet(page).getByRole("button", { name: "Next token", exact: true }).click();
  await expect(sheet(page).getByRole("heading", { name: "local" })).toBeVisible();
  await sheet(page).getByRole("button", { name: "Previous token", exact: true }).click();
  await expect(sheet(page).getByRole("heading", { name: "deterministic" })).toBeVisible();
});

test("tokens behind the collapsed sheet stay selectable and the page does not scroll sideways", async ({ page }) => {
  await page.getByRole("button", { name: "Inspect token deterministic", exact: true }).click();
  await expect(sheet(page)).toBeVisible();
  const other = page.getByRole("button", { name: "Inspect token This", exact: true });
  await other.scrollIntoViewIfNeeded();
  await other.click();
  await expect(sheet(page).getByRole("heading", { name: "This" })).toBeVisible();
  for (const width of [320, 390]) {
    await page.setViewportSize({ width, height: 700 });
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  }
});
