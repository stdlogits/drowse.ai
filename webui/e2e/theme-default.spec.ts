import { expect, test, type Page } from "@playwright/test";
import { resolve } from "node:path";

// The theme is Ledger's: auto (the system's look, stored nowhere), or light or dark, stepped by the
// one theme button. theme-init.js applies a stored choice before first paint; createTheme owns it
// after. The browser's theme colour follows the page's paper.
const devUrl = "http://127.0.0.1:4176";
const paper = (page: Page) => page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue("--paper").trim().toLowerCase());
const themeColors = (page: Page) => page.locator('meta[name="theme-color"]').evaluateAll(metas => metas.map(meta => (meta as HTMLMetaElement).content.toLowerCase()));

test.beforeEach(async ({ page }) => {
  await page.emulateMedia({ colorScheme: "light", reducedMotion: "reduce" });
});

test("fresh pages follow the system look and store nothing", async ({ page }) => {
  for (const path of ["/", "/credits", "/app?layoutFixture=base"]) {
    await page.goto(`${devUrl}${path}`);
    await expect(page.locator("html")).not.toHaveAttribute("data-theme", /.+/);
    expect(await paper(page)).toBe("#ffffff");
    expect(await page.evaluate(() => localStorage.getItem("drowse.theme"))).toBeNull();
  }
  await page.emulateMedia({ colorScheme: "dark" });
  await expect.poll(() => paper(page)).toBe("#0b0d12");
  await expect(page.locator("html")).not.toHaveAttribute("data-theme", /.+/);
});

test("the theme button steps to the system's opposite, holds it through reloads, and steps back", async ({ page }) => {
  await page.goto(devUrl);
  await page.getByRole("button", { name: "Use the dark theme", exact: true }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  expect(await page.evaluate(() => localStorage.getItem("drowse.theme"))).toBe("dark");
  await expect.poll(() => themeColors(page)).toEqual(["#0b0d12", "#0b0d12"]);
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await expect.poll(() => themeColors(page)).toEqual(["#0b0d12", "#0b0d12"]);
  await page.goto(`${devUrl}/app?layoutFixture=base`);
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await page.emulateMedia({ colorScheme: "dark" });
  await page.emulateMedia({ colorScheme: "light" });
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");

  // Dark on a light system steps to the system's own look, then back to auto.
  await page.goto(devUrl);
  await page.getByRole("button", { name: "Use the light theme", exact: true }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await page.getByRole("button", { name: "Follow the system's theme", exact: true }).click();
  await expect(page.locator("html")).not.toHaveAttribute("data-theme", /.+/);
  expect(await page.evaluate(() => localStorage.getItem("drowse.theme"))).toBeNull();
  await expect.poll(() => themeColors(page)).toEqual(["#ffffff", "#ffffff"]);
});

test("the runtime follows the system when storage is unavailable", async ({ page }) => {
  await page.goto(`${devUrl}/outside-the-workbench`);
  const mode = await page.evaluate(async url => {
    delete document.documentElement.dataset.theme;
    Object.defineProperty(window, "localStorage", { configurable: true, get() { throw new Error("Storage unavailable"); } });
    const { startTheme, theme } = await import(url);
    startTheme();
    return { choice: theme.choice, mode: theme.mode };
  }, `/@fs${resolve("../drowse/webui/src/lib/theme.ts")}?bootstrap-test`);
  expect(mode).toEqual({ choice: "auto", mode: "light" });
  await expect(page.locator("html")).not.toHaveAttribute("data-theme", /.+/);
});
