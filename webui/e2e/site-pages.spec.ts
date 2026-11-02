import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { siteTitle } from "../scripts/site-metadata.mjs";
import { devUrl, setAppearance, settle } from "./workbench-navigation";

// The hosted site's own pages: Credits, share metadata and icons, and the landing's copy.

/** Wait two frames: layout that depends on a resize or a route change lands a frame later. */
async function layoutSettled(page: Page) {
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
}

async function expectWebGpuRequirement(page: Page) {
  const note = page.locator("main [data-webgpu]");
  await expect(note).toHaveAttribute("data-webgpu", /^(available|unavailable)$/);
  await expect(note).toHaveText(await note.getAttribute("data-webgpu") === "available"
    ? "WebGPU is available in this browser. Drowse checks your device's memory next."
    : "This browser can't run Drowse yet. Try a current version of Chrome, Edge or Safari 26 or later on a computer with a graphics processor.");
}

test("Credits lists the team and thanks, reflows, and links to ways to contribute", async ({ page, browserName }, testInfo) => {
  test.slow();
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  const nav = page.getByRole("navigation", { name: "Primary navigation" });
  await expect(nav.getByRole("link", { name: "Contribute", exact: true })).toHaveAttribute("href", "/contribute");
  await expect(page.getByRole("link", { name: /^(Source|View source)$/ })).toHaveCount(0);
  await nav.getByRole("link", { name: "Credits", exact: true }).click();
  await expect(page).toHaveURL(/\/credits$/);
  await expect(page).toHaveTitle("Credits · Drowse");
  await expect(page.getByRole("heading", { name: "Credits.", exact: true })).toBeVisible();
  await expect(page.locator(".team-member")).toHaveCount(2);
  await expect(page.getByRole("link", { name: "@_a9lim", exact: true })).toHaveAttribute("href", "https://x.com/_a9lim");
  await expect(page.getByRole("link", { name: "@treetowntree", exact: true })).toHaveAttribute("href", "https://x.com/treetowntree");
  await expect(page.getByRole("link", { name: "Website a9l.im", exact: true })).toHaveAttribute("href", "https://a9l.im");
  await expect(page.getByRole("link", { name: "Website logits.ml", exact: true })).toHaveAttribute("href", "https://logits.ml");
  await expect(page.getByRole("link", { name: "@transkatgirl", exact: true })).toHaveAttribute("href", "https://x.com/transkatgirl");
  await expect(page.getByRole("link", { name: "@motion_so", exact: true })).toHaveAttribute("href", "https://x.com/motion_so");
  await expect(page.locator(".thanks-list li").nth(1).getByRole("link", { name: "@voooooogel", exact: true })).toHaveAttribute("href", "https://x.com/voooooogel");
  await expect(page.locator('.team-member .verification-badge[data-status="blue"]')).toHaveCount(2);
  await expect(page.locator('.thanks-list a[href="https://x.com/transkatgirl"] .verification-badge')).toHaveAttribute("data-status", "blue");
  await expect(page.locator('.thanks-list a[href="https://x.com/motion_so"] .verification-badge')).toHaveAttribute("data-status", "gold");
  await expect.poll(() => page.locator(".portrait, .thanks-portrait").evaluateAll(
    images => images.every(image => image instanceof HTMLImageElement && image.complete && image.naturalWidth > 0),
  )).toBe(true);
  await expect(page.locator(".page-brand")).toHaveAttribute("href", "/");
  await expect(page.locator(".page-header").getByRole("link", { name: "Open Drowse", exact: true })).toHaveAttribute("href", "/app");
  if (browserName === "webkit") {
    // WebKit's default keyboard mode skips links.
    await page.getByRole("link", { name: "Skip to content" }).focus();
  } else {
    await page.keyboard.press("Tab");
  }
  await expect(page.getByRole("link", { name: "Skip to content" })).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page.locator("#credits-main")).toBeFocused();
  for (const theme of ["light", "dark"]) {
    await page.setViewportSize({ width: 1440, height: 1000 });
    await setAppearance(page, theme === "light" ? "Light" : "Dark");
    for (const width of [1440, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      await expect.poll(() => page.evaluate(() => [...document.querySelectorAll("body *")]
        .filter(element => !element.closest('nav[aria-label="Primary navigation"], .citation pre code'))
        .filter(element => element.getBoundingClientRect().right > innerWidth + 1)
        .map(element => `${element.tagName}.${element.className}: ${element.getBoundingClientRect().right}`))).toEqual([]);
      for (const link of await nav.getByRole("link").all()) {
        await link.scrollIntoViewIfNeeded();
        const box = (await link.boundingBox())!;
        expect(box.x).toBeGreaterThanOrEqual(0);
        expect(box.x + box.width).toBeLessThanOrEqual(width + 1);
      }
      await settle(page);
      expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
    }
  }
  await page.reload();
  await expect(page.getByRole("heading", { name: "Credits.", exact: true })).toBeVisible();
  await page.getByRole("link", { name: "Drowse home", exact: true }).click();
  await expect(page.getByRole("navigation", { name: "Primary navigation" })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  expect(errors).toEqual([]);
});

test("pages have static share metadata and the brand icons", async ({ page, request }) => {
  const response = await request.get("/");
  expect(response.status()).toBe(200);
  const html = await response.text();
  expect(html).toContain(`<title>${siteTitle}</title>`);
  expect(html).toContain(`property="og:title" content="${siteTitle}"`);
  expect(html).toContain(`name="twitter:title" content="${siteTitle}"`);
  expect(html).toContain('property="og:site_name" content="Drowse"');
  expect(html).toContain('name="twitter:card" content="summary_large_image"');
  expect(html).toContain('property="og:image:width" content="1200"');
  expect(html).not.toContain("__DROWSE_");
  const image = await request.get("/social/drowse.png");
  expect(image.ok()).toBe(true);
  expect(image.headers()["content-type"]).toContain("image/png");
  await page.goto("/");
  await expect(page).toHaveTitle(siteTitle);
  expect(await page.locator('link[rel="icon"]').evaluateAll(links => links.map(link => link.getAttribute("href")))).toEqual(["/favicon.ico", "/favicon.svg"]);
  await expect(page.locator('link[rel="apple-touch-icon"]')).toHaveAttribute("href", "/icons/apple-touch-icon.png");
  for (const path of ["/favicon.ico", "/favicon.svg", "/icons/apple-touch-icon.png"]) expect((await request.get(path)).ok()).toBe(true);
});

test("landing browser copy is consistent across platforms and action labels stay on one line", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.addInitScript(() => {
    const query = new URL(location.href).searchParams;
    const platform = query.get("platform") ?? "macOS";
    const version = query.get("version");
    Object.defineProperties(navigator, {
      userAgent: { configurable: true, value: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Version/26.0 Safari/605.1.15" },
      platform: { configurable: true, value: "MacIntel" },
      maxTouchPoints: { configurable: true, value: 0 },
      userAgentData: { configurable: true, value: version === null ? undefined : {
        platform, mobile: false, getHighEntropyValues: async () => ({ platformVersion: version }),
      } },
    });
  });
  await page.goto(devUrl);
  const contentStoreUrl = `/@fs/${resolve("src/hosted/runtime/contentStore.ts")}`;
  await page.evaluate(async url => {
    await (await import(url)).hasSelectedInstalledModel();
    await new Promise<void>((resolve, reject) => {
      const request = indexedDB.open("drowse-hosted-runtime", 5);
      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const db = request.result;
        const transaction = db.transaction(["installs", "settings"], "readwrite");
        transaction.objectStore("installs").put({ id: "landing-layout-fixture", kind: "model", objectHashes: [], installedAt: 1 });
        transaction.objectStore("settings").put({ key: "selected_model_variant_id", value: "landing-layout-fixture" });
        transaction.oncomplete = () => { db.close(); resolve(); };
        transaction.onerror = () => { db.close(); reject(transaction.error); };
      };
    });
  }, contentStoreUrl);
  for (const query of [
    "?platform=macOS&version=26.0.0",
    "?platform=macOS&version=15.6.0",
    "?platform=Windows&version=26.0.0",
    "?platform=Linux&version=26.0.0",
    "",
  ]) {
    await page.goto(`${devUrl}/${query}`);
    const recommendation = page.locator(".hero-action-row > span");
    await expectWebGpuRequirement(page);
    const action = page.locator(".hero-action-row").getByRole("link", { name: "Open Drowse", exact: true });
    await expect(action).toHaveAttribute("href", "/app");
    const returningAction = page.locator(".closing-action");
    await expect(returningAction.getByRole("link")).toHaveText("Open Drowse");
    await expect(returningAction.getByRole("link")).toHaveAttribute("href", "/app");
    await expect(returningAction.locator("p")).toHaveText("Choose your model and tools inside the app.");
    for (const width of [320, 390, 768, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      for (const element of [action, recommendation]) {
        if (element === action) {
          expect(await element.evaluate(element => {
            const range = document.createRange();
            range.selectNodeContents(element);
            return range.getClientRects().length;
          })).toBe(1);
        }
        const box = (await element.boundingBox())!;
        expect(box.x).toBeGreaterThanOrEqual(0);
        expect(box.x + box.width).toBeLessThanOrEqual(width);
      }
    }
  }
  await page.setViewportSize({ width: 320, height: 900 });
});

test("landing explains the workbench and separates chat and base model uses", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(devUrl);
  const features = page.locator(".capabilities");
  await expect(features.getByRole("heading", { level: 2 })).toHaveText("Test what shapes a reply.");
  await expect(features.locator("li")).toHaveCount(3);
  await expect(features).toContainText("predictions across layers");
  const models = page.locator(".models");
  await expect(models).not.toContainText("Best quality");
  await expect(models).not.toContainText("Larger");
  await expect(models).not.toContainText("GB");
  await expect(models.getByRole("region", { name: "Chat models", exact: true })).toContainText("Gemma 3 1B");
  await expect(models.getByRole("region", { name: "Chat models", exact: true })).toContainText("Gemma 3 4B");
  const base = models.getByRole("region", { name: "Base models", exact: true });
  await expect(base.locator(".model-row")).toHaveCount(4);
  await expect(base).toContainText("Continue raw text.");
  await expect(models).not.toContainText("M means million parameters");
  await expect(models).not.toContainText("These candidates have existing");
  await expect(models).not.toContainText("validation needed");
  await expect(base.getByRole("link")).toHaveCount(4);
  for (const link of await base.getByRole("link").all()) {
    await expect(link).toHaveAttribute("href", /^\/app\?choose=1&model=[^&]+$/);
  }
  await expect(page.locator(".closing-action").getByRole("link", { name: "Open Drowse", exact: true })).toHaveAttribute("href", "/app");
  await expect(page.locator(".closing-action p")).toHaveText("Choose your model and tools inside the app.");
  for (const theme of ["Light", "Dark"]) {
    await page.setViewportSize({ width: 1440, height: 900 });
    await setAppearance(page, theme);
    await expect(page.locator("html")).toHaveAttribute("data-theme", theme.toLowerCase());
    for (const width of [320, 390, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      await models.scrollIntoViewIfNeeded();
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      expect(await models.locator(".model-row").evaluateAll(rows => rows.every(row => row.scrollWidth <= row.clientWidth))).toBe(true);
    }
  }
});
