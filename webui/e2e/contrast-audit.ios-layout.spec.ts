import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Locator, type Page, type TestInfo } from "@playwright/test";
import { resolve } from "node:path";
import { composer, openMenuItem, selectRoom, sendMessage, settle, toolDialog } from "./workbench-navigation";
import { waitForReply } from "./workbench-navigation";

const devUrl = "http://127.0.0.1:4176";
const moduleUrl = (path: string) => `/@fs${resolve(path)}`;

test.describe.configure({ mode: "parallel" });

const appEntryUrl = (page: Page) => page.locator('script[type="module"][src*="/main.ts"]')
  .evaluate(script => (script as HTMLScriptElement).src);

async function openPage(page: Page, path: string) {
  await page.goto(`${devUrl}${path}`);
  // Vite adds a timestamp after source edits; importing the bare path would mount a second app.
  await page.evaluate(async url => { await import(url); }, await appEntryUrl(page));
}

async function setTheme(page: Page, theme: string) {
  await page.evaluate(async ({ url, theme }) => (await import(url)).setTheme(theme), {
    url: moduleUrl("../drowse/webui/src/lib/theme.ts"), theme,
  });
}

async function measureAction(button: Locator) {
  const foreground = await button.evaluate(element => {
    const style = getComputedStyle(element);
    return { color: style.color, opacity: style.opacity, backdrop: getComputedStyle(element.parentElement!).backgroundColor };
  });
  const screenshot = await button.screenshot({
    scale: "css",
    style: "* { -webkit-text-fill-color: transparent !important; text-shadow: none !important; }",
  });
  await button.page().evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
  return button.page().evaluate(async ({ bytes, foreground }) => {
    const bitmap = await createImageBitmap(new Blob([new Uint8Array(bytes)], { type: "image/png" }));
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const context = canvas.getContext("2d")!;
    context.drawImage(bitmap, 0, 0);
    const backgrounds = [0.25, 0.5, 0.75].map(x => Array.from(context.getImageData(
      Math.floor(canvas.width * x), Math.floor(canvas.height / 2), 1, 1,
    ).data).slice(0, 3).map(c => c / 255));
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = foreground.color;
    context.fillRect(0, 0, 1, 1);
    let ink = Array.from(context.getImageData(0, 0, 1, 1).data).slice(0, 3).map(c => c / 255);
    if (Number(foreground.opacity) < 1) {
      context.clearRect(0, 0, 1, 1);
      context.fillStyle = foreground.backdrop;
      context.fillRect(0, 0, 1, 1);
      const backdrop = Array.from(context.getImageData(0, 0, 1, 1).data).slice(0, 3).map(c => c / 255);
      ink = ink.map((c, i) => c * Number(foreground.opacity) + backdrop[i] * (1 - Number(foreground.opacity)));
    }
    bitmap.close();
    const luminance = (rgb: number[]) => rgb.map(c => c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4)
      .reduce((sum, c, i) => sum + c * [0.2126, 0.7152, 0.0722][i], 0);
    const apcaLuminance = (rgb: number[]) => rgb.reduce((sum, c, i) => sum + c ** 2.4 * [0.2126729, 0.7151522, 0.072175][i], 0);
    const clamp = (y: number) => y < 0.022 ? y + (0.022 - y) ** 1.414 : y;
    return backgrounds.map(background => {
      const light = [luminance(ink), luminance(background)].sort((a, b) => b - a);
      const textY = clamp(apcaLuminance(ink));
      const bgY = clamp(apcaLuminance(background));
      const raw = bgY > textY ? (bgY ** 0.56 - textY ** 0.57) * 1.14 : (bgY ** 0.65 - textY ** 0.62) * 1.14;
      return {
        ...foreground, background: background.map(c => Math.round(c * 255)),
        wcag: (light[0] + 0.05) / (light[1] + 0.05),
        apca: Math.abs(Math.abs(raw) < 0.1 ? 0 : (raw > 0 ? raw - 0.027 : raw + 0.027) * 100),
      };
    });
  }, { bytes: [...screenshot], foreground });
}

async function expectActionContrast(button: Locator, label: string) {
  const samples = await measureAction(button);
  expect.soft(Math.min(...samples.map(s => s.wcag)), `${label}: ${JSON.stringify(samples)}`).toBeGreaterThanOrEqual(4.5);
  expect.soft(Math.min(...samples.map(s => s.apca)), `${label}: ${JSON.stringify(samples)}`).toBeGreaterThanOrEqual(60);
  return samples;
}

async function auditActionStates(page: Page, selector: string, label: string) {
  const candidates = await page.locator(selector).locator("button:visible, a:visible, summary:visible").evaluateAll(elements => {
    const seen = new Set<string>();
    return elements.flatMap((element, index) => {
      if (element.matches(":disabled, .skip-link") || element.closest("[inert]") || !element.textContent?.trim()
        || element.getAttribute("aria-label")?.startsWith("Inspect token")) return [];
      const style = getComputedStyle(element);
      const key = [element.tagName, element.className, style.color, style.backgroundColor, style.backgroundImage].join("|");
      if (seen.has(key)) return [];
      seen.add(key);
      element.setAttribute("data-contrast-control", String(index));
      return { index, text: element.textContent.trim().slice(0, 80) };
    });
  });
  const checked = [];
  for (const candidate of candidates) {
    const control = page.locator(`${selector} [data-contrast-control="${candidate.index}"]`);
    await control.scrollIntoViewIfNeeded();
    for (const state of ["rest", "hover"]) {
      if (state === "hover") {
        // A control inside a clipped scroller can sit outside the viewport; its rest state still counts.
        if (!await control.hover({ force: true, timeout: 2_000 }).then(() => true, () => false)) continue;
      } else await page.mouse.move(0, 0);
      const pair = await control.evaluate(element => {
        const style = getComputedStyle(element);
        let opacity = 1;
        for (let parent: Element | null = element; parent; parent = parent.parentElement) opacity *= Number(getComputedStyle(parent).opacity);
        const canvas = document.createElement("canvas");
        const context = canvas.getContext("2d")!;
        const rgba = (color: string) => {
          context.clearRect(0, 0, 1, 1);
          context.fillStyle = color;
          context.fillRect(0, 0, 1, 1);
          return Array.from(context.getImageData(0, 0, 1, 1).data).map(c => c / 255);
        };
        const ink = rgba(style.color), background = rgba(style.backgroundColor);
        if (background[3] < 0.99 || ink[3] < 0.99 || opacity < 0.99) return null;
        const luminance = (rgb: number[]) => rgb.slice(0, 3).map(c => c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4)
          .reduce((sum, c, i) => sum + c * [0.2126, 0.7152, 0.0722][i], 0);
        const light = [luminance(ink), luminance(background)].sort((a, b) => b - a);
        return { color: style.color, background: style.backgroundColor, ratio: (light[0] + 0.05) / (light[1] + 0.05) };
      });
      if (pair) {
        checked.push({ ...candidate, state, ...pair });
        if (pair.ratio < 4.5) await expectActionContrast(control, `${label} ${candidate.text} ${state}`);
      }
    }
    await control.evaluate(element => element.removeAttribute("data-contrast-control"));
  }
  await page.mouse.move(0, 0);
  return checked;
}

test.beforeEach(async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.addInitScript(() => {
    const storage = navigator.storage ?? {};
    Object.defineProperty(navigator, "storage", { configurable: true, value: storage });
    Object.defineProperty(storage, "persist", { configurable: true, value: async () => false });
    Object.defineProperty(storage, "persisted", { configurable: true, value: async () => false });
  });
});

async function auditSurface(page: Page, testInfo: TestInfo, theme: string, name: string, selector = "body") {
  await settle(page);
  await page.mouse.move(0, 0);
  const result = await new AxeBuilder({ page }).include(selector).withRules(["color-contrast"]).analyze();
  const summarize = ({ nodes }: any) => nodes.map((node: any) => ({ target: node.target, html: node.html, failure: node.failureSummary }));
  const violations = result.violations.flatMap(summarize);
  const controls = await auditActionStates(page, selector, `${theme} ${name}`);
  await testInfo.attach("contrast-surface-audit", {
    body: JSON.stringify({ name, violations, incomplete: result.incomplete.flatMap(summarize), controls }, null, 2),
    contentType: "application/json",
  });
  expect.soft(violations, name).toEqual([]);
  console.log(`${theme}: ${name} (${violations.length} contrast failures)`);
}

async function populatedWorkbench(page: Page, theme: string) {
  await openPage(page, "/app?layoutFixture=instruments");
  await expect(composer(page)).toBeVisible();
  await setTheme(page, theme);
  await sendMessage(page, "A contrast check with a populated conversation.");
}

for (const theme of ["dark", "light"]) {
  for (const path of ["/", "/credits", "/contact", "/missing-page", "/app?layoutFixture=setup"]) {
    test(`public text retains contrast in ${theme}: ${path}`, async ({ page }, testInfo) => {
      await openPage(page, path);
      await expect(page.locator("main")).toBeVisible();
      if (path.includes("layoutFixture=setup")) await expect(page.locator(".model-grid > button").first()).toBeVisible();
      await setTheme(page, theme);
      await page.evaluate(() => document.fonts.ready);
      for (const width of [390, 1440]) {
        await page.setViewportSize({ width, height: 900 });
        await auditSurface(page, testInfo, theme, `${path} ${width}`);
      }
    });
  }
  test(`workbench rooms and the Loom retain contrast in ${theme}`, async ({ page }, testInfo) => {
    test.setTimeout(120_000);
    await populatedWorkbench(page, theme);
    await page.setViewportSize({ width: 390, height: 900 });
    await auditSurface(page, testInfo, theme, "Workbench 390", "main");
    await page.setViewportSize({ width: 1440, height: 900 });
    for (const room of ["Lens", "Artifacts", "Workbench"]) {
      await selectRoom(page, room);
      await auditSurface(page, testInfo, theme, `${room} 1440`, "main");
    }
    await page.getByRole("complementary", { name: "Thread", exact: true }).getByRole("button", { name: "Map", exact: true }).click();
    await expect(page.getByRole("region", { name: "Loom", exact: true })).toBeVisible();
    await auditSurface(page, testInfo, theme, "Loom map", "main");
  });
  test(`Menu dialogs retain contrast in ${theme}`, async ({ page }, testInfo) => {
    test.setTimeout(120_000);
    await page.setViewportSize({ width: 1440, height: 900 });
    await populatedWorkbench(page, theme);
    for (const [item, label] of [["Saved chats", "Saved chats"], ["Transcript", "Transcript"], ["System prompt", "System prompt"],
      ["Sampling", "Sampling settings"], ["Help and shortcuts", "Help and shortcuts"], ["Feedback", "Feedback"]]) {
      await openMenuItem(page, item);
      await expect(toolDialog(page, label)).toBeVisible();
      await auditSurface(page, testInfo, theme, label, '[role="dialog"]');
      await page.keyboard.press("Escape");
    }
  });
  test(`base completion text retains contrast in ${theme}`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await openPage(page, "/app?layoutFixture=base");
    await expect(page.locator(".shell")).toBeVisible();
    await setTheme(page, theme);
    await auditSurface(page, testInfo, theme, "Base completion");
  });
}
