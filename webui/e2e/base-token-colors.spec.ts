import { expect, test } from "@playwright/test";
import { devUrl, waitForReply } from "./workbench-navigation";

test("base completion colours match inspection, survive an unsaved edit, and follow scrolling", async ({ page }) => {
  await page.setViewportSize({ width: 1100, height: 898 });
  await page.goto(`${devUrl}/app?layoutFixture=base`);
  const editor = page.getByRole("textbox", { name: "Editable completion buffer" });
  const mirror = page.locator(".color-mirror");
  await editor.fill("A field note: ");
  await page.getByRole("button", { name: "Continue text", exact: true }).click();
  await expect(mirror.locator("span[style*='background-color']").first()).toBeAttached();
  await waitForReply(page);
  const text = await editor.inputValue();
  const colors = () => mirror.locator("span[style*='background-']").evaluateAll(elements => elements.map(element => element.getAttribute("style")));
  const editColors = await colors();
  expect(editColors.length).toBeGreaterThan(0);
  await page.getByRole("button", { name: "Inspect tokens", exact: true }).click();
  expect(await page.locator(".inspect .tok").evaluateAll(elements => elements.map(element => element.getAttribute("style")))).toEqual(editColors);
  await page.getByRole("button", { name: "Edit text", exact: true }).click();
  await editor.fill(`${text} Unsaved`);
  await expect(mirror).toHaveCount(1);
  expect(await colors()).toEqual(editColors);
  await expect(mirror.locator(".origin-draft")).toHaveText(" Unsaved");
  await expect(page.locator(".color-notice")).toHaveCount(0);
  await page.getByRole("button", { name: "Discard", exact: true }).click();
  await expect(editor).toHaveValue(text);
  expect(await colors()).toEqual(editColors);

  // Raw mode's Colors control (drowse PR 71): the reading tools open under the completion's head and
  // change what its tokens are coloured by.
  const colorsToggle = page.getByRole("button", { name: "Colors", exact: true });
  await expect(colorsToggle).toHaveAttribute("aria-expanded", "false");
  await colorsToggle.click();
  await expect(colorsToggle).toHaveAttribute("aria-expanded", "true");
  const picker = page.getByRole("button", { name: /^Color completion tokens by/ });
  const pick = async (option: string) => {
    await picker.click();
    await page.getByRole("option", { name: option, exact: true }).click();
    await expect(picker).toContainText(option);
  };
  // The fixture's entropy and surprisal tint alike, so the switch is told by the picker and by No color.
  await pick("Sampler entropy");
  await expect.poll(async () => (await colors()).length).toBe(editColors.length);
  await pick("No color");
  await expect(mirror.locator("span[style*='background-']")).toHaveCount(0);
  await pick("Token surprisal");
  await expect.poll(colors).toEqual(editColors);
  await colorsToggle.click();
  await expect(colorsToggle).toHaveAttribute("aria-expanded", "false");
  await expect(picker).toHaveCount(0);
  await editor.fill(Array.from({ length: 30 }, (_, index) => `Field note ${index + 1}: the marmots gathered beside the rocks.\n`).join(""));
  await page.getByRole("button", { name: "Continue text", exact: true }).click();
  await waitForReply(page);
  const expectAligned = async () => {
    await expect.poll(() => page.evaluate(() => {
      const editor = document.querySelector<HTMLTextAreaElement>("#completion-buffer")!;
      const mirror = document.querySelector<HTMLElement>(".color-mirror")!;
      const editorStyle = getComputedStyle(editor);
      const mirrorStyle = getComputedStyle(mirror);
      return {
        text: mirror.textContent === `${editor.value}\u200b`,
        scroll: Math.abs(mirror.scrollTop - editor.scrollTop) < 1,
        width: mirror.clientWidth === editor.clientWidth,
        font: mirrorStyle.font === editorStyle.font,
        padding: mirrorStyle.padding === editorStyle.padding,
      };
    })).toEqual({ text: true, scroll: true, width: true, font: true, padding: true });
  };
  await editor.focus();
  await page.keyboard.press("Control+Home");
  await expectAligned();
  await page.keyboard.press("Control+End");
  await expectAligned();
  for (const viewport of [{ width: 320, height: 780 }, { width: 1440, height: 900 }]) {
    await page.setViewportSize(viewport);
    await expectAligned();
  }
});
