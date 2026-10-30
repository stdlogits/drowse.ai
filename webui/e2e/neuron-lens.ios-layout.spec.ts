import { expect, test, type Page } from "@playwright/test";
import type { CatalogIndex, HeadRow, ReadInfo, Terms } from "../src/hosted/ui/neurons/data";

const dev = "http://127.0.0.1:4176";
// Details name tokens by vocab.json id.
const vocab = ["rain", " weather", "下雨", "雨水", "天气", " dry", " she", " her"];
const terms: Terms = {
  "下雨": ["to rain", "xià yǔ", "zh"],
  "雨水": ["rainwater", "yǔ shuǐ", "zh"],
  "天气": ["weather", "tiān qì", "zh"],
};
const header: ReadInfo[] = [{ col: "read_S", trust: "primary", twin: false }, { col: "write_R", trust: "primary", twin: false }];
const pole = (sign: 1 | -1) => ({
  pole: sign, score: sign > 0 ? 4 : 1, shared: [], measured: [], silent: false, dark: false, detector: null,
  label: sign > 0
    ? { fires: "rain words (下雨, 雨水)", fc: 0.8, writes: "weather words (天气)", wc: 0.6, note: "in·S names rain in two languages." }
    : { fires: "dry weather", fc: 0.3, writes: "weak", wc: 0.1, note: "" },
  act: { top: 3.2, max: 5, rate: 0.001, slices: [0.001, 0.002, 0, 0.0005], pos: 0.5 },
  peaks: [["rain", 3]],
  contexts: sign > 0 ? [["it started to ⟦rain⟧ again"], ["昨天⟦下雨⟧了", "It ⟦rained⟧ yesterday", "zh"]] : [],
  drivers: [[0, 2.5]],
  legible: { read_S: 0.01, read_R: 1, write_S: 1, write_R: 0.5 },
  reads: sign > 0 ? [[90, [0, 1, 2, 3]], [80, [1, 4, 0, 2]]] : [[10, [5]], [5, []]],
});
const index: CatalogIndex = {
  model: "Qwen3-8B", id: "qwen3-8b", layers: 36, neuronsPerLayer: 12288, headsPerLayer: 32, chunk: 256, docBlock: 2048, labelled: true, media: [],
  layerMeta: [{ layer: 8, band: "crossover", count: 1 }],
  neurons: [[8, 344, "rain words (下雨, 雨水)", "dry weather", 5, 0, 80, 30]],
};
const headRow: HeadRow = [20, 17, 4, "after 'with' and 'that'", "gendered tokens (she, her)", "copies she and her", 20, 30, 45, 2, 3.1, [0.26, 0.07, 0.11, 0.04, 0.34]];

async function serve(page: Page) {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.route("**/neurons-data/qwen3-8b/index.json", route => route.fulfill({ json: index }));
  await page.route("**/neurons-data/qwen3-8b/vocab.json", route => route.fulfill({ json: vocab }));
  await page.route("**/neurons-data/qwen3-8b/terms.json", route => route.fulfill({ json: terms }));
  await page.route("**/neurons-data/qwen3-8b/neurons/L08/c001.json", route => route.fulfill({ json: { layer: 8, reads: header, neurons: [{ n: 344, poles: [pole(1), pole(-1)] }] } }));
  await page.route("**/neurons-data/qwen3-8b/heads/index.json", route => route.fulfill({ json: { heads: [headRow] } }));
  await page.route("**/neurons-data/qwen3-8b/heads/L20.json", route => route.fulfill({ json: {
    layer: 20,
    reads: [{ col: "ov_R", trust: "primary", twin: false }, { col: "qk_S", trust: "primary", twin: false }],
    heads: [{
      h: 17, kv: 4, roles: 2, score: 3.1, dark: false, modes: [0.1, 0.1], pattern: headRow[11],
      label: { fires: headRow[3], fc: 0.2, attends: headRow[4], ac: 0.3, writes: headRow[5], wc: 0.45, note: "Copies pronouns." },
      act: { top: 23, max: 26, rate: 0.46, slices: [0.45, 0.42, 0.6, 0.42] },
      copy: { ovR: 0.48, ovS: 0.97, self: 1.3, qkOv: 0, qkCtx: 0 },
      reads: [[90, [6, 7]], [40, [6]]], outRz: [7], drivers: [[1, 3]], keys: [[6, 0.012]],
      peaks: [], sources: [[" she", 2]], contexts: [["⟪ she⟫ said that ⟦ her⟧ coat"]],
      legible: { ov_R: 0.5, ov_copy: 0.8, ov_S: 0.9, qk_S: 0.9 },
    }],
  } }));
}

test("the old address redirects, and a neuron shows Haiku's labels over their raw reads", async ({ page }) => {
  await serve(page);
  await page.goto(`${dev}/neurons?layer=8&neuron=344`);
  await expect(page).toHaveURL(/\/neuron-lens\?layer=8&neuron=344$/);
  await expect(page.getByRole("heading", { name: "Neuron Lens", level: 1 })).toBeVisible();

  const summary = page.locator(".summary");
  // Wide screens head the columns; phones head each cell instead.
  await expect(summary.getByText("Triggered by", { exact: true }).filter({ visible: true }).first()).toBeVisible();
  await expect(summary.getByText("Steers outputs", { exact: true }).filter({ visible: true }).first()).toBeVisible();
  await expect(summary.locator(".label").first()).toContainText("rain words");
  const inputs = summary.locator(".cell .raw").first();
  await expect(inputs.locator(".source")).toHaveText("in·S");
  await expect(inputs.locator(".tok")).toHaveCount(4);
});

test("Translate to English covers tokens, labels and examples, and the tooltip shows each original", async ({ page }) => {
  await serve(page);
  await page.goto(`${dev}/neuron-lens?layer=8&neuron=344`);
  const summary = page.locator(".summary");
  const inputs = summary.locator(".cell .raw").first();
  const rain = inputs.getByRole("button", { name: "下雨", exact: true });
  await rain.hover();
  const tip = page.getByRole("tooltip");
  await expect(tip).toContainText("xià yǔ");
  await expect(tip).toContainText("to rain");
  await expect(tip).toContainText("Chinese");
  await expect(rain).toHaveAttribute("aria-describedby", await tip.getAttribute("id") ?? "");
  await page.keyboard.press("Escape");
  await expect(tip).toBeHidden();

  const translate = page.getByRole("checkbox", { name: "Translate to English", exact: true });
  await page.getByText("Translate to English", { exact: true }).click();
  await expect(translate).toBeChecked();
  await expect(inputs.locator(".tok .morph-source")).toHaveText(["to rain", "rainwater"]);
  await expect(summary.locator(".label").first().locator(".term .morph-source")).toHaveText(["to rain", "rainwater"]);
  await expect(summary.locator(".label").first().getByRole("button", { name: "to rain (translated from 下雨)", exact: true })).toBeVisible();
  const example = page.locator(".pole .ctx.translated").first();
  await expect(example).toContainText("It rained yesterday");
  await expect(example.locator("mark.hit")).toHaveText("rained");
  await example.getByRole("button").focus();
  await expect(tip).toContainText("昨天⟦下雨⟧了");

  // The switch remembers its state for this viewer.
  await page.reload();
  await expect(translate).toBeChecked();
  await page.getByText("Translate to English", { exact: true }).click();
  await expect(translate).not.toBeChecked();
  await expect(inputs.getByRole("button", { name: "雨水", exact: true })).toBeVisible();
  await expect(page.locator(".pole .ctx").nth(1)).toContainText("昨天");
});

test("a head shows where it fires, what it attends to and what it writes", async ({ page }) => {
  await serve(page);
  await page.goto(`${dev}/neuron-lens?layer=20&head=17`);
  await expect(page.getByRole("button", { name: "Attention heads" })).toHaveAttribute("aria-pressed", "true");
  const summary = page.locator(".summary");
  for (const heading of ["Fires when", "Attends to", "Writes"]) await expect(summary.getByText(heading, { exact: true })).toBeVisible();
  await expect(summary).toContainText("gendered tokens (she, her)");
  await expect(page.locator(".detail-bar")).toContainText("Duplicate-token head");
  await expect(page.locator(".pattern .on dt")).toHaveText(["Duplicate"]);
  await expect(page.locator(".ctx mark.src")).toHaveText(" she");
  await expect(page.locator(".ctx mark.hit")).toHaveText(" her");
});
