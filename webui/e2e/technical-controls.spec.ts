import { expect, test, type Page } from "@playwright/test";
import { composer, devUrl, inspector, sourceUrl, storesUrl } from "./workbench-navigation";

// Sampling controls under the hosted runtime: the Inspector's Sampling region and the Sampling
// settings dialog, bounded by the hosted model's context.

const registryUrl = sourceUrl("lib/runtime/registry.ts");

async function open(page: Page) {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(`${devUrl}/app?layoutFixture=instruments`);
  await expect(composer(page)).toBeVisible();
}

/** The Inspector's Sampling section; a phone opens the sheet from the composer's sampling line. */
async function samplingSection(page: Page) {
  const section = inspector(page).getByRole("region", { name: "Sampling", exact: true });
  if (!await section.isVisible()) await page.getByRole("button", { name: /^temperature / }).click();
  await expect(section).toBeVisible();
  return section;
}

test("max tokens follows the model limit and clamps oversized values", async ({ page }) => {
  await open(page);
  async function setModelLimit(limit: number) {
    await page.evaluate(async ({ storesUrl, registryUrl, limit }) => {
      const registry = await import(registryUrl);
      registry.getHostedController().snapshot.contextTokens = limit;
      const stores = await import(storesUrl);
      stores.sessionState.info = { ...stores.sessionState.info };
      stores.hydrateSamplingFromInfo();
    }, { storesUrl, registryUrl, limit });
  }
  await setModelLimit(2048);
  const slider = (await samplingSection(page)).getByRole("slider", { name: "Max tokens", exact: true });
  await expect(slider).toHaveAttribute("max", "2048");
  await slider.focus();
  await slider.press("End");
  await expect(slider).toHaveValue("2048");
  await setModelLimit(1024);
  await expect(slider).toHaveAttribute("max", "1024");
  await expect(slider).toHaveValue("1024");
  expect(await page.evaluate(async url => {
    const stores = await import(url);
    stores.setSampling("max_tokens", 999999);
    const stored = stores.samplingState.max_tokens;
    stores.samplingState.max_tokens = 999999;
    const sent = stores.buildSamplingPayload().max_tokens;
    return { stored, sent };
  }, storesUrl)).toEqual({ stored: 1024, sent: 1024 });
});

test("sampling settings keep large supported values on desktop and phone layouts", async ({ page }) => {
  await open(page);
  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 900 });
    await page.evaluate(async url => (await import(url)).openDrawer("advanced_sampling"), storesUrl);
    const dialog = page.getByRole("dialog", { name: "Sampling settings", exact: true });
    await expect(dialog).toBeVisible();
    for (const [label, value] of [["Top-k", "262144"], ["Temperature value", "3"], ["Seed", "9007199254740991"]]) {
      const input = dialog.getByRole("spinbutton", { name: label, exact: true });
      await input.fill(value);
      await input.press("Tab");
      await expect(input).toHaveValue(value);
    }
    expect(await page.evaluate(async url => {
      const { samplingState } = await import(url);
      return { top_k: samplingState.top_k, temperature: samplingState.temperature, seed: samplingState.seed };
    }, storesUrl)).toEqual({ top_k: 262144, temperature: 3, seed: Number.MAX_SAFE_INTEGER });
    expect(await dialog.evaluate(el => el.scrollWidth <= el.clientWidth + 1)).toBe(true);
    await page.keyboard.press("Escape");
    await expect(dialog).toHaveCount(0);
  }
});

test("Thinking shows only for models that support it and follows model changes", async ({ page }) => {
  await open(page);
  const section = await samplingSection(page);
  // A forced switch sits in an inert row, which the role query would otherwise skip.
  const thinking = section.getByRole("switch", { name: /^Thinking/, includeHidden: true });
  // A model change: the in-page runtime's session (what a settings patch answers with) and the
  // dashboard's copy of it.
  const setModel = (supported: boolean, optional: boolean) => page.evaluate(async ({ storesUrl, registryUrl, supported, optional }) => {
    const flags = { supports_thinking: supported, thinking_is_optional: optional };
    const runtime = (await import(registryUrl)).getRuntimeClient() as { session: object };
    runtime.session = { ...runtime.session, ...flags };
    const { sessionState } = await import(storesUrl);
    sessionState.info = { ...sessionState.info, ...flags };
  }, { storesUrl, registryUrl, supported, optional });
  const inert = () => thinking.evaluate(element => element.closest("[inert]") !== null);

  await setModel(false, false);
  await expect(thinking).toHaveCount(0);

  // Optional: a plain switch the person turns on and off.
  await setModel(true, true);
  await expect(thinking).toHaveAccessibleName("Thinking");
  expect(await inert()).toBe(false);
  const before = await thinking.getAttribute("aria-checked");
  await thinking.click();
  await expect(thinking).toHaveAttribute("aria-checked", before === "true" ? "false" : "true");
  await thinking.click();
  await expect(thinking).toHaveAttribute("aria-checked", before ?? "false");

  // Mandatory: shown on, said to be always on, and out of reach.
  await setModel(true, false);
  await expect(thinking).toHaveAccessibleName("Thinking, always on");
  await expect(thinking).toHaveAttribute("aria-checked", "true");
  expect(await inert()).toBe(true);

  await setModel(false, false);
  await expect(thinking).toHaveCount(0);
});
