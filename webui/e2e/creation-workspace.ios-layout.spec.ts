import { expect, test, type Locator, type Page } from "@playwright/test";
import { devUrl, selectRoom, sourceUrl } from "./workbench-navigation";

// The hosted builder: Artifacts › Build, generating examples and fitting them through the hosted
// controller (the runtime's generate and fit are replaced by a harness the test drives).

const registryUrl = sourceUrl("lib/runtime/registry.ts");

async function prepare(page: Page) {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(`${devUrl}/app?layoutFixture=instruments`);
  await expect(page.locator(".shell")).toBeVisible({ timeout: 30_000 });
  await page.evaluate(async url => {
    const registry = await import(url);
    registry.getRuntimeCapabilities().operations.fitting = { available: true, reasons: [] };
    const runtime = registry.getRuntimeClient();
    const harness: any = { requests: [], fits: [] };
    (window as any).creationHarness = harness;
    runtime.manifolds.generate = (request: unknown, emit: unknown) => {
      harness.requests.push(request);
      harness.emit = emit;
      return new Promise((resolve, reject) => { harness.finishGeneration = resolve; harness.reject = reject; });
    };
    runtime.manifolds.fit = (namespace: string, name: string, request: unknown, emit: unknown) => {
      harness.fits.push({ namespace, name, request });
      harness.emit = emit;
      return new Promise((resolve, reject) => { harness.finishFit = resolve; harness.reject = reject; });
    };
    registry.getHostedController().cancelFitting = async () => {
      harness.reject(Object.assign(new Error("Creation cancelled"), { code: "FITTING_CANCELLED" }));
    };
  }, registryUrl);
  await selectRoom(page, "Artifacts");
  await page.getByRole("group", { name: "Artifacts sections", exact: true }).getByRole("button", { name: "Build", exact: true }).click();
  await expect(builder(page)).toBeVisible();
}

const builder = (page: Page) => page.getByRole("region", { name: "Build manifold", exact: true });

async function fillConcepts(page: Page) {
  await builder(page).getByRole("textbox", { name: "Name", exact: true }).fill("pirate");
  await builder(page).getByRole("textbox", { name: /^Concepts \(at least 2\)/ }).fill("pirate, assistant");
}

/** A budget is a radio drawn under its label; a person clicks the label. */
async function pickBudget(panel: Locator, name: RegExp) {
  await panel.locator("label").filter({ has: panel.page().getByRole("radio", { name }) }).click();
  await expect(panel.getByRole("radio", { name })).toBeChecked();
}

async function emit(page: Page, data: unknown) {
  await page.evaluate(data => (window as any).creationHarness.emit({ event: "progress", data }), data);
}

test("budgets and mode changes keep the draft and reach the generation request", async ({ page }) => {
  await prepare(page);
  await fillConcepts(page);
  const panel = builder(page);
  await expect(panel.getByText("96 examples total", { exact: true })).toBeVisible();
  await pickBudget(panel, /^More examples/);
  await expect(panel.getByText("192 examples total", { exact: true })).toBeVisible();
  await pickBudget(panel, /^Thorough/);
  await expect(panel.getByText("384 examples total", { exact: true })).toBeVisible();
  await pickBudget(panel, /^More examples/);
  const modes = panel.getByRole("group", { name: "Authoring mode", exact: true });
  await modes.getByRole("button", { name: "Your examples", exact: true }).click();
  await modes.getByRole("button", { name: "Generate", exact: true }).click();
  await expect(panel.getByRole("textbox", { name: "Name", exact: true })).toHaveValue("pirate");
  await expect(panel.getByText("192 examples total", { exact: true })).toBeVisible();
  await panel.getByRole("button", { name: "Generate and fit", exact: true }).click();
  await expect.poll(() => page.evaluate(() => (window as any).creationHarness.requests.length)).toBe(1);
  expect(await page.evaluate(() => (window as any).creationHarness.requests[0].samples_per_prompt)).toBe(2);
  expect(await page.evaluate(() => (window as any).creationHarness.requests[0].fit_mode)).toBe("pca");
  await page.getByRole("button", { name: "Cancel", exact: true }).first().click();
});

test("generation progress survives leaving the room, shows a measured ETA, and chains into fitting", async ({ page }) => {
  await prepare(page);
  await fillConcepts(page);
  await page.clock.install();
  await pickBudget(builder(page), /^More examples/);
  await builder(page).getByRole("button", { name: "Generate and fit", exact: true }).click();
  await expect.poll(() => page.evaluate(() => (window as any).creationHarness.requests.length)).toBe(1);
  await emit(page, { message: 'Generating "pirate" response 1/96...' });
  const progress = page.getByRole("progressbar").first();
  await expect(progress).toHaveAttribute("aria-valuenow", "0");
  await expect(page.getByText("Estimating this step…", { exact: true }).first()).toBeVisible();
  await page.clock.runFor(3000);
  await emit(page, { message: 'Generating "pirate" response 4/96...' });
  await expect(page.getByText(/About .* left in this step/).first()).toBeVisible();
  await selectRoom(page, "Workbench");
  await expect(progress).toBeVisible();
  await expect(progress).toHaveAttribute("aria-valuetext", "3 of 192 completed in this step");
  await page.evaluate(() => (window as any).creationHarness.finishGeneration({}));
  await expect.poll(() => page.evaluate(() => (window as any).creationHarness.fits.length)).toBe(1);
  await emit(page, { stage: "capturing", completed: 24, total: 192 });
  await expect(page.getByText("Reading examples", { exact: true }).first()).toBeVisible();
  await expect(progress).toHaveAttribute("aria-valuetext", "24 of 192 completed in this step");
  await page.evaluate(() => (window as any).creationHarness.finishFit({}));
  await expect(page.getByText("Saved and ready to steer or probe.", { exact: true }).first()).toBeVisible();
});

test("failed fits retry saved examples and cancellation restores usable controls", async ({ page }) => {
  await prepare(page);
  await fillConcepts(page);
  await builder(page).getByRole("button", { name: "Generate and fit", exact: true }).click();
  await expect.poll(() => page.evaluate(() => (window as any).creationHarness.requests.length)).toBe(1);
  await page.evaluate(() => (window as any).creationHarness.finishGeneration({}));
  await expect.poll(() => page.evaluate(() => (window as any).creationHarness.fits.length)).toBe(1);
  await page.evaluate(() => (window as any).creationHarness.reject(new Error("Test fit failed")));
  await expect(page.getByRole("alert").filter({ hasText: "Examples were saved, but fitting failed" })).toBeVisible();
  await page.getByRole("button", { name: "Fit saved examples", exact: true }).click();
  await expect.poll(() => page.evaluate(() => (window as any).creationHarness.fits.length)).toBe(2);
  expect(await page.evaluate(() => (window as any).creationHarness.requests.length)).toBe(1);
  await page.getByRole("button", { name: "Cancel", exact: true }).first().click();
  await expect(page.getByText("Creation cancelled", { exact: true }).first()).toBeVisible();
  await expect(page.getByRole("button", { name: "Fit saved examples", exact: true })).toBeEnabled();
  await expect(builder(page).getByRole("textbox", { name: "Name", exact: true })).toBeEnabled();
});

test("authored validation waits for Build and focuses each blocking field", async ({ page }) => {
  await prepare(page);
  const panel = builder(page);
  await panel.getByRole("group", { name: "Authoring mode", exact: true }).getByRole("button", { name: "Your examples", exact: true }).click();
  const build = panel.getByRole("button", { name: "Build", exact: true });
  const name = panel.getByRole("textbox", { name: "Name", exact: true });
  await expect(build).toBeEnabled();
  await build.click();
  await expect(name).toBeFocused();

  await name.fill("friendly_scale");
  const high = panel.getByRole("spinbutton", { name: "x high value" });
  await high.fill("0");
  await build.click();
  await expect(high).toBeFocused();
  await expect(high).toHaveAttribute("aria-invalid", "true");
  await expect(panel.getByText("The high value must be greater than the low value.")).toBeVisible();

  await high.fill("1");
  await build.click();
  const addNode = panel.getByRole("button", { name: "Add node", exact: true });
  await expect(addNode).toBeFocused();
  for (let index = 0; index < 5; index += 1) await addNode.click();
  await build.click();
  const firstStatements = panel.getByRole("textbox", { name: "Statements for node node_1" });
  await expect(firstStatements).toBeFocused();
  await expect(firstStatements).toHaveAttribute("aria-invalid", "true");
  await expect(panel.getByText("Add at least one example statement.").first()).toBeVisible();
});

test("the hosted build offers only the linear fit", async ({ page }) => {
  await prepare(page);
  const panel = builder(page);
  await expect(panel.getByRole("region", { name: "Fit method" })).toContainText("curved layouts are not in the hosted release");
  await expect(panel.getByRole("radiogroup", { name: "Fit method" })).toHaveCount(0);
  await panel.getByRole("group", { name: "Authoring mode", exact: true }).getByRole("button", { name: "Your examples", exact: true }).click();
  await panel.getByRole("switch", { name: "Find the layout", exact: true }).click();
  await expect(panel.getByRole("radiogroup", { name: "Fit method" })).toHaveCount(0);
});
