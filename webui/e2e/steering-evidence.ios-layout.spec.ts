import { expect, test, type Locator, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { composer, devUrl, sendMessage, sourceUrl, storesUrl, waitForReply } from "./workbench-navigation";

// Steering in 6.0 lives in the composer's chips ("+ Steer"); each reply records the recipe it ran
// with, and "Compare unsteered" replays the turn without it.

const registryUrl = sourceUrl("lib/runtime/registry.ts");

async function open(page: Page) {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(`${devUrl}/app?layoutFixture=instruments`);
  await expect(composer(page)).toBeVisible();
}

/** The reply's steering marker: "steered" (its tip is the recipe) or "unsteered". */
const marker = (turn: Locator) => turn.locator(".dw-under span").filter({ hasText: /^(steered|unsteered)$/ });

const submits = (page: Page) => page.evaluate(() =>
  (window as any).steeringRequests.filter((request: any) => request.type === "submit"));

test("+ Steer opens with Enter and Space", async ({ page }) => {
  await open(page);
  const add = page.getByRole("group", { name: "Steering", exact: true }).getByRole("button", { name: "+ Steer", exact: true });
  for (const key of ["Enter", "Space"]) {
    await add.focus();
    await add.press(key);
    await expect(add).toHaveAttribute("aria-expanded", "true");
    await add.press(key);
    await expect(add).toHaveAttribute("aria-expanded", "false");
  }
});

test("chat records steering, compares the turn unsteered, and clears steering", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await open(page);
  await page.evaluate(async ({ storesUrl, registryUrl }) => {
    const stores = await import(storesUrl);
    const registry = await import(registryUrl);
    const channel = await stores.ensureRuntimeChannel();
    const original = channel.send.bind(channel);
    (window as any).steeringRequests = [];
    channel.send = (payload: unknown) => { (window as any).steeringRequests.push(payload); return original(payload); };
    await registry.getRuntimeClient().tree.castPut("assistant", { steering: "0.25 fixture/calm.focused" });
    stores.applyRecipe("0.5 fixture/calm.focused");
  }, { storesUrl, registryUrl });
  await expect.poll(() => page.evaluate(async url => (await import(url)).currentSteeringExpression(), storesUrl))
    .toBe("0.5 fixture/calm.focused");

  await sendMessage(page, "Tell me something interesting");
  expect((await submits(page)).at(-1).steering).toBe("0.5 fixture/calm.focused");
  const reply = page.locator(".chat .msg:not(.user)").last();
  await expect(marker(reply)).toHaveText("steered");
  await expect(marker(reply)).toHaveAttribute("data-tip", "0.5 fixture/calm.focused");

  // A later recipe change leaves the recorded reply alone.
  await page.evaluate(async url => (await import(url)).applyRecipe("0.8 fixture/calm.focused"), storesUrl);
  await expect(marker(reply)).toHaveAttribute("data-tip", "0.5 fixture/calm.focused");

  const compare = reply.getByRole("button", { name: "Compare unsteered", exact: true });
  await compare.click();
  await expect.poll(async () => (await page.evaluate(() => (window as any).steeringRequests))
    .some((request: any) => request.stateless)).toBe(true);
  await waitForReply(page);
  const comparison = (await page.evaluate(() => (window as any).steeringRequests)).find((request: any) => request.stateless);
  expect(comparison.recipe_override).toBe("unsteered");
  expect(comparison.steering).toBe("");
  expect(comparison.input).toEqual([{ role: "user", content: "Tell me something interesting", label: null }]);
  expect(comparison.parent_node_id).toBeTruthy();

  for (const width of [1440, 320]) {
    await page.setViewportSize({ width, height: 1000 });
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    expect((await new AxeBuilder({ page }).include(".chat").analyze()).violations).toEqual([]);
  }

  await page.evaluate(async url => (await import(url)).applyRecipe(""), storesUrl);
  await sendMessage(page, "Now without steering");
  expect((await submits(page)).at(-1).steering).toBe("");
  await expect(marker(page.locator(".chat .msg:not(.user):not(.shadow)").last())).toHaveText("unsteered");
  expect(errors).toEqual([]);
});
