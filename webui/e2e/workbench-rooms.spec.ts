import { expect, test } from "@playwright/test";
import { composer, openFixtureWorkbench, selectRoom } from "./workbench-navigation";

// drowse 6.0's rooms under the hosted shell: the header's tabs (the Menu on a phone), the model
// pill, and the address bar (rooms are hash routes, #/lens).

test.beforeEach(async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
});

test("room tabs, the model pill and the drowse mark move between rooms", async ({ page }) => {
  await openFixtureWorkbench(page);
  const tabs = page.getByRole("navigation", { name: "Rooms", exact: true });
  // The fixture runtime reads no units, so the Units room is hidden.
  await expect(tabs.getByRole("button")).toHaveText(["Workbench", "Lens", "Probes", "Artifacts", "Model"]);
  for (const room of ["Lens", "Probes", "Artifacts", "Workbench"]) {
    await selectRoom(page, room);
    await expect(tabs.getByRole("button", { name: room, exact: true })).toHaveAttribute("aria-pressed", "true");
  }
  await page.getByRole("button", { name: /: model room$/ }).click();
  await expect(page.locator("main h1")).toHaveText("Model");
  await expect(page).toHaveURL(/#\/model$/);
  await page.getByRole("banner").getByRole("button", { name: "drowse", exact: true }).click();
  await expect(page.locator("main h1")).toHaveText("Workbench");
  await expect(composer(page)).toBeVisible();
});

test("a phone reaches every room from the Menu", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openFixtureWorkbench(page);
  await expect(page.getByRole("navigation", { name: "Rooms", exact: true })).toHaveCount(0);
  for (const room of ["Lens", "Artifacts", "Workbench"]) await selectRoom(page, room);
  await expect(composer(page)).toBeVisible();
});

test("Back and Forward move between rooms without leaving the workbench", async ({ page }) => {
  // Product bug (lanes/A5.md, "Back leaves the workbench"): HostedRoot's popstate handler returns
  // to Your chats on any history step while the workbench is open, including a room's hash step.
  test.fixme(true, "HostedRoot treats a room's hash popstate as leaving the app; see lanes/A5.md");
  await openFixtureWorkbench(page);
  await selectRoom(page, "Lens");
  await page.goBack();
  await expect(page.locator("main h1")).toHaveText("Workbench");
  await expect(composer(page)).toBeVisible();
  await page.goForward();
  await expect(page.locator("main h1")).toHaveText("Lens");
});
