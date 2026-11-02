import { expect, test } from "@playwright/test";
import { inspectToken, openLayoutWorkbench, selectRoom, sendMessage } from "./workbench-navigation";

// The Units room and a token's anatomy on a phone (the layout fixture's units; units.spec has the
// wide screen): reached from the Menu, contained in the width, every read reachable.

test.use({ hasTouch: true });

const contained = () => document.documentElement.scrollWidth - document.documentElement.clientWidth <= 1;

test("a phone reads units in the Units room and a token's anatomy in the sheet", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 390, height: 844 });
  await openLayoutWorkbench(page, "instruments");
  await sendMessage(page, "Read my units on a phone.");

  await selectRoom(page, "Units");
  const selected = page.getByRole("region", { name: "Selected unit", exact: true });
  const results = page.getByRole("region", { name: "Results", exact: true });
  await expect(results).toContainText("Most active in this reply");
  const row = results.getByRole("button", { name: /^L[0-3] n\d{1,2} [−-]?\d/ }).nth(2);
  const id = (await row.innerText()).match(/^L\d n\d+/)![0];
  await row.scrollIntoViewIfNeeded();
  await row.click();
  await expect(selected.getByRole("heading", { level: 2 })).toHaveText(id);
  const silence = selected.getByRole("button", { name: "Silence", exact: true });
  await silence.scrollIntoViewIfNeeded();
  await expect(silence).toBeInViewport();
  const search = page.getByRole("searchbox", { name: "Search units", exact: true });
  await search.scrollIntoViewIfNeeded();
  await search.fill("L3 h0");
  await search.press("Enter");
  await expect(results).toContainText("Matches");
  // A unit chosen before stays in the detail until the match is chosen (lanes/N3.md).
  await results.getByRole("button", { name: /^L3 h0\b/ }).click();
  await expect(selected.getByRole("heading", { level: 2 })).toHaveText("L3 h0");
  await expect.poll(() => page.evaluate(contained)).toBe(true);

  await selectRoom(page, "Workbench");
  const token = await inspectToken(page, "deterministic");
  const pushed = token.getByRole("region", { name: "Units that pushed the token", exact: true });
  await expect(pushed.getByRole("button", { name: /^L[0-3] [nh]\d{1,2} [+−]\d/ })).toHaveCount(8);
  const layers = token.getByRole("region", { name: "Layer by layer", exact: true });
  await layers.scrollIntoViewIfNeeded();
  await expect(layers.getByRole("button", { name: /^L[0-3] / })).toHaveCount(4);
  await expect.poll(() => page.evaluate(contained)).toBe(true);
});
