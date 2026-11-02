import { expect, test, type Page } from "@playwright/test";
import { chooseFixtureModel, composer, devUrl, openLayoutWorkbench, selectRoom, sendMessage, sourceUrl } from "./workbench-navigation";

// drowse 6.0's two lenses in the browser: the S lens reads a layer as input tokens, the R lens as
// output words. The Lens room reads both at a token (the layout fixture's token anatomy, through
// the tiny gemma3 model's S lens at out {0,1} and R lens at out {0,1}, mid {1,3}); the Artifacts
// room's Lenses section is drowse's 2x2 (LensQuad, no fit form) with the hosted pack recovery rows;
// a first run offers the S lens and units beside the R lens.

test.beforeEach(async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 1280, height: 900 });
});

/** Each layer's row of the Lens room's table: [layer, S read, R say]. */
const layerRows = (page: Page) => page.getByRole("table", { name: "Each layer through both lenses", exact: true })
  .getByRole("row").filter({ has: page.getByRole("rowheader") })
  .evaluateAll(rows => rows.map(row => [...row.querySelectorAll('th, td, [role="rowheader"], [role="cell"]')].map(cell => cell.textContent!.trim())));

test("the Lens room reads each layer through the S and R lenses", async ({ page }) => {
  await openLayoutWorkbench(page, "instruments");
  await sendMessage(page, "Read me through both lenses.");
  await selectRoom(page, "Lens");
  const table = page.getByRole("table", { name: "Each layer through both lenses", exact: true });

  // Whole, with no lens probe over the reply, the room reads the reply's last token.
  await expect(page.getByRole("region", { name: /^Last token / })).toBeVisible();
  await expect(table.getByRole("columnheader")).toHaveText(["", "Reads S lens", "Says R lens"]);
  const check = async () => {
    await expect.poll(async () => (await layerRows(page)).map(([layer]) => layer)).toEqual(["L0", "L1", "L2", "L3"]);
    const [l0, l1, l2, l3] = await layerRows(page);
    // The S lens covers L0 and L1 here; the R lens L0, L1 and L3.
    for (const read of [l0[1], l1[1], l0[2], l1[2], l3[2]]) expect(read).not.toBe("–");
    expect([l2[1], l3[1], l2[2]]).toEqual(["–", "–", "–"]);
  };
  await check();
  const last = await layerRows(page);

  // At a token: the same reads, of that token (the fixture replays a reply as bytes, so another
  // position reads differently).
  await page.getByRole("region", { name: "Tokens", exact: true }).getByRole("button", { name: /^deterministic \(token \d+\)$/ }).click();
  await expect(page.getByRole("region", { name: "At deterministic", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Whole reply", exact: true })).toHaveAttribute("aria-pressed", "false");
  await expect.poll(() => layerRows(page)).not.toEqual(last);
  await check();
});

test("the four lenses show what is loaded and offer the missing packs", async ({ page }) => {
  const lenses = page.getByRole("region", { name: "Lenses", exact: true });
  const quad = lenses.getByRole("region", { name: "The four lenses", exact: true });
  const missing = (name: string) => lenses.getByText(`No ${name} pack is loaded. Open Models to download or update the compatible pack, then reopen the model. Your chats stay saved.`, { exact: true });
  const open = async (fixture: string) => {
    await page.goto(`${devUrl}/app?layoutFixture=${fixture}#/artifacts/lenses`);
    await expect(quad).toBeVisible();
    await expect(quad.getByRole("heading", { level: 2 })).toHaveText(["S-lens", "R-lens"]);
    // The hosted section draws drowse's grid without its local fit.
    await expect(page.getByRole("main")).not.toContainText(/Fit locally|Lens fit progress|Documents fitted/);
  };

  // No pack: every lens empty, and both packs offered.
  await open("1");
  await expect(lenses).toContainText("none prepared");
  await expect(quad.getByRole("img", { name: "No S-lens", exact: true })).toHaveCount(2);
  await expect(quad.getByRole("img", { name: "No R-lens", exact: true })).toHaveCount(2);
  await expect(missing("R lens")).toBeVisible();
  await expect(missing("S lens")).toBeVisible();
  await expect(lenses.getByRole("button", { name: "Open Models", exact: true })).toHaveCount(2);

  // The R pack loaded: R holds its out layers and is in use; only the S pack is offered.
  await open("instruments");
  await expect(lenses).toContainText("R in use");
  await expect(quad.getByRole("img", { name: "Holds L0–L1", exact: true })).toBeVisible();
  await expect(quad.getByRole("img", { name: "No mid site", exact: true })).toBeVisible();
  await expect(quad.getByRole("button", { name: /^fixture-jlens 64 documents/ })).toHaveAttribute("aria-pressed", "true");
  await expect(missing("R lens")).toHaveCount(0);
  await expect(missing("S lens")).toBeVisible();

  // A base model reads with the R lens alone: no S pack is offered.
  await open("base");
  await expect(missing("R lens")).toBeVisible();
  await expect(missing("S lens")).toHaveCount(0);
});

test("a first run offers the S lens and units, and units bring their lenses", async ({ page }) => {
  await page.goto(`${devUrl}/app?layoutFixture=setup&fixtureInstruments=1`);
  await chooseFixtureModel(page);
  const picker = page.locator(".tool-picker");
  const total = picker.locator(".tool-total");
  const sLens = picker.getByRole("checkbox", { name: /^Early-layer reads \(S lens\) Fixture S-lens\. Read early layers as input tokens\./ });
  const units = picker.getByRole("checkbox", { name: /^Neurons and heads \(units\) Fixture units\. Read, rank and ablate neurons and heads\./ });
  // The R lens comes with an instruct model; S and units start chosen beside it.
  await expect(picker.locator(".required-tool", { hasText: "Word insights (R lens)" })).toContainText("Included");
  await expect(sLens).toBeChecked();
  await expect(units).toBeChecked();
  await expect(total).toContainText("tools 6.0 KB");
  await expect(page.locator(".model-grid > button").first()).toContainText("Also selected: Fixture S-lens, Fixture units");

  // Units read through the S lens: dropping it drops units, and choosing units chooses it again.
  await sLens.uncheck();
  await expect(units).not.toBeChecked();
  await expect(total).toContainText("tools 1.0 KB");
  await units.check();
  await expect(sLens).toBeChecked();
  await expect(total).toContainText("tools 6.0 KB");
  await units.uncheck();
  await expect(sLens).toBeChecked();
  await expect(total).toContainText("tools 3.0 KB");
  await units.check();

  await page.getByRole("button", { name: "Download and open", exact: true }).click();
  await expect(composer(page)).toBeVisible();
  expect(await page.evaluate(async url => (await import(url)).getHostedController().snapshot.installedPackIds, sourceUrl("lib/runtime/registry.ts")))
    .toEqual(["fixture-jlens", "fixture-slens", "fixture-units"]);
});
