import { expect, test, type Locator, type Page } from "@playwright/test";
import {
  devUrl, inspectToken, inspector, openLayoutWorkbench, selectRoom, sendMessage, sourceUrl, storesUrl, waitForReply,
} from "./workbench-navigation";

// Half B of drowse 6.0 in the browser: the Units room, a token's anatomy, a unit's activity and
// unit steering. The layout fixture answers every units member through the real host ports
// (BrowserUnitsService, units/fixtureUnits.ts) over the tiny gemma3 model: 4 layers, 96 neurons
// and 4 heads a layer, a byte vocabulary, R lens out {0,1} mid {1,3}, S lens out {0,1} mid {0,2},
// no neuron-lens catalog. Its node replay reads a reply as its text's bytes (48 tokens here), so
// the reads are drowse-shaped and deterministic, not a model's.

const registryUrl = sourceUrl("lib/runtime/registry.ts");
const NEURON = /^L[0-3] n\d{1,2}$/;
const HEAD = /^L[0-3] h[0-3]$/;
/** A unit's row: its ID, then its value ("L2 n84 −3.4"). */
const ROW = /^L[0-3] [nh]\d{1,2} [−-]?\d/;

test.beforeEach(async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 1280, height: 900 });
});

const room = (page: Page) => ({
  atlas: page.getByRole("region", { name: "Atlas", exact: true }),
  results: page.getByRole("region", { name: "Results", exact: true }),
  selected: page.getByRole("region", { name: "Selected unit", exact: true }),
  search: page.getByRole("searchbox", { name: "Search units", exact: true }),
});

const expression = (page: Page) =>
  page.evaluate(async url => (await import(url)).currentSteeringExpression(), storesUrl);

/** Records what the page sends the runtime, to read a generation's steering. */
async function recordSubmits(page: Page): Promise<void> {
  await page.evaluate(async (url) => {
    const stores = await import(url);
    const channel = await stores.ensureRuntimeChannel();
    const original = channel.send.bind(channel);
    (window as any).runtimeRequests = [];
    channel.send = (payload: unknown) => { (window as any).runtimeRequests.push(payload); return original(payload); };
  }, storesUrl);
}
const submittedSteering = (page: Page) => page.evaluate(() =>
  (window as any).runtimeRequests.filter((request: any) => request.type === "submit").map((request: any) => request.steering));

test("the Units room reads the reply's neurons and heads", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await openLayoutWorkbench(page);
  await sendMessage(page, "Read my units.");
  await selectRoom(page, "Units");
  const { atlas, results, selected } = room(page);

  // Neurons, strongest, active here: the whole reply at each unit's peak, every layer a column.
  await expect(page.getByRole("region", { name: "Tokens", exact: true }).getByRole("button", { name: "Whole reply", exact: true }))
    .toHaveAttribute("aria-pressed", "true");
  await expect(atlas).toContainText("This reply, at each unit's peak");
  const columns = atlas.getByRole("group", { name: "Layers", exact: true }).getByRole("button", { name: /^Layer \d$/ });
  await expect(columns).toHaveCount(4);
  await expect(columns.first()).toHaveAccessibleName("Layer 0");
  await expect(results).toContainText("Most active in this reply");
  await expect(results).toContainText("384 units");
  const rows = results.getByRole("button", { name: ROW });
  await expect(rows).toHaveCount(8);
  await expect(selected.getByRole("heading", { level: 2 })).toHaveText(NEURON);
  for (const action of ["Push +", "Push −", "Silence"]) await expect(selected.getByRole("button", { name: action, exact: true })).toBeVisible();
  await expect(selected).toContainText("write norm");
  // Its activation on every token of the replayed reply.
  await expect(selected.getByRole("group", { name: /on each of the 48 tokens/ }).getByRole("button")).toHaveCount(48);
  await expect(selected.getByRole("heading", { name: "Writes", exact: true })).toBeVisible();

  // Another row chooses that unit.
  const second = (await rows.nth(1).getAttribute("aria-label") ?? await rows.nth(1).innerText()).match(/^L\d [nh]\d+/)![0];
  await rows.nth(1).click();
  await expect(selected.getByRole("heading", { level: 2 })).toHaveText(second);
  await expect(rows.nth(1)).toHaveAttribute("aria-pressed", "true");

  // Heads: 4 a layer, read by what they copy; a head is silenced, never pushed.
  await page.getByRole("group", { name: "Units", exact: true }).getByRole("button", { name: "Heads", exact: true }).click();
  await expect(results).toContainText("16 units");
  await expect(rows.first()).toHaveAccessibleName(/^L[0-3] h[0-3] /);
  // The chosen neuron stays chosen until a head is.
  await expect(selected.getByRole("heading", { level: 2 })).toHaveText(second);
  await rows.first().click();
  await expect(selected.getByRole("heading", { level: 2 })).toHaveText(HEAD);
  await expect(selected.getByRole("heading", { name: "Copies", exact: true })).toBeVisible();
  await expect(selected).toContainText("on the neutral text");
  await expect(selected.getByRole("button", { name: /^Push/ })).toHaveCount(0);
  await expect(selected.getByRole("button", { name: "Silence", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Writes a word", exact: true })).toHaveCount(0);

  // A token of the reply: the room reads there instead of the whole reply.
  await page.getByRole("region", { name: "Tokens", exact: true }).getByRole("button", { name: /^deterministic \(token \d+\)$/ }).click();
  await expect(page.getByRole("button", { name: "Whole reply", exact: true })).toHaveAttribute("aria-pressed", "false");
  await expect(atlas).not.toContainText("This reply, at each unit's peak");
  await expect(selected.getByRole("heading", { level: 2 })).toHaveText(HEAD);
  expect(errors).toEqual([]);
});

test("the Units room's search finds a word's writers and a unit by its ID", async ({ page }) => {
  await openLayoutWorkbench(page);
  await sendMessage(page, "Search my units.");
  await selectRoom(page, "Units");
  const { atlas, results, selected, search } = room(page);
  await page.getByRole("button", { name: "Writes a word", exact: true }).click();
  await expect(page.getByRole("button", { name: "Writes a word", exact: true })).toHaveAttribute("aria-pressed", "true");

  // A word: the units whose writes read as it (unitTop by writes), each read through its lens.
  await search.fill("e");
  await search.press("Enter");
  await expect(results).toContainText("Writes e");
  await expect(results).toContainText("48 units");
  await expect(atlas).toContainText("Writes e");
  await expect(selected.getByRole("heading", { level: 2 })).toHaveText(NEURON);
  await expect(selected.getByRole("heading", { name: "Writes", exact: true })).toBeVisible();
  await expect(selected).toContainText(/[RS] · (out|mid)/);
  // A chosen writer gives way to a searched ID (stdlogits/drowse#73).
  const writers = results.getByRole("button", { name: ROW });
  await writers.nth(1).click();
  await expect(writers.nth(1)).toHaveAttribute("aria-pressed", "true");

  // An ID: that unit alone, read from its weights.
  await search.fill("L1 h2");
  await search.press("Enter");
  await expect(results).toContainText("Matches");
  await expect(results).toContainText("1 unit");
  await expect(selected.getByRole("heading", { level: 2 })).toHaveText("L1 h2");
  await expect(selected.getByRole("heading", { name: "Copies", exact: true })).toBeVisible();
  await search.fill("L2 n7");
  await search.press("Enter");
  await expect(selected.getByRole("heading", { level: 2 })).toHaveText("L2 n7");

  await page.getByRole("button", { name: "Clear the search", exact: true }).click();
  await expect(search).toHaveValue("");
  await expect(results).not.toContainText("Matches");
});

test("without a catalog the units carry no labels and the Artifacts room says so", async ({ page }) => {
  await openLayoutWorkbench(page);
  await sendMessage(page, "Label my units.");
  await selectRoom(page, "Units");
  // No catalog, no Labelled filter.
  await expect(page.getByRole("group", { name: "Units shown", exact: true }).getByRole("button"))
    .toHaveText(["Active here", "Writes a word"]);

  await selectRoom(page, "Artifacts");
  await page.getByRole("group", { name: "Artifacts sections", exact: true }).getByRole("button", { name: "Units", exact: true }).click();
  const catalog = page.getByRole("region", { name: "Unit catalog", exact: true });
  await expect(catalog).toContainText("no labels");
  await expect(catalog).toContainText("no neuron-lens catalog for this model");
  await catalog.getByRole("button", { name: "Read units", exact: true }).click();
  await expect(page.locator("main h1")).toHaveText("Units");
  await expect(page).toHaveURL(/#\/units$/);

  // The members behind it, through the dashboard's own runtime client: the catalog is unbound,
  // labels come back empty with the catalog's reason, and find answers a residual vector.
  const answers = await page.evaluate(async (url) => {
    const units = (await import(url)).getRuntimeClient().instruments;
    const catalog = await units.unitCatalog();
    const labels = await units.unitLabels(["neuron/L0n1", "head/L1h2"]);
    const find = await units.unitFind({ vector: Array.from({ length: 64 }, (_, k) => Math.sin(k)), layer: 1 });
    return {
      catalog: [catalog.available, catalog.bound],
      labels: [labels.catalog.bound, labels.labels.length, labels.catalog.reason],
      find: [find.target, Object.keys(find.layers), find.layers["1"].neurons.length > 0, find.layers["1"].neurons.every((hit: { unit: string; label: unknown }) =>
        /^neuron\/L1n\d+$/.test(hit.unit) && hit.label === null)],
    };
  }, registryUrl);
  expect(answers.catalog).toEqual([false, false]);
  expect(answers.labels.slice(0, 2)).toEqual([false, 0]);
  expect(answers.labels[2]).toMatch(/^no neuron-lens catalog for this model/);
  expect(answers.find).toEqual(["vector", ["1"], true, true]);
});

test("a token's anatomy and a unit's activity show in the Token view", async ({ page }) => {
  await openLayoutWorkbench(page, "instruments");
  await sendMessage(page, "Show me a token's anatomy.");
  const token = await inspectToken(page, "deterministic");

  // The units that pushed the token, signed, each with Push (a neuron, on the side it fired) and Silence.
  const pushed = token.getByRole("region", { name: "Units that pushed the token", exact: true });
  await expect(pushed).toContainText("Pushed toward deterministic");
  await expect(pushed.getByRole("button", { name: /^L[0-3] [nh]\d{1,2} [+−]\d/ })).toHaveCount(8);
  await expect(pushed.getByRole("button", { name: /^Silence L\d [nh]\d+$/ })).toHaveCount(8);
  await expect(pushed.getByRole("button", { name: /^Push L\d n\d+/ }).first()).toBeVisible();
  await expect(pushed.getByRole("button", { name: /^Push L\d h\d+/ })).toHaveCount(0);
  // Where the top heads read from, the token's probability by depth, and each layer's reads:
  // the S lens early, the R lens late.
  await expect(token.getByRole("region", { name: "Where the top heads read from", exact: true }).getByRole("button", { name: HEAD }).first()).toBeVisible();
  await expect(token.getByRole("region", { name: "By depth", exact: true }).getByRole("img", { name: "Probability of deterministic by layer" })).toBeVisible();
  const layers = token.getByRole("region", { name: "Layer by layer", exact: true }).getByRole("button", { name: /^L[0-3] / });
  await expect(layers).toHaveCount(4);
  await expect(layers.first()).toHaveAccessibleName(/^L0 S /);
  await expect(layers.last()).toHaveAccessibleName(/^L3 R /);

  // A unit's ID selects it into the Inspector: its reads and its activation on the reply.
  const neuron = pushed.getByRole("button", { name: /^L[0-3] n\d{1,2} [+−]\d/ }).first();
  const id = (await neuron.getAttribute("aria-label") ?? await neuron.innerText()).match(/^L\d n\d+/)![0];
  await neuron.click();
  await expect(inspector(page).getByRole("heading", { level: 2, name: id, exact: true })).toBeVisible();
  for (const heading of ["Writes", "Fires on"]) await expect(inspector(page).getByRole("heading", { name: heading, exact: true })).toBeVisible();
  const activity = inspector(page).getByRole("region", { name: "On this conversation", exact: true });
  await expect(activity).toContainText(/fires on \d+ of 48 tokens/);
  await expect(activity.getByRole("img", { name: `${id} on each of the 48 tokens of the reply` })).toBeVisible();
});

test("neurons and heads steer from the picker, the recipe and the rooms", async ({ page }) => {
  await openLayoutWorkbench(page, "instruments");
  await recordSubmits(page);
  const steering = page.getByRole("group", { name: "Steering", exact: true });
  const add = steering.getByRole("button", { name: "+ Steer", exact: true });
  const unit = page.getByRole("searchbox", { name: "Neuron or head", exact: true });

  // Typed into the picker: a neuron pushed, a head silenced (exact mean-ablation).
  await add.click();
  await expect(page.getByRole("group", { name: "Steer with", exact: true }).getByRole("button", { name: "Neuron or head", exact: true }))
    .toHaveAttribute("aria-pressed", "true");
  await unit.fill("L1 n5");
  await page.getByRole("button", { name: "Push", exact: true }).click();
  await expect.poll(() => expression(page)).toBe("3 neuron/L1n5");
  await page.keyboard.press("Escape");
  await add.click();
  await unit.fill("L2 h1");
  await expect(page.getByRole("button", { name: "Push", exact: true })).toHaveCount(0);
  await page.getByRole("button", { name: "Silence", exact: true }).click();
  await expect.poll(() => expression(page)).toBe("3 neuron/L1n5 + !head/L2h1");
  await expect(steering.getByRole("button", { name: /L1 n5/ })).toHaveAccessibleName(/^3 L1 n5 /);
  await expect(steering.getByRole("button", { name: /L2 h1/ })).toHaveAccessibleName(/^without L2 h1 /);

  // The generation carries the expression, and the reply records it.
  await sendMessage(page, "Steer with units.");
  expect((await submittedSteering(page)).at(-1)).toBe("3 neuron/L1n5 + !head/L2h1");
  const reply = page.locator(".chat .msg:not(.user)").last();
  const marker = reply.locator(".dw-under span").filter({ hasText: /^(steered|unsteered)$/ });
  await expect(marker).toHaveText("steered");
  await expect(marker).toHaveAttribute("data-tip", "3 neuron/L1n5 + !head/L2h1");
  // The stored reply node: the steering it ran with and its recipe (what a replay or a saved chat
  // reads back).
  const stored = await page.evaluate(async (url) => {
    const tree = await (await import(url)).getRuntimeClient().tree.get();
    const reply = tree.nodes.find((node: any) => node.id === tree.active_node_id);
    return [reply.role, reply.applied_steering, reply.recipe?.steering];
  }, registryUrl);
  expect(stored).toEqual(["assistant", "3 neuron/L1n5 + !head/L2h1", "3 neuron/L1n5 + !head/L2h1"]);

  // A typed recipe from elsewhere (an agent's, a saved chat's) becomes the same chips.
  await page.evaluate(async url => (await import(url)).applyRecipe("0.3 neuron/L0n1 + !neuron/L3n86"), storesUrl);
  await expect.poll(() => expression(page)).toBe("0.3 neuron/L0n1 + !neuron/L3n86");
  await expect(steering.getByRole("button", { name: /L0 n1/ })).toHaveAccessibleName(/^0\.3 L0 n1 /);
  await expect(steering.getByRole("button", { name: /L3 n86/ })).toHaveAccessibleName(/^without L3 n86 /);

  // The Units room's Silence and the Token view's Push land in the same strip.
  await page.evaluate(async url => (await import(url)).applyRecipe(""), storesUrl);
  await expect.poll(() => expression(page)).toBe("");
  await selectRoom(page, "Units");
  const { selected } = room(page);
  const chosen = (await selected.getByRole("heading", { level: 2 }).innerText()).trim();
  await selected.getByRole("button", { name: "Silence", exact: true }).click();
  const [layer, index] = chosen.match(/^L(\d+) n(\d+)$/)!.slice(1);
  const silenced = `!neuron/L${layer}n${index}`;
  await expect.poll(() => expression(page)).toBe(silenced);
  await selectRoom(page, "Workbench");
  const token = await inspectToken(page, "deterministic");
  const push = token.getByRole("region", { name: "Units that pushed the token", exact: true }).getByRole("button", { name: /^Push L\d n\d+/ }).first();
  const label = (await push.getAttribute("aria-label"))!;
  const [, pushLayer, pushIndex] = label.match(/^Push L(\d+) n(\d+)/)!;
  await push.click();
  // UNIT_PUSH_ALPHA (3), negative on a neuron's − pole; a push on the silenced neuron replaces it.
  const negative = label.endsWith("on its − pole");
  const unitName = `neuron/L${pushLayer}n${pushIndex}`;
  await expect.poll(() => expression(page)).toBe(pushLayer === layer && pushIndex === index
    ? `${negative ? "-" : ""}3 ${unitName}`
    : `${silenced} ${negative ? "-" : "+"} 3 ${unitName}`);
});

// The base layout fixture's model is Pythia, whose library is residual (standard-v4-residual): its
// units members answer through BrowserUnitsService with the lock's refusal (fixtureHostedRuntime.ts).
const PYTHIA_REFUSAL = "layer 0: block has no softmax self-attention ('self_attn'; found attention, input_layernorm, mlp, post_attention_dropout, post_attention_layernorm, post_mlp_dropout)";

/** A base model's continuation, with the R lens installed (the Token view reads the anatomy
 *  where a lens is ready). */
async function continueBaseText(page: Page): Promise<void> {
  await page.goto(`${devUrl}/app?layoutFixture=base&fixtureInstruments=1`);
  await page.getByRole("textbox", { name: "Editable completion buffer" }).fill("A field note: ");
  await page.getByRole("button", { name: "Continue text", exact: true }).click();
  await waitForReply(page);
}

/** The completion's Token view on one of the model's tokens (docked in the Inspector). */
async function inspectBaseToken(page: Page): Promise<Locator> {
  await page.getByRole("button", { name: "Inspect tokens", exact: true }).click();
  await page.getByRole("button", { name: /^Inspect token deterministic\b/ }).first().click();
  const token = inspector(page).getByRole("region", { name: "Token", exact: true });
  await expect(token).toBeVisible();
  return token;
}

test("a model without unit sites refuses every unit read with the lock's refusal", async ({ page }) => {
  await continueBaseText(page);
  // Through the dashboard's own runtime client: each read, and labels (which check each unit
  // against the model's blocks, native `_parse`), answer native's 422 UnsupportedBlockError with
  // the refusal; the catalog answers, as on any model (disk only natively).
  const answers = await page.evaluate(async (url) => {
    const client = (await import(url)).getRuntimeClient();
    const units = client.instruments;
    const tree = await client.tree.get();
    const refused = async (call: () => Promise<unknown>) => {
      try { await call(); return "answered"; } catch (error: any) { return [error.code, error.status, error.message]; }
    };
    return {
      reads: [
        await refused(() => units.tokenAnatomy(tree.active_node_id, 0)),
        await refused(() => units.unitActivity(tree.active_node_id)),
        await refused(() => units.unitRead("neuron/L0n1")),
        await refused(() => units.unitTop("a")),
        await refused(() => units.unitFind({ vector: Array.from({ length: 64 }, (_, k) => Math.sin(k)), layer: 1 })),
        await refused(() => units.unitLabels(["neuron/L0n1"])),
      ],
      catalog: await units.unitCatalog().then((catalog: any) => [catalog.available, catalog.bound]),
    };
  }, registryUrl);
  expect(answers.reads).toEqual(Array(6).fill(["UnsupportedBlockError", 422, PYTHIA_REFUSAL]));
  expect(answers.catalog).toEqual([false, false]);

  // The Units room shows the failure with Try again, the Token view's Anatomy an alert (their
  // wording: next test).
  await selectRoom(page, "Units");
  await expect(page.getByRole("main").getByRole("button", { name: "Try again", exact: true }).first()).toBeVisible();
  await expect(page.getByRole("main").getByRole("button", { name: ROW })).toHaveCount(0);
  await selectRoom(page, "Workbench");
  const token = await inspectBaseToken(page);
  await expect(token.getByRole("region", { name: "Anatomy", exact: true }).getByRole("alert")).toBeVisible();
  await expect(token.getByRole("region", { name: "Units that pushed the token", exact: true })).toHaveCount(0);
});

test("the refusal reads as the model's refusal, not another tab's lock", async ({ page }) => {
  // userFacingError upper-cases the code to UNSUPPORTEDBLOCKERROR, whose BLOCK once matched its
  // tab-lock pattern (stdlogits/drowse#73).
  await continueBaseText(page);
  await selectRoom(page, "Units");
  await expect(page.getByRole("main")).toContainText(PYTHIA_REFUSAL);
  await expect(page.getByRole("main")).not.toContainText("Another Drowse tab");
  await selectRoom(page, "Workbench");
  const token = await inspectBaseToken(page);
  await expect(token.getByRole("region", { name: "Anatomy", exact: true }).getByRole("alert")).toHaveText(PYTHIA_REFUSAL);
});
