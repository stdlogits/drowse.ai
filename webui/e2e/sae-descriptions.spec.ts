import { expect, test, type Locator, type Page } from "@playwright/test";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { inspectToken, openLayoutWorkbench, sendMessage, sourceUrl } from "./workbench-navigation";

// SAE feature labels in the Token view's Features: published Gemma Scope labels come from the
// bundled index (no request leaves the device, and they stay once offline); labels for any other
// dictionary load from Neuronpedia by feature id, report a failure and retry.

const registryUrl = sourceUrl("lib/runtime/registry.ts");
// Enough features that the Features list offers its filter (more than eight).
const featureIds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 2286];

interface DescriptionSource { model: string; source: string; repository: string; folder: string }

/** Binds the in-page runtime's SAE to `source` at `layer` and makes its token replay answer
 *  unlabelled features, so the Token view has to find their labels. */
async function bindSae(page: Page, source: DescriptionSource, layer: number) {
  await page.evaluate(async ({ registryUrl, source, layer, featureIds }) => {
    const runtime = (await import(registryUrl)).getRuntimeClient();
    const instruments = runtime.instruments;
    const sources = instruments.sources.bind(instruments);
    instruments.sources = async (family: string) => {
      const result = await sources(family);
      return family === "sae"
        ? { ...result, sources: result.sources.map((row: object) => ({ ...row, layer, description_source: source })) }
        : result;
    };
    const tokenReadout = instruments.tokenReadout.bind(instruments);
    instruments.tokenReadout = async (family: string, ...rest: unknown[]) => {
      const result = await tokenReadout(family, ...rest);
      const sae = result.measurements.instruments.sae;
      if (family === "sae" && sae) {
        sae.binding.layer = layer;
        sae.readout.features = featureIds.map((id: number, i: number) => ({ id, activation: 9 - i * 0.5, label: null, max_act: 10 }));
      }
      return result;
    };
  }, { registryUrl, source, layer, featureIds });
}

async function openFeatures(page: Page): Promise<Locator> {
  await sendMessage(page, "Inspect the feature labels.");
  const token = await inspectToken(page, "deterministic");
  const features = token.getByRole("region", { name: "Features", exact: true });
  await expect(features.getByRole("listitem")).toHaveCount(featureIds.length);
  return features;
}

const featureRow = (features: Locator, id: number) =>
  features.getByRole("listitem").filter({ has: features.page().getByText(new RegExp(`^L\\d+ f${id}$`)) });

async function open(page: Page) {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 1280, height: 1000 });
  await openLayoutWorkbench(page, "instruments");
}

for (const [size, layer] of [["270m", 12], ["1b", 13], ["4b", 17]] as const) {
  test(`published SAE labels for Gemma ${size} come from the bundled index and stay offline`, async ({ page }) => {
    const index = JSON.parse(await readFile(resolve(`../drowse/webui/src/lib/data/sae-descriptions/gemma-3-${size}-it.json`), "utf8"));
    const label = (id: number): string => index.explanations[String(id)][0];
    const external: string[] = [];
    await page.route("https://www.neuronpedia.org/**", async route => {
      external.push(route.request().url());
      await route.abort();
    });
    await open(page);
    await bindSae(page, index.source, layer);
    const features = await openFeatures(page);

    const row = featureRow(features, 2286);
    await expect(row).toContainText(label(2286));
    await expect(features).toContainText("labels from Neuronpedia");
    await expect(features.getByRole("button", { name: /^(Load|Retry) labels$/ })).toHaveCount(0);
    await row.getByRole("button").first().click();
    await expect(row.getByRole("link", { name: /^Neuronpedia/ }))
      .toHaveAttribute("href", `https://www.neuronpedia.org/${index.source.model}/${index.source.source}/2286`);
    expect(external).toEqual([]);

    // Offline, the labels stay and the filter finds a feature by its label.
    await page.context().setOffline(true);
    const query = label(2286);
    await features.getByRole("searchbox", { name: "Find a feature", exact: true }).fill(query);
    const expected = featureIds.filter(id => `f${id} sae/${id} ${label(id)}`.toLowerCase().includes(query.toLowerCase()));
    await expect(features.getByRole("listitem")).toHaveCount(expected.length);
    await expect(row).toContainText(label(2286));
    await page.context().setOffline(false);
    expect(external).toEqual([]);
  });
}

test("labels for an unpublished dictionary load by feature id, report a failure and retry", async ({ page }) => {
  const source = {
    model: "gemma-3-12b-it",
    source: "20-gemmascope-2-res-16k",
    repository: "google/gemma-scope-2-12b-it",
    folder: "resid_post/layer_20_width_16k_l0_medium",
  };
  const requests: { method: string; url: string; body: string | null }[] = [];
  let online = false;
  await page.route("https://www.neuronpedia.org/**", async route => {
    const request = route.request();
    requests.push({ method: request.method(), url: request.url(), body: request.postData() });
    if (!online) return route.abort();
    const id = Number(new URL(request.url()).pathname.split("/").at(-1));
    await route.fulfill({
      status: 200,
      headers: { "access-control-allow-origin": "*" },
      contentType: "application/json",
      body: JSON.stringify({
        modelId: source.model, layer: source.source, index: id,
        source: { hfRepoId: source.repository, hfFolderId: source.folder },
        explanations: [{ description: `published label ${id}`, explanationModelName: "fixture-explainer" }],
      }),
    });
  });
  await open(page);
  await bindSae(page, source, 20);
  const features = await openFeatures(page);

  // The automatic lookup fails: the rows stay unlabelled, and the section says so and offers a retry.
  await expect(features.getByRole("alert")).toHaveText("Some labels did not load");
  const retry = features.getByRole("button", { name: "Retry labels", exact: true });
  await expect(retry).toBeVisible();
  await expect(featureRow(features, 2286)).not.toContainText("published label");

  online = true;
  await retry.click();
  await expect(featureRow(features, 2286)).toContainText("published label 2286");
  await expect(featureRow(features, 0)).toContainText("published label 0");
  await expect(features).toContainText("labels from Neuronpedia");
  await expect(features.getByRole("alert")).toHaveCount(0);
  await expect(features.getByRole("button", { name: /^(Load|Retry) labels$/ })).toHaveCount(0);

  // Only feature ids leave: GETs to the feature records, with no body.
  expect(requests.length).toBeGreaterThan(0);
  for (const request of requests) {
    expect(request.method).toBe("GET");
    expect(request.body).toBeNull();
    expect(request.url).toMatch(new RegExp(`^https://www\\.neuronpedia\\.org/api/feature/${source.model}/${source.source}/\\d+$`));
  }
});
