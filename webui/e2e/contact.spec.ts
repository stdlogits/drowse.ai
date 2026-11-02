import { expect, test, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { composer, devUrl as dev, openMenuItem, setAppearance, settle } from "./workbench-navigation";

const endpoint = "https://drowse.ai/api/contact";
const feedbackDialog = (page: Page) => page.getByRole("dialog", { name: "Feedback", exact: true });

/** The workbench's Feedback dialog: the header Menu's Feedback item. */
async function openFeedback(page: Page, load = true) {
  if (load) {
    await page.goto(`${dev}/app?layoutFixture=1`);
    await expect(composer(page)).toBeVisible();
  }
  await openMenuItem(page, "Feedback");
  await expect(feedbackDialog(page)).toBeVisible();
}

async function fillMessage(page: Page) {
  await page.getByRole("textbox", { name: "Title", exact: true }).fill("A synthetic feedback test");
  await page.getByRole("textbox", { name: "Message", exact: true }).fill("Synthetic test content. No real email is sent.");
}

test("contact email is available in public footers, feedback and help", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  const address = "contact@drowse.ai";
  for (const path of ["/", "/contact", "/credits"]) {
    await page.goto(`${dev}${path}`);
    await expect(page.getByRole("navigation", { name: "Footer", exact: true }).getByRole("link", { name: "Email us", exact: true })).toHaveAttribute("href", `mailto:${address}`);
    expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1);
  }
  await openFeedback(page);
  await expect(feedbackDialog(page).getByRole("link", { name: address, exact: true })).toHaveAttribute("href", `mailto:${address}`);
  await page.getByRole("button", { name: "Close feedback", exact: true }).click();
  await openMenuItem(page, "Help and shortcuts");
  await expect(page.getByRole("region", { name: "Help reference", exact: true }).getByRole("link", { name: address, exact: true })).toHaveAttribute("href", `mailto:${address}`);
});

test("feedback keeps focus inside, validates, and preserves a closed draft", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await openFeedback(page);
  const dialog = feedbackDialog(page);
  await expect(dialog.getByRole("link", { name: /Contact us/ })).toHaveAttribute("href", "/contact");
  await expect(dialog.getByRole("button", { name: "Close feedback", exact: true })).toBeFocused();
  await page.keyboard.press("Shift+Tab");
  await expect(dialog.getByRole("link", { name: "contact@drowse.ai", exact: true })).toBeFocused();
  await page.keyboard.press("Tab");
  await expect(dialog.getByRole("button", { name: "Close feedback", exact: true })).toBeFocused();
  await page.getByRole("button", { name: "Send message", exact: true }).click();
  await expect(page.getByRole("textbox", { name: "Title", exact: true })).toBeFocused();
  await expect(page.getByText("Add a short title.")).toBeVisible();
  await fillMessage(page);
  await page.getByRole("textbox", { name: /^Email/ }).fill("invalid");
  await page.getByRole("button", { name: "Send message", exact: true }).click();
  await expect(page.getByRole("textbox", { name: /^Email/ })).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(dialog).not.toBeVisible();
  await openFeedback(page, false);
  await expect(page.getByRole("textbox", { name: "Title", exact: true })).toHaveValue("A synthetic feedback test");
  await page.getByRole("button", { name: "Topic" }).click();
  await page.keyboard.press("Escape");
  await expect(dialog).toBeVisible();
});

test("offline and failed sends retain text and retry the same reference before success", async ({ page, context }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await openFeedback(page);
  await fillMessage(page);
  let requests: Record<string, string>[] = [];
  await page.route(endpoint, async route => {
    const body = route.request().postDataJSON();
    requests.push(body);
    await route.fulfill({ status: requests.length === 1 ? 503 : 200, contentType: "application/json", headers: { "Access-Control-Allow-Origin": dev }, body: JSON.stringify(requests.length === 1 ? { error: "unavailable" } : { status: "sent", reference: body.requestId }) });
  });
  await context.setOffline(true);
  await page.getByRole("button", { name: "Send message", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("offline");
  expect(requests).toHaveLength(0);
  await context.setOffline(false);
  await page.getByRole("button", { name: "Send message", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("contact service isn't responding");
  await expect(page.getByRole("textbox", { name: "Message", exact: true })).toHaveValue("Synthetic test content. No real email is sent.");
  await page.getByRole("button", { name: "Check delivery", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Message sent" })).toBeVisible();
  expect(requests).toHaveLength(2);
  expect(requests[0].requestId).toBe(requests[1].requestId);
  expect(requests[0].email).toBe("");
  expect(Object.keys(requests[0]).sort()).toEqual(["body", "email", "reason", "requestId", "source", "title", "topic", "website"]);
  await expect(page.getByText("You didn't leave an email, so we can't reply.")).toBeVisible();
  await page.getByRole("button", { name: "Done", exact: true }).click();
});

test("sending is single-flight and survives closing the popup", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await openFeedback(page);
  await fillMessage(page);
  let release!: () => void;
  let count = 0;
  const pending = new Promise<void>(resolve => { release = resolve; });
  await page.route(endpoint, async route => {
    count++;
    await pending;
    await route.fulfill({ status: 200, contentType: "application/json", headers: { "Access-Control-Allow-Origin": dev }, body: JSON.stringify({ status: "sent", reference: route.request().postDataJSON().requestId }) });
  });
  await page.getByRole("button", { name: "Send message", exact: true }).click();
  await expect(page.getByRole("button", { name: "Sending…", exact: true })).toBeDisabled();
  await expect.poll(() => count).toBe(1);
  await page.getByRole("button", { name: "Close feedback" }).click();
  release();
  await openFeedback(page, false);
  await expect(page.getByRole("heading", { name: "Message sent" })).toBeVisible();
  expect(count).toBe(1);
});

test("unconfirmed, rate-limited, and invalid success responses never lose the draft", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(`${dev}/contact`);
  await fillMessage(page);
  const references: string[] = [];
  await page.route(endpoint, async route => {
    const reference = route.request().postDataJSON().requestId;
    references.push(reference);
    const responses = [
      { status: 200, contentType: "text/html", body: "<!doctype html><title>Pages fallback</title>" },
      { status: 202, contentType: "application/json", body: JSON.stringify({ status: "pending", reference }) },
      { status: 429, contentType: "application/json", body: JSON.stringify({ error: "rate_limited" }) },
      { status: 200, contentType: "application/json", body: JSON.stringify({ status: "sent", reference: "wrong-reference" }) },
      { status: 200, contentType: "application/json", body: JSON.stringify({ status: "sent", reference }) },
    ];
    await route.fulfill({ ...responses[references.length - 1], headers: { "Access-Control-Allow-Origin": dev } });
  });
  for (const message of ["couldn't send this", "haven't confirmed", "Too many attempts", "couldn't send this"]) {
    await page.getByRole("button", { name: /^(Send message|Check delivery)$/ }).click();
    await expect(page.getByRole("alert")).toContainText(message);
    await expect(page.getByRole("heading", { name: "Message sent" })).not.toBeVisible();
    await expect(page.getByRole("textbox", { name: "Message", exact: true })).toHaveValue("Synthetic test content. No real email is sent.");
  }
  await page.getByRole("button", { name: /^(Send message|Check delivery)$/ }).click();
  await expect(page.getByRole("heading", { name: "Message sent" })).toBeVisible();
  expect(references).toHaveLength(5);
  expect(new Set(references).size).toBe(1);
});

test("a slow connection times out and leaves a recoverable draft", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(`${dev}/contact`);
  await fillMessage(page);
  await page.clock.install();
  let release!: () => void;
  let started = false;
  const pending = new Promise<void>(resolve => { release = resolve; });
  await page.route(endpoint, async route => {
    started = true;
    await pending;
    await route.fulfill({ status: 503, contentType: "application/json", body: "{}", headers: { "Access-Control-Allow-Origin": dev } });
  });
  await page.getByRole("button", { name: "Send message", exact: true }).click();
  await expect.poll(() => started).toBe(true);
  await page.clock.fastForward(25_001);
  await expect(page.getByRole("alert")).toContainText("Your draft is still here.");
  release();
  await expect(page.getByRole("heading", { name: "Message sent" })).not.toBeVisible();
  await expect(page.getByRole("textbox", { name: "Title", exact: true })).toHaveValue("A synthetic feedback test");
});

test("feedback fits a narrow phone and exposes accessible validation and clear controls", async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 568 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await openFeedback(page);
  const dialog = feedbackDialog(page);
  await fillMessage(page);
  await page.getByRole("textbox", { name: /^Email/ }).fill("invalid");
  await page.getByRole("button", { name: "Send message", exact: true }).click();
  expect(await dialog.evaluate(node => node.scrollWidth - node.clientWidth)).toBeLessThanOrEqual(1);
  for (const theme of ["dark", "light"]) {
    await page.evaluate(value => { document.documentElement.dataset.theme = value; }, theme);
    await settle(page);
    const audit = await new AxeBuilder({ page }).include('[role="dialog"]').analyze();
    expect(audit.violations).toEqual([]);
  }
  await page.getByRole("button", { name: "Clear draft", exact: true }).click();
  await expect(page.getByRole("textbox", { name: "Title", exact: true })).toHaveValue("A synthetic feedback test");
  await page.getByRole("button", { name: "Confirm clear draft", exact: true }).click();
  await expect(page.getByRole("textbox", { name: "Title", exact: true })).toHaveValue("");
  await page.keyboard.press("Escape");
  await expect(dialog).toHaveCount(0);
});

test("topics follow the reason, preserve shared choices, and submit the matching category", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(`${dev}/contact`);
  await fillMessage(page);
  const reason = page.getByRole("button", { name: /^Reason / });
  const topic = page.getByRole("button", { name: /^Topic / });
  await expect(reason).toHaveAccessibleName("Reason General feedback");
  await expect(topic).toHaveAccessibleName("Topic Drowse in general");
  async function chooseReason(label: string) {
    await reason.click();
    await page.getByRole("option", { name: label, exact: true }).click();
  }
  async function topicLabels() {
    await topic.click();
    const labels = await page.getByRole("listbox", { name: "Topic", exact: true }).getByRole("option").allTextContents();
    await page.keyboard.press("Escape");
    return labels.map(label => label.trim());
  }
  await topic.click();
  await page.getByRole("option", { name: "Models and downloads", exact: true }).click();
  await chooseReason("Help and support");
  await expect(topic).toContainText("Models and downloads");
  expect(await topicLabels()).toContain("Getting started");
  await chooseReason("Bug report");
  await expect(topic).toContainText("Models and downloads");
  expect(await topicLabels()).not.toContain("Getting started");
  await chooseReason("Feature suggestion");
  await expect(topic).toContainText("Models and downloads");
  await chooseReason("Media and press");
  await expect(topic).toContainText("Interview request");
  expect(await topicLabels()).toEqual(["Interview request", "Coverage and fact checking", "Images and press materials", "Something else"]);
  await chooseReason("Research and collaboration");
  await expect(topic).toContainText("Research collaboration");
  expect(await topicLabels()).toEqual(["Research collaboration", "Using Drowse in a study", "Methods and validation", "Sharing results", "Grants and research funding", "Something else"]);
  await topic.click();
  await page.getByRole("option", { name: "Grants and research funding", exact: true }).click();
  await chooseReason("Funding and sponsorship");
  await expect(topic).toContainText("Grants and research funding");
  expect(await topicLabels()).toEqual(["Sponsoring Drowse", "Grants and research funding", "Something else"]);
  await chooseReason("Teaching with Drowse");
  await expect(topic).toContainText("Using Drowse in a course");
  expect(await topicLabels()).toEqual(["Using Drowse in a course", "Teaching materials", "Something else"]);
  await topic.click();
  await page.getByRole("option", { name: "Something else", exact: true }).click();
  await chooseReason("Media and press");
  await expect(topic).toContainText("Something else");
  await chooseReason("Other inquiry");
  await expect(topic).toContainText("Not applicable");
  await expect(topic).toBeDisabled();
  await chooseReason("Media and press");
  await expect(topic).toBeEnabled();
  await expect(topic).toContainText("Interview request");
  await topic.click();
  await page.getByRole("option", { name: "Images and press materials", exact: true }).click();
  await expect(page.getByRole("textbox", { name: "Title", exact: true })).toHaveValue("A synthetic feedback test");
  await expect(page.getByRole("textbox", { name: "Message", exact: true })).toHaveValue("Synthetic test content. No real email is sent.");
  await page.route(endpoint, async route => {
    const body = route.request().postDataJSON();
    expect(body.reason).toBe("media");
    expect(body.topic).toBe("press-materials");
    await route.fulfill({ status: 200, contentType: "application/json", headers: { "Access-Control-Allow-Origin": dev }, body: JSON.stringify({ status: "sent", reference: body.requestId }) });
  });
  await page.getByRole("button", { name: "Send message", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Message sent" })).toBeVisible();
});

test("contact categories, success, and mobile layout are accessible", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(`${dev}/contact`);
  await expect(page).toHaveTitle("Contact us · Drowse");
  await page.getByRole("button", { name: "Reason" }).click();
  for (const label of ["Help and support", "Bug report", "Media and press", "Research and collaboration"]) await expect(page.getByRole("option", { name: label, exact: true })).toBeVisible();
  await page.getByRole("option", { name: "Media and press", exact: true }).click();
  await fillMessage(page);
  await page.getByRole("textbox", { name: /^Email/ }).fill("test@example.com");
  await page.route(endpoint, async route => {
    const body = route.request().postDataJSON();
    expect(body.source).toBe("contact");
    expect(body.reason).toBe("media");
    expect(body.topic).toBe("interview");
    await route.fulfill({ status: 200, contentType: "application/json", headers: { "Access-Control-Allow-Origin": dev }, body: JSON.stringify({ status: "sent", reference: body.requestId }) });
  });
  for (const theme of ["dark", "light"]) {
    await setAppearance(page, theme);
    await expect(page.locator("html")).toHaveAttribute("data-theme", theme);
    for (const width of [320, 390, 768, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      await settle(page);
      expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBeLessThanOrEqual(1);
      const audit = await new AxeBuilder({ page }).include("main").analyze();
      expect(audit.violations).toEqual([]);
    }
  }
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Send message", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Message sent" })).toBeVisible();
  await expect(page.getByText("We can reply to test@example.com.")).toBeVisible();
  await page.getByRole("button", { name: "Send another message", exact: true }).click();
  await expect(page.getByRole("textbox", { name: "Message", exact: true })).toHaveValue("");
});

