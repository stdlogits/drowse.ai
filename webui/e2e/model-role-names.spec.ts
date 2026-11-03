import { expect, test } from "@playwright/test";
import { composer, openFixtureWorkbench, storesUrl, waitForReply } from "./workbench-navigation";

test("changing the model role updates requests, messages, and saved recipes", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.addInitScript(() => {
    const messages: any[] = [];
    (window as any).__roleRequests = messages;
    const postMessage = Worker.prototype.postMessage;
    Worker.prototype.postMessage = function(message, ...rest: any[]) {
      if (message?.command === "submit") messages.push(structuredClone(message.payload));
      return postMessage.call(this, message, ...rest);
    };
  });
  await openFixtureWorkbench(page);
  await page.getByRole("button", { name: "user → assistant", exact: true }).click();
  const role = page.getByRole("combobox", { name: "Model writes as", exact: true });
  const labels: string[] = [];
  for (const name of ["pirate", "forest_guide", "assistant"]) {
    await role.fill(name);
    await role.press("Tab");
    await composer(page).fill("Arrgh");
    await page.getByRole("button", { name: "Send", exact: true }).click();
    await expect.poll(() => page.evaluate(() => (window as any).__roleRequests.length)).toBe(labels.length + 1);
    await expect(page.locator(".chat .msg:not(.user)")).toHaveCount(labels.length + 1);
    await waitForReply(page);
    expect(await page.evaluate(() => (window as any).__roleRequests.at(-1).sampling.assistant_role ?? null))
      .toBe(name === "assistant" ? null : name);
    labels.push(name);
    // The seat names a custom role; the default assistant seat names the model.
    await expect(page.locator(".chat .msg:not(.user) > .dw-seat"))
      .toHaveText(labels.map(label => label === "assistant" ? "qwen3-1.7b-fixture" : label));
    expect(await page.evaluate(async url => {
      const stores = await import(url);
      const generated = [...stores.loomTree.nodes.values()].filter((node: any) => node.role === "assistant");
      return generated.map((node: any) => [node.role_label, node.recipe.sampling.assistant_role]);
    }, storesUrl)).toEqual(labels.map(label => label === "assistant" ? [null, null] : [label, label]));
  }
});
