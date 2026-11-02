import assert from "node:assert/strict";
import { createServer } from "vite";
import { readFile } from "node:fs/promises";
import { parse } from "svelte/compiler";

const server = await createServer({
  appType: "custom", logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
try {
  const { completionAnchor } = await server.ssrLoadModule("/../drowse/webui/src/lib/completionSelection.ts");
  const turns = [{ nodeId: "prompt", text: "Once " }, { nodeId: "reply", text: "upon a moon." }];
  assert.deepEqual(completionAnchor([], "root", 0), { parentNodeId: "root" });
  assert.deepEqual(completionAnchor(turns, "root", 0), { parentNodeId: "root" });
  assert.deepEqual(completionAnchor(turns, "root", 3), { parentNodeId: "root", branch: { nodeId: "prompt", text: "Onc" } });
  assert.deepEqual(completionAnchor(turns, "root", 5), { parentNodeId: "prompt" });
  assert.deepEqual(completionAnchor(turns, "root", 9), { parentNodeId: "prompt", branch: { nodeId: "reply", text: "upon" } });
  assert.deepEqual(completionAnchor(turns, "root", 17), { parentNodeId: "reply" });
  assert.deepEqual(completionAnchor([{ nodeId: "unicode", text: "A 🌙 moon" }], "root", 4), { parentNodeId: "root", branch: { nodeId: "unicode", text: "A 🌙" } });
  assert.throws(() => completionAnchor([{ text: "draft" }], "root", 1), /not saved/);
  const { render } = await server.ssrLoadModule("svelte/server");
  const { completionTokenViews, projectEditedTokens } = await server.ssrLoadModule("/../drowse/webui/src/lib/rawTokenInspection.ts");
  const capturedTokens = ["I ", "love ", "marmots", "."].map((text, tokenId) => ({ text, tokenId, logprob: -tokenId - 0.1 }));
  const originalViews = completionTokenViews([
    { role: "user", generated: false, text: "Prompt: ", nodeId: "prompt" },
    { role: "assistant", generated: true, text: "I love marmots.", nodeId: "original", tokens: capturedTokens },
  ]);
  const insertedViews = projectEditedTokens(originalViews, "Prompt: I really love marmots.");
  assert.equal(insertedViews.map(view => view.text).join(""), "Prompt: I really love marmots.");
  assert.equal(insertedViews.find(view => view.text === "really ").tok, null);
  assert.equal(insertedViews.find(view => view.text === "really ").source, "draft");
  assert.equal(insertedViews.find(view => view.text === "love ").tok, capturedTokens[1]);
  assert.equal(insertedViews.find(view => view.text === "love ").contextChanged, true);
  assert.equal(insertedViews.find(view => view.text === "I ").contextChanged, false);
  assert.equal(insertedViews.find(view => view.text === "love ").nodeId, "original");
  const promptEdit = projectEditedTokens(originalViews, "New prompt: I love marmots.");
  assert.ok(promptEdit.filter(view => view.tok).every(view => view.contextChanged));
  const withinPrompt = projectEditedTokens(originalViews, "Prompt: extra I love marmots.");
  assert.equal(withinPrompt[0].source, "user", "unchanged user text retains its origin");
  const twiceEdited = projectEditedTokens(insertedViews, "Prompt: I really love marmots!");
  assert.equal(twiceEdited.find(view => view.text === "love ").tok, capturedTokens[1], "unaffected middle tokens survive successive edits");
  assert.ok(twiceEdited.filter(view => view.tok).every(view => view.tok.text === view.text), "only whole unchanged tokens retain recorded scores");
  const partialEdit = projectEditedTokens(originalViews, "Prompt: I loved marmots.");
  assert.ok(!partialEdit.some(view => view.tok === capturedTokens[1]), "a changed token never inherits the old token's probability");
  const appended = projectEditedTokens(originalViews, "Prompt: I love marmots. More");
  assert.ok(appended.filter(view => view.tok).every(view => !view.contextChanged), "appending does not invalidate earlier context");
  assert.deepEqual(projectEditedTokens(originalViews, ""), []);
  assert.equal(completionTokenViews([{ role: "user", generated: false, text: "I ", tokens: [capturedTokens[0]] }])[0].source, "user", "capturing a user token does not make it model-authored");
  assert.equal(completionTokenViews([{ role: "assistant", generated: true, text: "different", tokens: capturedTokens }])[0].tok, null, "mismatched text cannot borrow old measurements");
  for (const text of ["😀", "😁", "Prompt: I love 🦫.", "\nPrompt:\tI love marmots.\n", "Prompt: I love", ""]) {
    const projected = projectEditedTokens(originalViews, text);
    assert.equal(projected.map(view => view.text).join(""), text);
    for (const view of projected) assert.equal(view.text.isWellFormed(), true);
  }
  const emojiViews = projectEditedTokens([], "😀");
  assert.equal(projectEditedTokens(emojiViews, "😁")[0].text, "😁");
  const { tokenProbabilityRows } = await server.ssrLoadModule("/../drowse/webui/src/lib/tokenProbabilities.ts");
  const candidateRows = tokenProbabilityRows({ text: " mean", tokenId: 6, logprob: Math.log(0.001), topAlts: [
    { id: 1, text: " love", logprob: Math.log(0.451) },
    { id: 2, text: " wear", logprob: Math.log(0.047) },
  ] });
  assert.deepEqual(candidateRows.map(row => row.text), [" love", " wear", " mean"]);
  assert.equal(candidateRows.at(-1).chosen, true, "the clicked token stays visible even outside the captured top K");
  assert.equal(tokenProbabilityRows({ text: " mean", tokenId: 6, logprob: -1, topAlts: [{ id: 6, text: " mean", logprob: -1 }] }).length, 1);
  assert.deepEqual(tokenProbabilityRows({ text: "unknown" }), []);
  console.log("Token inspection: draft provenance, unchanged scores, changed-context labels, Unicode, and chosen-token probabilities passed");
  const { default: BaseModelTag } = await server.ssrLoadModule("/../drowse/webui/src/lib/ui/BaseModelTag.svelte");
  assert.match(render(BaseModelTag, { props: { plain: true } }).body, /class="[^"]*plain/);
  assert.doesNotMatch(render(BaseModelTag).body, /class="[^"]*plain/);
  const tagSource = await readFile(new URL("../../drowse/webui/src/lib/ui/BaseModelTag.svelte", import.meta.url), "utf8");
  assert.match(tagSource, /\.base-tag\.plain\s*\{[^}]*padding: 0;[^}]*background: none;/);
  const pickerSource = await readFile(new URL("../src/hosted/ui/HostedApp.svelte", import.meta.url), "utf8");
  assert.match(pickerSource, /<strong><span id=\{`model-name-\$\{model.id\}`\}>\{model.name\}<\/span>\{#if model.modelType === "base"\}<BaseModelTag plain \/>/);
  assert.match(pickerSource, /aria-labelledby=\{`model-name-\$\{model.id\}`\}/,
    "model cards are named by the model name alone; tier, size, and fit are descriptions");
  assert.match(pickerSource, /modelsBySize\(snapshot.models.filter\(\(model\) => model.modelType === "base"\)\)/);
  const { installRuntimeClient } = await server.ssrLoadModule("/../drowse/webui/src/lib/runtime/registry.ts");
  installRuntimeClient({ mode: "browser" });
  const { default: TokenLogitsPopover } = await server.ssrLoadModule("/../drowse/webui/src/lib/ui/TokenLogitsPopover.svelte");
  const probabilityTable = render(TokenLogitsPopover, { props: {
    token: { text: " mean", tokenId: 6, logprob: Math.log(0.001), topAlts: [
      { id: 1, text: " love", logprob: Math.log(0.451) },
      { id: 2, text: " wear", logprob: Math.log(0.047) },
      { id: 3, text: " like", logprob: Math.log(0.038) },
      { id: 4, text: " enjoy", logprob: Math.log(0.025) },
      { id: 5, text: " adore", logprob: Math.log(0.014) },
    ] }, anchor: {}, contextChanged: true, onclose() {},
  } }).body;
  assert.equal((probabilityTable.match(/<tr\b/g) ?? []).length, 7, "five captured alternatives plus the chosen token and header remain visible");
  for (const probability of ["0.451", "0.047", "0.038", "0.025", "0.014", "0.001"]) {
    assert.ok(probabilityTable.includes(probability), "probabilities are not renormalized over the captured subset");
  }
  assert.match(probabilityTable, /Recorded before your edit/);
  assert.match(probabilityTable, /not raw logits/);
  assert.match(probabilityTable, /this token/);
  const { sessionState } = await server.ssrLoadModule("/../drowse/webui/src/lib/stores/session.svelte.ts");
  const { genUiMode, effectiveRawMode, loadGenUiMode, setGenUiMode } =
    await server.ssrLoadModule("/../drowse/webui/src/lib/stores/chat.svelte.ts");
  sessionState.info = { model_id: "fixture/base", is_base_model: true, config: {} };
  genUiMode.mode = "chat";
  assert.equal(effectiveRawMode(), true, "stale chat state cannot override a base model");
  loadGenUiMode();
  assert.equal(genUiMode.mode, "raw");
  setGenUiMode("chat");
  assert.equal(genUiMode.mode, "raw", "base models cannot enable a chat template");
  sessionState.info = { model_id: "fixture/instruct", is_base_model: false, config: {} };
  setGenUiMode("chat");
  assert.equal(effectiveRawMode(), false);
  setGenUiMode("raw");
  assert.equal(effectiveRawMode(), true, "instruct models retain optional raw mode");
  console.log("Base mode enforcement and instruct-mode compatibility passed");

  const { chatLog, genStatus } = await server.ssrLoadModule("/../drowse/webui/src/lib/stores/chat.svelte.ts");
  const { default: RawBuffer } = await server.ssrLoadModule("/../drowse/webui/src/panels/RawBuffer.svelte");
  const { default: StatusFooter } = await server.ssrLoadModule("/../drowse/webui/src/panels/StatusFooter.svelte");
  sessionState.info = { model_id: "fixture/base", is_base_model: true, config: {} };
  chatLog.turns = [{ role: "assistant", text: "I love marmots because", generated: true,
    tokens: [{ text: "I", logprob: -0.2 }] }];
  genStatus.active = true;
  genStatus.startedAt = 1;
  genStatus.tokensSoFar = 1;
  const streaming = render(RawBuffer).body;
  assert.match(streaming, /<textarea[^>]*readonly/);
  assert.match(streaming, /<label for="completion-buffer"[^>]*>Text completion<\/label>/);
  // 6.0: the Inspect segment stays in the Buffer mode group and refuses the change while streaming;
  // its tip says why.
  assert.match(streaming, /<button(?=[^>]*aria-pressed="false")(?=[^>]*data-tip="Stop or finish generation before inspecting")[^>]*>Inspect tokens<\/button>/);
  genStatus.active = false;
  genStatus.finishReason = "cancelled";
  const textContent = node => node.type === "Text"
    ? node.data
    : (node.nodes ?? node.fragment?.nodes ?? []).map(textContent).join("");
  const plain = html => textContent(parse(html, { modern: true }).fragment).replace(/\s+/g, " ").trim();
  const stopped = plain(render(StatusFooter).body);
  assert.match(render(StatusFooter).body, /morph-source/, "current text exists independently of animation");
  const statusSource = await readFile(new URL("../../drowse/webui/src/panels/StatusFooter.svelte", import.meta.url), "utf8");
  assert.match(statusSource, /MorphText/, "status slots share the accessible motion renderer");
  genStatus.tokensSoFar = 24;
  genStatus.tokPerSec = 7.34;
  genStatus.finishedAt = genStatus.startedAt + 7700;
  const metrics = plain(render(StatusFooter).body);
  assert.match(metrics, /24 tokens/);
  assert.match(metrics, /7\.3 tokens\/s/);
  assert.match(metrics, /7\.7s/);
  assert.match(stopped, /Stopped/);
  assert.ok(stopped.indexOf("Stopped") < stopped.indexOf("tokens/s"), "stop reason precedes secondary metrics");
  genStatus.finishReason = "length";
  assert.match(plain(render(StatusFooter).body), /Token limit/);
  genStatus.finishReason = null;
  assert.match(plain(render(StatusFooter).body), /Ended/, "an error or unknown finish must not be called complete");
  genStatus.finishReason = "stop";
  genStatus.tokensSoFar = 0;
  chatLog.turns = [{ role: "user", text: "I love marmots because", generated: false }];
  const saved = render(RawBuffer).body;
  assert.match(saved, /Edit saved/);
  assert.doesNotMatch(saved, /tokens\/s/);
  chatLog.turns = [{ role: "assistant", text: "", generated: true }];
  assert.doesNotMatch(render(RawBuffer).body, /Edit saved/, "zero-token model output is not an authored save");
  console.log("Base completion rendering: streaming lock, inspection guard, visible label, finish reasons, and save status passed");

  const { highlightState } = await server.ssrLoadModule("/../drowse/webui/src/lib/stores/probes.svelte.ts");
  const { tokenRowToScore } = await server.ssrLoadModule("/../drowse/webui/src/lib/stores/loom.svelte.ts");
  const { highlightStyleString } = await server.ssrLoadModule("/../drowse/webui/src/lib/highlight.ts");
  const { SURPRISE_TARGET, ENTROPY_TARGET } = await server.ssrLoadModule("/../drowse/webui/src/lib/tokens.ts");
  const measured = [
    { text: " likely", logprob: -0.1, sampler_entropy: 0.2 },
    { text: " unusual", logprob: -3, sampler_entropy: 1.4 },
  ].map(tokenRowToScore);
  chatLog.turns = [
    { role: "user", text: "Prompt", generated: false },
    { role: "assistant", text: " likely unusual", generated: true, tokens: measured },
  ];
  highlightState.target = SURPRISE_TARGET;
  highlightState.compareTwo = false;
  const coloredBuffer = render(RawBuffer).body;
  assert.match(coloredBuffer, /class="edit-actions /, "edit controls keep a stable place when the draft is clean");
  assert.match(coloredBuffer, /class="color-mirror [^"]*"[^>]*aria-hidden="true"/);
  assert.match(coloredBuffer, /<span(?![^>]*style=)[^>]*>Prompt<\/span>/, "unmeasured prompt text stays untinted");
  for (const token of measured) {
    assert.ok(coloredBuffer.includes(highlightStyleString(token)), "rehydrated scores use the shared highlight ramp in edit mode");
  }
  assert.notEqual(highlightStyleString(measured[0]), highlightStyleString(measured[1]), "different surprisal values have different tints");
  genStatus.active = true;
  assert.match(render(RawBuffer).body, /class="color-mirror /, "the mirror stays available while measured tokens stream");
  genStatus.active = false;
  highlightState.target = ENTROPY_TARGET;
  assert.ok(render(RawBuffer).body.includes(highlightStyleString(measured[1])), "sampler entropy colors use the same mirror");
  measured[0].probes = { "fixture/trait": -0.5, "sae/7": 0.2 };
  measured[1].probes = { "fixture/trait": 0.5, "sae/7": 0.8 };
  highlightState.target = "fixture/trait";
  // A reading paints its value: each token's mirror span carries the shared ramp's style for its
  // own score, and different scores take different tints.
  for (const [target, what] of [["fixture/trait", "probe"], ["sae/7", "SAE"]]) {
    highlightState.target = target;
    const body = render(RawBuffer).body;
    assert.match(body, /class="color-mirror /, `${what} readings colour the mirror`);
    for (const token of measured) {
      const style = highlightStyleString(token);
      assert.match(style, /\S/, `${what} readings have a tint`);
      assert.ok(body.includes(style), `${what} readings colour their tokens with the shared ramp`);
    }
    assert.notEqual(highlightStyleString(measured[0]), highlightStyleString(measured[1]), `different ${what} readings have different tints`);
  }
  highlightState.target = SURPRISE_TARGET;
  highlightState.compareTwo = true;
  highlightState.compareTarget = ENTROPY_TARGET;
  highlightState.smoothBlend = false;
  assert.match(render(RawBuffer).body, /background-image: linear-gradient/);
  highlightState.smoothBlend = true;
  assert.ok(render(RawBuffer).body.includes(highlightStyleString(measured[1])), "smooth comparison uses the shared ramp");
  highlightState.compareTwo = false;
  chatLog.turns[1].tokens = [{ text: " likely unusual", logprob: 0, samplerEntropy: 0.2 }];
  assert.match(render(RawBuffer).body, /Recorded token surprisal is zero/);
  highlightState.compareTwo = true;
  assert.doesNotMatch(render(RawBuffer).body, /stay uncolored/, "nonzero second-channel colors are not described as uncolored");
  highlightState.compareTwo = false;
  chatLog.turns[1].tokens = [{ text: " likely unusual" }];
  assert.match(render(RawBuffer).body, /No readings were recorded for this color/);
  chatLog.turns[1].tokens = [{ text: "mismatched", logprob: -1 }];
  assert.doesNotMatch(render(RawBuffer).body, /class="color-mirror /, "token/text mismatch never paints misaligned scores");
  highlightState.target = null;
  assert.doesNotMatch(render(RawBuffer).body, /class="color-mirror |class="color-notice /, "No color removes the mirror and notices");
  highlightState.compareTarget = null;
  highlightState.smoothBlend = false;
  const { default: Chat } = await server.ssrLoadModule("/../drowse/webui/src/panels/Chat.svelte");
  assert.doesNotMatch(render(Chat, { props: { headersVisible: false } }).body, /aria-label="Active model"/);
  sessionState.info = { model_id: "fixture/instruct", is_base_model: false, config: {} };
  setGenUiMode("chat");
  assert.doesNotMatch(render(Chat, { props: { headersVisible: false } }).body, /aria-label="Active model"/);
  sessionState.info = { model_id: "fixture/base", is_base_model: true, config: {} };
  console.log("Base token colors: edit and live mirrors, surprise, entropy, compare, zero/missing readings, text alignment, and model identity passed");

  const { default: JLensMissingState } = await server.ssrLoadModule("/src/hosted/ui/JLensMissingState.svelte");
  const { default: JLensSourceSection } = await server.ssrLoadModule("/src/hosted/ui/JLensSourceSection.svelte");
  const missingBase = render(JLensMissingState, { props: { modelId: "fixture/base" } }).body;
  assert.match(missingBase, /No lens for base/);
  assert.match(missingBase, />Choose a lens<\/button>/);
  assert.match(render(JLensSourceSection).body, /Word insights are optional for base models/);
  sessionState.info = { model_id: "fixture/instruct", is_base_model: false, config: {} };
  const sources = render(JLensSourceSection).body;
  assert.match(sources, /download or update the compatible pack/);
  assert.match(sources, /Open Models/);
  assert.doesNotMatch(sources, /Word insights are optional for base models|download this one again/);
  const missingSource = await readFile(new URL("../src/hosted/ui/JLensMissingState.svelte", import.meta.url), "utf8");
  assert.match(missingSource, /goto\("artifacts", "lenses"\)/, "Choose a lens opens the Artifacts room's Lenses section");
  console.log("Optional base R lens, the lens missing state and pack recovery copy passed");

  const { default: Landing } = await server.ssrLoadModule("/src/hosted/ui/Landing.svelte");
  const landing = render(Landing).body;
  for (const name of ["Gemma 3 270M", "Gemma 3 1B", "Gemma 3 4B", "Qwen3 1.7B", "Qwen3 4B", "Gemma 3 1B PT", "GPT-2 Base (124M)", "Pythia 70M Deduped Base", "Qwen 3.5 2B Base"]) {
    assert.ok(landing.includes(name), `${name} remains represented on the landing page`);
  }
  assert.doesNotMatch(landing, /Under evaluation|\bbeta\b|Polythetic|Saklas/);
  assert.doesNotMatch(landing, /Install in Drowse|Download files|huggingface\.co/,
    "the model roster never exposes install or file links");
  assert.equal((landing.match(/href="\/app\?choose=1&amp;model=[^"]+"/g) ?? []).length, 9,
    "each listed model opens the app's model picker with that model selected");
  assert.equal((landing.match(/class="model-group /g) ?? []).length, 2, "both model types share one group layout");
  assert.equal((landing.match(/class="model-row /g) ?? []).length, 9);
  const landingMain = landing.match(/<main\b[\s\S]*<\/main>/)?.[0] ?? "";
  assert.equal((landingMain.match(/href="\/app"[^>]*>Open Drowse<\/a>/g) ?? []).length, 2,
    "both primary actions in the page body use the same label and app entry");
  assert.match(landing, /Run a language model on your device/);
  assert.doesNotMatch(landing, /Readouts are measurements|not a transcript of the model’s thoughts|J-lens and SAE tools require compatible packs/);
  assert.doesNotMatch(landing, /Interpretability, hands-on|section-signal|reading-section|footer-surface/,
    "the homepage has no eyebrow copy or added background wrappers");
  const landingSource = await readFile(new URL("../src/hosted/ui/Landing.svelte", import.meta.url), "utf8");
  assert.match(landing, /Follow instructions\.<br\b[^>]*>Explore a conversation\./);
  assert.match(landing, /Continue raw text\.<br\b[^>]*>Study next-token predictions\./);
  const setupSource = await readFile(new URL("../src/hosted/ui/HostedApp.svelte", import.meta.url), "utf8");
  assert.doesNotMatch(setupSource, /Each download includes a precomputed J-lens and SAE/);
  console.log("Unified informational home, shared app entry, qualified model identities, and research copy passed");
} finally {
  await server.close();
}
