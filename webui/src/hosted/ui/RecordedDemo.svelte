<script lang="ts">
  import { MorphText, Segmented, fadeOut, rise } from "@stdlogits/ledger";
  import DemoReply from "./DemoReply.svelte";
  import Slider from "../../../../drowse/webui/src/lib/Slider.svelte";
  import RecordedTokenText from "./RecordedTokenText.svelte";
  import recording from "../data/neighbor-example.json";

  let mode = $state<"steering" | "prediction">("steering");
  let setting = $state(3);
  let coloring = $state<"surprisal" | "probability">("surprisal");
  const promptPreview = recording.prompt.split(" Write your reply")[0];
  const options: { value: "steering" | "prediction"; label: string }[] = [{ value: "steering", label: "Steer" }, { value: "prediction", label: "Inspect" }];
  const colorOptions: { value: "surprisal" | "probability"; label: string }[] = [{ value: "surprisal", label: "Surprisal" }, { value: "probability", label: "Probability" }];
  const run = $derived(recording.runs[setting]);
  const tokens = $derived(run.tokens.filter(token => token.text.length > 0));
  // At negative strengths the recorded replies stay close to the unsteered one, so the ends
  // describe the direction of the influence rather than promising a detached reply.
  const directionLabel = (alpha: number) => alpha === 0 ? "Unsteered" : alpha < 0 ? "Less welcoming" : "More welcoming";
  const valueText = $derived(`${directionLabel(run.alpha)}, strength ${run.alpha.toFixed(2)}`);
  const modelNames: Record<string, string> = { "gemma3-4b-instruct": "Gemma 3 4B" };
  const recordedOn = new Intl.DateTimeFormat("en", { dateStyle: "long", timeZone: "UTC" }).format(new Date(recording.recordedAt));
  const provenance = `Recorded with ${modelNames[recording.modelId] ?? recording.modelId} in the Drowse browser runtime on ${recordedOn} · seed ${recording.sampling.seed} · temperature ${recording.sampling.temperature}. Real output, not edited.`;
  let announcement = $state("");
  // Announce once when a new setting is committed, not on every drag tick: drowse's Slider fires
  // change for a keyboard step and at the end of a drag that moved it.
  function announce() {
    announcement = `${valueText}. Reply: ${run.text.trim()}`;
  }

  function fitContent(node: HTMLElement) {
    const panel = node.parentElement!;
    const measure = () => { panel.style.height = `${node.getBoundingClientRect().height}px`; };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    return { destroy() { observer.disconnect(); } };
  }

</script>

<section class="capability-demo" aria-label="Recorded Drowse example">
  <div class="demo-content" use:fitContent>
  <header>
    <Segmented {options} value={mode} onchange={value => mode = value} label="Example capability" />
  </header>
  <p class="context">{promptPreview}</p>
    <div class="alpha">
      <span>Welcoming influence</span>
      <span class="strength"><MorphText text={`${directionLabel(run.alpha)} · Strength (α) ${run.alpha.toFixed(2)}`} />{#if run.expression}<code class="expression">{run.expression}</code>{/if}</span>
    </div>
    <div class="slider-field" onchange={announce}>
      <Slider value={setting} min={0} max={recording.runs.length - 1} step={1} oninput={value => setting = value}
        ariaLabel="Welcoming influence" displayValue={valueText} />
    </div>
    <div class="scale-labels" aria-hidden="true"><span>Less welcoming</span><span>Unsteered</span><span>More welcoming</span></div>
    <p class="sr-only" aria-live="polite">{announcement}</p>
  <div class="mode-stage">
  {#key mode}
  <div class="mode-content" in:rise out:fadeOut>
    {#if mode === "steering"}
      <p class="result"><DemoReply text={run.text} /></p>
    {:else}
      <div class="inspect-toolbar">
        <p class="inspect-hint">Hover or click a token to inspect.</p>
        <Segmented options={colorOptions} value={coloring} onchange={value => coloring = value} label="Token coloring" />
      </div>
      <div class="result token-text">
        <RecordedTokenText {tokens} layout="inline" {coloring} />
      </div>
    {/if}
  </div>
  {/key}
  </div>
  <p class="provenance">{provenance}</p>
  </div>
</section>

<style>
  /* A recorded run of the browser runtime: steer the reply with the slider, or inspect its tokens. */
  .capability-demo { min-width: 0; color: var(--ink2); font: var(--type-body); transition: height var(--glide) var(--ease); }
  .demo-content { display: flow-root; }
  .mode-stage { position: relative; display: grid; }
  .mode-stage > .mode-content { grid-area: 1 / 1; }
  .mode-content { display: flow-root; min-width: 0; }
  header { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: var(--s4); margin-bottom: var(--s5); }
  .context { max-width: 70ch; margin: 0 0 var(--s5); text-wrap: pretty; }
  .alpha, .scale-labels { display: flex; justify-content: space-between; gap: var(--s4); font-variant-numeric: tabular-nums; }
  .alpha { flex-wrap: wrap; align-items: baseline; row-gap: var(--s2); margin-bottom: var(--s3); color: var(--ink); font: var(--type-control); -webkit-user-select: none; user-select: none; }
  .alpha > span { min-width: 0; max-width: 100%; overflow-wrap: anywhere; }
  .alpha :global(.morph-paint) { overflow: visible; }
  .scale-labels, .inspect-hint { color: var(--ink3); font: var(--type-meta); -webkit-user-select: none; user-select: none; }
  .strength { display: inline-flex; flex-wrap: wrap; align-items: baseline; justify-content: end; gap: var(--s1) var(--s2); }
  .expression { padding: 0 var(--s1); border-radius: var(--r-chip); background: var(--wash); color: var(--ink); font: var(--type-meta); overflow-wrap: anywhere; }
  .provenance { max-width: 70ch; margin-top: var(--s5); color: var(--ink3); font: var(--type-meta); text-wrap: pretty; }
  .result { max-width: 70ch; margin: var(--s5) 0 0; color: var(--ink); font: var(--type-lead); text-wrap: pretty; overflow-wrap: anywhere; white-space: pre-wrap; }
  .inspect-toolbar { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: var(--s2); margin-top: var(--s5); }
  .token-text { margin-block: var(--s4) var(--s5); white-space: normal; }
  @media (max-width: 720px) { header :global(.seg) { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); width: 100%; } }
</style>
