<script lang="ts">
  import type { Snippet } from "svelte";
  import FluentIcon from "../../../../../drowse/webui/src/lib/ui/FluentIcon.svelte";
  import MorphText from "../../../../../drowse/webui/src/lib/ui/MorphText.svelte";
  import InfoTip from "../../../../../drowse/webui/src/lib/ui/InfoTip.svelte";
  import TranslateSwitch from "./TranslateSwitch.svelte";

  // The top of a neuron or head: its ID and tags, stepping through the layer's list, and the translate switch.
  let { id, caution, cautionLabel = "Why this depth is harder to read", unitName, onstep, position, total, hrefFor, tags }: {
    id: string;
    caution?: string;
    cautionLabel?: string;
    unitName: string;
    /** Move to the previous (-1) or next (+1) unit in the layer's list. */
    onstep: (delta: -1 | 1) => void;
    position: number;
    total: number;
    hrefFor: (delta: -1 | 1) => string;
    tags?: Snippet;
  } = $props();

  let copied = $state(false);
  async function copyLink() {
    try {
      await navigator.clipboard.writeText(window.location.href);
      copied = true;
      setTimeout(() => (copied = false), 1600);
    } catch { /* clipboard blocked: the address bar still has the link */ }
  }

  function step(event: MouseEvent, delta: -1 | 1) {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    onstep(delta);
  }
</script>

<div class="detail-bar">
  <div class="id-row">
    <h2 id="unit-title" tabindex="-1"><MorphText text={id} duration={340} /></h2>
    {#if caution}<span class="caution"><InfoTip icon="info" label={cautionLabel} text={caution} /></span>{/if}
    {@render tags?.()}
  </div>
  <div class="tools">
    <TranslateSwitch />
    <div class="stepper">
      <span class="position" aria-live="polite"><MorphText text={`${(position + 1).toLocaleString("en-US")} / ${total.toLocaleString("en-US")}`} duration={240} /></span>
      <a class="icon-button" href={hrefFor(-1)} aria-label={`Previous ${unitName}`} aria-keyshortcuts="k" onclick={(event) => step(event, -1)}><FluentIcon name="up" size={18} /></a>
      <a class="icon-button" href={hrefFor(1)} aria-label={`Next ${unitName}`} aria-keyshortcuts="j" onclick={(event) => step(event, 1)}><FluentIcon name="down" size={18} /></a>
      <button type="button" class="icon-button" aria-label={copied ? "Link copied" : `Copy link to this ${unitName}`} onclick={copyLink}>
        <span class="swap-icon" class:copied><span class="i-copy"><FluentIcon name="copy" size={18} /></span><span class="i-check"><FluentIcon name="check" size={18} /></span></span>
      </button>
    </div>
  </div>
</div>

<style>
  .detail-bar { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: var(--space-3) var(--space-5); }
  .id-row { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-2) var(--space-3); min-width: 0; }
  h2 { margin: 0; color: var(--fg); font-family: var(--font-mono); font-weight: var(--weight-data-medium); font-size: var(--text-feature-title); line-height: 1.15; letter-spacing: -0.01em; font-variant-numeric: tabular-nums; }
  h2:focus { outline: none; }
  .caution { display: inline-flex; color: var(--fg-muted); }
  .tools { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-2) var(--space-5); }
  .stepper { display: flex; align-items: center; gap: var(--space-2); }
  .position { margin-inline-end: var(--space-2); color: var(--fg-muted); font-size: var(--text-xs); font-variant-numeric: tabular-nums; }
  .icon-button {
    position: relative; display: inline-grid; place-items: center; width: 36px; height: 36px; border: 0; border-radius: var(--radius);
    background: var(--glass); color: var(--fg-dim); font: inherit; text-decoration: none; cursor: pointer;
    transition: background-color var(--dur-fast) var(--ease-out), color var(--dur-fast) var(--ease-out), scale var(--dur-fast) var(--ease-out);
  }
  .icon-button::after { content: ""; position: absolute; inset: -4px; }
  .icon-button:hover { background: var(--glass-strong); color: var(--fg); }
  .icon-button:active { scale: var(--press-scale); }
  .icon-button:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  .swap-icon { display: grid; }
  .swap-icon > span { grid-area: 1 / 1; display: grid; transition: opacity var(--dur-slow) cubic-bezier(0.2, 0, 0, 1), scale var(--dur-slow) cubic-bezier(0.2, 0, 0, 1), filter var(--dur-slow) cubic-bezier(0.2, 0, 0, 1); }
  .i-check, .copied .i-copy { opacity: 0; scale: 0.25; filter: blur(4px); }
  .copied .i-check { opacity: 1; scale: 1; filter: blur(0); }
  @media (prefers-reduced-motion: reduce) { .icon-button { transition: none; } .icon-button:active { scale: none; } .swap-icon > span { transition: opacity var(--dur-fast) linear; } .i-check, .copied .i-copy, .copied .i-check { scale: 1; filter: none; } }
</style>
