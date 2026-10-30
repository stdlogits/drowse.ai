<script lang="ts">
  import LabelText from "./LabelText.svelte";
  import Token from "./Token.svelte";
  import { confidenceWord, type LensToken } from "./data";

  // One of Haiku's labels with its confidence, and the raw tokens it rests on beneath it.
  let { label, confidence, tokens, source, sourceHelp }: {
    label: string;
    confidence: number;
    tokens: LensToken[];
    /** Where the raw tokens come from: a lens read's short name, or "in real text". */
    source: string;
    sourceHelp?: string;
  } = $props();
</script>

<div class="cell" class:low={confidence < 0.25}>
  <p class="label"><LabelText text={label} interactive /></p>
  <p class="conf"><span class="meter" aria-hidden="true"><span style:--c={Math.max(confidence, 0.04)}></span></span>{confidenceWord(confidence)}<span class="pct">{Math.round(confidence * 100)}%</span></p>
  {#if tokens.length}
    <p class="raw"><span class="source" title={sourceHelp}>{source}</span>{#each tokens as [token, count, flags]}<Token {token} {count} {flags} size="sm" />{/each}</p>
  {/if}
</div>

<style>
  .cell { display: grid; align-content: start; gap: var(--space-2); min-width: 0; }
  .label { margin: 0; color: var(--fg); font-family: var(--font-structure); font-size: var(--text-md); font-weight: var(--weight-structure); line-height: 1.4; text-wrap: pretty; }
  .low .label { color: var(--fg-dim); }
  .conf { display: flex; align-items: center; gap: var(--space-2); margin: 0; color: var(--fg-muted); font-size: var(--text-xs); }
  .pct { font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
  .meter { position: relative; display: inline-block; width: 40px; height: 4px; border-radius: var(--radius-pill); background: var(--glass-strong); overflow: hidden; }
  .meter span { position: absolute; inset: 0; border-radius: inherit; background: var(--neuron-tint); transform-origin: left; scale: var(--c) 1; transition: scale var(--dur-slow) var(--ease-out); }
  .raw { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-1); margin: var(--space-1) 0 0; min-width: 0; }
  .source { margin-inline-end: var(--space-1); color: var(--fg-muted); font-family: var(--font-mono); font-size: var(--text-2xs); white-space: nowrap; }
  @media (prefers-reduced-motion: reduce) { .meter span { transition: none; } }
</style>
