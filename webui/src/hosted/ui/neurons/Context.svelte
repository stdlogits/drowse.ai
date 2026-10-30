<script lang="ts">
  import { segments, type Context } from "./data";
  import { languageName, translation } from "./translate.svelte";

  // One top firing: ⟦ ⟧ marks the token where the unit fired, ⟪ ⟫ the token an attention head read
  // from. With the switch on, a translated context shows its English; the badge shows the original.
  let { context, clip = true }: { context: Context; clip?: boolean } = $props();

  const english = $derived(translation.on && context[1] ? context[1] : "");
  const parts = $derived.by(() => {
    const list = segments(english || context[0]).map((part) => ({ ...part }));
    if (!clip || list.length < 2) return list;
    const first = list[0];
    const last = list[list.length - 1];
    if (first.mark === null && first.text.length > 64) first.text = `…${first.text.slice(-64)}`;
    if (last.mark === null && last.text.length > 40) last.text = `${last.text.slice(0, 40)}…`;
    return list;
  });
</script>

<p class="ctx" class:translated={english !== ""}>{#each parts as part, i (i)}{#if part.mark}<mark class={part.mark}>{part.text || "␣"}</mark>{:else}{part.text}{/if}{/each}{#if english}<button type="button" class="lang" data-original={context[0]} data-lang={context[2]}>{languageName(context[2] ?? "")}<span class="sr-only">: translated by Claude Haiku. Original: {context[0]}</span></button>{/if}</p>

<style>
  .ctx { margin: 0; padding: var(--space-3) var(--space-4); border-radius: var(--radius); background: var(--neuron-well); color: var(--fg-dim); font-family: var(--font-mono); font-size: var(--text-sm); line-height: 1.7; overflow-wrap: anywhere; white-space: pre-wrap; }
  .ctx.translated { font-family: var(--font-reading); box-shadow: inset 2px 0 0 color-mix(in srgb, var(--accent-amber) 70%, transparent); }
  /* The highlight grows with a shadow, not padding, so the token stays joined to its word. */
  mark { border-radius: var(--radius-sm); color: var(--fg); }
  mark.hit { background: color-mix(in oklch, var(--neuron-tint) 34%, transparent); box-shadow: 0 0 0 2px color-mix(in oklch, var(--neuron-tint) 34%, transparent); }
  mark.src { background: transparent; box-shadow: 0 0 0 1.5px color-mix(in oklch, var(--neuron-tint) 80%, transparent); }
  .lang {
    display: inline-flex; align-items: center; margin-inline-start: var(--space-2); padding: 0 var(--space-2); border: 0; border-radius: var(--radius-pill);
    background: color-mix(in srgb, var(--accent-amber) 16%, transparent); color: var(--neuron-translated-ink, var(--accent-amber));
    font-family: var(--font-reading); font-size: var(--text-2xs); line-height: 1.7; white-space: nowrap; vertical-align: 0.1em; cursor: help;
  }
  .lang:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  .sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
</style>
