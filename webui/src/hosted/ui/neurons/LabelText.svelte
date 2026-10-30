<script lang="ts">
  import MorphText from "../../../../../drowse/webui/src/lib/ui/MorphText.svelte";
  import { FOREIGN_RUN, inlineEnglish } from "./translate.svelte";

  // Prose such as a Haiku label, whose quoted examples may be in another language. Those runs are
  // terms: hover or focus shows their translation, and the switch morphs them into English inline.
  // Cards already wrapped in links keep plain spans; standalone text can focus each term.
  let { text, fallback = "Unclear", interactive = false }: { text: string; fallback?: string; interactive?: boolean } = $props();

  // Punctuation hugging a term stays on its line: a button is an atomic inline even when styled
  // inline, so without this a comma after a focusable term can wrap onto a line of its own.
  const OPEN = /[(\[{«「『《“‘"']+$/u;
  const CLOSE = /^[)\]}»」』》”’"'.,;:!?、。，；：！？…]+/u;

  const parts = $derived.by(() => {
    const source = text || fallback;
    const out: { text: string; term: boolean; lead?: string; tail?: string }[] = [];
    let last = 0;
    for (const match of source.matchAll(FOREIGN_RUN)) {
      const at = match.index ?? 0;
      const before = source.slice(last, at);
      const lead = before.match(OPEN)?.[0] ?? "";
      if (before.length > lead.length) out.push({ text: before.slice(0, before.length - lead.length), term: false });
      const end = at + match[0].length;
      const tail = source.slice(end).match(CLOSE)?.[0] ?? "";
      out.push({ text: match[0], term: true, lead, tail });
      last = end + tail.length;
    }
    if (last < source.length) out.push({ text: source.slice(last), term: false });
    return out;
  });
</script>

<span class="label-text">{#each parts as part, i (i)}{#if part.term}{@const english = inlineEnglish(part.text)}<span class="glue">{part.lead}<svelte:element this={interactive ? "button" : "span"} type={interactive ? "button" : undefined} class="term" class:translated={english !== ""} data-term={part.text} lang={english ? "en" : undefined}><MorphText text={english || part.text} numbers={false} duration={260} />{#if english}<span class="sr-only">{` (translated from ${part.text})`}</span>{/if}</svelte:element>{part.tail}</span>{:else}{part.text}{/if}{/each}</span>

<style>
  .label-text { overflow-wrap: anywhere; }
  /* Inline so a long translated term wraps with the text instead of overflowing its box. */
  .label-text :global(.morph-text) { display: inline; }
  .glue { white-space: nowrap; }
  .term { display: inline; white-space: normal; margin: 0; padding: 0; border: 0; border-radius: var(--radius-sm); background: transparent; color: inherit; font: inherit; text-align: inherit; text-decoration: underline dotted color-mix(in srgb, currentColor 45%, transparent); text-underline-offset: 0.2em; cursor: help; }
  button.term:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  .translated { color: var(--neuron-translated-ink, var(--accent-amber)); font-style: italic; text-decoration: none; }
  .sr-only { position: absolute; inset-inline-start: 0; top: 0; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
</style>
