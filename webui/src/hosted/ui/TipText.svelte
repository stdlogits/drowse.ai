<script lang="ts">
  // A sentence that is still needed, kept in Ledger's tip: a quiet label that shows it on hover and
  // on focus. The page root carries use:tips.
  let { label, text }: { label: string; text: string } = $props();
</script>

<!-- svelte-ignore a11y_no_noninteractive_tabindex (the tip opens on focus, so a keyboard reaches it) -->
<span class="tip-text" tabindex="0" role="note" data-tip={text}>{label}</span>

<style>
  .tip-text { color: var(--ink3); font: var(--type-meta); text-decoration: underline; text-decoration-color: var(--wash2); text-decoration-thickness: 2px; text-underline-offset: 4px; cursor: default; transition: color var(--fast) var(--ease), text-decoration-color var(--fast) var(--ease); }
  .tip-text:focus-visible { color: var(--ink); }
  @media (hover: hover) { .tip-text:hover { color: var(--ink); text-decoration-color: var(--ink3); } }
</style>
