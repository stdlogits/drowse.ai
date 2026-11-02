<script lang="ts">
  import type { Snippet } from "svelte";
  import { tips } from "@stdlogits/ledger";
  import SiteHeader from "../SiteHeader.svelte";
  import PageFooter from "../PageFooter.svelte";
  import PwaUpdatePrompt from "../PwaUpdatePrompt.svelte";

  // A reading page (About, Learn, Privacy, Contribute, the licences): a title and a lead, then the
  // page's parts, as neuron-lens.com draws its own.
  let { title, heading, lede, children, markdown, current }: {
    title: string;
    heading: string;
    lede: string;
    children: Snippet;
    /** Path of the page's Markdown twin, when it has one. */
    markdown?: string;
    current?: "learn" | "privacy" | "about" | "contribute";
  } = $props();

  $effect(() => { document.title = title; });
</script>

<PwaUpdatePrompt />

<div class="site-page" use:tips>
  <a class="skip-link" href="#site-page-main">Skip to content</a>
  <SiteHeader {current} />
  <main id="site-page-main" tabindex="-1">
    <header class="page-intro">
      <h1>{heading}</h1>
      <p class="lede">{lede}</p>
    </header>
    {@render children()}
    {#if markdown}<p class="alternate"><a href={markdown} type="text/markdown">Read this page as Markdown</a></p>{/if}
  </main>
  <PageFooter {current} />
</div>

<style>
  .site-page { display: flex; flex-direction: column; min-height: 100svh; overflow-x: clip; }
  .skip-link { position: fixed; z-index: 60; top: var(--s2); inset-inline-start: var(--s2); padding: var(--s2) var(--s4); border-radius: var(--r-pill); background: var(--ink); color: var(--paper); font: var(--type-control); translate: 0 calc(-100% - var(--s4)); transition: translate var(--fast) var(--ease); }
  .skip-link:focus { translate: 0 0; }
  main { flex: 1; display: grid; grid-template-columns: minmax(0, 1fr); align-content: start; gap: var(--s8); width: 100%; max-width: var(--page); margin: 0 auto; padding: var(--s7) var(--gutter) var(--s9); }
  main:focus { outline: none; }
  .page-intro { display: grid; justify-items: start; gap: var(--s5); max-width: 52rem; }
  h1 { color: var(--ink); font: var(--type-title); letter-spacing: -0.04em; }
  .lede { max-width: 60ch; color: var(--ink2); font: var(--type-lead); }
  .alternate a { color: var(--ink3); font: var(--type-small); text-decoration: underline; text-decoration-color: var(--wash2); text-decoration-thickness: 2px; text-underline-offset: 4px; transition: color var(--fast) var(--ease); }
  @media (hover: hover) { .alternate a:hover { color: var(--ink); } }
  @media (max-width: 720px) { main { gap: var(--s7); padding-top: var(--s5); } }
</style>
