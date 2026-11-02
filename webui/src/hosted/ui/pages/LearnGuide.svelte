<script lang="ts">
  import SitePage from "./SitePage.svelte";
  import GuideContent from "./GuideContent.svelte";
  import { htmlGuides } from "../../../../scripts/site-content.mjs";

  let { path }: { path: string } = $props();
  const page = $derived(htmlGuides.find(guide => guide.path === path && path.startsWith("/learn/"))!);
</script>

<SitePage current="learn" title={`${page.heading} · Drowse`} heading={page.heading} lede={page.intro} markdown={`${page.path}.md`}>
  <nav class="breadcrumb" aria-label="Breadcrumb"><a href="/learn">Learn</a></nav>
  <GuideContent {page} <style>
  .breadcrumb a { color: var(--ink3); font: var(--type-control); transition: color var(--fast) var(--ease); }
  @media (hover: hover) { .breadcrumb a:hover { color: var(--ink); } }
</style>
