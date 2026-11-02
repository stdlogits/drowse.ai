<script lang="ts">
  import SitePage from "./SitePage.svelte";
  import GuideContent from "./GuideContent.svelte";
  import { Icon } from "@stdlogits/ledger";
  import { htmlGuides } from "../../../../scripts/site-content.mjs";

  const page = htmlGuides.find(guide => guide.path === "/learn")!;
  const guides = htmlGuides.filter(guide => guide.path.startsWith("/learn/"));
</script>

<SitePage current="learn" title="Learn · Drowse" heading={page.heading} lede={page.intro} markdown="/learn.md">
  <section class="guides" aria-labelledby="guides-title">
    <h2 id="guides-title">Guides</h2>
    <ul>
      {#each guides as guide}
        <li>
          <a href={guide.path}>
            <span class="guide-title">{guide.heading}</span>
            <span class="guide-summary">{guide.description}</span>
            <Icon name="right" />
          </a>
        </li>
      {/each}
    </ul>
  </section>
  <GuideContent {page} />
</SitePage>

<style>
  /* The guides, as rows of wash: a title and its one line. */
  .guides { display: grid; gap: var(--s4); }
  h2 { color: var(--ink3); font: var(--type-head); }
  ul { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--s2); }
  a { position: relative; display: grid; align-content: start; gap: var(--s2); height: 100%; padding: var(--s5) var(--s7) var(--s5) var(--s5); border-radius: var(--r-row); background: var(--wash); color: var(--ink); transition: background-color var(--fast) var(--ease), scale var(--fast) var(--ease); }
  a:active { scale: 0.99; }
  a > :global(svg) { position: absolute; top: var(--s5); right: var(--s5); color: var(--ink3); }
  .guide-title { font: var(--type-label); text-wrap: balance; }
  .guide-summary { color: var(--ink2); font: var(--type-small); text-wrap: pretty; }
  @media (hover: hover) { a:hover { background: var(--wash2); } }
  @media (max-width: 720px) { ul { grid-template-columns: minmax(0, 1fr); } }
</style>
