<script lang="ts">
  import { Icon } from "@stdlogits/ledger";
  import { htmlGuidePaths, type GuidePage, type GuideLink } from "../../../../scripts/site-content.mjs";

  let { page }: { page: GuidePage } = $props();

  // Internal guides without an HTML page (such as /developers) link to their Markdown resource.
  const htmlRoutes = new Set(["/", "/app", "/credits", "/contact", "/contribute", "/third-party-licenses", ...htmlGuidePaths]);
  const internalHref = (href: string) => {
    const path = href.split(/[?#]/, 1)[0];
    return htmlRoutes.has(path) ? href : `${path}.md`;
  };
  const isExternal = (link: GuideLink) => !link.href.startsWith("/");
  const slug = (text: string) => text.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
</script>

{#each page.sections as section}
  <section class="guide-section" aria-labelledby={`section-${slug(section.heading)}`}>
    <h2 id={`section-${slug(section.heading)}`}>{section.heading}</h2>
    <div class="section-body">
      {#each section.paragraphs ?? [] as paragraph}<p>{paragraph}</p>{/each}
      {#if section.bullets}
        <ul class="bullets">{#each section.bullets as bullet}<li>{bullet}</li>{/each}</ul>
      {/if}
      {#if section.code}<pre class="code"><code>{section.code}</code></pre>{/if}
      {#if section.links?.length}
        <ul class="links">
          {#each section.links as link}
            <li>
              {#if isExternal(link)}
                <a href={link.href} rel="noreferrer"><span>{link.label}</span><Icon name="external" /></a>
              {:else}
                <a href={internalHref(link.href)}><span>{link.label}</span><Icon name="right" /></a>
              {/if}
            </li>
          {/each}
        </ul>
      {/if}
    </div>
  </section>
{/each}

<style>
  /* A guide's parts: each a caption beside what it is about; captions stay in view while their part
     scrolls past on wide screens (neuron-lens.com's reading pages). */
  .guide-section { display: grid; grid-template-columns: minmax(0, 300px) minmax(0, 1fr); gap: var(--s5) var(--s7); align-items: start; min-width: 0; }
  h2 { position: sticky; top: var(--s5); color: var(--ink); font: var(--type-heading); letter-spacing: -0.02em; }
  .section-body { display: grid; gap: var(--s5); min-width: 0; }
  p, li { color: var(--ink2); font: var(--type-body); }
  p { max-width: 68ch; }
  .bullets { display: grid; gap: var(--s2); max-width: 68ch; padding-inline-start: var(--s5); list-style: disc; }
  .code { overflow-x: auto; margin: 0; padding: var(--s4) var(--s5); border-radius: var(--r-box); background: var(--wash); color: var(--ink); font: var(--type-small); font-variant-numeric: tabular-nums; }
  .links { display: grid; gap: var(--s2); }
  .links a { display: flex; align-items: center; justify-content: space-between; gap: var(--s4); min-height: 48px; padding: var(--s3) var(--s4); border-radius: var(--r-box); background: var(--wash); color: var(--ink); font: var(--type-control); overflow-wrap: anywhere; transition: background-color var(--fast) var(--ease), scale var(--fast) var(--ease); }
  .links a:active { scale: 0.99; }
  .links a > :global(svg) { flex: none; color: var(--ink3); }
  @media (hover: hover) { .links a:hover { background: var(--wash2); } }
  @media (max-width: 900px) {
    .guide-section { grid-template-columns: minmax(0, 1fr); }
    h2 { position: static; }
  }
</style>
