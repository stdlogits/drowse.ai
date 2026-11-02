<script lang="ts">
  import { Icon } from "@stdlogits/ledger";
  import SitePage from "./SitePage.svelte";

  const groups = [
    {
      id: "drowse", title: "Drowse",
      body: "Drowse is free software under the GNU Affero General Public License, version 3 or later.",
      links: [{ label: "GNU AGPL v3 or later", href: "/LICENSE" }],
    },
    {
      id: "software", title: "Open-source software",
      body: "The npm packages whose code ships in this build, with each package's license and notice files.",
      links: [{ label: "Third-party software notices", href: "/third-party-licenses.txt" }],
    },
    {
      id: "fonts", title: "Type and motion",
      body: "Wix Madefor is under the SIL Open Font License 1.1. Torph, which morphs text that changes in place, is under the MIT License.",
      links: [
        { label: "Wix Madefor", href: "/LICENSE-Wix-Madefor.txt" },
        { label: "Torph", href: "/LICENSE-Torph.txt" },
      ],
    },
    {
      id: "models", title: "Models and packs",
      body: "Models and instrument packs are downloaded separately and are covered by their own licenses.",
      links: [],
    },
  ];
</script>

<SitePage title="Third-party licenses · Drowse" heading="Licenses." lede="The open-source work that Drowse is built on, and the terms it ships under.">
  {#each groups as group}
    <section class="license-group" aria-labelledby={`licenses-${group.id}`}>
      <div class="copy">
        <h2 id={`licenses-${group.id}`}>{group.title}</h2>
        <p>{group.body}</p>
      </div>
      {#if group.links.length}
        <ul>
          {#each group.links as link}
            <li><a href={link.href} type="text/plain"><span>{link.label}</span><Icon name="right" /></a></li>
          {/each}
        </ul>
      {/if}
    </section>
  {/each}
</SitePage>

<style>
  /* Each group a caption beside its licence files, as rows of wash. */
  .license-group { display: grid; grid-template-columns: minmax(0, 300px) minmax(0, 1fr); gap: var(--s4) var(--s7); align-items: start; }
  .copy { display: grid; gap: var(--s2); }
  h2 { color: var(--ink); font: var(--type-label); }
  p { max-width: 60ch; color: var(--ink2); font: var(--type-small); }
  ul { display: grid; gap: var(--s2); }
  a { display: flex; align-items: center; justify-content: space-between; gap: var(--s4); min-height: 48px; padding: var(--s3) var(--s4); border-radius: var(--r-box); background: var(--wash); color: var(--ink); font: var(--type-control); transition: background-color var(--fast) var(--ease), scale var(--fast) var(--ease); }
  a:active { scale: 0.99; }
  a > :global(svg) { flex: none; color: var(--ink3); }
  @media (hover: hover) { a:hover { background: var(--wash2); } }
  @media (max-width: 900px) { .license-group { grid-template-columns: minmax(0, 1fr); } }
</style>
