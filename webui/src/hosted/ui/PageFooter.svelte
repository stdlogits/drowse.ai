<script lang="ts">
  import { CONTACT_ADDRESS } from "../../../../drowse/webui/src/lib/contact";
  import { Icon } from "@stdlogits/ledger";

  type Current = "home" | "credits" | "contact" | "learn" | "privacy" | "about" | "contribute";
  let { current, hosted = true, compact = false }: { current?: Current; hosted?: boolean; compact?: boolean } = $props();

  const releaseBuild = typeof __DROWSE_HOSTED_RELEASE__ !== "undefined" && __DROWSE_HOSTED_RELEASE__;
  const sourceRevision = typeof __DROWSE_SOURCE_REVISION__ === "string" ? __DROWSE_SOURCE_REVISION__ : "preview";
  // AGPL section 13 source offer: release builds link the exact revision they were built from.
  const sourceUrl = typeof __DROWSE_SOURCE_URL__ === "string" ? __DROWSE_SOURCE_URL__ : "https://github.com/stdlogits/drowse.ai";
  const version = typeof __DROWSE_VERSION__ === "string" ? __DROWSE_VERSION__ : "";
  const repositoryUrl = "https://github.com/stdlogits/drowse";

  const site = $derived(hosted ? "" : "https://drowse.ai");
  type FooterLink = { label: string; href: string; page?: Current; detail?: string; external?: boolean };
  const groups: { title: string; links: FooterLink[] }[] = $derived([
    { title: "Drowse", links: [
      { label: "Home", href: `${site}/`, page: "home" },
      { label: "Open Drowse", href: `${site}/app` },
      { label: "Models", href: `${site}/app?choose=1` },
      { label: "Learn", href: `${site}/learn`, page: "learn" },
    ] },
    { title: "Project", links: [
      { label: "Credits", href: `${site}/credits`, page: "credits" },
      { label: "Contribute", href: `${site}/contribute`, page: "contribute" },
      { label: "Changelog", href: `${repositoryUrl}/releases`, external: true },
    ] },
    { title: "Trust", links: [
      { label: "Privacy", href: `${site}/privacy`, page: "privacy" },
      { label: "License", href: `${site}/LICENSE`, detail: "GNU AGPL v3 or later" },
      { label: "Third-party licenses", href: `${site}/third-party-licenses` },
      { label: "Security", href: `${repositoryUrl}/security/policy`, external: true },
    ] },
    { title: "Contact", links: [
      { label: "Contact us", href: `${site}/contact`, page: "contact" },
      { label: "Email us", href: `mailto:${CONTACT_ADDRESS}` },
    ] },
  ]);
</script>

<!-- The site's footer: the pages in four quiet groups, then the version and the source offer. -->
<footer class="page-footer" class:compact>
  <nav class="footer-groups" aria-label="Footer">
    {#each groups as group (group.title)}
      <div class="footer-group">
        <h2 class="head">{group.title}</h2>
        <ul>
          {#each group.links as link (link.label)}
            <li>
              <a href={link.href} aria-current={link.page && current === link.page ? "page" : undefined} rel={link.external ? "noreferrer" : undefined}>
                {link.label}{#if link.detail}<span class="link-detail">{link.detail}</span>{/if}
                {#if link.external}<Icon name="external" size={14} /><span class="sr-only">(opens GitHub)</span>{/if}
              </a>
            </li>
          {/each}
        </ul>
      </div>
    {/each}
  </nav>
  <div class="project-info">
    <span translate="no">Drowse{version ? ` ${version}` : ""}</span>
    {#if releaseBuild}<span class="build num" {...{ "aria-description": sourceRevision }}>Build {sourceRevision.slice(0, 12)}</span>{/if}
    <a href={sourceUrl} rel="noreferrer">
      {releaseBuild ? "Source for this build" : "Source code"}
      <Icon name="external" size={14} /><span class="sr-only">(opens GitHub)</span>
    </a>
  </div>
</footer>

<style>
  .page-footer { display: grid; gap: var(--s6); width: 100%; max-width: var(--page); margin: 0 auto; padding: var(--s7) var(--gutter) max(var(--s7), env(safe-area-inset-bottom)); color: var(--ink3); font: var(--type-small); }
  .footer-groups { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: var(--s5) var(--s6); }
  .footer-group { display: grid; align-content: start; gap: var(--s3); min-width: 0; }
  ul { display: grid; gap: var(--s2); }
  a { display: inline-flex; flex-wrap: wrap; align-items: center; gap: 0 var(--s1); color: var(--ink2); transition: color var(--fast) var(--ease); }
  a[aria-current="page"] { color: var(--ink); }
  .link-detail { flex-basis: 100%; color: var(--ink3); font: var(--type-meta); }
  .project-info { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s2) var(--s5); }
  .project-info a { color: var(--ink3); }
  @media (hover: hover) { a:hover, .project-info a:hover { color: var(--ink); } }
  .compact { gap: var(--s4); padding-top: var(--s5); }
  .compact .footer-groups { gap: var(--s4) var(--s5); }
  @media (max-width: 720px) {
    .footer-groups { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    a { min-height: 32px; }
  }
</style>
