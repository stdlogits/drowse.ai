<script lang="ts">
  import { Icon } from "@stdlogits/ledger";
  import SitePage from "./SitePage.svelte";
  import { contributeLinks } from "../../../../scripts/site-content.mjs";

  const version = typeof __DROWSE_VERSION__ === "string" ? __DROWSE_VERSION__ : "unknown";
  const revision = typeof __DROWSE_ARTIFACT_SOURCE_REVISION__ === "string" ? __DROWSE_ARTIFACT_SOURCE_REVISION__ : "";
  const channel = typeof __DROWSE_HOSTED_CHANNEL__ === "string" ? __DROWSE_HOSTED_CHANNEL__ : "preview";
  const sourceUrl = revision ? `${contributeLinks.repository}/tree/${revision}` : contributeLinks.repository;
  // Only build identity is prefilled. Prompts, chats, and model output stay out of the report.
  const issueBody = [
    "**What happened?**", "", "", "**Steps to reproduce**", "1. ", "", "**What did you expect?**", "", "",
    "**Environment**",
    `- Drowse ${version} (${channel}${revision ? `, build ${revision.slice(0, 12)}` : ""})`,
    "- Browser and device: ",
    "- Model: ",
    "", "_Please leave out private prompts, chats, and personal data._",
  ].join("\n");
  const issueUrl = `${contributeLinks.newIssue}?body=${encodeURIComponent(issueBody)}`;

  const ways = [
    { id: "bug", title: "Report a bug", body: "Tell us what happened and how to reproduce it. The report fills in this build's version for you; leave out private prompts and chats.", links: [{ label: "Open an issue", href: issueUrl }] },
    { id: "code", title: "Fix or improve code", body: "Read the contributing guide, then pick an issue that fits. Small, focused changes are easiest to review.", links: [{ label: "Contributing guide", href: contributeLinks.contributing }, { label: "Good first issues", href: contributeLinks.goodFirstIssues }] },
    { id: "model", title: "Add a model or pack", body: "Hosted models ship only with a verified artifact closure and tests on real devices. Start with the browser runtime guide.", links: [{ label: "Browser runtime guide", href: contributeLinks.hostedModels }] },
    { id: "teach", title: "Study or teach with it", body: "Using Drowse in a course, workshop, or study? Tell us what you need.", links: [{ label: "Contact us about teaching", href: "/contact?reason=teaching" }] },
    { id: "support", title: "Support the project", body: "Ask about funding and sponsorship through the contact form.", links: [{ label: "Contact us about funding", href: "/contact?reason=funding" }] },
    { id: "security", title: "Report a security issue privately", body: "Please don't open a public issue for a vulnerability. Follow the security policy instead.", links: [{ label: "Security policy", href: contributeLinks.securityPolicy }] },
  ];
</script>

<SitePage current="contribute" title="Contribute · Drowse" heading="Help build Drowse." lede="Drowse is open source. Here are the ways you can help.">
  <ul class="ways">
    {#each ways as way}
      <li class="way">
        <h2 id={`way-${way.id}`}>{way.title}</h2>
        <p>{way.body}</p>
        <div class="actions">
          {#each way.links as link}
            {@const external = !link.href.startsWith("/")}
            <a href={link.href} rel={external ? "noreferrer" : undefined}>
              <span>{link.label}</span><Icon name={external ? "external" : "right"} />
            </a>
          {/each}
        </div>
      </li>
    {/each}
  </ul>

  <section class="source" aria-labelledby="source-title">
    <h2 id="source-title">Source for this build</h2>
    <p><code>{revision ? revision.slice(0, 12) : "unknown"}</code> · Drowse {version}</p>
    <a class="source-action" href={sourceUrl} rel="noreferrer">View on GitHub</a>
  </section>
</SitePage>

<style>
  /* Ways to help, each a wash with its one or two ways in; then the source for this build. */
  .ways { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 320px), 1fr)); gap: var(--s2); }
  .way { display: grid; align-content: start; gap: var(--s3); padding: var(--s5); border-radius: var(--r-row); background: var(--wash); }
  h2 { color: var(--ink); font: var(--type-label); }
  p { color: var(--ink2); font: var(--type-small); }
  .actions { display: flex; flex-wrap: wrap; gap: var(--s2); margin-top: var(--s1); }
  .actions a { display: inline-flex; align-items: center; gap: var(--s2); height: 36px; padding: 0 var(--s4); border-radius: var(--r-pill); background: var(--paper); color: var(--ink); font: var(--type-control); transition: background-color var(--fast) var(--ease), scale var(--fast) var(--ease); }
  .actions a:active { scale: 0.97; }
  .actions a > :global(svg) { color: var(--ink3); }
  .source { display: grid; justify-items: start; gap: var(--s3); }
  .source h2 { color: var(--ink3); font: var(--type-head); }
  .source code { color: var(--ink); }
  .source-action { display: inline-flex; align-items: center; height: 44px; padding: 0 var(--s5); border-radius: var(--r-pill); background: var(--ink); color: var(--paper); font: var(--type-control); transition: opacity var(--fast) var(--ease), scale var(--fast) var(--ease); }
  .source-action:active { scale: 0.97; }
  @media (hover: hover) {
    .actions a:hover { background: var(--wash2); }
    .source-action:hover { opacity: 0.86; }
  }
  @media (max-width: 720px) { .actions a { height: 44px; } }
</style>
