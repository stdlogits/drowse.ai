<script lang="ts">
  import { Icon, MorphText, tips } from "@stdlogits/ledger";
  import a9limPortrait from "../../assets/credits/a9lim.jpg";
  import treetownPortrait from "../../assets/credits/treetown.jpg";
  import katPortrait from "../../assets/credits/transkatgirl.jpg";
  import vogelPortrait from "../../assets/credits/voooooogel.jpg";
  import motionPortrait from "../../assets/credits/motion_so.jpg";
  import VerificationBadge from "./VerificationBadge.svelte";
  import SiteHeader from "./SiteHeader.svelte";
  import PageFooter from "./PageFooter.svelte";
  import PwaUpdatePrompt from "./PwaUpdatePrompt.svelte";
  import { onMount } from "svelte";

  const pageDescription = "The people, research, models and software behind Drowse, a local workbench for exploring language models.";
  const releaseBuild = typeof __DROWSE_HOSTED_RELEASE__ !== "undefined" && __DROWSE_HOSTED_RELEASE__;
  const sourceRevision = typeof __DROWSE_SOURCE_REVISION__ === "string" ? __DROWSE_SOURCE_REVISION__ : "preview";
  const version = typeof __DROWSE_VERSION__ === "string" ? __DROWSE_VERSION__ : "5.3.0";
  const repositoryUrl = "https://github.com/stdlogits/drowse";
  // X profile verification snapshot, checked 2026-09-06 (voooooogel: 2026-09-07).
  const team = [
    { name: "a9lim", handle: "@_a9lim", profile: "https://x.com/_a9lim", website: "a9l.im", portrait: a9limPortrait, verification: "blue" },
    { name: "treetown", handle: "@treetowntree", profile: "https://x.com/treetowntree", website: "logits.ml", portrait: treetownPortrait, verification: "blue" },
  ] as const;
  const thanks = [
    { handle: "@transkatgirl", profile: "https://x.com/transkatgirl", portrait: katPortrait, verification: "blue" },
    { handle: "@voooooogel", profile: "https://x.com/voooooogel", portrait: vogelPortrait, verification: "blue" },
    { handle: "@motion_so", profile: "https://x.com/motion_so", portrait: motionPortrait, verification: "gold" },
  ] as const;
  const providerNames = { claude: "Anthropic", openai: "OpenAI" } as const;
  const models = [
    { name: "Opus 4.6", provider: "claude" },
    { name: "Opus 4.7", provider: "claude" },
    { name: "Opus 4.8", provider: "claude" },
    { name: "Opus 5", provider: "claude" },
    { name: "Fable 5", provider: "claude" },
    { name: "GPT-5.6 Sol", provider: "openai" },
    { name: "GPT-6 Astra", provider: "openai" },
  ] as const;

  type License = { label: string; href: string };
  type Credit = { name: string; by?: string; note: string; href?: string; license?: License };
  const thirdParty = "/third-party-licenses";
  // Sources follow browser-runtime/runtime-lock.json. Licenses follow the hosted model closure checks.
  const catalogModels: Credit[] = [
    { name: "Gemma 3", by: "Google", href: "https://ai.google.dev/gemma",
      note: "270M, 1B and 4B instruction-tuned, and 1B base. Instruction-tuned weights come from Unsloth's copies. Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms.",
      license: { label: "Gemma Terms of Use", href: "https://ai.google.dev/gemma/terms" } },
    { name: "Qwen3 and Qwen3.5", by: "Qwen team, Alibaba Cloud", href: "https://huggingface.co/Qwen",
      note: "Qwen3 1.7B and 4B, and Qwen3.5 2B base.", license: { label: "Apache 2.0", href: thirdParty } },
    { name: "GPT-2", by: "OpenAI", href: "https://huggingface.co/openai-community/gpt2",
      note: "The original 124M model.", license: { label: "Modified MIT", href: "https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/LICENSE" } },
    { name: "Pythia", by: "EleutherAI", href: "https://huggingface.co/EleutherAI/pythia-70m-deduped",
      note: "70M, trained on deduplicated data.", license: { label: "Apache 2.0", href: thirdParty } },
  ];
  // From the research lineage in the project README.
  const research: Credit[] = [
    { name: "Representation Engineering", by: "Zou et al., 2023", href: "https://arxiv.org/abs/2310.01405", note: "The approach Drowse builds on." },
    { name: "repeng", by: "Theia Vogel", href: "https://github.com/vgel/repeng", note: "The best-known compact implementation of representation engineering." },
    { name: "Difference-of-means extraction", by: "Im and Li, 2025", href: "https://arxiv.org/abs/2502.02716", note: "Two-pole extraction follows this method." },
    { name: "Manifold steering", by: "Goodfire", href: "https://arxiv.org/abs/2605.05115", note: "Manifold steering follows this work." },
    { name: "The Assistant Axis", by: "Anthropic", href: "https://arxiv.org/abs/2601.10387", note: "The personas source is derived from its framing." },
    { name: "Verbalizable workspace", by: "Gurnee et al., 2026", href: "https://transformer-circuits.pub/2026/workspace/index.html", note: "The R and S lenses implement this method." },
    { name: "Neuron Lens", by: "Standard Logits", href: "https://neuron-lens.com", note: "Neuron and head reads follow its definitions." },
  ];
  const interpretability: Credit[] = [
    { name: "SAELens", by: "SAELens contributors", href: "https://github.com/jbloomAus/SAELens", note: "Loads published sparse autoencoders in the Python edition." },
    { name: "Gemma Scope 2", by: "Google DeepMind", note: "Sparse autoencoders for Gemma 3, used for SAE readings." },
    { name: "Neuronpedia", href: "https://www.neuronpedia.org", note: "Feature descriptions for Gemma Scope 2 features." },
  ];
  const software: Credit[] = [
    { name: "WebLLM", by: "MLC", href: "https://github.com/mlc-ai/web-llm", note: "Runs models in the browser. Drowse uses its own fork, @drowse/web-llm.", license: { label: "Apache 2.0", href: thirdParty } },
    { name: "Apache TVM", by: "Apache Software Foundation", href: "https://github.com/apache/tvm", note: "Compiles models for WebGPU.", license: { label: "Apache 2.0", href: thirdParty } },
    { name: "Svelte", by: "Svelte contributors", href: "https://svelte.dev", note: "The interface framework.", license: { label: "MIT", href: thirdParty } },
    { name: "Vite", by: "Vite contributors", href: "https://vite.dev", note: "Builds the site.", license: { label: "MIT", href: thirdParty } },
    { name: "fflate", by: "Arjun Barrett", href: "https://github.com/101arrowz/fflate", note: "Compression for archives and downloads.", license: { label: "MIT", href: thirdParty } },
    { name: "noble-hashes", by: "Paul Miller", href: "https://github.com/paulmillr/noble-hashes", note: "Checks downloaded files.", license: { label: "MIT", href: thirdParty } },
    { name: "YAML", by: "Eemeli Aro", href: "https://eemeli.org/yaml/", note: "Reads and writes transcripts.", license: { label: "ISC", href: thirdParty } },
    { name: "Torph", by: "Lochie Axon", href: "https://torph.lochie.me", note: "Text morphing.", license: { label: "MIT", href: "/LICENSE-Torph.txt" } },
    { name: "Wix Madefor", by: "The Wix Madefor Project Authors", href: "https://github.com/wix/wixmadefor", note: "Display and text font.", license: { label: "SIL Open Font License 1.1", href: "/LICENSE-Wix-Madefor.txt" } },
    { name: "Ledger", by: "Standard Logits", href: "https://github.com/stdlogits/ledger", note: "The design language every Standard Logits surface is drawn in.", license: { label: "AGPL-3.0", href: thirdParty } },
  ];
  const creditGroups = [
    { id: "research", title: "Research Drowse builds on", lede: "If you use Drowse in published research, cite these methods too.", items: research },
    { id: "interpretability", title: "Interpretability tools and data", lede: "Tools and data behind SAE readings and feature labels.", items: interpretability },
    { id: "catalog-models", title: "Models", lede: "Open models you can run in Drowse. Each keeps its own license.", items: catalogModels },
    { id: "software", title: "Software and type", lede: "Open-source projects Drowse ships with or is built with.", items: software },
  ];

  const citation = [
    "@software{drowse,",
    "  author = {{a9lim} and {treetown}},",
    "  title = {Drowse},",
    "  year = {2026},",
    `  version = {${version}},`,
    ...(releaseBuild && /^[0-9a-f]{40}$/.test(sourceRevision) ? [`  commit = {${sourceRevision}},`] : []),
    `  url = {${repositoryUrl}},`,
    "}",
  ].join("\n");
  let citationBlock = $state<HTMLElement>();
  let citationStatus = $state("");
  let citationTimer: ReturnType<typeof setTimeout> | undefined;

  function copyWithSelection(text: string): boolean {
    const field = document.createElement("textarea");
    field.value = text;
    field.setAttribute("readonly", "");
    field.className = "sr-only";
    document.body.append(field);
    field.select();
    try { return document.execCommand("copy"); }
    catch { return false; }
    finally { field.remove(); }
  }

  async function copyCitation() {
    let copied = false;
    try { await navigator.clipboard.writeText(citation); copied = true; }
    catch { copied = copyWithSelection(citation); }
    if (!copied && citationBlock) {
      const range = document.createRange();
      range.selectNodeContents(citationBlock);
      const selection = window.getSelection();
      selection?.removeAllRanges();
      selection?.addRange(range);
    }
    citationStatus = copied ? "Citation copied." : "Couldn't copy. The citation is selected, so you can copy it yourself.";
    clearTimeout(citationTimer);
    citationTimer = setTimeout(() => { citationStatus = ""; }, 5000);
  }

  // Update the page's one description tag instead of adding a second one.
  let mounted = $state(false);
  onMount(() => {
    mounted = true;
    document.title = "Credits · Drowse";
    let meta = document.head.querySelector<HTMLMetaElement>('meta[name="description"]');
    const created = !meta;
    const previous = meta?.content ?? "";
    if (!meta) {
      meta = document.createElement("meta");
      meta.name = "description";
      document.head.append(meta);
    }
    const tag = meta;
    tag.content = pageDescription;
    return () => {
      clearTimeout(citationTimer);
      if (created) tag.remove();
      else tag.content = previous;
    };
  });
</script>

<!-- The update prompt touches window on teardown, so it only renders in the browser. -->
{#if mounted}<PwaUpdatePrompt />{/if}

<div class="credits-shell" use:tips>
  <a class="skip-link" href="#credits-main">Skip to content</a>
  <SiteHeader current="credits" />

  <main id="credits-main" tabindex="-1">
    <section class="credits-intro" aria-labelledby="credits-title">
      <h1 id="credits-title">Credits.</h1>
      <p class="lede">The people behind Drowse.</p>
    </section>

    <section class="part" aria-labelledby="team-title">
      <div class="caption">
        <h2 id="team-title">The team</h2>
        <p>The people who build and maintain Drowse.</p>
      </div>
      <div class="team-list">
        {#each team as member (member.name)}
          <article class="team-member">
            <img class="portrait" src={member.portrait} width="64" height="64" alt="" />
            <div class="member-copy">
              <h3><span>{member.name}</span><VerificationBadge status={member.verification} /></h3>
              <a class="member-handle" href={member.profile} rel="noreferrer">{member.handle}</a>
              <a class="member-website" href={`https://${member.website}`} rel="noreferrer">
                <span class="sr-only">Website </span>{member.website}<Icon name="external" size={14} />
              </a>
            </div>
          </article>
        {/each}
      </div>
    </section>

    <section class="part" aria-labelledby="thanks-title">
      <div class="caption">
        <h2 id="thanks-title">Special thanks</h2>
        <p>People and groups who inspired, supported or contributed to Drowse.</p>
      </div>
      <ul class="thanks-list">
        {#each thanks as contributor (contributor.handle)}
          <li>
            <a href={contributor.profile} aria-label={contributor.handle} aria-describedby={`${contributor.handle}-verification`} rel="noreferrer">
              <img class="thanks-portrait" src={contributor.portrait} width="40" height="40" alt="" />
              <span class="thanks-name"><span>{contributor.handle}</span><VerificationBadge status={contributor.verification} id={`${contributor.handle}-verification`} /></span>
              <Icon name="external" size={14} />
            </a>
          </li>
        {/each}
      </ul>
    </section>

    <section class="part" aria-labelledby="models-title">
      <div class="caption">
        <h2 id="models-title">Models of Technical Staff</h2>
        <p>The models that worked hard to bring Drowse to life.</p>
      </div>
      <ul class="models-list">
        {#each models as model (model.name)}
          <li class="model-credit"><span class="model-name">{model.name}</span><span class="model-by">{providerNames[model.provider]}</span></li>
        {/each}
      </ul>
    </section>

    {#each creditGroups as group (group.id)}
      <section class="part credit-section" aria-labelledby={`${group.id}-title`}>
        <div class="caption">
          <h2 id={`${group.id}-title`}>{group.title}</h2>
          <p>{group.lede}</p>
        </div>
        <div class="credit-body">
          <ul class="credit-list">
            {#each group.items as item (item.name)}
              <li class="credit-card">
                <h3 class="credit-name">
                  {#if item.href}
                    <a href={item.href} rel="noreferrer">{item.name}{#if item.href.startsWith("https://github.com/")}<span class="sr-only">(opens GitHub)</span>{/if}</a>
                  {:else}{item.name}{/if}
                </h3>
                <span class="credit-detail">{#if item.by}<span class="credit-by">{item.by}</span><span class="sep" aria-hidden="true">·</span>{/if}<span class="credit-note">{item.note}</span></span>
                <span class="credit-license">{#if item.license}<a href={item.license.href} rel={item.license.href.startsWith("/") ? undefined : "noreferrer"}><span class="sr-only">License: </span>{item.license.label}</a>{/if}</span>
              </li>
            {/each}
          </ul>
          {#if group.id === "software"}
            <p class="credit-footnote">Full license texts for every bundled package are on the <a href={thirdParty}>third-party licenses</a> page.</p>
          {/if}
        </div>
      </section>
    {/each}

    <section class="part" aria-labelledby="cite-title">
      <div class="caption">
        <h2 id="cite-title">Cite Drowse</h2>
        <p>If Drowse helped your work, cite the version you used{releaseBuild ? " and this build's commit" : ""}, along with the exact model checkpoint.</p>
      </div>
      <div class="citation">
        <!-- svelte-ignore a11y_no_noninteractive_tabindex (The scrollable citation must be keyboard-accessible.) -->
        <pre bind:this={citationBlock} tabindex="0" role="region" aria-label="BibTeX citation"><code>{citation}</code></pre>
        <div class="citation-actions">
          <button type="button" class="dw-quiet" onclick={copyCitation}><MorphText text={citationStatus === "Citation copied." ? "Copied" : "Copy BibTeX"} /></button>
          <span class="citation-status" role="status">{citationStatus}</span>
        </div>
      </div>
    </section>

    <section class="contribute-section" aria-labelledby="contribute-title">
      <div class="caption">
        <h2 id="contribute-title">Want to help?</h2>
        <p>Report a bug, suggest a change, or send a fix. If something broke, include the steps to reproduce it.</p>
      </div>
      <a class="contribute-action" href="/contribute">Ways to contribute</a>
    </section>
  </main>

  <PageFooter current="credits" />
</div>

<style>
  /* Credits as a ledger (neuron-lens.com's): each part a caption beside its rows. */
  .credits-shell { display: flex; flex-direction: column; min-height: 100svh; }
  .skip-link { position: fixed; z-index: 60; top: var(--s2); inset-inline-start: var(--s2); padding: var(--s2) var(--s4); border-radius: var(--r-pill); background: var(--ink); color: var(--paper); font: var(--type-control); translate: 0 calc(-100% - var(--s4)); transition: translate var(--fast) var(--ease); }
  .skip-link:focus { translate: 0 0; }
  main { flex: 1; display: grid; grid-template-columns: minmax(0, 1fr); align-content: start; gap: var(--s8); width: 100%; max-width: var(--page); margin: 0 auto; padding: var(--s7) var(--gutter) var(--s9); }
  main:focus { outline: none; }
  .credits-intro { display: grid; justify-items: start; gap: var(--s5); }
  h1 { color: var(--ink); font: var(--type-title); letter-spacing: -0.04em; }
  .lede { color: var(--ink2); font: var(--type-lead); }

  .part { display: grid; grid-template-columns: minmax(0, 300px) minmax(0, 1fr); gap: var(--s4) var(--s7); align-items: start; min-width: 0; }
  .caption { position: sticky; top: var(--s5); display: grid; gap: var(--s2); }
  h2 { color: var(--ink); font: var(--type-label); }
  .caption p { color: var(--ink3); font: var(--type-small); }
  a { transition: color var(--fast) var(--ease), text-decoration-color var(--fast) var(--ease), background-color var(--fast) var(--ease); }

  .team-list { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 280px), 1fr)); gap: var(--s2); }
  .team-member { display: flex; align-items: center; gap: var(--s4); padding: var(--s4); border-radius: var(--r-row); background: var(--wash); }
  .portrait { flex: none; width: 64px; height: 64px; border-radius: var(--r-pill); object-fit: cover; }
  .member-copy { display: grid; gap: 2px; min-width: 0; }
  h3 { display: flex; align-items: center; gap: var(--s1); color: var(--ink); font: var(--type-label); }
  .member-handle, .member-website { display: inline-flex; align-items: center; gap: var(--s1); color: var(--ink2); font: var(--type-small); }
  .member-website { color: var(--ink3); }

  .thanks-list { display: flex; flex-wrap: wrap; gap: var(--s2); }
  .thanks-list a { display: flex; align-items: center; gap: var(--s3); padding: var(--s2) var(--s4) var(--s2) var(--s2); border-radius: var(--r-pill); background: var(--wash); color: var(--ink); font: var(--type-control); }
  .thanks-portrait { width: 40px; height: 40px; border-radius: var(--r-pill); object-fit: cover; }
  .thanks-name { display: inline-flex; align-items: center; gap: var(--s1); }
  .thanks-list a > :global(svg) { color: var(--ink3); }

  .models-list { display: flex; flex-wrap: wrap; gap: var(--s2); }
  .model-credit { display: inline-flex; align-items: baseline; gap: var(--s2); padding: var(--s2) var(--s4); border-radius: var(--r-pill); background: var(--wash); }
  .model-name { color: var(--ink); font: var(--type-control); }
  .model-by { color: var(--ink3); font: var(--type-meta); }

  .credit-body { display: grid; gap: var(--s4); min-width: 0; }
  .credit-list { display: grid; gap: var(--s3); }
  .credit-card { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 2px var(--s5); align-items: baseline; }
  .credit-name { color: var(--ink); font: var(--type-body); }
  .credit-detail { grid-column: 1; color: var(--ink2); font: var(--type-small); }
  .credit-by { color: var(--ink2); }
  .sep { margin: 0 var(--s2); color: var(--ink3); }
  .credit-license { grid-column: 2; grid-row: 1; color: var(--ink3); font: var(--type-small); text-align: end; white-space: nowrap; }
  .credit-name a, .credit-license a, .credit-footnote a, .member-handle { text-decoration: underline; text-decoration-color: var(--wash2); text-decoration-thickness: 2px; text-underline-offset: 4px; }
  .credit-footnote { color: var(--ink3); font: var(--type-small); }

  .citation { display: grid; gap: var(--s3); min-width: 0; }
  pre { margin: 0; overflow-x: auto; padding: var(--s4) var(--s5); border-radius: var(--r-box); background: var(--wash); color: var(--ink); font: var(--type-small); font-variant-numeric: tabular-nums; }
  .citation-actions { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s3); }
  .citation-status { color: var(--ink3); font: var(--type-small); }

  .contribute-section { display: grid; grid-template-columns: minmax(0, 300px) auto; gap: var(--s4) var(--s7); align-items: center; justify-content: start; }
  .contribute-action { display: inline-flex; align-items: center; height: 44px; padding: 0 var(--s5); border-radius: var(--r-pill); background: var(--ink); color: var(--paper); font: var(--type-control); transition: opacity var(--fast) var(--ease), scale var(--fast) var(--ease); }
  .contribute-action:active { scale: 0.97; }

  @media (hover: hover) {
    .member-handle:hover, .member-website:hover, .credit-name a:hover, .credit-license a:hover, .credit-footnote a:hover { color: var(--ink); text-decoration-color: var(--ink3); }
    .thanks-list a:hover { background: var(--wash2); }
    .contribute-action:hover { opacity: 0.86; }
  }
  @media (max-width: 900px) {
    main { gap: var(--s7); padding-top: var(--s5); }
    .part, .contribute-section { grid-template-columns: minmax(0, 1fr); }
    .caption { position: static; }
    .credit-card { grid-template-columns: minmax(0, 1fr); }
    .credit-license { grid-column: 1; grid-row: auto; text-align: start; }
  }
</style>
