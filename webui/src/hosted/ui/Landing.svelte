<script lang="ts">
  import { onMount, tick } from "svelte";
  import { Icon, MorphText, Token, glide, tips } from "@stdlogits/ledger";
  import CapabilityDemo from "./CapabilityDemo.svelte";
  import SiteHeader from "./SiteHeader.svelte";
  import PageFooter from "./PageFooter.svelte";
  import { siteTitle, sourceRepository } from "../../../scripts/site-metadata.mjs";

  // The hero's reading: one real prediction from the recorded example below (src/hosted/data/
  // neighbor-example.json: Gemma 3 4B in the browser runtime, steered toward welcoming at 0.10,
  // token 22 of its reply). The five likeliest next tokens after "We'd love for you to feel", and
  // the one it wrote. The title's slot holds the written one, or whichever row is pointed at or
  // chosen. scripts/landing-models.test.mjs checks every number against the recording.
  type Next = { token: string; p: number; wrote?: true };
  const heroModel = "Gemma 3 4B";
  const heroSteering = "steered toward welcoming, 0.10";
  const heroContext = "We’d love for you to feel";
  const heroNext: Next[] = [
    { token: " more", p: 0.4544 },
    { token: " welcome", p: 0.2118 },
    { token: " at", p: 0.2046, wrote: true },
    { token: " settled", p: 0.0919 },
    { token: " like", p: 0.0357 },
  ];
  const heroMax = Math.max(...heroNext.map((next) => next.p));
  const percent = (p: number) => `${(p * 100).toFixed(1)}%`;
  const written = heroNext.find((next) => next.wrote)!;
  let chosen = $state.raw(written);
  let pointed = $state.raw<Next | null>(null);
  const shown = $derived(pointed ?? chosen);
  function leave(event: FocusEvent) {
    const into = event.relatedTarget;
    if (!(into instanceof Node) || !(event.currentTarget as HTMLElement).contains(into)) pointed = null;
  }

  // The prerendered page states the requirement; the probe only replaces it with a result.
  let webGpu = $state<"unknown" | "available" | "unavailable">("unknown");

  async function probeWebGpu(): Promise<typeof webGpu> {
    const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
    if (typeof gpu?.requestAdapter !== "function") return "unavailable";
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      // Asks for an adapter only: no device, model, or runtime code is loaded here.
      const adapter = await Promise.race([
        gpu.requestAdapter(),
        new Promise<"timeout">((resolve) => { timer = setTimeout(() => resolve("timeout"), 3000); }),
      ]);
      return adapter === "timeout" ? "unknown" : adapter ? "available" : "unavailable";
    } catch {
      return "unavailable";
    } finally {
      clearTimeout(timer);
    }
  }

  // Mounting replaces the prerendered DOM, which drops the browser's jump to `#section`.
  let hashScrollY: number | null = null;
  function restoreHash() {
    let id = "";
    try { id = decodeURIComponent(location.hash.slice(1)); } catch { return; }
    const target = id ? document.getElementById(id) : null;
    if (!target) return;
    target.scrollIntoView({ block: "start", behavior: "instant" });
    if (target.hasAttribute("tabindex")) target.focus({ preventScroll: true });
    hashScrollY = window.scrollY;
  }

  async function demoReady() {
    // The recorded example changes the page height; jump again once it is in place,
    // unless the reader has already scrolled away from the target.
    if (hashScrollY === null) return;
    await tick();
    requestAnimationFrame(() => { if (hashScrollY !== null && Math.abs(window.scrollY - hashScrollY) < 4) restoreHash(); });
  }

  onMount(() => {
    document.title = siteTitle;
    let active = true;
    void probeWebGpu().then((result) => { if (active) webGpu = result; });
    requestAnimationFrame(restoreHash);
    return () => { active = false; };
  });

  const requirement = $derived(
    webGpu === "available"
      ? "WebGPU is available in this browser. Drowse checks your device's memory next."
      : webGpu === "unavailable"
        ? "This browser can't run Drowse yet. Try a current version of Chrome, Edge or Safari 26 or later on a computer with a graphics processor."
        : "Needs a browser with WebGPU, such as current Chrome, Edge or Safari 26 or later.",
  );

  const capabilities = [
    {
      title: "Inspect the prediction",
      body: "See which words the model could have chosen at each step. Add word insights (an R lens) to read predictions across layers, or feature insights (SAE) to inspect learned features.",
    },
    {
      title: "Change an influence",
      body: "An influence nudges the model's internal activity toward a concept, word, or feature while it writes. Strength sets how hard it pushes. The model's weights never change.",
    },
    {
      title: "Compare what changes",
      body: "Branch from any token in the Loom, Drowse's tree of continuations, and compare the new reply with the original. Your work autosaves in this browser.",
    },
  ];

  // Mirrors the verified runtime lock and the display names in the signed catalog.
  // `id` is the catalog variant the app's `?model=` parameter selects;
  // scripts/landing-models.test.mjs checks this list against browser-runtime/runtime-lock.json.
  // Download range (model plus required core pack) from signed catalog sequence 9.
  const storageRange = "0.3 to 2.6 GB";
  const modelGroups = [
    {
      id: "chat-models",
      title: "Chat models",
      description: ["Follow instructions.", "Explore a conversation."],
      models: [
        { name: "Gemma 3 270M", id: "gemma3-270m-instruct-q0f32", provider: "Google" },
        { name: "Gemma 3 1B", id: "gemma3-1b-instruct-q4f32_1", provider: "Google" },
        { name: "Gemma 3 4B", id: "gemma3-4b-instruct-q4f32_1", provider: "Google" },
        { name: "Qwen3 1.7B", id: "qwen3-1.7b-q4f32_1", provider: "Qwen" },
        { name: "Qwen3 4B", id: "qwen3-4b-q4f32_1", provider: "Qwen" },
      ],
    },
    {
      id: "base-models",
      title: "Base models",
      description: ["Continue raw text.", "Study next-token predictions."],
      models: [
        { name: "Gemma 3 1B PT", id: "gemma3-1b-pt-q4f32_1", provider: "Google" },
        { name: "GPT-2 Base (124M)", id: "gpt2-base-q0f32", provider: "OpenAI" },
        { name: "Pythia 70M Deduped Base", id: "pythia-70m-base-q0f32", provider: "EleutherAI" },
        { name: "Qwen 3.5 2B Base", id: "qwen35-2b-base-q4f32_1", provider: "Qwen" },
      ],
    },
  ];

  const requirements = [
    {
      title: "A browser with WebGPU",
      body: "Current Chrome, Edge, Brave, or Arc, or Safari 26 or later. Firefox support is a preview.",
    },
    {
      title: "Storage",
      body: `${storageRange} per model, plus optional insight packs. Your browser keeps them, so each model downloads once.`,
    },
    {
      title: "Memory",
      body: "About 4 GB for the smallest chat model, up to 16 GB or more for the 4B models.",
    },
    {
      title: "Phones",
      body: "Phones run compact models as a preview. Use a computer for the larger models.",
    },
  ];

  const glossary = [
    { term: "Influence", definition: "A change Drowse adds to the model's internal activity while it writes, pulling replies toward or away from a concept. Also called steering." },
    { term: "Strength (α)", definition: "How hard an influence pushes. 0 leaves the model unsteered; negative values push the other way." },
    { term: "Word insights (R lens)", definition: "A lens that shows which words each layer is leaning toward, before the model makes its final prediction." },
    { term: "Feature insights (SAE)", definition: "A sparse autoencoder splits the model's activity into features: learned patterns that often match a recognizable idea." },
    { term: "Loom", definition: "A tree of continuations. Branch from any token, try another path, and compare the results." },
    { term: "Base model", definition: "A model trained only to continue text, without chat tuning. It predicts what comes next instead of following requests." },
  ];

  const pythonInstall = `${sourceRepository}#installation`;

  // Answers that hold a link are split around it.
  const faq: { question: string; answer: string[]; link?: { href: string; label: string } }[] = [
    { question: "Does anything leave my device?", answer: ["Models and optional packs download from Hugging Face. Some feature descriptions are fetched from Neuronpedia when you look them up; that request names the feature, not your text. Your prompts, chats, and saved work stay in this browser. The contact form sends only what you enter in it. ", "."], link: { href: "/privacy", label: "Read the privacy details" } },
    { question: "Why is the download large?", answer: [`A model is mostly its weights: hundreds of millions to billions of numbers. Each model is ${storageRange}, plus any insight packs you choose. Your browser keeps the files, so you only download a model once.`] },
    { question: "Will it work on my phone?", answer: ["Phones with WebGPU can run compact models as a preview, within the memory the browser allows. For the larger models and insight packs, use a computer."] },
    { question: "Is steering the same as fine-tuning?", answer: ["No. Fine-tuning trains the model further and changes its weights. Steering adds an influence to the model's activity while it writes. Set the strength to 0 and you get the unsteered model back."] },
    { question: "What's a base model?", answer: ["A model trained to continue text, before any chat or instruction tuning. It doesn't answer requests; it predicts what comes next. That makes it useful for studying raw next-token predictions."] },
    { question: "Can I use it in a class?", answer: ["Yes. Drowse is free and open source, and it needs only a browser. Each student downloads models to their own device, so check the storage and memory needs above first. Each model's license still applies."] },
    { question: "How do I cite it?", answer: ["Cite the Drowse version and the exact model checkpoint you used, alongside the methods Drowse builds on. The ", " page lists them."], link: { href: "/credits", label: "Credits" } },
    { question: "How is this different from TransformerLens or Neuronpedia?", answer: ["TransformerLens is a Python library for writing interpretability code against a model's internals. Neuronpedia is a hosted platform for exploring and sharing published features. Drowse is an interactive workbench that runs the model on your device, so you can inspect, steer, and branch without code or a server. They work well together."] },
  ];
</script>

<a class="skip-link" href="#main">Skip to content</a>

<div class="landing-shell" use:tips>
  <SiteHeader current="home" />

  <main id="main" tabindex="-1">
    <section class="hero" aria-labelledby="hero-title">
      <div class="hero-copy">
        <h1 id="hero-title">See inside<br />your model.</h1>
        <p class="lede">
          Run a language model on your device. Inspect its predictions and test
          how changes to its internal activity affect its replies.
        </p>
        <div class="hero-action-row">
          <a class="primary-action" href="/app">Open Drowse</a>
          <span class="requirement" data-webgpu={webGpu}>{requirement}</span>
        </div>
        <p class="trust-line">Open source. Inference and saved work stay on your device. <a href="/privacy">What leaves your device <span aria-hidden="true">→</span></a></p>
        <p class="audience">For researchers, students, educators, and anyone curious about how language models work.</p>
      </div>
      <figure class="reading rise" style:--i={1} aria-labelledby="reading-title">
        <figcaption id="reading-title"><span>{heroModel}</span><span>{heroSteering}</span></figcaption>
        <p class="said">{heroContext} <span class="slot" data-tip-title={shown.token.trim()} data-tip={`${percent(shown.p)}${shown.wrote ? ", the token it wrote" : ""}`} data-tip-sub={heroModel}><MorphText text={shown.token.trim()} numbers={false} /></span></p>
        <div class="next" role="group" aria-label={`What ${heroModel} expected after “${heroContext}”. Choose one to put it in the sentence.`} use:glide onpointerleave={() => (pointed = null)} onfocusout={leave}>
          {#each heroNext as next, i (next.token)}
            <button type="button" class="row" aria-pressed={next === chosen} style:--i={i} onclick={() => (chosen = next)} onpointerenter={() => (pointed = next)} onfocus={() => (pointed = next)}>
              <Token token={next.token} {i} />
              <span class="bar" aria-hidden="true"><i style:--w={next.p / heroMax}></i></span>
              <span class="pct">{next.wrote ? "wrote · " : ""}<b>{percent(next.p)}</b></span>
            </button>
          {/each}
        </div>
      </figure>
    </section>

    <section class="capabilities" id="what-you-can-do" aria-labelledby="capabilities-title">
      <div class="section-intro">
        <h2 id="capabilities-title">Test what shapes a reply.</h2>
        <p class="section-description">Ask a question, then try it with different response controls. Compare the predictions and replies in your browser, with no notebook or server to set up.</p>
      </div>
      <ol>
        {#each capabilities as capability (capability.title)}
          <li>
            <h3>{capability.title}</h3>
            <p>{capability.body}</p>
          </li>
        {/each}
      </ol>
      <div class="demo-panel"><CapabilityDemo onready={demoReady} /></div>
    </section>

    <section class="models" id="models" aria-labelledby="models-title">
      <div class="section-intro">
        <h2 id="models-title">Choose a model.</h2>
        <p class="section-description">Use a chat model for conversation or a base model for text completion. Drowse checks what your device can run when you open the app.</p>
      </div>
      <div class="model-groups">
        {#each modelGroups as group (group.id)}
          <section class="model-group" aria-labelledby={`${group.id}-title`}>
            <h3 id={`${group.id}-title`}>{group.title}</h3>
            <p>{group.description[0]}<br />{group.description[1]}</p>
            <ul class="model-list">
              {#each group.models as model (model.id)}
                <li>
                  <a class="model-row" href={`/app?choose=1&model=${encodeURIComponent(model.id)}`} data-model={model.id}>
                    <span class="model-copy"><strong>{model.name}</strong><span>{model.provider}</span></span>
                    <Icon name="right" />
                  </a>
                </li>
              {/each}
            </ul>
          </section>
        {/each}
      </div>
      <p class="model-note">Gemma 3 270M runs at full precision, so its download is larger than Gemma 3 1B's.</p>
    </section>

    <section class="guide" id="requirements" aria-labelledby="requirements-title">
      <h2 id="requirements-title">What you need.</h2>
      <ul class="guide-grid">
        {#each requirements as item (item.title)}
          <li class="guide-panel">
            <h3>{item.title}</h3>
            <p>{item.body}</p>
          </li>
        {/each}
      </ul>
    </section>

    <section class="guide" id="glossary" aria-labelledby="glossary-title">
      <h2 id="glossary-title">Words you'll see.</h2>
      <dl class="guide-grid">
        {#each glossary as entry (entry.term)}
          <div class="guide-panel">
            <dt>{entry.term}</dt>
            <dd>{entry.definition}</dd>
          </div>
        {/each}
      </dl>
    </section>

    <section class="guide" id="faq" aria-labelledby="faq-title">
      <h2 id="faq-title">Common questions.</h2>
      <!-- Native details, so the answers open without script and for find-in-page. -->
      <div class="faq">
        {#each faq as item (item.question)}
          <details>
            <summary>{item.question}<Icon name="down" /></summary>
            <p>{#if item.link}{item.answer[0]}<a href={item.link.href}>{item.link.label}</a>{item.answer[1]}{:else}{item.answer[0]}{/if}</p>
          </details>
        {/each}
      </div>
    </section>

    <section class="guide closing" id="python" aria-labelledby="python-title">
      <div class="python-band">
        <h2 id="python-title">Need more models?</h2>
        <p>Drowse Local runs supported Hugging Face models with Python, fits your own lenses, trains SAEs, and serves OpenAI- and Ollama-compatible APIs.</p>
        <a class="text-link" href={pythonInstall}>Install from source <span aria-hidden="true">↗</span></a>
      </div>
      <div class="closing-action">
        <a class="primary-action" href="/app">Open Drowse</a>
        <p>Choose your model and tools inside the app.</p>
      </div>
    </section>
  </main>

  <PageFooter current="home" />
</div>

<style>
  /* The landing in Ledger: paper, the hero's copy beside its orb, then the parts of the page as
     captions over rows of wash. */
  .skip-link { position: fixed; z-index: 60; top: var(--s2); inset-inline-start: var(--s2); padding: var(--s2) var(--s4); border-radius: var(--r-pill); background: var(--ink); color: var(--paper); font: var(--type-control); translate: 0 calc(-100% - var(--s4)); transition: translate var(--fast) var(--ease); }
  .skip-link:focus { translate: 0 0; }
  .landing-shell { display: flex; flex-direction: column; min-height: 100svh; overflow-x: clip; }
  main { flex: 1; display: grid; grid-template-columns: minmax(0, 1fr); gap: var(--s9); width: 100%; max-width: var(--page); margin: 0 auto; padding: var(--s6) var(--gutter) var(--s9); }
  main:focus { outline: none; }
  a { transition: color var(--fast) var(--ease), background-color var(--fast) var(--ease), opacity var(--fast) var(--ease), scale var(--fast) var(--ease), text-decoration-color var(--fast) var(--ease); }

  .hero { display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(0, 0.9fr); align-items: center; gap: var(--s7); padding-block: var(--s6) var(--s4); }
  .hero-copy { display: grid; justify-items: start; gap: var(--s5); }
  h1 { color: var(--ink); font: var(--type-title); letter-spacing: -0.04em; overflow-wrap: anywhere; }
  .lede { max-width: 34ch; color: var(--ink2); font: var(--type-lead); }
  .hero-action-row { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s3) var(--s4); }
  .primary-action { display: inline-flex; align-items: center; height: 44px; padding: 0 var(--s5); border-radius: var(--r-pill); background: var(--ink); color: var(--paper); font: var(--type-control); white-space: nowrap; }
  .primary-action:active { scale: 0.97; }
  .requirement { max-width: 36ch; color: var(--ink3); font: var(--type-meta); }
  .trust-line, .audience { max-width: 52ch; color: var(--ink3); font: var(--type-small); }
  .trust-line a, .text-link, .faq a { color: var(--ink2); text-decoration: underline; text-decoration-color: var(--wash2); text-decoration-thickness: 2px; text-underline-offset: 4px; }
  /* The reading: the model's own prediction, so the last layer's place on the depth ramp. */
  .reading { --tint: var(--late); display: grid; gap: var(--s4); min-width: 0; margin: 0; }
  figcaption { display: flex; flex-wrap: wrap; gap: var(--s1) var(--s3); color: var(--ink3); font: var(--type-meta); }
  figcaption span:first-child { color: var(--ink2); }
  .said { color: var(--ink); font: var(--type-lead); text-wrap: pretty; }
  /* The slot: the token the list is about, marked as a unit's firing is, grown by its own shadow. */
  .slot { --mark: color-mix(in oklch, var(--tint) 22%, transparent); border-radius: var(--r-col); background: var(--mark); box-shadow: 0 0 0 2px var(--mark); cursor: help; }
  /* The rows share one grid (subgrid), so every bar has the same track and the numbers size the
     last column. */
  .next { --glide-bg: var(--wash); --glide-radius: var(--r-box); display: grid; grid-template-columns: minmax(0, 120px) minmax(0, 1fr) auto; gap: 2px var(--s4); margin-inline: calc(var(--s3) * -1); }
  .row { grid-column: 1 / -1; display: grid; grid-template-columns: subgrid; align-items: center; width: 100%; padding: var(--s2) var(--s3); border: 0; border-radius: var(--r-box); background: transparent; color: var(--ink); font: var(--type-small); text-align: start; cursor: pointer; transition: background-color var(--fast) var(--ease), scale var(--fast) var(--ease); }
  .row:active { scale: 0.99; }
  .row > :global(.tok) { justify-self: start; }
  .row[aria-pressed="true"] > :global(.tok) { background: var(--wash2); }
  /* Flat bars in the accent on a wash, as wide as each token's probability against the likeliest. */
  .bar { position: relative; height: 4px; border-radius: 2px; background: var(--wash); }
  .bar i { position: absolute; inset: 0 auto 0 0; width: max(2px, calc(var(--w) * 100%)); border-radius: 2px; background: var(--tint); transform-origin: left; animation: fill var(--glide) var(--ease) backwards; animation-delay: calc(80ms + var(--i, 0) * 24ms); }
  @keyframes fill { from { transform: scaleX(0); } }
  .pct { color: var(--ink3); font: var(--type-meta); font-variant-numeric: tabular-nums; text-align: end; white-space: nowrap; }
  .pct b { color: var(--ink); font-weight: 600; }

  h2 { color: var(--ink); font: var(--type-heading); letter-spacing: -0.02em; }
  h3 { color: var(--ink); font: var(--type-label); }
  .section-intro { display: grid; gap: var(--s3); max-width: 60ch; }
  .section-description { color: var(--ink2); font: var(--type-body); }

  .capabilities { display: grid; gap: var(--s6); }
  .capabilities ol { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: var(--s2); }
  .capabilities li, .guide-panel { display: grid; align-content: start; gap: var(--s2); padding: var(--s5); border-radius: var(--r-row); background: var(--wash); }
  .capabilities li p, .guide-panel p, dd { margin: 0; color: var(--ink2); font: var(--type-small); }
  .demo-panel { padding: var(--s6); border-radius: var(--r-row); background: var(--wash); --field: var(--wash2); --field-hover: var(--paper); }

  .models { display: grid; gap: var(--s6); }
  .model-groups { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--s5) var(--s6); }
  .model-group { display: grid; align-content: start; gap: var(--s3); }
  .model-group > p { color: var(--ink3); font: var(--type-small); }
  .model-list { display: grid; gap: var(--s2); }
  .model-row { display: flex; align-items: center; justify-content: space-between; gap: var(--s4); padding: var(--s3) var(--s4); border-radius: var(--r-box); background: var(--wash); color: var(--ink); }
  .model-row:active { scale: 0.99; }
  .model-row > :global(svg) { flex: none; color: var(--ink3); }
  .model-copy { display: grid; gap: 2px; min-width: 0; }
  .model-copy strong { font: var(--type-control); }
  .model-copy span { color: var(--ink3); font: var(--type-meta); }
  .model-note { color: var(--ink3); font: var(--type-small); }

  .guide { display: grid; gap: var(--s5); }
  .guide-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 260px), 1fr)); gap: var(--s2); margin: 0; }
  dt { color: var(--ink); font: var(--type-label); }

  .faq { display: grid; gap: var(--s2); max-width: 820px; }
  details { border-radius: var(--r-box); background: var(--wash); interpolate-size: allow-keywords; }
  summary { display: flex; align-items: center; justify-content: space-between; gap: var(--s4); min-height: 48px; padding: var(--s3) var(--s4); border-radius: var(--r-box); color: var(--ink); font: var(--type-control); list-style: none; cursor: pointer; }
  summary::-webkit-details-marker { display: none; }
  summary > :global(svg) { flex: none; color: var(--ink3); transition: rotate var(--glide) var(--ease); }
  details[open] summary > :global(svg) { rotate: 180deg; }
  details::details-content { block-size: 0; overflow: hidden; opacity: 0; transition: block-size var(--glide) var(--ease), opacity var(--glide) var(--ease), content-visibility var(--glide) allow-discrete; }
  details[open]::details-content { block-size: auto; opacity: 1; }
  details p { padding: 0 var(--s4) var(--s4); color: var(--ink2); font: var(--type-small); }

  .closing { grid-template-columns: minmax(0, 1fr) auto; align-items: center; gap: var(--s5) var(--s7); }
  .python-band { display: grid; justify-items: start; gap: var(--s3); max-width: 60ch; }
  .python-band p { color: var(--ink2); }
  .closing-action { display: grid; justify-items: start; gap: var(--s2); }
  .closing-action p { color: var(--ink3); font: var(--type-small); }

  @media (hover: hover) {
    .primary-action:hover { opacity: 0.86; }
    .trust-line a:hover, .text-link:hover, .faq a:hover { color: var(--ink); text-decoration-color: var(--ink3); }
    .model-row:hover { background: var(--wash2); }
    .row:not([aria-pressed="true"]):hover { background: color-mix(in oklch, var(--wash) 60%, transparent); }
    .row:hover > :global(.tok) { background: var(--wash2); }
    summary:hover { background: var(--wash2); }
  }
  @media (max-width: 900px) {
    main { gap: var(--s8); }
    .hero { grid-template-columns: minmax(0, 1fr); padding-block: 0; }
    .reading { max-width: 560px; }
    .capabilities ol, .model-groups { grid-template-columns: minmax(0, 1fr); }
    .closing { grid-template-columns: minmax(0, 1fr); }
  }
  @media (max-width: 720px) {
    main { padding-top: var(--s4); }
    .demo-panel { padding: var(--s4); }
  }
  @media (max-width: 520px) { .next { grid-template-columns: max-content minmax(0, 1fr) auto; column-gap: var(--s3); } }
  @media (prefers-reduced-motion: reduce) { .bar i { animation: none; } }
  @media (forced-colors: active) {
    .primary-action { border: 1px solid ButtonText; }
    .slot { outline: 2px solid Highlight; box-shadow: none; }
    .bar i { forced-color-adjust: none; background: Highlight; }
    .row[aria-pressed="true"] { outline: 1px solid Highlight; }
  }
</style>
