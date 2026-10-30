<script lang="ts">
  import { onMount, tick } from "svelte";
  import TabIdentity from "../../../../../drowse/webui/src/lib/ui/TabIdentity.svelte";
  import MorphText from "../../../../../drowse/webui/src/lib/ui/MorphText.svelte";
  import PageHeader from "../../../../../drowse/webui/src/lib/ui/PageHeader.svelte";
  import PageFooter from "../PageFooter.svelte";
  import PwaUpdatePrompt from "../PwaUpdatePrompt.svelte";
  import DepthRibbon from "../neurons/DepthRibbon.svelte";
  import HeadGrid from "../neurons/HeadGrid.svelte";
  import LayerStrip from "../neurons/LayerStrip.svelte";
  import NeuronDetail from "../neurons/NeuronDetail.svelte";
  import HeadDetail from "../neurons/HeadDetail.svelte";
  import NeuronSearch from "../neurons/NeuronSearch.svelte";
  import { tokenHover } from "../neurons/tokenHover";
  import { restoreTranslate } from "../neurons/translate.svelte";
  import {
    FLAG_DETECTOR, FLAG_MINUS_WRITES, FLAG_PLUS_WRITES, FLAG_PREDICTS, FLAG_SUPPRESSES, HEADS, LAYERS, MLP, ROLES,
    ROLE_COPY, ROLE_DUPLICATE, ROLE_INDUCTION, ROLE_NEXT, TOTAL_HEADS, TOTAL_NEURONS,
    chunkOf, depthColor, headId, loadHeadIndex, loadHeadLayer, loadIndex, loadLayerIndex, loadNeuronChunk, loadScores, neuronId, rolesOf,
    type CatalogIndex, type HeadDetail as Head, type HeadRow, type IndexRow, type NeuronDetail as Neuron, type StripCard, type StripFilter, type Unit,
  } from "../neurons/data";

  const format = (value: number) => value.toLocaleString("en-US");
  const PATH = "/neuron-lens";

  let index = $state<CatalogIndex | null>(null);
  let headRows = $state<HeadRow[] | null>(null);
  let failed = $state(false);
  let unit = $state<Unit>("neurons");
  let layer = $state(8);
  /** The neuron or head named in the URL. Without one, the layer's easiest is shown. */
  let neuron = $state<number | null>(null);
  let head = $state<number | null>(null);
  // "All" lists every one of the 442,368 neurons; the default lists the easiest to read.
  let everyNeuron = $state(false);
  let allScores = $state<Uint8Array[] | null>(null);
  let layerRows = $state<{ layer: number; rows: IndexRow[] } | null>(null);
  let neuronChunk = $state<{ layer: number; chunk: number; neurons: Neuron[] } | null>(null);
  let headLayer = $state<{ layer: number; heads: Head[] } | null>(null);
  let highlight = $state<{ neurons: Set<string>; heads: Set<string> }>({ neurons: new Set(), heads: new Set() });
  let roleFocus = $state<number | null>(null);
  let mounted = $state(false);
  let explorer: HTMLElement | undefined = $state();
  let stage: HTMLElement | undefined = $state();
  // Sticky panes sit just under the site header, whose height changes with width and zoom.
  let headerHeight = $state(92);
  // The headline figures count up once the catalog arrives (Torph morphs each step).
  let neuronCount = $state(0);
  let headCount = $state(0);

  // ---------- what the strip lists, and the selection

  const curatedRows = $derived((index?.neurons ?? []).filter((row) => row[0] === layer).sort((a, b) => b[4] - a[4]));
  const allRows = $derived(layerRows?.layer === layer ? [...layerRows.rows].sort((a, b) => b[4] - a[4]) : []);
  const neuronRows = $derived(everyNeuron ? allRows : curatedRows);
  const layerHeads = $derived((headRows ?? []).filter((row) => row[0] === layer).sort((a, b) => b[10] - a[10]));
  const order = $derived(unit === "heads" ? layerHeads.map((row) => row[1]) : neuronRows.map((row) => row[1]));
  const selected = $derived(unit === "heads" ? head ?? layerHeads[0]?.[1] ?? null : neuron ?? neuronRows[0]?.[1] ?? null);
  const position = $derived(selected === null ? -1 : order.indexOf(selected));
  const chunkSize = $derived(index?.chunk ?? 64);
  const neuronDetail = $derived(
    unit !== "neurons" || selected === null || neuronChunk?.layer !== layer || neuronChunk.chunk !== chunkOf(selected, chunkSize) ? null
    : neuronChunk.neurons.find((n) => n.n === selected) ?? null,
  );
  const headDetail = $derived(unit !== "heads" || selected === null || headLayer?.layer !== layer ? null : headLayer.heads.find((h) => h.h === selected) ?? null);

  const writesMark = (writes: boolean) => (writes ? "→ " : "");
  const neuronCards = $derived(neuronRows.map((row): StripCard => ({
    key: row[1],
    id: neuronId(row[0], row[1]),
    lines: [
      { mark: "+", markLabel: (row[5] & FLAG_PLUS_WRITES) ? "Plus pole writes" : "Plus pole", text: writesMark((row[5] & FLAG_PLUS_WRITES) !== 0) + row[2], conf: row[6] / 100 },
      { mark: "−", markLabel: (row[5] & FLAG_MINUS_WRITES) ? "Minus pole writes" : "Minus pole", text: writesMark((row[5] & FLAG_MINUS_WRITES) !== 0) + row[3], conf: row[7] / 100 },
    ],
    tags: [
      ...((row[5] & FLAG_PREDICTS) ? [{ label: "Predicts" }] : []),
      ...((row[5] & FLAG_SUPPRESSES) ? [{ label: "Suppresses" }] : []),
      ...((row[5] & FLAG_DETECTOR) ? [{ label: "One token" }] : []),
    ],
    flags: row[5],
  })));
  const headCards = $derived(layerHeads.map((row): StripCard => ({
    key: row[1],
    id: headId(row[0], row[1]),
    lines: [
      { mark: "attends", markLabel: "Attends to", text: row[4], conf: row[7] / 100 },
      { mark: "writes", markLabel: "Writes", text: row[5], conf: row[8] / 100 },
    ],
    tags: rolesOf(row[9]).slice(0, 2).map((role) => ({ label: role.short, color: role.color })),
    flags: row[9],
  })));
  const neuronFilters = $derived<StripFilter[]>([
    { value: "easiest", label: "Easiest", limit: 12 },
    { value: "all", label: `All ${format(neuronRows.length)}` },
    { value: "predicts", label: "Predictors", test: (card) => (card.flags & FLAG_PREDICTS) !== 0 },
    { value: "suppresses", label: "Suppressors", test: (card) => (card.flags & FLAG_SUPPRESSES) !== 0 },
    { value: "detector", label: "One-token", test: (card) => (card.flags & FLAG_DETECTOR) !== 0 },
  ]);
  const headFilters: StripFilter[] = [
    { value: "all", label: `All ${HEADS}` },
    ...ROLES.filter((role) => [ROLE_INDUCTION, ROLE_DUPLICATE, ROLE_NEXT, ROLE_COPY].includes(role.bit))
      .map((role) => ({ value: String(role.bit), label: role.short, test: (card: StripCard) => (card.flags & role.bit) !== 0 })),
  ];

  // ---------- URL state: /neuron-lens?layer=8&neuron=344, ?layer=20&head=17

  function readUrl() {
    const params = new URLSearchParams(window.location.search);
    const L = Number(params.get("layer"));
    const n = params.get("neuron");
    const h = params.get("head");
    const validHead = h !== null && /^\d{1,2}$/.test(h) && Number(h) < HEADS;
    return {
      layer: params.has("layer") && Number.isInteger(L) && L >= 0 && L < LAYERS ? L : validHead ? 20 : 8,
      neuron: n !== null && /^\d{1,5}$/.test(n) && Number(n) < MLP ? Number(n) : null,
      head: validHead ? Number(h) : null,
      unit: (validHead || params.get("unit") === "heads" ? "heads" : "neurons") as Unit,
      all: params.get("all") === "1",
    };
  }

  function urlFor(u: Unit, L: number, key: number | null): string {
    const params = new URLSearchParams({ layer: String(L) });
    if (u === "heads") {
      if (key !== null) params.set("head", String(key));
      else params.set("unit", "heads");
    } else {
      if (key !== null) params.set("neuron", String(key));
      if (everyNeuron) params.set("all", "1");
    }
    return `${PATH}?${params}`;
  }

  function writeUrl(push: boolean) {
    const url = urlFor(unit, layer, unit === "heads" ? head : neuron);
    if (url === window.location.pathname + window.location.search) return;
    history[push ? "pushState" : "replaceState"]({ neuronLens: true }, "", url);
  }

  // ---------- loading

  async function ensureHeads(L: number) {
    failed = false;
    try {
      if (!headRows) headRows = await loadHeadIndex();
      if (headLayer?.layer !== L) {
        const heads = await loadHeadLayer(L);
        if (layer === L) headLayer = { layer: L, heads };
      }
    } catch {
      if (layer === L) failed = true;
    }
  }

  async function ensureNeurons(L: number) {
    failed = false;
    try {
      if (everyNeuron) {
        if (!allScores) loadScores().then((scores) => { allScores = scores; }).catch(() => undefined);
        if (layerRows?.layer !== L) {
          const rows = await loadLayerIndex(L);
          if (layer === L) layerRows = { layer: L, rows };
        }
      }
      const n = neuron ?? neuronRows[0]?.[1];
      if (n === undefined || n === null) return;
      const chunk = chunkOf(n, chunkSize);
      if (neuronChunk?.layer === L && neuronChunk.chunk === chunk) return;
      const neurons = await loadNeuronChunk(L, chunk);
      if (layer === L) neuronChunk = { layer: L, chunk, neurons };
    } catch {
      if (layer === L) failed = true;
    }
  }

  const ensure = (L: number) => (unit === "heads" ? ensureHeads(L) : ensureNeurons(L));

  // The detail follows the selection: stepping can cross a chunk boundary, and the curated list
  // arrives after the first request.
  $effect(() => {
    if (unit !== "neurons" || selected === null || !index) return;
    if (neuronChunk?.layer === layer && neuronChunk.chunk === chunkOf(selected, chunkSize)) return;
    void ensureNeurons(layer);
  });

  // In All mode the ribbon's hover tip reads from per-layer indexes, loaded on first hover.
  let peekCache = $state<Record<number, IndexRow[]>>({});
  const peekRequested = new Set<number>();
  function peek(L: number, n: number): IndexRow | null {
    const rows = peekCache[L];
    if (rows) return rows[n] ?? null;
    if (!peekRequested.has(L)) {
      peekRequested.add(L);
      loadLayerIndex(L).then((file) => {
        const byNeuron: IndexRow[] = [];
        for (const row of file) byNeuron[row[1]] = row;
        peekCache = { ...peekCache, [L]: byNeuron };
      }).catch(() => peekRequested.delete(L));
    }
    return null;
  }

  // ---------- navigation

  function setUnit(next: Unit) {
    if (unit === next) return;
    unit = next;
    roleFocus = null;
    writeUrl(true);
    void ensure(layer);
  }

  function setEveryNeuron(on: boolean) {
    if (everyNeuron === on) return;
    everyNeuron = on;
    writeUrl(true);
    void ensureNeurons(layer);
  }

  /** Scrubbing replaces the history entry; a deliberate choice adds one. */
  function setLayer(L: number, commit: boolean) {
    if (L === layer && (unit === "heads" ? head : neuron) === null) return;
    layer = L;
    neuron = null;
    head = null;
    writeUrl(commit);
    void ensure(L);
  }

  async function openUnit(u: Unit, L: number, key: number, reveal = false, focus = true) {
    unit = u;
    layer = L;
    if (u === "heads") head = key;
    else {
      neuron = key;
      // A neuron outside the easiest set is listed among all of its layer's neurons.
      if (!everyNeuron && index && !index.neurons.some((row) => row[0] === L && row[1] === key)) everyNeuron = true;
    }
    writeUrl(true);
    void ensure(L);
    await tick();
    // Move focus to the unit, so the next Tab continues in it rather than back in the list.
    if (focus) document.getElementById("unit-title")?.focus({ preventScroll: true });
    const target = reveal ? explorer : stage;
    if (target) {
      const top = target.getBoundingClientRect().top;
      if (top < 0 || top > window.innerHeight * 0.7) target.scrollIntoView({ block: "start", behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
    }
  }

  function step(delta: -1 | 1) {
    if (!order.length || position < 0) return;
    const next = order[(position + delta + order.length) % order.length];
    if (unit === "heads") head = next;
    else neuron = next;
    writeUrl(false);
  }

  const stepHref = (delta: -1 | 1) => order.length && position >= 0
    ? urlFor(unit, layer, order[(position + delta + order.length) % order.length])
    : urlFor(unit, layer, selected);

  function pageKeys(event: KeyboardEvent) {
    const target = event.target as HTMLElement | null;
    if (event.metaKey || event.ctrlKey || event.altKey || target?.closest("input, textarea, select, [role='slider']")) return;
    if (event.key === "j" || event.key === "k") { event.preventDefault(); step(event.key === "j" ? 1 : -1); }
  }

  function countUp() {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) { neuronCount = TOTAL_NEURONS; headCount = TOTAL_HEADS; return; }
    const steps = [0.02, 0.09, 0.22, 0.41, 0.63, 0.82, 0.94, 1];
    steps.forEach((share, i) => setTimeout(() => { neuronCount = Math.round(TOTAL_NEURONS * share); headCount = Math.round(TOTAL_HEADS * share); }, 160 * (i + 1)));
  }

  function applyUrl() {
    const state = readUrl();
    unit = state.unit;
    layer = state.layer;
    neuron = state.neuron;
    head = state.head;
    everyNeuron = state.all;
    void ensure(state.layer);
  }

  onMount(() => {
    mounted = true;
    restoreTranslate();
    applyUrl();
    loadIndex().then((loaded) => {
      index = loaded;
      countUp();
      // A neuron named in the URL that isn't among the easiest opens in the All list.
      if (unit === "neurons" && neuron !== null && !everyNeuron && !loaded.neurons.some((row) => row[0] === layer && row[1] === neuron)) {
        everyNeuron = true;
        writeUrl(false);
        void ensureNeurons(layer);
      }
    }).catch(() => { failed = true; });
    // Heads are few: their index loads up front, for search and the Heads view.
    loadHeadIndex().then((rows) => { headRows = rows; }).catch(() => undefined);
    window.addEventListener("popstate", applyUrl);
    document.addEventListener("keydown", pageKeys);
    const header = document.querySelector<HTMLElement>(".page-header");
    const resize = header ? new ResizeObserver(() => (headerHeight = header.offsetHeight)) : null;
    if (header) resize!.observe(header);
    return () => {
      resize?.disconnect();
      window.removeEventListener("popstate", applyUrl);
      document.removeEventListener("keydown", pageKeys);
    };
  });

  const selectedId = $derived(selected === null ? null : unit === "heads" ? headId(layer, selected) : neuronId(layer, selected));
  const pageTitle = $derived(selectedId && (unit === "heads" ? head : neuron) !== null ? `${selectedId} · Neuron Lens · Drowse` : "Neuron Lens · Drowse");
  const loadingDetail = $derived(unit === "heads" ? !headDetail : !neuronDetail);
</script>

<TabIdentity title={pageTitle} />
{#if mounted}<PwaUpdatePrompt />{/if}

<div class="lens-page" style:--neuron-tint={depthColor(layer)} style:--neuron-sticky-top={`${headerHeight + 8}px`}>
  <a class="skip-link" href="#explore">Skip to the explorer</a>
  <PageHeader current="neurons" />
  <main id="neuron-lens-main" use:tokenHover>
    <section class="intro" aria-labelledby="lens-title">
      <h1 id="lens-title">Neuron Lens</h1>
      <p class="lede">What the neurons and attention heads inside Qwen3-8B respond to and what they make the model say, read from their weights and from 15 million tokens of text.</p>
      <dl class="figures">
        <div><dt>Neurons</dt><dd><MorphText text={format(neuronCount)} duration={200} /></dd></div>
        <div><dt>Attention heads</dt><dd><MorphText text={format(headCount)} duration={200} /></dd></div>
        <div><dt>Layers</dt><dd>{LAYERS}</dd></div>
      </dl>
      <NeuronSearch {index} heads={headRows} anyNeuron onopen={(u, L, k) => void openUnit(u, L, k, true)} onmatches={(keys) => (highlight = keys)} examples={["owl", "Bible", "she", "money", "圣经", "L20h17"]} />
    </section>

    <section class="explorer" id="explore" aria-label="Explorer" bind:this={explorer} tabindex="-1">
      <div class="panel ribbon-panel">
        <div class="ribbon-bar">
          <div class="segmented" role="group" aria-label="Units">
            <button type="button" aria-pressed={unit === "neurons"} onclick={() => setUnit("neurons")}>Neurons</button>
            <button type="button" aria-pressed={unit === "heads"} onclick={() => setUnit("heads")}>Attention heads</button>
          </div>
          {#if unit === "neurons"}
            <div class="segmented" role="group" aria-label="Neurons listed">
              <button type="button" aria-pressed={!everyNeuron} onclick={() => setEveryNeuron(false)}>Easiest <span>{index ? format(index.neurons.length) : ""}</span></button>
              <button type="button" aria-pressed={everyNeuron} onclick={() => setEveryNeuron(true)}>All <span>{format(TOTAL_NEURONS)}</span></button>
            </div>
          {/if}
        </div>
        {#if unit === "heads"}
          <HeadGrid rows={headRows} {layer} head={selected} highlight={highlight.heads} role={roleFocus} onlayer={setLayer} onhead={(L, h) => void openUnit("heads", L, h, false, false)} onrole={(bit) => (roleFocus = bit)} />
        {:else}
          <DepthRibbon {index} {layer} neuron={selected} highlight={highlight.neurons} scores={everyNeuron ? allScores : null} {peek} onlayer={setLayer} onneuron={(L, n) => void openUnit("neurons", L, n)} />
          {#if everyNeuron}<p class="ribbon-caption">Each column holds a layer's 12,288 neurons in order. Brighter cells are easier to read.</p>{/if}
        {/if}
      </div>

      {#if failed && !index}
        <p class="panel notice" role="alert">Unable to load the catalog. Check your connection and reload the page.</p>
      {:else}
        {#if unit === "heads"}
          <LayerStrip {layer} unitName="heads" cards={headCards} filters={headFilters} {selected} onselect={(h) => void openUnit("heads", layer, h)} hrefFor={(h) => urlFor("heads", layer, h)} />
        {:else}
          <LayerStrip {layer} unitName="neurons" cards={neuronCards} filters={neuronFilters} {selected} onselect={(n) => void openUnit("neurons", layer, n)} hrefFor={(n) => urlFor("neurons", layer, n)} />
        {/if}

        <div class="panel stage" bind:this={stage}>
          {#if unit === "heads" && headDetail}
            <HeadDetail head={headDetail} {layer} onstep={step} position={Math.max(0, position)} total={order.length} hrefFor={stepHref} />
          {:else if unit === "neurons" && neuronDetail}
            <NeuronDetail neuron={neuronDetail} {layer} onstep={step} position={Math.max(0, position)} total={order.length} hrefFor={stepHref} />
          {:else if failed}
            <div class="notice" role="alert">
              <p>Unable to load layer {layer}. Check your connection and try again.</p>
              <button type="button" class="text-button" onclick={() => void ensure(layer)}>Try again</button>
            </div>
          {:else if loadingDetail}
            <div class="skeleton" aria-live="polite" aria-busy="true">
              <span class="sr-only">Loading layer {layer}…</span>
              <span class="bone wide"></span><span class="bone block short"></span>
              <span class="bone"></span><span class="bone block"></span>
            </div>
          {/if}
        </div>
      {/if}
    </section>
  </main>
  <PageFooter current="neurons" />
</div>

<style>
  .lens-page {
    --neuron-tint-ink: color-mix(in oklch, var(--neuron-tint) 62%, var(--fg));
    --neuron-translated-ink: var(--accent-amber);
    /* Surfaces: hairline-edged panels in dark; white cards on soft shadows in light. */
    --neuron-panel: var(--bg-alt);
    --neuron-panel-edge: inset 0 0 0 1px var(--glass-line);
    --neuron-panel-lift: var(--shadow-card);
    --neuron-chip-bg: var(--glass);
    --neuron-well: var(--bg);
    display: flex; flex-direction: column; min-height: 100svh; overflow-x: clip; background: var(--bg); color: var(--fg);
    -webkit-font-smoothing: antialiased;
  }
  :global(:root[data-theme="light"]) .lens-page {
    --neuron-tint-ink: color-mix(in oklch, var(--neuron-tint) 65%, var(--fg));
    --neuron-panel-edge: var(--shadow-card);
    --neuron-panel-lift: var(--shadow-card-active);
    --neuron-chip-bg: var(--bg-alt);
    --neuron-well: var(--bg);
  }
  .skip-link { position: fixed; inset: 8px auto auto 8px; z-index: 10; padding: var(--space-sm); background: var(--accent); color: var(--text-on-accent); border-radius: var(--radius); text-decoration: none; transform: translateY(-160%); }
  .skip-link:focus { transform: none; }
  /* Flat panels, as in the workbench: no sheen gradients on this page's content. */
  main {
    --surface-sheen: none;
    --control-sheen: none;
    flex: 1; display: grid; gap: var(--space-xl); width: min(100%, var(--page-max)); margin-inline: auto;
    padding-inline: var(--page-gutter); padding-block-end: var(--section-space);
  }
  .panel { border-radius: var(--popup-radius); background: var(--neuron-panel); box-shadow: var(--neuron-panel-edge); }

  .intro { display: grid; justify-items: start; gap: var(--space-5); padding-block-start: var(--space-2xl); }
  h1 { margin: 0; color: var(--fg); font-family: var(--font-structure); font-weight: var(--weight-display); font-size: var(--text-explorer-title); line-height: 1.02; letter-spacing: -0.045em; text-wrap: balance; animation: rise-in 600ms var(--ease-enter) backwards; }
  .lede { max-width: 54ch; margin: 0; color: var(--fg-dim); font-size: var(--text-lg); line-height: 1.5; text-wrap: pretty; animation: rise-in 600ms var(--ease-enter) 80ms backwards; }
  @keyframes rise-in { from { opacity: 0; translate: 0 10px; } }
  .figures { display: flex; flex-wrap: wrap; gap: var(--space-4) var(--space-xl); margin: 0; animation: rise-in 600ms var(--ease-enter) 160ms backwards; }
  .figures div { display: grid; gap: 0; }
  .figures dt { color: var(--fg-muted); font-size: var(--text-sm); }
  .figures dd { margin: 0; color: var(--fg); font-family: var(--font-structure); font-size: var(--text-feature-title); font-weight: var(--weight-display); font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }

  .explorer { display: grid; gap: var(--space-6); min-width: 0; scroll-margin-top: var(--neuron-sticky-top); }
  .explorer:focus { outline: none; }
  .ribbon-panel { display: grid; gap: var(--space-4); padding: var(--space-5) var(--space-6); }
  .ribbon-bar { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: var(--space-3); }
  .ribbon-caption { margin: 0; color: var(--fg-muted); font-size: var(--text-sm); }
  /* One segmented control for every exclusive choice on the page (the strip's filters share it). */
  .segmented { display: inline-flex; gap: var(--space-1); padding: var(--space-1); border-radius: var(--radius-pill); background: var(--neuron-chip-bg); box-shadow: inset 0 0 0 1px var(--glass-line); }
  .segmented button { display: inline-flex; align-items: center; gap: var(--space-2); min-height: 32px; padding-inline: var(--space-4); border: 0; border-radius: var(--radius-pill); background: transparent; color: var(--fg-dim); font: inherit; font-size: var(--text-sm); font-weight: var(--weight-structure); cursor: pointer; transition: background-color var(--dur) var(--ease-out), color var(--dur) var(--ease-out), scale var(--dur-fast) var(--ease-out); }
  .segmented button span { color: var(--fg-muted); font-family: var(--font-mono); font-size: var(--text-xs); font-variant-numeric: tabular-nums; }
  .segmented button[aria-pressed="true"] { background: var(--fg); color: var(--bg); }
  .segmented button[aria-pressed="true"] span { color: color-mix(in srgb, var(--bg) 70%, var(--fg)); }
  .segmented button:active { scale: var(--press-scale); }
  .segmented button:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  @media (hover: hover) { .segmented button:not([aria-pressed="true"]):hover { background: var(--glass-strong); color: var(--fg); } }
  .stage { min-width: 0; padding: var(--space-6) var(--space-xl); scroll-margin-top: var(--neuron-sticky-top); }
  .notice { display: grid; justify-items: start; gap: var(--space-3); margin: 0; padding: var(--space-5); color: var(--fg-dim); font-size: var(--text-sm); }
  .notice p { margin: 0; }
  .text-button { min-height: var(--control-target); padding-inline: var(--space-4); border: 0; border-radius: var(--radius); background: var(--glass-strong); color: var(--fg); font: inherit; cursor: pointer; transition: background-color var(--dur-fast) var(--ease-out), scale var(--dur-fast) var(--ease-out); }
  .text-button:active { scale: var(--press-scale); }
  .text-button:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  @media (hover: hover) { .text-button:hover { background: var(--glass-bright); } }
  .skeleton { display: grid; gap: var(--space-3); }
  .bone { display: block; height: 14px; width: 40%; border-radius: var(--radius-sm); background: var(--glass-strong); animation: pulse 1.4s ease-in-out infinite; }
  .bone.wide { width: 24%; height: 32px; }
  .bone.block { width: 100%; height: 240px; border-radius: var(--radius-lg); }
  .bone.block.short { height: 160px; }
  @keyframes pulse { 50% { opacity: 0.5; } }
  .sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }

  @media (max-width: 760px) {
    .ribbon-panel { padding: var(--space-4) var(--space-3) var(--space-3); }
    .stage { padding: var(--space-5) var(--space-4); }
  }
  @media (prefers-reduced-motion: reduce) {
    h1, .lede, .figures, .bone { animation: none; }
    .segmented button, .text-button { transition: none; }
    .segmented button:active, .text-button:active { scale: none; }
  }

  :global(.term-tip) {
    position: absolute; left: 0; top: 0; z-index: 60; display: grid; gap: 0; width: max-content;
    padding: var(--space-3) var(--space-4); border-radius: var(--radius); background: var(--bg-elev); box-shadow: var(--shadow-card);
    color: var(--fg); font-size: var(--text-sm); line-height: 1.4;
    opacity: 0; translate: 0 3px; transition: opacity var(--dur-fast) var(--ease-out), translate var(--dur-fast) var(--ease-out);
  }
  :global(.term-tip[hidden]) { display: none; }
  :global(.term-tip.shown) { opacity: 1; translate: 0 0; }
  :global(.term-tip .tip-word) { font-size: var(--text-md); font-weight: var(--weight-structure); }
  :global(.term-tip .tip-roman) { color: var(--fg-dim); font-family: var(--font-mono); font-size: var(--text-xs); }
  :global(.term-tip .tip-meaning) { color: var(--fg); }
  :global(.term-tip .tip-original) { color: var(--fg); font-family: var(--font-mono); white-space: pre-wrap; overflow-wrap: anywhere; }
  :global(.term-tip .tip-meta) { margin-top: var(--space-1); color: var(--fg-muted); font-size: var(--text-xs); }
  @media (prefers-reduced-motion: reduce) { :global(.term-tip) { transition: none; } }
</style>
