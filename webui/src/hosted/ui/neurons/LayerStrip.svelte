<script lang="ts">
  import MorphText from "../../../../../drowse/webui/src/lib/ui/MorphText.svelte";
  import InfoTip from "../../../../../drowse/webui/src/lib/ui/InfoTip.svelte";
  import FluentIcon from "../../../../../drowse/webui/src/lib/ui/FluentIcon.svelte";
  import LabelText from "./LabelText.svelte";
  import { BANDS, MIDDLE_CAUTION, bandOf, type StripCard, type StripFilter } from "./data";

  // The selected layer: its name, filters, and its neurons or heads as a row of cards.
  let { layer, unitName, cards, filters, selected, onselect, hrefFor }: {
    layer: number;
    /** "neurons" or "heads", for labels. */
    unitName: string;
    cards: StripCard[];
    filters: StripFilter[];
    selected: number | null;
    onselect: (key: number) => void;
    hrefFor: (key: number) => string;
  } = $props();

  let filter = $state("");
  const active = $derived(filters.find((f) => f.value === filter) ?? filters[0]);
  const available = $derived(filters.filter((f) => !f.test || cards.some(f.test)));
  const shown = $derived.by(() => {
    const list = active?.test ? cards.filter(active.test) : cards;
    return active?.limit ? list.slice(0, active.limit) : list;
  });
  // Long lists (every neuron of a layer) render 48 cards at a time.
  const PAGE = 48;
  let visible = $state(PAGE);
  $effect.pre(() => { void layer; void filter; visible = PAGE; });
  const rendered = $derived(shown.slice(0, visible));
  // Each layer opens on the first filter; a unit picked elsewhere that isn't in view switches to the widest one.
  $effect.pre(() => { void layer; void unitName; filter = filters[0]?.value ?? ""; });
  $effect(() => {
    if (selected === null || !cards.length) return;
    if (!shown.some((card) => card.key === selected)) {
      const wide = filters.find((f) => !f.test && !f.limit);
      if (wide && filter !== wide.value) filter = wide.value;
      return;
    }
    const at = shown.findIndex((card) => card.key === selected);
    if (at >= visible) visible = Math.ceil((at + 1) / PAGE) * PAGE;
  });

  const band = $derived(BANDS[bandOf(layer)]);
  let track: HTMLOListElement | undefined = $state();

  // Keep the selected card in view as the selection moves (j/k, the ribbon, search).
  $effect(() => {
    void selected;
    const card = track?.querySelector<HTMLElement>('[aria-current="true"]');
    if (!card || !track) return;
    const left = card.offsetLeft - track.clientWidth / 2 + card.clientWidth / 2;
    track.scrollTo({ left, behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  });

  function page(delta: -1 | 1) {
    track?.scrollBy({ left: delta * track.clientWidth * 0.8, behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  }

  function open(event: MouseEvent, key: number) {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    onselect(key);
  }
</script>

<section class="strip" aria-labelledby="layer-title">
  <header class="strip-head">
    <div class="title">
      <h2 id="layer-title" tabindex="-1"><MorphText text={`Layer ${layer}`} duration={320} /></h2>
      <span class="band"><MorphText text={band.name} numbers={false} duration={260} /></span>
      <span class="about"><InfoTip icon="info" label={`About ${band.name.toLowerCase()}`} text={`${band.summary} ${band.reading}${bandOf(layer) === "dark" ? ` ${MIDDLE_CAUTION}` : ""}`} /></span>
    </div>
    <div class="controls">
      {#if available.length > 1}
        <div class="filters" role="group" aria-label={`Filter ${unitName}`}>
          {#each available as f (f.value)}
            <button type="button" aria-pressed={active?.value === f.value} onclick={() => (filter = f.value)}>{f.label}</button>
          {/each}
        </div>
      {/if}
      <div class="pagers">
        <button type="button" class="pager" aria-label={`Scroll ${unitName} back`} onclick={() => page(-1)}><FluentIcon name="back" size={18} /></button>
        <button type="button" class="pager" aria-label={`Scroll ${unitName} forward`} onclick={() => page(1)}><FluentIcon name="next" size={18} /></button>
      </div>
    </div>
  </header>

  {#if !cards.length}
    <p class="empty">Loading {unitName}…</p>
  {:else}
    <div class="track-wrap">
      <ol class="track" bind:this={track}>
        {#each rendered as card, i (card.key)}
          <li style:--i={Math.min(i, 10)}>
            <a class="card" href={hrefFor(card.key)} aria-current={selected === card.key ? "true" : undefined} onclick={(event) => open(event, card.key)}>
              <span class="card-top">
                <span class="card-id">{card.id}</span>
                {#each card.tags as tag (tag.label)}<span class="tag" style:--tag={tag.color}>{tag.label}</span>{/each}
              </span>
              {#each card.lines as line, j (j)}
                <span class="card-line" class:first={j === 0} class:low={line.conf < 0.25}>
                  <span class="mark" aria-hidden="true">{line.mark}</span><span class="sr-only">{line.markLabel}: </span>
                  <span class="t"><LabelText text={line.text} /></span>
                </span>
              {/each}
            </a>
          </li>
        {/each}
        {#if shown.length > visible}
          <li class="more-cell"><button type="button" class="more-card" onclick={() => (visible += PAGE)}>Show {Math.min(PAGE, shown.length - visible)} more<span>{visible.toLocaleString("en-US")} of {shown.length.toLocaleString("en-US")} shown</span></button></li>
        {/if}
      </ol>
      <span class="fade" aria-hidden="true"></span>
    </div>
  {/if}
</section>

<style>
  .strip { display: grid; gap: var(--space-4); min-width: 0; }
  .strip-head { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: var(--space-3) var(--space-5); }
  .title { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-2) var(--space-3); }
  h2 { margin: 0; color: var(--fg); font-family: var(--font-structure); font-weight: var(--weight-display); font-size: var(--text-feature-title); line-height: 1.1; letter-spacing: -0.03em; font-variant-numeric: tabular-nums; }
  h2:focus { outline: none; }
  .band { color: var(--neuron-tint-ink); font-size: var(--text-md); font-weight: var(--weight-structure); }
  .about { display: inline-flex; color: var(--fg-muted); }

  .filters { display: flex; flex-wrap: wrap; gap: var(--space-1); padding: var(--space-1); border-radius: var(--radius-pill); background: var(--neuron-chip-bg); box-shadow: inset 0 0 0 1px var(--glass-line); }
  .filters button { min-height: 32px; padding-inline: var(--space-4); border: 0; border-radius: var(--radius-pill); background: transparent; color: var(--fg-dim); font: inherit; font-size: var(--text-sm); cursor: pointer; transition: background-color var(--dur-fast) var(--ease-out), color var(--dur-fast) var(--ease-out), scale var(--dur-fast) var(--ease-out); }
  .filters button[aria-pressed="true"] { background: var(--fg); color: var(--bg); }
  .filters button:active { scale: var(--press-scale); }
  .filters button:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  @media (hover: hover) { .filters button:not([aria-pressed="true"]):hover { background: var(--glass-strong); color: var(--fg); } }

  .track-wrap { position: relative; min-width: 0; }
  /* A fade at the right edge says the row scrolls on. */
  .fade { position: absolute; inset: 0 0 var(--space-3) auto; width: var(--space-8); background: linear-gradient(to right, transparent, var(--bg)); pointer-events: none; }
  .track {
    display: grid; grid-auto-flow: column; grid-auto-columns: minmax(15rem, 17rem); gap: var(--space-3);
    margin: 0; padding: var(--space-1) var(--space-8) var(--space-3) var(--space-1); overflow-x: auto; scroll-snap-type: x proximity;
    scroll-padding-inline: var(--space-1); list-style: none; scrollbar-width: thin; overscroll-behavior-inline: contain;
  }
  .track li { scroll-snap-align: start; animation: card-in 360ms var(--ease-enter) backwards; animation-delay: calc(var(--i) * 30ms); }
  @keyframes card-in { from { opacity: 0; translate: 12px 0; } }
  .card {
    position: relative; display: grid; align-content: start; gap: var(--space-2); height: 100%; padding: var(--space-4) var(--space-5);
    border-radius: var(--radius-lg); background: var(--neuron-panel); box-shadow: var(--neuron-panel-edge); color: var(--fg); text-decoration: none;
    transition: box-shadow var(--dur) var(--ease-out), translate var(--dur) var(--ease-out), background-color var(--dur) var(--ease-out);
  }
  .card[aria-current="true"] { box-shadow: inset 0 0 0 2px var(--neuron-tint), var(--neuron-panel-lift); background: color-mix(in oklch, var(--neuron-tint) 10%, var(--neuron-panel)); }
  .card[aria-current="true"]::after { content: ""; position: absolute; inset: auto var(--space-5) 0; height: 3px; border-radius: var(--radius-pill) var(--radius-pill) 0 0; background: var(--neuron-tint); }
  .card:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  @media (hover: hover) { .card:hover { translate: 0 -3px; box-shadow: var(--neuron-panel-edge), var(--neuron-panel-lift); } }
  .card-top { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-2); }
  .card-id { color: var(--neuron-tint-ink); font-family: var(--font-mono); font-size: var(--text-sm); font-variant-numeric: tabular-nums; }
  .tag { display: inline-flex; align-items: center; gap: var(--space-1); padding: 0 var(--space-2); border-radius: var(--radius-pill); background: color-mix(in oklch, var(--tag, var(--fg-muted)) 16%, transparent); color: var(--fg-dim); font-size: var(--text-2xs); line-height: 1.7; }
  .card-line { display: grid; grid-template-columns: auto minmax(0, 1fr); gap: var(--space-2); min-width: 0; color: var(--fg-dim); font-size: var(--text-sm); line-height: 1.4; }
  .card-line.first { color: var(--fg); font-size: var(--text-md); }
  .card-line.low { color: var(--fg-muted); }
  .mark { min-width: 1ch; color: var(--fg-muted); font-size: var(--text-xs); line-height: 1.9; }
  .t { display: -webkit-box; -webkit-line-clamp: 2; line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; overflow-wrap: anywhere; }

  .controls { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-3); }
  .pagers { display: flex; gap: var(--space-2); }
  .pager {
    display: grid; place-items: center; width: var(--control-target); height: var(--control-target);
    border: 0; border-radius: 50%; background: var(--neuron-chip-bg); box-shadow: inset 0 0 0 1px var(--glass-line); color: var(--fg); cursor: pointer;
    transition: scale var(--dur-fast) var(--ease-out), background-color var(--dur-fast) var(--ease-out);
  }
  @media (hover: hover) { .pager:hover { background: var(--glass-strong); } }
  .pager:active { scale: var(--press-scale); }
  .pager:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  .empty { margin: 0; color: var(--fg-muted); }
  .more-card { display: grid; place-content: center; gap: var(--space-1); width: 100%; height: 100%; min-height: 120px; border: 0; border-radius: var(--radius-lg); background: transparent; box-shadow: inset 0 0 0 1px var(--glass-line); color: var(--fg); font: inherit; font-size: var(--text-md); cursor: pointer; transition: background-color var(--dur-fast) var(--ease-out), scale var(--dur-fast) var(--ease-out); }
  .more-card span { color: var(--fg-muted); font-size: var(--text-xs); }
  .more-card:active { scale: var(--press-scale); }
  .more-card:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  @media (hover: hover) { .more-card:hover { background: var(--bg-hover); } }
  .sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
  @media (max-width: 640px) { .pager { display: none; } .track { grid-auto-columns: minmax(78%, 1fr); } }
  @media (prefers-reduced-motion: reduce) {
    .track li { animation: none; }
    .card, .filters button, .pager { transition: none; }
    .card:hover { translate: none; }
    .filters button:active, .pager:active { scale: none; }
  }
</style>
