<script lang="ts">
  import DetailBar from "./DetailBar.svelte";
  import LabelCell from "./LabelCell.svelte";
  import LabelText from "./LabelText.svelte";
  import PoleColumn from "./PoleColumn.svelte";
  import { BANDS, LENSES, MIDDLE_CAUTION, bandOf, headline, neuronId, type LensToken, type NeuronDetail, type Pole } from "./data";

  let { neuron, layer, onstep, position, total, hrefFor }: {
    neuron: NeuronDetail;
    layer: number;
    /** Move to the previous (-1) or next (+1) shown neuron in the layer. */
    onstep: (delta: -1 | 1) => void;
    position: number;
    total: number;
    hrefFor: (delta: -1 | 1) => string;
  } = $props();

  const clearer = $derived(neuron.poles[0].score >= neuron.poles[1].score ? 0 : 1);
  // On phones the two poles share one column; start on the clearer one.
  let shown = $state<0 | 1>(0);
  let lastNeuron = -1;
  $effect.pre(() => {
    if (neuron.n !== lastNeuron) { lastNeuron = neuron.n; shown = clearer as 0 | 1; }
  });

  // Tabs only on phones, where one pole shows at a time; on wider screens both are plain columns.
  // Measured on the pane itself, since the rail beside it takes a varying share of the window.
  let width = $state(1000);
  const narrow = $derived(width < 720);
  const band = $derived(bandOf(layer));

  // Arrow keys move between the two pole tabs (APG tabs pattern).
  function tabKey(event: KeyboardEvent) {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    shown = event.key === "Home" ? 0 : event.key === "End" ? 1 : (shown === 0 ? 1 : 0);
    document.getElementById(`pole-tab-${shown}`)?.focus();
  }

  /** The raw tokens under a label: the depth's primary read, or in the middle layers the exact drivers. */
  function evidence(pole: Pole, side: "input" | "output"): { tokens: LensToken[]; source: string; help: string } {
    if (side === "input" && band === "dark" && pole.drivers.length) {
      return {
        tokens: headline(pole.drivers).map(([token]): LensToken => [token, 1, 2]),
        source: "in real text",
        help: "Exact drivers: tokens that most make it fire when put into real documents",
      };
    }
    const read = pole.reads.find((r) => r.trust === "primary" && LENSES[r.col].side === side);
    if (!read) return { tokens: [], source: "", help: "" };
    return { tokens: headline(read.tokens), source: LENSES[read.col].short, help: LENSES[read.col].describe };
  }
</script>

<section class="detail" class:narrow aria-labelledby="unit-title" data-link-scope bind:clientWidth={width}>
  <DetailBar id={neuronId(layer, neuron.n)} unitName="neuron" caution={band === "dark" ? MIDDLE_CAUTION : undefined} {onstep} {position} {total} {hrefFor}>
    {#snippet tags()}<span class="band-tag">{BANDS[band].name}</span>{/snippet}
  </DetailBar>

  {#key neuron.n}
  <section class="summary" aria-label="What each side responds to and what it does">
    <div class="sum-grid">
      <span class="col-head first">Triggered by</span>
      <span class="col-head">Steers outputs</span>
      {#each neuron.poles as pole, index (pole.pole)}
        {@const input = evidence(pole, "input")}
        {@const output = evidence(pole, "output")}
        <div class="row-head" class:clear={clearer === index}>
          <span class="sign" aria-hidden="true">{index === 0 ? "+" : "−"}</span>
          <span>{index === 0 ? "Positive side" : "Negative side"}</span>
          {#if clearer === index}<span class="clear-tag">clearer</span>{/if}
        </div>
        <div class="cell"><span class="cell-head">Triggered by</span><LabelCell label={pole.label.fires} confidence={pole.label.fc} tokens={input.tokens} source={input.source} sourceHelp={input.help} /></div>
        <div class="cell"><span class="cell-head">Steers outputs</span><LabelCell label={pole.label.writes} confidence={pole.label.wc} tokens={output.tokens} source={output.source} sourceHelp={output.help} /></div>
        {#if pole.label.note}<p class="note"><LabelText text={pole.label.note} interactive /></p>{/if}
      {/each}
    </div>
    <p class="credit">Labels by Claude Haiku 5.5, written from these reads and the first firings below. Confidence is Haiku's own.</p>
  </section>
  {/key}

  <div class="pole-switch" role="tablist" aria-label="Side" hidden={!narrow}>
    {#each [0, 1] as index}
      <button type="button" role="tab" id={`pole-tab-${index}`} aria-selected={shown === index} aria-controls={`pole-panel-${index}`} tabindex={shown === index ? 0 : -1} onclick={() => (shown = index as 0 | 1)} onkeydown={tabKey}>
        {index === 0 ? "+ Positive side" : "− Negative side"}{#if clearer === index}<span class="clear-dot" aria-hidden="true"></span><span class="sr-only">, clearer</span>{/if}
      </button>
    {/each}
  </div>

  {#key neuron.n}
  <div class="poles">
    {#each neuron.poles as pole, index (pole.pole)}
      <div class="pole-panel" class:hidden-phone={shown !== index} id={`pole-panel-${index}`} role={narrow ? "tabpanel" : undefined} aria-labelledby={narrow ? `pole-tab-${index}` : undefined}>
        <PoleColumn {pole} {layer} clearer={clearer === index} />
      </div>
    {/each}
  </div>
  {/key}
</section>

<style>
  .detail { display: grid; gap: var(--space-6); min-width: 0; }
  /* Stepping to another neuron slides the new one in; it plays once per change, never on hover. */
  .summary, .poles { animation: rise 380ms var(--ease-enter) backwards; }
  .poles { animation-delay: 60ms; }
  @keyframes rise { from { opacity: 0; translate: 0 8px; } }
  .band-tag { color: var(--neuron-tint-ink); font-size: var(--text-sm); font-weight: var(--weight-structure); }

  .summary { display: grid; gap: var(--space-3); padding: var(--space-5) var(--space-6); border-radius: var(--radius-lg); background: color-mix(in oklch, var(--neuron-tint) 6%, var(--neuron-chip-bg)); box-shadow: inset 0 0 0 1px color-mix(in oklch, var(--neuron-tint) 26%, transparent); }
  .sum-grid { display: grid; grid-template-columns: minmax(8.5rem, auto) minmax(0, 1fr) minmax(0, 1fr); gap: var(--space-4) var(--space-6); align-items: start; }
  .col-head { color: var(--neuron-tint-ink); font-size: var(--text-sm); font-weight: var(--weight-structure-bold); }
  .col-head.first { grid-column: 2; }
  .row-head { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-2); padding-top: 2px; color: var(--fg-dim); font-size: var(--text-sm); font-weight: var(--weight-structure); }
  .row-head.clear { color: var(--fg); }
  .sign { display: grid; place-items: center; flex: none; width: 24px; height: 24px; border-radius: var(--radius-sm); background: color-mix(in oklch, var(--neuron-tint) 20%, var(--neuron-chip-bg)); color: var(--fg); font-family: var(--font-structure); font-weight: var(--weight-display); line-height: 1; }
  .clear-tag { padding: 0 var(--space-2); border-radius: var(--radius-pill); background: color-mix(in oklch, var(--neuron-tint) 20%, transparent); color: var(--fg); font-size: var(--text-2xs); line-height: 1.7; }
  .cell { min-width: 0; }
  .cell-head { display: none; }
  .note { grid-column: 2 / -1; margin: calc(var(--space-2) * -1) 0 var(--space-2); color: var(--fg-muted); font-size: var(--text-sm); line-height: 1.55; text-wrap: pretty; }
  .credit { margin: 0; padding-top: var(--space-3); border-top: 1px solid var(--glass-line); color: var(--fg-muted); font-size: var(--text-xs); }

  .poles { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--space-8); align-items: start; }
  .pole-switch { display: none; }
  .sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
  .narrow {
    .summary { padding: var(--space-4); }
    .sum-grid { grid-template-columns: minmax(0, 1fr); gap: var(--space-3); }
    .col-head { display: none; }
    .row-head { padding-top: var(--space-2); }
    .row-head:not(:first-of-type) { margin-top: var(--space-2); border-top: 1px solid var(--glass-line); padding-top: var(--space-4); }
    .cell-head { display: block; margin-bottom: var(--space-1); color: var(--neuron-tint-ink); font-size: var(--text-xs); font-weight: var(--weight-structure-bold); }
    .note { grid-column: 1; margin: 0; }
    .poles { grid-template-columns: minmax(0, 1fr); }
    .pole-switch:not([hidden]) { position: sticky; top: var(--neuron-sticky-top); z-index: 2; display: grid; grid-template-columns: 1fr 1fr; gap: 0; padding: 0; border-radius: var(--radius); background: var(--bg-elev); }
    .pole-switch button { display: inline-flex; align-items: center; justify-content: center; gap: var(--space-2); min-height: var(--control-target); border: 0; border-radius: var(--radius); background: transparent; color: var(--fg-dim); font: inherit; font-size: var(--text-sm); cursor: pointer; transition: background-color var(--dur-fast) var(--ease-out), color var(--dur-fast) var(--ease-out); }
    .pole-switch button[aria-selected="true"] { background: var(--glass-bright); color: var(--fg); }
    .pole-switch button:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
    .clear-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--neuron-tint); }
    .hidden-phone { display: none; }
  }
  @media (prefers-reduced-motion: reduce) { .summary, .poles { animation: none; } .pole-switch button { transition: none; } }
</style>
