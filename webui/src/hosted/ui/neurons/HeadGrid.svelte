<script lang="ts">
  import LabelText from "./LabelText.svelte";
  import { ROLES, describeRole, headId, primaryRole, rolesOf, type HeadRow } from "./data";
  import { bandOf, bandRange, depthColor, type ModelInfo } from "./models";

  // Every attention head as one cell, layers left to right and heads top to bottom, coloured by its role.
  let { model, rows, layer, head, highlight, role = null, onlayer, onhead, onrole }: {
    model: ModelInfo;
    rows: HeadRow[] | null;
    layer: number;
    head: number | null;
    /** "layer:head" keys to light up, such as search matches. */
    highlight: Set<string>;
    /** A role bit to light up, from the legend. */
    role?: number | null;
    onlayer: (layer: number, commit: boolean) => void;
    onhead: (layer: number, head: number) => void;
    onrole: (bit: number | null) => void;
  } = $props();

  const COL = 32;
  const TOP = 20;
  const LAYERS = $derived(model.layers);
  const HEADS = $derived(model.heads);
  // Fewer heads a layer get taller cells: Qwen3-8B's 32 at 10 units, Gemma 4's 16 at 16.
  const PITCH = $derived(Math.max(10, Math.round(256 / model.heads)));
  const W = $derived(COL * LAYERS);
  const PLOT = $derived(TOP + HEADS * PITCH);
  const H = $derived(PLOT + 64);
  const LAYER_LIST = $derived(Array.from({ length: LAYERS }, (_, L) => L));
  const global = $derived(new Set(model.globalLayers));

  const byKey = $derived(new Map((rows ?? []).map((row) => [`${row[0]}:${row[1]}`, row])));
  const rankOf = $derived.by(() => {
    const scores = (rows ?? []).map((row) => row[10]).sort((a, b) => a - b);
    return (score: number) => {
      let lo = 0, hi = scores.length;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (scores[mid] < score) lo = mid + 1; else hi = mid; }
      return scores.length ? lo / scores.length : 0;
    };
  });
  // The legend lists the roles this model's heads have.
  const legend = $derived(ROLES.map((r) => ({ role: r, count: (rows ?? []).filter((row) => (row[9] & r.bit) !== 0).length })).filter((x) => x.count > 0));

  let svg: SVGSVGElement | undefined = $state();
  let scroller: HTMLDivElement | undefined = $state();
  // The drawn width, measured once mounted; until then, about the viewBox.
  let drawn = $state(1200);
  const k = $derived(W / Math.max(drawn, 1));
  let hover = $state<{ layer: number; head: number | null; x: number; y: number; flip: boolean } | null>(null);
  const tipRow = $derived(hover && hover.head !== null ? byKey.get(`${hover.layer}:${hover.head}`) ?? null : null);

  function locate(event: PointerEvent) {
    if (!svg || !scroller) return null;
    const box = svg.getBoundingClientRect();
    const x = ((event.clientX - box.left) / box.width) * W;
    const y = ((event.clientY - box.top) / box.height) * H;
    const L = Math.max(0, Math.min(LAYERS - 1, Math.floor(x / COL)));
    const h = Math.floor((y - TOP) / PITCH);
    const outer = scroller.getBoundingClientRect();
    const px = event.clientX - outer.left;
    const py = Math.max(90, Math.min(outer.height - 90, event.clientY - outer.top));
    return { layer: L, head: h >= 0 && h < HEADS ? h : null, x: px, y: py, flip: px > outer.width - 300 };
  }

  function up(event: PointerEvent) {
    if (event.button !== 0) return;
    const spot = locate(event);
    if (!spot) return;
    if (spot.head !== null) onhead(spot.layer, spot.head);
    else onlayer(spot.layer, true);
    if (event.pointerType !== "mouse") hover = null;
  }

  function key(event: KeyboardEvent) {
    const current = head ?? 0;
    if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
      event.preventDefault();
      onlayer(Math.max(0, Math.min(LAYERS - 1, layer + (event.key === "ArrowRight" ? 1 : -1))), true);
    } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      onhead(layer, Math.max(0, Math.min(HEADS - 1, current + (event.key === "ArrowDown" ? 1 : -1))));
    } else if (event.key === "Home" || event.key === "End") {
      event.preventDefault();
      onlayer(event.key === "Home" ? 0 : LAYERS - 1, true);
    } else if (event.key === "Enter") {
      event.preventDefault();
      onhead(layer, current);
    }
  }

  // On narrow screens the grid scrolls sideways; keep the selected layer in view.
  $effect(() => {
    void layer;
    if (!scroller || scroller.scrollWidth <= scroller.clientWidth + 1) return;
    const colWidth = scroller.scrollWidth / LAYERS;
    scroller.scrollTo({ left: layer * colWidth - scroller.clientWidth / 2 + colWidth / 2, behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  });

  const valueText = $derived(`Layer ${layer}${head !== null ? `, head ${head}` : ""}, ${bandOf(model, layer).name}${global.has(layer) ? ", a global layer" : ""}`);
</script>

<div class="grid-wrap" class:ready={!!rows}>
  <div class="scroller" bind:this={scroller}>
    <svg
      bind:this={svg}
      bind:clientWidth={drawn}
      viewBox={`0 0 ${W} ${H}`}
      role="slider"
      tabindex="0"
      aria-label="Layer and head"
      aria-valuemin={0}
      aria-valuemax={LAYERS - 1}
      aria-valuenow={layer}
      aria-valuetext={valueText}
      aria-describedby="head-grid-help"
      onpointermove={(event) => (hover = locate(event))}
      onpointerup={up}
      onpointerleave={() => (hover = null)}
      onkeydown={key}
    >
      {#each LAYER_LIST as L}
        <rect class="col-bg" class:active={L === layer} class:hot={hover?.layer === L && L !== layer} x={L * COL + 2} y={TOP - 8} width={COL - 4} height={HEADS * PITCH + 12} rx="8" style:--c={depthColor(model, L)} />
      {/each}
      {#each rows ?? [] as row (`${row[0]}:${row[1]}`)}
        {@const r = primaryRole(row[9] & ~128)}
        {@const lit = highlight.has(`${row[0]}:${row[1]}`) || (role !== null && (row[9] & role) !== 0)}
        <rect
          class="cell"
          class:dark={(row[9] & 128) !== 0 && !r}
          class:lit
          class:dim={(highlight.size > 0 || role !== null) && !lit}
          class:selected={row[0] === layer && row[1] === head}
          class:pointed={hover?.layer === row[0] && hover?.head === row[1]}
          x={row[0] * COL + 5}
          y={TOP + row[1] * PITCH + 1}
          width={COL - 10}
          height={PITCH - 2}
          rx="2"
          style:--c={r ? r.color : depthColor(model, row[0])}
          style:--o={r ? 0.55 + 0.45 * rankOf(row[10]) : 0.16 + 0.5 * rankOf(row[10])}
          style:--L={row[0]}
        />
      {/each}
      {#each LAYER_LIST as L}
        <text class="layer-num" class:active={L === layer} x={L * COL + COL / 2} y={PLOT + 18} font-size={11 * Math.min(k, 1.25)}>{L}</text>
        {#if global.has(L)}<circle class="global-mark" class:active={L === layer} cx={L * COL + COL / 2} cy={TOP - 13} r="2.5" />{/if}
      {/each}
      {#each model.bands as b (b.key)}
        <g class="band" class:active={bandOf(model, layer) === b}>
          <line x1={b.first * COL + 6} x2={(b.last + 1) * COL - 6} y1={PLOT + 32} y2={PLOT + 32} />
          <text x={((b.first + b.last + 1) * COL) / 2} y={PLOT + 52} font-size={12 * Math.min(k, 1.25)}>{b.last - b.first >= 2 ? b.name.replace(" layers", "") : bandRange(b)}</text>
        </g>
      {/each}
    </svg>

    {#if hover}
      <div class="tip" class:flip={hover.flip} style:--x={`${hover.x}px`} style:--y={`${hover.y}px`} aria-hidden="true">
        {#if tipRow}
          <strong>{headId(tipRow[0], tipRow[1])}</strong>
          {#if rolesOf(tipRow[9]).length}<span class="tip-roles">{rolesOf(tipRow[9]).map((x) => x.short).join(" · ")}</span>{/if}
          <span class="tip-over"><em>Attends to</em><LabelText text={tipRow[4]} /></span>
          <span class="tip-over"><em>Writes</em><LabelText text={tipRow[5]} /></span>
          <span class="tip-hint">Click to open</span>
        {:else}
          <strong>Layer {hover.layer}</strong>
          <span class="tip-line">{bandOf(model, hover.layer).name}{global.has(hover.layer) ? " · global attention" : model.globalLayers.length ? " · 1,024-token window" : ""}</span>
        {/if}
      </div>
    {/if}
  </div>
  <p id="head-grid-help" class="sr-only">Left and right arrows change layer, up and down change head, Enter opens it.</p>
  <div class="legend" role="group" aria-label="Light up heads by role">
    {#each legend as { role: r, count } (r.bit)}
      <button type="button" aria-pressed={role === r.bit} data-help={describeRole(r, model)} style:--c={r.color} onclick={() => onrole(role === r.bit ? null : r.bit)}>
        <span class="swatch" aria-hidden="true"></span>{r.short}<span class="count">{count}</span>
      </button>
    {/each}
    {#if model.globalLayers.length}
      <span class="global-key" data-help="Every sixth layer attends over the whole text; the others only within the last 1,024 tokens, with every two query heads sharing a key-value head."><span class="global-dot" aria-hidden="true"></span>Global layer</span>
    {/if}
  </div>
</div>

<style>
  .grid-wrap { position: relative; display: grid; gap: var(--space-3); min-width: 0; user-select: none; -webkit-user-select: none; }
  .scroller { position: relative; overflow-x: auto; overflow-y: visible; scrollbar-width: thin; overscroll-behavior-inline: contain; }
  svg { display: block; width: 100%; min-width: 56rem; height: auto; overflow: visible; cursor: pointer; border-radius: var(--radius-lg); }
  svg:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 4px; }
  .col-bg { fill: transparent; transition: fill var(--dur) var(--ease-out); }
  .col-bg.hot { fill: var(--bg-hover); }
  .col-bg.active { fill: color-mix(in oklch, var(--c) 14%, transparent); stroke: color-mix(in oklch, var(--c) 50%, transparent); stroke-width: 1; }
  .cell { fill: var(--c); opacity: var(--o); transform-box: fill-box; transform-origin: center; transition: opacity var(--dur) var(--ease-out), scale 180ms var(--ease-out); }
  .ready .cell { animation: cell-in 520ms var(--ease-enter) backwards; animation-delay: calc(var(--L) * 24ms); }
  @keyframes cell-in { from { opacity: 0; scale: 0.4; } }
  .cell.dark { fill: none; stroke: var(--c); stroke-width: 1; stroke-opacity: 0.5; }
  .cell.dim { opacity: 0.1; }
  .cell.lit { opacity: 1; }
  .cell.pointed { scale: 1.35; opacity: 1; }
  .cell.selected { opacity: 1; stroke: var(--fg); stroke-width: 1.6; scale: 1.25; }
  .layer-num { fill: var(--fg-muted); font-family: var(--font-mono); text-anchor: middle; font-variant-numeric: tabular-nums; }
  .layer-num.active { fill: var(--fg); font-weight: var(--weight-data-bold); }
  .global-mark { fill: var(--fg-muted); }
  .global-mark.active { fill: var(--fg); }
  .band line { stroke: var(--grid-line); stroke-width: 3; stroke-linecap: round; }
  .band text { fill: var(--fg-muted); font-family: var(--font-reading); text-anchor: middle; }
  .band.active line { stroke: var(--neuron-tint); }
  .band.active text { fill: var(--fg); }

  .tip {
    position: absolute; left: 0; top: 0; z-index: 3; display: grid; gap: var(--space-1); width: max-content; max-width: 20rem;
    padding: var(--space-3) var(--space-4); border-radius: var(--radius); background: var(--bg-elev); box-shadow: var(--shadow-card);
    color: var(--fg-dim); font-size: var(--text-sm); line-height: 1.4; pointer-events: none;
    transform: translate(calc(var(--x) + 16px), calc(var(--y) - 50%));
    animation: tip-in 140ms var(--ease-out) backwards;
  }
  .tip.flip { transform: translate(calc(var(--x) - 100% - 16px), calc(var(--y) - 50%)); }
  @keyframes tip-in { from { opacity: 0; } }
  .tip strong { color: var(--fg); font-family: var(--font-mono); font-size: var(--text-md); }
  .tip-roles { color: var(--fg); font-size: var(--text-xs); }
  .tip-over { display: grid; gap: 0; color: var(--fg); }
  .tip-over em { color: var(--fg-muted); font-size: var(--text-xs); font-style: normal; }
  .tip-line, .tip-hint { color: var(--fg-muted); font-size: var(--text-xs); }

  .legend { display: flex; flex-wrap: wrap; gap: var(--space-2); }
  .legend button { display: inline-flex; align-items: center; gap: var(--space-2); min-height: 32px; padding-inline: var(--space-3); border: 0; border-radius: var(--radius-pill); background: transparent; box-shadow: inset 0 0 0 1px var(--glass-line); color: var(--fg-dim); font: inherit; font-size: var(--text-xs); cursor: pointer; transition: background-color var(--dur-fast) var(--ease-out), color var(--dur-fast) var(--ease-out); }
  .legend button[aria-pressed="true"] { background: color-mix(in oklch, var(--c) 18%, transparent); box-shadow: inset 0 0 0 1px var(--c); color: var(--fg); }
  .legend button:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  @media (hover: hover) { .legend button:not([aria-pressed="true"]):hover { background: var(--bg-hover); color: var(--fg); } }
  .swatch { width: 10px; height: 8px; border-radius: var(--data-mark-radius); background: var(--c); }
  .count { color: var(--fg-muted); font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
  .global-key { display: inline-flex; align-items: center; gap: var(--space-2); min-height: 32px; padding-inline: var(--space-2); color: var(--fg-muted); font-size: var(--text-xs); }
  .global-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--fg-muted); }
  .sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
  @media (prefers-reduced-motion: reduce) { .ready .cell, .tip { animation: none; } .cell, .col-bg { transition: none; } .cell { scale: 1; } }
  @media (forced-colors: active) { .cell { fill: CanvasText; } .col-bg.active { stroke: Highlight; } }
</style>
