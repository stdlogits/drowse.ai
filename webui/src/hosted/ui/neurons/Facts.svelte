<script lang="ts">
  import { percent } from "./data";

  // How hard and how often a unit fires, and on which kinds of text.
  let { act, peakLabel = "Typical peak", rateLabel = "Fires on" }: {
    act: { top: number; rate: number; slices: [number, number, number, number] };
    peakLabel?: string;
    rateLabel?: string;
  } = $props();

  const sliceNames = ["Web", "Chinese", "Code", "Chat"];
  const sliceMax = $derived(Math.max(...act.slices, 1e-6));
</script>

<dl class="facts">
  <div><dt>{peakLabel}</dt><dd>{act.top.toFixed(2)}</dd></div>
  <div><dt>{rateLabel}</dt><dd>{percent(act.rate)} of tokens</dd></div>
  <div class="slices">
    <dt>Where</dt>
    <dd>
      <span class="slice-bars" role="img" aria-label={sliceNames.map((name, index) => `${name} ${percent(act.slices[index])}`).join(", ")}>
        {#each act.slices as value, index}
          <span class="slice"><span class="slice-bar"><span style:--h={Math.max(0.06, value / sliceMax)}></span></span><span class="slice-name">{sliceNames[index]}</span></span>
        {/each}
      </span>
    </dd>
  </div>
</dl>

<style>
  .facts { display: flex; flex-wrap: wrap; gap: var(--space-5) var(--space-8); margin: 0; }
  .facts > div { display: grid; gap: var(--space-1); align-content: start; }
  dt { color: var(--fg-muted); font-size: var(--text-xs); }
  dd { margin: 0; color: var(--fg); font-family: var(--font-mono); font-size: var(--text-sm); font-variant-numeric: tabular-nums; white-space: nowrap; }
  .slice-bars { display: flex; align-items: flex-end; gap: var(--space-2); }
  .slice { display: grid; justify-items: center; gap: 0; }
  .slice-bar { position: relative; display: block; width: 10px; height: 22px; border-radius: var(--data-mark-radius); background: var(--glass-strong); overflow: hidden; }
  .slice-bar span { position: absolute; inset: 0; background: var(--neuron-tint); border-radius: inherit; transform-origin: bottom; scale: 1 var(--h); }
  .slice-name { color: var(--fg-muted); font-family: var(--font-reading); font-size: var(--text-2xs); }
</style>
