<script lang="ts">
  import { mediaUrl, type MediaDoc } from "./data";
  import { playOnly, sourceName } from "./media";
  import type { Modality, ModelId } from "./models";

  // One image or sound a pole fires on, as the model saw it: a thumbnail, or the 2.56 s it heard.
  let { model, modality, doc, value, peak }: { model: ModelId; modality: Modality; doc: MediaDoc; value: number; peak: number } = $props();

  let url = $state<string | null>(null);
  let failed = $state(false);
  $effect(() => {
    const hosted = doc.h;
    url = null;
    failed = false;
    if (!hosted) return;
    mediaUrl(model, modality, hosted).then((u) => { if (doc.h === hosted) url = u; }).catch(() => (failed = true));
  });

  const label = $derived(doc.l[0] ?? "unlabelled");
  const caption = $derived(doc.en ? `${doc.en[0]} (machine translated)` : doc.c);
  const help = $derived([caption, `${sourceName(doc.s)} · mean activation ${value.toFixed(2)}`].filter(Boolean).join(" · "));

  let stop: (() => void) | null = null;
  let playing = $state(false);
  function toggle() {
    if (!url) return;
    if (playing) { stop?.(); return; }
    playing = true;
    stop = playOnly(url, () => { playing = false; stop = null; });
  }
  $effect(() => () => stop?.());
</script>

{#if modality === "image"}
  <figure class="tile image" style:--v={Math.max(0.08, value / peak)} data-help={help}>
    <span class="frame">{#if url}<img src={url} alt={label} width="192" height="192" />{:else}<span class="ph" class:failed></span>{/if}</span>
    <span class="bar" aria-hidden="true"></span>
    <figcaption><span class="label">{label}</span><span class="src">{sourceName(doc.s)}</span></figcaption>
  </figure>
{:else}
  <button type="button" class="tile sound" style:--v={Math.max(0.08, value / peak)} data-help={help} aria-pressed={playing} disabled={!url} onclick={toggle}
    aria-label={`${playing ? "Stop" : "Play"} ${label}, from ${sourceName(doc.s)}`}>
    <span class="play" aria-hidden="true"><svg viewBox="0 0 24 24" width="14" height="14">{#if playing}<rect x="6" y="6" width="12" height="12" rx="2" />{:else}<path d="M8 5.5v13a1 1 0 0 0 1.5.86l10.5-6.5a1 1 0 0 0 0-1.72L9.5 4.64A1 1 0 0 0 8 5.5z" />{/if}</svg></span>
    <span class="label">{label}</span><span class="src">{sourceName(doc.s)}</span>
    <span class="bar" aria-hidden="true"></span>
  </button>
{/if}

<style>
  .tile { position: relative; display: grid; gap: var(--space-1); min-width: 0; margin: 0; overflow: hidden; border-radius: var(--radius); background: var(--neuron-well); box-shadow: inset 0 0 0 1px var(--glass-line); color: var(--fg); }
  .frame { display: block; aspect-ratio: 1; overflow: hidden; background: var(--glass); }
  .frame img { display: block; width: 100%; height: 100%; object-fit: cover; animation: fade 260ms var(--ease-out); }
  @keyframes fade { from { opacity: 0; } }
  .ph { display: block; width: 100%; height: 100%; animation: pulse 1.4s ease-in-out infinite; background: var(--glass-strong); }
  .ph.failed { animation: none; background: var(--glass); }
  @keyframes pulse { 50% { opacity: 0.5; } }
  figcaption { display: grid; gap: 0; padding: var(--space-1) var(--space-2) var(--space-2); }
  .label { overflow: hidden; font-size: var(--text-xs); line-height: 1.3; text-overflow: ellipsis; white-space: nowrap; }
  .src { overflow: hidden; color: var(--fg-muted); font-size: var(--text-2xs); text-overflow: ellipsis; white-space: nowrap; }
  .bar { position: absolute; inset: auto 0 0 0; height: 3px; background: var(--neuron-tint); transform-origin: left; scale: var(--v) 1; opacity: 0.8; }

  .sound { width: 100%; grid-template-columns: auto minmax(0, 1fr); grid-template-rows: auto auto; align-items: center; gap: 0 var(--space-2); padding: var(--space-2) var(--space-3) var(--space-3) var(--space-2); border: 0; font: inherit; text-align: start; cursor: pointer; transition: box-shadow var(--dur-fast) var(--ease-out); }
  .sound .play { grid-row: 1 / 3; display: grid; place-items: center; width: 30px; height: 30px; border-radius: 50%; background: var(--glass-strong); color: var(--fg); transition: background-color var(--dur-fast) var(--ease-out), color var(--dur-fast) var(--ease-out); }
  .sound .play svg { fill: currentColor; }
  .sound[aria-pressed="true"] .play { background: var(--fg); color: var(--bg); }
  .sound:disabled { cursor: default; }
  .sound:disabled .play { opacity: 0.4; }
  .sound:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  @media (hover: hover) { .sound:not(:disabled):hover { box-shadow: inset 0 0 0 1px var(--neuron-tint); } }
  @media (prefers-reduced-motion: reduce) { .ph { animation: none; } .frame img { animation: none; } .sound, .sound .play { transition: none; } }
</style>
