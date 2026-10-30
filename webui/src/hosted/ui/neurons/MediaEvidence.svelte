<script lang="ts">
  import MediaTile from "./MediaTile.svelte";
  import Token from "./Token.svelte";
  import { loadMediaDoc, type MediaDoc, type PoleMedia } from "./data";
  import { frameSamples, sourceName } from "./media";
  import { MODALITY_NAMES, type Modality, type ModelInfo } from "./models";

  // What one pole of a Gemma 4 neuron responds to in images or sound (findings §14, §15): its label,
  // the documents it fires on most, the patch or frame optimised to drive it, what it writes there,
  // and whether the same concept reaches it through text.
  let { model, modality, media, docBlock }: { model: ModelInfo; modality: Modality; media: PoleMedia; docBlock: number } = $props();

  const names = $derived(MODALITY_NAMES[modality]);
  let docs = $state<(MediaDoc | null)[]>([]);
  $effect(() => {
    const asked = media;
    docs = [];
    Promise.all(asked.top.map(([doc]) => loadMediaDoc(model.id, modality, doc, docBlock).catch(() => null)))
      .then((found) => { if (asked === media) docs = found; });
  });
  const peak = $derived(Math.max(...media.top.map(([, v]) => v), 1e-6));
  // Documents this site can show become tiles; the rest are grouped by label and linked to their source.
  const shown = $derived(media.top.map(([, value], i) => ({ doc: docs[i], value })).filter((x): x is { doc: MediaDoc; value: number } => !!x.doc?.h));
  const linked = $derived.by(() => {
    const groups = new Map<string, { label: string; source: string; link: string; count: number }>();
    for (const doc of docs) {
      if (!doc || doc.h) continue;
      const label = doc.l[0] ?? "unlabelled";
      const key = `${doc.s}:${label}`;
      const group = groups.get(key);
      if (group) group.count += 1;
      else groups.set(key, { label, source: doc.s, link: doc.u, count: 1 });
    }
    return [...groups.values()];
  });
  const loading = $derived(docs.length === 0);

  const fmt = (v: number | null, digits = 2) => (v === null ? "–" : v.toFixed(digits));
  const signed = (v: number | null) => (v === null ? "–" : `${v >= 0 ? "+" : ""}${Math.abs(v) >= 100 ? Math.round(v).toLocaleString("en-US") : v.toPrecision(3)}`);
  const RANKS = ["Its name fires it", "The text S-lens reads it", "The text R-lens writes it"] as const;
  const RANK_HELP = [
    "How hard the pole fires on \"Here is an example: <label>.\", as the label's rank among all labels (1 is first; past 0.95 passes, 5% by chance)",
    "The label's tokens in the pole's text S-lens read (in·S), as a rank among all labels",
    "The label's tokens in the R-lens read of the pole's write (out·R), as a rank among all labels",
  ];

  // ---------- the optimised sound: one 40 ms frame repeated over the 64 slots, as the model heard it

  let playing = $state(false);
  let context: AudioContext | null = null;
  let source: AudioBufferSourceNode | null = null;
  const frame = $derived(media.opt?.wave ? frameSamples(media.opt.wave) : null);
  const wavePath = $derived.by(() => {
    if (!frame) return "";
    const step = frame.length / 160;
    let d = "";
    for (let i = 0; i < 160; i += 1) d += `${i ? "L" : "M"}${i} ${20 - 18 * frame[Math.floor(i * step)]}`;
    return d;
  });

  function play() {
    if (!frame) return;
    if (playing) { source?.stop(); return; }
    context ??= new AudioContext();
    const buffer = context.createBuffer(1, frame.length * 64, 16000);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < 64; i += 1) data.set(frame, i * frame.length);
    // Fade the ends so the loop starts and stops without a click; keep it quiet.
    for (let i = 0; i < 400; i += 1) { data[i] *= i / 400; data[data.length - 1 - i] *= i / 400; }
    const gain = context.createGain();
    gain.gain.value = 0.35;
    source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(gain).connect(context.destination);
    source.onended = () => { playing = false; source = null; };
    source.start();
    playing = true;
  }
  $effect(() => () => { source?.stop(); void context?.close(); context = null; });
</script>

<section class="media" aria-label={names.title}>
  <header class="head">
    <h4>{names.title}</h4>
    <p class="label"><span class="name">{media.label}</span>
      <span class="stat" data-help={media.reliable
        ? `Past the shuffled-label null for its layer and in the top 5 of both halves of the ${names.plural}. AUROC is on held-out ${names.plural}, after choosing the label on the others; z is how far its mean activation on the label's ${names.plural} sits from its mean on all of them, toward this side, in standard deviations.`
        : `The label it fires on most, but it fails the shuffled-label null or isn't stable across halves of the ${names.plural}: a weak lead.`}>
        {media.reliable ? "Reliable" : "Not reliable"} · AUROC {fmt(media.auroc)} · z {fmt(media.z, 1)}
      </span>
    </p>
  </header>

  {#if loading}
    <div class="tiles pending" aria-hidden="true">{#each media.top.slice(0, 4) as [doc] (doc)}<span class="tile-ph"></span>{/each}</div>
  {:else}
    {#if shown.length}
      <ol class={modality === "image" ? "tiles" : "sounds"} aria-label={`The ${names.plural} it fires on most`}>
        {#each shown as { doc, value } (doc.h?.join(":"))}
          <li><MediaTile model={model.id} {modality} {doc} {value} {peak} /></li>
        {/each}
      </ol>
    {/if}
    {#if linked.length}
      <p class="linked">
        <span class="linked-head">{shown.length ? "Also" : `Its top ${names.plural}`}, from collections shown by link only:</span>
        {#each linked as group, i (`${group.source}:${group.label}`)}
          {#if i}<span class="sep" aria-hidden="true">·</span>{/if}
          {#if group.link}<a href={group.link} target="_blank" rel="noopener noreferrer">{group.label}</a>{:else}<span>{group.label}</span>{/if}{#if group.count > 1}<span class="count">×{group.count}</span>{/if}
          <span class="src">{sourceName(group.source)}</span>
        {/each}
      </p>
    {/if}
  {/if}

  {#if media.opt}
    {@const o = media.opt}
    <div class="opt">
      {#if o.png}
        <span class="patch" style:--png={`url(${o.png})`} role="img" aria-label="The patch optimised to drive this pole, repeated over the image"></span>
      {:else if frame}
        <button type="button" class="play" aria-pressed={playing} onclick={play} aria-label={playing ? "Stop the optimised sound" : "Play the optimised sound"}>
          <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">{#if playing}<rect x="6" y="6" width="12" height="12" rx="2" />{:else}<path d="M8 5.5v13a1 1 0 0 0 1.5.86l10.5-6.5a1 1 0 0 0 0-1.72L9.5 4.64A1 1 0 0 0 8 5.5z" />{/if}</svg>
        </button>
        <svg class="wave" viewBox="0 0 160 40" preserveAspectRatio="none" aria-hidden="true"><path d={wavePath} /></svg>
      {/if}
      <p class="opt-text">
        {o.png ? "A patch optimised to drive it, in every slot of an image." : "A 40 ms sound optimised to drive it, repeated for 2.56 s."}
        {#if o.gain !== null && o.gain > 0}On held-out {names.plural} it raises the pole by <b>{signed(o.gain)}</b>{#if o.lens !== null}, where the lens read it started from gives {signed(o.lens)}{/if}.
        {:else}It doesn't raise the pole on held-out {names.plural}.{/if}
      </p>
    </div>
  {/if}

  {#if media.write.length}
    <div class="chips-block">
      <h5 data-help={`The R-lens fitted on ${names.plural}, read on the pole's write: what it makes the model say in ${names.noun} contexts`}>In {names.noun} contexts it makes the model say</h5>
      <div class="chips">{#each media.write.slice(0, 10) as [token, count, flags]}<Token {token} {count} {flags} size="sm" />{/each}</div>
    </div>
  {/if}

  {#if media.ranks && media.reliable}
    <dl class="ranks" aria-label="The same label through text">
      {#each media.ranks as rank, i}
        <div class:pass={rank !== null && rank >= 0.95} data-help={RANK_HELP[i]}>
          <dt>{RANKS[i]}</dt>
          <dd><span class="meter" aria-hidden="true"><span style:--r={rank ?? 0}></span></span><span class="num">{fmt(rank)}</span></dd>
        </div>
      {/each}
    </dl>
  {/if}
</section>

<style>
  .media { container-type: inline-size; display: grid; gap: var(--space-4); min-width: 0; padding: var(--space-4) var(--space-5); border-radius: var(--radius-lg); background: var(--neuron-chip-bg); box-shadow: inset 0 0 0 1px var(--glass-line); }
  .head { display: grid; gap: var(--space-1); }
  h4 { margin: 0; color: var(--fg-muted); font-size: var(--text-sm); font-weight: var(--weight-structure); }
  .label { display: flex; flex-wrap: wrap; align-items: baseline; gap: var(--space-1) var(--space-3); margin: 0; }
  .name { color: var(--fg); font-family: var(--font-structure); font-size: var(--text-lg); font-weight: var(--weight-structure); letter-spacing: -0.01em; }
  .stat { color: var(--fg-muted); font-family: var(--font-mono); font-size: var(--text-2xs); }

  .tiles { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: var(--space-2); margin: 0; padding: 0; list-style: none; }
  .sounds { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--space-2); margin: 0; padding: 0; list-style: none; }
  .tiles.pending { min-height: 4rem; }
  .tile-ph { aspect-ratio: 1; border-radius: var(--radius); background: var(--glass-strong); animation: pulse 1.4s ease-in-out infinite; }
  @keyframes pulse { 50% { opacity: 0.5; } }
  .linked { margin: 0; color: var(--fg-dim); font-size: var(--text-sm); line-height: 1.7; }
  .linked-head { color: var(--fg-muted); font-size: var(--text-xs); margin-inline-end: var(--space-1); }
  .linked a { color: var(--fg); text-decoration-color: color-mix(in oklch, var(--fg) 30%, transparent); text-underline-offset: 2px; }
  .linked a:hover { text-decoration-color: var(--neuron-tint); }
  .linked .count { color: var(--fg-muted); font-family: var(--font-mono); font-size: var(--text-2xs); }
  .linked .src { margin-inline-start: var(--space-1); color: var(--fg-muted); font-size: var(--text-2xs); }
  .linked .sep { margin-inline: var(--space-2); color: var(--fg-muted); }

  .opt { display: grid; grid-template-columns: auto minmax(0, 1fr); align-items: center; gap: var(--space-2) var(--space-4); }
  .patch { width: 96px; height: 96px; border-radius: var(--radius); background-image: var(--png); background-size: 32px 32px; image-rendering: pixelated; box-shadow: inset 0 0 0 1px var(--glass-line); }
  .play { display: grid; place-items: center; width: 44px; height: 44px; border: 0; border-radius: 50%; background: var(--fg); color: var(--bg); cursor: pointer; transition: scale var(--dur-fast) var(--ease-out); }
  .play svg { fill: currentColor; }
  .play:active { scale: var(--press-scale); }
  .play:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  .wave { grid-column: 2; grid-row: 1; width: 100%; height: 40px; }
  .wave path { fill: none; stroke: var(--neuron-tint); stroke-width: 1.2; vector-effect: non-scaling-stroke; }
  .wave + .opt-text { grid-column: 1 / -1; }
  .opt-text { margin: 0; color: var(--fg-dim); font-size: var(--text-sm); line-height: 1.5; }
  .opt-text b { color: var(--fg); font-family: var(--font-mono); font-weight: var(--weight-data-bold); }

  .chips-block { display: grid; gap: var(--space-2); }
  h5 { justify-self: start; margin: 0; color: var(--fg-muted); font-size: var(--text-xs); font-weight: var(--weight-structure); }
  .chips { display: flex; flex-wrap: wrap; gap: var(--space-1); min-width: 0; }

  .ranks { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: var(--space-3); margin: 0; }
  .ranks div { display: grid; gap: var(--space-1); min-width: 0; }
  .ranks dt { color: var(--fg-muted); font-size: var(--text-2xs); line-height: 1.3; }
  .ranks dd { display: flex; align-items: center; gap: var(--space-2); margin: 0; }
  .meter { position: relative; flex: 1; height: 4px; border-radius: var(--radius-pill); background: var(--glass-strong); overflow: hidden; }
  .meter span { position: absolute; inset: 0; border-radius: inherit; background: var(--fg-muted); transform-origin: left; scale: var(--r) 1; }
  .pass .meter span { background: var(--neuron-tint); }
  .num { color: var(--fg-dim); font-family: var(--font-mono); font-size: var(--text-2xs); font-variant-numeric: tabular-nums; }
  .pass .num { color: var(--fg); }

  @container (max-width: 30rem) { .tiles { grid-template-columns: repeat(3, minmax(0, 1fr)); } .ranks { grid-template-columns: minmax(0, 1fr); } }
  @container (max-width: 17rem) { .sounds { grid-template-columns: minmax(0, 1fr); } }
  @media (prefers-reduced-motion: reduce) { .play { transition: none; } .tile-ph { animation: none; } }
</style>
