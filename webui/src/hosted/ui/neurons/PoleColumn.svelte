<script lang="ts">
  import LensReadView from "./LensReadView.svelte";
  import Token from "./Token.svelte";
  import Expander from "./Expander.svelte";
  import Context from "./Context.svelte";
  import Facts from "./Facts.svelte";
  import MediaEvidence from "./MediaEvidence.svelte";
  import { LENSES, type LensRead, type NeuronSide, type Pole, type PoleMedia } from "./data";
  import { MODALITY_NAMES, bandOf, writeVerb, type Modality, type ModelInfo } from "./models";

  // The evidence behind one pole's labels: its examples, the reads that hold at its depth, its images
  // and sounds (Gemma 4), and the rest on request.
  let { model, pole, layer, clearer, docBlock = 2048 }: { model: ModelInfo; pole: Pole; layer: number; clearer: boolean; docBlock?: number } = $props();

  const trusted = (side: "input" | "output") => pole.reads.filter((read) => LENSES[read.col].side === side && read.trust !== "weak");
  const inputs = $derived(trusted("input"));
  const outputs = $derived(trusted("output"));
  const others = $derived(pole.reads.filter((read) => read.trust === "weak"));
  // In the middle layers the examples and exact drivers describe a neuron better than any lens, so they come first.
  const middle = $derived(!!bandOf(model, layer).dark);
  // Reliable media labels lead; the others wait under More evidence.
  const mediaEntries = $derived(Object.entries(pole.media ?? {}) as [Modality, PoleMedia][]);
  const reliableMedia = $derived(mediaEntries.filter(([, m]) => m.reliable));
  const weakMedia = $derived(mediaEntries.filter(([, m]) => !m.reliable));
  const contexts = $derived(pole.contexts);
  const sideNames: Record<NeuronSide, string> = { read_S: "in·S", read_R: "in·R", write_S: "out·S", write_R: "out·R" };
  const holds = $derived((Object.entries(pole.legible) as [NeuronSide, number][]).filter(([, q]) => q <= 0.05));

  function twinName(read: LensRead): string | undefined {
    if (!read.twin) return undefined;
    const family = read.col.replace(/_z$/, "");
    const first = pole.reads.find((other) => other !== read && other.col.replace(/_z$/, "") === family);
    return first ? LENSES[first.col].short : undefined;
  }
</script>

<article class="pole" class:silent={pole.silent} aria-label={pole.pole > 0 ? "Positive side" : "Negative side"}>
  <header class="pole-head">
    <span class="sign" aria-hidden="true">{pole.pole > 0 ? "+" : "−"}</span>
    <h3>{pole.pole > 0 ? "Positive side" : "Negative side"}</h3>
    {#if clearer}<span class="badge tint">Clearer side</span>{/if}
    {#if pole.silent}<span class="badge">Silent</span>{/if}
    {#if pole.detector === "token"}<span class="badge">Fires on one token</span>{/if}
    {#if pole.detector === "next"}<span class="badge">Fires before one token</span>{/if}
    {#each reliableMedia as [modality] (modality)}<span class="badge tint">{modality === "image" ? "Images" : "Sound"}</span>{/each}
    {#if pole.shape?.kind === "predicts"}<span class="badge">Predicts the next word</span>{/if}
    {#if pole.shape?.kind === "suppresses"}<span class="badge">Holds back the next word</span>{/if}
  </header>

  {#snippet examples()}
    <section class="part">
      <h4>Where it fires</h4>
      {#if contexts.length}
        <div class="contexts">{#each contexts.slice(0, 3) as context, i (i)}<div style:--i={i}><Context {context} /></div>{/each}</div>
      {:else}
        <p class="empty">No firings recorded for this side.</p>
      {/if}
    </section>
  {/snippet}

  {#snippet drivers()}
    {#if pole.drivers.length}
      <section class="part">
        <h4>Words that set it off in real text</h4>
        <div class="chips">{#each pole.drivers.slice(0, 12) as [token, value]}<Token {token} {value} size="sm" />{/each}</div>
      </section>
    {/if}
  {/snippet}

  <!-- A reliable image or sound label is measured on held-out documents, not read from weights: it leads. -->
  {#each reliableMedia as [modality, media] (modality)}<MediaEvidence {model} {modality} {media} {docBlock} />{/each}

  {#if middle}{@render examples()}{@render drivers()}{/if}

  {#if inputs[0]}
    <section class="part">
      <h4>What triggers it</h4>
      <LensReadView read={inputs[0]} compact />
    </section>
  {/if}

  {#if outputs[0]}
    <section class="part">
      <h4>{writeVerb(model, layer)}</h4>
      <LensReadView read={outputs[0]} compact />
    </section>
  {/if}

  {#if !middle}{@render examples()}{/if}


  <Expander label="More evidence" openLabel="Less evidence">
    <div class="more">
      {#if pole.measured.length || pole.shared.length}
        <div class="agree">
          {#if pole.measured.length}
            <p class="agree-row"><span class="agree-key"><span class="dot measured" aria-hidden="true"></span>Confirmed on real text</span>
              <span class="chips">{#each pole.measured.slice(0, 10) as token}<Token {token} flags={2} size="sm" />{/each}</span></p>
          {/if}
          {#if pole.shared.length}
            <p class="agree-row"><span class="agree-key"><span class="dot shared" aria-hidden="true"></span>Lenses agree</span>
              <span class="chips">{#each pole.shared.slice(0, 10) as token}<Token {token} flags={1} size="sm" />{/each}</span></p>
          {/if}
        </div>
      {/if}

      <Facts act={pole.act} />

      <p class="tests">
        {#if holds.length}Reads that hold at its own top firings: {holds.map(([side, q]) => `${sideNames[side]} (q ${q < 0.001 ? "< 0.001" : q.toFixed(3)})`).join(", ")}.
        {:else if pole.silent}Too quiet to test its reads.
        {:else}No weight-only read holds at its own top firings{pole.detector ? ", but it fires on one token" : ""}.{/if}
      </p>

      {#each [...inputs.slice(1), ...outputs.slice(1)] as read (read.col)}<LensReadView {read} twinOf={twinName(read)} />{/each}

      {#if !middle}{@render drivers()}{/if}

      {#if contexts.length > 3}
        <div class="contexts">{#each contexts.slice(3) as context, i (i)}<Context {context} />{/each}</div>
      {/if}

      {#each weakMedia as [modality, media] (modality)}
        <p class="tests">{MODALITY_NAMES[modality].title}, it fires most on <b>{media.label}</b>, but the label isn't reliable (AUROC {media.auroc?.toFixed(2) ?? "–"}).</p>
      {/each}

      {#if others.length}
        <Expander label={`${others.length} weaker reads at this depth`} openLabel="Hide weaker reads">
          <div class="others">{#each others as read (read.col)}<LensReadView {read} twinOf={twinName(read)} />{/each}</div>
        </Expander>
      {/if}
    </div>
  </Expander>
</article>

<style>
  .pole { display: grid; align-content: start; gap: var(--space-5); min-width: 0; }
  .silent { opacity: 0.6; }
  .pole-head { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-2) var(--space-3); }
  .sign {
    display: grid; place-items: center; flex: none; width: 28px; height: 28px; border-radius: var(--radius);
    background: color-mix(in oklch, var(--neuron-tint) 20%, var(--neuron-chip-bg)); color: var(--fg);
    font-family: var(--font-structure); font-size: var(--text-md); font-weight: var(--weight-display); line-height: 1;
  }
  h3 { margin: 0; color: var(--fg); font-size: var(--text-md); font-weight: var(--weight-structure-bold); letter-spacing: -0.01em; }
  .badge { display: inline-flex; align-items: center; min-height: 22px; padding-inline: var(--space-2); border-radius: var(--radius-pill); background: var(--glass-strong); color: var(--fg-dim); font-size: var(--text-xs); white-space: nowrap; }
  .badge.tint { background: color-mix(in oklch, var(--neuron-tint) 18%, var(--glass)); color: var(--fg); }

  .part { display: grid; gap: var(--space-3); min-width: 0; }
  h4 { margin: 0; color: var(--fg-muted); font-size: var(--text-sm); font-weight: var(--weight-structure); }
  .contexts { display: grid; gap: var(--space-2); }
  .contexts > div { animation: ctx-in 320ms var(--ease-enter) backwards; animation-delay: calc(var(--i, 0) * 60ms); }
  @keyframes ctx-in { from { opacity: 0; translate: 0 4px; } }
  .empty { margin: 0; color: var(--fg-muted); font-size: var(--text-sm); }

  .more { display: grid; gap: var(--space-5); }
  .agree { display: grid; gap: var(--space-3); }
  .agree-row { display: grid; gap: var(--space-2); margin: 0; }
  .agree-key { display: inline-flex; align-items: center; gap: var(--space-2); color: var(--fg-muted); font-size: var(--text-xs); }
  .dot { width: 6px; height: 6px; border-radius: 50%; }
  .dot.measured { background: var(--accent-green); }
  .dot.shared { background: var(--neuron-tint); }
  .chips { display: flex; flex-wrap: wrap; gap: var(--space-1); min-width: 0; }
  .tests { margin: 0; color: var(--fg-dim); font-size: var(--text-sm); line-height: 1.55; }
  .tests b { color: var(--fg); font-weight: var(--weight-structure); }
  .others { display: grid; }
  @media (prefers-reduced-motion: reduce) { .contexts > div { animation: none; } }
</style>
