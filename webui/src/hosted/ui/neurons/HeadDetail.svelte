<script lang="ts">
  import DetailBar from "./DetailBar.svelte";
  import LabelCell from "./LabelCell.svelte";
  import LabelText from "./LabelText.svelte";
  import LensReadView from "./LensReadView.svelte";
  import Context from "./Context.svelte";
  import Expander from "./Expander.svelte";
  import Facts from "./Facts.svelte";
  import Token from "./Token.svelte";
  import {
    LENSES, PATTERN_NAMES, PATTERN_THRESHOLDS, describeRole, headId, headline, rolesOf,
    type HeadDetail, type HeadMedia, type LensRead, type LensToken, type ReadColumn,
  } from "./data";
  import { MODALITY_NAMES, bandOf, headBandNote, type Modality, type ModelInfo } from "./models";

  let { model, head, layer, labelled = true, groupSize, onstep, position, total, hrefFor }: {
    model: ModelInfo;
    head: HeadDetail;
    labelled?: boolean;
    /** Query heads that read this head's key-value head, itself included. */
    groupSize: number;
    layer: number;
    onstep: (delta: -1 | 1) => void;
    position: number;
    total: number;
    hrefFor: (delta: -1 | 1) => string;
  } = $props();

  const read = (col: ReadColumn) => head.reads.find((r) => r.col === col);
  const ovPrimary = $derived(head.reads.find((r) => r.trust === "primary" && r.col.startsWith("ov_") && !r.col.includes("neg") && !r.twin));
  const ovFamily = $derived(ovPrimary?.col ?? "ov_R");
  const shown = $derived(new Set<ReadColumn>(["qk_S", ovFamily, `${ovFamily}_neg` as ReadColumn, "qk_self_S"]));
  const more = $derived(head.reads.filter((r) => !shown.has(r.col)));
  const roles = $derived(rolesOf(head.roles));
  const band = $derived(bandOf(model, layer));
  const stat = $derived(model.headTest);
  const mediaEntries = $derived(Object.entries(head.media ?? {}) as [Modality, HeadMedia][]);
  const fmt = (v: number | null, digits = 2) => (v === null ? "–" : v.toFixed(digits));
  const tests = $derived([
    ["OV·R predicts the next token", head.legible.ov_R],
    ["it promotes what it attends to", head.legible.ov_copy],
  ].filter(([, q]) => (q as number) <= 0.05) as [string, number][]);

  const wordsOf = (pairs: [string, number][], flags = 2): LensToken[] => headline(pairs).map(([token]) => [token, 1, flags]);
  const topOf = (r: LensRead | undefined): LensToken[] => headline(r?.tokens ?? []);

  function twinName(r: LensRead): string | undefined {
    if (!r.twin) return undefined;
    const family = r.col.replace(/_z$/, "");
    const first = head.reads.find((other) => other !== r && other.col === family);
    return first ? LENSES[first.col].short : undefined;
  }
</script>

<section class="detail" aria-labelledby="unit-title" data-link-scope>
  <DetailBar id={headId(layer, head.h)} unitName="head" caution={band.dark ? model.caution : undefined} cautionLabel={model.cautionLabel} {onstep} {position} {total} {hrefFor}>
    {#snippet tags()}
      <span class="kv" data-help={groupSize > 1 ? `The key-value head it shares with ${groupSize - 1 === 1 ? "one other query head" : `${groupSize - 1} other query heads`}` : "Its own key-value head"}>kv {head.kv}</span>
      {#if model.globalLayers.includes(layer)}<span class="kv" data-help="This layer attends over the whole text, with one key-value head whose keys are also its values; the others attend within the last 1,024 tokens">global</span>{/if}
      {#each roles as role (role.bit)}<span class="role" style:--role={role.color} data-help={describeRole(role, model)}>{role.name}</span>{/each}
    {/snippet}
  </DetailBar>

  {#key head.h}
  <section class="summary" aria-label="Where it fires, what it attends to and what it writes">
    <div class="sum-grid">
      <div class="cell"><span class="cell-head">Fires when</span>
        <LabelCell label={head.label.fires} confidence={head.label.fc} tokens={wordsOf(head.drivers)} source="as the query" sourceHelp="Exact drivers: query tokens that make its output largest in real text" /></div>
      <div class="cell"><span class="cell-head">Attends to</span>
        <LabelCell label={head.label.attends} confidence={head.label.ac} tokens={wordsOf(head.keys)} source="as the key" sourceHelp="Key tokens it attends to most from the positions after them, in real text" /></div>
      <div class="cell"><span class="cell-head">Writes</span>
        <LabelCell label={head.label.writes} confidence={head.label.wc} tokens={topOf(ovPrimary)} source={ovPrimary ? LENSES[ovPrimary.col].short : ""} sourceHelp={ovPrimary ? LENSES[ovPrimary.col].describe : ""} /></div>
    </div>
    {#if head.label.note}<p class="note"><LabelText text={head.label.note} interactive /></p>{/if}
    {#if labelled}<p class="credit">Labels by Claude Haiku 5.5, written from these reads, the attention pattern and the first firings below. Confidence is Haiku's own. {model.labelChecks.heads}</p>{/if}
  </section>

  <div class="body">
    <section class="part pattern-part">
      <h3>Attention pattern</h3>
      <dl class="pattern">
        {#each PATTERN_NAMES as name, i}
          {@const value = head.pattern[i]}
          <div class:on={value > PATTERN_THRESHOLDS[i]}>
            <dt>{name}</dt>
            <dd>
              <span class="track" aria-hidden="true"><span class="fill" style:--v={Math.max(0.02, Math.min(1, value))}></span><span class="tick" style:--t={PATTERN_THRESHOLDS[i]}></span></span>
              <span class="num">{value.toFixed(2)}</span>
            </dd>
          </div>
        {/each}
      </dl>
      <p class="hint">{headBandNote(model, layer)} Ticks mark where the catalog names a pattern.</p>
    </section>

    {#if mediaEntries.length}
      <section class="part">
        <h3>In images and sound</h3>
        <div class="media-grid">
          {#each mediaEntries as [modality, m] (modality)}
            {@const names = MODALITY_NAMES[modality]}
            <dl class="scores media-scores" aria-label={names.title}>
              <div class="media-title">{names.title}</div>
              <div data-help={`Attention from the text after the ${names.noun} to its slots, as a multiple of an even spread over the keys`}><dt>Reads the {names.noun}</dt><dd>{fmt(m.read)}×</dd></div>
              <div data-help={`From each slot of the ${names.noun}, attention to the other slots, and to its neighbours`}><dt>Looks within it</dt><dd>{fmt(m.within)}{#if m.nbr !== null}<span class="sub"> · neighbours {fmt(m.nbr)}</span>{/if}</dd></div>
              <div class:on={(m.content ?? 0) > 3} data-help={`Zeroing the head, how much worse the model tells the ${names.noun}'s label from another's, as a z over 64 ${names.plural} (past 3 is notable)`}><dt>Carries its content</dt><dd>z {fmt(m.content, 1)}</dd></div>
              <div class:on={m.writeQ !== null && m.writeQ <= 0.05} data-help={`At the word after the ${names.noun}, the right class's mean percentile among the classes in what the head writes (0.5 by chance), with its q over all heads`}><dt>Names it</dt><dd>{fmt(m.write)}{#if m.writeQ !== null}<span class="sub"> · q {m.writeQ < 0.001 ? "< 0.001" : fmt(m.writeQ, 3)}</span>{/if}</dd></div>
            </dl>
          {/each}
        </div>
        <p class="hint">From 64 images and 64 sounds, each followed by its label (findings §16). The model tells images apart far better than sounds, so the sound scores say little.</p>
      </section>
    {/if}

    <section class="part">
      <h3>Where it fires</h3>
      <p class="legend"><span class="key hit">fired here</span><span class="key src">read from</span></p>
      <div class="contexts">{#each head.contexts as context, i (i)}<div style:--i={i}><Context {context} clip={false} /></div>{/each}</div>
    </section>

    <div class="columns">
      <section class="part">
        <h3>Where it looks</h3>
        {#if read("qk_S")}<LensReadView read={read("qk_S")!} compact />{/if}
        {#if head.keys.length}
          <div class="chips-block"><h4>In real text it reads from</h4><div class="chips">{#each head.keys.slice(0, 12) as [token, value]}<Token {token} {value} size="sm" />{/each}</div></div>
        {/if}
        {#if read("qk_self_S")}<LensReadView read={read("qk_self_S")!} />{/if}
      </section>
      <section class="part">
        <h3>What it writes</h3>
        {#if ovPrimary}<LensReadView read={ovPrimary} compact />{/if}
        {#if head.outRz.length}
          <div class="chips-block"><h4>What its firings write, in context</h4><div class="chips">{#each head.outRz.slice(0, 12) as [token, count, flags]}<Token {token} {count} {flags} size="sm" />{/each}</div></div>
        {/if}
        {#if read(`${ovFamily}_neg` as ReadColumn)}<LensReadView read={read(`${ovFamily}_neg` as ReadColumn)!} />{/if}
      </section>
    </div>

    <Expander label="More evidence" openLabel="Less evidence">
      <div class="more">
        <Facts act={head.act} peakLabel="Typical output" rateLabel="Active on" />
        <dl class="scores">
          <div><dt>Copies what it reads (OV·R)</dt><dd>{head.copy.ovR >= 0 ? "+" : ""}{head.copy.ovR.toFixed(2)}</dd></div>
          <div><dt>Seeks its own token</dt><dd>{head.copy.self >= 0 ? "+" : ""}{head.copy.self.toFixed(2)}</dd></div>
          <div><dt>Attends to what it copies</dt><dd>{head.copy.qkOv >= 0 ? "+" : ""}{head.copy.qkOv.toFixed(2)}</dd></div>
        </dl>
        <p class="tests">
          {#if tests.length}Tests that hold at its own top firings: {tests.map(([name, q]) => `${name} (${stat} ${q < 0.001 ? "< 0.001" : q.toFixed(3)})`).join("; ")}{stat === "p" ? ", uncorrected: with 16 heads a layer, no test can pass a false-discovery correction" : ""}.
          {:else if head.dark}No test holds at its own top firings, and it has no token mode or pattern.
          {:else}No read holds at its own top firings{head.modes[0] >= 1 / 3 || head.modes[1] >= 1 / 3 ? ", but a third of its firings share a token" : ""}.{/if}
        </p>
        {#if head.drivers.length}
          <div class="chips-block"><h4>Query tokens that make its output largest</h4><div class="chips">{#each head.drivers.slice(0, 12) as [token, value]}<Token {token} {value} size="sm" />{/each}</div></div>
        {/if}
        {#if head.sources.length}
          <div class="chips-block"><h4>Tokens it read from at its top firings</h4><div class="chips">{#each head.sources as [token, count]}<Token {token} {count} size="sm" />{/each}</div></div>
        {/if}
        {#each more as r (r.col)}<LensReadView read={r} twinOf={twinName(r)} />{/each}
      </div>
    </Expander>
  </div>
  {/key}
</section>

<style>
  .detail { display: grid; gap: var(--space-6); min-width: 0; }
  .summary, .body { animation: rise 380ms var(--ease-enter) backwards; }
  .body { animation-delay: 60ms; }
  @keyframes rise { from { opacity: 0; translate: 0 8px; } }
  .kv { padding: 0 var(--space-2); border-radius: var(--radius-sm); background: var(--glass); color: var(--fg-dim); font-family: var(--font-mono); font-size: var(--text-xs); line-height: 1.8; }
  .role { display: inline-flex; align-items: center; gap: var(--space-2); padding: 0 var(--space-3); border-radius: var(--radius-pill); background: color-mix(in oklch, var(--role) 16%, transparent); color: var(--fg); font-size: var(--text-xs); line-height: 1.8; white-space: nowrap; }
  .role::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: var(--role); }

  .summary { display: grid; gap: var(--space-3); padding: var(--space-5) var(--space-6); border-radius: var(--radius-lg); background: color-mix(in oklch, var(--neuron-tint) 6%, var(--neuron-chip-bg)); box-shadow: inset 0 0 0 1px color-mix(in oklch, var(--neuron-tint) 26%, transparent); }
  .sum-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(14rem, 1fr)); gap: var(--space-5) var(--space-6); }
  .cell { display: grid; gap: var(--space-2); align-content: start; min-width: 0; }
  .cell-head { color: var(--neuron-tint-ink); font-size: var(--text-sm); font-weight: var(--weight-structure-bold); }
  .note { margin: 0; color: var(--fg-muted); font-size: var(--text-sm); line-height: 1.55; text-wrap: pretty; }
  .credit { margin: 0; padding-top: var(--space-3); border-top: 1px solid var(--glass-line); color: var(--fg-muted); font-size: var(--text-xs); }

  .body { display: grid; gap: var(--space-7, var(--space-6)); }
  .part { display: grid; gap: var(--space-3); align-content: start; min-width: 0; }
  h3 { margin: 0; color: var(--fg); font-size: var(--text-md); font-weight: var(--weight-structure-bold); }
  h4 { margin: 0; color: var(--fg-muted); font-size: var(--text-xs); font-weight: var(--weight-structure); }
  .columns { display: grid; grid-template-columns: repeat(auto-fit, minmax(18rem, 1fr)); gap: var(--space-6) var(--space-8); align-items: start; }

  .pattern { display: grid; grid-template-columns: repeat(auto-fit, minmax(9rem, 1fr)); gap: var(--space-3) var(--space-5); margin: 0; }
  .pattern > div { display: grid; gap: var(--space-1); }
  .pattern dt { color: var(--fg-muted); font-size: var(--text-xs); }
  .pattern .on dt { color: var(--fg); font-weight: var(--weight-structure); }
  .pattern dd { display: flex; align-items: center; gap: var(--space-2); margin: 0; }
  .track { position: relative; flex: 1; height: 6px; border-radius: var(--radius-pill); background: var(--glass-strong); }
  .fill { position: absolute; inset: 0; border-radius: inherit; background: color-mix(in oklch, var(--neuron-tint) 55%, var(--glass-strong)); transform-origin: left; scale: var(--v) 1; }
  .on .fill { background: var(--neuron-tint); }
  .tick { position: absolute; top: -3px; bottom: -3px; left: calc(var(--t) * 100%); width: 2px; border-radius: var(--data-mark-radius); background: var(--fg-dim); }
  .num { min-width: 3.2ch; color: var(--fg-dim); font-family: var(--font-mono); font-size: var(--text-xs); font-variant-numeric: tabular-nums; text-align: end; }
  .hint { margin: 0; color: var(--fg-muted); font-size: var(--text-sm); line-height: 1.5; }

  .legend { display: flex; gap: var(--space-4); margin: 0; color: var(--fg-muted); font-size: var(--text-xs); }
  .key { display: inline-flex; align-items: center; gap: var(--space-2); }
  .key::before { content: ""; width: 14px; height: 10px; border-radius: var(--data-mark-radius); }
  .key.hit::before { background: color-mix(in oklch, var(--neuron-tint) 34%, transparent); }
  .key.src::before { box-shadow: inset 0 0 0 1.5px color-mix(in oklch, var(--neuron-tint) 80%, transparent); }
  .contexts { display: grid; gap: var(--space-2); }
  .contexts > div { animation: ctx-in 320ms var(--ease-enter) backwards; animation-delay: calc(var(--i, 0) * 60ms); }
  @keyframes ctx-in { from { opacity: 0; translate: 0 4px; } }
  .chips-block { display: grid; gap: var(--space-2); }
  .chips { display: flex; flex-wrap: wrap; gap: var(--space-1); min-width: 0; }

  .more { display: grid; gap: var(--space-5); }
  .scores { display: flex; flex-wrap: wrap; gap: var(--space-4) var(--space-8); margin: 0; }
  .scores > div { display: grid; gap: var(--space-1); }
  .scores dt { color: var(--fg-muted); font-size: var(--text-xs); }
  .scores dd { margin: 0; color: var(--fg); font-family: var(--font-mono); font-size: var(--text-sm); font-variant-numeric: tabular-nums; }
  .tests { margin: 0; color: var(--fg-dim); font-size: var(--text-sm); line-height: 1.55; }
  .media-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr)); gap: var(--space-4) var(--space-8); }
  .media-scores { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--space-3) var(--space-5); }
  .media-title { grid-column: 1 / -1; color: var(--fg-dim); font-size: var(--text-sm); font-weight: var(--weight-structure); }
  .media-scores .on dt { color: var(--fg); font-weight: var(--weight-structure); }
  .media-scores .on dd { color: var(--neuron-tint-ink); }
  .sub { color: var(--fg-muted); font-size: var(--text-xs); }
  @media (prefers-reduced-motion: reduce) { .summary, .body, .contexts > div { animation: none; } }
</style>
