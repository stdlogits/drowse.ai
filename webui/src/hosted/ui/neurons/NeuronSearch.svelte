<script lang="ts">
  import FluentIcon from "../../../../../drowse/webui/src/lib/ui/FluentIcon.svelte";
  import MorphText from "../../../../../drowse/webui/src/lib/ui/MorphText.svelte";
  import { HEADS, headId, loadSearch, neuronId, parseRef, stem, type CatalogIndex, type HeadRow, type SearchFile, type Unit } from "./data";

  interface Result { unit: Unit; layer: number; index: number; side: 0 | 1 | 2; title: string; matched: string; score: number }

  let { index, heads, onopen, onmatches, examples = [], anyNeuron = false }: {
    index: CatalogIndex | null;
    heads: HeadRow[] | null;
    /** With every neuron available, an ID opens any neuron, not just the easiest. */
    anyNeuron?: boolean;
    onopen: (unit: Unit, layer: number, index: number) => void;
    /** Every matching unit, as "layer:index" keys per unit, so the ribbon can light them up. */
    onmatches: (keys: { neurons: Set<string>; heads: Set<string> }, query: string) => void;
    examples?: string[];
  } = $props();

  let query = $state("");
  let results = $state<Result[]>([]);
  let total = $state(0);
  let active = $state(-1);
  let open = $state(false);
  let loading = $state(false);
  let searchFailed = $state(false);
  let data: SearchFile | null = null;
  let input: HTMLInputElement | undefined = $state();
  let timer: ReturnType<typeof setTimeout> | undefined;

  async function ensure(): Promise<SearchFile | null> {
    if (data) return data;
    loading = true;
    searchFailed = false;
    try { data = await loadSearch(); } catch { data = null; searchFailed = true; } finally { loading = false; }
    return data;
  }

  /** A head's clearest label. */
  function headTitle(row: HeadRow): string {
    const labels: [string, number][] = [[row[3], row[6]], [row[4], row[7]], [row[5], row[8]]];
    return labels.reduce((best, next) => (next[1] > best[1] ? next : best))[0];
  }

  function report(list: Result[]) {
    const keys = { neurons: new Set<string>(), heads: new Set<string>() };
    for (const r of list) keys[r.unit].add(`${r.layer}:${r.index}`);
    return keys;
  }

  function run(text: string) {
    const trimmed = text.trim();
    if (!trimmed || !index) { results = []; total = 0; onmatches({ neurons: new Set(), heads: new Set() }, ""); return; }
    const ref = parseRef(trimmed);
    if (ref) {
      let found: Result | null = null;
      if (ref.unit === "heads") {
        const row = heads?.find((r) => r[0] === ref.layer && r[1] === ref.index);
        found = { unit: "heads", layer: ref.layer, index: ref.index, side: 2, title: row ? headTitle(row) : "Open this head", matched: "", score: 1 };
      } else {
        const row = index.neurons.find((r) => r[0] === ref.layer && r[1] === ref.index);
        found = row ? { unit: "neurons", layer: row[0], index: row[1], side: 0, title: row[2], matched: "", score: 1 }
          : anyNeuron ? { unit: "neurons", layer: ref.layer, index: ref.index, side: 0, title: "Open this neuron", matched: "", score: 1 } : null;
      }
      results = found ? [found] : [];
      total = results.length;
      onmatches(report(results), trimmed);
      active = results.length ? 0 : -1;
      return;
    }
    if (!data) {
      if (searchFailed) return;
      void ensure().then(() => { if (data) run(query); });
      return;
    }
    const key = stem(trimmed);
    const lower = trimmed.toLowerCase();
    const cjk = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/u.test(lower);
    // Keyed by unit: curated neuron poles as ref * 2 + side, heads as 1e6 + layer * 32 + head.
    const scores = new Map<number, { score: number; matched: string }>();
    for (const [form, refs] of Object.entries(data.forms)) {
      let factor = 0;
      if (form === key || form === lower) factor = 3;
      else if (key.length >= 2 && form.startsWith(key)) factor = 1.4;
      else if (key.length >= 3 && form.includes(key)) factor = 0.7;
      else if (cjk && form.includes(lower)) factor = 1.2;
      if (!factor) continue;
      for (const [ref, side, weight] of refs) {
        const id = side === 2 ? 1e6 + ref : ref * 2 + side;
        const value = weight * factor;
        const previous = scores.get(id);
        if (!previous) scores.set(id, { score: value, matched: data.display[form] ?? form });
        else { previous.score += value * 0.35; if (value > previous.score) previous.matched = data.display[form] ?? form; }
      }
    }
    const ranked = [...scores.entries()].sort((a, b) => b[1].score - a[1].score);
    const all: Result[] = [];
    const seen = new Set<string>();
    for (const [id, hit] of ranked) {
      let r: Result | null = null;
      if (id >= 1e6) {
        const ref = id - 1e6;
        const row = heads?.find((h) => h[0] === Math.floor(ref / HEADS) && h[1] === ref % HEADS);
        r = { unit: "heads", layer: Math.floor(ref / HEADS), index: ref % HEADS, side: 2, title: row ? headTitle(row) : "", matched: hit.matched, score: hit.score };
      } else {
        const row = index.neurons[id >> 1];
        const side = (id & 1) as 0 | 1;
        r = { unit: "neurons", layer: row[0], index: row[1], side, title: side === 0 ? row[2] : row[3], matched: hit.matched, score: hit.score };
      }
      const k = `${r.unit}:${r.layer}:${r.index}`;
      if (seen.has(k)) continue;
      seen.add(k);
      all.push(r);
    }
    total = all.length;
    results = all.slice(0, 12);
    active = results.length ? 0 : -1;
    onmatches(report(all), trimmed);
  }

  function changed() {
    open = true;
    clearTimeout(timer);
    timer = setTimeout(() => run(query), 70);
  }

  function choose(result: Result) {
    open = false;
    onopen(result.unit, result.layer, result.index);
  }

  function key(event: KeyboardEvent) {
    if (event.key === "ArrowDown") { event.preventDefault(); open = true; active = Math.min(results.length - 1, active + 1); }
    else if (event.key === "ArrowUp") { event.preventDefault(); active = Math.max(0, active - 1); }
    else if (event.key === "Enter" && active >= 0 && results[active]) { event.preventDefault(); choose(results[active]); }
    else if (event.key === "Escape") {
      if (open && results.length) open = false;
      else { query = ""; run(""); }
    }
  }

  function tryExample(text: string) {
    query = text;
    open = true;
    input?.focus();
    void ensure().then(() => run(text));
  }

  // "/" focuses search from anywhere on the page, as on most explorers.
  $effect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "/" || event.metaKey || event.ctrlKey || event.altKey) return;
      const target = event.target as HTMLElement | null;
      if (target?.closest("input, textarea, select, [contenteditable='true']")) return;
      event.preventDefault();
      input?.focus();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  });

  const status = $derived(
    !query.trim() ? ""
    : searchFailed ? "Unable to load search. Check your connection and try again."
    : loading ? "Loading search…"
    : !results.length ? `Nothing matches “${query.trim()}”. Try another word, or an ID like L8 344 or L20h17.`
    : total === 1 ? "1 match."
    : `${total} matches. Showing the top ${results.length}.`,
  );
</script>

<div class="search" role="search">
  <div class="field" class:open={open && results.length > 0}>
    <FluentIcon name="search" size={20} />
    <input
      bind:this={input}
      bind:value={query}
      type="search"
      role="combobox"
      aria-label="Search neurons and heads by concept or ID"
      aria-autocomplete="list"
      aria-expanded={open && results.length > 0}
      aria-controls="neuron-results"
      aria-activedescendant={open && active >= 0 ? `neuron-result-${active}` : undefined}
      placeholder="Search a word, or an ID like L8 344 or L20h17"
      autocomplete="off"
      spellcheck="false"
      oninput={changed}
      onfocus={() => { open = true; searchFailed = false; void ensure().then(() => { if (data && query.trim()) run(query); }); }}
      onblur={() => setTimeout(() => (open = false), 120)}
      onkeydown={key}
    />
    <kbd aria-hidden="true">/</kbd>
  </div>
  <p class="status" aria-live="polite">{#if status}<MorphText text={status} duration={200} />{/if}</p>
  {#if open && results.length}
    <ul class="results" id="neuron-results" role="listbox" aria-label="Matches">
      {#each results as result, i (`${result.unit}:${result.layer}:${result.index}`)}
        <li
          id={`neuron-result-${i}`}
          role="option"
          aria-selected={i === active}
          class:active={i === active}
          onpointerdown={(event) => { event.preventDefault(); choose(result); }}
          onpointerenter={() => (active = i)}
          style:--L={result.layer}
        >
          <span class="r-id">{result.unit === "heads" ? headId(result.layer, result.index) : neuronId(result.layer, result.index)}</span>
          <span class="r-sign" aria-label={result.side === 2 ? "head" : result.side === 0 ? "plus pole" : "minus pole"}>{result.side === 2 ? "h" : result.side === 0 ? "+" : "−"}</span>
          <span class="r-title">{result.title || "unclear"}</span>
          {#if result.matched}<span class="r-match">{result.matched}</span>{/if}
        </li>
      {/each}
    </ul>
  {/if}
  {#if examples.length && !query}
    <p class="examples"><span>Try</span>{#each examples as example}<button type="button" onclick={() => tryExample(example)}>{example}</button>{/each}</p>
  {/if}
</div>

<style>
  .search { position: relative; display: grid; gap: var(--space-3); width: min(100%, 40rem); }
  .field {
    display: flex; align-items: center; gap: var(--space-3); min-height: 56px; padding-inline: var(--space-5);
    border-radius: var(--radius-lg); background: var(--input-well, var(--glass)); box-shadow: var(--shadow-well, inset 0 0 0 1px var(--glass-line));
    color: var(--fg-muted); transition: box-shadow var(--dur-fast) var(--ease-out);
  }
  .field:focus-within { box-shadow: var(--shadow-well, none), 0 0 0 2px var(--focus-ring); color: var(--fg); }
  input { flex: 1; min-width: 0; height: 54px; border: 0; background: transparent; color: var(--fg); font: inherit; font-size: var(--text-md); outline: none; }
  input::placeholder { color: var(--fg-muted); }
  input::-webkit-search-cancel-button { filter: grayscale(1); }
  kbd { display: grid; place-items: center; min-width: 24px; height: 24px; border-radius: var(--radius-sm); box-shadow: inset 0 0 0 1px var(--glass-line); color: var(--fg-muted); font-family: var(--font-mono); font-size: var(--text-xs); }
  .field:focus-within kbd { display: none; }
  .status { min-height: 1.2em; margin: 0; color: var(--fg-muted); font-size: var(--text-xs); }
  /* Stays rendered while empty, so screen readers announce its first message. */
  .status:empty { min-height: 0; margin-block-end: calc(var(--space-3) * -1); }
  .results {
    position: absolute; inset: 60px 0 auto; z-index: 6; display: grid; max-height: min(26rem, 60vh); margin: 0; padding: var(--space-2); overflow-y: auto;
    border-radius: var(--radius-lg); background: var(--bg-elev); box-shadow: var(--shadow-card); list-style: none;
    animation: drop var(--dur) var(--ease-enter) backwards; transform-origin: top;
  }
  @keyframes drop { from { opacity: 0; transform: translateY(-4px) scale(0.99); } }
  .results li { display: grid; grid-template-columns: 5.5rem 1.5ch minmax(0, 1fr) auto; align-items: center; gap: var(--space-3); min-height: 44px; padding: var(--space-2) var(--space-3); border-radius: var(--radius); cursor: pointer; }
  .results li.active { background: var(--bg-hover); }
  .r-id { color: var(--neuron-tint-ink); font-family: var(--font-mono); font-size: var(--text-xs); font-variant-numeric: tabular-nums; }
  .r-sign { color: var(--fg-muted); font-weight: var(--weight-display); }
  .r-title { overflow: hidden; color: var(--fg); font-size: var(--text-sm); text-overflow: ellipsis; white-space: nowrap; }
  .r-match { max-width: 10ch; overflow: hidden; padding: 0 var(--space-xs); border-radius: var(--radius-sm); background: var(--glass-strong); color: var(--fg-dim); font-family: var(--font-mono); font-size: var(--text-xs); text-overflow: ellipsis; white-space: nowrap; }
  .examples { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-2); margin: 0; color: var(--fg-muted); font-size: var(--text-sm); }
  .examples span { margin-inline-end: var(--space-1); }
  .examples button { min-height: var(--control-target); padding-inline: var(--space-4); border: 0; border-radius: var(--radius-pill); background: var(--glass); box-shadow: inset 0 0 0 1px var(--glass-line); color: var(--fg-dim); font: inherit; font-size: var(--text-sm); cursor: pointer; transition: background-color var(--dur-fast) var(--ease-out), color var(--dur-fast) var(--ease-out); }
  .examples button:hover { background: var(--glass-strong); color: var(--fg); }
  .examples button:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  @media (max-width: 520px) { .results li { grid-template-columns: 5.5rem 1.5ch minmax(0, 1fr); } .r-match { display: none; } }
  @media (prefers-reduced-motion: reduce) { .results { animation: none; } .field, .examples button { transition: none; } }
</style>
