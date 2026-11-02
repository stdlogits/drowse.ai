<script lang="ts">
  import { onMount, tick } from "svelte";
  import { flip } from "svelte/animate";
  import { GLIDE, IN, MorphText, OUT, reducedMotion } from "@stdlogits/ledger";
  import { tokenProbabilityRows, visibleTokenText } from "../../../../drowse/webui/src/lib/tokenProbabilities";
  import { probabilityScore, scoreToRgb as highlightColor, surpriseScore } from "../../../../drowse/webui/src/lib/tokens";
  import type { TokenScore } from "../../../../drowse/webui/src/lib/types";

  let { tokens, highlights = true, layout = "message", coloring = "surprisal" }: {
    tokens: TokenScore[];
    highlights?: boolean;
    layout?: "message" | "inline";
    coloring?: string;
  } = $props();
  let active = $state<number | null>(null);
  let displayed = $state<number | null>(null);
  let anchor: HTMLElement | null = null;
  let panel: HTMLDivElement;
  let pinned = false;
  let fade: Animation | null = null;
  let hoverAfter = 0;
  let hoverTimer: ReturnType<typeof setTimeout> | undefined;
  let closeTimer: ReturnType<typeof setTimeout> | undefined;
  // Without the Popover API the panel is a fixed element toggled by class (see the @supports rule).
  let fallbackOpen = $state(false);
  // Roving tabindex: one Tab stop for the whole text; arrow keys, Home and End move between tokens.
  let current = $state<number | null>(null);
  const uid = $props.id();
  const popoverSupported = () => typeof panel?.showPopover === "function";
  function showPanel() {
    if (!popoverSupported()) { fallbackOpen = true; return; }
    try { if (!panel.matches(":popover-open")) panel.showPopover(); } catch { fallbackOpen = true; }
  }
  function hidePanel() {
    fallbackOpen = false;
    if (!popoverSupported()) return;
    try { if (panel.matches(":popover-open")) panel.hidePopover(); } catch { /* Already closed. */ }
  }
  function panelOpen() {
    if (!popoverSupported()) return fallbackOpen;
    try { return panel.matches(":popover-open") || fallbackOpen; } catch { return fallbackOpen; }
  }
  function spokenToken(text: string) {
    const word = text.trim();
    const space = (part: string) => part.includes("\n") ? "line break" : "space";
    if (!word) return text.includes("\n") ? "line break" : "space";
    const lead = text.match(/^\s+/)?.[0], trail = text.match(/\s+$/)?.[0];
    const notes = [lead && `a leading ${space(lead)}`, trail && `a trailing ${space(trail)}`].filter(Boolean);
    return `“${word}”${notes.length ? `, with ${notes.join(" and ")}` : ""}`;
  }
  const token = $derived(displayed === null ? null : tokens[displayed]);
  const rows = $derived(token ? tokenProbabilityRows(token) : []);
  $effect.pre(() => {
    tokens;
    cancelTimers();
    fade?.cancel();
    active = null;
    displayed = null;
    anchor = null;
    pinned = false;
    current = null;
    if (panel) hidePanel();
  });
  const groups = $derived.by(() => {
    const result: Array<{ space: string; pieces: Array<{ text: string; index: number; first: boolean }> }> = [];
    let group = { space: "", pieces: [] as Array<{ text: string; index: number; first: boolean }> };
    tokens.forEach((token, index) => {
      let first = true;
      for (const part of token.text.match(/\s+|\S+/g) ?? []) {
        if (/^\s+$/.test(part)) {
          if (group.pieces.length) { result.push(group); group = { space: "", pieces: [] }; }
          group.space += part;
        } else { group.pieces.push({ text: part, index, first }); first = false; }
      }
    });
    if (group.pieces.length) result.push(group);
    return result;
  });
  const rovingIndex = $derived(current ?? (layout === "inline" ? 0 : groups[0]?.pieces[0]?.index ?? 0));

  function place() {
    if (!anchor || active === null) return;
    const viewport = window.visualViewport;
    const left = viewport?.offsetLeft ?? 0, top = viewport?.offsetTop ?? 0;
    const width = viewport?.width ?? innerWidth, height = viewport?.height ?? innerHeight;
    panel.style.maxHeight = `${height - 24}px`;
    const rect = anchor.getBoundingClientRect(), box = panel.getBoundingClientRect();
    const x = Math.max(left + 12, Math.min(rect.left, left + width - box.width - 12));
    const below = top + height - rect.bottom - 12;
    const y = below >= box.height || rect.top - top < below ? rect.bottom + 8 : rect.top - box.height - 8;
    panel.style.left = `${x}px`;
    panel.style.top = `${Math.max(top + 12, Math.min(y, top + height - box.height - 12))}px`;
  }

  function cancelTimers() { clearTimeout(hoverTimer); clearTimeout(closeTimer); }
  function close(restoreFocus = false) {
    cancelTimers();
    if (active === null) return;
    const trigger = anchor;
    const opacity = getComputedStyle(panel).opacity;
    fade?.cancel();
    active = null;
    pinned = false;
    hoverAfter = performance.now() + 250;
    const exit = panel.animate([{ opacity }, { opacity: 0 }], {
      duration: reducedMotion() ? 0 : OUT,
      easing: "cubic-bezier(0.2, 0, 0, 1)",
    });
    fade = exit;
    exit.onfinish = () => { if (fade === exit) { hidePanel(); fade = null; } };
    if (restoreFocus) trigger?.focus({ preventScroll: true });
  }
  async function open(index: number, element: HTMLElement, pin = false, focus = false) {
    cancelTimers();
    anchor = element;
    active = index;
    displayed = index;
    pinned = pin;
    await tick();
    if (active !== index || anchor !== element) return;
    const opacity = panelOpen() ? getComputedStyle(panel).opacity : "0";
    fade?.cancel();
    showPanel();
    place();
    fade = panel.animate([{ opacity }, { opacity: 1 }], {
      duration: reducedMotion() ? 0 : IN,
      easing: "cubic-bezier(0.2, 0, 0, 1)",
    });
    if (focus) panel.focus({ preventScroll: true });
  }
  function hover(event: PointerEvent, index: number) {
    if (event.pointerType !== "mouse" || pinned || performance.now() < hoverAfter) return;
    cancelTimers();
    const element = event.currentTarget as HTMLElement;
    hoverTimer = setTimeout(() => void open(index, element), 1000);
  }
  function leave() {
    clearTimeout(hoverTimer);
    if (!pinned) closeTimer = setTimeout(() => close(), 220);
  }
  function move(event: KeyboardEvent) {
    const buttons = Array.from((event.currentTarget as HTMLElement).closest(".recorded-tokens")!.querySelectorAll<HTMLButtonElement>(".recorded-token"));
    const current = buttons.indexOf(event.currentTarget as HTMLButtonElement);
    const next = event.key === "ArrowRight" ? Math.min(current + 1, buttons.length - 1)
      : event.key === "ArrowLeft" ? Math.max(current - 1, 0)
      : event.key === "Home" ? 0 : event.key === "End" ? buttons.length - 1 : null;
    if (next !== null) { event.preventDefault(); buttons[next].focus(); }
  }

  onMount(() => {
    const resize = new ResizeObserver(place);
    resize.observe(panel);
    const outside = (event: PointerEvent) => {
      if (active !== null && !panel.contains(event.target as Node) && !anchor?.contains(event.target as Node)) close();
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape" && active !== null) { event.preventDefault(); close(pinned); }
    };
    const focusOutside = (event: FocusEvent) => {
      if (active !== null && !panel.contains(event.target as Node) && !anchor?.contains(event.target as Node)) close();
    };
    const scroll = (event: Event) => { if (!(event.target instanceof Node) || !panel?.contains(event.target)) place(); };
    document.addEventListener("pointerdown", outside);
    document.addEventListener("keydown", escape);
    document.addEventListener("focusin", focusOutside);
    window.addEventListener("resize", place);
    window.addEventListener("scroll", scroll, true);
    window.visualViewport?.addEventListener("resize", place);
    window.visualViewport?.addEventListener("scroll", place);
    return () => {
      cancelTimers();
      fade?.cancel();
      resize.disconnect();
      document.removeEventListener("pointerdown", outside);
      document.removeEventListener("keydown", escape);
      document.removeEventListener("focusin", focusOutside);
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", scroll, true);
      window.visualViewport?.removeEventListener("resize", place);
      window.visualViewport?.removeEventListener("scroll", place);
    };
  });
</script>

{#snippet tokenButton(text: string, index: number, first = true)}<button
    type="button" class="recorded-token" tabindex={index === rovingIndex && first ? 0 : -1} class:highlighted={highlights} class:selected={active === index} class:whitespace={text.trim().length === 0}
    style:--token-tint={highlightColor(coloring === "probability" ? probabilityScore(tokens[index].logprob) : surpriseScore(tokens[index].logprob), undefined, "surprise")}
    aria-label={`Inspect token ${spokenToken(tokens[index].text)}`}
    aria-haspopup="dialog" aria-expanded={active === index} aria-controls={active === index ? `${uid}-panel` : undefined}
    onpointerenter={event => hover(event, index)} onpointerleave={leave}
    onfocus={() => current = index}
    onclick={event => void open(index, event.currentTarget, true, event.detail === 0)}
    onkeydown={move}
  >{#if layout === "inline"}<MorphText {text} numbers={false} duration={240} />{:else}{text}{/if}</button>{/snippet}

<svelte:element this={layout === "message" ? "h1" : "div"} class="message-title" class:inline={layout === "inline"} aria-label={tokens.map(token => token.text).join("").trim()}><span class="recorded-tokens">{#if layout === "inline"}{#each tokens as piece, index (index)}<span class="inline-piece" animate:flip={{ duration: reducedMotion() ? 0 : GLIDE }}>{@render tokenButton(piece.text, index)}</span>{/each}{:else}{#each [groups.slice(0, 2), groups.slice(2)] as section, sectionIndex}{sectionIndex === 0 ? "" : " "}<span class={sectionIndex === 0 ? "message-prefix" : "message-body"}>{#each section as group, groupIndex}{groupIndex === 0 ? "" : group.space}<span class="word">{#each group.pieces as piece}{@render tokenButton(piece.text, piece.index, piece.first)}{/each}</span>{/each}</span>{/each}{/if}</span></svelte:element>

<div bind:this={panel} id={`${uid}-panel`} class="recorded-token-panel" class:fallback-open={fallbackOpen} popover="manual" role="dialog" tabindex="-1"
  aria-labelledby={`${uid}-title`} inert={active === null} onpointerenter={cancelTimers} onpointerleave={leave}>
  {#if token}
    <header>
      <div><span class="source">Model token · Gemma 3 4B</span><h2 id={`${uid}-title`}>{visibleTokenText(token.text)}</h2></div>
      <button class="close" type="button" aria-label="Close token probabilities" onclick={() => close(true)}>Close</button>
    </header>
    <dl><div><dt>Token ID</dt><dd>{token.tokenId}</dd></div><div><dt>Surprisal</dt><dd>{(-(token.logprob ?? 0) / Math.LN2).toFixed(2)} bits</dd></div></dl>
    <table aria-label="Recorded token probabilities">
      <thead><tr><th scope="col">Token</th><th scope="col">Probability</th></tr></thead>
      <tbody>{#each rows as row, index (index)}<tr class:chosen={row.chosen}>
        <td><code>{visibleTokenText(row.text)}</code>{#if row.chosen}<span class="chosen-label">this token</span>{/if}</td>
        <td>{Math.exp(row.logprob) >= 0.001 ? Math.exp(row.logprob).toFixed(3) : Math.exp(row.logprob).toExponential(2)}</td>
      </tr>{/each}</tbody>
    </table>
  {/if}
</div>

<style>
  /* A recorded reply, each token a button tinted by how surprising it was (the dashboard's token
     tint, from --tint); selecting one opens its alternatives in a frosted panel, Ledger's tip. */
  .message-title { margin: 0; font: inherit; text-wrap: balance; }
  .recorded-tokens { display: flex; flex-direction: column; white-space: pre-wrap; }
  .message-prefix { display: block; margin-bottom: var(--s2); color: var(--ink); font: var(--type-title); letter-spacing: -0.04em; }
  .message-body { display: block; color: var(--ink2); font: var(--type-lead); }
  .word { display: inline-block; max-width: 100%; white-space: nowrap; }
  .recorded-token { display: inline-block; min-height: 44px; margin: 0; padding: 0; border: 0; border-radius: var(--r-chip); background: transparent; color: inherit; font: inherit; vertical-align: baseline; cursor: pointer; transition: background-color var(--fast) var(--ease), box-shadow var(--fast) var(--ease); }
  .recorded-token.highlighted { background: var(--token-tint); }
  .recorded-token.selected { box-shadow: 0 0 0 2px var(--ink); }
  @media (hover: hover) { .recorded-token:hover { box-shadow: 0 0 0 2px var(--ink); } }
  .inline { text-wrap: pretty; }
  .inline .recorded-tokens { display: inline; }
  .inline-piece { display: inline-block; max-width: 100%; vertical-align: baseline; }
  .inline .recorded-token { display: inline; min-height: 0; white-space: pre-wrap; overflow-wrap: anywhere; }
  .inline .recorded-token.whitespace { display: inline-block; min-width: 0.35em; white-space: pre; }

  .recorded-token-panel {
    position: fixed;
    inset: auto;
    width: min(320px, calc(100vw - 24px));
    max-height: min(420px, calc(100dvh - 24px));
    margin: 0;
    padding: var(--s3) var(--s4);
    overflow: auto;
    border: 0;
    border-radius: var(--r-box);
    background: color-mix(in oklch, var(--paper) 72%, transparent);
    -webkit-backdrop-filter: blur(20px) saturate(1.6);
    backdrop-filter: blur(20px) saturate(1.6);
    color: var(--ink);
    font: var(--type-small);
    white-space: normal;
  }
  .recorded-token-panel:focus { outline: none; }
  header { display: flex; align-items: start; justify-content: space-between; gap: var(--s3); }
  header > div { display: grid; gap: 2px; min-width: 0; }
  .recorded-token-panel h2 { color: var(--ink); font: 600 16px/24px var(--display); overflow-wrap: anywhere; }
  .source, dt { color: var(--ink3); font: var(--type-meta); }
  .close { height: 32px; padding: 0 var(--s3); border: 0; border-radius: var(--r-pill); background: var(--wash); color: var(--ink); font: var(--type-control); cursor: pointer; transition: background-color var(--fast) var(--ease), scale var(--fast) var(--ease); }
  .close:active { scale: 0.97; }
  @media (hover: hover) { .close:hover { background: var(--wash2); } }
  dl { display: flex; flex-wrap: wrap; gap: var(--s1) var(--s3); margin: var(--s2) 0; font-variant-numeric: tabular-nums; }
  dl div { display: flex; gap: var(--s1); }
  dd { margin: 0; }
  table { width: 100%; border-collapse: collapse; table-layout: fixed; }
  th { color: var(--ink3); font: var(--type-head); text-align: start; }
  th, td { padding: var(--s1) var(--s2); }
  th:last-child, td:last-child { width: 6rem; text-align: end; font-variant-numeric: tabular-nums; }
  td { color: var(--ink2); }
  code { overflow-wrap: anywhere; }
  .chosen td { background: color-mix(in oklch, var(--tint) 14%, transparent); color: var(--ink); }
  .chosen td:first-child { border-radius: var(--r-chip) 0 0 var(--r-chip); }
  .chosen td:last-child { border-radius: 0 var(--r-chip) var(--r-chip) 0; }
  .chosen-label { margin-inline-start: var(--s1); color: var(--ink3); font: var(--type-meta); }
  @supports not ((backdrop-filter: blur(1px)) or (-webkit-backdrop-filter: blur(1px))) { .recorded-token-panel { background: var(--paper); } }
  @media (prefers-reduced-transparency: reduce), (prefers-contrast: more) { .recorded-token-panel { background: var(--paper); } }
  .recorded-token-panel.fallback-open { display: block; z-index: 1000; }
  @supports not selector(:popover-open) { .recorded-token-panel:not(.fallback-open) { display: none; } }
  @media (forced-colors: active) { .recorded-token-panel { border: 1px solid CanvasText; } }
</style>
