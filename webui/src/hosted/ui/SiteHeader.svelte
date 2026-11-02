<script lang="ts">
  import type { Snippet } from "svelte";
  import { ThemeButton } from "@stdlogits/ledger";
  import { theme } from "../../../../drowse/webui/src/lib/theme";

  // The site's header, drawn as the dashboard's is: the name set in type is the mark, then the
  // pages as quiet links, then the one way in and the theme. Inside /app (Chats and Models) the
  // links are the app's own two places.
  type Current = "home" | "chats" | "models" | "credits" | "contact" | "learn" | "privacy" | "about" | "contribute";

  let { current, variant, onChats, onModels, showChats = true, siteNavigation = true, homeHref = "/", leading }: {
    current?: Current;
    /** "site" for public pages, "app" inside /app. Chats and Models default to "app". */
    variant?: "site" | "app";
    onChats?: () => void;
    onModels?: () => void;
    /** Hide the Chats link, for example before the first chat exists. */
    showChats?: boolean;
    siteNavigation?: boolean;
    homeHref?: string;
    leading?: Snippet;
  } = $props();

  const app = $derived((variant ?? (current === "chats" || current === "models" ? "app" : "site")) === "app");

  /** A plain click runs the in-app move; a modified one opens the link as the browser would. */
  function navigate(event: MouseEvent, action?: () => void) {
    if (!action || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    action();
  }
</script>

<header class="page-header" class:app>
  <div class="page-leading">
    {@render leading?.()}
    <a class="page-brand" href={homeHref} aria-label="Drowse home" aria-current={current === "home" ? "page" : undefined} translate="no">drowse</a>
  </div>
  {#if siteNavigation}
    <nav aria-label="Primary navigation">
      {#if app}
        {#if showChats}<a href="/app" aria-current={current === "chats" ? "page" : undefined} onclick={(event) => navigate(event, onChats)}>Chats</a>{/if}
        <a href="/app?choose=1" aria-current={current === "models" ? "page" : undefined} onclick={(event) => navigate(event, onModels)}>Models</a>
      {:else}
        <a href="/learn" aria-current={current === "learn" ? "page" : undefined}>Learn</a>
        <a href="/app?choose=1" onclick={(event) => navigate(event, onModels)}>Models</a>
        <a href="/credits" aria-current={current === "credits" ? "page" : undefined}>Credits</a>
        <a href="/contribute" aria-current={current === "contribute" ? "page" : undefined}>Contribute</a>
      {/if}
    </nav>
  {/if}
  <div class="header-end">
    {#if siteNavigation && !app}<a class="open-action" href="/app">Open Drowse</a>{/if}
    <ThemeButton {theme} />
  </div>
</header>

<style>
  .page-header {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr) auto;
    grid-template-areas: "leading nav end";
    align-items: center;
    gap: var(--s2) var(--s6);
    width: 100%;
    max-width: var(--page);
    margin: 0 auto;
    padding: var(--s4) var(--gutter);
  }
  .page-leading { grid-area: leading; display: flex; align-items: center; gap: var(--s2); min-width: 0; }
  .page-brand { color: var(--ink); font: var(--type-mark); letter-spacing: -0.02em; white-space: nowrap; }
  nav { grid-area: nav; display: flex; align-items: center; gap: var(--s5); min-width: 0; overflow-x: auto; scrollbar-width: none; }
  nav::-webkit-scrollbar { display: none; }
  nav a { flex: none; color: var(--ink3); font: var(--type-small); white-space: nowrap; transition: color var(--fast) var(--ease); }
  nav a[aria-current="page"] { color: var(--ink); }
  .header-end { grid-area: end; display: flex; align-items: center; gap: var(--s2); }
  .open-action { display: inline-flex; align-items: center; height: 36px; padding: 0 var(--s4); border-radius: var(--r-pill); background: var(--ink); color: var(--paper); font: var(--type-control); white-space: nowrap; transition: opacity var(--fast) var(--ease), scale var(--fast) var(--ease); }
  .open-action:active { scale: 0.97; }
  @media (hover: hover) {
    nav a:hover { color: var(--ink); }
    .open-action:hover { opacity: 0.86; }
  }
  /* A phone: the mark and the way in on the first row, the pages under them. */
  @media (max-width: 720px) {
    .page-header { grid-template-columns: minmax(0, 1fr) auto; grid-template-areas: "leading end" "nav nav"; gap: var(--s2) var(--s4); }
    .page-header.app { grid-template-columns: auto minmax(0, 1fr) auto; grid-template-areas: "leading nav end"; }
    nav { gap: var(--s4); min-height: 44px; }
    .open-action { height: 44px; }
    nav a { position: relative; }
    nav a::after { content: ""; position: absolute; left: 50%; top: 50%; width: max(100%, 44px); height: 44px; translate: -50% -50%; }
  }
  @media (forced-colors: active) { .open-action { border: 1px solid ButtonText; } }
</style>
