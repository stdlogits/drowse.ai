<script lang="ts">
  import { onDestroy, onMount, tick, type Component } from "svelte";
  import { IN, OUT, fadeIn, fadeOut, rise } from "@stdlogits/ledger";
  import { createShellController } from "../../../hosted/shell-controller";
  import { mountHostedWebMcp } from "../../../../drowse/webui/src/lib/webmcp";
  import { registerPageActions } from "../../../../drowse/webui/src/lib/webmcp/lifecycle";
  import {
    installHostedController,
    installRuntimeCapabilities,
    installRuntimeClient,
  } from "../../../../drowse/webui/src/lib/runtime/registry";
  import HostedApp from "./HostedApp.svelte";
  import HostedHome from "./HostedHome.svelte";
  import SiteHeader from "./SiteHeader.svelte";
  import PageFooter from "./PageFooter.svelte";
  import PwaUpdatePrompt from "./PwaUpdatePrompt.svelte";
  import { conversationLibrary, flushConversationAutosave } from "../../../../drowse/webui/src/lib/stores/savedConversations.svelte";
  import type { SavedConversationSummary } from "../../../../drowse/webui/src/lib/conversationLibrary";
  import type { HostedShellController, HostedShellSnapshot } from "../../../../drowse/webui/src/lib/shell";
  import { userFacingError } from "../../../../drowse/webui/src/lib/runtime/userFacingError";
  import {
    chunkLoadFailureMessage,
    completeChunkRecovery,
    isChunkLoadError,
    recoverFromChunkLoadError,
    reloadHostedApp,
  } from "../runtime/chunkRecovery";
  import {
    forgetEntryModel,
    readEntryMemory,
    rememberCompletedOnboarding,
  } from "./entryMemory";
  import {
    clearConversationOpen,
    queueConversationOpen,
  } from "../../../../drowse/webui/src/lib/conversationOpen";

  type EntryView = "resolving" | "home" | "models";
  type HistoryMode = "push" | "replace";
  interface EntryHistoryState {
    drowseConversation?: { id: string; modelId: string };
  }

  let { controller }: { controller?: HostedShellController } = $props();
  const appController = $derived(controller ?? createShellController());
  let Workbench = $state<Component | null>(null);
  let workbenchError = $state<string | null>(null);
  let workbenchNeedsReload = $state(false);
  let workbenchReloading = $state(false);
  let runtimeRecovery = $state<string | null>(null);
  let shellSnapshot = $state<HostedShellSnapshot | null>(null);
  let entryView: EntryView = $state("resolving");
  let firstRun = $state(true);
  let runtimeDialog: HTMLElement | null = $state(null);
  let dialogInitialFocus: HTMLButtonElement | null = $state(null);
  let allowingTakeover = $state(false);
  let preparingPwaUpdate = $state(false);
  let returningHome = $state(false);
  let disposed = false;
  let reloading = false;
  let focusedDialogKey: string | null = null;
  let previousFocus: HTMLElement | null = null;
  let restoreFocusOnClose = true;
  let workbenchPromise: Promise<void> | null = null;
  let viewNonce = $state(0);
  let installedModelIds: Set<string> | null = null;
  const initialUrl = new URL(window.location.href);
  let shownEntry = `${initialUrl.pathname}${initialUrl.search}`;
  const requestedModelPicker = initialUrl.searchParams.get("choose") === "1";
  const requestedWorkbenchReopen = initialUrl.searchParams.get("reopen") === "1";
  const takeoverPhase = $derived(shellSnapshot?.takeover?.phase ?? "idle");
  const takeoverReason = $derived(shellSnapshot?.takeover?.reason ?? null);
  const downloadActive = $derived(
    ["requesting_persistence", "downloading", "cancelling"].includes(shellSnapshot?.download.phase ?? "idle"),
  );
  const takeoverForDownload = $derived(
    downloadActive || /\bdownloading\b/u.test(takeoverReason ?? ""),
  );
  const dialogKey = $derived(
    runtimeRecovery !== null
      ? "recovery"
      : takeoverPhase === "idle"
        ? null
        : takeoverPhase,
  );
  const dialogOpen = $derived(dialogKey !== null);
  const route = $derived(Workbench ? "workbench" : shellSnapshot === null ? "resolving" : entryView);
  const routeKey = $derived(`${route}:${viewNonce}`);
  let routeMounted = $state(false);
  $effect(() => {
    // The shell's views name themselves; the workbench is the app, as the dashboard's own
    // index.html titles it (in the site's casing).
    if (route === "resolving") document.title = "Opening workspace · Drowse";
    else if (route === "workbench") document.title = "Drowse";
  });

  const FOCUSABLE = [
    "button:not([disabled])",
    "[href]",
    "input:not([disabled])",
    "select:not([disabled])",
    "textarea:not([disabled])",
    '[tabindex]:not([tabindex="-1"])',
  ].join(",");

  $effect(() => {
    const nextKey = dialogKey;
    if (nextKey === focusedDialogKey) return;
    const wasOpen = focusedDialogKey !== null;
    focusedDialogKey = nextKey;
    if (nextKey !== null) {
      if (!wasOpen) {
        previousFocus = document.activeElement instanceof HTMLElement
          ? document.activeElement
          : null;
        restoreFocusOnClose = nextKey === "requested";
      }
      void tick().then(() => {
        if (!disposed && focusedDialogKey === nextKey) {
          (dialogInitialFocus ?? runtimeDialog)?.focus();
        }
      });
      return;
    }
    const target = previousFocus;
    previousFocus = null;
    if (wasOpen && restoreFocusOnClose) {
      queueMicrotask(() => {
        if (!disposed && focusedDialogKey === null && target?.isConnected) {
          target.focus();
        }
      });
    }
  });

  function reloadToOnboarding(chooseModels = false): void {
    if (disposed || reloading) return;
    reloading = true;
    const url = new URL("/app", window.location.href);
    const currentUrl = new URL(window.location.href);
    if (currentUrl.searchParams.get("fixture") === "1") {
      url.searchParams.set("fixture", "1");
      for (const key of ["fixtureSae", "fixtureSlow"]) {
        const value = currentUrl.searchParams.get(key);
        if (value !== null) url.searchParams.set(key, value);
      }
    }
    if (currentUrl.searchParams.get("choose") === "1") {
      url.searchParams.set("choose", "1");
    } else if (currentUrl.searchParams.get("reopen-after-unload") === "1") {
      url.searchParams.set("reopen", "1");
    } else if (chooseModels) {
      url.searchParams.set("choose", "1");
    }
    window.location.replace(url.href);
  }

  function prepareWorkbenchUrl(): void {
    const url = new URL(window.location.href);
    url.searchParams.delete("choose");
    url.searchParams.delete("model");
    url.searchParams.delete("reopen-after-unload");
    url.searchParams.set("reopen", "1");
    window.history.replaceState(window.history.state, "", url);
    shownEntry = entryKey();
  }

  // The path and query of the history entry on screen. The workbench's rooms are hash routes
  // (drowse's rooms store pushes #/lens, #/artifacts/build, ...), so a history step that keeps
  // the path and query is a room move the dashboard's router follows, not a shell move.
  function entryKey(href: string = window.location.href): string {
    const url = new URL(href);
    return `${url.pathname}${url.search}`;
  }

  // User-initiated Chats/Models moves add a history entry so Back returns to
  // the previous view instead of leaving the app. Reaching the URL already
  // shown replaces the entry, so repeated clicks never stack duplicates.
  // The shell's views have no rooms: a room's hash left from the workbench
  // would otherwise reopen the next workbench in that room.
  function commitEntryUrl(url: URL, mode: HistoryMode, state: EntryHistoryState | null): void {
    url.hash = "";
    if (mode === "push" && url.href !== window.location.href) {
      window.history.pushState(state, "", url);
    } else {
      window.history.replaceState(state ?? window.history.state, "", url);
    }
    shownEntry = entryKey();
  }

  function showModels(
    modelVariantId?: string,
    conversation?: SavedConversationSummary,
    mode: HistoryMode = "push",
  ): void {
    if (conversation) queueConversationOpen(conversation.id, conversation.modelId);
    else clearConversationOpen();
    const url = new URL(window.location.href);
    url.searchParams.delete("reopen");
    url.searchParams.delete("reopen-after-unload");
    url.searchParams.set("choose", "1");
    if (modelVariantId) url.searchParams.set("model", modelVariantId);
    else url.searchParams.delete("model");
    commitEntryUrl(url, mode, conversation
      ? { drowseConversation: { id: conversation.id, modelId: conversation.modelId } }
      : {});
    entryView = "models";
  }

  function showHome(mode: HistoryMode = "push"): void {
    clearConversationOpen();
    const url = new URL(window.location.href);
    url.searchParams.delete("choose");
    url.searchParams.delete("model");
    url.searchParams.delete("reopen");
    url.searchParams.delete("reopen-after-unload");
    commitEntryUrl(url, mode, {});
    entryView = "home";
  }

  function focusPageHeading(): void {
    const heading = document.querySelector<HTMLElement>(".page-route h1");
    if (!heading) return;
    heading.tabIndex = -1;
    heading.focus({ preventScroll: true });
  }

  function onPopState(event: PopStateEvent): void {
    if (disposed || reloading || entryView === "resolving") return;
    const url = new URL(window.location.href);
    const target: EntryView = url.searchParams.get("choose") === "1" ? "models" : "home";
    const sameEntry = entryKey(url.href) === shownEntry;
    shownEntry = entryKey(url.href);
    if (Workbench) {
      if (sameEntry) return;
      if (!returningHome) void returnToChats(target === "models" ? "models" : "chats");
      return;
    }
    const pending = (event.state as EntryHistoryState | null)?.drowseConversation;
    if (target === "models" && pending) queueConversationOpen(pending.id, pending.modelId);
    else clearConversationOpen();
    // The models view reads ?model= when it mounts; remount it when only the
    // query changed so Back/Forward between two deep links stays accurate.
    if (target === entryView) {
      viewNonce += 1;
      void tick().then(focusPageHeading);
    }
    entryView = target;
  }

  function onBeforeUnload(event: BeforeUnloadEvent): void {
    if (!downloadActive || reloading || preparingPwaUpdate) return;
    event.preventDefault();
    event.returnValue = "";
  }

  function trackRemovedModels(snapshot: HostedShellSnapshot): void {
    if (snapshot.phase !== "supported") return;
    const installed = new Set(snapshot.models.filter(model => model.installed).map(model => model.id));
    const previous = installedModelIds;
    installedModelIds = installed;
    if (!previous) return;
    const replacement = [...installed][0] ?? null;
    for (const id of previous) {
      if (!installed.has(id) && snapshot.models.some(model => model.id === id)) {
        forgetEntryModel(id, replacement);
      }
    }
  }

  function enterPage(event: Event): void {
    (event.currentTarget as HTMLElement).inert = false;
    if (!routeMounted) return;
    (event.currentTarget as HTMLElement).classList.add("page-entering");
    window.scrollTo({ top: 0, behavior: "instant" });
    const heading = (event.currentTarget as HTMLElement).querySelector("h1");
    if (heading) {
      heading.tabIndex = -1;
      heading.focus({ preventScroll: true });
    }
  }

  async function returnToChats(destination: "chats" | "models" = "chats"): Promise<void> {
    if (returningHome) return;
    const modelVariantId = shellSnapshot?.runtime.modelVariantId ?? shellSnapshot?.selectedModelVariantId;
    returningHome = true;
    try {
      await appController.prepareForReload();
      await flushConversationAutosave();
      Workbench = null;
      if (destination === "models") showModels(modelVariantId ?? undefined, undefined, "replace");
      else showHome("replace");
      await tick();
      document.querySelector<HTMLElement>(destination === "models" ? ".app-shell h1" : ".home-shell h1")?.focus({ preventScroll: true });
    } finally {
      returningHome = false;
    }
  }

  async function resolveEntryView(): Promise<void> {
    const memory = readEntryMemory();
    let hasSavedChats = false;
    if (!memory && !requestedModelPicker && !requestedWorkbenchReopen) {
      try {
        hasSavedChats = await conversationLibrary.hasAny();
      } catch {
        hasSavedChats = false;
      }
    }
    if (disposed || !appController) return;
    // Returning visitors see their chats right away. The device check keeps
    // running in the background; the chats page gates every open on it.
    const knownReturning = memory !== null || hasSavedChats;
    const checking = appController.check();
    if (knownReturning && !requestedModelPicker && !requestedWorkbenchReopen) {
      firstRun = false;
      entryView = "home";
      await checking;
      return;
    }
    await checking;
    if (disposed) return;
    const checked = appController.current();
    const installedModel = checked.models.find((model) => model.setupComplete);
    const returning = knownReturning || installedModel !== undefined;
    if (!memory && installedModel) rememberCompletedOnboarding(installedModel.id);
    firstRun = !returning;
    entryView = requestedModelPicker || requestedWorkbenchReopen || !returning
      ? "models"
      : "home";
  }

  function denyTakeover(): void {
    if (takeoverPhase !== "requested") return;
    appController?.denyBusyTakeover();
  }

  async function allowTakeover(): Promise<void> {
    if (!appController || takeoverPhase !== "requested") return;
    allowingTakeover = true;
    restoreFocusOnClose = false;
    await appController.allowBusyTakeover();
    reloadToOnboarding(true);
  }

  async function prepareForPwaUpdate(): Promise<void> {
    await appController?.prepareForReload();
  }

  function onPwaApplyingChange(applying: boolean): void {
    preparingPwaUpdate = applying;
    if (
      !applying &&
      Workbench &&
      appController?.current().runtime.phase === "unloaded"
    ) {
      reloadToOnboarding();
    }
  }

  function onDialogKeydown(event: KeyboardEvent): void {
    event.stopPropagation();
    if (event.key === "Escape") {
      event.preventDefault();
      denyTakeover();
      return;
    }
    if (event.key !== "Tab" || !runtimeDialog) return;
    const focusable = [...runtimeDialog.querySelectorAll<HTMLElement>(FOCUSABLE)]
      .filter((element) => element.offsetParent !== null);
    if (focusable.length === 0) {
      event.preventDefault();
      runtimeDialog.focus();
      return;
    }
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    const active = document.activeElement;
    if (
      event.shiftKey &&
      (active === first || active === runtimeDialog || !runtimeDialog.contains(active))
    ) {
      event.preventDefault();
      last.focus();
    } else if (
      !event.shiftKey &&
      (active === last || !runtimeDialog.contains(active))
    ) {
      event.preventDefault();
      first.focus();
    }
  }

  async function mountWorkbench(): Promise<void> {
    if (!appController || Workbench || workbenchPromise) return;
    const capabilities = appController.capabilities();
    if (!capabilities) return;
    workbenchPromise = (async () => {
      workbenchError = null;
      workbenchNeedsReload = false;
      try {
        installRuntimeClient(appController.runtimeClient());
        installRuntimeCapabilities(capabilities);
        installHostedController(appController.hostedController());
        const component = (await import("../../../../drowse/webui/src/App.svelte")).default;
        if (!disposed) {
          if (appController.current().runtime.phase === "ready") {
            Workbench = component;
            completeChunkRecovery("workbench");
            // The models view clears its choice from the query as it opens the workbench, and a
            // recovery reload's marker leaves it just above: read the entry once both are gone.
            shownEntry = entryKey();
          } else {
            reloadToOnboarding();
          }
        }
      } catch (error) {
        if (
          await recoverFromChunkLoadError(
            error,
            "workbench",
            () => appController.prepareForReload(),
          )
        ) return;
        workbenchNeedsReload = isChunkLoadError(error);
        workbenchError = workbenchNeedsReload
          ? chunkLoadFailureMessage
          : userFacingError(
              error,
              "The workbench could not open. Reopen the model and try again.",
            );
      }
    })();
    try {
      await workbenchPromise;
    } finally {
      workbenchPromise = null;
    }
  }

  async function retryWorkbench(): Promise<void> {
    if (!workbenchNeedsReload) return mountWorkbench();
    if (workbenchReloading) return;
    workbenchReloading = true;
    try {
      await reloadHostedApp("workbench", async () => {
        await appController.prepareForReload();
        await flushConversationAutosave();
      });
    } catch (error) {
      workbenchError = userFacingError(error, "The app could not refresh. Please try again.");
      workbenchReloading = false;
    }
  }

  onMount(() => {
    if (!appController) return;
    const disposeTools = mountHostedWebMcp(appController);
    const disposeActions = registerPageActions({
      state: () => ({ route, recovery: runtimeRecovery ?? workbenchError }),
      home: () => Workbench ? returnToChats("chats") : showHome(),
      models: () => Workbench ? returnToChats("models") : showModels(),
      leave: async url => {
        returningHome = true;
        try {
          await appController.prepareForReload();
          await flushConversationAutosave();
          window.setTimeout(() => window.location.assign(url), 0);
        } catch (error) {
          returningHome = false;
          throw error;
        }
      },
      retry: retryWorkbench, allowTakeover, denyTakeover,
    });
    const unsubscribe = appController.subscribe((snapshot) => {
      shellSnapshot = snapshot;
      trackRemovedModels(snapshot);
      if (snapshot.runtime.phase === "ready") {
        const modelVariantId = snapshot.runtime.modelVariantId ?? snapshot.selectedModelVariantId;
        if (modelVariantId) rememberCompletedOnboarding(modelVariantId);
        firstRun = false;
        prepareWorkbenchUrl();
        runtimeRecovery = null;
        void mountWorkbench();
      } else if (Workbench && snapshot.runtime.phase === "failed") {
        runtimeRecovery = snapshot.runtime.reason;
      } else if (
        Workbench &&
        snapshot.runtime.phase === "unloaded" &&
        !allowingTakeover &&
        !preparingPwaUpdate &&
        !returningHome
      ) {
        reloadToOnboarding(true);
      }
    });
    void resolveEntryView();
    window.addEventListener("popstate", onPopState);
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => {
      unsubscribe(); disposeTools(); disposeActions();
      window.removeEventListener("popstate", onPopState);
      window.removeEventListener("beforeunload", onBeforeUnload);
    };
  });

  onDestroy(() => {
    disposed = true;
    appController?.dispose();
  });
</script>


<div class="hosted-background" inert={dialogOpen}>
  <PwaUpdatePrompt
    showOfflineReady={!Workbench}
    {downloadActive}
    onPauseDownload={() => appController.cancelDownload()}
    onPrepareReload={prepareForPwaUpdate}
    onApplyingChange={onPwaApplyingChange}
  />

  <div class="route-stage">
  {#key routeKey}
  <div class="page-route" data-route={route}
    in:rise={{ duration: routeMounted ? IN : 0 }}
    out:fadeOut={{ duration: routeMounted ? OUT : 0 }}
    onintrostart={enterPage}
    onintroend={(event) => {
      if (event.currentTarget.dataset.route !== "resolving") routeMounted = true;
      event.currentTarget.classList.remove("page-entering");
    }}
    onoutrostart={(event) => { event.currentTarget.inert = true; }}
  >
  {#if Workbench}
    <div class="workbench-route" inert={returningHome}>
      <Workbench onhome={() => returnToChats()} onmodels={() => returnToChats("models")} />
    </div>
  {:else if entryView === "resolving" || shellSnapshot === null}
    <div class="entry-loading">
      <SiteHeader variant="app" siteNavigation={false} />
      <main aria-busy="true" aria-live="polite">
        <p class="loading-placeholder">Opening your local workspace…</p>
      </main>
      <PageFooter />
    </div>
  {:else if entryView === "home"}
    <div class="home-route">
      <HostedHome
        controller={appController}
        snapshot={shellSnapshot}
        onChooseModels={showModels}
        onChats={focusPageHeading}
        {workbenchError}
        {workbenchNeedsReload}
        {workbenchReloading}
        onRetryWorkbench={() => void retryWorkbench()}
      />
    </div>
  {:else}
    <HostedApp
      controller={appController}
      {firstRun}
      onBack={() => showHome()}
      {workbenchError}
      {workbenchNeedsReload}
      {workbenchReloading}
      onRetryWorkbench={() => void retryWorkbench()}
    />
  {/if}
  </div>
  {/key}
  </div>
</div>

{#if dialogOpen}
  <div
    class="runtime-dialog-backdrop"
    in:fadeIn
    out:fadeOut
  >
    <div
      bind:this={runtimeDialog}
      class="runtime-dialog"
      role="alertdialog"
      aria-modal="true"
      aria-labelledby="runtime-dialog-title"
      aria-describedby="runtime-dialog-description"
      aria-busy={takeoverPhase === "allowing"}
      tabindex="-1"
      onkeydown={onDialogKeydown}
      in:rise
      out:fadeOut
      onintrostart={(event) => { event.currentTarget.inert = false; }}
      onoutrostart={(event) => { event.currentTarget.inert = true; }}
    >
      {#key dialogKey}
        <div
          class="runtime-dialog-content"
          in:rise
          out:fadeOut
        >
          {#if runtimeRecovery}
            <h1 id="runtime-dialog-title">Reload Drowse to recover</h1>
            <div id="runtime-dialog-description">
              <p>{runtimeRecovery}</p>
              <p>Your installed model and saved conversation remain on this device.</p>
            </div>
            <div class="runtime-dialog-actions">
              <button
                bind:this={dialogInitialFocus}
                type="button"
                class="dw-button"
                onclick={() => reloadToOnboarding()}
              >Reload Drowse</button>
            </div>
          {:else if takeoverPhase === "requested"}
            <h1 id="runtime-dialog-title">{takeoverForDownload ? "Another tab wants to take over this download" : "Another tab wants to use the GPU"}</h1>
            <div id="runtime-dialog-description">
              <p>{takeoverReason}</p>
              <p>{takeoverForDownload
                ? "Pause the download here and let the other tab continue. Files already downloaded are kept."
                : "Stop the current task, unload the model, and move it to the other tab."}</p>
            </div>
            <div class="runtime-dialog-actions">
              <button
                bind:this={dialogInitialFocus}
                type="button"
                class="dw-quiet"
                onclick={denyTakeover}
              >Keep working here</button>
              <button
                type="button"
                class="dw-button"
                onclick={() => void allowTakeover()}
              >{takeoverForDownload ? "Pause and release" : "Release runtime"}</button>
            </div>
          {:else}
            <h1 id="runtime-dialog-title">Releasing the local runtime</h1>
            <div id="runtime-dialog-description" role="status" aria-live="polite">
              <p>{takeoverReason}</p>
              <p>Drowse is stopping the current task and unloading the model before returning to setup.</p>
            </div>
          {/if}
        </div>
      {/key}
    </div>
  </div>
{/if}

<style>
  .hosted-background { display: contents; }

  .route-stage { position: relative; display: grid; min-width: 0; }
  .route-stage > :global(*) { grid-area: 1 / 1; min-width: 0; }
  .page-route { min-width: 0; }
  .workbench-route, .home-route { min-width: 0; }

  .entry-loading { display: flex; flex-direction: column; justify-content: space-between; min-height: 100dvh; }
  .entry-loading > main { display: grid; place-content: center; min-height: 50dvh; padding: var(--s6) var(--gutter); }

  /* A dialog over the page: the dashboard's scrim and a paper sheet, no line or shadow. */
  .runtime-dialog-backdrop {
    position: fixed;
    inset: 0;
    z-index: 10000;
    display: grid;
    place-items: center;
    padding: max(var(--s5), env(safe-area-inset-top)) max(var(--s5), env(safe-area-inset-right)) max(var(--s5), env(safe-area-inset-bottom)) max(var(--s5), env(safe-area-inset-left));
    overflow-y: auto;
    overscroll-behavior: contain;
    background: color-mix(in oklab, var(--wash2) 88%, transparent);
    -webkit-backdrop-filter: blur(8px);
    backdrop-filter: blur(8px);
  }
  .runtime-dialog { width: min(100%, 520px); padding: var(--s6); border-radius: var(--r-row); background: var(--paper); color: var(--ink2); }
  .runtime-dialog:focus { outline: none; }
  .runtime-dialog-content { display: grid; gap: var(--s4); min-width: 0; }
  .runtime-dialog h1 { color: var(--ink); font: var(--type-heading); letter-spacing: -0.02em; }
  .runtime-dialog-content > div { display: grid; gap: var(--s3); }
  .runtime-dialog-actions { display: flex; flex-wrap: wrap; gap: var(--s2); margin-top: var(--s2); }
  @media (prefers-reduced-transparency: reduce) { .runtime-dialog-backdrop { background: var(--wash2); } }
  @media (max-width: 720px) {
    .runtime-dialog-backdrop { align-items: end; padding: var(--s4) 0 0; }
    .runtime-dialog { width: 100%; padding: var(--s5) var(--s4) max(var(--s5), env(safe-area-inset-bottom)); border-radius: var(--r-row) var(--r-row) 0 0; }
    .runtime-dialog-actions > :global(button) { flex: 1 1 auto; }
  }
</style>
