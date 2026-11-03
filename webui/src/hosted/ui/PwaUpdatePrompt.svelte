<script lang="ts">
  import { MorphText, fadeOut, rise } from "@stdlogits/ledger";
  import { onDestroy, onMount } from "svelte";
  import { registerUpdateActions } from "../../../../drowse/webui/src/lib/webmcp/lifecycle";
  import { registerSW } from "virtual:pwa-register";
  import { userFacingError } from "../../../../drowse/webui/src/lib/runtime/userFacingError";
  import { hostedNetworkIsOffline } from "../runtime/networkState";
  import {
    UPDATE_REMINDER_KEY, readUpdateReminder, markUpdateShown,
    deferUpdate, clearUpdateReminder,
  } from "../runtime/updateReminder";

  const UPDATE_CONTROL_TIMEOUT_MS = 10_000;
  const CONFIRMATION_MS = 5_000;

  let {
    onPrepareReload,
    onApplyingChange,
    onPauseDownload,
    downloadActive = false,
    showOfflineReady = true,
  }: {
    onPrepareReload?: () => Promise<void>;
    onApplyingChange?: (applying: boolean) => void;
    /** Pauses the active model download so the reload doesn't interrupt it mid-write. */
    onPauseDownload?: () => Promise<void>;
    downloadActive?: boolean;
    showOfflineReady?: boolean;
  } = $props();

  let updateAvailable = $state(false);
  let offlineReady = $state(false);
  let applying = $state(false);
  let registrationError = $state<string | null>(null);
  let updateServiceWorker: ((reloadPage?: boolean) => Promise<void>) | null = null;
  let noticeEl: HTMLElement | null = $state(null);
  let returnFocus: HTMLElement | null = null;
  let registrationStarted = false;
  let waitingForOnline = false;
  let destroyed = false;
  let updatePending = false;
  let firstUpdateEntrance = $state(false);
  let confirmation = $state<string | null>(null);
  let confirmingPause = $state(false);
  let reminderTimer: number | undefined;
  let confirmationTimer: number | undefined;
  let deferredStartTimer: number | undefined;

  onMount(() => {
    void initializeServiceWorker();
    window.addEventListener("storage", syncReminderStorage);
    window.addEventListener("focus", refreshUpdateNotice);
    document.addEventListener("visibilitychange", refreshUpdateNotice);
    navigator.serviceWorker?.addEventListener("controllerchange", updateControlled);
    return registerUpdateActions({
      state: () => ({ available: updateAvailable, applying, error: registrationError }),
      apply: applyUpdate, remindLater,
    });
  });

  onDestroy(() => {
    if (typeof window === "undefined") return;
    destroyed = true;
    window.clearTimeout(deferredStartTimer);
    window.removeEventListener("load", deferUpdateMonitoring);
    stopWaitingForOnline();
    window.clearTimeout(reminderTimer);
    window.clearTimeout(confirmationTimer);
    window.removeEventListener("storage", syncReminderStorage);
    window.removeEventListener("focus", refreshUpdateNotice);
    document.removeEventListener("visibilitychange", refreshUpdateNotice);
    navigator.serviceWorker?.removeEventListener("controllerchange", updateControlled);
  });

  function refreshUpdateNotice(): void {
    window.clearTimeout(reminderTimer);
    if (destroyed || !updatePending || applying) return;
    const reminder = readUpdateReminder();
    const remaining = reminder.remindAt - Date.now();
    if (remaining > 0) {
      updateAvailable = false;
      reminderTimer = window.setTimeout(refreshUpdateNotice, Math.min(remaining, 2_147_483_647));
    } else if (!document.hidden && !updateAvailable) {
      firstUpdateEntrance = !reminder.shown;
      markUpdateShown();
      confirmation = null;
      offlineReady = false;
      updateAvailable = true;
    }
  }

  function syncReminderStorage(event: StorageEvent): void {
    if (event.key === UPDATE_REMINDER_KEY || event.key === null) refreshUpdateNotice();
  }

  function updateControlled(): void {
    clearUpdateReminder();
    updatePending = false;
    updateAvailable = false;
    confirmation = null;
    window.clearTimeout(reminderTimer);
    window.clearTimeout(confirmationTimer);
  }

  function remindLater(): void {
    if (!updateAvailable || applying) return;
    confirmingPause = false;
    confirmation = `Okay! We'll remind you in ${deferUpdate()}.`;
    offlineReady = false;
    refreshUpdateNotice();
    window.clearTimeout(confirmationTimer);
    confirmationTimer = window.setTimeout(() => { confirmation = null; }, CONFIRMATION_MS);
    restoreFocus();
  }

  async function initializeServiceWorker(): Promise<void> {
    if ("serviceWorker" in navigator && hostedNetworkIsOffline()) {
      const registration = await navigator.serviceWorker.getRegistration("/");
      if (destroyed) return;
      if (registration?.active && navigator.serviceWorker.controller) {
        offlineReady = true;
        waitForOnline();
        return;
      }
    }
    // The app needs the worker for offline models; on the public pages, installing it can wait until
    // the page has loaded and gone idle so the precache doesn't compete with first paint.
    if (/^\/app(?:\/|$)/.test(window.location.pathname)) startUpdateMonitoring();
    else if (document.readyState === "complete") deferUpdateMonitoring();
    else window.addEventListener("load", deferUpdateMonitoring, { once: true });
  }

  function deferUpdateMonitoring(): void {
    if (destroyed) return;
    // Safari only gained requestIdleCallback recently.
    if (typeof window.requestIdleCallback === "function") window.requestIdleCallback(startUpdateMonitoring, { timeout: 5_000 });
    else deferredStartTimer = globalThis.setTimeout(startUpdateMonitoring, 2_000) as unknown as number;
  }

  function waitForOnline(): void {
    if (destroyed || registrationStarted || waitingForOnline) return;
    waitingForOnline = true;
    window.addEventListener("online", resumeUpdateMonitoring);
    if (!hostedNetworkIsOffline()) resumeUpdateMonitoring();
  }

  function stopWaitingForOnline(): void {
    if (!waitingForOnline) return;
    waitingForOnline = false;
    window.removeEventListener("online", resumeUpdateMonitoring);
  }

  function resumeUpdateMonitoring(): void {
    stopWaitingForOnline();
    startUpdateMonitoring();
  }

  function startUpdateMonitoring(): void {
    if (destroyed || registrationStarted) return;
    registrationStarted = true;
    stopWaitingForOnline();
    updateServiceWorker = registerSW({
      immediate: true,
      onNeedRefresh: () => {
        if (destroyed) return;
        updatePending = true;
        offlineReady = false;
        refreshUpdateNotice();
      },
      onRegisteredSW: (_url, registration) => {
        if (destroyed || !registration) return;
        if (!registration.waiting && !registration.installing) clearUpdateReminder();
      },
      onNeedReload: () => undefined,
      onOfflineReady: () => {
        if (!destroyed && !updatePending) offlineReady = true;
      },
      onRegisterError: (error) => {
        registrationError = userFacingError(
          error,
          "Offline support could not start. Reload while connected and try again.",
        );
      },
    });
  }

  async function applyUpdate(pauseConfirmed = false): Promise<void> {
    if (!updateServiceWorker || applying) return;
    if (downloadActive && !pauseConfirmed) {
      confirmingPause = true;
      return;
    }
    applying = true;
    try {
      onApplyingChange?.(true);
      if (downloadActive) await onPauseDownload?.();
      confirmingPause = false;
      await onPrepareReload?.();
      const controlling = waitForUpdatedController();
      await updateServiceWorker(false);
      await controlling;
      window.location.reload();
    } catch (error) {
      registrationError = userFacingError(
        error,
        "Unable to apply the update safely. Reload the page and try again.",
      );
      updateAvailable = false;
      confirmingPause = false;
      applying = false;
      onApplyingChange?.(false);
    }
  }

  function waitForUpdatedController(): Promise<void> {
    return new Promise((resolve, reject) => {
      const previousController = navigator.serviceWorker.controller;
      const onControllerChange = () => {
        const nextController = navigator.serviceWorker.controller;
        if (!nextController || nextController === previousController) return;
        window.clearTimeout(timeout);
        navigator.serviceWorker.removeEventListener("controllerchange", onControllerChange);
        resolve();
      };
      const timeout = window.setTimeout(() => {
        navigator.serviceWorker.removeEventListener("controllerchange", onControllerChange);
        reject(new Error("The updated app could not take control. Try reloading this page."));
      }, UPDATE_CONTROL_TIMEOUT_MS);
      navigator.serviceWorker.addEventListener("controllerchange", onControllerChange);
    });
  }

  function rememberReturnFocus(event: FocusEvent): void {
    if (event.relatedTarget instanceof HTMLElement && !noticeEl?.contains(event.relatedTarget)) {
      returnFocus = event.relatedTarget;
    }
  }

  function dismissNotice(): void {
    offlineReady = false;
    registrationError = null;
    restoreFocus();
  }

  function restoreFocus(): void {
    const target = returnFocus;
    queueMicrotask(() => {
      if (target?.isConnected) {
        target.focus();
        return;
      }
      const main = document.querySelector<HTMLElement>("main");
      if (!main) return;
      const heroAction = main.querySelector<HTMLAnchorElement>(".hero .primary-action");
      if (heroAction) {
        heroAction.focus({ preventScroll: true });
        return;
      }
      main.tabIndex = -1;
      main.focus();
    });
  }
</script>

{#if updateAvailable || (offlineReady && showOfflineReady) || registrationError}
  <div
    bind:this={noticeEl}
    class="pwa-notice"
    class:first-update={updateAvailable && firstUpdateEntrance}
    class:passive={offlineReady && !updateAvailable && !registrationError}
    role={registrationError ? "alert" : "status"}
    aria-live={registrationError ? "assertive" : "polite"}
    aria-atomic="true"
    aria-busy={applying}
    onfocusin={rememberReturnFocus}
    in:rise={{ duration: updateAvailable ? 0 : undefined }}
    out:fadeOut
  >
    <div>
      {#if updateAvailable}
        <strong>A Drowse update is ready.</strong>
        {#if confirmingPause && downloadActive}
          <span>Updating will pause your download. Files already downloaded are kept, and you can resume after the update.</span>
        {:else}
          <span>Your installed models and conversations will stay on this device.</span>
        {/if}
      {:else if offlineReady && !registrationError}
        <strong>The interface is ready offline.</strong>
        <span>Once a model is installed, Drowse can reopen and generate offline.</span>
      {:else}
        <strong>Offline app setup failed.</strong>
        <span>{registrationError}</span>
      {/if}
    </div>
    <div class="pwa-actions">
      {#if updateAvailable}
        <button type="button" class="primary" disabled={applying} aria-busy={applying} onclick={() => void applyUpdate(confirmingPause)}>
          <MorphText text={applying ? "Saving local work…" : confirmingPause && downloadActive ? "Pause and update" : "Update and reload"} numbers={false} />
        </button>
        <button type="button" disabled={applying} onclick={remindLater}>
          Update later
        </button>
      {:else}
        <button
          type="button"
          onclick={dismissNotice}
        >Dismiss</button>
      {/if}
    </div>
  </div>
{/if}

{#if confirmation}
  <div
    class="pwa-notice pwa-confirmation"
    role="status"
    aria-live="polite"
    aria-atomic="true"
    in:rise
    out:fadeOut
  >
    <span>{confirmation}</span>
  </div>
{/if}

<style>
  /* An update or offline notice: a wash card in the corner, one sentence and its two choices. */
  .pwa-notice {
    position: fixed;
    z-index: 45;
    inset-inline-end: calc(var(--s5) + env(safe-area-inset-right, 0px));
    bottom: calc(var(--s5) + env(safe-area-inset-bottom, 0px));
    display: grid;
    gap: var(--s3);
    width: min(440px, calc(100vw - var(--s5) * 2 - env(safe-area-inset-left, 0px) - env(safe-area-inset-right, 0px)));
    max-height: calc(100dvh - var(--s6) - env(safe-area-inset-top, 0px) - env(safe-area-inset-bottom, 0px));
    overflow-y: auto;
    padding: var(--s4) var(--s5);
    border-radius: var(--r-row);
    background: var(--wash2);
    color: var(--ink2);
    font: var(--type-small);
  }
  .pwa-notice > div:first-child { display: grid; gap: var(--s1); }
  .pwa-notice strong { color: var(--ink); font: var(--type-label); }
  .pwa-actions { display: flex; flex-wrap: wrap; gap: var(--s2); }
  .pwa-actions button { display: inline-flex; align-items: center; justify-content: center; height: 36px; padding: 0 var(--s4); border: 0; border-radius: var(--r-pill); background: var(--paper); color: var(--ink); font: var(--type-control); white-space: nowrap; cursor: pointer; transition: opacity var(--fast) var(--ease), background-color var(--fast) var(--ease), scale var(--fast) var(--ease); }
  .pwa-actions button.primary { padding: 0 var(--s5); background: var(--ink); color: var(--paper); }
  .pwa-actions button:active:not(:disabled) { scale: 0.97; }
  .pwa-actions button:disabled { opacity: 0.4; cursor: default; }
  @media (hover: hover) {
    .pwa-actions button:hover:not(:disabled) { background: var(--wash); }
    .pwa-actions button.primary:hover:not(:disabled) { background: var(--ink); opacity: 0.86; }
  }
  .pwa-confirmation { display: flex; align-items: center; gap: var(--s3); }
  @media (max-width: 720px) {
    .pwa-notice { inset-inline: var(--s4); bottom: calc(var(--s4) + env(safe-area-inset-bottom, 0px)); width: auto; padding: var(--s4); }
    .pwa-actions button { flex: 1 1 auto; height: 44px; }
  }
  @media (forced-colors: active) { .pwa-notice { border: 1px solid CanvasText; } }
</style>
