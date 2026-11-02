<script lang="ts">
  import { Expander, Icon, MorphText, fadeIn, fadeOut, rise, tips } from "@stdlogits/ledger";
  import BaseModelTag from "../../../../drowse/webui/src/lib/ui/BaseModelTag.svelte";
  import RollingNumber from "../../../../drowse/webui/src/lib/ui/RollingNumber.svelte";
  import { onMount } from "svelte";
  import { registerModelPicker } from "../../../../drowse/webui/src/lib/webmcp/lifecycle";
  import TipText from "./TipText.svelte";
  import SiteHeader from "./SiteHeader.svelte";
  import PageFooter from "./PageFooter.svelte";
  import { formatEtaRange } from "../../lib/runtime/eta";
  import type {
    HostedCheckState,
    HostedModelOption,
    HostedShellController,
    HostedShellSnapshot,
  } from "../../../../drowse/webui/src/lib/shell";
  import type { BrowserRuntimeClass } from "../../../../drowse/webui/src/lib/runtime/contracts";
  import { userFacingError } from "../../../../drowse/webui/src/lib/runtime/userFacingError";
  import { reloadInstructions, appRefreshSafetyMessage } from "../runtime/chunkRecovery";
  import { clearConversationOpen } from "../../../../drowse/webui/src/lib/conversationOpen";
  import {
    defaultModelSelection,
    downloadSizeSummary,
    downloadUnavailableReason,
    modelsBySize,
    remainingDownloadParts,
  } from "./modelSelection";
  import { formatBytes } from "../../lib/format/bytes";
  import { storageShortfallBytes } from "../../lib/runtime/recommendation";

  let {
    controller,
    firstRun = true,
    onBack,
    onChats,
    workbenchError = null,
    workbenchNeedsReload = false,
    workbenchReloading = false,
    onRetryWorkbench,
  }: {
    controller: HostedShellController;
    firstRun?: boolean;
    onBack?: () => void;
    onChats?: () => void;
    workbenchError?: string | null;
    workbenchNeedsReload?: boolean;
    workbenchReloading?: boolean;
    onRetryWorkbench?: () => void;
  } = $props();
  let snapshot = $state<HostedShellSnapshot>({
    phase: "idle",
    headline: "Check this device",
    detail: "Preparing the compatibility check.",
    checks: [],
    models: [],
    download: {
      available: false,
      phase: "locked",
      reason: "Checking the signed distribution configuration.",
    },
    runtime: {
      available: false,
      phase: "locked",
      reason: "Checking the local model engine.",
    },
  });
  $effect(() => { document.title = "Models · Drowse"; });
  // The device check's details open by themselves when the check fails.
  let deviceDetailsOpen = $state(false);
  $effect(() => {
    if (snapshot.phase === "unsupported" || snapshot.phase === "failed") deviceDetailsOpen = true;
  });
  let selectedModel = $state<string | null>(null);
  // The model chosen for the visitor on first run, so its reason can be explained.
  let autoSelectedModel = $state<string | null>(null);
  let automaticSelectionDone = false;
  let discardConfirmation = $state<string | null>(null);
  let discardingPartial = $state(false);
  let discardError = $state<string | null>(null);
  let baseModelsOpen = $state(false);
  let unsafeConfirmation = $state<string | null>(null);
  let confirmSessionReset = $state(false);
  let deleteConfirmation = $state<string | null>(null);
  let deletingModel = $state(false);
  let deletionError = $state<string | null>(null);
  let deletionNotice = $state<string | null>(null);
  const modelOperationBusy = $derived(deletingModel || snapshot.runtime.phase === "loading" || ["requesting_persistence", "downloading", "cancelling"].includes(snapshot.download.phase));
  async function deleteSelectedModel() {
    const id = deleteConfirmation;
    if (!id || modelOperationBusy) return;
    const name = snapshot.models.find(model => model.id === id)?.name ?? "Model";
    deletingModel = true;
    deletionError = null;
    try {
      await controller.deleteModel(id);
      deleteConfirmation = null;
      deletionNotice = `${name} removed from this device. Your chats and analysis packs are kept. Download the model again to continue its chats.`;
    } catch (error) {
      deletionError = userFacingError(error, "The model could not be removed. Close other Drowse tabs and try again.");
    } finally {
      deletingModel = false;
    }
  }
  let packSelectionError = $state<string | null>(null);
  let persistenceRetrying = $state(false);
  let persistenceRetryAttempted = $state(false);
  const persistenceRequestAvailable = typeof navigator.storage?.persist === "function";
  const storageUnprotected = $derived((snapshot.storage !== undefined && snapshot.storage.persisted !== true) || snapshot.download.persistenceDenied === true);
  let appleMobile = $state(false);
  let runtimeClass = $state<BrowserRuntimeClass | null>(null);
  let automaticOpenStarted = false;
  const workbenchReopenRequested = new URL(window.location.href).searchParams.get("reopen") === "1";
  const requestedModelId = new URL(window.location.href).searchParams.get("model");
  const selectedOption = $derived(
    snapshot.models.find((model) => model.id === selectedModel) ?? null,
  );
  const unfinishedDownloadPhases = ["requesting_persistence", "downloading", "cancelling", "paused", "failed"];
  // Download progress, pauses, and failures belong to one model. Other models only
  // see the global idle or locked state, so a paused download never relabels them.
  const selectedDownloadPhase = $derived(
    selectedModel !== null && snapshot.download.modelVariantId === selectedModel
      ? snapshot.download.phase
      : snapshot.download.available ? "idle" : "locked",
  );
  const selectedProgress = $derived(
    selectedModel !== null && snapshot.download.modelVariantId === selectedModel
      ? snapshot.download.progress
      : undefined,
  );
  const otherDownloadActive = $derived(
    snapshot.download.modelVariantId !== undefined &&
      snapshot.download.modelVariantId !== selectedModel &&
      ["requesting_persistence", "downloading", "cancelling"].includes(snapshot.download.phase),
  );
  const keptBytes = $derived(
    !selectedOption || selectedOption.setupComplete
      ? 0
      : selectedDownloadPhase === "paused" && selectedProgress
        ? selectedProgress.bytesReceived
        : selectedOption.partialBytes ?? 0,
  );
  const downloadsLocked = $derived(
    snapshot.phase === "supported" && !snapshot.download.available,
  );
  const modelChoicesAvailable = $derived(
    snapshot.phase === "supported" &&
      snapshot.models.some((model) => model.fit !== "blocked") &&
      (snapshot.download.available || snapshot.models.some((model) => model.setupComplete)),
  );
  const displayHeadline = $derived(
    modelChoicesAvailable
      ? firstRun
        ? "Choose your first model"
        : "Models"
      : snapshot.headline,
  );
  const displayDetail = $derived(
    modelChoicesAvailable
      ? firstRun
        ? "The workbench opens automatically after download and setup."
        : "Open an installed model or download another one. Your saved chats are kept separately."
      : snapshot.detail,
  );
  const showDownloadAction = $derived(
    selectedOption !== null && selectedOption.fit !== "blocked" &&
    snapshot.download.available && (
      !selectedOption?.setupComplete ||
      (selectedOption?.remainingDownloadBytes ?? 0) > 0 ||
      unfinishedDownloadPhases.includes(selectedDownloadPhase)
    ),
  );
  const downloadComplete = $derived(
    snapshot.phase === "supported" && selectedOption?.setupComplete === true &&
    selectedOption.fit !== "blocked" && !showDownloadAction &&
    !unfinishedDownloadPhases.includes(selectedDownloadPhase),
  );
  const passedChecks = $derived(
    snapshot.checks.filter((check) => check.state === "pass").length,
  );
  const advisoryChecks = $derived(
    snapshot.checks.filter((check) => check.state === "warn").length,
  );
  const checkCountLabel = $derived(
    `${passedChecks} of ${snapshot.checks.length} passed` +
      (advisoryChecks ? ` · ${advisoryChecks} note${advisoryChecks === 1 ? "" : "s"}` : ""),
  );
  const checkStatus = $derived(
    snapshot.phase === "checking" || snapshot.phase === "idle"
      ? "checking"
      : snapshot.checks.length > 0
        ? passedChecks === snapshot.checks.length
          ? "ready"
          : passedChecks > 0 ? "partial" : "unavailable"
        : snapshot.phase === "supported" ? "ready" : "unavailable",
  );
  const previewBrowser = $derived(
    runtimeClass === "desktop-webkit"
      ? "Safari preview"
      : runtimeClass === "desktop-gecko"
        ? "Firefox preview"
        : null,
  );
  const checkSummaryLabel = $derived(
    snapshot.phase === "checking"
      ? "Checking browser and graphics"
      : snapshot.phase === "supported"
          ? appleMobile
            ? advisoryChecks
              ? `iPhone preview checks passed with ${advisoryChecks} note${advisoryChecks === 1 ? "" : "s"}`
              : "iPhone preview checks passed"
            : previewBrowser
              ? advisoryChecks
                ? `${previewBrowser} checks passed with ${advisoryChecks} note${advisoryChecks === 1 ? "" : "s"}`
                : `${previewBrowser} checks passed`
          : advisoryChecks
            ? `Device ready with ${advisoryChecks} note${advisoryChecks === 1 ? "" : "s"}`
            : "Device ready"
        : snapshot.phase === "unsupported" || snapshot.phase === "failed"
          ? "Review these checks"
          : "Device check",
  );

  const checkLabel: Record<HostedCheckState, string> = {
    pending: "Waiting",
    running: "Checking",
    pass: "Ready",
    warn: "Needs verification",
    fail: "Unavailable",
  };

  const tierLabel: Record<HostedModelOption["tier"], string> = {
    fastest: "Fastest",
    balanced: "Balanced",
    quality: "Best quality",
  };

  const fitLabel: Record<HostedModelOption["fit"], string> = {
    recommended: "Recommended",
    eligible: "Good fit",
    uncertain: "Needs confirmation",
    blocked: "Not supported",
  };

  const formatAvailable = (bytes?: number) =>
    bytes == null ? "Storage estimate unavailable" : `${formatBytes(bytes)} available`;

  const spaceNeeded = (model: HostedModelOption) =>
    model.setupComplete || model.catalogAvailable === false
      ? 0
      : storageShortfallBytes(model.remainingDownloadBytes, snapshot.storage?.availableBytes);

  // Card size: what the next download still needs, or the stored size once set up.
  const cardSize = (model: HostedModelOption) =>
    model.setupComplete || model.remainingDownloadBytes === 0
      ? formatBytes(model.firstRunBytes)
      : formatBytes(model.remainingDownloadBytes);

  const toolChips = (model: HostedModelOption) => [
    ...(model.firstRunPacks.some((pack) => pack.kind === "jlens") ? ["Word insights (R lens)"] : []),
    ...(model.firstRunPacks.some((pack) => packKind(pack) === "slens") ? ["Early-layer reads (S lens)"] : []),
    ...(model.firstRunPacks.some((pack) => packKind(pack) === "units") ? ["Neurons and heads (units)"] : []),
    ...(model.firstRunPacks.some((pack) => pack.kind === "sae") ? ["Feature insights (SAE)"] : []),
  ];

  const measuredSpeed = (model: HostedModelOption) =>
    model.measuredTokensPerSecond
      ? `Ran at ~${model.measuredTokensPerSecond >= 10 ? Math.round(model.measuredTokensPerSecond) : model.measuredTokensPerSecond.toFixed(1)} tok/s on this device`
      : null;

  const storedBytes = (model: HostedModelOption) =>
    model.modelDownloadBytes + model.firstRunPacks
      .filter((pack) => pack.installed)
      .reduce((total, pack) => total + pack.bytes, 0);
  const installedModels = $derived(snapshot.models.filter((model) => model.installed));
  const comparableModels = $derived(
    modelsBySize(snapshot.models.filter((model) => model.catalogAvailable !== false))
      .toSorted((left, right) => Number(left.modelType === "base") - Number(right.modelType === "base")),
  );

  const packName = (pack: HostedModelOption["firstRunPacks"][number]) =>
    pack.name;

  // 6.0's pack kinds (jlens, slens, units, sae); the shared shell type names jlens and sae only
  // until drowse's `HostedModelOption` carries the other two.
  const packKind = (pack: HostedModelOption["firstRunPacks"][number]) => pack.kind as string;

  const packTitle = (pack: HostedModelOption["firstRunPacks"][number]) =>
    pack.kind === "jlens"
      ? "Word insights"
      : packKind(pack) === "slens"
        ? "Early-layer reads"
        : packKind(pack) === "units"
          ? "Neurons and heads"
          : /gemma scope/i.test(pack.name)
            ? "Gemma Scope features"
            : "Feature explorer";

  const packTag = (pack: HostedModelOption["firstRunPacks"][number]) =>
    pack.kind === "jlens"
      ? "optional readout"
      : packKind(pack) === "slens"
        ? "S lens"
        : packKind(pack) === "units"
          ? "units"
          : "optional SAE";

  const packDescription = (pack: HostedModelOption["firstRunPacks"][number]) =>
    pack.kind === "jlens"
      ? `${pack.name}. Read word predictions across layers. This lens is already fitted; your browser doesn't train it.`
      : packKind(pack) === "slens"
        ? `${pack.name}. Read early layers as input tokens. Already fitted; each layer downloads the first time it is read.`
        : packKind(pack) === "units"
          ? `${pack.name}. Read, rank and ablate neurons and heads. Already computed; each layer's statistics download the first time they are read.`
          : `${pack.name}. Inspect and steer learned features. This SAE is already trained; your browser doesn't train it.`;
  const appleStorageHelp =
    "Adding Drowse to your iPhone or iPad Home Screen can help it keep data. Back up your chats first. Existing chats and models won't transfer to the Home Screen app.";
  const macDesktop = /Macintosh|Mac OS X/i.test(navigator.userAgent) ||
    /^Mac/i.test(navigator.platform);
  const firefoxSwitchGuidance = macDesktop
    ? "Update Firefox to the latest version and check again. If it remains incompatible, switch to Safari or a Chromium-based browser such as Chrome, Edge, Brave, or Arc."
    : "Update Firefox to the latest version and check again. If it remains incompatible, switch to a Chromium-based browser such as Chrome, Edge, or Brave.";
  // Never suggest the browser that just failed the check.
  const alternativeBrowsers = (() => {
    const agent = navigator.userAgent;
    const current = /Firefox|FxiOS/i.test(agent)
      ? "Firefox"
      : /Edg\//i.test(agent)
        ? "Edge"
        : /Chrome|CriOS|Chromium/i.test(agent)
          ? "Chrome"
          : /Safari/i.test(agent) ? "Safari" : null;
    const options = ["Safari", "Firefox", "Chrome", "Edge"].filter((name) => name !== current);
    return `${options.slice(0, -1).join(", ")}, or ${options.at(-1)}`;
  })();
  const firefoxIncompatible = $derived(
    runtimeClass === "desktop-gecko" &&
      snapshot.headline === "Firefox is incompatible with the current model runtime",
  );
  const recoveryGuidance = $derived(
    appleMobile
      ? "Review the failed check above. On iPhone or iPad, Drowse needs Safari 26 or later, WebGPU, and working local storage."
      : runtimeClass === "desktop-webkit"
        ? "Review the failed check above. Drowse needs Safari 26 on macOS 26, working local storage, and a graphics adapter that passes the WebGPU compute check. Lockdown Mode can disable WebGPU."
        : runtimeClass === "desktop-gecko"
          ? firefoxSwitchGuidance
          : `Use a current ${alternativeBrowsers} browser on a device with WebGPU, then check again.`,
  );
  const lockedDownloadGuidance = $derived(
    snapshot.catalogError
      ? "Drowse couldn't load the verified model list. Check your connection, then check again."
      : firefoxIncompatible
        ? null
        : "Downloads are unavailable right now. Check again after reconnecting or updating your browser.",
  );
  const runtimeGateTitle = $derived(
    snapshot.runtime.phase === "loading"
      ? "Loading the model"
      : snapshot.runtime.phase === "failed"
        ? "The model could not open"
        : "Ready to open",
  );
  const runtimeGateAction = $derived(
    snapshot.runtime.phase === "failed" ? "Try opening again" : "Open workbench",
  );

  const speedClass = (speed: string) => {
    const lowerBound = Number.parseFloat(speed);
    if (!Number.isFinite(lowerBound)) return "Measured after the first reply";
    if (lowerBound >= 40) return "Very fast";
    if (lowerBound >= 20) return "Fast";
    if (lowerBound >= 8) return "Moderate";
    return "Slower";
  };

  const clearModelChoiceRequest = () => {
    const url = new URL(window.location.href);
    url.searchParams.delete("choose");
    url.searchParams.delete("model");
    url.searchParams.set("reopen", "1");
    window.history.replaceState(window.history.state, "", url);
  };

  const openInstalledModel = async (next: HostedShellSnapshot) => {
    if (
      !workbenchReopenRequested || automaticOpenStarted || next.phase !== "supported" ||
      !next.runtime.available || next.runtime.phase !== "unloaded"
    ) return;
    const model = next.models.find(
      (candidate) => candidate.id === next.selectedModelVariantId,
    ) ?? next.models.find((candidate) => candidate.setupComplete);
    if (!model?.setupComplete || model.fit === "blocked" || model.requiresOomRetry) return;
    automaticOpenStarted = true;
    selectedModel = model.id;
    try {
      await controller.open(model.id);
      clearModelChoiceRequest();
    } catch {}
  };

  onMount(() => {
    const disposeModelPicker = registerModelPicker(id => {
      selectedModel = id;
      if (snapshot.models.find(model => model.id === id)?.modelType === "base") baseModelsOpen = true;
    });
    snapshot = controller.current();
    const unsubscribe = controller.subscribe((next) => {
      snapshot = next;
      const capabilities = controller.capabilities();
      appleMobile = capabilities?.signals.appleMobile === true;
      runtimeClass = capabilities?.signals.runtimeClass ?? null;
      if (!selectedModel || !next.models.some((model) => model.id === selectedModel)) {
        const requestedSelection = requestedModelId ?? (next.download.phase !== "idle" ? next.download.modelVariantId : undefined);
        selectedModel = next.models.find((model) => model.id === requestedSelection)?.id ?? null;
        if (
          selectedModel === null && firstRun && requestedModelId === null && !automaticSelectionDone &&
          next.phase === "supported" && next.catalogError === undefined && next.download.available &&
          !unfinishedDownloadPhases.includes(next.download.phase)
        ) {
          automaticSelectionDone = true;
          selectedModel = defaultModelSelection(next.models.filter((model) => model.fit !== "blocked"), null);
          autoSelectedModel = selectedModel;
        }
        if (next.models.find((model) => model.id === selectedModel)?.modelType === "base") {
          baseModelsOpen = true;
        }
      }
      void openInstalledModel(next);
    });
    return () => { unsubscribe(); disposeModelPicker(); };
  });

  const etaLabel = (progress = selectedProgress) => {
    if (!progress) return "Preparing download";
    if (progress.bytesTotal > 0 && progress.bytesReceived >= progress.bytesTotal) {
      return "Complete";
    }
    if (progress.offline) return "Offline. Waiting for a connection";
    if (progress.stalled) return "Waiting for download data";
    if (progress.calculatingEta || !progress.etaSeconds) return "Calculating…";
    return `${formatEtaRange(progress.etaSeconds)} remaining`;
  };

  const progressPercent = (progress = selectedProgress) =>
    progress && progress.bytesTotal > 0
      ? Math.min(100, (progress.bytesReceived / progress.bytesTotal) * 100)
      : 0;

  const startDownload = async (explicitUnsafeOverride = false) => {
    if (!selectedModel || modelOperationBusy) return;
    if (selectedOption?.fit === "uncertain" && !explicitUnsafeOverride) {
      unsafeConfirmation = selectedModel;
      return;
    }
    const modelId = selectedModel;
    unsafeConfirmation = null;
    try {
      await controller.download(modelId, { explicitUnsafeOverride });
      const next = controller.current();
      const installed = next.models.find((model) => model.id === modelId);
      if (
        installed?.installed && next.download.phase === "installed" &&
        next.runtime.available &&
        next.runtime.phase !== "ready"
      ) {
        await controller.open(modelId);
        clearModelChoiceRequest();
      }
    } catch {}
  };

  const setPackSelected = (packId: string, selected: boolean) => {
    if (!selectedModel) return;
    try {
      controller.setOptionalPackSelected(selectedModel, packId, selected);
      packSelectionError = null;
    } catch (error) {
      packSelectionError = userFacingError(
        error,
        "That model tool could not be selected. Choose another tool or model and try again.",
      );
    }
  };

  const retryPersistence = async () => {
    if (persistenceRetrying) return;
    persistenceRetrying = true;
    try {
      persistenceRetryAttempted = !(await controller.retryPersistence());
    } catch {
      persistenceRetryAttempted = true;
    } finally {
      persistenceRetrying = false;
    }
  };

  const discardPartialDownload = async () => {
    const id = discardConfirmation;
    if (!id || !controller.discardPartialDownload || modelOperationBusy) return;
    discardingPartial = true;
    discardError = null;
    try {
      await controller.discardPartialDownload(id);
      discardConfirmation = null;
    } catch (error) {
      discardError = userFacingError(error, "The kept files could not be removed. Close other Drowse tabs and try again.");
    } finally {
      discardingPartial = false;
    }
  };

  const openModel = async (resetSession = false) => {
    if (!selectedModel) return;
    try {
      await controller.open(selectedModel, { resetSession });
      confirmSessionReset = false;
      clearModelChoiceRequest();
    } catch {}
  };
</script>

{#snippet storageProtectionNotice()}
  <div class="persistence-notice" role="status" aria-label="Storage protection">
    <div>
      <strong>Local storage is available</strong>
      <p class="persistence-help">
        {persistenceRetryAttempted
          ? "Downloads and chats still save on this device. Your browser may remove them during automatic cleanup, so back up your chats."
          : persistenceRequestAvailable
            ? "Downloads and chats save on this device. Storage protection is optional and helps prevent automatic cleanup."
            : "Downloads and chats save on this device, but the browser may remove them during automatic cleanup. Back up your chats."}
      </p>
      {#if appleMobile}<p class="persistence-help">{appleStorageHelp}</p>{/if}
    </div>
    {#if persistenceRequestAvailable}
      <button
        class="quiet-action"
        type="button"
        disabled={persistenceRetrying}
        aria-busy={persistenceRetrying}
        onclick={() => void retryPersistence()}
      >{persistenceRetrying ? "Requesting protection…" : "Protect storage"}</button>
    {/if}
  </div>
{/snippet}

<a class="skip-link" href="#device-check">Skip to device check</a>

<div class="app-shell" use:tips class:hardware-compatible={snapshot.phase === "supported"} class:download-complete={downloadComplete}>
  <SiteHeader current="models" variant="app" onChats={onChats ?? onBack} onModels={() => document.querySelector<HTMLElement>("#models-title")?.focus()} />

  <main id="device-check">
    <div class="intro" data-page-group="0">
      <h1 id="models-title" tabindex="-1"><MorphText text={displayHeadline} numbers={false} /></h1>
      <p>{displayDetail}</p>
      {#if firefoxIncompatible}<p>{firefoxSwitchGuidance}</p>{/if}
    </div>

    <div class="status-region" role="status" aria-live="polite" aria-atomic="true">
      <MorphText text={snapshot.download.phase === "downloading"
        ? "Local setup download in progress."
        : snapshot.download.phase === "cancelling"
          ? "Pausing local setup download."
          : snapshot.download.phase === "paused"
            ? `Local setup download paused. ${snapshot.download.reason}`
          : snapshot.download.phase === "installed"
            ? "Local setup download complete and verified."
            : snapshot.runtime.phase === "loading"
              ? "Loading the model on this device."
              : snapshot.runtime.phase === "ready"
                ? "The local model is ready."
        : snapshot.phase === "checking"
        ? "Device check in progress."
        : snapshot.phase === "supported"
          ? snapshot.runtime.available &&
              (snapshot.download.available || snapshot.models.some((model) => model.setupComplete))
            ? "Check complete."
            : "Check complete. Review model compatibility below."
          : snapshot.phase === "unsupported"
            ? "Check failed. Review the items below."
            : ""} />
    </div>
    <div class="alert-region" role="alert" aria-live="assertive" aria-atomic="true">
      {snapshot.phase === "failed"
        ? snapshot.detail
        : snapshot.download.phase === "failed"
          ? `Local setup download failed. ${snapshot.download.reason}`
        : snapshot.runtime.phase === "failed"
          ? snapshot.runtime.reason
          : ""}
    </div>

    {#if workbenchError}
      <section
        class="workbench-error"
        role="alert"
        in:rise
        out:fadeOut
      >
        <div>
          <strong>The workbench could not open.</strong>
          <p>{workbenchError}</p>
          {#if workbenchNeedsReload}
            <p>{appRefreshSafetyMessage}</p>
            <p>{reloadInstructions()}</p>
          {/if}
        </div>
        <button
          type="button"
          disabled={workbenchReloading}
          onclick={onRetryWorkbench}
        ><MorphText text={workbenchReloading ? "Refreshing app…" : workbenchNeedsReload ? "Reload Drowse" : "Retry workbench"} /></button>
      </section>
    {/if}

    <section class="check-panel" data-status={checkStatus} data-page-group="1" aria-label="Device check" aria-busy={snapshot.phase === "checking"}>
      <h2 class="sr-only">Device checks</h2>
      <div class="device-details">
        <Expander label={snapshot.checks.length > 0 ? `${checkSummaryLabel} · ${checkCountLabel}` : checkSummaryLabel} bind:open={deviceDetailsOpen}>
        <ul class="check-list">
          {#each snapshot.checks as check}
            <li class:failed={check.state === "fail"} class:warning={check.state === "warn"}>
              <div>
                <h3>{check.label}</h3>
                <p>{check.detail}</p>
              </div>
              <span class="check-label">{checkLabel[check.state]}</span>
            </li>
          {/each}
        </ul>
        </Expander>
      </div>

      {#if snapshot.phase === "unsupported" || snapshot.phase === "failed" || downloadsLocked}
        <div class="recovery" in:rise out:fadeOut>
          {#if snapshot.phase === "unsupported" || snapshot.phase === "failed"}
            <p>{recoveryGuidance}</p>
          {:else if lockedDownloadGuidance}
            <p>{lockedDownloadGuidance}</p>
          {/if}
          <button type="button" onclick={() => controller.check()}>Check again</button>
        </div>
      {/if}
    </section>

    {#if snapshot.phase === "supported"}
      <section
        class="model-panel"
        data-page-group="2"
        aria-labelledby="models-title"
        in:rise
        out:fadeOut
      >
        <div
          class="panel-heading"
          class:choice-heading={snapshot.models.some((model) => model.fit !== "blocked")}
        >
            <h2 id="models-title" class:sr-only={snapshot.models.some((model) => model.fit !== "blocked")}>
              {snapshot.models.some((model) => model.fit !== "blocked")
                ? appleMobile
                  ? "Choose a compact model"
                  : "Choose a model"
                : "Current model requirements"}
            </h2>
          <div class="heading-side">
            <TipText label="About model choices" text="Smaller models use less memory and download faster. Larger models usually give better answers. The recommendation is an estimate of what this device can run." />
            <code>{formatAvailable(snapshot.storage?.availableBytes)}</code>
          </div>
        </div>

        {#snippet modelCard(model: HostedModelOption)}
            {@const shortfall = spaceNeeded(model)}
            <button
              type="button"
              class:selected={selectedModel === model.id}
              class:blocked={model.fit === "blocked"}
              disabled={modelOperationBusy}
              aria-disabled={model.fit === "blocked" && model.catalogAvailable !== false}
              aria-pressed={selectedModel === model.id}
              aria-labelledby={`model-name-${model.id}`}
              aria-describedby={[
                `model-topline-${model.id}`,
                `model-fit-${model.id}`,
                ...(model.installed ? [`model-installed-${model.id}`] : []),
                ...(model.modelType === "base" ? [`model-kind-${model.id}`] : []),
              ].join(" ")}
              onclick={() => {
                if (model.id !== requestedModelId) clearConversationOpen();
                selectedModel = model.id;
                unsafeConfirmation = null;
                confirmSessionReset = false;
                deleteConfirmation = null;
                discardConfirmation = null;
                deletionError = null;
                discardError = null;
                packSelectionError = null;
              }}
            >
              <span class="model-topline" id={`model-topline-${model.id}`}>
                <span>{tierLabel[model.tier]}</span>
                <code>{model.catalogAvailable === false ? "Chat model" : cardSize(model)}</code>
              </span>
              <span class="model-name">
                <strong><span id={`model-name-${model.id}`}>{model.name}</span>{#if model.modelType === "base"}<BaseModelTag plain /><span class="sr-only" id={`model-kind-${model.id}`}>Base model</span>{/if}</strong>
                {#if model.firstRunPacks.some((pack) => pack.kind === "sae")}
                  <span class="sae-available" {...{ "aria-description": "Sparse autoencoder available" }}>SAE</span>
                {/if}
              </span>
              {#if selectedModel === model.id}
                <span class="model-meta">{model.context} · {model.language}</span>
                {#if measuredSpeed(model)}
                  <span class="model-detail">{measuredSpeed(model)}</span>
                {:else if model.expectedSpeed}
                  <span class="model-detail" {...{ "aria-description": (`Measured estimate: ${model.expectedSpeed}. A token is a short piece of text.`) }}>
                    Expected response speed: {speedClass(model.expectedSpeed)}
                  </span>
                {/if}
                {#if model.estimatedDownloadSeconds}
                  <span class="model-detail">Estimated download: <MorphText text={formatEtaRange(model.estimatedDownloadSeconds)} /></span>
                {/if}
                {#if model.catalogAvailable !== false && model.remainingDownloadBytes > 0}
                <span class="model-detail">
                  Download: <MorphText text={downloadSizeSummary(model)} />
                </span>
                {/if}
                {#if model.firstRunPacks.some((pack) => pack.selected && !pack.requiredForSetup)}
                  <span class="model-detail">
                    Also selected: {model.firstRunPacks.filter((pack) => pack.selected && !pack.requiredForSetup).map(packName).join(", ")}
                  </span>
                {/if}
                {#if autoSelectedModel !== model.id}<span class="model-reason">{model.reason}</span>{/if}
              {/if}
              <span class="model-fit {shortfall > 0 ? "blocked" : model.fit}" id={`model-fit-${model.id}`}>{model.catalogAvailable === false
                ? "Download unavailable"
                : shortfall > 0
                  ? `Needs ${formatBytes(shortfall)} more space`
                  : model.requiresOomRetry ? "Previously ran out of memory" : fitLabel[model.fit]}</span>
              {#if model.installed}
                <span class="installed-mark" id={`model-installed-${model.id}`}>
                  {model.setupComplete ? "Installed and verified" : "Model downloaded · tools still needed"}
                </span>
              {/if}
            </button>
        {/snippet}

        <div class="model-grid">
          {#each modelsBySize(snapshot.models.filter((model) => model.modelType !== "base")) as model}
            {@render modelCard(model)}
          {/each}
        </div>

        <div class="base-models">
        <Expander label="Base models · Text completion" bind:open={baseModelsOpen}>
          <p class="base-model-warning">
            Base models predict what comes next in a piece of text. They don't follow instructions reliably
            and can produce unreliable or offensive text. Use a chat model for everyday conversations.
          </p>
          {#if snapshot.models.some((model) => model.modelType === "base")}
            <div class="model-grid">
              {#each modelsBySize(snapshot.models.filter((model) => model.modelType === "base")) as model}
                {@render modelCard(model)}
              {/each}
            </div>
          {:else}
            <p class="base-model-description">No verified base-model downloads are available in this catalog yet. Choose a chat model above to continue.</p>
          {/if}
        </Expander>
        </div>

        {#if comparableModels.length > 1}
          <div class="compare-models">
          <Expander label="Compare models">
            <!-- svelte-ignore a11y_no_noninteractive_tabindex (keyboard access for the horizontally scrolling table) -->
            <div class="compare-scroll" role="region" aria-label="Model comparison" tabindex="0">
              <table>
                <thead>
                  <tr>
                    <th scope="col">Model</th>
                    <th scope="col">Download</th>
                    <th scope="col">Speed</th>
                    <th scope="col">Conversation length</th>
                    <th scope="col">Languages</th>
                    <th scope="col">Tools</th>
                    <th scope="col">License</th>
                    <th scope="col">Fit on this device</th>
                  </tr>
                </thead>
                <tbody>
                  {#each comparableModels as model (model.id)}
                    <tr class:selected-row={selectedModel === model.id}>
                      <th scope="row">{model.name}{#if model.modelType === "base"} <span class="compare-note">Base</span>{/if}</th>
                      <td>{model.setupComplete ? "On this device" : formatBytes(model.remainingDownloadBytes)}</td>
                      <td>{measuredSpeed(model) ?? (model.expectedSpeed ? `${speedClass(model.expectedSpeed)} · ${model.expectedSpeed}` : "Measured after the first reply")}</td>
                      <td>{model.context}</td>
                      <td>{model.language}</td>
                      <td>{toolChips(model).join(", ") || "Response controls"}</td>
                      <td>{model.license}</td>
                      <td>{spaceNeeded(model) > 0 ? `Needs ${formatBytes(spaceNeeded(model))} more space` : fitLabel[model.fit]}<span class="compare-reason">{model.reason}</span></td>
                    </tr>
                  {/each}
                </tbody>
              </table>
            </div>
          </Expander>
          </div>
        {/if}

        {#if selectedOption}
          {#key selectedOption.id}
          <div class="selected-model-details" in:rise>
          {#if autoSelectedModel === selectedOption.id && selectedOption.fit !== "blocked"}
            <div class="why-this">
              <Expander label="Why this one?">
                <p>{selectedOption.fit === "recommended" ? `${selectedOption.name} is the recommended model for this device. ` : ""}{selectedOption.reason}</p>
              </Expander>
            </div>
          {/if}
          <p class="model-provenance">
            <a href={selectedOption.sourceUrl} target="_blank" rel="noreferrer">
              {selectedOption.name} source
            </a>
            <span aria-hidden="true">·</span>
            <span>{selectedOption.license}</span>
          </p>

          {#if selectedOption.fit !== "blocked"}
          <section class="tool-picker" aria-labelledby="tools-title">
            <div class="tool-picker-heading">
              <div>
                <h3 id="tools-title">{selectedOption.firstRunPacks.some((pack) => !pack.requiredForSetup) ? "Tools for" : "Included with"} {selectedOption.name}</h3>
              </div>
              <TipText
                label="About model tools"
                text={selectedOption.modelType === "base"
                  ? "This setup includes response controls and compatible SAE features when available. Word insights are optional for base models."
                  : "This setup includes response controls, word insights, and compatible SAE features when available. These packs are already trained."}
              />
            </div>
            <div class="tool-option required-tool">
              <span class="tool-check" aria-hidden="true"><Icon name="check" /></span>
              <div>
                <strong>Response controls</strong>
                <p>Measure and shape replies.</p>
              </div>
              <span>Included</span>
            </div>
            {#each selectedOption.firstRunPacks.filter((pack) => pack.requiredForSetup) as pack (pack.id)}
              <div class="tool-option required-tool" class:tool-installed={pack.installed}>
                <span class="tool-check" aria-hidden="true"><Icon name="check" /></span>
                <div>
                  <strong>{pack.kind === "jlens" ? "Word insights" : "Feature insights"} <span>({pack.kind === "jlens" ? "R lens" : "SAE"})</span></strong>
                  <p>{pack.kind === "jlens" ? `${pack.name}. Read word predictions across layers and steer toward a word.` : packDescription(pack)}</p>
                </div>
                <span><MorphText text={pack.installed ? "Included · already downloaded" : `Included · ${formatBytes(pack.bytes)}`} /></span>
              </div>
            {/each}
            {#if selectedOption.setupIssue}
              <p class="tool-error" role="alert">{selectedOption.setupIssue}</p>
            {:else}
              {#if selectedOption.toolNotice}
                <p class="tool-empty">{selectedOption.toolNotice}</p>
              {/if}
              {#if selectedOption.firstRunPacks.some((pack) => !pack.requiredForSetup)}
                {#each selectedOption.firstRunPacks.filter((pack) => !pack.requiredForSetup) as pack (pack.id)}
                  <label class="tool-option" class:tool-installed={pack.installed}>
                    <input
                      type="checkbox"
                      checked={pack.selected}
                      disabled={pack.installed || snapshot.download.phase === "requesting_persistence" || snapshot.download.phase === "downloading" || snapshot.download.phase === "cancelling"}
                      onchange={(event) => setPackSelected(pack.id, event.currentTarget.checked)}
                    />
                    <div>
                      <strong>{pack.kind === "jlens" ? selectedOption.modelType === "base" ? pack.name : "Alternative R lens" : packTitle(pack)} <span>({packTag(pack)})</span></strong>
                      <p>{packDescription(pack)}</p>
                    </div>
                    <span><MorphText text={pack.installed ? "Already downloaded" : formatBytes(pack.bytes)} /></span>
                  </label>
                {/each}
              {:else if !selectedOption.toolNotice}
                <p class="tool-empty">No other tools are available for this model and context size.</p>
              {/if}
            {/if}
            <div class="tool-total">
              <span>{selectedOption.installed ? "Download remaining" : "First download"}</span>
              <strong><MorphText text={formatBytes(selectedOption.remainingDownloadBytes) +
                (remainingDownloadParts(selectedOption).modelBytes > 0 && remainingDownloadParts(selectedOption).toolBytes > 0
                  ? ` · ${downloadSizeSummary(selectedOption)}`
                  : "")} /></strong>
            </div>
            {#if packSelectionError}
              <p class="tool-error" role="alert">{packSelectionError}</p>
            {/if}
          </section>
          {/if}
          </div>
          {/key}
        {/if}

        {#if selectedOption?.installed && !showDownloadAction}
          <div class="release-gate runtime-gate">
            <div>
              <strong><MorphText text={runtimeGateTitle} /></strong>
              <p>{snapshot.runtime.reason}</p>
              {#if storageUnprotected}
                {@render storageProtectionNotice()}
              {/if}
            </div>
            <button
              type="button"
              disabled={modelOperationBusy || !selectedOption.setupComplete || !snapshot.runtime.available || snapshot.runtime.resetSessionAvailable}
              onclick={() => {
                if (selectedOption.fit === "uncertain") unsafeConfirmation = selectedModel;
                else void openModel();
              }}
            >
              <MorphText text={!selectedOption.setupComplete ? "Add selected tools first" : snapshot.runtime.phase === "loading" ? "Loading…" : ["requesting_persistence", "downloading", "cancelling"].includes(selectedDownloadPhase) ? "Finishing setup…" : snapshot.runtime.resetSessionAvailable ? "Conversation needs attention" : selectedOption.fit === "uncertain" ? "Review warning" : runtimeGateAction} />
            </button>
          </div>
          {#if selectedOption.fit === "uncertain" && unsafeConfirmation === selectedModel}
            <div
              class="unsafe-warning"
              role="alert"
              in:rise
              out:fadeOut
            >
              <div>
                <strong>{selectedOption.requiresOomRetry ? "This model ran out of memory with these settings." : "Check this model on your device"}</strong>
                <p>{selectedOption.reason} The model may run out of memory or lose its graphics connection.</p>
              </div>
              <div class="warning-actions">
                <button class="safe-action" type="button" onclick={() => (unsafeConfirmation = null)}>Choose another model</button>
                <button class="warning-action" type="button" onclick={() => void openModel()}>{selectedOption.requiresOomRetry ? "Retry this model" : "Load anyway"}</button>
              </div>
            </div>
          {/if}
        {/if}

        {#if selectedOption?.installed && snapshot.runtime.resetSessionAvailable}
          <div
            class="unsafe-warning"
            role="alert"
            in:rise
            out:fadeOut
          >
            <div>
              <strong>The saved conversation cannot open with this runtime.</strong>
              <p>Starting over removes the conversation but keeps the model.</p>
            </div>
            <div class="warning-actions">
              {#if confirmSessionReset}
                <button class="safe-action" type="button" onclick={() => (confirmSessionReset = false)}>Keep conversation</button>
                <button class="danger-action" type="button" onclick={() => void openModel(true)}>Remove conversation and start over</button>
              {:else}
                <button class="safe-action" type="button" onclick={() => (confirmSessionReset = true)}>Review start-over option</button>
              {/if}
            </div>
          </div>
        {/if}

        {#if showDownloadAction}
          <div class="release-gate download-gate">
            <div>
              {#if selectedDownloadPhase !== "idle" && selectedDownloadPhase !== "locked"}
              <strong><MorphText text={selectedDownloadPhase === "installed" ? "Setup complete" : selectedDownloadPhase === "requesting_persistence" ? "Preparing download" : selectedDownloadPhase === "downloading" ? "Downloading model files" : selectedDownloadPhase === "cancelling" ? "Pausing download" : selectedDownloadPhase === "paused" ? "Download paused" : "Download interrupted"} /></strong>
              {#if snapshot.download.reason !== "Ready to download."}<p>{snapshot.download.reason}</p>{/if}
              {:else if keptBytes > 0}
              <strong>Download paused</strong>
              {/if}
              {#if keptBytes > 0 && !["requesting_persistence", "downloading", "cancelling"].includes(selectedDownloadPhase)}
                <p class="kept-copy">{formatBytes(keptBytes)} already downloaded and kept on this device. Resume to finish setup.</p>
              {/if}
              {#if otherDownloadActive}
                <p>Another model is downloading. Wait for it to finish or pause it first.</p>
              {/if}
              {#if storageUnprotected}
                {@render storageProtectionNotice()}
              {/if}
              {#if selectedProgress}
                <div
                  class="progress-track"
                  role="progressbar"
                  aria-label="Local setup download"
                  aria-valuemin="0"
                  aria-valuemax="100"
                  aria-valuenow={Math.round(progressPercent())}
                  aria-valuetext={`${Math.round(progressPercent())}% complete. ${etaLabel()}.`}
                  in:fadeIn
                  out:fadeOut
                >
                  <span style={`width: ${progressPercent()}%`}></span>
                </div>
                <p class="progress-copy">
                  <RollingNumber value={Math.round(progressPercent())} />% · <MorphText text={etaLabel()} />
                </p>
              {/if}
            </div>
            {#if selectedDownloadPhase === "downloading" || selectedDownloadPhase === "cancelling"}
              <button
                type="button"
                disabled={selectedDownloadPhase === "cancelling"}
                onclick={() => void controller.cancelDownload()}
              >
                <MorphText text={selectedDownloadPhase === "cancelling" ? "Pausing…" : "Pause download"} />
              </button>
            {:else}
              <button
                type="button"
                disabled={modelOperationBusy || !selectedModel || selectedOption?.setupComplete || selectedDownloadPhase === "installed"}
                onclick={() => void startDownload(false)}
              >
                <MorphText text={selectedOption?.setupComplete ? "Installed" : selectedDownloadPhase === "paused" || (keptBytes > 0 && selectedDownloadPhase !== "failed") ? "Resume download" : selectedDownloadPhase === "failed" ? "Retry download" : selectedOption?.fit === "uncertain" ? "Review warning" : selectedDownloadPhase === "requesting_persistence" ? "Preparing…" : selectedOption?.installed ? "Add tools and open" : "Download and open"} />
              </button>
            {/if}
            {#if keptBytes > 0 && controller.discardPartialDownload && ["idle", "paused", "failed"].includes(selectedDownloadPhase)}
              {#if discardConfirmation === selectedModel}
                <div class="delete-confirmation">
                  <p>Discard the {formatBytes(keptBytes)} already downloaded? The next download starts from the beginning.</p>
                  <div class="warning-actions">
                    <button type="button" class="secondary" disabled={discardingPartial} onclick={() => discardConfirmation = null}>Keep files</button>
                    <button type="button" class="delete-model" disabled={modelOperationBusy || discardingPartial} onclick={() => void discardPartialDownload()}><MorphText text={discardingPartial ? "Discarding…" : "Discard files"} /></button>
                  </div>
                </div>
              {:else}
                <button type="button" class="quiet-action" disabled={modelOperationBusy} onclick={() => { discardConfirmation = selectedModel; discardError = null; }}>Discard partial download</button>
              {/if}
              {#if discardError}<p role="alert" class="tool-error">{discardError}</p>{/if}
            {/if}
          </div>
          {#if selectedOption?.fit === "uncertain" && !selectedOption.installed && unsafeConfirmation === selectedModel}
            <div
              class="unsafe-warning"
              role="alert"
              in:rise
              out:fadeOut
            >
              <div>
                <strong>{selectedOption.requiresOomRetry ? "This model ran out of memory with these settings." : "Check this model on your device"}</strong>
                <p>{selectedOption.reason} The model may run out of memory or lose its graphics connection.</p>
              </div>
              <div class="warning-actions">
                <button class="safe-action" type="button" onclick={() => (unsafeConfirmation = null)}>Choose another model</button>
                <button class="warning-action" type="button" onclick={() => void startDownload(true)}>{selectedOption.requiresOomRetry ? "Retry download" : "Download anyway"}</button>
              </div>
            </div>
          {/if}
        {:else if (!snapshot.download.available || selectedOption?.fit === "blocked") && !selectedOption?.setupComplete}
          <div class="release-gate unavailable-gate">
            <div>
              <strong>{selectedOption?.fit === "blocked" ? "This model is unavailable" : "Downloads are unavailable"}</strong>
              <p>{downloadUnavailableReason(selectedOption, snapshot.download)}</p>
            </div>
            <button type="button" disabled>Download and open</button>
          </div>
        {/if}

        {#if installedModels.length > 0}
          <section class="model-storage" aria-labelledby="stored-title">
            <div class="stored-heading">
              <h3 id="stored-title">Stored on this device</h3>
              <p>Remove model files to free storage. Your chats and analysis packs are kept.</p>
            </div>
            <ul class="stored-list">
              {#each installedModels as model (model.id)}
                <li>
                  <div class="stored-row">
                    <div>
                      <strong>{model.name}</strong>
                      <p>
                        Model {formatBytes(model.modelDownloadBytes)}{#each model.firstRunPacks.filter((pack) => pack.installed) as pack (pack.id)} · {pack.kind === "jlens" ? "Word insights" : "Feature insights"} {formatBytes(pack.bytes)}{/each}
                      </p>
                    </div>
                    <span class="stored-size">{formatBytes(storedBytes(model))}</span>
                    {#if deleteConfirmation !== model.id}
                      <button type="button" class="secondary" disabled={modelOperationBusy} onclick={() => { deleteConfirmation = model.id; deletionNotice = null; deletionError = null; }}>Delete from device</button>
                    {/if}
                  </div>
                  {#if deleteConfirmation === model.id}
                    <div class="delete-confirmation">
                      <p>Delete {model.name} from this device? You will need to download it again to continue its chats.</p>
                      <div class="warning-actions">
                        <button type="button" class="secondary" disabled={deletingModel} onclick={() => deleteConfirmation = null}>Cancel</button>
                        <button type="button" class="delete-model" disabled={modelOperationBusy} onclick={() => void deleteSelectedModel()}><MorphText text={deletingModel ? "Deleting…" : "Delete model"} /></button>
                      </div>
                    </div>
                  {/if}
                </li>
              {/each}
            </ul>
            <div class="stored-total">
              <span>Total for these models</span>
              <strong>{formatBytes(installedModels.reduce((total, model) => total + storedBytes(model), 0))}</strong>
            </div>
            {#if snapshot.storage?.usageBytes != null && snapshot.storage?.quotaBytes != null}
              <p class="stored-quota">This browser is using {formatBytes(snapshot.storage.usageBytes)} of {formatBytes(snapshot.storage.quotaBytes)} it allows Drowse{snapshot.storage.availableBytes != null ? ` · ${formatBytes(snapshot.storage.availableBytes)} available` : ""}.</p>
            {:else if snapshot.storage?.availableBytes != null}
              <p class="stored-quota">{formatAvailable(snapshot.storage.availableBytes)}.</p>
            {/if}
            {#if deletionError}<p role="alert" class="tool-error">{deletionError}</p>{/if}
          </section>
        {/if}
        {#if deletionNotice}<p role="status">{deletionNotice}</p>{/if}
      </section>
    {/if}

  </main>

  <PageFooter />
</div>

<style>
  /* The models page in Ledger: paper, washes for each surface (wash2 for the chosen model), ink in
     three weights, no lines. The device check sits above the model list; the gates under the list
     are the one way forward. */
  .skip-link { position: fixed; z-index: 1000; top: var(--s2); inset-inline-start: var(--s2); padding: var(--s2) var(--s4); border-radius: var(--r-pill); background: var(--ink); color: var(--paper); font: var(--type-control); translate: 0 calc(-100% - var(--s4)); transition: translate var(--fast) var(--ease); }
  .skip-link:focus { translate: 0 0; }

  .app-shell { display: flex; flex-direction: column; min-height: 100dvh; }
  main { flex: 1; display: grid; grid-template-columns: minmax(0, 1fr); align-content: start; gap: var(--s7); width: 100%; max-width: var(--page); margin: 0 auto; padding: var(--s6) var(--gutter) var(--s8); }
  main:focus { outline: none; }

  .intro { display: grid; justify-items: start; gap: var(--s4); }
  h1 { color: var(--ink); font: var(--type-title); letter-spacing: -0.04em; }
  h1:focus { outline: none; }
  .intro p { max-width: 60ch; color: var(--ink2); font: var(--type-lead); }
  .intro > p + p { font: var(--type-body); }

  .status-region, .alert-region { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }

  /* A plain sentence and one way out. */
  .workbench-error, .recovery, .release-gate, .unsafe-warning, .persistence-notice, .delete-confirmation { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: var(--s3) var(--s5); padding: var(--s5); border-radius: var(--r-row); background: var(--wash); }
  .workbench-error > div, .release-gate > div, .unsafe-warning > div, .persistence-notice > div { display: grid; gap: var(--s2); flex: 1 1 320px; min-width: 0; }
  .workbench-error strong, .release-gate strong, .unsafe-warning strong, .persistence-notice strong { color: var(--ink); font: var(--type-label); }
  .workbench-error p, .recovery p, .release-gate p, .unsafe-warning p, .persistence-notice p, .delete-confirmation p { color: var(--ink2); font: var(--type-small); overflow-wrap: anywhere; }
  .persistence-notice { padding: var(--s4); background: var(--wash2); }
  .delete-confirmation { flex-basis: 100%; background: var(--wash2); }
  .warning-actions { display: flex; flex-wrap: wrap; gap: var(--s2); }

  /* Buttons: the gates' and the recovery's are filled; a choice that keeps things is quiet; a
     follow-up is text. A destructive one is confirmed first, then filled like any other. */
  button { display: inline-flex; align-items: center; justify-content: center; gap: var(--s2); height: 36px; padding: 0 var(--s5); border: 0; border-radius: var(--r-pill); background: var(--ink); color: var(--paper); font: var(--type-control); white-space: nowrap; cursor: pointer; transition: opacity var(--fast) var(--ease), background-color var(--fast) var(--ease), color var(--fast) var(--ease), scale var(--fast) var(--ease); }
  button:disabled { opacity: 0.4; cursor: default; }
  button:active:not(:disabled) { scale: 0.97; }
  .secondary, .safe-action { padding: 0 var(--s4); background: var(--paper); color: var(--ink); }
  .quiet-action { height: auto; padding: 0; background: transparent; color: var(--ink3); }
  @media (hover: hover) {
    button:hover:not(:disabled) { opacity: 0.86; }
    .secondary:hover:not(:disabled), .safe-action:hover:not(:disabled) { opacity: 1; background: var(--wash2); }
    .quiet-action:hover:not(:disabled) { opacity: 1; color: var(--ink); }
  }

  /* The device check. */
  .check-panel { display: grid; gap: var(--s4); }
  .device-details { padding: var(--s4) var(--s5); border-radius: var(--r-row); background: var(--wash); }
  .check-list { display: grid; gap: var(--s3); padding-top: var(--s4); }
  .check-list li { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: var(--s1) var(--s4); align-items: baseline; }
  .check-list h3 { color: var(--ink); font: var(--type-label); }
  .check-list p { grid-column: 1; color: var(--ink2); font: var(--type-small); overflow-wrap: anywhere; }
  .check-label { grid-column: 2; grid-row: 1; color: var(--ink3); font: var(--type-meta); }
  .check-list li.failed .check-label, .check-list li.warning .check-label { color: var(--ink); }

  /* The models. */
  .model-panel { display: grid; gap: var(--s5); }
  .panel-heading { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: var(--s2) var(--s4); }
  h2 { color: var(--ink); font: var(--type-heading); letter-spacing: -0.02em; }
  .heading-side { display: flex; flex-wrap: wrap; align-items: baseline; gap: var(--s4); color: var(--ink3); font: var(--type-meta); }
  .heading-side code { font: var(--type-meta); }
  .model-grid { display: grid; gap: var(--s2); }
  .model-grid > button {
    display: grid;
    grid-template-columns: minmax(120px, 180px) minmax(0, 1fr) minmax(120px, auto);
    gap: var(--s1) var(--s5);
    align-items: baseline;
    justify-content: stretch;
    height: auto;
    padding: var(--s4) var(--s5);
    border-radius: var(--r-row);
    background: var(--wash);
    color: var(--ink2);
    font: var(--type-small);
    white-space: normal;
    text-align: start;
  }
  .model-grid > button.selected { background: var(--wash2); }
  .model-grid > button[aria-disabled="true"] { opacity: 0.55; }
  .model-grid > button:active:not(:disabled) { scale: 0.995; }
  @media (hover: hover) { .model-grid > button:hover:not(:disabled) { opacity: 1; background: var(--wash2); } }
  .model-topline { display: grid; gap: 2px; color: var(--ink3); font: var(--type-meta); }
  .model-topline code { font: var(--type-meta); }
  .model-name { display: flex; flex-wrap: wrap; align-items: baseline; gap: var(--s2); min-width: 0; }
  .model-name strong { display: inline-flex; flex-wrap: wrap; align-items: baseline; gap: var(--s2); color: var(--ink); font: var(--type-label); overflow-wrap: anywhere; }
  .sae-available { padding: 0 var(--s2); border-radius: var(--r-chip); background: var(--paper); color: var(--ink2); font: var(--type-meta); }
  .model-meta, .model-detail, .model-reason { grid-column: 2; color: var(--ink2); font: var(--type-small); }
  .model-meta { color: var(--ink3); }
  .model-fit { grid-column: 3; grid-row: 1; justify-self: end; color: var(--ink3); font: var(--type-meta); text-align: end; }
  .model-fit.recommended, .model-fit.blocked { color: var(--ink); }
  .installed-mark { grid-column: 3; justify-self: end; color: var(--ink3); font: var(--type-meta); text-align: end; }

  .base-models, .compare-models, .why-this { display: grid; gap: var(--s3); }
  .base-models :global(.inner) { display: grid; gap: var(--s4); padding-top: var(--s3); }
  .base-model-warning, .base-model-description, .why-this p { max-width: 64ch; color: var(--ink2); font: var(--type-small); }
  .compare-scroll { overflow-x: auto; margin-top: var(--s3); border-radius: var(--r-box); background: var(--wash); }
  .compare-scroll:focus-visible { outline-offset: 2px; }
  .compare-scroll table { width: 100%; min-width: 760px; border-collapse: collapse; font: var(--type-small); }
  .compare-scroll th, .compare-scroll td { padding: var(--s3) var(--s4); text-align: start; vertical-align: top; }
  .compare-scroll thead th { color: var(--ink3); font: var(--type-head); }
  .compare-scroll tbody th { color: var(--ink); font-weight: 500; }
  .compare-scroll td { color: var(--ink2); }
  .compare-scroll .selected-row > * { background: var(--wash2); }
  .compare-note, .compare-reason { color: var(--ink3); font: var(--type-meta); }
  .compare-reason { display: block; margin-top: 2px; }

  .selected-model-details { display: grid; gap: var(--s4); }
  .model-provenance { display: flex; flex-wrap: wrap; gap: var(--s2); color: var(--ink3); font: var(--type-small); }
  .model-provenance a { color: var(--ink2); text-decoration: underline; text-decoration-color: var(--wash2); text-decoration-thickness: 2px; text-underline-offset: 4px; transition: color var(--fast) var(--ease); }
  @media (hover: hover) { .model-provenance a:hover { color: var(--ink); } }

  .tool-picker { display: grid; gap: var(--s2); }
  .tool-picker-heading { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: var(--s2) var(--s4); margin-bottom: var(--s2); }
  .tool-picker h3 { color: var(--ink); font: var(--type-label); }
  .tool-option { display: grid; grid-template-columns: 24px minmax(0, 1fr) auto; gap: var(--s1) var(--s3); align-items: start; padding: var(--s3) var(--s4); border-radius: var(--r-box); background: var(--wash); color: var(--ink2); font: var(--type-small); }
  label.tool-option { cursor: pointer; transition: background-color var(--fast) var(--ease); }
  @media (hover: hover) { label.tool-option:hover { background: var(--wash2); } }
  label.tool-option:has(input:focus-visible) { outline: 2px solid var(--ink); outline-offset: 2px; }
  .tool-option input { width: 18px; height: 18px; margin: 2px 0 0; accent-color: var(--ink); }
  .tool-check { display: grid; place-items: center; height: 20px; color: var(--ink); }
  .tool-option strong { color: var(--ink); font: var(--type-control); }
  .tool-option strong span { color: var(--ink3); font-weight: 400; }
  .tool-option > span:last-child { color: var(--ink3); font: var(--type-meta); text-align: end; }
  .tool-empty, .tool-error { color: var(--ink2); font: var(--type-small); }
  .tool-error { color: var(--ink); }
  .tool-total { display: flex; flex-wrap: wrap; justify-content: space-between; gap: var(--s2); padding: var(--s2) var(--s4) 0; color: var(--ink3); font: var(--type-small); }
  .tool-total strong { color: var(--ink); font-weight: 500; }

  /* Download progress: a wash track with an ink bar, numbers beside it. */
  .progress-track { position: relative; width: 100%; height: 4px; margin-top: var(--s2); border-radius: 2px; background: var(--wash2); overflow: hidden; }
  .progress-track span { position: absolute; inset: 0 auto 0 0; border-radius: 2px; background: var(--ink); transition: width var(--glide) var(--ease); }
  .progress-copy { color: var(--ink3); font: var(--type-meta); }
  .kept-copy { color: var(--ink2); }

  /* Models stored on this device. */
  .model-storage { display: grid; gap: var(--s4); }
  .model-storage h3 { color: var(--ink); font: var(--type-label); }
  .model-storage p { color: var(--ink2); font: var(--type-small); overflow-wrap: anywhere; }
  .stored-list { display: grid; gap: var(--s2); }
  .stored-list > li { display: grid; gap: var(--s3); padding: var(--s4) var(--s5); border-radius: var(--r-row); background: var(--wash); }
  .stored-row { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s2) var(--s5); }
  .stored-row > div { flex: 1 1 240px; min-width: 0; }
  .stored-row strong { color: var(--ink); font: var(--type-label); }
  .stored-size { color: var(--ink3); font: var(--type-meta); }
  .stored-total { display: flex; flex-wrap: wrap; justify-content: space-between; gap: var(--s2); padding: 0 var(--s5); color: var(--ink3); font: var(--type-small); }
  .stored-total strong { color: var(--ink); font-weight: 500; }
  .stored-quota { padding: 0 var(--s5); color: var(--ink3); font: var(--type-meta); }

  @media (max-width: 720px) {
    main { gap: var(--s6); padding-top: var(--s4); }
    button { height: 44px; }
    .quiet-action { height: auto; min-height: 44px; }
    .model-grid > button { grid-template-columns: minmax(0, 1fr) auto; padding: var(--s4); }
    .model-topline { grid-column: 1; grid-row: 1; }
    .model-name { grid-column: 1 / -1; }
    .model-meta, .model-detail, .model-reason { grid-column: 1 / -1; }
    .model-fit { grid-column: 2; grid-row: 1; }
    .installed-mark { grid-column: 1 / -1; justify-self: start; text-align: start; }
    .workbench-error, .recovery, .release-gate, .unsafe-warning, .persistence-notice { padding: var(--s4); }
    .release-gate > button, .warning-actions > button { flex: 1 1 auto; }
  }
</style>
