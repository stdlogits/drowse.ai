<script lang="ts">
  import { Expander, IconButton, MorphText } from "@stdlogits/ledger";
  import RollingNumber from "../../../drowse/webui/src/lib/ui/RollingNumber.svelte";
  import ResetSettings from "../../../drowse/webui/src/lib/ui/ResetSettings.svelte";
  import { onMount } from "svelte";
  import {
    getHostedController,
    getRuntimeCapabilities,
  } from "../../../drowse/webui/src/lib/runtime/registry";
  import { initialRuntimeSnapshot } from "../lib/runtime/state";
  import { userFacingError } from "../../../drowse/webui/src/lib/runtime/userFacingError";
  import { formatEtaRange } from "../lib/runtime/eta";
  import { hostedNetworkIsOffline } from "../hosted/runtime/networkState";
  import { optionalPackHardwareBlock } from "../hosted/runtime/optionalPackCompatibility";
  import {
    instrumentPackDependencies,
    type OptionalInstrumentPackKind,
  } from "../lib/runtime/instrumentPackSafety";
  import type {
    CatalogInstrumentPack,
    DownloadProgress,
    ModelVariant,
    RuntimeSnapshot,
    VerifiedCatalog,
  } from "../../../drowse/webui/src/lib/runtime/contracts";
  import {
    closeDrawer,
    currentSteeringExpression,
    probeRack,
    steerRack,
  } from "../../../drowse/webui/src/lib/stores.svelte";

  let { params = null }: { params?: unknown } = $props();
  const embedded = $derived(
    (params as { embedded?: boolean } | null)?.embedded === true,
  );

  const controller = getHostedController();
  const capabilities = getRuntimeCapabilities();
  let snapshot = $state<RuntimeSnapshot>(controller?.snapshot ?? initialRuntimeSnapshot());
  let storage = $state(capabilities?.storage ?? null);
  let busy = $state<string | null>(null);
  let error = $state<string | null>(
    controller ? null : "The hosted runtime controller is unavailable.",
  );
  let confirmClear = $state(false);
  let confirmDelete = $state<string | null>(null);
  let confirmPackDelete = $state<string | null>(null);
  let verifiedCatalog = $state<VerifiedCatalog | null>(null);
  let catalogLoading = $state(false);
  let catalogError = $state<string | null>(null);
  let packProgress = $state<DownloadProgress | null>(null);
  let cancellingPack = $state(false);
  let persistenceWarning = $state<string | null>(null);
  let notice = $state<string | null>(null);
  let filesOpen = $state(false);
  let dataOpen = $state(false);
  let diagnosticsOpen = $state(false);
  const MODEL_CHOICE_CLEANUP_TIMEOUT_MS = 10_000;

  const selectedVariantId = $derived(
    snapshot.modelVariantId ?? snapshot.selectedModelVariantId,
  );
  const selectedVariant = $derived.by<ModelVariant | null>(() => {
    if (!verifiedCatalog || !selectedVariantId) return null;
    return verifiedCatalog.document.models
      .flatMap((model) => model.variants)
      .find((variant) => variant.id === selectedVariantId) ?? null;
  });
  const modelTools = $derived(
    selectedVariant?.packs.filter((pack) => pack.kind !== "core") ?? [],
  );
  const requiredJlens = $derived(
    modelTools.find((pack) => pack.kind === "jlens") ?? null,
  );
  const selectedModelName = $derived.by(() => {
    if (!verifiedCatalog || !selectedVariantId) return selectedVariantId;
    const model = verifiedCatalog.document.models.find((candidate) =>
      candidate.variants.some((variant) => variant.id === selectedVariantId)
    );
    return model?.displayName ?? selectedVariantId;
  });
  const catalogPacks = $derived(
    verifiedCatalog?.document.models.flatMap((model) =>
      model.variants.flatMap((variant) => variant.packs)
    ) ?? [],
  );
  const requiredJlensIds = $derived.by(() => new Set(
    verifiedCatalog?.document.models.flatMap((model) =>
      model.variants.flatMap((variant) => {
        const pack = variant.packs.find((candidate) => candidate.kind === "jlens");
        return pack ? [pack.id] : [];
      })
    ) ?? [],
  ));
  const deletablePackIds = $derived(
    snapshot.installedPackIds.filter((packId) => !requiredJlensIds.has(packId)),
  );
  const lifecycleLabel = $derived(({
    uninitialized: "Not started",
    checking: "Checking this device",
    unloaded: "Saved on this device",
    loading: "Opening model",
    ready: "Ready",
    failed: "Needs attention",
  } as Record<string, string>)[snapshot.lifecycle] ?? snapshot.lifecycle);

  onMount(() => {
    const unsubscribe = controller?.subscribe((next) => (snapshot = next));
    if (controller) void loadCatalog();
    return unsubscribe;
  });

  const describe = (value: unknown) => userFacingError(value);

  const formatBytes = (bytes: number | null | undefined) => {
    if (bytes == null) return "unavailable";
    if (bytes >= 1_000_000_000) return `${(bytes / 1_000_000_000).toFixed(1)} GB`;
    return `${Math.round(bytes / 1_000_000)} MB`;
  };

  const formatPercent = (progress: DownloadProgress) =>
    progress.bytesTotal === 0
      ? 0
      : Math.min(100, Math.round(progress.bytesReceived / progress.bytesTotal * 100));

  const formatEta = (progress: DownloadProgress) => {
    if (progress.stalled) return "stalled";
    if (progress.calculatingEta || !progress.etaSeconds) return "calculating time remaining…";
    if (progress.etaSeconds[1] <= 0) return "verifying…";
    return `${formatEtaRange(progress.etaSeconds)} remaining`;
  };

  function packUnavailableReason(pack: CatalogInstrumentPack): string | null {
    if (!selectedVariant) return "Choose this model before adding a tool.";
    if (!snapshot.installedModelVariantIds.includes(selectedVariant.id)) {
      return "Download this model before adding a tool.";
    }
    const activeContext = snapshot.modelVariantId === selectedVariant.id &&
        snapshot.contextTokens !== null
      ? selectedVariant.contextProfiles.find(
          (profile) => profile.contextTokens === snapshot.contextTokens,
        )
      : null;
    if (
      snapshot.modelVariantId === selectedVariant.id &&
      (activeContext === undefined || activeContext === null ||
        !pack.compatibleContextBindingSha256.includes(activeContext.bindingSha256))
    ) return "This tool does not support the active conversation length.";
    const compatibleBindings = activeContext
      ? [activeContext.bindingSha256]
      : pack.compatibleContextBindingSha256;
    const concurrent = selectedVariant.packs.filter((candidate) =>
      candidate.id !== pack.id && !candidate.required && candidate.kind !== "core" &&
      snapshot.installedPackIds.includes(candidate.id) &&
      candidate.compatibleContextBindingSha256.some((binding) =>
        compatibleBindings.includes(binding)
      )
    );
    // Lens packs are alternative readout sources. Keep the standard J-lens
    // installed while downloading an R-lens so the user can switch between
    // them without replacing either pack. SAEs remain one-per-model because
    // the runtime can hold only one active feature dictionary.
    const sameKind = pack.kind === "sae"
      ? concurrent.find((candidate) => candidate.kind === pack.kind)
      : undefined;
    if (sameKind) {
      return `Remove ${sameKind.displayName} before installing another feature pack.`;
    }
    if (!capabilities) return "Run the device check before adding a tool.";
    return optionalPackHardwareBlock(
      pack,
      selectedVariant,
      capabilities,
      concurrent,
    )?.message ?? null;
  }

  const packSourceUrl = (pack: CatalogInstrumentPack) =>
    `https://huggingface.co/${pack.sourceRepository}/tree/${pack.sourceRevision}`;

  async function run(label: string, operation: () => Promise<void>): Promise<void> {
    busy = label;
    error = null;
    try {
      await operation();
    } catch (cause) {
      error = describe(cause);
    } finally {
      busy = null;
    }
  }

  async function refreshStorage(): Promise<void> {
    if (!controller) return;
    await run("storage", async () => {
      storage = await controller.refreshStorage();
    });
  }

  async function protectStorage(): Promise<void> {
    if (!controller) return;
    await run("persistence", async () => {
      const requested = await controller.requestPersistence();
      storage = await controller.refreshStorage();
      if (requested || storage.persisted === true) {
        persistenceWarning = null;
        notice = "Downloaded files are protected from automatic browser cleanup.";
      } else {
        persistenceWarning = "Protection is still off. Open Storage and downloads, then select Protect storage to try again. If your browser declines, avoid clearing data for this site and keep enough free device storage.";
      }
    });
  }

  async function loadCatalog(): Promise<void> {
    if (!controller) return;
    catalogLoading = true;
    catalogError = null;
    try {
      verifiedCatalog = await controller.catalog({
        offline: hostedNetworkIsOffline(),
        preferCached: true,
      });
    } catch (cause) {
      catalogError = describe(cause);
    } finally {
      catalogLoading = false;
    }
  }

  async function installPack(pack: CatalogInstrumentPack): Promise<void> {
    if (!controller || !selectedVariant) return;
    packProgress = null;
    persistenceWarning = null;
    notice = null;
    await run(`install-pack:${pack.id}`, async () => {
      if (!(await controller.requestPersistence())) {
        persistenceWarning = "Protection is still off. Open Storage and downloads, then select Protect storage to try again. If your browser declines, avoid clearing data for this site and keep enough free device storage.";
      }
      const result = await controller.downloadPack(
        selectedVariant.id,
        pack.id,
        (progress) => (packProgress = progress),
      );
      notice = result.cancelled
        ? `${pack.displayName} download paused. Its verified contiguous prefix is kept for resume.`
        : snapshot.modelVariantId === selectedVariant.id
          ? `${pack.displayName} is installed. Unload and reopen this model to activate it; availability still depends on its exact binding and local capability checks.`
          : `${pack.displayName} is installed. It will be checked and loaded the next time this model opens.`;
    });
    packProgress = null;
  }

  async function cancelPackDownload(): Promise<void> {
    if (!controller || cancellingPack) return;
    cancellingPack = true;
    error = null;
    try {
      await controller.cancelDownload();
    } catch (cause) {
      error = describe(cause);
    } finally {
      cancellingPack = false;
    }
  }

  async function unloadModel(): Promise<void> {
    if (!controller) return;
    const url = new URL(window.location.href);
    url.searchParams.delete("reopen");
    url.searchParams.set("choose", "1");
    window.history.replaceState(window.history.state, "", url);
    await run("unload", () => controller.unload());
  }

  async function chooseModel(): Promise<void> {
    if (!controller) return;
    await run("switch", async () => {
      const url = new URL("/app", window.location.href);
      if (new URL(window.location.href).searchParams.get("fixture") === "1") {
        url.searchParams.set("fixture", "1");
      }
      url.searchParams.set("choose", "1");
      const currentUrl = new URL(window.location.href);
      currentUrl.searchParams.delete("reopen");
      currentUrl.searchParams.set("choose", "1");
      window.history.replaceState(window.history.state, "", currentUrl);
      if (snapshot.modelVariantId !== null) {
        let timer: number | undefined;
        const timeout = new Promise<void>((resolve) => {
          timer = window.setTimeout(resolve, MODEL_CHOICE_CLEANUP_TIMEOUT_MS);
        });
        await Promise.race([
          controller.unload().catch(() => undefined),
          timeout,
        ]);
        if (timer !== undefined) window.clearTimeout(timer);
      }
      window.location.replace(url.href);
    });
  }

  async function deleteModel(modelVariantId: string): Promise<void> {
    if (!controller) return;
    await run(`delete:${modelVariantId}`, async () => {
      if (snapshot.modelVariantId === modelVariantId) {
        const url = new URL(window.location.href);
        url.searchParams.delete("reopen");
        url.searchParams.set("choose", "1");
        window.history.replaceState(window.history.state, "", url);
      }
      await controller.deleteModel(modelVariantId);
      confirmDelete = null;
    });
  }

  async function deletePack(packId: string): Promise<void> {
    if (!controller) return;
    if (requiredJlensIds.has(packId)) {
      error = "Word insights are included with the model and cannot be removed separately. Delete the model to remove all of its required files.";
      return;
    }
    const pack = catalogPacks.find((candidate) => candidate.id === packId);
    if (pack?.kind !== "core") {
      const kind: OptionalInstrumentPackKind | null =
        pack?.kind === "jlens" || pack?.kind === "sae" ? pack.kind : null;
      const references = instrumentPackDependencies({
        kind,
        steeringExpression: currentSteeringExpression(),
        steeringEntries: steerRack.entries,
        probes: [...probeRack.entries.values()].map((entry) => entry.info),
      });
      if (references.length > 0) {
        error = `Remove this tool from response shaping and watched readings before deleting it (${references.join("; ")}).`;
        return;
      }
    }
    const closedActiveModel = snapshot.modelVariantId !== null;
    await run(`delete-pack:${packId}`, async () => {
      if (closedActiveModel) {
        const url = new URL(window.location.href);
        url.searchParams.set("reopen", "1");
        url.searchParams.set("reopen-after-unload", "1");
        window.history.replaceState(window.history.state, "", url);
      }
      await controller.deletePack(packId);
      confirmPackDelete = null;
      notice = closedActiveModel
        ? "The model tool was deleted and the active model was closed. Reopen the model to continue."
        : "The model tool was deleted from this device.";
    });
  }

  async function clearAll(): Promise<void> {
    if (!controller) return;
    await run("clear", async () => {
      const url = new URL(window.location.href);
      url.searchParams.delete("reopen");
      url.searchParams.set("choose", "1");
      window.history.replaceState(window.history.state, "", url);
      await controller.clearAll();
    });
  }
</script>

<section
  class="drawer-shell"
  class:embedded
  role={embedded ? "region" : undefined}
  aria-label={embedded ? "Model controls" : "Model settings drawer"}
  data-model-variant-id={snapshot.modelVariantId ?? ""}
>
  {#if !embedded}
    <header class="header">
      <h2 class="title">Model settings</h2>
      <IconButton icon="close" label="Close" onclick={closeDrawer} />
    </header>
  {/if}

  <div class="body" aria-busy={busy !== null}>
    {#if error}<p class="error" role="alert">{error}</p>{/if}

    <section class="panel model-card">
      <div class="model-heading">
        <div class="model-copy">
          <h3>{selectedModelName ?? "No model open"}</h3>
          <p>
            {lifecycleLabel}<MorphText text={snapshot.contextTokens
              ? ` · ${snapshot.contextTokens.toLocaleString()}-token conversation memory`
              : ""} />
          </p>
        </div>
        <span class="status" class:ready={snapshot.lifecycle === "ready"}>{lifecycleLabel}</span>
      </div>
      <div class="actions">
        <button class="primary" aria-busy={busy === "switch"} type="button" disabled={busy !== null} onclick={() => void chooseModel()}>
          <MorphText text={busy === "switch"
            ? "Closing model…"
            : snapshot.modelVariantId === null
              ? "Choose a model"
              : "Change model"} />
        </button>
        <button
          type="button"
          disabled={busy !== null || snapshot.modelVariantId === null}
         
          aria-busy={busy === "unload"}
          onclick={() => void unloadModel()}
        ><MorphText text={busy === "unload" ? "Closing…" : "Close model"} /></button>
      </div>
      {#if busy === "switch"}
        <p class="action-status" role="status">Saving this conversation and releasing graphics memory. Model choices will open automatically.</p>
      {/if}
    </section>

    <section class="panel"><ResetSettings full /></section>

    <section class="panel tools-card">
      <div class="section-head">
        <div>
          <h3>Included files and additions</h3>
        </div>
        <button class="quiet" aria-busy={catalogLoading} type="button" disabled={busy !== null || catalogLoading} onclick={() => void loadCatalog()}>
          <MorphText text={catalogLoading ? "Checking…" : "Check again"} />
        </button>
      </div>
      {#if persistenceWarning}<p class="warning" role="status">{persistenceWarning}</p>{/if}
      {#if notice}<p class="notice" role="status">{notice}</p>{/if}
      {#if verifiedCatalog && !verifiedCatalog.allowDownloads}
        <p class="warning" role="status">You are offline. Downloaded tools still work.</p>
      {/if}
      {#if catalogError}
        <p class="error" role="alert">Unable to check model tools: {catalogError}</p>
      {:else if catalogLoading && !verifiedCatalog}
        <p class="empty">Checking model tools…</p>
      {:else if selectedVariantId === null}
        <p class="empty">Choose a model to see its tools.</p>
      {:else if selectedVariant === null}
        <p class="empty">Tool information is not available for this model.</p>
      {:else if modelTools.length === 0}
        <p class="empty">This model has no separate reading tools.</p>
      {:else}
        <ul class="pack-list">
          {#each modelTools as pack}
            {@const installed = snapshot.installedPackIds.includes(pack.id)}
            {@const unavailableReason = packUnavailableReason(pack)}
            {@const downloading = busy === `install-pack:${pack.id}`}
            <li data-pack-id={pack.id}>
              <div class="pack-details">
                <strong>{pack.id === requiredJlens?.id ? "Word insights · included" : pack.kind === "jlens" ? "Alternative R lens" : "Feature explorer"}</strong>
                <span>{pack.displayName} · {formatBytes(pack.bytes)}</span>
              </div>
              {#if pack.id === requiredJlens?.id}
                <span class:installed={installed}>{installed ? "Ready" : "Required file missing"}</span>
              {:else if downloading}
                <div class="pack-download" aria-live="polite">
                  {#if packProgress}
                    <progress
                      value={packProgress.bytesReceived}
                      max={packProgress.bytesTotal}
                      aria-label={`${pack.displayName} download progress`}
                    ></progress>
                    <span><RollingNumber value={formatPercent(packProgress)} />% · <MorphText text={formatEta(packProgress)} /></span>
                  {:else}
                    <span>Preparing verified download…</span>
                  {/if}
                  <button
                    type="button"
                    disabled={cancellingPack}
                    onclick={() => void cancelPackDownload()}
                  ><MorphText text={cancellingPack ? "Cancelling…" : "Pause download"} /></button>
                </div>
              {:else if installed}
                <span class="installed">Ready</span>
              {:else}
                <button
                  type="button"
                  disabled={busy !== null || unavailableReason !== null || !verifiedCatalog?.allowDownloads}
                  {...{ "aria-description": (!verifiedCatalog?.allowDownloads
                    ? "Reconnect to download this tool"
                    : unavailableReason ?? `Install ${pack.displayName}`) }}
                  onclick={() => void installPack(pack)}
                >Download</button>
              {/if}
            </li>
          {/each}
        </ul>
      {/if}
    </section>

    <Expander bind:open={filesOpen} label="Storage and downloads">
      <div class="disclosure-stack">
        <section class="subsection">
          <div class="section-head">
            <div>
              <h3>Space on this device</h3>
              <p>{storage?.persisted ? "Downloaded files are protected from automatic browser cleanup." : "Protection from automatic browser cleanup is off. Select Protect storage to ask your browser to keep these files. If it declines, avoid clearing data for this site and keep enough free device storage."}</p>
            </div>
            <button
              class="quiet"
              type="button"
              disabled={busy !== null}
              onclick={() => void (storage?.persisted ? refreshStorage() : protectStorage())}
            >{busy === "persistence" ? "Checking…" : storage?.persisted ? "Refresh" : "Protect storage"}</button>
          </div>
          <dl class="facts compact">
            <div><dt>Used</dt><dd>{formatBytes(storage?.usageBytes)}</dd></div>
            <div><dt>Available</dt><dd>{formatBytes(storage?.availableBytes)}</dd></div>
          </dl>
        </section>

        <section class="subsection" aria-label="Downloaded models">
          <h3>Downloaded models</h3>
          {#if snapshot.installedModelVariantIds.length === 0}
            <p class="empty">No model is downloaded.</p>
          {:else}
            <ul class="models">
              {#each snapshot.installedModelVariantIds as modelVariantId}
                <li>
                  <code>{modelVariantId}</code>
                  {#if confirmDelete === modelVariantId}
                    <span>Delete this model and its conversation?</span>
                    <button type="button" class="danger" disabled={busy !== null} onclick={() => void deleteModel(modelVariantId)}>Delete model</button>
                    <button type="button" disabled={busy !== null} onclick={() => (confirmDelete = null)}>Cancel</button>
                  {:else}
                    <button type="button" disabled={busy !== null} onclick={() => (confirmDelete = modelVariantId)}>Delete</button>
                  {/if}
                </li>
              {/each}
            </ul>
          {/if}
        </section>

        <section class="subsection">
          <h3>Downloaded additions</h3>
          {#if deletablePackIds.length === 0}
            <p class="empty">No additional feature or R lens packs are downloaded.</p>
          {:else}
            <ul class="models">
              {#each deletablePackIds as packId}
                {@const catalogPack = catalogPacks.find((pack) => pack.id === packId)}
                <li data-pack-id={packId}>
                  <span class="installed-pack-details">
                    <strong>{catalogPack?.kind === "jlens" ? "Alternative R lens" : catalogPack?.kind === "sae" ? "Feature explorer" : "Response controls"}</strong>
                    <code>{catalogPack?.displayName ?? packId}</code>
                  </span>
                  {#if confirmPackDelete === packId}
                    <span>Delete this tool from this device?</span>
                    <button type="button" class="danger" disabled={busy !== null} onclick={() => void deletePack(packId)}>Delete tool</button>
                    <button type="button" disabled={busy !== null} onclick={() => (confirmPackDelete = null)}>Cancel</button>
                  {:else}
                    <button type="button" disabled={busy !== null} onclick={() => (confirmPackDelete = packId)}>Delete</button>
                  {/if}
                </li>
              {/each}
            </ul>
          {/if}
        </section>
      </div>
    </Expander>

    <Expander bind:open={dataOpen} label="Reset and technical details">
      <div class="disclosure-stack">
        <section class="subsection danger-zone">
          <h3>Clear local data</h3>
          <p>Remove every Drowse model, tool, conversation, response control, and preference stored by this site.</p>
          {#if confirmClear}
            <div class="actions">
              <button type="button" class="danger" disabled={busy !== null} onclick={() => void clearAll()}><MorphText text={busy === "clear" ? "Clearing…" : "Clear all local data"} /></button>
              <button type="button" disabled={busy !== null} onclick={() => (confirmClear = false)}>Cancel</button>
            </div>
          {:else}
            <button type="button" disabled={busy !== null} onclick={() => (confirmClear = true)}>Review local data</button>
          {/if}
        </section>

        <Expander bind:open={diagnosticsOpen} label="Technical details">
          <dl class="facts diagnostics">
            <div><dt>Model ID</dt><dd><code>{snapshot.modelVariantId ?? "None"}</code></dd></div>
            <div><dt>Runtime version</dt><dd>{capabilities?.runtimeVersion ?? "Unknown"}</dd></div>
            <div><dt>Graphics adapter</dt><dd>{capabilities?.webGpu.fallback ?? "Unknown"}</dd></div>
            <div><dt>Speed check</dt><dd>{capabilities?.signals.calibrationScore?.toFixed(2) ?? "Unavailable"}</dd></div>
            <div><dt>Device signature</dt><dd><code>{capabilities?.deviceSignature ?? "Unknown"}</code></dd></div>
          </dl>
          {#if modelTools.length > 0}
            <div class="sources">
              {#each modelTools as pack}
                <a href={packSourceUrl(pack)} target="_blank" rel="noreferrer">{pack.displayName} source</a>
              {/each}
            </div>
          {/if}
        </Expander>
      </div>
    </Expander>

    <footer class="product-footer" aria-label="About Drowse">
      <div class="product-identity"><span class="wordmark" translate="no">Drowse</span><span class="version">v 0.1</span></div>
      <nav aria-label="Project links">
        <span>GNU AGPL v3 or later</span>
        <span aria-hidden="true">-</span>
        <a href="https://drowse.ai/contribute" target="_blank" rel="noreferrer">Contribute</a>
        <span aria-hidden="true">-</span>
        <a href="/credits" target="_blank" rel="noreferrer">Credits</a>
      </nav>
      <p>© 2026 Drowse Contributors</p>
    </footer>
  </div>
</section>

<style>
  /* The browser runtime's model settings (the Model room's Runtime section, or a dialog): the open
     model, its tools, then storage and the reset in folds. Washes for each part, no lines. */
  .drawer-shell { display: flex; flex-direction: column; min-height: 0; color: var(--ink2); }
  .header { display: flex; align-items: center; justify-content: space-between; gap: var(--s4); padding: var(--s5) var(--s5) 0; }
  .title { color: var(--ink); font: var(--type-mark); }
  .body { display: grid; gap: var(--s4); padding: var(--s5); overflow-y: auto; }
  .embedded .body { padding: 0; overflow: visible; }

  .panel { display: grid; gap: var(--s3); padding: var(--s5); border-radius: var(--r-row); background: var(--wash); }
  .model-heading, .section-head { display: flex; flex-wrap: wrap; align-items: start; justify-content: space-between; gap: var(--s2) var(--s4); }
  .model-copy, .section-head > div { display: grid; gap: var(--s1); flex: 1 1 240px; min-width: 0; }
  h3 { color: var(--ink); font: var(--type-label); overflow-wrap: anywhere; }
  p { font: var(--type-small); overflow-wrap: anywhere; }
  .status { padding: 2px var(--s2); border-radius: var(--r-chip); background: var(--paper); color: var(--ink3); font: var(--type-meta); }
  .status.ready { color: var(--ink); }
  .actions { display: flex; flex-wrap: wrap; gap: var(--s2); }
  .action-status, .empty { color: var(--ink3); }
  .error, .warning { color: var(--ink); }
  .notice { color: var(--ink2); }

  /* Buttons: the one way forward filled, the rest quiet on paper, a follow-up as text. */
  button:not(:global(.icon)) { display: inline-flex; align-items: center; justify-content: center; gap: var(--s2); height: 36px; padding: 0 var(--s4); border: 0; border-radius: var(--r-pill); background: var(--paper); color: var(--ink); font: var(--type-control); white-space: nowrap; cursor: pointer; transition: opacity var(--fast) var(--ease), background-color var(--fast) var(--ease), color var(--fast) var(--ease), scale var(--fast) var(--ease); }
  button:not(:global(.icon)):disabled { opacity: 0.4; cursor: default; }
  button:not(:global(.icon)):active:not(:disabled) { scale: 0.97; }
  button.primary, button.danger { padding: 0 var(--s5); background: var(--ink); color: var(--paper); }
  button.quiet { padding: 0 var(--s2); background: transparent; color: var(--ink3); }
  @media (hover: hover) {
    button:not(:global(.icon)):hover:not(:disabled) { background: var(--wash2); }
    button.primary:hover:not(:disabled), button.danger:hover:not(:disabled) { background: var(--ink); opacity: 0.86; }
    button.quiet:hover:not(:disabled) { background: transparent; color: var(--ink); }
  }

  .pack-list, .models { display: grid; gap: var(--s2); }
  .pack-list li, .models li { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: var(--s2) var(--s4); padding: var(--s3) var(--s4); border-radius: var(--r-box); background: var(--paper); }
  .pack-details, .installed-pack-details { display: grid; gap: 2px; flex: 1 1 200px; min-width: 0; }
  .pack-details strong, .installed-pack-details strong { color: var(--ink); font: var(--type-control); }
  .pack-details span, .installed-pack-details code, .models code { color: var(--ink3); font: var(--type-meta); overflow-wrap: anywhere; }
  .pack-list li > span { color: var(--ink3); font: var(--type-meta); }
  .pack-list li > span.installed, .installed { color: var(--ink); font: var(--type-meta); }
  .pack-list li button, .models li button:not(.danger) { background: var(--wash); }
  .pack-download { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s2) var(--s3); flex: 1 1 100%; color: var(--ink3); font: var(--type-meta); }
  progress { flex: 1 1 120px; height: 4px; border: 0; border-radius: 2px; background: var(--wash); overflow: hidden; appearance: none; }
  progress::-webkit-progress-bar { border-radius: 2px; background: var(--wash); }
  progress::-webkit-progress-value { border-radius: 2px; background: var(--ink); transition: width var(--glide) var(--ease); }
  progress::-moz-progress-bar { border-radius: 2px; background: var(--ink); }

  .disclosure-stack { display: grid; gap: var(--s3); padding-top: var(--s3); }
  .subsection { display: grid; gap: var(--s3); padding: var(--s4); border-radius: var(--r-row); background: var(--wash); }
  .facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: var(--s2) var(--s4); margin: 0; }
  .facts > div { display: grid; gap: 2px; min-width: 0; }
  dt { color: var(--ink3); font: var(--type-meta); }
  dd { margin: 0; color: var(--ink); font: var(--type-small); font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
  .diagnostics { padding-top: var(--s3); }
  .sources { display: flex; flex-wrap: wrap; gap: var(--s2) var(--s4); padding-top: var(--s3); }
  .sources a, .product-footer a { color: var(--ink2); font: var(--type-small); text-decoration: underline; text-decoration-color: var(--wash2); text-decoration-thickness: 2px; text-underline-offset: 4px; transition: color var(--fast) var(--ease); }
  @media (hover: hover) { .sources a:hover, .product-footer a:hover { color: var(--ink); } }

  .product-footer { display: grid; gap: var(--s2); padding-top: var(--s4); color: var(--ink3); font: var(--type-meta); }
  .product-identity { display: flex; align-items: baseline; gap: var(--s2); }
  .wordmark { color: var(--ink); font: var(--type-control); }
  .product-footer nav { display: flex; flex-wrap: wrap; gap: var(--s2); }
  .product-footer p { font: var(--type-meta); }

  @media (max-width: 720px) {
    .header { padding: var(--s4) var(--s4) 0; }
    .body { padding: var(--s4); }
    .panel { padding: var(--s4); }
    button:not(:global(.icon)) { height: 44px; }
  }
</style>
