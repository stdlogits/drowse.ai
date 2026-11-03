<script lang="ts">
  import { MorphText } from "@stdlogits/ledger";
  import TipText from "./TipText.svelte";
  import { formatEtaRange } from "../../lib/runtime/eta";
  import { userFacingError } from "../../../../drowse/webui/src/lib/runtime/userFacingError";
  import { formatBytes } from "../../lib/format/bytes";
  import { modelsForChooser } from "./modelSelection";
  import type { HostedModelOption, HostedShellController, HostedShellSnapshot } from "../../../../drowse/webui/src/lib/shell";

  let { controller, snapshot, disabled = false, onSelect }: {
    controller: HostedShellController;
    snapshot: HostedShellSnapshot;
    disabled?: boolean;
    onSelect: (model: HostedModelOption) => void;
  } = $props();

  let armedId = $state<string | null>(null);
  let pendingId = $state<string | null>(null);
  let pausing = $state(false);
  let failure = $state<{ id: string; message: string } | null>(null);
  const busy = $derived(pendingId !== null || ["requesting_persistence", "downloading", "cancelling"].includes(snapshot.download.phase));
  const ARMED_TIMEOUT_MS = 8_000;

  const tierLabel: Record<HostedModelOption["tier"], string> = {
    fastest: "Fastest",
    balanced: "Balanced",
    quality: "Best quality",
  };

  // A confirmation that is not completed soon expires, so a later stray press
  // never starts a download the visitor has forgotten about.
  $effect(() => {
    if (armedId === null) return;
    const timer = setTimeout(() => { armedId = null; }, ARMED_TIMEOUT_MS);
    return () => clearTimeout(timer);
  });

  function toolChips(model: HostedModelOption): string[] {
    return [
      ...(model.firstRunPacks.some((pack) => pack.kind === "jlens") ? ["Word insights (R lens)"] : []),
      ...(model.firstRunPacks.some((pack) => pack.kind === "slens") ? ["Early-layer reads (S lens)"] : []),
      ...(model.firstRunPacks.some((pack) => pack.kind === "units") ? ["Neurons and heads (units)"] : []),
      ...(model.firstRunPacks.some((pack) => pack.kind === "sae") ? ["Feature insights (SAE)"] : []),
    ];
  }

  function unavailable(model: HostedModelOption): boolean {
    return disabled || busy || model.fit === "blocked" || (model.setupComplete
      ? !snapshot.runtime.available
      : !snapshot.download.available || model.catalogAvailable === false);
  }

  function downloadSize(model: HostedModelOption): string {
    return formatBytes(model.remainingDownloadBytes);
  }

  function cardSize(model: HostedModelOption): string {
    return model.setupComplete || model.remainingDownloadBytes === 0
      ? formatBytes(model.firstRunBytes)
      : downloadSize(model);
  }

  async function choose(model: HostedModelOption): Promise<void> {
    if (unavailable(model)) return;
    if (model.setupComplete) {
      armedId = null;
      onSelect(model);
      return;
    }
    if (armedId !== model.id) {
      armedId = model.id;
      return;
    }
    armedId = null;
    pendingId = model.id;
    failure = null;
    try {
      await controller.download(model.id, {
        explicitUnsafeOverride: model.fit === "uncertain" || model.requiresOomRetry === true,
      });
      // Open the model as soon as its setup finishes, as the Models page does.
      const installed = controller.current().models.find((candidate) => candidate.id === model.id);
      if (installed?.setupComplete && !disabled && controller.current().runtime.available) {
        onSelect(installed);
      }
    } catch (cause) {
      failure = { id: model.id, message: userFacingError(cause, "Download failed. Try again.") };
    } finally {
      pendingId = null;
    }
  }

  async function pause(id: string): Promise<void> {
    if (pausing || snapshot.download.phase !== "downloading") return;
    pausing = true;
    try {
      await controller.cancelDownload();
    } catch (cause) {
      failure = { id, message: userFacingError(cause, "Could not pause the download. Try again.") };
    } finally {
      pausing = false;
    }
  }
</script>

<div class="model-choice-groups">
  {#each ["chat", "base"] as kind}
    {@const models = modelsForChooser(snapshot.models.filter(model => (model.modelType === "base" ? "base" : "chat") === kind))}
    {#if models.length}
      <section class="model-choice-group" aria-labelledby={`new-chat-${kind}-title`}>
        <div class="model-group-heading">
          <h3 id={`new-chat-${kind}-title`}>{kind === "base" ? "Base models" : "Chat models"}</h3>
          {#if kind === "base"}<TipText label="What are base models?" text="Base models predict what comes next in a piece of text. For conversation or instructions, choose a chat model." />{/if}
        </div>
        <div class="model-choices">
          {#each models as model (model.id)}
            {@const download = snapshot.download.modelVariantId === model.id ? snapshot.download : null}
            {@const active = !model.setupComplete && (pendingId === model.id || (download !== null && ["requesting_persistence", "downloading", "cancelling"].includes(download.phase)))}
            {@const progress = download?.progress}
            {@const determinate = progress !== undefined && progress.bytesTotal > 0}
            {@const percent = determinate ? Math.max(0, Math.min(100, Math.round(progress.bytesReceived / progress.bytesTotal * 100))) : 0}
            {@const failed = failure?.id === model.id ? failure.message : download?.phase === "failed" ? download.reason : null}
            {@const paused = download?.phase === "paused"}
            {@const armed = armedId === model.id && !model.setupComplete}
            {@const status = download?.phase === "cancelling" ? "Pausing…"
              : download?.phase === "requesting_persistence" || (pendingId === model.id && download?.phase !== "downloading") ? "Preparing download…"
              : progress?.offline ? "Offline. Waiting for a connection"
              : progress?.stalled ? "Waiting for data…"
              : determinate && progress.bytesReceived >= progress.bytesTotal ? "Verifying files…"
              : progress?.etaSeconds ? `${formatEtaRange(progress.etaSeconds)} remaining` : "Downloading…"}
            <div class="model-choice" class:installed={model.installed} class:armed class:active data-model-id={model.id}>
              {#if model.catalogAvailable !== false}
                <span class="model-download-size"><span class="sr-only">{model.setupComplete ? "Stored size: " : "Download size: "}</span>{cardSize(model)}</span>
              {/if}
              <button type="button" class="model-choice-button" disabled={unavailable(model)}
                onclick={() => void choose(model)}
                onkeydown={(event) => { if (event.key === "Escape") armedId = null; }}
                onblur={() => { if (armedId === model.id) armedId = null; }}
                aria-labelledby={`chooser-name-${model.id}`}
                aria-describedby={[
                  `chooser-status-${model.id}`,
                  `chooser-chips-${model.id}`,
                  ...(armed && (model.fit === "uncertain" || model.requiresOomRetry) ? [`model-warning-${model.id}`] : []),
                ].join(" ")}>
                <span class="model-choice-copy">
                  <span class="model-name" id={`chooser-name-${model.id}`}>{model.name}</span>
                  <span class="model-chips" id={`chooser-chips-${model.id}`}>
                    {#if model.fit === "recommended"}<span class="chip recommended">Recommended for this device</span>{/if}
                    <span class="chip">{tierLabel[model.tier]}</span>
                    {#each toolChips(model) as chip (chip)}<span class="chip">{chip}</span>{/each}
                  </span>
                  <span class="model-status" id={`chooser-status-${model.id}`} aria-live={active || armed ? "polite" : undefined}><MorphText text={model.fit === "blocked" ? "Not supported on this device"
                    : model.setupComplete ? "Ready to use"
                    : active ? "Download in progress"
                    : model.catalogAvailable === false || !snapshot.download.available ? "Download unavailable"
                    : armed ? `Download ${downloadSize(model)}? Select again to confirm`
                    : paused ? "Download paused. Select to resume"
                    : failed ? "Download failed. Select to retry"
                    : model.installed ? "Model downloaded. Select to add its tools"
                    : "Select to download"} /></span>
                </span>
              </button>
              {#if armed && (model.fit === "uncertain" || model.requiresOomRetry)}
                <p class="model-warning" id={`model-warning-${model.id}`} role="alert">{model.reason} The browser may run out of memory. Select again only if you want to continue.</p>
              {/if}
              {#if active || (paused && !model.setupComplete)}
                <div class="model-download">
                  <div class="progress-track" role="progressbar" aria-label={`Download ${model.name}`}
                    aria-valuemin="0" aria-valuemax="100" aria-valuenow={determinate ? percent : undefined}
                    aria-valuetext={paused ? "Download paused" : `${determinate ? `${percent}% · ` : ""}${status}`}>
                    <span class:indeterminate={!determinate} style:width={`${determinate ? percent : 18}%`}></span>
                  </div>
                  <div class="download-detail">
                    <span><MorphText text={paused ? "Downloaded files kept" : status} numbers={false} />{#if determinate} · <MorphText text={`${percent}%`} />{/if}</span>
                    {#if active}<button type="button" class="pause-download" disabled={pausing || download?.phase !== "downloading"} onclick={() => void pause(model.id)}>Pause</button>{/if}
                  </div>
                </div>
              {/if}
              {#if failed && !model.setupComplete}<p class="model-error" role="alert">{failed}</p>{/if}
            </div>
          {/each}
        </div>
      </section>
    {/if}
  {/each}
</div>

<style>
  /* The model chooser inside Your chats: a row of wash per model, chips on paper, the status in
     ink3, a thin ink bar while it downloads. */
  .model-choice-groups { display: grid; gap: var(--s5); }
  .model-choice-group { display: grid; gap: var(--s3); }
  .model-group-heading { display: flex; flex-wrap: wrap; align-items: baseline; gap: var(--s2) var(--s4); }
  h3 { color: var(--ink3); font: var(--type-head); }
  .model-choices { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 280px), 1fr)); gap: var(--s2); }
  .model-choice { position: relative; display: grid; gap: var(--s2); align-content: start; padding: var(--s4); border-radius: var(--r-row); background: var(--paper); transition: background-color var(--fast) var(--ease); }
  .model-choice.armed, .model-choice.active { background: var(--wash2); }
  .model-download-size { position: absolute; top: var(--s4); right: var(--s4); color: var(--ink3); font: var(--type-meta); }
  .model-choice-button { display: grid; gap: var(--s2); width: 100%; min-width: 0; padding: 0; border: 0; background: transparent; color: inherit; font: inherit; text-align: start; cursor: pointer; transition: scale var(--fast) var(--ease); }
  .model-choice-button:active:not(:disabled) { scale: 0.99; }
  .model-choice-button:disabled { opacity: 0.4; cursor: default; }
  .model-choice-button:focus-visible { outline-offset: var(--s2); border-radius: var(--r-box); }
  .model-choice-copy { display: grid; gap: var(--s2); min-width: 0; padding-right: var(--s8); }
  .model-name { color: var(--ink); font: var(--type-label); overflow-wrap: anywhere; }
  .model-chips { display: flex; flex-wrap: wrap; gap: var(--s1); }
  .chip { padding: 2px var(--s2); border-radius: var(--r-chip); background: var(--wash); color: var(--ink2); font: var(--type-meta); }
  .model-choice.armed .chip, .model-choice.active .chip { background: var(--paper); }
  .chip.recommended { color: var(--ink); }
  .model-status { color: var(--ink3); font: var(--type-small); }
  .model-choice.armed .model-status { color: var(--ink); }
  .model-warning, .model-error { color: var(--ink); font: var(--type-small); }
  .model-download { display: grid; gap: var(--s2); }
  .progress-track { position: relative; height: 4px; border-radius: 2px; background: var(--wash); overflow: hidden; }
  .progress-track span { display: block; height: 100%; border-radius: 2px; background: var(--ink); transition: width var(--glide) var(--ease); }
  .progress-track span.indeterminate { opacity: 0.35; }
  .download-detail { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: var(--s2); color: var(--ink3); font: var(--type-meta); }
  .pause-download { height: 32px; padding: 0 var(--s3); border: 0; border-radius: var(--r-pill); background: var(--paper); color: var(--ink); font: var(--type-control); cursor: pointer; transition: background-color var(--fast) var(--ease), scale var(--fast) var(--ease); }
  .pause-download:active:not(:disabled) { scale: 0.97; }
  .pause-download:disabled { opacity: 0.4; cursor: default; }
  @media (hover: hover) {
    .model-choice:has(.model-choice-button:hover:not(:disabled)) { background: var(--wash2); }
    .pause-download:hover:not(:disabled) { background: var(--wash); }
  }
  @media (max-width: 720px) { .pause-download { height: 44px; } }
  /* Forced colours flatten the paper and the washes: keep each choice, the armed or installed one
     and the download bar distinct with system colours. */
  @media (forced-colors: active) {
    .model-choice { border: 1px solid CanvasText; }
    .model-choice.armed, .model-choice.active, .model-choice.installed { border: 2px solid Highlight; }
    .progress-track { border: 1px solid CanvasText; }
    .progress-track span { background: Highlight; forced-color-adjust: none; }
  }
</style>
