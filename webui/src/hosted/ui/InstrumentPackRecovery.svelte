<script lang="ts">
  // A lens or SAE source section with nothing loaded: one plain sentence and one way out (Models,
  // where the model's packs download).
  import { onMount } from "svelte";
  import { sessionState } from "../../../../drowse/webui/src/lib/stores/session.svelte";
  import {
    currentCatalogInstrumentAvailability,
    type CatalogInstrumentAvailability,
  } from "../../../../drowse/webui/src/lib/runtime/instrumentAvailability";

  let { kind }: { kind: "jlens" | "slens" | "units" | "sae" } = $props();
  let availability = $state<CatalogInstrumentAvailability>("unknown");
  const name = $derived({ jlens: "R lens", slens: "S lens", units: "units", sae: "SAE" }[kind]);
  const message = $derived(
    availability === "unavailable"
      ? `No compatible ${name} pack is published for this model.`
      : availability === "context-unavailable"
        ? `No ${name} pack supports this conversation length. Open Models to choose a supported length.`
        : `No ${name} pack is loaded. Open Models to download or update the compatible pack, then reopen the model. Your chats stay saved.`,
  );

  onMount(() => {
    let mounted = true;
    // The shared helper filters packs by kind; its type names jlens and sae only until drowse's
    // carries 6.0's slens and units (lanes/B6.md, needs).
    void currentCatalogInstrumentAvailability(kind as "jlens" | "sae").then((value) => {
      if (mounted) availability = value;
    });
    return () => { mounted = false; };
  });
</script>

<div class="pack-recovery">
  <p>{message}</p>
  {#if kind === "jlens" && sessionState.info?.is_base_model}
    <p>Word insights are optional for base models. Text completion works without a compatible R lens pack.</p>
  {/if}
  <button type="button" class="dw-quiet" onclick={() => window.dispatchEvent(new Event("drowse:models"))}>Open Models</button>
</div>

<style>
  .pack-recovery { display: grid; justify-items: start; gap: var(--s3); }
  p { color: var(--ink2); font: var(--type-small); }
</style>
