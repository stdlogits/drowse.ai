<script lang="ts">
  // The browser runtime's lens sources, in #/artifacts/lenses under the Artifacts room's Lenses head:
  // the R lens packs this model has, the one in use, and the way to a pack when none is loaded.
  import { onMount } from "svelte";
  import {
    instrumentFamily,
    lensSourceState,
    refreshLensSources,
    sessionState,
    useLensSource,
  } from "../../../../drowse/webui/src/lib/stores.svelte";
  import InstrumentSourceSection from "../../../../drowse/webui/src/panels/rack/InstrumentSourceSection.svelte";
  import InstrumentPackRecovery from "./InstrumentPackRecovery.svelte";

  const fitted = $derived(sessionState.info?.jlens_fitted === true);
  const allowSourceSwitch = $derived(
    instrumentFamily("lens")?.capabilities.source_switch === true,
  );
  const visibleSources = $derived(
    allowSourceSwitch
      ? lensSourceState.sources
      : lensSourceState.sources.filter((source) => source.active === true),
  );
  const sourceBusy = $derived(lensSourceState.loading || lensSourceState.busy);
  let selectedSource = $state("");

  onMount(() => void refreshLensSources());
</script>

<InstrumentSourceSection
  ready={fitted}
  sources={visibleSources}
  bind:value={selectedSource}
  busy={sourceBusy}
  sourceError={lensSourceState.error}
  working={false}
  allowLocal={false}
  onuse={(source) => void useLensSource(source)}
  providerOptions={[]}
  providerPlaceholder="compatible R lens pack"
  onfetch={() => undefined}
  localActionLabel=""
  localActionDisabled
  onlocal={() => undefined}
>
  {#snippet messages()}
    {#if !fitted}<InstrumentPackRecovery kind="jlens" />{/if}
  {/snippet}
</InstrumentSourceSection>
