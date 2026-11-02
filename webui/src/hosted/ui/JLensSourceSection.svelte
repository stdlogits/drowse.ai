<script lang="ts">
  // The browser runtime's lens sources, in #/artifacts/lenses under the Artifacts room's Lenses head:
  // 6.0's four lenses (drowse's 2x2, S and R at out and mid) over the installed packs, the R pack in
  // use chosen among those installed, and the way to a pack for a family with none loaded. The
  // runtime offers no fit or fetch, so it renders drowse's LensQuad with the default column (no fit
  // form) rather than the native section, and the shared columns show sources and coverage only.
  import LensQuad from "../../../../drowse/webui/src/panels/rooms/lens/LensQuad.svelte";
  import { lensSourceState, sessionState } from "../../../../drowse/webui/src/lib/stores.svelte";
  import InstrumentPackRecovery from "./InstrumentPackRecovery.svelte";

  const rLoaded = $derived(sessionState.info?.jlens_fitted === true);
  const sLoaded = $derived(lensSourceState.sources.some((row) => row.family === "S" && row.active === true));
</script>

<LensQuad />
{#if !rLoaded}<InstrumentPackRecovery kind="jlens" />{/if}
{#if !sLoaded && sessionState.info?.is_base_model !== true}<InstrumentPackRecovery kind="slens" />{/if}
