<script lang="ts">
  // No lens for this model yet: the Artifacts room's Lenses section (#/artifacts/lenses) shows the
  // browser runtime's lens packs and the way to one. Swapped in for the dashboard's own through
  // @runtime-jlens-missing, with the same prop.
  import { goto } from "../../../../drowse/webui/src/lib/stores/rooms.svelte";
  import EmptyState from "../../../../drowse/webui/src/drawers/token/EmptyState.svelte";

  let { modelId }: { modelId: string | null } = $props();
</script>

<EmptyState title={modelId ? `No lens for ${modelId.split("/").at(-1)}` : "No lens yet"}>
  <button type="button" class="dw-quiet" onclick={() => goto("artifacts", "lenses")}>Choose a lens</button>
</EmptyState>
