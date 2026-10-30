<script lang="ts">
  import { loadMediaCredits, type MediaCredits } from "./data";
  import type { ModelInfo } from "./models";

  // Where the images and sounds shown come from, with their licences: the attribution their
  // collections ask for. Collections not listed are linked to their source, never shown.
  let { model }: { model: ModelInfo } = $props();

  let credits = $state<MediaCredits | null>(null);
  $effect(() => {
    const asked = model.id;
    credits = null;
    loadMediaCredits(asked).then((c) => { if (model.id === asked) credits = c; }).catch(() => undefined);
  });
</script>

{#if credits}
  <section class="credits" aria-labelledby="media-credits-title">
    <h2 id="media-credits-title">Images and sounds</h2>
    <p class="lead">Each pole's top images and sounds come from labelled collections ({model.name}'s exemplars, neuron-lens findings §14). The ones below are shown as the model saw them; the rest, such as ImageNet and AudioSet's YouTube clips, are named and linked to their source.</p>
    <div class="lists">
      {#each model.media as modality (modality)}
        <dl>
          <dt class="kind">{modality === "image" ? "Images" : "Sounds"}</dt>
          {#each Object.entries(credits[modality] ?? {}) as [source, c] (source)}
            <dd><span class="name">{c.name}</span> <span class="license">{c.license}</span> <span class="by">{c.credit}</span></dd>
          {/each}
        </dl>
      {/each}
    </div>
  </section>
{/if}

<style>
  .credits { display: grid; gap: var(--space-3); max-width: 72rem; }
  h2 { margin: 0; color: var(--fg); font-size: var(--text-md); font-weight: var(--weight-structure-bold); }
  .lead { max-width: 70ch; margin: 0; color: var(--fg-dim); font-size: var(--text-sm); line-height: 1.55; }
  .lists { display: grid; grid-template-columns: repeat(auto-fit, minmax(20rem, 1fr)); gap: var(--space-4) var(--space-8); }
  dl { display: grid; gap: var(--space-1); margin: 0; }
  .kind { margin-bottom: var(--space-1); color: var(--fg-muted); font-size: var(--text-xs); font-weight: var(--weight-structure); }
  dd { margin: 0; color: var(--fg-dim); font-size: var(--text-xs); line-height: 1.5; }
  .name { color: var(--fg); }
  .license { padding: 0 var(--space-1); border-radius: var(--radius-sm); background: var(--glass); font-family: var(--font-mono); font-size: var(--text-2xs); }
</style>
