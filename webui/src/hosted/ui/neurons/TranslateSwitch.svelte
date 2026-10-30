<script lang="ts">
  import { setTranslate, translation } from "./translate.svelte";

  const HELP = "Machine translations by Claude Haiku 5.5, made when the data was built";
  const helpId = $props.id();
</script>

<label class="translate" data-help={HELP}>
  <input type="checkbox" checked={translation.on} aria-describedby={helpId} onchange={(event) => setTranslate(event.currentTarget.checked)} />
  <span class="switch" aria-hidden="true"></span>
  <span>Translate to English</span>
</label>
<!-- Outside the label, so the switch's sibling selectors still match; hidden text still describes. -->
<span id={helpId} hidden>{HELP}</span>

<style>
  .translate { position: relative; display: inline-flex; align-items: center; gap: var(--space-3); min-height: var(--control-target); color: var(--fg-dim); font-size: var(--text-sm); white-space: nowrap; cursor: pointer; }
  .translate input { position: absolute; opacity: 0; width: 1px; height: 1px; }
  .switch { position: relative; flex: none; width: 36px; height: 20px; border-radius: var(--radius-pill); background: var(--glass-strong); box-shadow: inset 0 0 0 1px var(--glass-line); transition: background-color var(--dur) var(--ease-out); }
  .switch::after { content: ""; position: absolute; top: 3px; left: 3px; width: 14px; height: 14px; border-radius: 50%; background: var(--fg-dim); transition: translate var(--dur) var(--ease-out), background-color var(--dur) var(--ease-out); }
  .translate input:checked + .switch { background: color-mix(in srgb, var(--accent-amber) 60%, transparent); }
  .translate input:checked + .switch::after { translate: 16px 0; background: var(--fg); }
  .translate input:checked ~ span:last-child { color: var(--fg); }
  .translate input:focus-visible + .switch { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  @media (prefers-reduced-motion: reduce) { .switch, .switch::after { transition: none; } }
</style>
