<script lang="ts">
  import { onMount, tick } from "svelte";
  import { Icon } from "@stdlogits/ledger";
  import { dropdownMotion } from "../../../../drowse/webui/src/lib/dropdownMotion.svelte";

  let { name, disabled = false, duplicateDisabled = false, busy = false, onduplicate, ondelete }: {
    name: string;
    disabled?: boolean;
    duplicateDisabled?: boolean;
    busy?: boolean;
    onduplicate: () => void;
    ondelete: () => void;
  } = $props();

  const id = $props.id();
  const presence = dropdownMotion();
  let open = $state(false);
  let active = $state(0);
  let trigger: HTMLButtonElement | null = $state(null);
  let popover: HTMLDivElement | null = $state(null);
  let surface: HTMLDivElement | null = $state(null);
  let position = $state("");

  function place() {
    if (!open || !trigger || !popover || !surface) return;
    const rect = trigger.getBoundingClientRect();
    const viewport = window.visualViewport;
    const x = viewport?.offsetLeft ?? 0;
    const y = viewport?.offsetTop ?? 0;
    const width = viewport?.width ?? innerWidth;
    const bottom = y + (viewport?.height ?? innerHeight);
    const menuWidth = Math.min(184, width - 16);
    const left = Math.max(x + 8, Math.min(rect.right - menuWidth, x + width - menuWidth - 8));
    const height = surface.scrollHeight;
    const flip = rect.bottom + 8 + height > bottom - 8 && rect.top - height - 8 >= y + 8;
    const top = Math.max(y + 8, Math.min(flip ? rect.top - height - 8 : rect.bottom + 8, bottom - height - 8));
    const bounds = popover.getBoundingClientRect();
    const styles = getComputedStyle(popover);
    const offsetX = parseFloat(styles.left) - bounds.left;
    const offsetY = parseFloat(styles.top) - bounds.top;
    position = `left:${left + offsetX}px;top:${top + offsetY}px;width:${menuWidth}px;max-height:${Math.max(0, bottom - top - 8)}px;`;
    surface.dataset.origin = flip ? "bottom-right" : "top-right";
  }

  function focusItem(index: number) {
    active = index;
    surface?.querySelectorAll<HTMLButtonElement>("button")[index]?.focus({ preventScroll: true });
  }

  async function show(last = false) {
    if (disabled) return;
    open = true;
    presence.mount();
    await tick();
    if (!open || !popover || !surface) return;
    popover.showPopover();
    place();
    await tick();
    if (!open || !surface) return;
    presence.show(surface);
    focusItem(last || duplicateDisabled ? 1 : 0);
  }

  function close(restoreFocus = false) {
    open = false;
    if (restoreFocus) trigger?.focus({ preventScroll: true });
    presence.close(surface);
  }

  function run(action: () => void) {
    close(true);
    action();
  }

  function onKeydown(event: KeyboardEvent) {
    if (event.key === "Escape" && open) {
      event.preventDefault();
      event.stopPropagation();
      close(true);
    } else if (event.key === "Tab" && open) {
      close(true);
    } else if (["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) {
      event.preventDefault();
      if (!open) void show(event.key === "ArrowUp" || event.key === "End");
      else focusItem(duplicateDisabled || event.key === "End" ? 1 : event.key === "Home" ? 0 : 1 - active);
    } else if (open && !event.ctrlKey && !event.metaKey && !event.altKey && /^[dD]$/.test(event.key)) {
      event.preventDefault();
      focusItem(duplicateDisabled ? 1 : 1 - active);
    }
  }

  function outside(event: Event) {
    if (open && !popover?.contains(event.target as Node) && !trigger?.contains(event.target as Node)) close();
  }

  $effect(() => { if (disabled && open) close(); });

  onMount(() => {
    document.addEventListener("pointerdown", outside, true);
    document.addEventListener("focusin", outside);
    window.addEventListener("scroll", place, true);
    const viewport = window.visualViewport;
    viewport?.addEventListener("resize", place);
    viewport?.addEventListener("scroll", place);
    return () => {
      open = false;
      document.removeEventListener("pointerdown", outside, true);
      document.removeEventListener("focusin", outside);
      window.removeEventListener("scroll", place, true);
      viewport?.removeEventListener("resize", place);
      viewport?.removeEventListener("scroll", place);
      presence.destroy();
    };
  });
</script>

<svelte:window onresize={place} />

<button bind:this={trigger} type="button" class="chat-menu-trigger" {disabled}
  aria-label={`More options for ${name}`} aria-haspopup="menu" aria-expanded={open}
  aria-controls={presence.mounted ? id : undefined} aria-busy={busy}
  onclick={() => open ? close() : void show()} onkeydown={onKeydown}>
  <Icon name="menu" />
</button>

{#if presence.mounted}
  <div bind:this={popover} class="chat-menu-popover" popover="manual" style={position} inert={!open}>
    <div bind:this={surface} id={id} class="chat-menu" role="menu" tabindex="-1" aria-label={`Options for ${name}`} onkeydown={onKeydown}>
      <button type="button" role="menuitem" tabindex={active === 0 ? 0 : -1} disabled={disabled || duplicateDisabled}
        aria-label={`Duplicate ${name}`} onfocus={() => active = 0} onclick={() => run(onduplicate)}>Duplicate</button>
      <button type="button" role="menuitem" tabindex={active === 1 ? 0 : -1} {disabled}
        onfocus={() => active = 1} onclick={() => run(ondelete)}>Delete</button>
    </div>
  </div>
{/if}

<style>
  /* A row's few more actions: a round icon button that opens a small wash2 menu (the dashboard's
     header menu, in miniature). It rises in and fades out on Ledger's curve. */
  .chat-menu-trigger { display: grid; place-items: center; flex: none; width: 36px; height: 36px; padding: 0; border: 0; border-radius: var(--r-pill); background: transparent; color: var(--ink3); cursor: pointer; touch-action: manipulation; transition: background-color var(--fast) var(--ease), color var(--fast) var(--ease), scale var(--fast) var(--ease); }
  .chat-menu-trigger:active:not(:disabled) { scale: 0.94; }
  .chat-menu-trigger:disabled { opacity: 0.4; cursor: default; }
  .chat-menu-trigger[aria-expanded="true"] { background: var(--wash2); color: var(--ink); }
  @media (hover: hover) { .chat-menu-trigger:hover:not(:disabled) { background: var(--wash2); color: var(--ink); } }
  .chat-menu-popover { position: fixed; inset: auto; width: min(184px, calc(100vw - 16px)); margin: 0; padding: 0; border: 0; background: transparent; overflow: visible; pointer-events: none; }
  .chat-menu {
    --dropdown-close-dur: var(--out);
    display: grid;
    gap: 2px;
    max-height: inherit;
    padding: var(--s1);
    overflow: auto;
    overscroll-behavior: contain;
    border-radius: var(--r-box);
    background: var(--wash2);
    color: var(--ink2);
    font: var(--type-control);
    opacity: 0;
    translate: 0 -4px;
    pointer-events: auto;
    transition: opacity var(--out) var(--ease), translate var(--out) var(--ease);
  }
  .chat-menu:focus { outline: none; }
  .chat-menu:global(.is-open) { opacity: 1; translate: 0 0; transition-duration: var(--in); }
  .chat-menu button { display: flex; align-items: center; min-width: 0; min-height: 40px; padding: 0 var(--s3); border: 0; border-radius: var(--r-col); background: transparent; color: var(--ink2); font: inherit; text-align: start; cursor: pointer; touch-action: manipulation; transition: background-color var(--fast) var(--ease), color var(--fast) var(--ease); }
  .chat-menu button:disabled { opacity: 0.4; cursor: default; }
  .chat-menu button:focus-visible { outline-offset: -2px; }
  @media (hover: hover) { .chat-menu button:hover:not(:disabled) { background: var(--paper); color: var(--ink); } }
  @media (pointer: coarse) { .chat-menu-trigger { width: 44px; height: 44px; } .chat-menu button { min-height: 44px; } }
  @media (prefers-reduced-motion: reduce) { .chat-menu { translate: none; } }
</style>
