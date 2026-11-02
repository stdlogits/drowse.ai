<script lang="ts">
  import { tips } from "@stdlogits/ledger";
  import SiteHeader from "./SiteHeader.svelte";
  import PageFooter from "./PageFooter.svelte";
  import RecordedTokenText from "./RecordedTokenText.svelte";
  import { nextMessage, recording } from "./notFoundMessages";

  // Not found: one of the recorded replies a model gave when asked about a missing page, shown as
  // drowse shows a reply (each word selectable for its alternatives), then the two ways back.
  const message = recording.messages[nextMessage()];
  $effect(() => { document.title = "Page not found · Drowse"; });
</script>

<a class="skip-link" href="#not-found-main">Skip to content</a>
<div class="not-found-shell" use:tips>
  <SiteHeader />
  <main id="not-found-main" tabindex="-1">
    <section class="message-stage" aria-label="Page not found">
      <p class="code meta">404 · page not found</p>
      <div class="message" data-message-id={message.id}>
        <RecordedTokenText tokens={message.tokens} />
      </div>
      <p class="token-caption">Drowse shows what the model predicted for each word. Select a word to see the alternatives.</p>
      <div class="recovery-actions">
        <a class="primary-action" href="/">Back to home</a>
        <a class="secondary-action" href="/app">Open Drowse</a>
      </div>
    </section>
  </main>
  <PageFooter />
</div>

<style>
  .skip-link { position: fixed; z-index: 60; top: var(--s2); inset-inline-start: var(--s2); padding: var(--s2) var(--s4); border-radius: var(--r-pill); background: var(--ink); color: var(--paper); font: var(--type-control); translate: 0 calc(-100% - var(--s4)); transition: translate var(--fast) var(--ease); }
  .skip-link:focus { translate: 0 0; }
  .not-found-shell { display: flex; flex-direction: column; min-height: 100svh; }
  main { flex: 1; display: grid; align-content: center; width: 100%; max-width: 760px; margin: 0 auto; padding: var(--s8) var(--gutter); }
  main:focus { outline: none; }
  .message-stage { display: grid; justify-items: start; gap: var(--s5); }
  .message { color: var(--ink); font: var(--type-lead); }
  .token-caption { max-width: 52ch; color: var(--ink3); font: var(--type-small); }
  .recovery-actions { display: flex; flex-wrap: wrap; gap: var(--s2); }
  .recovery-actions a { display: inline-flex; align-items: center; height: 44px; padding: 0 var(--s5); border-radius: var(--r-pill); font: var(--type-control); transition: opacity var(--fast) var(--ease), background-color var(--fast) var(--ease), scale var(--fast) var(--ease); }
  .recovery-actions a:active { scale: 0.97; }
  .primary-action { background: var(--ink); color: var(--paper); }
  .secondary-action { background: var(--wash); color: var(--ink); }
  @media (hover: hover) {
    .primary-action:hover { opacity: 0.86; }
    .secondary-action:hover { background: var(--wash2); }
  }
</style>
