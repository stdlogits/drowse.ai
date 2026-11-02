<script lang="ts">
  import { tips } from "@stdlogits/ledger";
  import SiteHeader from "./SiteHeader.svelte";
  import PageFooter from "./PageFooter.svelte";
  import ContactForm from "../../../../drowse/webui/src/lib/ui/ContactForm.svelte";
  import PwaUpdatePrompt from "./PwaUpdatePrompt.svelte";
  import { CONTACT_ADDRESS, CONTACT_REASONS, contactTopics } from "../../../../drowse/webui/src/lib/contact";
  import { contactDrafts } from "../../../../drowse/webui/src/lib/stores/contact.svelte";
  import { onMount } from "svelte";

  // Links such as /contact?reason=teaching preselect a reason, unless a draft is already in progress.
  function applyReasonFromUrl(): void {
    const reason = new URLSearchParams(window.location.search).get("reason");
    const fields = contactDrafts.contact.fields;
    if (!reason || fields.title || fields.body || !CONTACT_REASONS.some(({ value }) => value === reason)) return;
    fields.reason = reason;
    fields.topic = contactTopics(reason)[0]?.value ?? fields.topic;
  }

  const pageDescription = "Get in touch with the Drowse team. Ask for help, share feedback, or discuss working together.";

  // Update the page's one description tag instead of adding a second one.
  let mounted = $state(false);
  onMount(() => {
    mounted = true;
    document.title = "Contact us · Drowse";
    applyReasonFromUrl();
    let meta = document.head.querySelector<HTMLMetaElement>('meta[name="description"]');
    const created = !meta;
    const previous = meta?.content ?? "";
    if (!meta) {
      meta = document.createElement("meta");
      meta.name = "description";
      document.head.append(meta);
    }
    const tag = meta;
    tag.content = pageDescription;
    return () => {
      if (created) tag.remove();
      else tag.content = previous;
    };
  });
</script>

<!-- The update prompt touches window on teardown, so it only renders in the browser. -->
{#if mounted}<PwaUpdatePrompt />{/if}
<div class="contact-page" use:tips>
  <a class="skip-link" href="#contact-main">Skip to content</a>
  <SiteHeader current="contact" />
  <main id="contact-main" tabindex="-1">
    <div class="heading">
      <h1>Contact us.</h1>
      <p class="intro">Have a question or an idea for Drowse? Send us a message.</p>
      <div class="email-option"><span>Prefer email?</span><a href={`mailto:${CONTACT_ADDRESS}`}>{CONTACT_ADDRESS}</a></div>
    </div>
    <div class="form-surface"><ContactForm source="contact" /></div>
  </main>
  <PageFooter current="contact" />
</div>

<style>
  /* Contact: the title and the email beside the form, which sits on a wash. */
  .contact-page { display: flex; flex-direction: column; min-height: 100svh; }
  .skip-link { position: fixed; z-index: 60; top: var(--s2); inset-inline-start: var(--s2); padding: var(--s2) var(--s4); border-radius: var(--r-pill); background: var(--ink); color: var(--paper); font: var(--type-control); translate: 0 calc(-100% - var(--s4)); transition: translate var(--fast) var(--ease); }
  .skip-link:focus { translate: 0 0; }
  main { flex: 1; display: grid; grid-template-columns: minmax(0, 360px) minmax(0, 1fr); align-items: start; gap: var(--s7); width: 100%; max-width: var(--page); margin: 0 auto; padding: var(--s7) var(--gutter) var(--s9); }
  main:focus { outline: none; }
  .heading { position: sticky; top: var(--s5); display: grid; justify-items: start; gap: var(--s4); }
  h1 { color: var(--ink); font: var(--type-title); letter-spacing: -0.04em; }
  .intro { color: var(--ink2); font: var(--type-lead); }
  .email-option { display: flex; flex-wrap: wrap; gap: var(--s2); color: var(--ink3); font: var(--type-small); }
  .email-option a { color: var(--ink2); text-decoration: underline; text-decoration-color: var(--wash2); text-decoration-thickness: 2px; text-underline-offset: 4px; overflow-wrap: anywhere; transition: color var(--fast) var(--ease); }
  @media (hover: hover) { .email-option a:hover { color: var(--ink); } }
  .form-surface { --field: var(--wash2); --field-hover: var(--paper); min-width: 0; padding: var(--s6); border-radius: var(--r-row); background: var(--wash); }
  @media (max-width: 900px) {
    main { grid-template-columns: minmax(0, 1fr); gap: var(--s6); padding-top: var(--s5); }
    .heading { position: static; }
    .form-surface { padding: var(--s4); }
  }
</style>
