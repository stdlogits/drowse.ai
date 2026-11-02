// The hosted shell's chrome: the site header and footer, the help that replaces hover titles, and the
// saved-chat row's order. The look is Ledger's and is held by check-ledger.mjs; this checks what the
// chrome does.
import assert from "node:assert/strict";
import { readdir, readFile } from "node:fs/promises";
import { extname, join } from "node:path";
import { parse } from "svelte/compiler";
import { createServer } from "vite";

// No native hover titles: help is Ledger's tip (data-tip) or an accessible description.
async function svelteFiles(directory) {
  const files = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) files.push(...await svelteFiles(path));
    else if (extname(entry.name) === ".svelte") files.push(path);
  }
  return files;
}
const titled = [];
for (const file of [...await svelteFiles("src/hosted"), ...await svelteFiles("src/drawers")]) {
  const source = await readFile(file, "utf8");
  const visit = node => {
    if (!node || typeof node !== "object") return;
    if (node.type === "RegularElement" && node.attributes?.some(attribute => attribute.name === "title")) titled.push(file);
    for (const value of Object.values(node)) {
      if (Array.isArray(value)) value.forEach(visit);
      else if (value && typeof value === "object") visit(value);
    }
  };
  visit(parse(source, { modern: true }).fragment);
}
assert.deepEqual(titled, [], "native hover titles; use data-tip (Ledger's tip) or aria-description");

const server = await createServer({ appType: "custom", logLevel: "silent", server: { middlewareMode: true, watch: null } });
try {
  const { render } = await server.ssrLoadModule("svelte/server");
  const { default: SiteHeader } = await server.ssrLoadModule("/src/hosted/ui/SiteHeader.svelte");
  const { default: PageFooter } = await server.ssrLoadModule("/src/hosted/ui/PageFooter.svelte");

  // The site header: the name is the mark and always leads home; pages are links; one way in; the theme.
  const site = render(SiteHeader, { props: { current: "credits" } }).body;
  assert.match(site, /<a class="page-brand[^"]*" href="\/" aria-label="Drowse home"[^>]*>drowse<\/a>/);
  for (const [href, label] of [["/learn", "Learn"], ["/app?choose=1", "Models"], ["/credits", "Credits"], ["/contribute", "Contribute"]]) {
    assert.match(site, new RegExp(`href="${href.replace("?", "\\?")}"[^>]*>${label}</a>`), `${label} is a link`);
  }
  assert.match(site, /href="\/credits" aria-current="page"/, "the current page is marked");
  assert.match(site, /class="open-action[^"]*" href="\/app">Open Drowse<\/a>/);
  assert.match(site, /class="theme[^"]*"[^>]*aria-label="Use the [a-z]+ theme"|aria-label="Follow the system's theme"/, "the theme button is Ledger's");
  const app = render(SiteHeader, { props: { current: "chats", variant: "app" } }).body;
  assert.match(app, /href="\/app" aria-current="page"[^>]*>Chats<\/a>/);
  assert.doesNotMatch(app, /Open Drowse|href="\/learn"/, "inside the app the header offers the app's own places");
  assert.doesNotMatch(render(SiteHeader, { props: { current: "chats", variant: "app", showChats: false } }).body, />Chats<\/a>/);
  assert.doesNotMatch(render(SiteHeader, { props: { siteNavigation: false } }).body, /<nav/);
  const headerSource = await readFile("src/hosted/ui/SiteHeader.svelte", "utf8");
  assert.match(headerSource, /event\.metaKey \|\| event\.ctrlKey \|\| event\.shiftKey \|\| event\.altKey/, "modified clicks keep the browser's navigation");

  // The footer: the AGPL source offer, the licences, contact without printing the address.
  const footer = render(PageFooter, { props: { current: "privacy" } }).body;
  assert.match(footer, /href="https:\/\/github\.com\/stdlogits\/drowse\.ai"[^>]*>\s*Source code/, "a preview build offers its source");
  assert.match(footer, /href="\/LICENSE"/);
  assert.match(footer, /href="\/third-party-licenses"/);
  assert.match(footer, /href="\/privacy" aria-current="page"/);
  assert.match(footer, /\(opens GitHub\)/, "external GitHub links say where they go");
  assert.match(footer, /href="mailto:[^"]+"[^>]*>\s*Email us/);
  assert.doesNotMatch(footer, />[^<]*@[^<]*\.[a-z]+</, "the footer says Email us instead of printing the address");
  assert.match(render(PageFooter, { props: { hosted: false } }).body, /href="https:\/\/drowse\.ai\/credits"/, "off-site footers link back to drowse.ai");

  // A saved chat's timestamp sits right before its options menu.
  const home = await readFile("src/hosted/ui/HostedHome.svelte", "utf8");
  assert.match(home, /class="chat-meta">\s*<time[\s\S]*?<\/time>\s*<SavedChatMenu/);
  console.log("Site chrome: no hover titles, header and footer links, source offer, and saved-chat row order passed");
} finally {
  await server.close();
}
