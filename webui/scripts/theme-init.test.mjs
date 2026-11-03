// The hosted pages' pre-paint script (public-hosted/theme-init.js) is Ledger's theme preflight with
// the same key and semantics, plus the browser theme colour and quiet view transitions. It runs on
// the same inputs as Ledger's own preflight and must leave the page in the same state.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { runInNewContext } from "node:vm";
import { themeInitScript } from "./site-publication.mjs";
import { themeColors } from "./site-metadata.mjs";

const [hosted, ledger, index, theme, tokens] = await Promise.all([
  readFile("public-hosted/theme-init.js", "utf8"),
  readFile("../node_modules/@stdlogits/ledger/src/theme-preflight.js", "utf8"),
  readFile("hosted/index.html", "utf8"),
  readFile("../drowse/webui/src/lib/theme.ts", "utf8"),
  readFile("../node_modules/@stdlogits/ledger/src/styles/tokens.css", "utf8"),
]);

const KEY = "drowse.theme";
assert.ok(index.includes(`<script src="/theme-init.js" data-ledger-theme-key="${KEY}"></script>`), "index.html names the theme key");
assert.match(theme, new RegExp(`THEME_STORAGE_KEY = "${KEY.replace(".", "\\.")}"`), "createTheme keeps the theme under the same key");

/** Runs a pre-paint script against a fake document; returns what it left behind. */
function run(source, stored, { blocked = false } = {}) {
  const root = { dataset: {} };
  const metas = [{ media: "(prefers-color-scheme: dark)", content: themeColors.dark }, { media: "(prefers-color-scheme: light)", content: themeColors.light }];
  const listeners = {};
  let quieted = 0;
  runInNewContext(source, {
    localStorage: { getItem: key => { if (blocked) throw new Error("blocked"); return key === KEY ? stored : null; } },
    window: { addEventListener: (name, handler) => { listeners[name] = handler; } },
    document: {
      currentScript: { getAttribute: name => name === "data-ledger-theme-key" ? KEY : null },
      documentElement: root,
      activeViewTransition: { ready: { catch() { quieted += 1; } } },
      querySelectorAll: selector => selector.includes("theme-color") ? metas : [],
    },
  });
  return { theme: root.dataset.theme, metas: metas.map(meta => meta.content), listeners, quieted };
}

for (const [stored, blocked] of [["light"], ["dark"], [null], ["auto"], ["invalid"], ["dark", true]]) {
  const page = run(hosted, stored, { blocked });
  const reference = run(ledger, stored, { blocked });
  assert.equal(page.theme, reference.theme, `stored ${stored}${blocked ? " (storage blocked)" : ""}: same data-theme as Ledger's preflight`);
  if (stored === "light" || stored === "dark") {
    if (!blocked) assert.deepEqual(page.metas, [themeColors[stored], themeColors[stored]], `a stored ${stored} paints both theme colours`);
  } else {
    assert.equal(page.theme, undefined, "auto leaves data-theme unset, so the tokens follow the system");
    assert.deepEqual(page.metas, [themeColors.dark, themeColors.light], "auto keeps each look's own theme colour");
  }
  assert.equal(typeof page.listeners.pageswap, "function", "skipped page transitions are handled before the app mounts");
  assert.equal(typeof page.listeners.pagereveal, "function");
  assert.equal(page.quieted, 1, "a view transition already running is quieted");
}

// The theme colours are Ledger's paper in each theme.
const paper = selector => tokens.slice(tokens.indexOf(selector)).match(/--paper:\s*(#[0-9a-f]{6})/i)[1].toLowerCase();
assert.deepEqual(themeColors, { light: paper(":root {"), dark: paper(':root[data-theme="dark"]') });
assert.ok(hosted.includes(themeColors.light) && hosted.includes(themeColors.dark), "theme-init.js paints Ledger's papers");

// Inlined at build with its digest in the CSP.
const { source, cspSource } = await themeInitScript();
assert.equal(source, hosted.trim());
assert.match(cspSource, /^'sha256-[A-Za-z0-9+/]+=*'$/);
console.log("Theme: Ledger's preflight semantics, shared key, paper theme colours, quiet view transitions and the CSP digest passed");
