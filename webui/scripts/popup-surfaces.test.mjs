import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const surfaces = [
  ["../drowse/webui/src/App.svelte", ".drawer"],
  ["../drowse/webui/src/lib/Toaster.svelte", ".toast"],
  ["../drowse/webui/src/lib/Select.svelte", ".sk-select-popover"],
  ["../drowse/webui/src/lib/Combobox.svelte", ".popover"],
  ["../drowse/webui/src/lib/ui/InfoTip.svelte", ".info-popover"],
  ["../drowse/webui/src/lib/ui/TokenLogitsPopover.svelte", ".token-logits-popover"],
  ["../drowse/webui/src/lib/ui/WorkbenchMenu.svelte", ".workbench-menu"],
  ["src/hosted/ui/SavedChatMenu.svelte", ".chat-menu"],
  ["src/hosted/ui/PwaUpdatePrompt.svelte", ".pwa-notice"],
  ["src/hosted/ui/HostedRoot.svelte", ".runtime-dialog"],
  ["../drowse/webui/src/panels/CommandPalette.svelte", ".palette"],
  ["../drowse/webui/src/panels/loom/LoomSidebar.svelte", ".loom-menu"],
  ["../drowse/webui/src/panels/loom/LoomSidebar.svelte", ".loom-modal"],
];
const tokens = await readFile("../drowse/webui/src/lib/style/tokens.css", "utf8");
const app = await readFile("../drowse/webui/src/App.svelte", "utf8");
assert.match(app, /\.drawer\s*\{\s*position:\s*fixed;/, "Drawers anchor to the viewport, not below page banners");
assert.match(app, /\.drawer-backdrop\s*\{\s*position:\s*fixed;/);
assert.match(tokens, /--radius:\s*8px;/);
assert.match(tokens, /--radius-lg:\s*12px;/);
assert.match(tokens, /--popup-radius:\s*var\(--radius-lg\);/);
assert.doesNotMatch(tokens, /--radius-lg:\s*calc/, "Popup corners must not grow with content padding");
for (const path of ["../drowse/webui/src/lib/ui/DrawerCloseButton.svelte", "../drowse/webui/src/lib/ui/InfoTip.svelte"]) {
  assert.match(await readFile(path, "utf8"), /border-radius:\s*var\(--radius\);/, `${path}: controls use compact corners`);
}
const sampling = await readFile("../drowse/webui/src/drawers/AdvancedSamplingDrawer.svelte", "utf8");
assert.match(sampling, /container:\s*sampling-settings \/ inline-size;/);
assert.match(sampling, /@container sampling-settings \(max-width: 40rem\)/);
assert.doesNotMatch(sampling.match(/\.panel\s*\{([^}]*)\}/)?.[1] ?? "", /background:|box-shadow:|padding:|border-radius:/, "Sampling sections do not add nested boxes or insets");
assert.match(sampling, /\.header\s*\{[^}]*flex:\s*0 0 auto;/);
assert.match(sampling, /\.body\s*\{[^}]*min-height:\s*0;[^}]*overflow:\s*auto;/);
for (const [path, selector] of surfaces) {
  const source = await readFile(path, "utf8");
  const start = source.indexOf(`${selector} {`);
  assert.ok(start >= 0, `${path}: missing ${selector}`);
  const rule = source.slice(start, source.indexOf("}", start));
  for (const token of ["bg", "border", "radius", "shadow"]) {
    assert.ok(rule.includes(`var(--popup-${token})`), `${path} ${selector}: missing shared ${token}`);
  }
  assert.ok(!rule.includes("gradient("), `${path}: popup uses the shared sheen`);
  if (selector === ".drawer") {
    assert.match(rule, /color-mix\(in srgb, var\(--popup-bg\) 90%, transparent\)/);
  } else {
    assert.ok(!rule.includes("backdrop-filter:"), `${path}: popup must be opaque`);
  }
}
console.log(`${surfaces.length} popup surfaces share the same material tokens`);
