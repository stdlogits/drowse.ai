// The hosted pages' pre-paint script, inlined into every document (its digest is in the CSP).
// It is Ledger's theme-preflight.js with the same key and semantics: a stored light or dark choice
// sets data-theme before first paint, and with none (auto) nothing is set and the tokens follow the
// system; createTheme (lib/theme.ts) owns the theme after. It also points the browser's theme colour
// at the chosen paper, and keeps an aborted cross-document view transition out of the console.
(function () {
  function quiet(transition) {
    if (transition) transition.ready.catch(function () {});
  }
  quiet(document.activeViewTransition);
  window.addEventListener("pageswap", function (event) { quiet(event.viewTransition); });
  window.addEventListener("pagereveal", function (event) { quiet(event.viewTransition); });
  try {
    var script = document.currentScript;
    var key = script && script.getAttribute("data-ledger-theme-key");
    var mode = key && localStorage.getItem(key);
    if (mode !== "light" && mode !== "dark") return;
    document.documentElement.dataset.theme = mode;
    var metas = document.querySelectorAll('meta[name="theme-color"]');
    for (var i = 0; i < metas.length; i++) metas[i].content = mode === "light" ? "#ffffff" : "#0b0d12";
  } catch (error) {}
})();
