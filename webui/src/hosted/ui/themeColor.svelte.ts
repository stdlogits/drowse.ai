import { theme } from "../../../../drowse/webui/src/lib/theme";

// The browser's theme colour follows the page's paper. index.html carries one meta per system look
// and theme-init.js points them at a stored choice before first paint; this keeps them on the paper
// the page has as the theme button or the system changes it.
export function followThemeColor(): () => void {
  return $effect.root(() => {
    $effect(() => {
      void theme.mode;
      const paper = getComputedStyle(document.documentElement).getPropertyValue("--paper").trim();
      if (!paper) return;
      for (const meta of document.querySelectorAll<HTMLMetaElement>('meta[name="theme-color"]')) meta.content = paper;
    });
  });
}
