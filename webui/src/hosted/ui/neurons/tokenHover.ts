// One listener for every token chip in the explorer: hovering a token lights up its variants in
// every read of the same unit, and a term from another language shows its translation, its
// romanization and its language. A translated context's badge shows the original text.
import { ensureTerms, languageName, termFor, translation } from "./translate.svelte";

const TARGET = ".tok, .term, .lang[data-original]";

export function tokenHover(root: HTMLElement) {
  const tip = document.createElement("div");
  tip.className = "term-tip";
  tip.setAttribute("role", "tooltip");
  tip.id = `term-tip-${Math.random().toString(36).slice(2, 8)}`;
  tip.hidden = true;
  document.body.append(tip);

  let linked: Element[] = [];
  let current: HTMLElement | null = null;

  function clearLinked() {
    for (const element of linked) element.classList.remove("linked");
    linked = [];
  }

  function hideTip() {
    tip.hidden = true;
    tip.classList.remove("shown");
    current?.removeAttribute("aria-describedby");
  }

  function line(className: string, text: string, lang?: string) {
    const span = document.createElement("span");
    span.className = className;
    span.textContent = text;
    if (lang) span.lang = lang;
    return span;
  }

  function fill(token: HTMLElement): boolean {
    const original = token.dataset.original;
    if (original) {
      tip.replaceChildren(
        line("tip-original", original, token.dataset.lang),
        line("tip-meta", `Translated from ${languageName(token.dataset.lang ?? "")} by Claude Haiku`),
      );
      return true;
    }
    const term = token.dataset.term;
    if (!term) return false;
    if (!translation.terms) void ensureTerms().then(() => { if (current === token) showTip(token); });
    const entry = termFor(term);
    const children = [line("tip-word", term, entry?.[2])];
    if (entry) {
      if (entry[1]) children.push(line("tip-roman", entry[1]));
      children.push(line("tip-meaning", entry[0]), line("tip-meta", `${languageName(entry[2])} · machine translated`));
    } else {
      children.push(line("tip-meta", translation.terms || translation.failed ? "No translation: likely a piece of a longer word." : "Loading translation…"));
    }
    tip.replaceChildren(...children);
    return true;
  }

  function showTip(token: HTMLElement) {
    if (!fill(token)) return hideTip();
    tip.hidden = false;
    const box = token.getBoundingClientRect();
    const width = Math.min(340, window.innerWidth - 24);
    tip.style.maxWidth = `${width}px`;
    const tipBox = tip.getBoundingClientRect();
    const left = Math.max(12, Math.min(window.innerWidth - tipBox.width - 12, box.left + box.width / 2 - tipBox.width / 2));
    const above = box.top - tipBox.height - 8;
    const top = above > 8 ? above : box.bottom + 8;
    tip.style.transform = `translate(${Math.round(left)}px, ${Math.round(top + window.scrollY)}px)`;
    token.setAttribute("aria-describedby", tip.id);
    requestAnimationFrame(() => tip.classList.add("shown"));
  }

  function enter(token: HTMLElement) {
    if (token === current) return;
    leave();
    current = token;
    const key = token.dataset.stem;
    const scope = token.closest("[data-link-scope]");
    if (key && scope) {
      linked = [...scope.querySelectorAll(`.tok[data-stem="${CSS.escape(key)}"]`)];
      if (linked.length > 1) for (const element of linked) element.classList.add("linked");
      else linked = [];
    }
    showTip(token);
  }

  function leave() {
    clearLinked();
    hideTip();
    current = null;
  }

  const over = (event: Event) => {
    const token = (event.target as Element | null)?.closest?.<HTMLElement>(TARGET);
    if (token && root.contains(token)) enter(token);
    else if (current && !current.contains(document.activeElement)) leave();
  };
  const out = (event: PointerEvent | FocusEvent) => {
    // Focus-driven scrolling also changes the element under a stationary pointer.
    // Those pointer events must not dismiss the keyboard user's focused glossary.
    if (event.type === "pointerout" && current?.contains(document.activeElement)) return;
    const next = event.relatedTarget as Element | null;
    // The pointer may move onto the gloss itself to read it (WCAG 1.4.13).
    if (next && tip.contains(next)) return;
    if (!next || !current?.contains(next)) {
      const nextToken = next?.closest?.<HTMLElement>(TARGET);
      if (!nextToken) leave();
    }
  };
  // Touch: a tap shows the gloss; tapping elsewhere hides it.
  const down = (event: PointerEvent) => {
    if (event.pointerType === "mouse" || tip.contains(event.target as Node)) return;
    const token = (event.target as Element | null)?.closest?.<HTMLElement>(TARGET);
    if (token && root.contains(token)) enter(token);
    else leave();
  };
  const key = (event: KeyboardEvent) => { if (event.key === "Escape") leave(); };
  const scroll = () => {
    if (!current) return;
    // Moving to a term can scroll it into view. Keep the glossary attached while
    // it still has focus or the pointer, and dismiss it when it leaves the viewport.
    const box = current.getBoundingClientRect();
    if ((current.contains(document.activeElement) || current.matches(":hover")) && box.bottom > 0 && box.top < window.innerHeight) showTip(current);
    else leave();
  };

  root.addEventListener("pointerover", over);
  root.addEventListener("pointerout", out);
  root.addEventListener("focusin", over);
  root.addEventListener("focusout", out);
  const tipLeave = (event: PointerEvent) => {
    const next = event.relatedTarget as Element | null;
    if (!next || !current?.contains(next)) leave();
  };
  tip.addEventListener("pointerleave", tipLeave);
  document.addEventListener("pointerdown", down);
  document.addEventListener("keydown", key);
  window.addEventListener("scroll", scroll, { passive: true });

  return {
    destroy() {
      root.removeEventListener("pointerover", over);
      root.removeEventListener("pointerout", out);
      root.removeEventListener("focusin", over);
      root.removeEventListener("focusout", out);
      tip.removeEventListener("pointerleave", tipLeave);
      document.removeEventListener("pointerdown", down);
      document.removeEventListener("keydown", key);
      window.removeEventListener("scroll", scroll);
      tip.remove();
    },
  };
}

