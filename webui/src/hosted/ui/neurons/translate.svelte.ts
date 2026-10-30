// English for text in other languages, machine-translated by Claude Haiku 5.5 when the data is built
// (scripts/neurons/translate.py). Tokens and quoted terms come from terms.json, loaded on first need;
// whole contexts carry their translation in the detail files.
import { loadTerms, type Terms } from "./data";
import { DEFAULT_MODEL, type ModelId } from "./models";

const KEY = "drowse.neuronLens.translate";

export const translation = $state<{ on: boolean; terms: Terms | null; failed: boolean }>({ on: false, terms: null, failed: false });

// Each model has its own vocabulary, so its own terms file.
let model: ModelId = DEFAULT_MODEL;
let pending: Promise<void> | null = null;
export function ensureTerms(): Promise<void> {
  const asked = model;
  pending ??= loadTerms(asked)
    .then((terms) => { if (asked === model) { translation.terms = terms; translation.failed = false; } })
    .catch(() => { pending = null; translation.failed = true; });
  return pending;
}

export function setTermsModel(next: ModelId) {
  if (next === model) return;
  model = next;
  pending = null;
  translation.terms = null;
  if (translation.on) void ensureTerms();
}

/** The switch remembers its state per viewer; the page works the same without storage. */
export function setTranslate(on: boolean) {
  translation.on = on;
  try { localStorage.setItem(KEY, on ? "1" : "0"); } catch { /* storage blocked */ }
  if (on) void ensureTerms();
}

export function restoreTranslate() {
  try { if (localStorage.getItem(KEY) === "1") setTranslate(true); } catch { /* storage blocked */ }
}

/** Letters outside the Latin script: worth a lookup even before the terms have loaded. */
export const FOREIGN = /[^\P{L}\p{Script=Latin}]/u;
/** Runs of such letters (with their combining marks) inside running text, such as a label's quoted examples. */
export const FOREIGN_RUN = /(?:[^\P{L}\p{Script=Latin}]\p{M}*)+/gu;

export const termFor = (text: string) => translation.terms?.[text.trim()];

/** Whether a token or term may have a translation: non-Latin letters, or a known entry once loaded. */
export const maybeForeign = (text: string) => FOREIGN.test(text) || termFor(text) !== undefined;

/** The English to show inline in place of a term, short enough for a chip. */
export function inlineEnglish(text: string): string {
  if (!translation.on) return "";
  const entry = termFor(text);
  if (!entry?.[0]) return "";
  const english = entry[0];
  return english.length > 28 ? `${english.slice(0, 27)}…` : english;
}

let names: Intl.DisplayNames | null = null;
export function languageName(code: string): string {
  try {
    names ??= new Intl.DisplayNames(["en"], { type: "language" });
    return (code && names.of(code)) || code;
  } catch {
    return code;
  }
}
