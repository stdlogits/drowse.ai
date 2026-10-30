// Data for the Neuron Lens explorer. The files are prebuilt by scripts/neurons/build-neuron-catalog.py
// from the neuron-lens catalog and served as static JSON next to the page.

export const DATA_BASE = "/neurons-data";
export const LAYERS = 36;
export const MLP = 12288;
export const HEADS = 32;
export const TOTAL_NEURONS = LAYERS * MLP;
export const TOTAL_HEADS = LAYERS * HEADS;

export type Unit = "neurons" | "heads";
export type Band = "embed" | "early" | "crossover" | "concept" | "dark" | "predict" | "suppress";
export type Trust = "primary" | "trusted" | "weak";
export type NeuronColumn =
  | "read_S" | "read_S_z" | "read_R" | "read_R_z"
  | "write_S" | "write_S_z" | "write_R" | "write_R_z";
export type HeadColumn =
  | "ov_R" | "ov_R_z" | "ov_R_neg" | "ov_R_neg_z" | "ov_S" | "ov_S_z" | "ov_S_neg" | "ov_S_neg_z"
  | "qk_S" | "qk_S_z" | "qk_S_neg" | "qk_S_neg_z" | "qk_self_S" | "qk_self_S_z";
export type ReadColumn = NeuronColumn | HeadColumn;

/** [token, variants merged into it, flags: 1 another lens names it too, 2 confirmed by measurement]. */
export type LensToken = [string, number, number];
export interface ReadInfo {
  col: ReadColumn;
  trust: Trust;
  /** True when this is the raw or z twin of a read listed above it: only its extra tokens are listed. */
  twin: boolean;
}
export interface LensRead extends ReadInfo {
  /** Mean top-8 score as a percentile of the layer's poles (of all heads, for heads). */
  pct: number;
  tokens: LensToken[];
}
/** A top firing: the text with the token marked ⟦ ⟧ (and, for heads, the attended token ⟪ ⟫), then its English and language when translated. */
export type Context = [text: string, en?: string, lang?: string];

export type NeuronSide = "read_S" | "read_R" | "write_S" | "write_R";
export interface NeuronLabel { fires: string; fc: number; writes: string; wc: number; note: string }
export interface Pole {
  pole: 1 | -1;
  reads: LensRead[];
  /** Variant keys two lens families name. */
  shared: string[];
  /** Variant keys a lens names and the pole's drivers or peaks confirm. */
  measured: string[];
  label: NeuronLabel;
  /** Typical and largest firing, the share of tokens it fires on (overall, then web, Chinese, code, chat), and the share of firings past position 0. */
  act: { top: number; max: number; rate: number; slices: [number, number, number, number]; pos: number };
  peaks: [string, number][];
  contexts: Context[];
  drivers: [string, number][];
  /** Each side's q-value at the pole's own top firings: under 0.05, the read holds. */
  legible: Record<NeuronSide, number>;
  silent: boolean;
  dark: boolean;
  detector: "token" | "next" | null;
  shape?: { kurtPct: number; skew: number; kind: "predicts" | "suppresses" | null };
  score: number;
}
export interface NeuronDetail { n: number; poles: [Pole, Pole] }

export interface HeadLabel { fires: string; fc: number; attends: string; ac: number; writes: string; wc: number; note: string }
export interface HeadDetail {
  h: number;
  kv: number;
  label: HeadLabel;
  /** Attention to position 0, the previous token and itself; induction and duplicate-token scores on repeated text. */
  pattern: [number, number, number, number, number];
  roles: number;
  act: { top: number; max: number; rate: number; slices: [number, number, number, number] };
  copy: { ovR: number; ovS: number; self: number; qkOv: number; qkCtx: number };
  reads: LensRead[];
  /** What its firings write in context (the R-lens read of its actual output, layer-contrastive). */
  outRz: LensToken[];
  /** As the query: tokens that make its output largest. */
  drivers: [string, number][];
  /** As the key: tokens it attends to most. */
  keys: [string, number][];
  peaks: [string, number][];
  sources: [string, number][];
  contexts: Context[];
  legible: { ov_R: number; ov_copy: number; ov_S: number; qk_S: number };
  /** Share of top firings on one query token, and on one attended token. */
  modes: [number, number];
  dark: boolean;
  score: number;
}

/** [layer, neuron, title of the + pole, title of the − pole, score, flags, each title's confidence 0–100]. */
export type IndexRow = [number, number, string, string, number, number, number, number];
export const FLAG_DETECTOR = 1;
export const FLAG_PREDICTS = 2;
export const FLAG_SUPPRESSES = 4;
/** The + (or −) title is the pole's writes label rather than its fires label. */
export const FLAG_PLUS_WRITES = 8;
export const FLAG_MINUS_WRITES = 16;
export const FLAG_DARK = 32;

/** [layer, head, kv head, fires, attends, writes, their confidences 0–100, roles, score, pattern]. */
export type HeadRow = [number, number, number, string, string, string, number, number, number, number, number, HeadDetail["pattern"]];

export interface LayerMeta {
  layer: number;
  band: Band;
  missing?: boolean;
  count?: number;
  stats?: { silent: number; detector: number; legible: number; dark: number };
}

export interface CatalogIndex {
  model: string;
  layers: number;
  neuronsPerLayer: number;
  headsPerLayer: number;
  chunk: number;
  layerMeta: LayerMeta[];
  /** The easiest neurons to read, by legibility and label confidence. */
  neurons: IndexRow[];
}

/** Word -> [ref, side, weight]: side 0/1 is a curated neuron's + or − pole (ref indexes CatalogIndex.neurons), 2 a head (ref = layer × 32 + head). */
export interface SearchFile { forms: Record<string, [number, 0 | 1 | 2, number][]>; display: Record<string, string> }
/** Token or quoted term -> [English, romanization, ISO 639-1 language]. */
export type Terms = Record<string, [string, string, string]>;

// ---------- loading

const cache = new Map<string, Promise<unknown>>();

function memo<T>(key: string, make: () => Promise<T>): Promise<T> {
  let pending = cache.get(key) as Promise<T> | undefined;
  if (!pending) {
    pending = make();
    // A failed request can be retried later instead of failing forever.
    pending.catch(() => cache.delete(key));
    cache.set(key, pending);
  }
  return pending;
}

const load = <T>(name: string) => memo(name, () => fetch(`${DATA_BASE}/${name}`).then((response) => {
  if (!response.ok) throw new Error(`${name}: ${response.status}`);
  return response.json() as Promise<T>;
}));

const pad = (value: number, width = 2) => String(value).padStart(width, "0");

type RawToken = number | [number, number, number];
type RawRead = [number, RawToken[]];
type RawPairs = [number, number][];
type RawPole = Omit<Pole, "reads" | "drivers"> & { reads: RawRead[]; drivers: RawPairs };
type RawHead = Omit<HeadDetail, "reads" | "drivers" | "keys" | "outRz"> & { reads: RawRead[]; drivers: RawPairs; keys: RawPairs; outRz: RawToken[] };

// Details name tokens by id; vocab.json decodes every id once.
const loadVocab = () => load<string[]>("vocab.json");
const tokensOf = (raw: RawToken[], vocab: string[]): LensToken[] =>
  raw.map((t) => (typeof t === "number" ? [vocab[t] ?? "", 1, 0] : [vocab[t[0]] ?? "", t[1], t[2]]));
const readsOf = (header: ReadInfo[], raw: RawRead[], vocab: string[]): LensRead[] =>
  header.map((info, i) => ({ ...info, pct: raw[i]?.[0] ?? 0, tokens: tokensOf(raw[i]?.[1] ?? [], vocab) }));
const pairsOf = (raw: RawPairs, vocab: string[]): [string, number][] => raw.map(([t, value]) => [vocab[t] ?? "", value]);

export const loadIndex = () => load<CatalogIndex>("index.json");
export const loadSearch = () => load<SearchFile>("search.json");
export const loadTerms = () => load<Terms>("terms.json");
/** Every neuron of a layer: [neuron, title +, title −, score, flags, confidence +, confidence −]. */
export const loadLayerIndex = (layer: number) => memo(`rows:${layer}`, async () => {
  const file = await load<{ layer: number; neurons: [number, string, string, number, number, number, number][] }>(`neurons/L${pad(layer)}/index.json`);
  return file.neurons.map((row): IndexRow => [layer, ...row]);
});
/** Per layer, a base64 string of 12,288 bytes: each neuron's readability, 0–255. */
export const loadScores = () => memo("scores", async () => (await load<{ layers: string[] }>("scores.json")).layers.map(decodeScores));
export const loadHeadIndex = () => memo("heads", async () => (await load<{ heads: HeadRow[] }>("heads/index.json")).heads);

export const chunkOf = (neuron: number, size: number) => Math.floor(neuron / size);
export const loadNeuronChunk = (layer: number, chunk: number) => memo(`chunk:${layer}:${chunk}`, async () => {
  const [file, vocab] = await Promise.all([
    load<{ layer: number; reads: ReadInfo[]; neurons: { n: number; poles: [RawPole, RawPole] }[] }>(`neurons/L${pad(layer)}/c${pad(chunk, 3)}.json`),
    loadVocab(),
  ]);
  return file.neurons.map((neuron): NeuronDetail => ({
    n: neuron.n,
    poles: neuron.poles.map((pole) => ({ ...pole, reads: readsOf(file.reads, pole.reads, vocab), drivers: pairsOf(pole.drivers, vocab) })) as [Pole, Pole],
  }));
});
export const loadHeadLayer = (layer: number) => memo(`heads:${layer}`, async () => {
  const [file, vocab] = await Promise.all([load<{ layer: number; reads: ReadInfo[]; heads: RawHead[] }>(`heads/L${pad(layer)}.json`), loadVocab()]);
  return file.heads.map((head): HeadDetail => ({
    ...head,
    reads: readsOf(file.reads, head.reads, vocab),
    outRz: tokensOf(head.outRz, vocab),
    drivers: pairsOf(head.drivers, vocab),
    keys: pairsOf(head.keys, vocab),
  }));
});

function decodeScores(encoded: string): Uint8Array {
  const binary = atob(encoded);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) out[i] = binary.charCodeAt(i);
  return out;
}

// ---------- depth bands: what the catalog's checks say to trust at each depth (neuron-lens findings §6, §7)

export interface BandInfo { name: string; range: string; summary: string; reading: string }

export const BANDS: Record<Band, BandInfo> = {
  embed: {
    name: "Token layer", range: "L0",
    summary: "Neurons here mostly respond to one literal input token.",
    reading: "The input-token read (in·S) holds for 78% of poles at their own top firings. Writes are weak this early.",
  },
  early: {
    name: "Early layers", range: "L1–L7",
    summary: "Neurons respond to input tokens a few positions back, and often write back what they detect.",
    reading: "Trust the input-token reads (in·S). From about L4, the write read (out·R) predicts what pushing the neuron does.",
  },
  crossover: {
    name: "Crossover", range: "L8",
    summary: "The layer where the input-token reads and the concept reads both carry signal.",
    reading: "Both lenses read both sides here, and agree most. Tokens they share are the strongest evidence.",
  },
  concept: {
    name: "Concept layers", range: "L9–L12",
    summary: "Neurons respond to concepts more than to literal tokens.",
    reading: "The concept read (in·R) takes over from in·S, which has gone to noise. out·R predicts steering, rarely the next word.",
  },
  dark: {
    name: "Middle layers", range: "L13–L20",
    summary: "82% of poles here pass no test of a weight-only read. They respond to context.",
    reading: "Read the examples and the words that set it off in real text first. Lens reads are only a cross-check here.",
  },
  predict: {
    name: "Prediction layers", range: "L21–L31",
    summary: "Neurons fire just before a predictable word, and many write that word.",
    reading: "in·R is often the word about to come next. From about L23 the write read names the next word.",
  },
  suppress: {
    name: "Suppression layers", range: "L32–L35",
    summary: "Many neurons push down the word the model is about to say, to calibrate it.",
    reading: "Read both sides' writes: a side whose top tokens are the next word holds it back.",
  },
};

/** Shown beside layers 13–20, where every weight-only lens is at its weakest (findings §7, §12). */
export const MIDDLE_CAUTION = "Weight-only reads are least reliable at layers 13–20: 82% of neuron poles and three quarters of heads here pass none of the tests at their own top firings. Most respond to context built up over many words rather than to one word. Trust the examples first.";

export const BAND_ORDER: Band[] = ["embed", "early", "crossover", "concept", "dark", "predict", "suppress"];

export function bandOf(layer: number): Band {
  if (layer === 0) return "embed";
  if (layer <= 7) return "early";
  if (layer === 8) return "crossover";
  if (layer <= 12) return "concept";
  if (layer <= 20) return "dark";
  if (layer <= 31) return "predict";
  return "suppress";
}

/** Heads by depth (findings §11, §12). */
export function headBandNote(layer: number): string {
  if (layer <= 8) return "Early heads often attend by token identity or position. Their writes read best through the S-lens (OV·S).";
  if (layer <= 20) return "Three quarters of mid-network heads pass no test of a weight-only read. Where a head looks says little about what it writes.";
  if (layer <= 27) return "Late heads start writing the token that follows what they attend to.";
  return "27% of these heads write the next token as their OV read predicts, and many attend to the class they copy or suppress.";
}

// ---------- lenses

export interface LensInfo { short: string; name: string; side: "input" | "output"; describe: string }

export const LENSES: Record<ReadColumn, LensInfo> = {
  read_S: { short: "in·S", name: "Words that trigger it", side: "input", describe: "Input tokens, here or a few positions back, whose presence drives the neuron (the S-lens on its read direction)." },
  read_S_z: { short: "in·S z", name: "Distinctive words that trigger it", side: "input", describe: "The same, scored against the layer's other neurons so that common tokens drop out." },
  read_R: { short: "in·R", name: "Ideas that trigger it", side: "input", describe: "What the model is getting ready to say when this neuron is driven: the concept it responds to (the R-lens on its read direction)." },
  read_R_z: { short: "in·R z", name: "Distinctive ideas that trigger it", side: "input", describe: "The same, scored against the layer's other neurons so that common tokens drop out." },
  write_S: { short: "out·S", name: "Words it acts like", side: "output", describe: "Its push looks to later layers as if these input tokens were present." },
  write_S_z: { short: "out·S z", name: "Distinctive words it acts like", side: "output", describe: "The same, scored against the layer's other neurons." },
  write_R: { short: "out·R", name: "Pushes the model to say", side: "output", describe: "What its push makes the model say (the R-lens on its write)." },
  write_R_z: { short: "out·R z", name: "Distinctive words it pushes the model to say", side: "output", describe: "What's specific to this neuron's push, with tokens every neuron promotes removed." },
  ov_R: { short: "OV·R", name: "Copies", side: "output", describe: "Reading a token, what the head makes the model say: each token's place in its own column of the OV map." },
  ov_R_z: { short: "OV·R z", name: "Distinctively copies", side: "output", describe: "The same, scored against the layer's other heads." },
  ov_R_neg: { short: "OV·R −", name: "Suppresses", side: "output", describe: "Reading a token, what the head pushes the model away from saying." },
  ov_R_neg_z: { short: "OV·R − z", name: "Distinctively suppresses", side: "output", describe: "The same, scored against the layer's other heads." },
  ov_S: { short: "OV·S", name: "Copies the footprint of", side: "output", describe: "Reading a token, the head's write looks to later layers as if that token were there." },
  ov_S_z: { short: "OV·S z", name: "Distinctively copies the footprint of", side: "output", describe: "The same, scored against the layer's other heads." },
  ov_S_neg: { short: "OV·S −", name: "Erases the footprint of", side: "output", describe: "Reading a token, the head's write removes that token's footprint." },
  ov_S_neg_z: { short: "OV·S − z", name: "Distinctively erases", side: "output", describe: "The same, scored against the layer's other heads." },
  qk_S: { short: "QK·S", name: "Attends to", side: "input", describe: "Key tokens its average query favours, whatever the query." },
  qk_S_z: { short: "QK·S z", name: "Distinctively attends to", side: "input", describe: "The same, scored against the layer's other heads." },
  qk_S_neg: { short: "QK·S −", name: "Avoids", side: "input", describe: "Key tokens its average query looks away from." },
  qk_S_neg_z: { short: "QK·S − z", name: "Distinctively avoids", side: "input", describe: "The same, scored against the layer's other heads." },
  qk_self_S: { short: "QK self", name: "Seeks itself", side: "input", describe: "Tokens that, as the query, favour an earlier copy of themselves: what duplicate-token heads do." },
  qk_self_S_z: { short: "QK self z", name: "Distinctively seeks itself", side: "input", describe: "The same, scored against the layer's other heads." },
};

/** How to phrase a neuron's output side at this depth: steering before L21, prediction after. */
export function writeVerb(layer: number): string {
  if (layer < 4) return "Writes (weak this early)";
  if (layer < 21) return "When pushed, it promotes";
  if (layer < 32) return "It predicts the next word";
  return "It predicts or holds back the next word";
}

// ---------- heads

export const ROLE_INDUCTION = 1;
export const ROLE_DUPLICATE = 2;
export const ROLE_PREVIOUS = 4;
export const ROLE_SELF = 8;
export const ROLE_SINK = 16;
export const ROLE_NEXT = 32;
export const ROLE_COPY = 64;
export const ROLE_DARK = 128;

export interface RoleInfo { bit: number; name: string; short: string; color: string; describe: string }

/** In order of precedence: a head is drawn in the colour of its first role. */
export const ROLES: RoleInfo[] = [
  { bit: ROLE_INDUCTION, name: "Induction head", short: "Induction", color: "var(--accent)", describe: "Continues repeated text: after A B … A, it attends to B and predicts it (score past 0.3)." },
  { bit: ROLE_DUPLICATE, name: "Duplicate-token head", short: "Duplicate", color: "var(--accent-amber)", describe: "Attends to earlier copies of the current token (score past 0.3). It copies little, likely marking that the token was seen." },
  { bit: ROLE_PREVIOUS, name: "Previous-token head", short: "Previous", color: "var(--accent-red)", describe: "Attends mostly to the token just before (past 0.5)." },
  { bit: ROLE_SELF, name: "Self-attending head", short: "Self", color: "var(--accent-red)", describe: "Attends mostly to its own position (past 0.5)." },
  { bit: ROLE_NEXT, name: "Next-token writer", short: "Next token", color: "var(--accent-green)", describe: "At its own top firings, the attended token's OV column promotes the token that follows (q < 0.05)." },
  { bit: ROLE_COPY, name: "Copier", short: "Copies", color: "var(--neuron-late)", describe: "At its own top firings, it promotes the token it attends to (q < 0.05)." },
  { bit: ROLE_SINK, name: "Sink head", short: "Sink", color: "var(--fg-muted)", describe: "Puts over 90% of its attention on the first position, a resting place: it acts where it leaves the sink." },
];

export const rolesOf = (bits: number) => ROLES.filter((role) => (bits & role.bit) !== 0);
export const primaryRole = (bits: number) => ROLES.find((role) => (bits & role.bit) !== 0) ?? null;
export const PATTERN_NAMES = ["Sink", "Previous", "Self", "Induction", "Duplicate"] as const;
/** The thresholds the catalog names a pattern at. */
export const PATTERN_THRESHOLDS = [0.9, 0.5, 0.5, 0.3, 0.3] as const;

// ---------- ids

export const neuronId = (layer: number, neuron: number) => `L${layer}·n${neuron}`;
export const headId = (layer: number, head: number) => `L${layer}·h${head}`;

export type UnitRef = { unit: "neurons"; layer: number; index: number } | { unit: "heads"; layer: number; index: number };

/** "L8 344", "l8/n344", "8.344", "L8·n344" -> a neuron; "L20h17", "L20 h17", "20.h17" -> a head. */
export function parseRef(query: string): UnitRef | null {
  const text = query.trim();
  const head = text.match(/^l?\s*(\d{1,2})\s*[\s./·:,_-]?\s*h\s*(\d{1,2})$/i);
  if (head) {
    const layer = Number(head[1]);
    const index = Number(head[2]);
    return layer < LAYERS && index < HEADS ? { unit: "heads", layer, index } : null;
  }
  const neuron = text.match(/^l?\s*(\d{1,2})\s*[\s./·:,_-]\s*n?\s*(\d{1,5})$/i);
  if (!neuron) return null;
  const layer = Number(neuron[1]);
  const index = Number(neuron[2]);
  return layer < LAYERS && index < MLP ? { unit: "neurons", layer, index } : null;
}

// ---------- tokens and text

export interface TokenParts { lead: boolean; text: string; trail: boolean; byte: boolean; empty: boolean }

/** Splits a token for display: a leading space marks a word start, byte pieces are marked, whitespace is made visible. */
export function tokenParts(raw: string): TokenParts {
  if (raw.startsWith("\u0001")) return { lead: false, text: raw.slice(1), trail: false, byte: true, empty: false };
  const lead = raw.startsWith(" ");
  const trail = raw.length > 1 && raw.endsWith(" ") && raw.trim() !== "";
  let text = raw.trim() === "" ? raw : raw.slice(lead ? 1 : 0, trail ? -1 : undefined);
  text = text.replace(/\n/g, "⏎").replace(/\r/g, "␍").replace(/\t/g, "⇥");
  if (text.trim() === "" && text.length) text = text.replace(/ /g, "␣");
  return { lead, text, trail, byte: false, empty: text === "" };
}

/** Variant key, matching the builder's stem(): case, leading punctuation and common English suffixes removed. */
export function stem(token: string): string {
  if (token.startsWith("\u0001")) return token;
  let key = token.trim().toLowerCase().replace(/^[_\-.]+/, "").replace(/^[/([]+/, "");
  for (const suffix of ["ing", "ed", "es", "s", "ly"]) {
    if (key.endsWith(suffix) && key.length - suffix.length >= 4) { key = key.slice(0, -suffix.length); break; }
  }
  return key.endsWith("e") && key.length > 4 ? key.slice(0, -1) : key;
}

/** A token worth showing as a word: not a byte piece, whitespace or bare punctuation. */
export const wordy = (token: string) => !token.startsWith("\u0001") && /[\p{L}\p{N}]/u.test(token);

/** The first few tokens of a list for a summary: words first, or the raw tokens when a unit reads only punctuation. */
export function headline<T extends [string, ...unknown[]]>(list: T[], count = 5): T[] {
  const words = list.filter(([token]) => wordy(token));
  return (words.length ? words : list.filter(([token]) => !token.startsWith("\u0001"))).slice(0, count);
}

export interface Segment { text: string; mark: "hit" | "src" | null }

/** A context split at its markers: ⟦ ⟧ the token where the unit fired, ⟪ ⟫ the token a head attended to. */
export function segments(text: string): Segment[] {
  const out: Segment[] = [];
  let last = 0;
  for (const match of text.matchAll(/⟦([^⟧]*)⟧|⟪([^⟫]*)⟫/g)) {
    const at = match.index ?? 0;
    if (at > last) out.push({ text: text.slice(last, at), mark: null });
    out.push({ text: match[1] ?? match[2] ?? "", mark: match[1] !== undefined ? "hit" : "src" });
    last = at + match[0].length;
  }
  if (last < text.length) out.push({ text: text.slice(last), mark: null });
  return out;
}

/** Layer colour on the violet → cyan depth ramp, as used in the neuron-lens animation. */
export function depthColor(layer: number): string {
  return `color-mix(in oklch, var(--neuron-deep) ${Math.round(100 - (layer / 35) * 100)}%, var(--neuron-late))`;
}

export function percent(value: number | undefined, digits = 0): string {
  if (value == null || !Number.isFinite(value)) return "–";
  const pct = value * 100;
  if (pct === 0) return "0%";
  if (pct < 0.001) return "<0.001%";
  // Firing rates are often hundredths of a percent: keep two significant figures there.
  if (pct < 1) return `${Number(pct.toPrecision(2))}%`;
  return `${pct.toFixed(digits)}%`;
}

/** A Haiku confidence, 0–1, in words. */
export function confidenceWord(value: number): string {
  return value >= 0.7 ? "Confident" : value >= 0.45 ? "Fairly sure" : value >= 0.25 ? "Tentative" : "A guess";
}

// ---------- the layer strip: cards for neurons or heads

export interface StripCard {
  key: number;
  id: string;
  /** Up to two labelled lines: a mark (+, −, or a verb) and Haiku's title, with its confidence 0–1. */
  lines: { mark: string; markLabel: string; text: string; conf: number }[];
  tags: { label: string; color?: string }[];
  flags: number;
}
export interface StripFilter { value: string; label: string; test?: (card: StripCard) => boolean; limit?: number }
