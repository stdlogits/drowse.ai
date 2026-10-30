// The models the Neuron Lens explorer covers: their shape, and what the neuron-lens findings say to
// trust at each depth (Qwen3-8B: findings §6, §7, §11, §12; Gemma 4 12B: §13-§16). The builder writes
// each layer's band key into index.json; the words for each band live here.

export type ModelId = "qwen3-8b" | "gemma-4-12b";
export type Modality = "image" | "audio";

export interface BandInfo {
  key: string;
  first: number;
  last: number;
  name: string;
  summary: string;
  reading: string;
  /** The band where weight-only reads are weakest: examples and exact drivers lead there. */
  dark?: boolean;
}

export interface ModelInfo {
  id: ModelId;
  name: string;
  lede: string;
  layers: number;
  mlp: number;
  heads: number;
  bands: BandInfo[];
  /** Shown beside the dark band's units. */
  caution: string;
  cautionLabel: string;
  /** By depth: (last layer, note on reading its heads). */
  headNotes: [number, string][];
  /** By depth: (last layer, the heading over a neuron's output read). */
  writeVerbs: [number, string][];
  /** The head tests' statistic: a false-discovery q, or (with too few heads a layer for one) an uncorrected p. */
  headTest: "q" | "p";
  /** Every sixth layer of Gemma 4 attends globally; the others within a window. */
  globalLayers: number[];
  media: Modality[];
  examples: string[];
  start: { neurons: number; heads: number };
  /** What the blind checks of Haiku's labels found (neuron-lens findings §8, §12, §13), shown beneath them. */
  labelChecks: { neurons: string; heads: string };
  /** The two labels a head's card in the layer strip shows: the ones that held up when checked blind. */
  headCard: ["fires" | "attends" | "writes", "fires" | "attends" | "writes"];
}

export const MODELS: Record<ModelId, ModelInfo> = {
  "qwen3-8b": {
    id: "qwen3-8b",
    name: "Qwen3-8B",
    lede: "What the neurons and attention heads inside Qwen3-8B respond to and what they make the model say, read from their weights and from 15 million tokens of text.",
    layers: 36,
    mlp: 12288,
    heads: 32,
    bands: [
      {
        key: "embed", first: 0, last: 0, name: "Token layer",
        summary: "Neurons here mostly respond to one literal input token.",
        reading: "The input-token read (in·S) holds for 78% of poles at their own top firings. Writes are weak this early.",
      },
      {
        key: "early", first: 1, last: 7, name: "Early layers",
        summary: "Neurons respond to input tokens a few positions back, and often write back what they detect.",
        reading: "Trust the input-token reads (in·S). From about L4, the write read (out·R) predicts what pushing the neuron does.",
      },
      {
        key: "crossover", first: 8, last: 8, name: "Crossover",
        summary: "The layer where the input-token reads and the concept reads both carry signal.",
        reading: "Both lenses read both sides here, and agree most. Tokens they share are the strongest evidence.",
      },
      {
        key: "concept", first: 9, last: 12, name: "Concept layers",
        summary: "Neurons respond to concepts more than to literal tokens.",
        reading: "The concept read (in·R) takes over from in·S, which has gone to noise. out·R predicts steering, rarely the next word.",
      },
      {
        key: "dark", first: 13, last: 20, name: "Middle layers", dark: true,
        summary: "82% of poles here pass no test of a weight-only read. They respond to context.",
        reading: "Read the examples and the words that set it off in real text first. Lens reads are only a cross-check here.",
      },
      {
        key: "predict", first: 21, last: 31, name: "Prediction layers",
        summary: "Neurons fire just before a predictable word, and many write that word.",
        reading: "in·R is often the word about to come next. From about L23 the write read names the next word.",
      },
      {
        key: "suppress", first: 32, last: 35, name: "Suppression layers",
        summary: "Many neurons push down the word the model is about to say, to calibrate it.",
        reading: "Read both sides' writes: a side whose top tokens are the next word holds it back.",
      },
    ],
    caution: "Weight-only reads are least reliable at layers 13–20: 82% of neuron poles and three quarters of heads here pass none of the tests at their own top firings. Most respond to context built up over many words rather than to one word. Trust the examples first.",
    cautionLabel: "Why layers 13–20 are harder to read",
    headNotes: [
      [8, "Early heads often attend by token identity or position. Their writes read best through the S-lens (OV·S)."],
      [20, "Three quarters of mid-network heads pass no test of a weight-only read. Where a head looks says little about what it writes."],
      [27, "Late heads start writing the token that follows what they attend to."],
      [35, "27% of these heads write the next token as their OV read predicts, and many attend to the class they copy or suppress."],
    ],
    writeVerbs: [[3, "Writes (weak this early)"], [20, "When pushed, it promotes"], [31, "It predicts the next word"], [35, "It predicts or holds back the next word"]],
    headTest: "q",
    globalLayers: [],
    media: [],
    examples: ["owl", "Bible", "she", "money", "圣经", "L20h17"],
    start: { neurons: 8, heads: 20 },
    labelChecks: {
      neurons: "Checked blind on 100 poles, the firing labels pick out held-out top firings at AUROC 0.82 (another pole's label: 0.56), and a judge picks the write label out of four from steered text 61% of the time (chance 25%).",
      heads: "Checked blind on 100 heads, the firing labels pick out held-out top firings at AUROC 0.71 (another head's label: 0.51), and a judge picks the write label out of four from what the head writes in context 49% of the time (chance 25%).",
    },
    headCard: ["attends", "writes"],
  },
  "gemma-4-12b": {
    id: "gemma-4-12b",
    name: "Gemma 4 12B",
    lede: "What the neurons and attention heads inside Gemma 4 12B respond to in text, images and sound, and what they make the model say, read from their weights, 15 million tokens of text, 50,000 images and 47,000 sounds.",
    layers: 48,
    mlp: 15360,
    heads: 16,
    bands: [
      {
        key: "token", first: 0, last: 4, name: "Token layers",
        summary: "Neurons here mostly respond to one literal input token.",
        reading: "The input-token read (in·S) holds for a third to nearly all poles here. Writes are too weak to read this early.",
      },
      {
        key: "early", first: 5, last: 13, name: "Early layers",
        summary: "Neurons respond to tokens a few positions back. The input-token read fades by L9, then returns at L10–L13, where many neurons write back what they detect.",
        reading: "Trust in·S at L10–L13, and the write's footprint (out·S) at L11–L13. From L11 the write read (out·R) predicts what pushing the neuron does.",
      },
      {
        key: "middle", first: 14, last: 32, name: "Middle layers", dark: true,
        summary: "Most poles here pass no test of a weight-only read. They respond to context, and many to images: image concepts meet their names in text at these depths.",
        reading: "Read the examples and the words that set it off in real text first. out·R still predicts what pushing the neuron does, though rarely the next word.",
      },
      {
        key: "late", first: 33, last: 47, name: "Late layers",
        summary: "Many neurons write the word the model is about to say.",
        reading: "out·R names the next word for 21–37% of poles at L33–L40, fewer after. Input reads are weak here: trust the examples.",
      },
    ],
    caution: "Weight-only reads are least reliable at layers 14–32: 48–85% of neuron poles by layer and 70% of heads pass none of the tests at their own top firings. Most respond to context built up over many words rather than to one word. Trust the examples first.",
    cautionLabel: "Why layers 14–32 are harder to read",
    headNotes: [
      [13, "Early heads often attend by token identity or position: for two thirds of them, a third of their top firings share one query token. With 16 heads a layer, no OV read passes its test this early."],
      [32, "70% of mid-network heads pass no test of a weight-only read. Text reads images most at L23–L31, and zeroing L24–L35's heads costs an image's label most."],
      [47, "OV·R holds for about a fifth of these heads (uncorrected): they write the token that follows what they attend to. Heads at L35–L46 most often write the class of an image they read."],
    ],
    writeVerbs: [[10, "Writes (weak this early)"], [32, "When pushed, it promotes"], [47, "It predicts the next word"]],
    headTest: "p",
    globalLayers: [5, 11, 17, 23, 29, 35, 41, 47],
    media: ["image", "audio"],
    examples: ["panda", "mosque", "Arabic", "rain", "熊猫", "L28h6"],
    start: { neurons: 23, heads: 28 },
    labelChecks: {
      neurons: "Not yet checked blind on this model.",
      heads: "Checked blind on 80 heads, the firing labels pick out held-out top firings at AUROC 0.71 (another head's label: 0.47), but the write labels are at chance: a judge picks one out of four from what the head writes in context 22% of the time (chance 25%). Read the write labels as guesses.",
    },
    headCard: ["fires", "attends"],
  },
};

export const MODEL_ORDER: ModelId[] = ["qwen3-8b", "gemma-4-12b"];
export const DEFAULT_MODEL: ModelId = "qwen3-8b";
export const isModelId = (value: string | null): value is ModelId => value !== null && value in MODELS;

export const bandOf = (model: ModelInfo, layer: number): BandInfo => model.bands.find((band) => layer <= band.last) ?? model.bands[model.bands.length - 1];
const byDepth = (steps: [number, string][], layer: number) => (steps.find(([last]) => layer <= last) ?? steps[steps.length - 1])[1];
export const headBandNote = (model: ModelInfo, layer: number) => byDepth(model.headNotes, layer);
/** How to phrase a neuron's output side at this depth: steering early, prediction late. */
export const writeVerb = (model: ModelInfo, layer: number) => byDepth(model.writeVerbs, layer);
export const bandRange = (band: BandInfo) => (band.first === band.last ? `L${band.first}` : `L${band.first}–${band.last}`);

/** Layer colour on the violet → cyan depth ramp, as used in the neuron-lens animation. */
export function depthColor(model: ModelInfo, layer: number): string {
  return `color-mix(in oklch, var(--neuron-deep) ${Math.round(100 - (layer / (model.layers - 1)) * 100)}%, var(--neuron-late))`;
}

export const MODALITY_NAMES: Record<Modality, { noun: string; plural: string; title: string }> = {
  image: { noun: "image", plural: "images", title: "In images" },
  audio: { noun: "sound", plural: "sounds", title: "In sound" },
};
