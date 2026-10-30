// Gemma 4's image and sound collections (neuron-lens media_sets.py; findings §14), by the source name
// the catalog's documents carry.

export const SOURCE_NAMES: Record<string, string> = {
  imagenet: "ImageNet", coco: "COCO", places365: "Places365", sun397: "SUN397", cityscapes: "Cityscapes",
  food101: "Food-101", pets: "Oxford Pets", flowers: "Oxford Flowers", cars: "Stanford Cars", aircraft: "FGVC Aircraft",
  dtd: "Describable Textures", quickdraw: "Quick, Draw!", eurosat: "EuroSAT", xray: "Chest X-rays", mnist: "MNIST",
  rico: "RICO app screens", flickr8k: "Flickr8k", wikiart: "WikiArt", docvqa: "DocVQA", chartqa: "ChartQA",
  caltech101: "Caltech-101", anime: "Anime faces", cartoon: "Cartoons", sprite: "Pixel sprites", logo: "Logos",
  emoji: "Emoji", newyorker: "New Yorker cartoons", actions: "Human actions", textvqa: "TextVQA", iam: "IAM handwriting",
  letters: "Handwritten letters", im2latex: "Typeset equations", ai2d: "AI2D diagrams", websight: "WebSight pages",
  pubtabnet: "PubTabNet tables", aigen: "AI-generated images", birds: "CUB birds", signs: "Traffic signs",
  products: "Fashion products", fashionmnist: "Fashion-MNIST", skin: "Skin lesions", histology: "PatchCamelyon",
  galaxy: "Galaxy10", aerial: "UC Merced aerial", underwater: "Aquarium", slides: "SlideVQA", expressions: "Facial expressions",
  celeba: "CelebA", synth: "Synthetic probe", manga: "Manga109",
  audioset: "AudioSet", fsd50k: "FSD50K", esc50: "ESC-50", clotho: "Clotho", fleurs: "FLEURS", edacc: "EdAcc",
  voxpopuli: "VoxPopuli", expresso: "Expresso", genre: "Music genres", instrument: "Instrument timbre", mls: "Multilingual LibriSpeech",
};

export const sourceName = (source: string) => SOURCE_NAMES[source] ?? source;

/** A 40 ms frame (640 int16 samples, base64), tiled 64 times as the model saw it: 2.56 s at 16 kHz. */
export function frameSamples(wave: string): Float32Array {
  const bytes = Uint8Array.from(atob(wave), (c) => c.charCodeAt(0));
  const ints = new Int16Array(bytes.buffer);
  return Float32Array.from(ints, (v) => v / 32768);
}

// One sound at a time across the page: starting another stops the one playing.
let playing: { audio: HTMLAudioElement; stop: () => void } | null = null;
export function playOnly(url: string, onstop: () => void): () => void {
  playing?.stop();
  const audio = new Audio(url);
  const stop = () => {
    audio.pause();
    if (playing?.audio === audio) playing = null;
    onstop();
  };
  audio.onended = stop;
  playing = { audio, stop };
  void audio.play().catch(stop);
  return stop;
}
