// The tiny fixture session as the host ports see it, per arch: the reference backend over the
// generated weights, the hand-built lenses, the corpus-count token mask and the neutral means
// (unit-stats golden), mirroring drowse's `tests/support/browser_units.py::fixture`.
import { archGolden, decode, golden } from "./units-goldens.mjs";

export const DROPPED_SYMBOLS = "QXZqxz#$%^`|";

export async function unitsFixture(server, arch, options = {}) {
  const [tiny, cpu, context, profiles] = await Promise.all([
    server.ssrLoadModule("/src/hosted/runtime/units/tinyModels.ts"),
    server.ssrLoadModule("/src/hosted/runtime/units/cpuBackend.ts"),
    server.ssrLoadModule("/src/hosted/runtime/units/referenceContext.ts"),
    server.ssrLoadModule("/src/hosted/runtime/structuredHookProfile.ts"),
  ]);
  const file = golden("units-tiny-models-v1.json");
  const record = file.arches[arch];
  const model = tiny.tinyModelWeights(file, arch);
  const tokenizer = tiny.tinyTokenizer(record);
  const backend = cpu.cpuBlockSitesBackend(model, {
    profile: profiles.STANDARD_STRUCTURED_HOOK_PROFILE_V4,
    contextTokens: 256,
  });
  const dropped = new Set([...DROPPED_SYMBOLS].map((symbol) => tokenizer.idOfByte(symbol.charCodeAt(0))));
  const keep = Uint8Array.from({ length: model.vocabSize }, (_, v) =>
    v < tokenizer.size && tokenizer.byteOf[v] >= 0 && !tokenizer.decode([v]).includes("�") && !dropped.has(v) ? 1 : 0);
  const stats = archGolden("unit-stats", arch);
  const means = { out: Float32Array.from(decode(stats.out_mean)), mid: Float32Array.from(decode(stats.mid_mean)) };
  const lenses = tiny.tinyLenses(file, arch);
  const selected = new Map([...lenses].filter(([key]) =>
    (options.families ?? ["R", "S"]).includes(key[0]) && !(options.drop ?? []).includes(key)));
  const readContext = context.referenceReadContext({
    backend,
    model,
    modelId: options.modelId ?? "<model>",
    band: { lo: file.band[0], hi: file.band[1] },
    lenses: selected,
    keep,
    mask: { source: "corpus", detail: "corpus counts over 1,000,000 tokens of units-tiny-fixture, kept at 1+" },
    means,
    decode: (id) => tokenizer.decode([id]),
  });
  return { file, record, model, tokenizer, backend, keep, means, lenses, context: readContext, stats };
}
