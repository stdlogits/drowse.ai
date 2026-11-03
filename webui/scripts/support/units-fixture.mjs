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

/** `wᵀ P w` over a `[d, d]` projector (the golden stores the null basis's projector). */
export function projectorEnergy(projector, d) {
  return async (w) => {
    let quad = 0;
    for (let i = 0; i < d; i += 1) {
      let row = 0;
      for (let j = 0; j < d; j += 1) row += projector[i * d + j] * w[j];
      quad += w[i] * row;
    }
    return quad;
  };
}

/** The fixture as B6's packs would serve it (`UnitsPackSource`): unit-stats, unit-artifacts,
 *  slens statistics, the hand-built lenses and the exported fixture whitener. */
export function unitsPackSource(fx, arch, whiteners, options = {}) {
  const stats = fx.stats;
  const artifacts = archGolden("unit-artifacts", arch);
  const slens = archGolden("slens", arch);
  const tables = new Map(artifacts.layers.map((entry) => [
    entry.layer,
    new Map(Object.entries(entry.tensors).map(([key, value]) => [key, Float32Array.from(decode(value))])),
  ]));
  const mask = new Uint32Array(Math.ceil(fx.model.vocabSize / 32));
  for (const v of slens.mask.kept) mask[v >>> 5] |= 1 << (v & 31);
  const lens = (family) => {
    const sites = ["out", "mid"].filter((site) => fx.lenses.has(`${family}:${site}`) && !(options.drop ?? []).includes(`${family}:${site}`));
    if (sites.length === 0) return null;
    return {
      source: `local:${family.toLowerCase()}`,
      identity: `units-tiny-${arch}-${family.toLowerCase()}`,
      sites,
      layers: (site) => [...(fx.lenses.get(`${family}:${site}`)?.keys() ?? [])].sort((a, b) => a - b),
      fetch: async (site, layer) => fx.lenses.get(`${family}:${site}`).get(layer),
    };
  };
  return {
    band: { lo: fx.file.band[0], hi: fx.file.band[1] },
    statistics: async () => ({
      statistics: {
        documents: stats.n_documents,
        neuronMean: Float32Array.from(decode(stats.neuron_mean)),
        neuronScale: Float32Array.from(decode(stats.neuron_scale)),
        headMean: Float32Array.from(decode(stats.head_mean)),
        headScale: Float32Array.from(decode(stats.head_scale)),
      },
      absent: null,
    }),
    means: async () => fx.means,
    lens,
    embeddingStatistics: async () => ({
      meanE: Float32Array.from(decode(slens.mean_E)),
      sigmaE: Float32Array.from(decode(slens.sigma_E)),
      mask,
      source: slens.mask.source,
      detail: slens.mask.detail,
      maskDigest: "tokenizer",
    }),
    layerArtifacts: async (layer) => tables.get(layer),
    nullProjection: projectorEnergy(Float32Array.from(decode(artifacts.null.projector)), fx.model.hiddenSize),
    headKeptIds: async () => Int32Array.from(artifacts.head_kept_ids),
    whitener: async (layer) => {
      const entry = whiteners?.arches[arch].find((row) => row.layer === layer);
      return entry ? {
        columns: fx.model.hiddenSize, rank: entry.eigenvalues.length, ridge: entry.ridge,
        mean: Float64Array.from(entry.mean), basis: Float64Array.from(entry.basis.flat()),
        eigenvalues: Float64Array.from(entry.eigenvalues), inverseScales: new Float64Array(entry.eigenvalues.length),
      } : null;
    },
  };
}
