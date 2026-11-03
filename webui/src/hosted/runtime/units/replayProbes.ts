// The attached lens probes scored over a replay's block outputs (native
// `LensInstrument.score_probes({L: taps.out[L]})`, which the token anatomy's envelope carries):
// a `jlens/` probe's per-layer probability of its token through the R lens and the model's
// readout (`token_readout_rows`), an `slens/` probe's footprint z through the S lens
// (`_InputBlock.readings`).
import type { ProbeInfo, ScalarReadingJSON } from "../../../../../drowse/webui/src/lib/types";
import { f32Tensor, i32Tensor } from "../modelBackend";
import { roundReading } from "../measurementRows";
import { layerDepth, outputTransport, transport, type UnitsReadContext } from "./lensReads";
import { slensProbeRows } from "./slens";

export type LensProbe = Extract<ProbeInfo, { family: "lens" }>;

export const UNIT_MEAN_TOKEN_PROBABILITY = "mean_token_probability";
export const UNIT_MEAN_FOOTPRINT_Z = "mean_footprint_z";

/** Readings of `probes` over `out` (`[L, d]`, every layer's block output at one position): the
 *  families whose lens is resident score, the others are skipped (native reads only resident
 *  families). */
export async function lensProbeReadings(
  context: UnitsReadContext,
  out: Float32Array,
  probes: readonly LensProbe[],
): Promise<Record<string, ScalarReadingJSON>> {
  const readings: Record<string, ScalarReadingJSON> = {};
  const d = context.hiddenSize;
  const lens = context.r;
  if (lens !== null) {
    for (const probe of probes) {
      if (probe.namespace !== "jlens" || probe.token_id === null) continue;
      const layers = probe.layers.filter((layer) => layer >= 0 && layer < context.nLayers);
      if (layers.length === 0) continue;
      const p: number[] = [];
      for (const layer of layers) {
        const { matrix } = await outputTransport(context, lens, layer, "out");
        const row = out.slice(layer * d, (layer + 1) * d);
        const x = matrix === null ? row : await transport(context, row, matrix, true);
        const scored = await context.backend.vocabTokenLogits(f32Tensor(x, [1, d]), i32Tensor(Int32Array.of(probe.token_id), [1, 1]));
        p.push(Math.exp(scored.logits.data[0] - scored.logSumExp.data[0]));
      }
      const depths = layers.map((layer) => layerDepth(layer, context.nLayers));
      const mass = Math.max(p.reduce((sum, value) => sum + value, 0), 1e-12);
      const center = p.reduce((sum, value, i) => sum + value * depths[i], 0) / mass;
      const variance = p.reduce((sum, value, i) => sum + value * (depths[i] - center) ** 2, 0) / mass;
      readings[probe.name] = {
        value: roundReading(p.reduce((sum, value) => sum + value, 0) / p.length),
        unit: UNIT_MEAN_TOKEN_PROBABILITY,
        per_layer: Object.fromEntries(layers.map((layer, i) => [String(layer), roundReading(p[i])])),
        depth: { center: [roundReading(center)], spread: [roundReading(Math.sqrt(Math.max(variance, 0)))], basis: "readout_probability_mass" },
      };
    }
  }
  const slens = probes.filter((probe) => probe.namespace === "slens" && probe.token_id !== null);
  if (slens.length > 0 && context.s !== null && context.means.out !== null) {
    const scored = await slensReadings(context, out, slens.map((probe) => ({ name: probe.name, tokenId: probe.token_id! })));
    Object.assign(readings, scored.readings);
  }
  return readings;
}

/** `slens/` readings over `out` (`[L, d]`): per probe the footprint z through the S lens
 *  (native `_InputBlock.readings`, rounded as the wire rounds them) and its exact w_S-weighted
 *  mean z, the scalar a gate reads (`gate_scalars`). */
export async function slensReadings(
  context: UnitsReadContext,
  out: Float32Array,
  probes: readonly { readonly name: string; readonly tokenId: number }[],
): Promise<{ readings: Record<string, ScalarReadingJSON>; values: Record<string, number> }> {
  const readings: Record<string, ScalarReadingJSON> = {};
  const values: Record<string, number> = {};
  if (probes.length === 0) return { readings, values };
  const { layers, rows } = await slensProbeRows(context, out, Int32Array.from(probes, (probe) => probe.tokenId));
  probes.forEach((probe, i) => {
    // `[3 + c]`: the w_S-weighted mean z, depth center, depth spread, then z per S:out layer.
    const row = rows[i];
    values[probe.name] = row[0];
    readings[probe.name] = {
      value: roundReading(row[0]),
      unit: UNIT_MEAN_FOOTPRINT_Z,
      per_layer: Object.fromEntries(layers.map((layer, l) => [String(layer), roundReading(row[3 + l])])),
      depth: { center: [roundReading(row[1])], spread: [roundReading(row[2])], basis: "footprint_z" },
    };
  });
  return { readings, values };
}
