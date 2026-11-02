// What the browser loom persists of a measurement envelope: drowse's
// `readout/measurements.py` projections (`token_row_measurements`,
// `summary_measurements`, `aggregate_measurements`), checked against
// `drowse/tests/fixtures/browser/measurement-envelope-v2.json` (`expectedRow`,
// `expectedNodeAggregate`). The loom reader's rules for what they produce are
// shared code (`lib/tokenRows.ts`: `tokenRowProblem`,
// `aggregateMeasurementsProblem`).
import type {
  MeasurementsEnvelopeJSON,
  ProbeReadingJSON,
  ScalarReadingJSON,
  TokenRowEnvelopeJSON,
  TokenRowMeasurementsJSON,
} from "../../../../drowse/webui/src/lib/types";
import { formatG } from "../../../../drowse/webui/src/lib/expression";
import { ROW_MEASUREMENTS_VERSION } from "../../../../drowse/webui/src/lib/tokenRows";

/** The measurement envelope version (drowse's `MEASUREMENTS_VERSION`). */
export const MEASUREMENTS_VERSION = ROW_MEASUREMENTS_VERSION;

/** `value` at 6 significant digits, as drowse's `round_reading` (`%g`: the exact
 *  binary value rounded to the nearest decimal, an exact tie to even). */
export function roundReading(value: number): number {
  return Number(formatG(value, 6));
}

type Instruments = MeasurementsEnvelopeJSON["instruments"];

/** A generated token row's form of its token envelope: the flat `scores` and
 *  each geometry reading's `fraction` and `coords`, rounded; no other channel. */
export function tokenRowMeasurements(envelope: MeasurementsEnvelopeJSON): TokenRowMeasurementsJSON {
  const instruments: TokenRowMeasurementsJSON["instruments"] = {};
  const geometry = envelope.instruments.geometry;
  if (geometry !== undefined) {
    instruments.geometry = {
      readings: Object.fromEntries(Object.entries(geometry.readings).map(([name, reading]) => [
        name,
        { fraction: roundReading(reading.fraction), coords: reading.coords.map(roundReading) },
      ])),
    };
  }
  return {
    version: envelope.version,
    scope: envelope.scope,
    provenance: envelope.provenance,
    instruments,
    ...(envelope.scores === undefined ? {} : { scores: roundedScores(envelope.scores) }),
  };
}

/** A generated row's `measurements`: the slim projection, or the summary with
 *  every trace when the call set `persist_per_layer_scores`. */
export function rowMeasurements(
  envelope: MeasurementsEnvelopeJSON,
  persistPerLayer: boolean,
): TokenRowEnvelopeJSON {
  return persistPerLayer
    ? summaryMeasurements(envelope, true)
    : tokenRowMeasurements(envelope);
}

/** An envelope's persisted summary: every reading whole but its traces (kept
 *  with `perLayer`), the lens readout's chips, the SAE features, the bindings
 *  and the scores, every float rounded. The inspector's live
 *  `subspace_coords_per_layer` trail is never kept. */
export function summaryMeasurements(
  envelope: MeasurementsEnvelopeJSON,
  perLayer: boolean,
): MeasurementsEnvelopeJSON {
  const source = envelope.instruments;
  const out: Instruments = {};
  if (source.geometry !== undefined) {
    out.geometry = {
      readings: Object.fromEntries(Object.entries(source.geometry.readings).map(
        ([name, reading]) => [name, summaryProbeReading(reading, perLayer)],
      )),
      ...(source.geometry.binding === undefined ? {} : { binding: { ...source.geometry.binding } }),
    };
  }
  if (source.lens !== undefined) {
    const lens = source.lens;
    const readout = lens.readout;
    out.lens = {
      binding: { sources: { ...lens.binding.sources }, steering: lens.binding.steering },
      ...(lens.readings === undefined ? {} : { readings: summaryScalarReadings(lens.readings) }),
      ...(readout === undefined ? {} : {
        readout: {
          aggregate: (readout.aggregate ?? []).map((chip) => ({
            token: chip.token,
            id: chip.id,
            strength: roundReading(chip.strength),
            com: roundReading(chip.com),
            spread: roundReading(chip.spread),
          })),
          ...(perLayer && readout.layers !== undefined ? {
            layers: readout.layers.map((row) => ({
              layer: row.layer,
              tokens: row.tokens.map((token) => ({
                token: token.token,
                id: token.id,
                logprob: roundReading(token.logprob),
              })),
            })),
          } : {}),
        },
      }),
    } as Instruments["lens"];
  }
  if (source.sae !== undefined) {
    const sae = source.sae;
    out.sae = {
      binding: { ...sae.binding },
      ...(sae.readings === undefined ? {} : { readings: summaryScalarReadings(sae.readings) }),
      ...(sae.readout === undefined ? {} : {
        readout: {
          features: sae.readout.features.map((feature) => ({
            id: feature.id,
            activation: roundReading(feature.activation),
            label: feature.label,
            max_act: feature.max_act === null || feature.max_act === undefined
              ? null
              : roundReading(feature.max_act),
          })),
        },
      }),
    } as Instruments["sae"];
  }
  if (source.units !== undefined) {
    out.units = {
      binding: { ...source.units.binding },
      readings: summaryScalarReadings(source.units.readings),
    };
  }
  return {
    version: envelope.version,
    scope: envelope.scope,
    provenance: envelope.provenance,
    instruments: out,
    ...(envelope.scores === undefined ? {} : { scores: roundedScores(envelope.scores) }),
  };
}

/** What a loom node keeps of its reply's aggregate envelope
 *  (`aggregate_measurements`): the summary without traces and without scores. */
export function aggregateMeasurements(envelope: MeasurementsEnvelopeJSON): MeasurementsEnvelopeJSON {
  const summary = summaryMeasurements(envelope, false);
  delete summary.scores;
  return summary;
}

/** The `done` frame's `scope: "aggregate"` envelope from the reply's
 *  last-content token envelope: the attached probes' readings, their bindings
 *  and the scores, without the token's readouts (drowse's aggregate binds a
 *  family only where it carries readings). Undefined when nothing remains. */
export function aggregateEnvelope(
  envelope: MeasurementsEnvelopeJSON,
): MeasurementsEnvelopeJSON | undefined {
  const source = envelope.instruments;
  const out: Instruments = {};
  if (source.geometry !== undefined && Object.keys(source.geometry.readings).length > 0) {
    out.geometry = structuredClone(source.geometry);
  }
  if (source.lens?.readings !== undefined && Object.keys(source.lens.readings).length > 0) {
    // The binding names the families whose data the aggregate keeps: its readings' (the token's
    // R readout stays behind), told apart by unit as native's `LENS_FAMILY_BY_UNIT` does.
    const families = new Set(Object.values(source.lens.readings).map((reading) =>
      reading.unit === "mean_footprint_z" ? "S" : "R"
    ));
    out.lens = {
      binding: {
        ...structuredClone(source.lens.binding),
        sources: Object.fromEntries(Object.entries(source.lens.binding.sources)
          .filter(([family]) => families.has(family))),
      },
      readings: structuredClone(source.lens.readings),
    };
  }
  if (source.sae?.readings !== undefined && Object.keys(source.sae.readings).length > 0) {
    out.sae = {
      binding: structuredClone(source.sae.binding),
      readings: structuredClone(source.sae.readings),
    };
  }
  if (source.units !== undefined && Object.keys(source.units.readings).length > 0) {
    out.units = structuredClone(source.units);
  }
  if (Object.keys(out).length === 0) return undefined;
  return {
    version: envelope.version,
    scope: "aggregate",
    provenance: envelope.provenance,
    instruments: out,
    ...(envelope.scores === undefined ? {} : { scores: { ...envelope.scores } }),
  };
}

function roundedScores(scores: Readonly<Record<string, number>>): Record<string, number> {
  return Object.fromEntries(Object.entries(scores).map(([name, value]) => [name, roundReading(value)]));
}

function ranked(pairs: readonly (readonly [string, number])[]): [string, number][] {
  return pairs.map(([label, value]) => [String(label), roundReading(value)]);
}

function summaryProbeReading(reading: ProbeReadingJSON, perLayer: boolean): ProbeReadingJSON {
  const out = {
    fraction: roundReading(reading.fraction),
    nearest: ranked(reading.nearest),
    coords: reading.coords.map(roundReading),
    residual: roundReading(reading.residual),
    assignment: ranked(reading.assignment ?? []),
    membership: roundReading(reading.membership ?? 0),
    depth_com: (reading.depth_com ?? []).map(roundReading),
    depth_spread: (reading.depth_spread ?? []).map(roundReading),
  } as ProbeReadingJSON;
  if (perLayer) {
    out.fraction_per_layer = roundedMap(reading.fraction_per_layer ?? {});
    out.coords_per_layer = Object.fromEntries(Object.entries(reading.coords_per_layer ?? {}).map(
      ([layer, coords]) => [layer, coords.map(roundReading)],
    ));
    out.residual_per_layer = roundedMap(reading.residual_per_layer ?? {});
  }
  return out;
}

function summaryScalarReadings(
  readings: Readonly<Record<string, ScalarReadingJSON>>,
): Record<string, ScalarReadingJSON> {
  return Object.fromEntries(Object.entries(readings).map(([name, reading]) => {
    const depth = reading.depth ?? null;
    const out: ScalarReadingJSON = {
      value: roundReading(reading.value),
      unit: reading.unit,
      per_layer: roundedMap(reading.per_layer),
      depth: depth === null ? null : {
        center: depth.center.map(roundReading),
        spread: depth.spread.map(roundReading),
        basis: depth.basis,
      },
    } as ScalarReadingJSON;
    if (reading.meta !== undefined && reading.meta !== null) {
      out.meta = Object.fromEntries(Object.entries(reading.meta).map(([key, value]) => [
        key,
        // drowse rounds every float meta value; its only producer (units) writes floats.
        typeof value === "number" && Number.isFinite(value) ? roundReading(value) : value,
      ])) as ScalarReadingJSON["meta"];
    }
    return [name, out];
  }));
}

function roundedMap(values: Readonly<Record<string, number>>): Record<string, number> {
  return Object.fromEntries(Object.entries(values).map(([key, value]) => [key, roundReading(value)]));
}
