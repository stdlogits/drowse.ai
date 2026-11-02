// The unit part of a browser steering program (B1 §2.2, §3.1-§3.3): unit references as the
// compiler resolves them, the soft reading scale a tap divides by, the units channel's readings
// from a forward's taps (native `UnitsInstrument.readings`), and the gate scores unit gates read.
import type { ScalarReadingJSON } from "../../../../../drowse/webui/src/lib/types";
import { UnsupportedBlockError, type BlockSitesForwardReadback } from "../modelBackend";
import { roundReading } from "../measurementRows";
import type {
  StructuredHookProgramBuffers,
  StructuredUnitTap,
  StructuredUnitTapScale,
} from "../structuredHookProgram";
import type { BlockSpec } from "../structuredHookProfile";
import type { UnitScales, UnitStatistics } from "./scales";
import { checkUnit, parseUnit, unitName, type UnitRef } from "./unitReader";

export const UNIT_ACTIVATION_OVER_SCALE = "activation_over_scale";

/** What the structured compiler needs to lower unit atoms, read off the loaded library and
 *  the core pack (B6): the attested block spec (null with the residual library's refusal),
 *  the neutral statistics and their reading scales, and the device-resolved directions the
 *  async preparation step filled (a neuron's write, an `slens/` footprint profile). */
export interface BrowserUnitProgramSource {
  readonly spec: BlockSpec | null;
  readonly refusal: string | null;
  readonly layerCount: number;
  readonly statistics: UnitStatistics | null;
  readonly scales: UnitScales | null;
  /** `s·γ_mo ⊙ W_down[:, n]` of a prepared neuron, else null. */
  neuronWrite(layer: number, index: number): Float32Array | null;
  /** `{L: K^{out}_L ẽ_v}` of a prepared `slens/<word>`, else null. */
  slensAtom(word: string): { readonly tokenId: number; readonly directions: ReadonlyMap<number, Float32Array> } | null;
  /** What an attached `slens/` probe's digest binds besides its word (native
   *  `LensInstrument.probe_hash`): the S lens identity, the centring out means' digest at the
   *  probe's layers, the token mask's digest. Null until a `slens/` atom or probe was prepared. */
  slensIdentity(layers: readonly number[]): { lens: string; centre: string; mask: string } | null;
  /** Reads off the device what `expression`'s `neuron/` pushes and `slens/` atoms and gates
   *  need, so the synchronous compile finds them (`neuronWrite`, `slensAtom`). */
  prepare(expression: string): Promise<void>;
  /** The same for one probe selector (`slens/<word>`: its atom and its digest's identity). */
  prepareProbe(selector: string): Promise<void>;
}

/** A `neuron/…` or `head/…` atom's unit, refused as native refuses it: 422 on a library
 *  without unit sites, 400 for a unit outside the model. */
export function resolveUnitAtom(source: BrowserUnitProgramSource, namespace: "neuron" | "head", concept: string): UnitRef {
  if (source.spec === null) throw new UnsupportedBlockError(source.refusal ?? "the model's blocks have no unit sites");
  return parseUnit(`${namespace}/${concept}`, source.spec, source.layerCount);
}

/** The neutral mean `α !unit` blends toward: `neuron_mean[n]` or `head_mean[h]` `[hd]`. */
export function unitMean(source: BrowserUnitProgramSource, ref: UnitRef): number | Float32Array {
  const stats = source.statistics;
  const spec = source.spec!;
  if (stats === null) {
    throw Object.assign(new Error("the model's neutral cache holds no unit statistics"), {
      code: "UnitsUnavailableError", status: 422, recoverable: false,
    });
  }
  checkUnit(ref, spec, source.layerCount);
  if (ref.kind === "neuron") return stats.neuronMean[ref.layer * spec.dFF + ref.index];
  const at = (ref.layer * spec.nHeads + ref.index) * spec.headDim;
  return stats.headMean.slice(at, at + spec.headDim);
}

/** The soft, floored scale a tap of `ref` divides by (`[+, −]` for a neuron). */
export function unitTapScale(scales: UnitScales, spec: Pick<BlockSpec, "dFF" | "nHeads">, ref: UnitRef): StructuredUnitTapScale {
  const round = (x: number) => Math.round(x * 1e6) / 1e6;
  if (ref.kind === "neuron") {
    const plus = (ref.layer * 2) * spec.dFF + ref.index;
    const minus = (ref.layer * 2 + 1) * spec.dFF + ref.index;
    return {
      source: "neutral",
      detail: scales.detail,
      top: [round(scales.neuronTop[plus]), round(scales.neuronTop[minus])],
      value: [Math.max(scales.neuron[plus], 1e-6), Math.max(scales.neuron[minus], 1e-6)],
    };
  }
  const at = ref.layer * spec.nHeads + ref.index;
  return { source: "neutral", detail: scales.detail, top: round(scales.headTop[at]), value: Math.max(scales.head[at], 1e-6) };
}

/** The live tap for `name` (an attached unit probe or a gate's unit) at `slot`. */
export function unitTap(source: BrowserUnitProgramSource, name: string, ref: UnitRef, slot: number): StructuredUnitTap {
  if (source.scales === null || source.spec === null) {
    throw Object.assign(new Error("the model's neutral cache holds no unit statistics"), {
      code: "UnitsUnavailableError", status: 422, recoverable: false,
    });
  }
  return { name, unit: unitName(ref), kind: ref.kind, layer: ref.layer, index: ref.index, slot, scale: unitTapScale(source.scales, source.spec, ref) };
}

/** One forward's tap value over its scale (native `UnitTap.values_at`): a neuron's `act` over its
 *  pole's scale picked by sign, a head's write norm over its own. */
export function tapValue(tap: StructuredUnitTap, raw: number): number {
  const scale = tap.scale.value;
  if (Array.isArray(scale)) return raw / (raw >= 0 ? scale[0] : scale[1]);
  return raw / (scale as number);
}

/** The units channel's readings for one forward's taps (`UnitsInstrument.readings`), the
 *  rounded scores, and the exact gate scores keyed by probe name and by unit. */
export function unitTapReadings(
  taps: readonly (StructuredUnitTap | null)[],
  values: ArrayLike<number>,
  nLayers: number,
): {
  readings: Record<string, ScalarReadingJSON>;
  scores: Record<string, number>;
  gateScores: Record<string, number>;
} {
  const readings: Record<string, ScalarReadingJSON> = {};
  const scores: Record<string, number> = {};
  const gateScores: Record<string, number> = {};
  const den = Math.max(nLayers - 1, 1);
  for (const tap of taps) {
    if (tap === null) continue;
    const raw = values[tap.slot];
    const value = tapValue(tap, raw);
    gateScores[tap.name] = value;
    gateScores[tap.unit] = value;
    scores[tap.name] = roundReading(value);
    readings[tap.name] = {
      value: roundReading(value),
      unit: UNIT_ACTIVATION_OVER_SCALE,
      per_layer: { [String(tap.layer)]: roundReading(value) },
      depth: { center: [roundReading(tap.layer / den)], spread: [0], basis: "single_layer" },
      meta: {
        unit: tap.unit,
        [tap.kind === "neuron" ? "activation" : "write_norm"]: Math.round(raw * 1e6) / 1e6,
        scale: tap.scale.source,
      },
    };
  }
  return { readings, scores, gateScores };
}

/** Gate scores for a replay's next segment from its last forward's taps. */
export function unitGateScores(
  program: StructuredHookProgramBuffers,
  readback: BlockSitesForwardReadback,
): Readonly<Record<string, number>> {
  const taps = program.measurementSchema?.unitTaps;
  if (!taps || readback.unitTaps === null) return {};
  return unitTapReadings(taps, readback.unitTaps.data, program.layerCount).gateScores;
}
