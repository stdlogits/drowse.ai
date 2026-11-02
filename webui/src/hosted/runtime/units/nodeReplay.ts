// Node replay on the block-sites backend (native `drowse/loom/replay.py::replay_segments` and
// `drowse/loom/node_replay.py::NodeReplayer._activity`): the forwards that replay a node as
// generation's triggers ran it, and the unit activity assembled from the replay's on-device
// reduction (B1 §2.3).
import {
  activityStateLayout,
  f32Tensor,
  type BlockSitesActivitySpan,
  type BlockSitesForwardReadback,
  type BlockSitesReplayRequest,
  type BlockSitesReplayResult,
  type BlockSitesUnitRef,
  type BrowserBlockSitesBackend,
} from "../modelBackend";
import {
  createStructuredHookControlEvaluator,
  type StructuredHookControlContext,
  type StructuredHookControlState,
  type StructuredHookProgramBuffers,
} from "../structuredHookProgram";
import type { BlockSpec } from "../structuredHookProfile";
import { round6 } from "./lensReads";
import { activityScaleTable, type UnitScales } from "./scales";
import { UnitRequestError, unitName, type UnitRef } from "./unitReader";

export const UNIT_ACTIVITY_UNIT = "activation_over_scale";
export const POPULATION_ENCODING = "float16-le-base64";
export const MAX_ACTIVITY_TOP = 1024;

/** The producing forward of a node's rows (native `ReplayForward`): `ids` is the prompt render
 *  then the raw record before the anchored row; `spans` are native's `spans()`, `(firstRow,
 *  stopRow, at)` with row `r` produced at column `at + r − firstRow` (negative: unproduced). */
export interface NodeReplayForward {
  readonly ids: Int32Array;
  readonly promptLength: number;
  readonly rawIds: Int32Array;
  readonly spans: readonly { readonly firstRow: number; readonly stopRow: number; readonly at: number }[];
}

/** One replay forward: positions `[start, stop)` under one control evaluation. */
export interface ReplaySegmentPlan {
  readonly start: number;
  readonly stop: number;
  readonly prefill: boolean;
  readonly genStep: number;
  /** Stepwise segments evaluate their gates from the previous forward's readback. */
  readonly gated: boolean;
}

/** Gate scores (`priorScores`) from a forward's readback, for the next segment's controls. */
export type ReplayGateScores = (
  program: StructuredHookProgramBuffers,
  readback: BlockSitesForwardReadback,
) => Readonly<Record<string, number>>;

function controlsOf(program: StructuredHookProgramBuffers): (import("../structuredHookProgram").StructuredHookControl | null)[] {
  return [...(program.controls?.affine ?? []), ...(program.controls?.curve ?? []), ...(program.controls?.units ?? [])];
}

/** Native `replay_segments` over a browser program: one forward when every control is
 *  always-on, else the prefill and decode runs split where any control's firing changes; a
 *  probe-gated program steps one decode position per forward. Thinking-phase windows refuse
 *  (`canThink`), as native does. */
export function replaySegments(
  program: StructuredHookProgramBuffers | null,
  promptLength: number,
  length: number,
  canThink = false,
): ReplaySegmentPlan[] {
  const windowed = program === null ? [] : controlsOf(program).filter((c) => c !== null && (c.phase.kind !== "both" || c.gate !== null));
  if (windowed.length === 0) return [{ start: 0, stop: length, prefill: false, genStep: 0, gated: false }];
  const decoded = length > promptLength;
  const gated = decoded && windowed.some((c) => c!.gate !== null && c!.phase.kind !== "prompt_only");
  const phased = decoded && canThink && windowed.some((c) => c!.phase.kind === "thinking_only" || c!.phase.kind === "after_thinking");
  if (phased) {
    throw Object.assign(new Error(
      "the replayed steering has thinking-phase terms (@after, @thinking), which fire on the decode loop's thinking state; the node does not record where they fired, so a replay cannot reproduce them. Replay without them (apply_steering=false drops the node's own steering).",
    ), { code: "SteeringCompositionError", status: 422, recoverable: true });
  }
  const prefillStop = Math.min(promptLength, length);
  const out: ReplaySegmentPlan[] = [];
  if (prefillStop > 0) out.push({ start: 0, stop: prefillStop, prefill: true, genStep: 0, gated });
  if (gated) {
    for (let position = prefillStop; position < length; position += 1) {
      out.push({ start: position, stop: position + 1, prefill: false, genStep: position - promptLength + 1, gated: true });
    }
    return out;
  }
  const evaluate = createStructuredHookControlEvaluator(program!);
  const fires = (step: number) => {
    const state = evaluate({ prefill: false, thinking: false, generatedTokens: step });
    return `${state.affineActive.join()}|${state.curveActive.join()}|${state.unitActive?.join() ?? ""}`;
  };
  let start = prefillStop;
  while (start < length) {
    const step = start - promptLength + 1;
    const state = fires(step);
    let stop = start + 1;
    while (stop < length && fires(stop - promptLength + 1) === state) stop += 1;
    out.push({ start, stop, prefill: false, genStep: step, gated: false });
    start = stop;
  }
  return out;
}

const NO_CONTROLS: StructuredHookControlState = { affineActive: new Uint32Array(0), curveActive: new Uint32Array(0) };

/** Drive a replay's segments through the backend (`drive_segments`), the controls of each
 *  evaluated as the decode loop would have (gates from the previous forward's readback). */
export async function driveReplay(
  backend: BrowserBlockSitesBackend,
  request: BlockSitesReplayRequest,
  forward: NodeReplayForward,
  options: { canThink?: boolean; gateScores?: ReplayGateScores; signal?: AbortSignal } = {},
): Promise<BlockSitesReplayResult> {
  const program = request.program;
  const segments = replaySegments(program, forward.promptLength, forward.ids.length, options.canThink);
  const evaluate = program === null ? null : createStructuredHookControlEvaluator(program);
  const replay = await backend.openReplay(request, options.signal);
  try {
    let readback: BlockSitesForwardReadback | null = null;
    for (const segment of segments) {
      const context: StructuredHookControlContext = {
        prefill: segment.prefill,
        thinking: false,
        generatedTokens: segment.genStep,
        ...(segment.gated && readback !== null && program !== null
          ? { priorMeasurements: readback.measurements.data, priorScores: options.gateScores?.(program, readback) ?? null }
          : {}),
      };
      const controls = evaluate === null ? NO_CONTROLS : evaluate(context);
      readback = await replay.forward({ tokenIds: forward.ids.slice(segment.start, segment.stop), controls });
    }
    return await replay.finish();
  } catch (error) {
    await replay.cancel();
    throw error;
  }
}

// ------------------------------------------------------------------ activity ---

export interface UnitActivityOptions {
  readonly top: number;
  readonly units: readonly UnitRef[];
  /** The token row whose population to return, or null. */
  readonly rawIndex: number | null;
  /** The population's layers (every layer when null). */
  readonly layers: readonly number[] | null;
}

/** IEEE binary16 bits of an fp32 value, clamped to ±65504, round to nearest even. */
export function float16Bits(value: number): number {
  const v = Math.max(-65504, Math.min(65504, Math.fround(value)));
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  f32[0] = v;
  const x = u32[0];
  const sign = (x >>> 16) & 0x8000;
  const exp = (x >>> 23) & 0xff;
  let mant = x & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);
  let e = exp - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00;
  if (e <= 0) {
    if (e < -10) return sign;
    mant |= 0x800000;
    const shift = 14 - e;
    const half = 1 << (shift - 1);
    let m = mant >>> shift;
    const rest = mant & ((1 << shift) - 1);
    if (rest > half || (rest === half && (m & 1))) m += 1;
    return sign | m;
  }
  let m = mant >>> 13;
  const rest = mant & 0x1fff;
  if (rest > 0x1000 || (rest === 0x1000 && (m & 1))) {
    m += 1;
    if (m === 0x400) {
      m = 0;
      e += 1;
      if (e >= 0x1f) return sign | 0x7c00;
    }
  }
  return sign | (e << 10) | m;
}

export function encodePopulation(values: ArrayLike<number>): string {
  const bytes = new Uint8Array(2 * values.length);
  for (let i = 0; i < values.length; i += 1) {
    const bits = float16Bits(values[i]);
    bytes[2 * i] = bits & 0xff;
    bytes[2 * i + 1] = bits >>> 8;
  }
  let text = "";
  for (let i = 0; i < bytes.length; i += 0x8000) text += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(text);
}

/** Native `unit_activity`'s argument checks (400s, verbatim). */
export function checkActivityOptions(options: UnitActivityOptions): void {
  if (!(Number.isSafeInteger(options.top) && options.top >= 1 && options.top <= MAX_ACTIVITY_TOP)) {
    throw new UnitRequestError("RequestError", "top must be in [1, 1024]");
  }
  if (options.layers !== null && options.rawIndex === null) {
    throw new UnitRequestError("RequestError", "layers selects a population, which needs raw_index");
  }
  const names = options.units.map(unitName);
  if (new Set(names).size !== names.length) throw new UnitRequestError("RequestError", "units names a unit twice");
}

/** `NodeReplayer._activity` on the backend: replay the node's producing forward with the
 *  activity reduction, then rank each kind by peak `|value|` (value desc, flat index asc), drop
 *  units that never ranked (rank −1), and pack traces and the population. */
export async function unitActivity(
  backend: BrowserBlockSitesBackend,
  model: { readonly nLayers: number; readonly spec: BlockSpec },
  scales: UnitScales,
  forward: NodeReplayForward,
  program: StructuredHookProgramBuffers | null,
  options: UnitActivityOptions,
  decode: (tokenId: number) => string,
  replayOptions: { canThink?: boolean; gateScores?: ReplayGateScores; signal?: AbortSignal } = {},
): Promise<Record<string, unknown>> {
  checkActivityOptions(options);
  const { nLayers, spec } = model;
  const F = spec.dFF;
  const H = spec.nHeads;
  const count = forward.rawIds.length;
  const spans: BlockSitesActivitySpan[] = forward.spans.map((span) => ({
    firstRow: span.firstRow,
    rows: span.stopRow - span.firstRow,
    firstPosition: span.at,
  }));
  const atSink = new Array<boolean>(count).fill(false);
  for (const span of forward.spans) {
    for (let row = span.firstRow; row < span.stopRow; row += 1) atSink[row] = span.at + row - span.firstRow < 1;
  }
  const traces: BlockSitesUnitRef[] = options.units.map((unit) => ({ kind: unit.kind, layer: unit.layer, index: unit.index }));
  const scaleTable = await backend.uploadUnitScales(f32Tensor(activityScaleTable(scales, spec, nLayers), [nLayers, 2 * F + H]));
  let result: BlockSitesReplayResult;
  try {
    result = await driveReplay(backend, {
      program,
      anatomyPosition: null,
      activity: { spans, rowCount: count, scales: scaleTable, traces, populationRow: options.rawIndex },
      patternLayers: null,
    }, forward, replayOptions);
  } finally {
    await backend.release(scaleTable);
  }
  const activity = result.activity!;
  const layout = activityStateLayout(spec);
  const state = activity.state.data;
  // Flat order native ranks in: [(neuron, L0), (head, L0), (neuron, L1), …].
  const entries: { kind: "neuron" | "head"; layer: number; index: number; flat: number; value: number; row: number; rank: number }[] = [];
  let flat = 0;
  for (let layer = 0; layer < nLayers; layer += 1) {
    const base = layer * layout.stride;
    for (let n = 0; n < F; n += 1) {
      entries.push({ kind: "neuron", layer, index: n, flat: flat++, value: state[base + layout.neuronValue + n], row: state[base + layout.neuronRow + n], rank: state[base + layout.neuronRank + n] });
    }
    for (let h = 0; h < H; h += 1) {
      entries.push({ kind: "head", layer, index: h, flat: flat++, value: state[base + layout.headValue + h], row: state[base + layout.headRow + h], rank: state[base + layout.headRank + h] });
    }
  }
  const ranked = (kind: "neuron" | "head") => entries
    .filter((entry) => entry.kind === kind)
    .sort((a, b) => b.rank - a.rank || a.flat - b.flat)
    .slice(0, options.top)
    .filter((entry) => entry.rank >= 0);
  const top = [...ranked("neuron"), ...ranked("head")].map((entry) => ({
    unit: unitName(entry),
    kind: entry.kind,
    layer: entry.layer,
    index: entry.index,
    peak: round6(entry.value),
    raw_index: entry.row,
    label: null,
  }));
  const tracesOut: Record<string, number[]> = {};
  options.units.forEach((unit, slot) => {
    tracesOut[unitName(unit)] = Array.from(activity.traces.data.subarray(slot * count, (slot + 1) * count), round6);
  });
  let population: Record<string, unknown> | null = null;
  if (options.rawIndex !== null) {
    const layers = [...new Set(options.layers ?? Array.from({ length: nLayers }, (_, layer) => layer))].sort((a, b) => a - b);
    const values = activity.population!.data;
    population = {
      raw_index: options.rawIndex,
      encoding: POPULATION_ENCODING,
      layers: layers.map((layer) => ({
        layer,
        neurons: encodePopulation(values.subarray(layer * (F + H), layer * (F + H) + F)),
        heads: encodePopulation(values.subarray(layer * (F + H) + F, (layer + 1) * (F + H))),
        n_neurons: F,
        n_heads: H,
      })),
    };
  }
  const sink = atSink.indexOf(true);
  return {
    node_id: "",
    steering: null,
    unit: UNIT_ACTIVITY_UNIT,
    scale: { source: scales.source, detail: scales.detail },
    tokens: Array.from(forward.rawIds, (id, i) => ({ raw_index: i, token_id: id, token: decode(id) })),
    sink_raw_index: sink >= 0 ? sink : null,
    top,
    traces: tracesOut,
    population,
  };
}
