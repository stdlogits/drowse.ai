// Token anatomy (native `drowse/loom/node_replay.py::NodeReplayer.token_anatomy`, `_anatomy`,
// `_contributions`) on the block-sites backend: one replay with the anatomy taps at the token's
// producing position, the four lens reads per layer, the direct contributions of every neuron
// and head through the R lens, and the top heads' attention from a pattern pass (B1 §2.3, §4.1,
// §6; C0's note: under full-depth packs the chosen heads are known only after contributions).
import {
  anatomyTapLayout,
  f32Tensor,
  i32Tensor,
  type BlockSitesLensLayer,
  type BrowserBlockSitesBackend,
} from "../modelBackend";
import type { StructuredHookProgramBuffers } from "../structuredHookProgram";
import type { BlockSpec } from "../structuredHookProfile";
import { readVectors, round6, type UnitsReadContext, type VectorReadRequest } from "./lensReads";
import { driveReplay, type NodeReplayForward, type ReplayGateScores } from "./nodeReplay";
import type { UnitScales } from "./scales";
import { UnitRequestError, unitName } from "./unitReader";

const SCALE_FLOOR = 1e-6;

export interface TokenAnatomyOptions {
  readonly k: number;
  readonly units: number;
  readonly patterns: number;
}

/** Native's knob checks (400s, verbatim). */
export function checkAnatomyOptions(options: TokenAnatomyOptions): void {
  if (!(Number.isSafeInteger(options.k) && options.k >= 1 && options.k <= 256)) throw new UnitRequestError("RequestError", "k must be in [1, 256]");
  if (!(Number.isSafeInteger(options.units) && options.units >= 1 && options.units <= 64)) throw new UnitRequestError("RequestError", "units must be in [1, 64]");
  if (!(Number.isSafeInteger(options.patterns) && options.patterns >= 0 && options.patterns <= 16)) throw new UnitRequestError("RequestError", "patterns must be in [0, 16]");
}

export interface AnatomyForward extends NodeReplayForward {
  /** The anchored row (its producing position is the forward's last column). */
  readonly rawIndex: number;
}

/** Top `n` indices of `values` by `|value|` desc, index asc. */
function topAbs(values: ArrayLike<number>, n: number): number[] {
  const order = Array.from({ length: values.length }, (_, i) => i);
  order.sort((a, b) => Math.abs(values[b]) - Math.abs(values[a]) || a - b);
  return order.slice(0, Math.min(n, values.length));
}

function unitRow(kind: "neuron" | "head", layer: number, index: number, contribution: number, activation: number | null, over: number | null) {
  return {
    unit: unitName({ kind, layer, index }),
    kind,
    layer,
    index,
    contribution: round6(contribution),
    activation: activation === null ? null : round6(activation),
    activation_over_scale: over === null ? null : round6(over),
  };
}

/** The `scope="replay"` envelope's `anatomy` record for one token, and the block outputs it read
 *  (`out` `[L, d]` at the anatomy position, after the steering), which the envelope's lens
 *  probes are scored over (native `score_probes({L: taps.out[L]})`). */
export async function tokenAnatomy(
  context: UnitsReadContext,
  model: { readonly spec: BlockSpec; readonly scales: UnitScales | null; readonly backend: BrowserBlockSitesBackend },
  forward: AnatomyForward,
  program: StructuredHookProgramBuffers | null,
  options: TokenAnatomyOptions,
  binding: { readonly sources: Readonly<Record<"R" | "S", string | null>>; readonly steering: string | null },
  replayOptions: { canThink?: boolean; gateScores?: ReplayGateScores; signal?: AbortSignal } = {},
): Promise<{ anatomy: Record<string, unknown>; out: Float32Array }> {
  checkAnatomyOptions(options);
  const { spec, scales, backend } = model;
  const n = context.nLayers;
  const d = context.hiddenSize;
  const F = spec.dFF;
  const H = spec.nHeads;
  const hd = spec.headDim;
  const position = forward.ids.length - 1;
  const tokenId = forward.rawIds[forward.rawIndex];
  const replay = await driveReplay(backend, { program, anatomyPosition: position, activity: null, patternLayers: null }, forward, replayOptions);
  const taps = replay.anatomyTaps!.data;
  const layout = anatomyTapLayout(d, spec);
  const tap = (layer: number, offset: number, width: number) => taps.slice(layer * layout.stride + offset, layer * layout.stride + offset + width);
  const delta = (layer: number, a: number, b: number) => {
    const x = tap(layer, a, d);
    const y = tap(layer, b, d);
    return x.map((value, i) => value - y[i]);
  };

  // ---- the four reads per layer
  const requests: VectorReadRequest[] = [];
  const injected: boolean[] = [];
  for (let layer = 0; layer < n; layer += 1) {
    const steering = delta(layer, layout.out, layout.outPre);
    injected.push(steering.some((value) => Math.abs(value) > 0));
    requests.push({ rows: tap(layer, layout.out, d), layer, site: "out", of: "residual" });
    requests.push({ rows: delta(layer, layout.mid, layout.blockIn), layer, site: "mid", of: "direction" });
    requests.push({ rows: delta(layer, layout.outPre, layout.mid), layer, site: "out", of: "direction" });
    requests.push({ rows: steering, layer, site: "out", of: "direction" });
  }
  const reads = await readVectors(context, requests, options.k);
  const finalOut = tap(n - 1, layout.out, d);
  const tokenLogit = await backend.vocabTokenLogits(f32Tensor(finalOut, [1, d]), i32Tensor(Int32Array.of(tokenId), [1, 1]));
  const probability = Math.exp(tokenLogit.logits.data[0] - tokenLogit.logSumExp.data[0]);

  // ---- contributions through the R lens
  let absent: string | null = null;
  const neuronC = new Map<number, Float32Array>();
  const headC = new Map<number, Float32Array>();
  const lens = context.r;
  if (lens === null) {
    absent = "no R-family lens is selected: the units that pushed the token are read through it (fetch or fit one)";
  } else {
    const u = (await backend.unembedCotangent(f32Tensor(finalOut, [1, d]), Int32Array.of(tokenId))).data;
    const cotangents = async (site: "out" | "mid"): Promise<Map<number, Float32Array>> => {
      const out = new Map<number, Float32Array>();
      if (!lens.sites.includes(site)) return out;
      const layers = [...lens.layers(site)].sort((a, b) => a - b);
      if (layers.length === 0) return out;
      const matrices: BlockSitesLensLayer[] = await Promise.all(layers.map((layer) => lens.matrix(site, layer)));
      const rows = new Float32Array(layers.length * d);
      layers.forEach((_, i) => rows.set(u, i * d));
      const g = await backend.lensTransport(f32Tensor(rows, [layers.length, 1, d]), matrices, false);
      layers.forEach((layer, i) => out.set(layer, g.data.slice(i * d, (i + 1) * d)));
      return out;
    };
    const gOut = await cotangents("out");
    if (gOut.size) gOut.set(n - 1, Float32Array.from(u));
    const gMid = await cotangents("mid");
    const outLayers = [...gOut.keys()].sort((a, b) => a - b);
    const midLayers = [...gMid.keys()].sort((a, b) => a - b);
    if (outLayers.length) {
      const g = new Float32Array(outLayers.length * d);
      const act = new Float32Array(outLayers.length * F);
      const mlpO = new Float32Array(outLayers.length * d);
      outLayers.forEach((layer, i) => {
        g.set(gOut.get(layer)!, i * d);
        act.set(tap(layer, layout.act, F), i * F);
        mlpO.set(tap(layer, layout.mlpO, d), i * d);
      });
      const c = await backend.neuronContributions(f32Tensor(g, [outLayers.length, d]), f32Tensor(act, [outLayers.length, F]), f32Tensor(mlpO, [outLayers.length, d]), Int32Array.from(outLayers));
      outLayers.forEach((layer, i) => neuronC.set(layer, c.data.slice(i * F, (i + 1) * F)));
    }
    if (midLayers.length) {
      const g = new Float32Array(midLayers.length * d);
      const z = new Float32Array(midLayers.length * H * hd);
      const attnO = new Float32Array(midLayers.length * d);
      midLayers.forEach((layer, i) => {
        g.set(gMid.get(layer)!, i * d);
        z.set(tap(layer, layout.z, H * hd), i * H * hd);
        attnO.set(tap(layer, layout.attnO, d), i * d);
      });
      const c = await backend.headContributions(f32Tensor(g, [midLayers.length, d]), f32Tensor(z, [midLayers.length, H * hd]), f32Tensor(attnO, [midLayers.length, d]), Int32Array.from(midLayers));
      midLayers.forEach((layer, i) => headC.set(layer, c.data.slice(i * H, (i + 1) * H)));
    }
    if (neuronC.size === 0 && headC.size === 0) absent = "the R lens holds no layer at either site";
  }

  // ---- per-layer units
  const neurons = new Map<number, ReturnType<typeof unitRow>[]>();
  const heads = new Map<number, ReturnType<typeof unitRow>[]>();
  for (const [layer, c] of neuronC) {
    const act = tap(layer, layout.act, F);
    neurons.set(layer, topAbs(c, options.units).map((i) => {
      let over: number | null = null;
      if (scales) {
        const pole = act[i] >= 0 ? scales.neuron[(layer * 2) * F + i] : scales.neuron[(layer * 2 + 1) * F + i];
        over = act[i] / Math.max(pole, SCALE_FLOOR);
      }
      return unitRow("neuron", layer, i, c[i], act[i], over);
    }));
  }
  if (headC.size) {
    const layers = [...headC.keys()];
    const z = new Float32Array(layers.length * H * hd);
    const attnO = new Float32Array(layers.length * d);
    layers.forEach((layer, i) => {
      z.set(tap(layer, layout.z, H * hd), i * H * hd);
      attnO.set(tap(layer, layout.attnO, d), i * d);
    });
    const norms = (await backend.headWriteNorms(f32Tensor(z, [layers.length, H * hd]), f32Tensor(attnO, [layers.length, d]), Int32Array.from(layers))).data;
    layers.forEach((layer, l) => {
      const c = headC.get(layer)!;
      heads.set(layer, topAbs(c, options.units).map((i) => {
        const norm = norms[l * H + i];
        const over = scales ? norm / Math.max(scales.head[layer * H + i], SCALE_FLOOR) : null;
        return unitRow("head", layer, i, c[i], norm, over);
      }));
    });
  }

  // ---- across layers: every neuron, then every head (sorted layers), by |c|
  const flat: { kind: "neuron" | "head"; layer: number; index: number; value: number }[] = [];
  for (const layer of [...neuronC.keys()].sort((a, b) => a - b)) neuronC.get(layer)!.forEach((value, index) => flat.push({ kind: "neuron", layer, index, value }));
  const headBase = flat.length;
  for (const layer of [...headC.keys()].sort((a, b) => a - b)) headC.get(layer)!.forEach((value, index) => flat.push({ kind: "head", layer, index, value }));
  const top = topAbs(flat.map((entry) => entry.value), options.units).map((i) => unitRow(flat[i].kind, flat[i].layer, flat[i].index, flat[i].value, null, null));

  // ---- patterns: the top heads' attention from the token's producing position
  let patterns: Record<string, unknown>[] | null = null;
  let contextTokens: string[] | null = null;
  const readable = lens !== null && lens.sites.includes("mid") && lens.layers("mid").length > 0;
  if (options.patterns > 0 && readable && headC.size > 0) {
    const headValues = flat.slice(headBase).map((entry) => entry.value);
    const chosen = topAbs(headValues, options.patterns).map((i) => flat[headBase + i]);
    const rows = await patternRows(backend, program, forward, chosen.map((entry) => ({ layer: entry.layer, head: entry.index })), replayOptions);
    patterns = chosen.map((entry, i) => ({
      unit: unitName({ kind: "head", layer: entry.layer, index: entry.index }),
      layer: entry.layer,
      head: entry.index,
      contribution: round6(entry.value),
      weights: Array.from(rows[i], round6),
    }));
    contextTokens = Array.from(forward.ids, (id) => context.decode(id));
  }

  const layers = Array.from({ length: n }, (_, layer) => {
    const [readsAs, attention, mlp, steering] = [0, 1, 2, 3].map((i) => reads[4 * layer + i][0]);
    return {
      layer,
      depth: readsAs.depth,
      reads_as: readsAs,
      attention,
      mlp,
      steering: injected[layer] ? steering : null,
      neurons: neurons.get(layer) ?? [],
      heads: heads.get(layer) ?? [],
    };
  });
  const out = new Float32Array(n * d);
  for (let layer = 0; layer < n; layer += 1) out.set(tap(layer, layout.out, d), layer * d);
  const anatomy = {
    token: {
      raw_index: forward.rawIndex,
      token_id: tokenId,
      token: context.decode(tokenId),
      position,
      prompt_length: forward.promptLength,
      probability: round6(probability),
    },
    binding: { sources: { ...binding.sources }, steering: binding.steering },
    layers,
    units: top,
    units_absent: absent,
    units_scale: scales === null ? null : scales.source,
    patterns,
    context: contextTokens,
  };
  return { anatomy, out };
}

/** The pattern pass: replay again capturing keys at the chosen heads' layers (at most the
 *  profile's `maxPatternLayers` per replay) and read their attention rows. */
async function patternRows(
  backend: BrowserBlockSitesBackend,
  program: StructuredHookProgramBuffers | null,
  forward: AnatomyForward,
  heads: readonly { layer: number; head: number }[],
  replayOptions: { canThink?: boolean; gateScores?: ReplayGateScores; signal?: AbortSignal },
): Promise<Float32Array[]> {
  const capacity = backend.capabilities()?.profile.maxPatternLayers ?? 0;
  if (capacity < 1) throw new Error("this model library captures no attention patterns");
  const position = forward.ids.length - 1;
  const layers = [...new Set(heads.map((entry) => entry.layer))].sort((a, b) => a - b);
  const out = new Array<Float32Array>(heads.length);
  for (let start = 0; start < layers.length; start += capacity) {
    const batch = layers.slice(start, start + capacity);
    const replay = await driveReplay(backend, { program, anatomyPosition: position, activity: null, patternLayers: batch }, forward, replayOptions);
    const capture = replay.patterns!;
    try {
      const wanted = heads.map((entry, i) => ({ entry, i })).filter(({ entry }) => batch.includes(entry.layer));
      const rows = await backend.attentionRows(capture, wanted.map(({ entry }) => entry));
      wanted.forEach(({ i }, j) => {
        out[i] = rows.data.slice(j * (position + 1), (j + 1) * (position + 1));
      });
    } finally {
      await backend.release(capture);
    }
  }
  return out;
}
