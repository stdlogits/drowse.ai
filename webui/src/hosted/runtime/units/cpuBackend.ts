// The CPU fp32 reference of hook ABI v6's device surface (`BrowserBlockSitesBackend`, B1 §2-§6)
// over dense weights: plain loops, B1 §4.1's formulas. Test-only. It lets the host ports run
// end to end without a GPU (the fixture runtime), and is the reference B4's kernels answer to.
// Nothing in the hosted app imports it.
import {
  activityStateLayout,
  anatomyTapLayout,
  f32Tensor,
  i32Tensor,
  initialActivityState,
  requireTensorShape,
  UnsupportedBlockError,
  validateBlockSitesCapabilities,
  type BlockSitesActivityRequest,
  type BlockSitesCapabilities,
  type BlockSitesEmbeddingStatistics,
  type BlockSitesForwardReadback,
  type BlockSitesLensLayer,
  type BlockSitesReplayRequest,
  type BlockSitesReplayResult,
  type BlockSitesReplaySegment,
  type BlockSitesResident,
  type BlockSitesResidentKind,
  type BlockSitesUnitRef,
  type BrowserBlockSitesBackend,
  type BrowserBlockSitesReplay,
  type F32Tensor,
  type I32Tensor,
  type LensFamily,
  type LensSite,
  type UnitWeightRowsKind,
} from "../modelBackend";
import {
  hasStructuredUnitSection,
  initialStructuredHookState,
  runStructuredHookLayer,
  runStructuredUnitSite,
  structuredUnitSectionLayout,
  type StructuredHookControlState,
  type StructuredHookProgramBuffers,
  type StructuredHookState,
} from "../structuredHookProgram";
import {
  BLOCK_SITES_EXACT_READOUT_ABI,
  BLOCK_SITES_HOOK_ABI,
  BLOCK_SITES_RUNTIME_ABI,
  isLocalAttentionLayer,
  UNIT_POSITIONS_FROM,
  type StructuredHookCapacityProfile,
} from "../structuredHookProfile";
import { DenseSequence, finalLogits, rmsNorm, rmsScale, type DenseModelWeights } from "./denseModel";

export interface CpuBlockSitesOptions {
  readonly profile: StructuredHookCapacityProfile;
  readonly contextTokens: number;
  readonly prefillChunkSize?: number;
  /** The refusal a residual profile answers with (a lock's `unitRefusal`). */
  readonly unitRefusal?: string;
}

interface StoredResident {
  readonly handle: BlockSitesResident;
  readonly data: unknown;
}

interface StoredLens {
  readonly family: LensFamily;
  readonly site: LensSite;
  readonly layer: number;
  readonly matrix: Float32Array;
}

interface StoredStatistics {
  readonly meanE: Float32Array;
  readonly sigmaE: Float32Array;
  readonly keep: Uint8Array;
}

interface StoredPatterns {
  readonly layers: readonly number[];
  /** Per pattern layer, keys `[context, KV·hd]` (post-RoPE). */
  readonly keys: Float32Array[];
  /** Per pattern layer, queries `[H·hd]` at the anatomy position. */
  readonly queries: Float32Array[];
  readonly position: number;
}

const sum = (n: number, f: (i: number) => number): number => {
  let s = 0;
  for (let i = 0; i < n; i += 1) s += f(i);
  return s;
};

/** The CPU reference backend over `model` under a `standard-v5` (or residual) profile. */
export function cpuBlockSitesBackend(
  model: DenseModelWeights,
  options: CpuBlockSitesOptions,
): BrowserBlockSitesBackend {
  const { spec } = model;
  const d = model.hiddenSize;
  const L = model.layerCount;
  const V = model.vocabSize;
  const H = spec.nHeads;
  const KV = spec.nKvHeads;
  const hd = spec.headDim;
  const F = spec.dFF;
  const units = options.profile.unitSites === 1;
  const capabilities = validateBlockSitesCapabilities({
    hookAbi: BLOCK_SITES_HOOK_ABI,
    runtimeAbi: BLOCK_SITES_RUNTIME_ABI,
    exactReadoutAbi: BLOCK_SITES_EXACT_READOUT_ABI,
    profile: options.profile,
    layerCount: L,
    hiddenSize: d,
    vocabSize: V,
    contextTokens: options.contextTokens,
    prefillChunkSize: options.prefillChunkSize ?? 2048,
    blockSpec: units ? spec : null,
    unitRefusal: units ? null : options.unitRefusal ?? "the reference model's blocks are not dense",
  });
  const residents = new Map<string, StoredResident>();
  let serial = 0;
  let grams: BlockSitesResident<"head_grams"> | null = null;
  const requireUnits = () => {
    if (!units) throw new UnsupportedBlockError(capabilities.unitRefusal!);
  };
  const resident = <K extends BlockSitesResidentKind>(
    kind: K,
    shape: readonly number[],
    data: unknown,
    extra: object = {},
  ): BlockSitesResident<K> => {
    const elements = shape.reduce((a, b) => a * b, 1);
    const handle = Object.freeze({ kind, id: `${kind}-${(serial += 1)}`, bytes: 4 * elements, shape: Object.freeze([...shape]), ...extra });
    residents.set(handle.id, { handle, data });
    return handle as BlockSitesResident<K>;
  };
  const stored = <T>(handle: BlockSitesResident, kind: BlockSitesResidentKind): T => {
    const entry = residents.get(handle.id);
    if (!entry || entry.handle.kind !== kind) throw new TypeError(`${handle.id} is not a resident ${kind}`);
    return entry.data as T;
  };

  // ------------------------------------------------------------ weight helpers ---
  const block = (layer: number) => {
    if (!Number.isSafeInteger(layer) || layer < 0 || layer >= L) throw new RangeError(`layer ${layer} is outside the model`);
    return model.blocks[layer];
  };
  /** `s·γ_mo ⊙ W_down[:, n]`. */
  const neuronWrite = (layer: number, n: number): Float32Array => {
    const b = block(layer);
    return Float32Array.from({ length: d }, (_, i) => (b.mlpOutNorm ? b.mlpOutNorm[i] : 1) * b.down[i * F + n]);
  };
  const unitRead = (layer: number, n: number): [Float32Array, Float32Array] => {
    const b = block(layer);
    const row = (w: Float32Array) => {
      const v = Float32Array.from({ length: d }, (_, i) => b.mlpInNorm[i] * w[n * d + i]);
      const norm = Math.max(Math.sqrt(sum(d, (i) => v[i] ** 2)), 1e-12);
      return v.map((x) => x / norm);
    };
    return [row(b.gate), row(b.up)];
  };
  /** `γ_ao ⊙ W_O[:, h·hd:(h+1)·hd]` `[d, hd]`. */
  const headWrite = (layer: number, h: number): Float32Array => {
    const b = block(layer);
    const out = new Float32Array(d * hd);
    for (let i = 0; i < d; i += 1) {
      for (let j = 0; j < hd; j += 1) out[i * hd + j] = (b.attnOutNorm ? b.attnOutNorm[i] : 1) * b.o[i * H * hd + h * hd + j];
    }
    return out;
  };
  const gramOf = (write: Float32Array): Float32Array => {
    const g = new Float32Array(hd * hd);
    for (let a = 0; a < hd; a += 1) {
      for (let c = 0; c < hd; c += 1) g[a * hd + c] = sum(d, (i) => write[i * hd + a] * write[i * hd + c]);
    }
    return g;
  };
  const allGrams = (() => {
    let cache: Float32Array | null = null;
    return () => {
      if (cache) return cache;
      cache = new Float32Array(L * H * hd * hd);
      for (let layer = 0; layer < L; layer += 1) {
        for (let h = 0; h < H; h += 1) cache.set(gramOf(headWrite(layer, h)), (layer * H + h) * hd * hd);
      }
      return cache;
    };
  })();
  /** `sqrt(max(0, zᵀ G z)) · r_ao` of head `h` from a `z` row and its `attn_o` row. */
  const headNorm = (layer: number, h: number, z: ArrayLike<number>, zOffset: number, attnO: ArrayLike<number> | null, oOffset: number): number => {
    const g = allGrams().subarray((layer * H + h) * hd * hd, (layer * H + h + 1) * hd * hd);
    let q = 0;
    for (let a = 0; a < hd; a += 1) {
      let row = 0;
      for (let c = 0; c < hd; c += 1) row += g[a * hd + c] * z[zOffset + h * hd + c];
      q += z[zOffset + h * hd + a] * row;
    }
    const r = spec.sandwich && attnO ? rmsScale(attnO, oOffset, d, spec.rmsEps) : 1;
    return Math.sqrt(Math.max(0, q)) * r;
  };
  const finalRows = (rows: Float32Array, n: number): Float32Array => rmsNorm(rows, model.finalNorm, d, spec.rmsEps).subarray(0, n * d);
  const logitsOf = (normed: Float32Array, row: number, v: number): number => sum(d, (i) => normed[row * d + i] * model.embed[v * d + i]);
  const statistics = (handle: BlockSitesResident<"embedding_statistics">) => stored<StoredStatistics>(handle, "embedding_statistics");
  const lens = (handle: BlockSitesLensLayer) => stored<StoredLens>(handle, "lens_layer");
  const variance = (stats: StoredStatistics, u: ArrayLike<number>, offset: number): number => {
    let s = 0;
    for (let i = 0; i < d; i += 1) {
      let row = 0;
      for (let j = 0; j < d; j += 1) row += stats.sigmaE[i * d + j] * u[offset + j];
      s += u[offset + i] * row;
    }
    return s;
  };
  const centredEmbedding = (stats: StoredStatistics, v: number): Float32Array =>
    Float32Array.from({ length: d }, (_, i) => model.embed[v * d + i] * spec.embedScale - stats.meanE[i]);
  /** Top `k` of `values` by value desc, index asc, skipping `−∞`. */
  const topIndices = (values: ArrayLike<number>, k: number): number[] => {
    const order = Array.from({ length: values.length }, (_, i) => i).filter((i) => values[i] !== -Infinity);
    order.sort((a, b) => (values[b] - values[a]) || a - b);
    return order.slice(0, k);
  };
  const checkK = (k: number) => {
    if (!Number.isSafeInteger(k) || k < 1 || k > options.profile.maxLensReadTopK) {
      throw new RangeError(`k must be in [1, ${options.profile.maxLensReadTopK}]`);
    }
  };

  // ------------------------------------------------------------------ replay ---
  const openReplay = async (request: BlockSitesReplayRequest): Promise<BrowserBlockSitesReplay> => {
    requireUnits();
    const program = request.program;
    const sequence = new DenseSequence(model, options.contextTokens);
    const hookState: StructuredHookState | null = program ? initialStructuredHookState(program) : null;
    const anatomy = request.anatomyPosition === null ? null : new Float32Array(L * anatomyTapLayout(d, spec).stride);
    const layout = anatomyTapLayout(d, spec);
    const activity = request.activity;
    const state = activity ? initialActivityState(L, spec) : null;
    const traces = activity ? new Float32Array(activity.traces.length * activity.rowCount) : null;
    const population = activity && activity.populationRow !== null ? new Float32Array(L * (F + H)) : null;
    const scales = activity ? stored<Float32Array>(activity.scales, "unit_scales") : null;
    const patternLayers = request.patternLayers ?? [];
    const keys = new Map<number, Float32Array>(patternLayers.map((layer) => [layer, new Float32Array(options.contextTokens * KV * hd)]));
    const queries = new Map<number, Float32Array>();
    let open = true;
    const rowOf = (spans: BlockSitesActivityRequest["spans"], position: number): number | null => {
      for (const span of spans) {
        if (position >= span.firstPosition && position < span.firstPosition + span.rows) {
          const row = span.firstRow + (position - span.firstPosition);
          return row;
        }
      }
      return null;
    };
    const forward = async (segment: BlockSitesReplaySegment): Promise<BlockSitesForwardReadback> => {
      if (!open) throw new Error("The replay is closed");
      const controls: StructuredHookControlState | null = program ? segment.controls : null;
      const active = program && controls
        ? { ...program, affineActive: controls.affineActive, curveActive: controls.curveActive, ...(controls.unitActive ? { unitActive: controls.unitActive } : {}) }
        : null;
      const first = sequence.length;
      const T = segment.tokenIds.length;
      const probes = new Float32Array(L * (program?.profile.maxProbes ?? 1));
      const taps = program && hasStructuredUnitSection(program) ? new Float32Array(options.profile.maxUnitTaps) : null;
      const geometryWidth = program ? program.profile.maxGeometryProbes * program.profile.geometryOutputStride : 0;
      const geometry = program?.geometryActive !== undefined ? new Float32Array(L * geometryWidth) : null;
      const hidden = program?.hiddenCapture === true ? new Float32Array(L * d) : null;
      const prefill = first === 0 || T > 1;
      sequence.forward(segment.tokenIds, {
        head: active && hasStructuredUnitSection(active)
          ? (layer, z, positions) => runStructuredUnitSite(active, layer, "z", z, positions)
          : undefined,
        neuron: active && hasStructuredUnitSection(active)
          ? (layer, act, positions) => runStructuredUnitSite(active, layer, "act", act, positions)
          : undefined,
        postBlock: active
          ? (layer, outPre) => {
              const out = new Float32Array(outPre.length);
              for (let t = 0; t < T; t += 1) {
                const result = runStructuredHookLayer(active, hookState!, layer, outPre.subarray(t * d, (t + 1) * d), { decode: !prefill });
                out.set(result.residual, t * d);
                if (t === T - 1) {
                  probes.set(result.probes, layer * active.profile.maxProbes);
                  if (geometry !== null && result.geometry !== undefined) geometry.set(result.geometry, layer * geometryWidth);
                  hidden?.set(result.residual, layer * d);
                }
              }
              return out;
            }
          : undefined,
        capture(layer, site) {
          for (let t = 0; t < T; t += 1) {
            const position = first + t;
            if (anatomy && position === request.anatomyPosition) {
              const base = layer * layout.stride;
              const put = (offset: number, source: Float32Array, width: number) =>
                anatomy.set(source.subarray(t * width, (t + 1) * width), base + offset);
              put(layout.blockIn, site.blockIn, d);
              put(layout.mid, site.mid, d);
              put(layout.outPre, site.outPre, d);
              put(layout.out, site.out, d);
              put(layout.attnO, site.attnO, d);
              put(layout.mlpO, site.mlpO, d);
              put(layout.act, site.act, F);
              put(layout.z, site.z, H * hd);
              if (keys.has(layer)) queries.set(layer, site.queries.slice(t * H * hd, (t + 1) * H * hd));
            }
            if (keys.has(layer)) keys.get(layer)!.set(site.keys.subarray(t * KV * hd, (t + 1) * KV * hd), position * KV * hd);
            if (activity && state && scales) {
              const row = rowOf(activity.spans, position);
              if (row === null) continue;
              const excluded = position < UNIT_POSITIONS_FROM;
              const sl = activityStateLayout(spec);
              const base = layer * sl.stride;
              const sBase = layer * (2 * F + H);
              const value = (kind: "neuron" | "head", index: number): number => {
                if (excluded) return 0;
                if (kind === "neuron") {
                  const a = site.act[t * F + index];
                  const scale = a >= 0 ? scales[sBase + index] : scales[sBase + F + index];
                  return a / scale;
                }
                return headNorm(layer, index, site.z, t * H * hd, site.attnO, t * d) / scales[sBase + 2 * F + index];
              };
              const update = (valueAt: number, rowAt: number, rankAt: number, v: number) => {
                const rank = excluded ? -1 : Math.fround(Math.abs(v));
                if (rank > state[rankAt]) {
                  state[valueAt] = v;
                  state[rowAt] = row;
                  state[rankAt] = rank;
                }
              };
              for (let n = 0; n < F; n += 1) {
                const v = value("neuron", n);
                update(base + sl.neuronValue + n, base + sl.neuronRow + n, base + sl.neuronRank + n, v);
                if (population && row === activity.populationRow) population[layer * (F + H) + n] = v;
              }
              for (let h = 0; h < H; h += 1) {
                const v = value("head", h);
                update(base + sl.headValue + h, base + sl.headRow + h, base + sl.headRank + h, v);
                if (population && row === activity.populationRow) population[layer * (F + H) + F + h] = v;
              }
              activity.traces.forEach((unit, slot) => {
                if (unit.layer === layer) traces![slot * activity.rowCount + row] = value(unit.kind, unit.index);
              });
            }
          }
          if (taps && active && T > 0) {
            const { meta } = structuredUnitSectionLayout(options.profile, L, hd);
            const m = active.unitMeta!;
            const t = T - 1;
            for (let e = m[meta.tapNeuronOffsets.offset + layer]; e < m[meta.tapNeuronOffsets.offset + layer + 1]; e += 1) {
              taps[m[meta.tapNeuronSlot.offset + e]] = site.act[t * F + m[meta.tapNeuronIndex.offset + e]];
            }
            for (let e = m[meta.tapHeadOffsets.offset + layer]; e < m[meta.tapHeadOffsets.offset + layer + 1]; e += 1) {
              taps[m[meta.tapHeadSlot.offset + e]] = headNorm(layer, m[meta.tapHeadIndex.offset + e], site.z, t * H * hd, site.attnO, t * d);
            }
          }
        },
      });
      // The oracle has no J-lens dictionary: a kind-3 slot reads NaN, so a gate on it refuses.
      if (program) {
        for (let slot = 0; slot < probes.length; slot += 1) {
          if (program.probeKind[slot] === 3) probes[slot] = Number.NaN;
        }
      }
      return {
        measurements: f32Tensor(probes, [L, program?.profile.maxProbes ?? 1]),
        unitTaps: taps ? f32Tensor(taps, [taps.length]) : null,
        geometry: geometry === null ? null : f32Tensor(geometry, [L, program!.profile.maxGeometryProbes, program!.profile.geometryOutputStride]),
        hidden: hidden === null ? null : f32Tensor(hidden, [L, d]),
      };
    };
    return {
      forward,
      async finish(): Promise<BlockSitesReplayResult> {
        open = false;
        // The bridge's rule: a replay finishes having fed every position it was sized to.
        const tokenCount = request.tokenCount ?? options.contextTokens;
        if (sequence.length !== tokenCount) {
          throw new Error(`REPLAY_INCOMPLETE: ${sequence.length} of ${tokenCount} positions were fed`);
        }
        let patterns: BlockSitesResident<"pattern_capture"> | null = null;
        if (patternLayers.length && request.anatomyPosition !== null) {
          patterns = resident("pattern_capture", [patternLayers.length, KV, request.anatomyPosition + 1, hd], {
            layers: patternLayers,
            keys: patternLayers.map((layer) => keys.get(layer)!),
            queries: patternLayers.map((layer) => queries.get(layer)!),
            position: request.anatomyPosition,
          } satisfies StoredPatterns);
        }
        return {
          anatomyTaps: anatomy ? f32Tensor(anatomy, [L, layout.stride]) : null,
          activity: activity && state
            ? {
                state: f32Tensor(state, [L, activityStateLayout(spec).stride]),
                traces: f32Tensor(traces!, [activity.traces.length, activity.rowCount]),
                population: population ? f32Tensor(population, [L, F + H]) : null,
              }
            : null,
          patterns,
        };
      },
      async cancel() {
        open = false;
      },
    };
  };

  // ------------------------------------------------------------------ surface ---
  const backend: BrowserBlockSitesBackend = {
    capabilities: () => capabilities,

    async bindStructuredProgram(program: StructuredHookProgramBuffers) {
      if (!hasStructuredUnitSection(program)) return program;
      requireUnits();
      const { meta } = structuredUnitSectionLayout(options.profile, L, hd);
      const m = program.unitMeta!;
      const tapGrams = new Float32Array(program.tapGrams!.length);
      let slot = 0;
      for (let layer = 0; layer < L; layer += 1) {
        for (let e = m[meta.tapHeadOffsets.offset + layer]; e < m[meta.tapHeadOffsets.offset + layer + 1]; e += 1) {
          tapGrams.set(gramOf(headWrite(layer, m[meta.tapHeadIndex.offset + e])), slot * hd * hd);
          slot += 1;
        }
      }
      return { ...program, tapGrams };
    },

    async residentHeadGrams() {
      requireUnits();
      grams ??= resident("head_grams", [L, H, hd, hd], allGrams());
      return grams;
    },
    async uploadUnitScales(scales: F32Tensor) {
      requireUnits();
      requireTensorShape(scales, "f32", [L, 2 * F + H], "unit scales");
      return resident("unit_scales", scales.shape, Float32Array.from(scales.data));
    },
    async uploadLensLayer(family, site, layer, matrix) {
      requireTensorShape(matrix, "f32", [d, d], "lens matrix");
      if (site === "mid") requireUnits();
      return resident("lens_layer", [1, d, d], { family, site, layer, matrix: Float32Array.from(matrix.data) } satisfies StoredLens, { family, site, layer }) as BlockSitesLensLayer;
    },
    async uploadEmbeddingStatistics(value: BlockSitesEmbeddingStatistics) {
      requireTensorShape(value.meanE, "f32", [d], "mean_E");
      requireTensorShape(value.sigmaE, "f32", [d, d], "Σ_E");
      requireTensorShape(value.mask, "u32", [Math.ceil(V / 32)], "token mask");
      const keep = Uint8Array.from({ length: V }, (_, v) => (value.mask.data[v >>> 5] >>> (v & 31)) & 1);
      return resident("embedding_statistics", [d + d * d], { meanE: Float32Array.from(value.meanE.data), sigmaE: Float32Array.from(value.sigmaE.data), keep } satisfies StoredStatistics);
    },
    async release(handle) {
      if (handle.kind !== "head_grams") residents.delete(handle.id);
    },

    openReplay,
    async attentionRows(capture, heads) {
      requireUnits();
      const data = stored<StoredPatterns>(capture, "pattern_capture");
      const p = data.position;
      const out = new Float32Array(heads.length * (p + 1));
      heads.forEach(({ layer, head }, n) => {
        const at = data.layers.indexOf(layer);
        if (at < 0) throw new RangeError(`layer ${layer} was not captured`);
        const g = Math.floor(head / (H / KV));
        const local = isLocalAttentionLayer(spec, layer) && spec.slidingWindow !== null;
        const lo = local ? Math.max(0, p - spec.slidingWindow! + 1) : 0;
        const scores = new Float64Array(p + 1).fill(-Infinity);
        for (let j = lo; j <= p; j += 1) {
          let s = sum(hd, (i) => data.queries[at][head * hd + i] * data.keys[at][(j * KV + g) * hd + i]) * spec.attnScaling;
          if (spec.attnLogitSoftcap !== null) s = Math.tanh(s / spec.attnLogitSoftcap) * spec.attnLogitSoftcap;
          scores[j] = s;
        }
        const max = Math.max(...scores);
        const total = sum(p + 1, (j) => (scores[j] === -Infinity ? 0 : Math.exp(scores[j] - max)));
        for (let j = 0; j <= p; j += 1) out[n * (p + 1) + j] = scores[j] === -Infinity ? 0 : Math.exp(scores[j] - max) / total;
      });
      return f32Tensor(out, [heads.length, p + 1]);
    },

    async unitWeightRows(kind: UnitWeightRowsKind, list: readonly BlockSitesUnitRef[]) {
      requireUnits();
      const rows: Float32Array[] = list.map((unit) => {
        if (kind === "neuron_write") return neuronWrite(unit.layer, unit.index);
        if (kind === "neuron_read") {
          const [a, b] = unitRead(unit.layer, unit.index);
          const out = new Float32Array(2 * d);
          out.set(a, 0);
          out.set(b, d);
          return out;
        }
        if (kind === "head_write") return headWrite(unit.layer, unit.index);
        const b = block(unit.layer);
        const g = Math.floor(unit.index / (H / KV));
        const out = new Float32Array(3 * hd * d);
        [[b.q, unit.index], [b.k, g], [b.v, g]].forEach(([w, index], role) => {
          for (let r = 0; r < hd; r += 1) {
            for (let i = 0; i < d; i += 1) out[(role * hd + r) * d + i] = b.inputNorm[i] * (w as Float32Array)[((index as number) * hd + r) * d + i];
          }
        });
        return out;
      });
      const shape = kind === "neuron_write" ? [list.length, d] : kind === "neuron_read" ? [list.length, 2, d] : kind === "head_write" ? [list.length, d, hd] : [list.length, 3, hd, d];
      const out = new Float32Array(rows.reduce((a, r) => a + r.length, 0));
      let at = 0;
      for (const r of rows) {
        out.set(r, at);
        at += r.length;
      }
      return f32Tensor(out, shape);
    },
    async headWriteGrams(heads) {
      requireUnits();
      const out = new Float32Array(heads.length * hd * hd);
      heads.forEach(({ layer, head }, n) => out.set(gramOf(headWrite(layer, head)), n * hd * hd));
      return f32Tensor(out, [heads.length, hd, hd]);
    },
    async layerWriteScores(x, layers) {
      requireUnits();
      const [c, m] = [x.shape[0], x.shape[1]];
      requireTensorShape(x, "f32", [layers.length, null, d], "write-score rows");
      const out = new Float32Array(c * m * F);
      for (let l = 0; l < c; l += 1) {
        const b = block(layers[l]);
        for (let r = 0; r < m; r += 1) {
          for (let n = 0; n < F; n += 1) {
            out[(l * m + r) * F + n] = sum(d, (i) => x.data[(l * m + r) * d + i] * (b.mlpOutNorm ? b.mlpOutNorm[i] : 1) * b.down[i * F + n]);
          }
        }
      }
      return f32Tensor(out, [c, m, F]);
    },
    async layerReadScores(x, layers) {
      requireUnits();
      requireTensorShape(x, "f32", [layers.length, null, d], "read-score rows");
      const [c, m] = [x.shape[0], x.shape[1]];
      const out = new Float32Array(c * 2 * m * F);
      for (let l = 0; l < c; l += 1) {
        for (let n = 0; n < F; n += 1) {
          const [a, b] = unitRead(layers[l], n);
          for (let r = 0; r < m; r += 1) {
            out[((l * 2 + 0) * m + r) * F + n] = sum(d, (i) => a[i] * x.data[(l * m + r) * d + i]);
            out[((l * 2 + 1) * m + r) * F + n] = sum(d, (i) => b[i] * x.data[(l * m + r) * d + i]);
          }
        }
      }
      return f32Tensor(out, [c, 2, m, F]);
    },
    async neuronContributions(g, act, mlpO, layers) {
      requireUnits();
      const c = layers.length;
      const out = new Float32Array(c * F);
      for (let l = 0; l < c; l += 1) {
        const b = block(layers[l]);
        const r = spec.sandwich ? rmsScale(mlpO.data, l * d, d, spec.rmsEps) : 1;
        for (let n = 0; n < F; n += 1) {
          const dot = sum(d, (i) => g.data[l * d + i] * (b.mlpOutNorm ? b.mlpOutNorm[i] : 1) * b.down[i * F + n]);
          out[l * F + n] = act.data[l * F + n] * dot * r;
        }
      }
      return f32Tensor(out, [c, F]);
    },
    async headContributions(g, z, attnO, layers) {
      requireUnits();
      const c = layers.length;
      const out = new Float32Array(c * H);
      for (let l = 0; l < c; l += 1) {
        const b = block(layers[l]);
        const r = spec.sandwich ? rmsScale(attnO.data, l * d, d, spec.rmsEps) : 1;
        for (let h = 0; h < H; h += 1) {
          let s = 0;
          for (let j = 0; j < hd; j += 1) {
            const col = h * hd + j;
            s += sum(d, (i) => g.data[l * d + i] * (b.attnOutNorm ? b.attnOutNorm[i] : 1) * b.o[i * H * hd + col]) * z.data[l * H * hd + col];
          }
          out[l * H + h] = r * s;
        }
      }
      return f32Tensor(out, [c, H]);
    },
    async headWriteNorms(z, attnO, layers) {
      requireUnits();
      const c = layers.length;
      const out = new Float32Array(c * H);
      for (let l = 0; l < c; l += 1) {
        for (let h = 0; h < H; h += 1) out[l * H + h] = headNorm(layers[l], h, z.data, l * H * hd, attnO.data, l * d);
      }
      return f32Tensor(out, [c, H]);
    },
    async vocabRowsTimes(request) {
      const n = request.ids.length;
      const stats = request.kind === "embed" ? statistics(request.statistics!) : null;
      const rowOf = (v: number): Float32Array => request.kind === "embed"
        ? centredEmbedding(stats!, v)
        : Float32Array.from({ length: d }, (_, i) => model.finalNorm[i] * model.embed[v * d + i]);
      let factor: Float32Array;
      let m: number;
      if ("lens" in request.factor) {
        const matrix = lens(request.factor.lens).matrix;
        m = d;
        factor = request.factor.transpose
          ? Float32Array.from({ length: d * d }, (_, k) => matrix[(k % d) * d + Math.floor(k / d)])
          : matrix;
      } else {
        requireTensorShape(request.factor, "f32", [d, null], "vocabulary factor");
        m = request.factor.shape[1];
        factor = request.factor.data;
      }
      const out = new Float32Array(n * m);
      for (let r = 0; r < n; r += 1) {
        const row = rowOf(request.ids[r]);
        for (let j = 0; j < m; j += 1) out[r * m + j] = sum(d, (i) => row[i] * factor[i * m + j]);
      }
      return f32Tensor(out, [n, m]);
    },
    async vocabReadoutTopK(rows, k) {
      checkK(k);
      const n = rows.shape[0];
      const normed = finalRows(rows.data, n);
      const ids = new Int32Array(n * k);
      const logits = new Float32Array(n * k);
      const lse = new Float32Array(n);
      for (let r = 0; r < n; r += 1) {
        const all = Float64Array.from({ length: V }, (_, v) => logitsOf(normed, r, v));
        const max = Math.max(...all);
        lse[r] = max + Math.log(sum(V, (v) => Math.exp(all[v] - max)));
        topIndices(all, k).forEach((v, j) => {
          ids[r * k + j] = v;
          logits[r * k + j] = all[v];
        });
      }
      return { ids: i32Tensor(ids, [n, k]), logits: f32Tensor(logits, [n, k]), logSumExp: f32Tensor(lse, [n]) };
    },
    async vocabTokenLogits(rows, ids) {
      const n = rows.shape[0];
      const j = ids.shape[1];
      const normed = finalRows(rows.data, n);
      const out = new Float32Array(n * j);
      const lse = new Float32Array(n);
      for (let r = 0; r < n; r += 1) {
        const all = Float64Array.from({ length: V }, (_, v) => logitsOf(normed, r, v));
        const max = Math.max(...all);
        lse[r] = max + Math.log(sum(V, (v) => Math.exp(all[v] - max)));
        for (let c = 0; c < j; c += 1) out[r * j + c] = all[ids.data[r * j + c]];
      }
      return { logits: f32Tensor(out, [n, j]), logSumExp: f32Tensor(lse, [n]) };
    },
    async vocabLogitMoments(rows, stats) {
      const n = rows.shape[0];
      const keep = stats ? statistics(stats).keep : null;
      const normed = finalRows(rows.data, n);
      const out = new Float32Array(2 * n);
      for (let r = 0; r < n; r += 1) {
        const kept = Array.from({ length: V }, (_, v) => v).filter((v) => !keep || keep[v]);
        const x = kept.map((v) => logitsOf(normed, r, v));
        const mean = x.reduce((a, b) => a + b, 0) / x.length;
        const c = x.map((v) => v - mean);
        const variance = Math.max(c.reduce((a, b) => a + b * b, 0) / x.length, 1e-30);
        out[2 * r] = c.reduce((a, b) => a + b ** 3, 0) / x.length / variance ** 1.5;
        out[2 * r + 1] = c.reduce((a, b) => a + b ** 4, 0) / x.length / variance ** 2;
      }
      return f32Tensor(out, [n, 2]);
    },
    async footprintTopK(u, k, handle) {
      checkK(k);
      const stats = statistics(handle);
      const n = u.shape[0];
      const kept = stats.keep.reduce((a, b) => a + b, 0);
      const width = Math.min(k, kept);
      const ids = new Int32Array(n * width);
      const z = new Float32Array(n * width);
      for (let r = 0; r < n; r += 1) {
        const scale = Math.sqrt(Math.max(variance(stats, u.data, r * d), 1e-30));
        const offset = sum(d, (i) => u.data[r * d + i] * stats.meanE[i]);
        const all = Float64Array.from({ length: V }, (_, v) => stats.keep[v]
          ? (sum(d, (i) => u.data[r * d + i] * model.embed[v * d + i]) * spec.embedScale - offset) / scale
          : -Infinity);
        topIndices(all, width).forEach((v, j) => {
          ids[r * width + j] = v;
          z[r * width + j] = all[v];
        });
      }
      return { ids: i32Tensor(ids, [n, width]), z: f32Tensor(z, [n, width]) };
    },
    async unembedCotangent(hFinal, ids) {
      const n = ids.length;
      const out = new Float32Array(n * d);
      for (let r = 0; r < n; r += 1) {
        const scale = rmsScale(hFinal.data, r * d, d, spec.rmsEps);
        for (let i = 0; i < d; i += 1) out[r * d + i] = scale * model.finalNorm[i] * model.embed[ids[r] * d + i];
      }
      return f32Tensor(out, [n, d]);
    },
    async slensProbes(request) {
      const stats = statistics(request.statistics);
      const P = request.eTilde.shape[0];
      const out = new Float32Array(request.lens.length * P);
      request.lens.forEach((handle, c) => {
        const { layer, matrix } = lens(handle);
        const x = Float32Array.from({ length: d }, (_, i) => request.hidden.data[layer * d + i] - request.meansOut.data[layer * d + i]);
        const u = Float32Array.from({ length: d }, (_, j) => sum(d, (i) => x[i] * matrix[i * d + j]));
        const scale = Math.sqrt(Math.max(variance(stats, u, 0), 1e-30));
        for (let p = 0; p < P; p += 1) out[c * P + p] = sum(d, (i) => u[i] * request.eTilde.data[p * d + i]) / scale;
      });
      return f32Tensor(out, [request.lens.length, P]);
    },
    async footprintDirections(handles, ids, statsHandle) {
      const stats = statistics(statsHandle);
      const out = new Float32Array(handles.length * ids.length * d);
      handles.forEach((handle, c) => {
        const { matrix } = lens(handle);
        ids.forEach((v, n) => {
          const e = centredEmbedding(stats, v);
          for (let i = 0; i < d; i += 1) out[(c * ids.length + n) * d + i] = sum(d, (j) => matrix[i * d + j] * e[j]);
        });
      });
      return f32Tensor(out, [handles.length, ids.length, d]);
    },

    async lensTransport(rows, handles, transpose) {
      requireTensorShape(rows, "f32", [handles.length, null, d], "transport rows");
      const m = rows.shape[1];
      const out = new Float32Array(rows.data.length);
      handles.forEach((handle, c) => {
        const { matrix } = lens(handle);
        for (let r = 0; r < m; r += 1) {
          for (let j = 0; j < d; j += 1) {
            out[(c * m + r) * d + j] = sum(d, (i) => rows.data[(c * m + r) * d + i] * (transpose ? matrix[j * d + i] : matrix[i * d + j]));
          }
        }
      });
      return f32Tensor(out, rows.shape);
    },
    async batchedMatmul(a, b) {
      const [B, m, k] = a.shape;
      const n = b.shape[2];
      requireTensorShape(b, "f32", [B, k, null], "matmul right");
      const out = new Float32Array(B * m * n);
      for (let z = 0; z < B; z += 1) {
        for (let i = 0; i < m; i += 1) {
          for (let j = 0; j < n; j += 1) out[(z * m + i) * n + j] = sum(k, (t) => a.data[(z * m + i) * k + t] * b.data[(z * k + t) * n + j]);
        }
      }
      return f32Tensor(out, [B, m, n]);
    },
    async batchedQuadraticForm(x, matrix) {
      const [B, n] = x.shape;
      const data = "data" in matrix ? matrix : null;
      const sigma = data ? null : statistics(matrix as BlockSitesResident<"embedding_statistics">).sigmaE;
      const shared = data ? data.shape.length === 2 : true;
      const out = new Float32Array(B);
      for (let z = 0; z < B; z += 1) {
        const M = sigma ?? data!.data;
        const base = shared ? 0 : z * n * n;
        let s = 0;
        for (let i = 0; i < n; i += 1) s += x.data[z * n + i] * sum(n, (j) => M[base + i * n + j] * x.data[z * n + j]);
        out[z] = s;
      }
      return f32Tensor(out, [B]);
    },
    async groupedRowNorms(x, width) {
      const [n, total] = x.shape;
      const g = total / width;
      const out = new Float32Array(n * g);
      for (let r = 0; r < n; r += 1) {
        for (let c = 0; c < g; c += 1) out[r * g + c] = Math.sqrt(sum(width, (i) => x.data[r * total + c * width + i] ** 2));
      }
      return f32Tensor(out, [n, g]);
    },
    async chanMerge(state, rows) {
      const [r, dim] = rows.shape;
      if (r === 0) return state;
      const mean = new Float32Array(dim);
      const m2 = new Float32Array(dim);
      for (let i = 0; i < dim; i += 1) {
        const chunkMean = sum(r, (t) => rows.data[t * dim + i]) / r;
        const chunkM2 = sum(r, (t) => (rows.data[t * dim + i] - chunkMean) ** 2);
        const total = state.count + r;
        const delta = chunkMean - state.mean.data[i];
        mean[i] = state.mean.data[i] + (delta * r) / total;
        m2[i] = state.m2.data[i] + chunkM2 + (delta * delta * state.count * r) / total;
      }
      return { count: state.count + r, mean: f32Tensor(mean, [dim]), m2: f32Tensor(m2, [dim]) };
    },
    async logSumExp(x) {
      const [n, m] = x.shape;
      const out = new Float32Array(n);
      for (let r = 0; r < n; r += 1) {
        let max = -Infinity;
        for (let j = 0; j < m; j += 1) max = Math.max(max, x.data[r * m + j]);
        out[r] = max + Math.log(sum(m, (j) => Math.exp(x.data[r * m + j] - max)));
      }
      return f32Tensor(out, [n]);
    },
    async topK(x, k) {
      const [n, m] = x.shape;
      if (!Number.isSafeInteger(k) || k < 1 || k > 1024) throw new RangeError("top-k takes 1 to 1024");
      const width = Math.min(k, m);
      const values = new Float32Array(n * width);
      const indices = new Int32Array(n * width);
      for (let r = 0; r < n; r += 1) {
        topIndices(x.data.subarray(r * m, (r + 1) * m), width).forEach((i, j) => {
          values[r * width + j] = x.data[r * m + i];
          indices[r * width + j] = i;
        });
      }
      return { values: f32Tensor(values, [n, width]), indices: i32Tensor(indices, [n, width]) };
    },
  };
  return backend;
}

/** The final logits of a replay's last hidden row (the reference's own head; for tests). */
export function cpuFinalLogits(model: DenseModelWeights, hidden: Float32Array): Float32Array {
  return finalLogits(model, hidden, hidden.length - model.hiddenSize);
}

export type { I32Tensor };
