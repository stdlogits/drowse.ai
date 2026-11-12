// Hook ABI v6's static reads and exact-readout-v2 over the WebLLM bridge (B1 §4.1,
// browser-runtime/HOOK_ABI.md §8). Each member validates its arguments, sends one `read` request
// (a short program of library calls over host tensors, residents and earlier outputs) and gets
// its outputs back with one device sync, so a member costs one round trip as native's packed
// readback does.
import type {
  DrowseBlockSitesRequest,
  DrowseReadArg,
  DrowseReadOutputs,
  DrowseReadStep,
} from "@drowse/web-llm";
import {
  f32Tensor,
  i32Tensor,
  requireTensorShape,
  type BlockSitesCapabilities,
  type BlockSitesEmbeddingStatistics,
  type BlockSitesLensLayer,
  type BlockSitesResident,
  type BlockSitesUnitRef,
  type BrowserBlockSitesBackend,
  type F32Tensor,
  type I32Tensor,
  type LensFamily,
  type LensSite,
  type UnitWeightRowsKind,
  unitRowsPerCall,
  VOCAB_READ_ROWS,
} from "./modelBackend";

/** The device's stand-in for −∞ (WGSL has no infinity literal): masked footprint ids and padded
 *  top-k entries come back as this and are reported as `-Infinity`. */
export const DEVICE_LOWEST = -3.4028234663852886e38;
const QUADRATIC_FLOOR = 1e-30;
const MAX_TOP_K = 1024;

type HostArray = Float32Array | Int32Array | Uint32Array;
type WebLlmReadArg = DrowseReadArg;
type WebLlmReadStep = DrowseReadStep;

/** The static-read requests of the bridge's `drowseBlockSites`. */
export type WebLlmStaticReadRequest = Extract<DrowseBlockSitesRequest, { op: "uploadResident" | "read" }>;

export interface WebLlmStaticReadDevice {
  call(request: WebLlmStaticReadRequest): Promise<unknown>;
}

type Members = Partial<Record<keyof BrowserBlockSitesBackend, unknown>>;

const host = (data: HostArray, shape: readonly number[]): WebLlmReadArg => ({
  tensor: { data, shape: [...shape] },
});

function finiteF32(value: F32Tensor, label: string, shape: readonly (number | null)[]): F32Tensor {
  requireTensorShape(value, "f32", shape, label);
  if (!value.data.every(Number.isFinite)) throw new RangeError(`${label} must be finite`);
  return value;
}

/** `-Infinity` to the device's lowest float; NaN and `+Infinity` are refused. */
function toDevice(value: F32Tensor, label: string): Float32Array {
  const out = new Float32Array(value.data.length);
  value.data.forEach((x, i) => {
    if (x === -Infinity) out[i] = DEVICE_LOWEST;
    else if (Number.isFinite(x)) out[i] = x;
    else throw new RangeError(`${label} must be finite or -Infinity`);
  });
  return out;
}

function fromDevice(values: Float32Array): Float32Array {
  return values.map((x) => (x <= DEVICE_LOWEST ? -Infinity : x));
}

/** The static-read members of a v5 library's block-sites surface (B1 §4.1). Unit members exist
 *  only on a units library; the caller installs the refusal on a residual one. */
export function webLlmStaticReads(
  capabilities: BlockSitesCapabilities,
  device: WebLlmStaticReadDevice,
  headGramsId: () => Promise<string>,
): Members {
  const units = capabilities.profile.unitSites === 1;
  const d = capabilities.hiddenSize;
  const L = capabilities.layerCount;
  const V = capabilities.vocabSize;
  const spec = capabilities.blockSpec;
  const topK = capabilities.profile.maxLensReadTopK;

  const run = async (steps: WebLlmReadStep[], read: string[]) => {
    const out = (await device.call({ op: "read", steps, read })) as DrowseReadOutputs;
    for (const name of read) {
      if (out?.tensors?.[name] === undefined) throw new TypeError(`The bridge read back no ${name}`);
    }
    return out.tensors;
  };
  const f32 = (o: { data: HostArray; shape: number[] }, shape?: readonly number[]) =>
    f32Tensor(o.data as Float32Array, shape ?? o.shape);
  /** `read` over `rows` in calls of at most `VOCAB_READ_ROWS` rows (each call's `[rows, V]`
   *  device output stays within the library's binding), its row-major outputs joined by row. */
  const inRowChunks = async <T extends Record<string, F32Tensor | I32Tensor>>(
    rows: F32Tensor,
    read: (chunk: F32Tensor, first: number) => Promise<T>,
  ): Promise<T> => {
    const [n, width] = rows.shape;
    if (n <= VOCAB_READ_ROWS) return read(rows, 0);
    const parts: T[] = [];
    for (let first = 0; first < n; first += VOCAB_READ_ROWS) {
      const last = Math.min(n, first + VOCAB_READ_ROWS);
      parts.push(await read(f32Tensor(rows.data.slice(first * width, last * width), [last - first, width]), first));
    }
    const joined: Record<string, F32Tensor | I32Tensor> = {};
    for (const name of Object.keys(parts[0])) {
      const head = parts[0][name];
      const shape = [n, ...head.shape.slice(1)];
      if (head.dtype === "i32") {
        const data = new Int32Array(parts.reduce((sum, part) => sum + part[name].data.length, 0));
        let at = 0;
        for (const part of parts) { data.set(part[name].data as Int32Array, at); at += part[name].data.length; }
        joined[name] = i32Tensor(data, shape);
      } else {
        const data = new Float32Array(parts.reduce((sum, part) => sum + part[name].data.length, 0));
        let at = 0;
        for (const part of parts) { data.set(part[name].data as Float32Array, at); at += part[name].data.length; }
        joined[name] = f32Tensor(data, shape);
      }
    }
    return joined as T;
  };

  const ids1 = (ids: Int32Array, label: string): Int32Array => {
    if (!(ids instanceof Int32Array) || ids.length < 1) {
      throw new TypeError(`${label} must be a non-empty Int32Array`);
    }
    if (ids.some((id) => id < 0 || id >= V)) throw new RangeError(`${label} must be vocabulary ids`);
    return ids;
  };
  const layerList = (layers: Int32Array, count: number, label: string): Int32Array => {
    if (!(layers instanceof Int32Array) || layers.length !== count) {
      throw new TypeError(`${label} must be an Int32Array of ${count} layers`);
    }
    if (layers.some((layer) => layer < 0 || layer >= L)) throw new RangeError(`${label} out of range`);
    return layers;
  };
  const statisticsId = (statistics: BlockSitesResident<"embedding_statistics"> | undefined) => {
    if (statistics?.kind !== "embedding_statistics") {
      throw new TypeError("This read needs the resident S embedding statistics");
    }
    return statistics.id;
  };
  const lensLayer = (lens: BlockSitesLensLayer, family: LensFamily | null, label: string) => {
    if (lens?.kind !== "lens_layer" || (family !== null && lens.family !== family)) {
      throw new TypeError(`${label} must be a resident ${family === null ? "" : `${family.toUpperCase()} `}lens layer`);
    }
    return lens.id;
  };
  const checkK = (k: number, most: number, width: number) => {
    if (!Number.isSafeInteger(k) || k < 1 || k > most || k > width) {
      throw new RangeError(`k must be in [1, ${Math.min(most, width)}]`);
    }
  };
  const topkSteps = (input: string, n: number, k: number): WebLlmReadStep => ({
    fn: "drowse_topk",
    args: [{ ref: input }, host(new Float32Array(n * k), [n, k]), host(new Int32Array(n * k), [n, k])],
    out: ["values", "ids"],
  });

  const members: Members = {};

  members.uploadLensLayer = async (
    family: LensFamily,
    site: LensSite,
    layer: number,
    matrix: F32Tensor,
  ): Promise<BlockSitesLensLayer> => {
    if ((family !== "r" && family !== "s") || (site !== "out" && site !== "mid")) {
      throw new TypeError("A lens layer is R or S at out or mid");
    }
    if (!Number.isSafeInteger(layer) || layer < 0 || layer >= L) throw new RangeError("Lens layer out of range");
    finiteF32(matrix, "A lens matrix", [d, d]);
    const handle = (await device.call({
      op: "uploadResident",
      kind: "lens_layer",
      tensors: [{ data: matrix.data, shape: [1, d, d] }],
    })) as { id: string; bytes: number };
    return Object.freeze({
      kind: "lens_layer", id: handle.id, bytes: handle.bytes, shape: Object.freeze([1, d, d]),
      family, site, layer,
    });
  };

  members.uploadEmbeddingStatistics = async (
    statistics: BlockSitesEmbeddingStatistics,
  ): Promise<BlockSitesResident<"embedding_statistics">> => {
    finiteF32(statistics.meanE, "mean_E", [d]);
    finiteF32(statistics.sigmaE, "Σ_E", [d, d]);
    requireTensorShape(statistics.mask, "u32", [Math.ceil(V / 32)], "The token mask");
    const handle = (await device.call({
      op: "uploadResident",
      kind: "embedding_statistics",
      tensors: [
        { data: statistics.meanE.data, shape: [d] },
        { data: statistics.sigmaE.data, shape: [d, d] },
        { data: statistics.mask.data, shape: [statistics.mask.data.length] },
      ],
    })) as { id: string; bytes: number };
    return Object.freeze({
      kind: "embedding_statistics", id: handle.id, bytes: handle.bytes, shape: Object.freeze([d]),
    });
  };

  // ---- vocabulary and footprints (every library) ----
  const rowsStep = (
    kind: "embed" | "unembed",
    ids: Int32Array,
    statistics?: BlockSitesResident<"embedding_statistics">,
  ): WebLlmReadStep =>
    kind === "embed"
      ? {
          fn: "drowse_embedding_rows",
          args: [host(ids, [ids.length]), { resident: statisticsId(statistics), index: 0 }],
          out: "rows",
        }
      : { fn: "drowse_unembedding_rows", args: [host(ids, [ids.length])], out: "rows" };

  members.vocabRowsTimes = async (request: {
    readonly kind: "embed" | "unembed";
    readonly ids: Int32Array;
    readonly factor: F32Tensor | { readonly lens: BlockSitesLensLayer; readonly transpose: boolean };
    readonly statistics?: BlockSitesResident<"embedding_statistics">;
  }): Promise<F32Tensor> => {
    if (request.kind !== "embed" && request.kind !== "unembed") throw new TypeError("kind is embed or unembed");
    const ids = ids1(request.ids, "Vocabulary ids");
    const n = ids.length;
    let product: WebLlmReadStep;
    let m: number;
    if ("lens" in request.factor) {
      const id = lensLayer(request.factor.lens, null, "The factor");
      m = d;
      product = {
        fn: request.factor.transpose ? "drowse_batched_matmul_nt" : "drowse_batched_matmul",
        args: [{ ref: "rows", shape: [1, n, d] }, { resident: id }],
        out: "out",
      };
    } else {
      finiteF32(request.factor, "The factor", [d, null]);
      m = request.factor.shape[1];
      product = {
        fn: "drowse_batched_matmul",
        args: [{ ref: "rows", shape: [1, n, d] }, host(request.factor.data, [1, d, m])],
        out: "out",
      };
    }
    const out = await run([rowsStep(request.kind, ids, request.statistics), product], ["out"]);
    return f32(out.out, [n, m]);
  };

  members.vocabReadoutTopK = async (rows: F32Tensor, k: number) => {
    finiteF32(rows, "Readout rows", [null, d]);
    checkK(k, topK, V);
    return inRowChunks(rows, async (chunk) => {
      const n = chunk.shape[0];
      const out = await run([
        { fn: "drowse_vocab_logits", args: [host(chunk.data, [n, d])], out: "logits" },
        { fn: "drowse_log_sum_exp", args: [{ ref: "logits" }], out: "lse" },
        topkSteps("logits", n, k),
      ], ["values", "ids", "lse"]);
      return {
        ids: i32Tensor(out.ids.data as Int32Array, [n, k]),
        logits: f32(out.values, [n, k]),
        logSumExp: f32(out.lse, [n]),
      };
    });
  };

  members.vocabTokenLogits = async (rows: F32Tensor, ids: I32Tensor) => {
    finiteF32(rows, "Readout rows", [null, d]);
    requireTensorShape(ids, "i32", [rows.shape[0], null], "Token ids");
    ids1(ids.data, "Token ids");
    const width = ids.shape[1];
    return inRowChunks(rows, async (chunk, first) => {
      const n = chunk.shape[0];
      const out = await run([
        { fn: "drowse_vocab_logits", args: [host(chunk.data, [n, d])], out: "logits" },
        { fn: "drowse_log_sum_exp", args: [{ ref: "logits" }], out: "lse" },
      ], ["logits", "lse"]);
      const logits = out.logits.data as Float32Array;
      const picked = new Float32Array(n * width);
      for (let row = 0; row < n; row += 1) {
        for (let j = 0; j < width; j += 1) {
          picked[row * width + j] = logits[row * V + ids.data[(first + row) * width + j]];
        }
      }
      return { logits: f32Tensor(picked, [n, width]), logSumExp: f32(out.lse, [n]) };
    });
  };

  members.vocabLogitMoments = async (
    rows: F32Tensor,
    statistics?: BlockSitesResident<"embedding_statistics">,
  ): Promise<F32Tensor> => {
    finiteF32(rows, "Readout rows", [null, d]);
    const mask: WebLlmReadArg = statistics === undefined
      ? host(new Uint32Array(1), [1])
      : { resident: statisticsId(statistics), index: 2 };
    const { moments } = await inRowChunks(rows, async (chunk) => {
      const n = chunk.shape[0];
      const out = await run([
        { fn: "drowse_vocab_logits", args: [host(chunk.data, [n, d])], out: "logits" },
        {
          fn: "drowse_logit_moments",
          args: [{ ref: "logits" }, mask, host(new Int32Array([statistics === undefined ? 0 : 1]), [1])],
          out: "moments",
        },
      ], ["moments"]);
      return { moments: f32(out.moments, [n, 2]) };
    });
    return moments;
  };

  members.footprintTopK = async (
    u: F32Tensor,
    k: number,
    statistics: BlockSitesResident<"embedding_statistics">,
  ) => {
    finiteF32(u, "Covectors", [null, d]);
    checkK(k, topK, V);
    const id = statisticsId(statistics);
    return inRowChunks(u, async (chunk) => {
      const n = chunk.shape[0];
      const out = await run([
        {
          fn: "drowse_footprint_scores",
          args: [host(chunk.data, [n, d]), { resident: id, index: 0 }, { resident: id, index: 1 }, { resident: id, index: 2 }],
          out: "scores",
        },
        topkSteps("scores", n, k),
      ], ["values", "ids"]);
      return {
        ids: i32Tensor(out.ids.data as Int32Array, [n, k]),
        z: f32Tensor(fromDevice(out.values.data as Float32Array), [n, k]),
      };
    });
  };

  members.slensProbes = async (request: {
    readonly hidden: F32Tensor;
    readonly meansOut: F32Tensor;
    readonly lens: readonly BlockSitesLensLayer[];
    readonly eTilde: F32Tensor;
    readonly statistics: BlockSitesResident<"embedding_statistics">;
  }): Promise<F32Tensor> => {
    finiteF32(request.hidden, "Hidden states", [L, d]);
    finiteF32(request.meansOut, "out means", [L, d]);
    finiteF32(request.eTilde, "Probe embeddings", [null, d]);
    const id = statisticsId(request.statistics);
    const probes = request.eTilde.shape[0];
    if (!Array.isArray(request.lens) || request.lens.length < 1) throw new TypeError("slensProbes needs S:out layers");
    const steps: WebLlmReadStep[] = [];
    const read: string[] = [];
    request.lens.forEach((lens, i) => {
      const lensId = lensLayer(lens, "s", "An S probe lens");
      if (lens.site !== "out") throw new TypeError("S probes read the S:out lens");
      const x = new Float32Array(d);
      for (let k = 0; k < d; k += 1) {
        x[k] = request.hidden.data[lens.layer * d + k] - request.meansOut.data[lens.layer * d + k];
      }
      steps.push(
        { fn: "drowse_batched_matmul", args: [host(x, [1, 1, d]), { resident: lensId }], out: `u${i}` },
        {
          fn: "drowse_batched_matmul_nt",
          args: [{ ref: `u${i}` }, host(request.eTilde.data, [1, probes, d])],
          out: `dot${i}`,
        },
        {
          fn: "drowse_quadratic_form",
          args: [{ ref: `u${i}`, shape: [1, d] }, { resident: id, index: 1, shape: [1, d, d] }],
          out: `q${i}`,
        },
      );
      read.push(`dot${i}`, `q${i}`);
    });
    const out = await run(steps, read);
    const z = new Float32Array(request.lens.length * probes);
    request.lens.forEach((_, i) => {
      const root = Math.sqrt(Math.max((out[`q${i}`].data as Float32Array)[0], QUADRATIC_FLOOR));
      const dots = out[`dot${i}`].data as Float32Array;
      for (let p = 0; p < probes; p += 1) z[i * probes + p] = dots[p] / root;
    });
    return f32Tensor(z, [request.lens.length, probes]);
  };

  members.footprintDirections = async (
    lens: readonly BlockSitesLensLayer[],
    ids: Int32Array,
    statistics: BlockSitesResident<"embedding_statistics">,
  ): Promise<F32Tensor> => {
    ids1(ids, "Footprint ids");
    if (!Array.isArray(lens) || lens.length < 1) throw new TypeError("Footprint directions need S:out layers");
    const n = ids.length;
    const steps: WebLlmReadStep[] = [rowsStep("embed", ids, statistics)];
    lens.forEach((layer, i) => {
      const id = lensLayer(layer, "s", "A footprint lens");
      steps.push({
        fn: "drowse_batched_matmul_nt",
        args: [{ ref: "rows", shape: [1, n, d] }, { resident: id }],
        out: `f${i}`,
      });
    });
    const names = lens.map((_, i) => `f${i}`);
    const out = await run(steps, names);
    const data = new Float32Array(lens.length * n * d);
    names.forEach((name, i) => data.set(out[name].data as Float32Array, i * n * d));
    return f32Tensor(data, [lens.length, n, d]);
  };

  // ---- exact-readout-v2 utilities (weight-free) ----
  members.lensTransport = async (
    rows: F32Tensor,
    lens: readonly BlockSitesLensLayer[],
    transpose: boolean,
  ): Promise<F32Tensor> => {
    finiteF32(rows, "Transport rows", [lens.length, null, d]);
    const m = rows.shape[1];
    const names = lens.map((_, i) => `t${i}`);
    const out = await run(lens.map((layer, i) => ({
      fn: transpose ? "drowse_batched_matmul_nt" : "drowse_batched_matmul",
      args: [host(rows.data.slice(i * m * d, (i + 1) * m * d), [1, m, d]), { resident: lensLayer(layer, null, "A transport lens") }],
      out: names[i],
    })), names);
    const data = new Float32Array(lens.length * m * d);
    names.forEach((name, i) => data.set(out[name].data as Float32Array, i * m * d));
    return f32Tensor(data, [lens.length, m, d]);
  };

  members.batchedMatmul = async (a: F32Tensor, b: F32Tensor): Promise<F32Tensor> => {
    finiteF32(a, "a", [null, null, null]);
    const [batch, m, k] = a.shape;
    finiteF32(b, "b", [null, k, null]);
    if (b.shape[0] !== batch && b.shape[0] !== 1) throw new TypeError("b's batch must be a's or 1");
    const n = b.shape[2];
    const out = await run([{
      fn: "drowse_batched_matmul", args: [host(a.data, a.shape), host(b.data, b.shape)], out: "out",
    }], ["out"]);
    return f32(out.out, [batch, m, n]);
  };

  members.batchedQuadraticForm = async (
    x: F32Tensor,
    matrix: F32Tensor | BlockSitesResident<"embedding_statistics">,
  ): Promise<F32Tensor> => {
    finiteF32(x, "x", [null, null]);
    const [b, n] = x.shape;
    let m: WebLlmReadArg;
    if ("data" in matrix) {
      if (matrix.shape.length === 2) finiteF32(matrix, "M", [n, n]);
      else finiteF32(matrix, "M", [null, n, n]);
      if (matrix.shape.length === 3 && matrix.shape[0] !== b && matrix.shape[0] !== 1) {
        throw new TypeError("M's batch must be x's or 1");
      }
      m = host(matrix.data, matrix.shape.length === 2 ? [1, n, n] : matrix.shape);
    } else {
      if (n !== d) throw new TypeError("Σ_E reads d-wide rows");
      m = { resident: statisticsId(matrix), index: 1, shape: [1, d, d] };
    }
    const out = await run([{ fn: "drowse_quadratic_form", args: [host(x.data, [b, n]), m], out: "q" }], ["q"]);
    return f32(out.q, [b]);
  };

  members.groupedRowNorms = async (x: F32Tensor, width: number): Promise<F32Tensor> => {
    finiteF32(x, "x", [null, null]);
    const [n, total] = x.shape;
    if (!Number.isSafeInteger(width) || width < 1 || total % width !== 0) {
      throw new RangeError("The group width must divide the row");
    }
    const groups = total / width;
    const out = await run([{
      fn: "drowse_grouped_row_norms", args: [host(x.data, [n, groups, width])], out: "norms",
    }], ["norms"]);
    return f32(out.norms, [n, groups]);
  };

  members.chanMerge = async (
    state: { readonly count: number; readonly mean: F32Tensor; readonly m2: F32Tensor },
    rows: F32Tensor,
  ) => {
    finiteF32(rows, "Rows", [null, null]);
    const [r, width] = rows.shape;
    finiteF32(state.mean, "The running mean", [width]);
    finiteF32(state.m2, "The running m2", [width]);
    if (!Number.isSafeInteger(state.count) || state.count < 0) throw new RangeError("count must be a count");
    const out = await run([{
      fn: "drowse_chan_merge",
      args: [
        host(new Float32Array([state.count]), [1]),
        host(state.mean.data, [width]),
        host(state.m2.data, [width]),
        host(rows.data, [r, width]),
      ],
      out: ["mean", "m2"],
    }], ["mean", "m2"]);
    return { count: state.count + r, mean: f32(out.mean, [width]), m2: f32(out.m2, [width]) };
  };

  members.logSumExp = async (x: F32Tensor): Promise<F32Tensor> => {
    requireTensorShape(x, "f32", [null, null], "x");
    const out = await run([{
      fn: "drowse_log_sum_exp", args: [host(toDevice(x, "x"), x.shape)], out: "lse",
    }], ["lse"]);
    return f32Tensor(fromDevice(out.lse.data as Float32Array), [x.shape[0]]);
  };

  members.topK = async (x: F32Tensor, k: number) => {
    requireTensorShape(x, "f32", [null, null], "x");
    const [n, m] = x.shape;
    checkK(k, MAX_TOP_K, m);
    const out = await run([
      {
        fn: "drowse_topk",
        args: [host(toDevice(x, "x"), x.shape), host(new Float32Array(n * k), [n, k]), host(new Int32Array(n * k), [n, k])],
        out: ["values", "ids"],
      },
    ], ["values", "ids"]);
    return {
      values: f32Tensor(fromDevice(out.values.data as Float32Array), [n, k]),
      indices: i32Tensor(out.ids.data as Int32Array, [n, k]),
    };
  };

  if (!units || spec === null) return members;

  // ---- unit reads (units libraries) ----
  const unitTable = (refs: readonly BlockSitesUnitRef[], kind: "neuron" | "head"): Int32Array => {
    if (!Array.isArray(refs) || refs.length < 1) throw new TypeError("Unit reads need at least one unit");
    const table = new Int32Array(2 * refs.length);
    refs.forEach((ref, i) => {
      const limit = kind === "neuron" ? spec.dFF : spec.nHeads;
      if (ref?.kind !== kind || !Number.isSafeInteger(ref.layer) || ref.layer < 0 || ref.layer >= L ||
          !Number.isSafeInteger(ref.index) || ref.index < 0 || ref.index >= limit) {
        throw new RangeError(`Unit ${i} is not a ${kind} of this model`);
      }
      table[2 * i] = ref.layer;
      table[2 * i + 1] = ref.index;
    });
    return table;
  };
  const hd = spec.headDim;

  members.unitWeightRows = async (
    kind: UnitWeightRowsKind,
    refs: readonly BlockSitesUnitRef[],
  ): Promise<F32Tensor> => {
    const neuron = kind === "neuron_write" || kind === "neuron_read";
    if (!neuron && kind !== "head_write" && kind !== "head_read") throw new TypeError(`Unknown unit rows ${String(kind)}`);
    const table = unitTable(refs, neuron ? "neuron" : "head");
    const n = refs.length;
    // In calls of at most `unitRowsPerCall` units, so each call's output stays within the binding.
    const unitFloats = neuron ? 3 * d : 4 * hd * d;
    const perCall = unitRowsPerCall(spec, neuron ? "neuron" : "head");
    const rows = new Float32Array(n * unitFloats);
    for (let first = 0; first < n; first += perCall) {
      const last = Math.min(n, first + perCall);
      const out = await run([{
        fn: neuron ? "drowse_neuron_rows" : "drowse_head_rows",
        args: [host(table.slice(2 * first, 2 * last), [last - first, 2])],
        out: "rows",
      }], ["rows"]);
      const part = out.rows.data as Float32Array;
      if (part.length !== (last - first) * unitFloats) throw new TypeError("The bridge read back unit rows of the wrong size");
      rows.set(part, first * unitFloats);
    }
    if (kind === "neuron_write" || kind === "neuron_read") {
      const take = kind === "neuron_write" ? [0] : [1, 2];
      const data = new Float32Array(n * take.length * d);
      for (let u = 0; u < n; u += 1) {
        take.forEach((slot, j) => data.set(rows.subarray((u * 3 + slot) * d, (u * 3 + slot + 1) * d), (u * take.length + j) * d));
      }
      return f32Tensor(data, kind === "neuron_write" ? [n, d] : [n, 2, d]);
    }
    const block = hd * d;
    if (kind === "head_read") {
      const data = new Float32Array(n * 3 * block);
      for (let u = 0; u < n; u += 1) data.set(rows.subarray((u * 4 + 1) * block, (u * 4 + 4) * block), u * 3 * block);
      return f32Tensor(data, [n, 3, hd, d]);
    }
    const data = new Float32Array(n * block);
    for (let u = 0; u < n; u += 1) {
      for (let i = 0; i < hd; i += 1) {
        for (let k = 0; k < d; k += 1) data[u * block + k * hd + i] = rows[u * 4 * block + i * d + k];
      }
    }
    return f32Tensor(data, [n, d, hd]);
  };

  members.headWriteGrams = async (
    heads: readonly { layer: number; head: number }[],
  ): Promise<F32Tensor> => {
    const table = unitTable(heads.map(({ layer, head }) => ({ kind: "head", layer, index: head })), "head");
    const grams = await headGramsId();
    const out = await run([{
      fn: "drowse_gather_head_grams",
      args: [{ resident: grams }, host(table, [heads.length, 2])],
      out: "grams",
    }], ["grams"]);
    return f32(out.grams, [heads.length, hd, hd]);
  };

  const scores = (fn: string, slots: number) => async (x: F32Tensor, layers: Int32Array): Promise<F32Tensor> => {
    finiteF32(x, "Score rows", [null, null, d]);
    const [c, m] = x.shape;
    layerList(layers, c, "Score layers");
    const out = await run([{ fn, args: [host(x.data, [c, m, d]), host(layers, [c])], out: "scores" }], ["scores"]);
    return f32(out.scores, slots === 1 ? [c, m, spec.dFF] : [c, slots, m, spec.dFF]);
  };
  members.layerWriteScores = scores("drowse_layer_write_scores", 1);
  members.layerReadScores = scores("drowse_layer_read_scores", 2);

  members.neuronContributions = async (g: F32Tensor, act: F32Tensor, mlpO: F32Tensor, layers: Int32Array) => {
    finiteF32(g, "g", [null, d]);
    const c = g.shape[0];
    finiteF32(act, "act", [c, spec.dFF]);
    finiteF32(mlpO, "mlp_o", [c, d]);
    layerList(layers, c, "Contribution layers");
    const out = await run([{
      fn: "drowse_neuron_contributions",
      args: [host(g.data, [c, d]), host(act.data, [c, spec.dFF]), host(mlpO.data, [c, d]), host(layers, [c])],
      out: "c",
    }], ["c"]);
    return f32(out.c, [c, spec.dFF]);
  };

  members.headContributions = async (g: F32Tensor, z: F32Tensor, attnO: F32Tensor, layers: Int32Array) => {
    finiteF32(g, "g", [null, d]);
    const c = g.shape[0];
    finiteF32(z, "z", [c, spec.nHeads * hd]);
    finiteF32(attnO, "attn_o", [c, d]);
    layerList(layers, c, "Contribution layers");
    const out = await run([{
      fn: "drowse_head_contributions",
      args: [host(g.data, [c, d]), host(z.data, [c, spec.nHeads * hd]), host(attnO.data, [c, d]), host(layers, [c])],
      out: "c",
    }], ["c"]);
    return f32(out.c, [c, spec.nHeads]);
  };

  members.headWriteNorms = async (z: F32Tensor, attnO: F32Tensor, layers: Int32Array) => {
    finiteF32(z, "z", [null, spec.nHeads * hd]);
    const c = z.shape[0];
    finiteF32(attnO, "attn_o", [c, d]);
    layerList(layers, c, "Norm layers");
    const grams = await headGramsId();
    const out = await run([{
      fn: "drowse_head_write_norms",
      args: [host(z.data, [c, spec.nHeads * hd]), host(attnO.data, [c, d]), host(layers, [c]), { resident: grams }],
      out: "norms",
    }], ["norms"]);
    return f32(out.norms, [c, spec.nHeads]);
  };

  members.unembedCotangent = async (hFinal: F32Tensor, ids: Int32Array): Promise<F32Tensor> => {
    finiteF32(hFinal, "Final residuals", [null, d]);
    const n = hFinal.shape[0];
    ids1(ids, "Cotangent ids");
    if (ids.length !== n) throw new TypeError("One id per final residual");
    const out = await run([{
      fn: "drowse_unembed_cotangent", args: [host(hFinal.data, [n, d]), host(ids, [n])], out: "u",
    }], ["u"]);
    return f32(out.u, [n, d]);
  };

  return members;
}
