// The fixture runtimes' units members (dev and e2e only): the real host ports over the tiny
// gemma3 fixture model on the CPU reference backend, with packs synthesized from it (a short
// neutral pass for the statistics and means, the hand-built lenses, the token mask over the
// byte vocabulary, an identity whitener and no null basis). Answers are drowse-shaped and
// deterministic, not a model's: they let the Units room, token anatomy and the S lens render.
import type { LoomTreeJSON } from "../../../../../drowse/webui/src/lib/types";
import { STANDARD_STRUCTURED_HOOK_PROFILE_V4, STANDARD_STRUCTURED_HOOK_PROFILE_V4_RESIDUAL } from "../structuredHookProfile";
import { cpuBlockSitesBackend } from "./cpuBackend";
import { DenseSequence, finalLogits, rmsScale, type DenseModelWeights } from "./denseModel";
import { embeddingStatistics } from "./referenceContext";
import type { UnitStatistics } from "./scales";
import { tinyLenses, tinyModelWeights, tinyTokenizer, type TinyModelFile, type TinyTokenizer } from "./tinyModels";
import { BrowserUnitsService, type UnitsNodeReplayPort, type UnitsPackSource } from "./unitsService";

const ARCH = "gemma3";
const NEUTRAL_TEXTS = 8;
const MAX_ROWS = 48;

interface FixtureUnits {
  readonly model: DenseModelWeights;
  readonly tokenizer: TinyTokenizer;
  readonly packs: UnitsPackSource;
}

let loaded: Promise<FixtureUnits> | null = null;

function fixtureUnits(): Promise<FixtureUnits> {
  loaded ??= (async () => {
    const file = (await import("../fixtures/units-tiny-models-v1.json")).default as unknown as TinyModelFile & {
      neutral: { statements: string[] };
    };
    const record = file.arches[ARCH];
    const model = tinyModelWeights(file, ARCH);
    const tokenizer = tinyTokenizer(record);
    return { model, tokenizer, packs: synthesizedPacks(file, model, tokenizer) };
  })();
  return loaded;
}

/** Neutral statistics from a short pass (positions ≥ 1): means, `act_top` per pole, head means
 *  and write-norm tops; the neutral `out` and `mid` means. */
function neutralPass(model: DenseModelWeights, tokenizer: TinyTokenizer, texts: readonly string[]) {
  const { spec } = model;
  const L = model.layerCount;
  const d = model.hiddenSize;
  const F = spec.dFF;
  const H = spec.nHeads;
  const hd = spec.headDim;
  const sums = {
    act: new Float64Array(L * F), z: new Float64Array(L * H * hd),
    out: new Float64Array(L * d), mid: new Float64Array(L * d),
  };
  const top = new Float32Array(L * 2 * F);
  const headTop = new Float32Array(L * H);
  let rows = 0;
  for (const text of texts) {
    const ids = [tokenizer.bos ?? 0, ...tokenizer.encode(text).slice(0, MAX_ROWS)];
    const sequence = new DenseSequence(model, ids.length);
    sequence.forward(ids, {
      capture(layer, site) {
        for (let t = 1; t < ids.length; t += 1) {
          for (let n = 0; n < F; n += 1) {
            const a = site.act[t * F + n];
            sums.act[layer * F + n] += a;
            const pole = a >= 0 ? 0 : 1;
            top[(layer * 2 + pole) * F + n] = Math.max(top[(layer * 2 + pole) * F + n], Math.abs(a));
          }
          for (let i = 0; i < H * hd; i += 1) sums.z[layer * H * hd + i] += site.z[t * H * hd + i];
          for (let i = 0; i < d; i += 1) {
            sums.out[layer * d + i] += site.out[t * d + i];
            sums.mid[layer * d + i] += site.mid[t * d + i];
          }
          const r = rmsScale(site.attnO, t * d, d, spec.rmsEps);
          for (let h = 0; h < H; h += 1) {
            let norm = 0;
            for (let i = 0; i < d; i += 1) {
              let w = 0;
              for (let j = 0; j < hd; j += 1) w += model.blocks[layer].o[i * H * hd + h * hd + j] * site.z[t * H * hd + h * hd + j];
              norm += ((model.blocks[layer].attnOutNorm?.[i] ?? 1) * w) ** 2;
            }
            headTop[layer * H + h] = Math.max(headTop[layer * H + h], Math.sqrt(norm) * r);
          }
        }
      },
    });
    rows += ids.length - 1;
  }
  const mean = (values: Float64Array) => Float32Array.from(values, (x) => x / rows);
  const statistics: UnitStatistics = {
    documents: texts.length,
    neuronMean: mean(sums.act),
    neuronScale: top,
    headMean: mean(sums.z),
    headScale: headTop,
  };
  return { statistics, out: mean(sums.out), mid: mean(sums.mid) };
}

function synthesizedPacks(file: TinyModelFile & { neutral: { statements: string[] } }, model: DenseModelWeights, tokenizer: TinyTokenizer): UnitsPackSource {
  const d = model.hiddenSize;
  const V = model.vocabSize;
  const F = model.spec.dFF;
  const neutral = neutralPass(model, tokenizer, file.neutral.statements.slice(0, NEUTRAL_TEXTS));
  const lenses = tinyLenses(file, ARCH);
  const keep = Uint8Array.from({ length: V }, (_, v) => (tokenizer.byteOf[v] >= 0 && !tokenizer.decode([v]).includes("�") ? 1 : 0));
  const stats = embeddingStatistics(model, keep);
  const kept = Int32Array.from([...keep.keys()].filter((v) => keep[v]));
  const layerArtifacts = new Map<number, Map<string, Float32Array>>();
  const artifacts = (layer: number) => {
    let table = layerArtifacts.get(layer);
    if (table) return table;
    table = new Map();
    const block = model.blocks[layer];
    const writes = Array.from({ length: F }, (_, n) => Float32Array.from({ length: d }, (_, i) => (block.mlpOutNorm?.[i] ?? 1) * block.down[i * F + n]));
    const reads = (sign: number) => Array.from({ length: F }, (_, n) => {
      const unit = (w: Float32Array) => {
        const v = Float32Array.from({ length: d }, (_, i) => block.mlpInNorm[i] * w[n * d + i]);
        const norm = Math.max(Math.hypot(...v), 1e-12);
        return v.map((x) => x / norm);
      };
      const a = unit(block.gate);
      const b = unit(block.up);
      return a.map((x, i) => x + sign * b[i]);
    });
    const populations: [string, "out" | "mid", Float32Array[]][] = [
      ["writes", "out", writes], ["fires+", "mid", reads(1)], ["fires-", "mid", reads(-1)],
    ];
    for (const [name, site, rows] of populations) {
      const r = layer === model.layerCount - 1 && site === "out" ? new Map([[layer, identity(d)]]) : lenses.get(`R:${site}`);
      const J = r?.get(layer);
      if (J) {
        const logits = rows.map((row) => {
          const x = Float32Array.from({ length: d }, (_, j) => row.reduce((s, value, i) => s + value * J[j * d + i], 0));
          const h = new Float32Array(d);
          h.set(x);
          return { x, logits: finalLogits(model, h) };
        });
        table.set(`R.${name}.rho`, Float32Array.from(logits, ({ x }) => rmsScale(x, 0, d, model.spec.rmsEps)));
        table.set(`R.${name}.lse`, Float32Array.from(logits, ({ logits: l }) => lse(l, 1)));
        if (name === "writes") table.set(`R.${name}.lse_neg`, Float32Array.from(logits, ({ logits: l }) => lse(l, -1)));
        const { mean, std } = moments(logits.map(({ logits: l }) => l), 1e-4);
        table.set(`R.${name}.mean`, mean);
        table.set(`R.${name}.std`, std);
      }
      const K = lenses.get(`S:${site}`)?.get(layer);
      if (K) {
        const us = rows.map((row) => Float32Array.from({ length: d }, (_, j) => row.reduce((s, value, i) => s + value * K[i * d + j], 0)));
        table.set(`S.${name}.norm`, Float32Array.from(us, (u) => Math.max(Math.hypot(...u), 1e-12)));
        table.set(`S.${name}.footprint_std`, Float32Array.from(us, (u) => Math.sqrt(Math.max(quadratic(stats.sigmaE, u), 1e-30))));
      }
    }
    for (const [place, names] of [["writes", ["writes"]], ["fires", ["fires+", "fires-"]]] as const) {
      const site = place === "writes" ? "out" : "mid";
      const K = lenses.get(`S:${site}`)?.get(layer);
      if (!K) continue;
      const rows = populations.filter(([name]) => (names as readonly string[]).includes(name)).flatMap(([, , r]) => r);
      const scores = rows.map((row) => {
        const u = Float32Array.from({ length: d }, (_, j) => row.reduce((s, value, i) => s + value * K[i * d + j], 0));
        const norm = Math.max(Math.hypot(...u), 1e-12);
        return Float32Array.from({ length: V }, (_, v) => {
          let s = 0;
          for (let i = 0; i < d; i += 1) s += (u[i] / norm) * (model.embed[v * d + i] * model.spec.embedScale - stats.meanE[i]);
          return s;
        });
      });
      const { mean, std } = moments(scores, 1e-6);
      table.set(`S.${place}.mean`, mean);
      table.set(`S.${place}.std`, std);
    }
    table.set("neurons.write_norms", Float32Array.from(writes, (w) => Math.hypot(...w)));
    layerArtifacts.set(layer, table);
    return table;
  };
  const lensOf = (family: "R" | "S") => ({
    source: `fixture:${family.toLowerCase()}`,
    identity: `fixture-${family.toLowerCase()}-lens`,
    sites: (["out", "mid"] as const).filter((site) => lenses.has(`${family}:${site}`)),
    layers: (site: "out" | "mid") => [...(lenses.get(`${family}:${site}`)?.keys() ?? [])].sort((a, b) => a - b),
    fetch: async (site: "out" | "mid", layer: number) => lenses.get(`${family}:${site}`)!.get(layer)!,
  });
  return {
    band: { lo: file.band[0], hi: file.band[1] },
    statistics: async () => ({ statistics: neutral.statistics, absent: null }),
    means: async () => ({ out: neutral.out, mid: neutral.mid }),
    lens: lensOf,
    embeddingStatistics: async () => ({ ...stats, source: "tokenizer", detail: "the fixture's byte vocabulary", maskDigest: "tokenizer" }),
    layerArtifacts: async (layer) => artifacts(layer),
    nullProjection: async () => 0,
    headKeptIds: async () => kept,
    whitener: async () => ({
      columns: d, rank: 0, ridge: 1, mean: new Float64Array(d), basis: new Float64Array(0),
      eigenvalues: new Float64Array(0), inverseScales: new Float64Array(0),
    }),
  };
}

function identity(d: number): Float32Array {
  const out = new Float32Array(d * d);
  for (let i = 0; i < d; i += 1) out[i * d + i] = 1;
  return out;
}

function lse(values: Float32Array, sign: number): number {
  let max = -Infinity;
  for (const x of values) max = Math.max(max, sign * x);
  let s = 0;
  for (const x of values) s += Math.exp(sign * x - max);
  return max + Math.log(s);
}

function moments(rows: readonly Float32Array[], floor: number): { mean: Float32Array; std: Float32Array } {
  const width = rows[0].length;
  const mean = new Float32Array(width);
  const std = new Float32Array(width);
  for (let v = 0; v < width; v += 1) {
    let s = 0;
    let q = 0;
    for (const row of rows) {
      s += row[v];
      q += row[v] * row[v];
    }
    const m = s / rows.length;
    mean[v] = m;
    std[v] = Math.sqrt(Math.max(q / rows.length - m * m, floor * floor));
  }
  return { mean, std };
}

function quadratic(matrix: Float32Array, u: Float32Array): number {
  const d = u.length;
  let s = 0;
  for (let i = 0; i < d; i += 1) {
    let row = 0;
    for (let j = 0; j < d; j += 1) row += matrix[i * d + j] * u[j];
    s += u[i] * row;
  }
  return s;
}

/** Node replays over a fixture loom: a node's text as byte tokens after its path's texts. */
function fixtureReplay(tree: () => LoomTreeJSON, tokenizer: TinyTokenizer): UnitsNodeReplayPort {
  return {
    async nodeForward(nodeId, rawIndex) {
      const nodes = new Map(tree().nodes.map((entry) => [entry.id, entry]));
      const node = nodes.get(nodeId);
      if (node === undefined || node.parent_id === null) {
        throw Object.assign(new Error(`Conversation node ${nodeId} has no exact raw token record`), {
          code: "TOKEN_REPLAY_UNAVAILABLE", status: 409, recoverable: true,
        });
      }
      const path: string[] = [];
      let at = nodes.get(node.parent_id);
      while (at !== undefined) {
        if (at.text) path.unshift(at.text);
        at = at.parent_id === null ? undefined : nodes.get(at.parent_id);
      }
      const prompt = [tokenizer.bos ?? 0, ...tokenizer.encode(path.join("\n")).slice(-MAX_ROWS)];
      const rows = tokenizer.encode(node.text || " ").slice(0, MAX_ROWS);
      const anchor = rawIndex ?? rows.length - 1;
      if (!(anchor >= 0 && anchor < rows.length)) {
        throw Object.assign(new Error(`Raw token index ${anchor} is outside node ${nodeId}`), {
          code: "INVALID_NODE_OPERATION", status: 400, recoverable: true,
        });
      }
      return {
        forward: {
          ids: Int32Array.from([...prompt, ...rows.slice(0, anchor)]),
          promptLength: prompt.length,
          rawIds: Int32Array.from(rows),
          rawIndex: anchor,
          spans: [{ firstRow: 0, stopRow: rows.length, at: prompt.length - 1 }],
        },
        program: null,
        steering: null,
        canThink: false,
      };
    },
  };
}

/** The fixture's units members over `tree` (the fake runtime's loom). With `unitRefusal` the
 *  library is a residual one (`standard-v4-residual`): every member answers that refusal. */
export function createFixtureUnitsService(modelId: string, tree: () => LoomTreeJSON, unitRefusal: string | null = null): {
  request(method: string, args: readonly unknown[]): Promise<unknown>;
} {
  let service: Promise<BrowserUnitsService> | null = null;
  const open = () => {
    service ??= fixtureUnits().then(({ model, tokenizer, packs }) => {
      const backend = cpuBlockSitesBackend(model, unitRefusal === null
        ? { profile: STANDARD_STRUCTURED_HOOK_PROFILE_V4, contextTokens: 2 * MAX_ROWS + 2 }
        : { profile: STANDARD_STRUCTURED_HOOK_PROFILE_V4_RESIDUAL, contextTokens: 2 * MAX_ROWS + 2, unitRefusal });
      const replay = fixtureReplay(tree, tokenizer);
      return new BrowserUnitsService({
        modelId,
        blockSites: () => backend,
        packs: () => packs,
        tokenTexts: async (ids) => ids.map((id) => tokenizer.decode([id])),
        resolveToken: async (word) => {
          if (word.startsWith("#")) return Number(word.slice(1));
          const ids = tokenizer.encode(word.trim());
          if (ids.length !== 1) {
            throw Object.assign(new Error(`'${word}' is not a single token in the fixture vocabulary`), {
              code: "MultiTokenWordError", status: 400, recoverable: true,
            });
          }
          return ids[0];
        },
        replay: () => replay,
      });
    });
    return service;
  };
  return {
    async request(method, args) {
      const units = await open();
      return units.request(method as Parameters<BrowserUnitsService["request"]>[0], args);
    },
  };
}
