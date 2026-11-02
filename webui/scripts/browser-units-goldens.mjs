#!/usr/bin/env node
// Hook ABI v5 on WebGPU: drowse's units goldens through a fixture library and the WebLLM bridge.
//
//   node scripts/browser-units-goldens.mjs --webllm WEB_LLM/lib/index.js \
//     --fixtures DIR --arch qwen3|gemma3 --quantization q0f32|q4f32_1 [--suite all|units|replay|reads]
//     [--headed]
//
// DIR holds build-units-fixtures.py's model directory `units-<arch>-<quantization>` and its
// library `libs/units-<arch>-<quantization>-v5.wasm`. The bridge's schema-4 descriptor is attested
// with the adapter's own function (C0's expected vector); then every block-sites, unit-ablation and
// unit-probes case runs as a raw greedy completion under a drowse-structured-v4 program with eight
// linear probes per layer (`out[layer, last, 0:8]`): per forward the greedy token, its logprob,
// the probe readings and the live unit taps are checked against the files' library tolerance.
// The replay suite drives `drowse_replay_batch_prefill` through the adapter's own block-sites
// surface (webLlmBlockSites.ts, served to the page by Vite): attention-patterns (every head's row at
// positions 2, 5, 8), token-anatomy (taps, write norms, the chosen heads' rows) and unit-activity
// (every unit's device peak, the host ranking, traces, the float16 population). The reads suite
// runs every static-read member of the adapter (browser-units-reads.mjs).
import { createServer as createHttpServer } from "node:http";
import { readFile } from "node:fs/promises";
import { basename, extname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { createServer as createViteServer } from "vite";
import { verifyReads } from "./browser-units-reads.mjs";
import { releaseToolBrowserOptions, releaseToolGpuArgs } from "./release-tool-storage-quota.mjs";

const options = parseArguments(process.argv.slice(2));
const webui = fileURLToPath(new URL("..", import.meta.url));
const goldens = resolve(webui, "../drowse/tests/fixtures/browser");
const modelDirectory = resolve(options.fixtures, `units-${options.arch}-${options.quantization}`);
const libraryPath = resolve(options.fixtures, "libs", `units-${options.arch}-${options.quantization}-v5.wasm`);
const golden = async (name) => JSON.parse(await readFile(resolve(goldens, `${name}-${options.arch}-v1.json`), "utf8"));
const tiny = JSON.parse(await readFile(resolve(goldens, "units-tiny-models-v1.json"), "utf8")).arches[options.arch];
const chatConfig = JSON.parse(await readFile(resolve(modelDirectory, "mlc-chat-config.json"), "utf8"));
const modelFiles = ["mlc-chat-config.json", "tensor-cache.json", "tokenizer.json",
  ...JSON.parse(await readFile(resolve(modelDirectory, "tensor-cache.json"), "utf8")).records.map((shard) => shard.dataPath)];

const vite = await createViteServer({
  root: webui, configFile: false, appType: "custom", logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
const [profiles, programs, engineModule] = await Promise.all([
  vite.ssrLoadModule("/src/hosted/runtime/structuredHookProfile.ts"),
  vite.ssrLoadModule("/src/hosted/runtime/structuredHookProgram.ts"),
  vite.ssrLoadModule("/src/hosted/runtime/webLlmEngine.ts"),
]);

// ------------------------------------------------------------------- fixture facts ---
const config = tiny.config;
const gemma = options.arch === "gemma3";
const blockSpec = profiles.validateBlockSpec({
  dFF: config.intermediate_size, nHeads: config.num_attention_heads,
  nKvHeads: config.num_key_value_heads, headDim: config.head_dim, sandwich: gemma,
  normOffset: gemma ? 1 : 0, activation: gemma ? "gelu_tanh" : "silu",
  embedScale: gemma ? Math.sqrt(config.hidden_size) : 1, rmsEps: config.rms_norm_eps, qkNorm: true,
  attnScaling: (gemma ? config.query_pre_attn_scalar : config.head_dim) ** -0.5,
  attnLogitSoftcap: null, finalLogitSoftcap: null,
  ropeTheta: gemma ? config.rope_parameters.full_attention.rope_theta : config.rope_parameters.rope_theta,
  ropeLocalTheta: gemma ? config.rope_parameters.sliding_attention.rope_theta : null,
  ropeGlobalScale: 1, ropeInGraph: gemma,
  slidingWindow: gemma ? config.sliding_window : null, slidingPattern: gemma ? 2 : null,
});
const L = config.num_hidden_layers;
const d = config.hidden_size;
const hd = config.head_dim;
const profile = profiles.STANDARD_STRUCTURED_HOOK_PROFILE_V4;
const layout = programs.structuredUnitSectionLayout(profile, L, hd);
const context = chatConfig.context_window_size;
const variant = {
  structuredHookProfile: "standard-v4",
  contextProfiles: [{ contextTokens: context }],
  runtimeIdentity: { layerMap: Array.from({ length: L }, (_, i) => i), hiddenSize: d },
};

const tensors = await loadTensorCache(modelDirectory);
const stats = await golden("unit-stats");
const decode = (value) => {
  if (Array.isArray(value)) return Float32Array.from(value);
  if (value.encoding !== "float32-le-base64") return Float32Array.from(value.values);
  const bytes = Buffer.from(value.data, "base64");
  return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
};
const neuronMean = decode(stats.neuron_mean);
const headMean = decode(stats.head_mean);
const neuronScale = decode(stats.scales.neuron);
const headScale = decode(stats.scales.head);

// A zero v4 program (no affine, no curves) with linear probes e_0…e_7 at every layer.
function baseProgram(affine = null) {
  const layers = Array.from({ length: L }, (_, layer) => ({
    probes: Array.from({ length: 8 }, (_, k) => {
      const direction = new Array(d).fill(0);
      direction[k] = 1;
      return { kind: "linear", direction, bias: 0 };
    }),
    ...(affine?.layer === layer ? { affineGroups: [affine.group] } : {}),
  }));
  return programs.compileStructuredHookProgram(d, layers, profile);
}

class UnitTables {
  constructor() {
    this.neurons = []; this.heads = []; this.neuronTaps = []; this.headTaps = []; this.triggers = [];
  }
  group(trigger) {
    const key = JSON.stringify(trigger);
    let index = this.triggers.findIndex((item) => JSON.stringify(item) === key);
    if (index < 0) { index = this.triggers.length; this.triggers.push(trigger); }
    return index;
  }
  ablate(kind, layer, index, keep, bias, trigger) {
    (kind === "neuron" ? this.neurons : this.heads).push({ layer, index, group: this.group(trigger), keep, bias });
  }
  tap(kind, layer, index) {
    const slot = this.neuronTaps.length + this.headTaps.length;
    (kind === "neuron" ? this.neuronTaps : this.headTaps).push({ layer, index, slot, gram: kind === "head" ? headGram(layer, index) : null });
    return slot;
  }
  section() {
    const meta = new Uint32Array(layout.metaLength);
    const coeff = new Float32Array(layout.coeffLength);
    const grams = new Float32Array(layout.tapGramsLength);
    const m = layout.meta;
    const c = layout.coeff;
    const csr = (entries, name) => {
      const ordered = entries.map((entry, at) => ({ entry, at })).sort((a, b) => a.entry.layer - b.entry.layer || a.at - b.at).map((x) => x.entry);
      for (let layer = 0; layer <= L; layer += 1) meta[m[name].offset + layer] = ordered.filter((e) => e.layer < layer).length;
      return ordered;
    };
    csr(this.neurons, "neuronOffsets").forEach((e, i) => {
      meta[m.neuronIndex.offset + i] = e.index; meta[m.neuronGroup.offset + i] = e.group;
      coeff[c.neuronKeep.offset + i] = e.keep; coeff[c.neuronBias.offset + i] = e.bias;
    });
    csr(this.heads, "headOffsets").forEach((e, i) => {
      meta[m.headIndex.offset + i] = e.index; meta[m.headGroup.offset + i] = e.group;
      coeff[c.headKeep.offset + i] = e.keep; coeff.set(e.bias, c.headBias.offset + i * hd);
    });
    csr(this.neuronTaps, "tapNeuronOffsets").forEach((e, i) => {
      meta[m.tapNeuronIndex.offset + i] = e.index; meta[m.tapNeuronSlot.offset + i] = e.slot;
    });
    csr(this.headTaps, "tapHeadOffsets").forEach((e, i) => {
      meta[m.tapHeadIndex.offset + i] = e.index; meta[m.tapHeadSlot.offset + i] = e.slot;
      grams.set(e.gram, i * hd * hd);
    });
    return { unitMeta: meta, unitCoeff: coeff, unitActive: new Float32Array(layout.activeLength), tapGrams: grams };
  }
}

function headGram(layer, head) {
  const suffix = `layers.${layer}.self_attn.o_proj`;
  const name = [...tensors.keys()].find((key) => key.includes(suffix));
  const prefix = name.slice(0, name.lastIndexOf("."));
  const weight = dequantized(prefix);
  const rows = d;
  const gamma = gemma ? tensors.get(prefix.replace("self_attn.o_proj", "post_attention_layernorm") + ".weight").values : null;
  const gram = new Float32Array(hd * hd);
  for (let a = 0; a < hd; a += 1) {
    for (let b = 0; b < hd; b += 1) {
      let sum = 0;
      for (let r = 0; r < rows; r += 1) {
        const g = gamma === null ? 1 : gamma[r];
        sum += g * weight[r * weight.columns + head * hd + a] * g * weight[r * weight.columns + head * hd + b];
      }
      gram[a * hd + b] = sum;
    }
  }
  return gram;
}

function dequantized(prefix) {
  const plain = tensors.get(prefix + ".weight");
  if (plain !== undefined) return Object.assign(Float64Array.from(plain.values), { columns: plain.shape[1] });
  const weight = tensors.get(prefix + ".q_weight");
  const scale = tensors.get(prefix + ".q_scale");
  const [rows, packed] = weight.shape;
  const columns = packed * 8;
  const out = new Float64Array(rows * columns);
  for (let r = 0; r < rows; r += 1) {
    for (let col = 0; col < columns; col += 1) {
      const word = weight.values[r * packed + (col >> 3)];
      const code = (word >>> ((col & 7) * 4)) & 15;
      out[r * columns + col] = (code - 7) * scale.values[r * scale.shape[1] + (col >> 5)];
    }
  }
  return Object.assign(out, { columns });
}

function parseTrigger(text) {
  if (text === undefined || text === null) return null;
  if (text === "prompt" || text === "before") return { prompt: true, generated: false };
  if (text === "response") return { prompt: false, generated: true, response: true };
  if (text.startsWith("first:")) return { prompt: false, first_n: Number(text.slice(6)) };
  const gate = /^when:(\S+)\s*(>=|<=|>|<)\s*(\S+)$/.exec(text);
  if (gate) return { gate: { probe: gate[1], op: gate[2], threshold: Number(gate[3]) } };
  throw new Error(`unsupported trigger ${text}`);
}

function parseUnit(name) {
  const match = /^(neuron|head)\/L(\d+)[nh](\d+)$/.exec(name);
  return { kind: match[1], layer: Number(match[2]), index: Number(match[3]) };
}

function terms(steering) {
  if (steering === null) return [];
  return steering.split(" + ").map((text) => {
    const match = /^(?:([-+]?[0-9.]+)\s+)?!(neuron|head)\/L(\d+)[nh](\d+)(?:@(.+))?$/.exec(text.trim());
    if (!match) throw new Error(`unsupported term ${text}`);
    return { alpha: match[1] ? Number(match[1]) : 1, kind: match[2], layer: Number(match[3]), index: Number(match[4]), trigger: parseTrigger(match[5]) };
  });
}

const scaleOf = (kind, layer, index) => kind === "neuron"
  ? [neuronScale[(layer * 2) * config.intermediate_size + index], neuronScale[(layer * 2 + 1) * config.intermediate_size + index]]
  : headScale[layer * config.num_attention_heads + index];

// A case for the page: the program, the triggers (gates resolved to tap slots and scales), and
// what each forward must read.
function pageCase({ name, tables, steering, gates, taps, affine, forwards, tolerance }) {
  const unit = new UnitTables();
  if (tables !== undefined) {
    for (const table of tables) {
      const trigger = table.trigger;
      if (table.site === "act") {
        table.entries.forEach((entry, at) => unit.ablate("neuron", table.layer, entry, table.keep[at], table.bias[at], trigger));
      } else {
        for (let start = 0; start < table.entries.length; start += hd) {
          unit.ablate("head", table.layer, table.entries[start] / hd, table.keep[start], Float32Array.from(table.bias.slice(start, start + hd)), trigger);
        }
      }
    }
  }
  for (const term of terms(steering ?? null)) {
    const mean = term.kind === "neuron"
      ? neuronMean[term.layer * config.intermediate_size + term.index]
      : headMean.subarray((term.layer * config.num_attention_heads + term.index) * hd, (term.layer * config.num_attention_heads + term.index + 1) * hd);
    const c = Math.fround(term.alpha);
    unit.ablate(term.kind, term.layer, term.index, Math.fround(1 - term.alpha),
      typeof mean === "number" ? Math.fround(c * mean) : Float32Array.from(mean, (v) => Math.fround(c * v)), term.trigger);
  }
  const slots = {};
  for (const unitName of new Set([...(taps ?? []), ...(gates ?? [])])) {
    const parsed = parseUnit(unitName);
    slots[unitName] = { ...parsed, slot: unit.tap(parsed.kind, parsed.layer, parsed.index), scale: scaleOf(parsed.kind, parsed.layer, parsed.index) };
  }
  const program = { ...baseProgram(affine), ...unit.section() };
  return {
    name,
    program: serialize(program),
    triggers: unit.triggers.map((trigger) => trigger === null ? null : {
      ...trigger, ...(trigger.gate ? { gate: { ...trigger.gate, ...slots[trigger.gate.probe] } } : {}),
    }),
    slots,
    forwards,
    tolerance,
  };
}

function serialize(program) {
  const out = {};
  for (const [key, value] of Object.entries(program)) {
    if (ArrayBuffer.isView(value)) out[key] = { type: value.constructor.name, values: Array.from(value) };
    else if (["hookAbi", "format", "hiddenSize", "layerCount"].includes(key)) out[key] = value;
  }
  return out;
}

const logSoftmax = (logits) => {
  let max = -Infinity;
  for (const value of logits) max = Math.max(max, value);
  let sum = 0;
  for (const value of logits) sum += Math.exp(value - max);
  const lse = max + Math.log(sum);
  return Array.from(logits, (value) => value - lse);
};

// ------------------------------------------------------------------------ the cases ---
const sites = await golden("block-sites");
const ablation = await golden("unit-ablation");
const probesGolden = await golden("unit-probes");
const push = await golden("neuron-push");
const library = sites.tolerance.library;
const prompt = sites.ids;
const cases = [];
const outRows = (value, row) => {
  const out = decode(value);
  const [layers, rows, width] = value.shape;
  return Array.from({ length: layers }, (_, layer) => Array.from(out.subarray((layer * rows + row) * width, (layer * rows + row) * width + 8)));
};
cases.push(pageCase({
  name: "block-sites", tables: [], forwards: [{
    token: null, logprobs: logSoftmax(decode(sites.logits_last)),
    out: outRows(sites.sites.out, prompt.length - 1),
  }], tolerance: library,
}));
for (const item of ablation.cases) {
  if (item.name === "mixed") continue;
  const affineTerm = item.name === "push-and-ablate"
    ? push.cases.find((c) => c.expression === "0.3 neuron/L2n7").layers[0]
    : null;
  cases.push(pageCase({
    name: `unit-ablation:${item.name}`,
    tables: item.tables,
    gates: item.tables.flatMap((t) => t.trigger.gate ? [t.trigger.gate.probe] : []),
    affine: affineTerm === null ? null : {
      layer: affineTerm.layer,
      group: { active: true, basis: [affineTerm.basis], neutral: affineTerm.neutral, target: [affineTerm.target], along: affineTerm.along, kappa: [0] },
    },
    forwards: item.forwards.map((forward, number) => {
      const lastRow = forward.out_rows.indexOf(forward.ids.length - 1);
      const out = outRows(forward.out, lastRow);
      if (affineTerm !== null) {
        // Probes read after the post-block hook, which adds the push at its layer.
        out[affineTerm.layer] = out[affineTerm.layer].map((value, k) => Math.fround(value + affineTerm.along * affineTerm.target * affineTerm.basis[k]));
      }
      return { token: item.forwards[number + 1]?.ids[0] ?? null, logprobs: logSoftmax(decode(forward.logits)), out };
    }),
    tolerance: library,
  }));
}
for (const item of probesGolden.cases) {
  const gateTerms = terms(item.steering).filter((t) => t.trigger?.gate);
  cases.push(pageCase({
    name: `unit-probes:${item.name}`,
    steering: item.steering,
    taps: [...probesGolden.probes, ...(item.gate_scalars.length ? [probesGolden.gate_unit] : [])],
    gates: gateTerms.map((t) => t.trigger.gate.probe),
    forwards: item.forwards.map((forward, number) => ({
      token: item.tokens[number], units: forward.units,
      gate: item.gate_scalars[number]?.[probesGolden.gate_unit] ?? null,
    })),
    tolerance: library,
  }));
}

// ---------------------------------------------------------------- replay cases (B4b) ---
// Each replay runs in the page through the adapter's own `webLlmBlockSitesBackend` over the
// bridge: `openReplay` (C0's `encodeReplayMeta`), segments of at most 16 positions (so the
// KV-continuing path is exercised), `finish`, and `attentionRows` on the pattern capture.
const REPLAY_PIECE = 16;
const RAW_FIRST_IDS = { qwen3: [45], gemma3: [32] };
const H = config.num_attention_heads;
const F = config.intermediate_size;
const unitScalesData = (() => {
  const width = 2 * F + H;
  const out = new Float32Array(L * width);
  for (let layer = 0; layer < L; layer += 1) {
    for (let n = 0; n < F; n += 1) {
      out[layer * width + n] = Math.max(neuronScale[(layer * 2) * F + n], 1e-6);
      out[layer * width + F + n] = Math.max(neuronScale[(layer * 2 + 1) * F + n], 1e-6);
    }
    for (let h = 0; h < H; h += 1) out[layer * width + 2 * F + h] = Math.max(headScale[layer * H + h], 1e-6);
  }
  return out;
})();
const emptyControls = (() => {
  const program = baseProgram();
  return { affine: program.affineActive.length, curve: program.curveActive.length };
})();

/** A replay case for the page: the program (null: unsteered), the segments as fed (each piece
 *  ≤ 16 positions, controls held), the request, and the heads whose rows to read. */
function replayCase({ name, steering = null, affine = null, unitTerms = [], segments, request, heads = [] }) {
  let program = null;
  let triggers = [];
  if (steering !== null || affine !== null || unitTerms.length > 0) {
    const unit = new UnitTables();
    for (const term of [...terms(steering), ...unitTerms]) {
      const mean = term.kind === "neuron"
        ? neuronMean[term.layer * F + term.index]
        : headMean.subarray((term.layer * H + term.index) * hd, (term.layer * H + term.index + 1) * hd);
      const c = Math.fround(term.alpha);
      const bias = term.bias !== undefined ? term.bias
        : typeof mean === "number" ? Math.fround(c * mean) : Float32Array.from(mean, (v) => Math.fround(c * v));
      unit.ablate(term.kind, term.layer, term.index, term.keep ?? Math.fround(1 - term.alpha), bias, term.trigger);
    }
    program = serialize({ ...baseProgram(affine), ...unit.section() });
    triggers = unit.triggers;
  }
  const pieces = [];
  for (const segment of segments) {
    for (let start = 0; start < segment.ids.length; start += REPLAY_PIECE) {
      pieces.push({ tokenIds: segment.ids.slice(start, start + REPLAY_PIECE), unitActive: [...segment.active, 0, 0, 0, 0].slice(0, 4) });
    }
  }
  const tokenCount = segments.reduce((sum, segment) => sum + segment.ids.length, 0);
  return { name, program, triggers, pieces, request: { ...request, tokenCount }, heads, controls: emptyControls };
}

const anatomyGolden = await golden("token-anatomy");
const patternsGolden = await golden("attention-patterns");
const activityGolden = await golden("unit-activity");
const pushAffine = (expression) => {
  const term = push.cases.find((c) => c.expression === expression).layers[0];
  return { layer: term.layer, group: { active: true, basis: [term.basis], neutral: term.neutral, target: [term.target], along: term.along, kappa: [0] } };
};
const allLayers = Array.from({ length: L }, (_, layer) => layer);
const everyHead = allLayers.flatMap((layer) => Array.from({ length: H }, (_, head) => ({ layer, head })));
const replayCases = [];
for (const item of patternsGolden.cases) {
  for (const row of item.rows) {
    replayCases.push({ ...replayCase({
      name: `attention-patterns:${item.name}@${row.position}`, steering: item.steering,
      segments: [{ ids: patternsGolden.ids, active: [true] }],
      request: { anatomyPosition: row.position, activity: null, patternLayers: allLayers },
      heads: everyHead,
    }), check: { kind: "patterns", row } });
  }
}
for (const item of anatomyGolden.cases) {
  const ids = item.intermediates.ids;
  const pushTerm = item.steering?.split(" + ").find((t) => t.startsWith("0.3 neuron/")) ?? null;
  const ablation = item.steering?.split(" + ").filter((t) => !t.startsWith("0.3 neuron/")).join(" + ") || null;
  const record = item.envelope.measurements.anatomy;
  replayCases.push({ ...replayCase({
    name: `token-anatomy:${item.name}`, steering: ablation, affine: pushTerm === null ? null : pushAffine(pushTerm),
    segments: [{ ids, active: [true] }],
    request: { anatomyPosition: ids.length - 1, activity: null, patternLayers: allLayers },
    heads: (record.patterns ?? []).map((p) => ({ layer: p.layer, head: p.head })),
  }), check: { kind: "anatomy", item } });
}
{
  const first = anatomyGolden.cases[0];
  const promptLength = first.envelope.measurements.anatomy.token.prompt_length;
  const chat = first.intermediates.ids.slice(0, promptLength);
  for (const item of activityGolden.cases) {
    const args = item.arguments;
    const tokens = item.result.tokens.map((t) => t.token_id);
    let ids; let spans; let segments; const unitTerms = [];
    if (item.name === "generated" || item.name === "segmented") {
      ids = [...chat, ...tokens.slice(0, -1)];
      spans = [{ firstRow: 0, rows: tokens.length, firstPosition: promptLength - 1 }];
      segments = item.result.steering === null
        ? [{ ids, active: [] }]
        : [{ ids: ids.slice(0, promptLength), active: [true] }, { ids: ids.slice(promptLength), active: [false] }];
    } else if (item.name === "sink" || item.name === "sink-population") {
      ids = [...RAW_FIRST_IDS[options.arch], ...tokens.slice(0, -1)];
      spans = [{ firstRow: 0, rows: tokens.length, firstPosition: 0 }];
      segments = [{ ids, active: [] }];
    } else if (item.name === "authored") {
      const start = chat.findIndex((_, i) => tokens.every((t, k) => chat[i + k] === t));
      ids = chat.slice(0, start + tokens.length - 1);
      spans = [{ firstRow: 0, rows: tokens.length, firstPosition: start - 1 }];
      segments = [{ ids, active: [] }];
    } else {
      ids = args.ids;
      const runs = args.runs;
      spans = runs.map(([first, at], i) => ({
        firstRow: first, rows: (runs[i + 1]?.[0] ?? args.raw_ids.length) - first, firstPosition: at - 1,
      }));
      if (item.name === "excluded-and-zero") {
        unitTerms.push({ kind: "neuron", layer: 1, index: 5, alpha: 1, keep: 0, bias: 0, trigger: null });
        segments = [{ ids, active: [true] }];
      } else {
        segments = [{ ids, active: [] }];
      }
    }
    replayCases.push({ ...replayCase({
      name: `unit-activity:${item.name}`, steering: item.result.steering, unitTerms, segments,
      request: {
        anatomyPosition: null, patternLayers: null,
        activity: {
          spans, rowCount: tokens.length,
          traces: (args.units ?? []).map((u) => parseUnit(u)),
          populationRow: args.raw_index ?? null,
        },
      },
    }), check: { kind: "activity", item, index: activityGolden.cases.indexOf(item) } });
  }
}

// ---------------------------------------------------------------------- the browser ---
const libraryBytes = await readFile(libraryPath);
const webllm = await readFile(resolve(options.webllm));
const http = createHttpServer(async (request, response) => {
  const url = new URL(request.url ?? "/", "http://127.0.0.1");
  const send = (body, type, status = 200) => {
    response.writeHead(status, { "content-type": type, "cross-origin-opener-policy": "same-origin", "cross-origin-embedder-policy": "require-corp" });
    response.end(body);
  };
  try {
    if (url.pathname === "/") return send("<!doctype html><title>units</title>", "text/html");
    if (url.pathname === "/webllm.js") return send(webllm, "text/javascript");
    if (url.pathname === "/model-lib.wasm") return send(libraryBytes, "application/wasm");
    if (url.pathname.startsWith("/model/")) {
      const name = basename(url.pathname);
      if (!modelFiles.includes(name)) return send("not found", "text/plain", 404);
      return send(await readFile(resolve(modelDirectory, name)), extname(name) === ".json" ? "application/json" : "application/octet-stream");
    }
    // The adapter's own modules (webLlmBlockSites.ts and what it imports), served by Vite.
    if (url.pathname.startsWith("/src/") || url.pathname.startsWith("/@") || url.pathname.startsWith("/node_modules/")) {
      return vite.middlewares(request, response, () => send("not found", "text/plain", 404));
    }
    return send("not found", "text/plain", 404);
  } catch (error) {
    return send(String(error), "text/plain", 500);
  }
});
await new Promise((ok) => http.listen(options.port, "127.0.0.1", ok));
const origin = `http://127.0.0.1:${http.address().port}`;
const browser = await chromium.launch(releaseToolBrowserOptions({
  headless: !options.headed, args: releaseToolGpuArgs({ headless: !options.headed }),
}));
let failures = 0;
try {
  const page = await browser.newPage();
  page.setDefaultTimeout(300_000);
  page.on("console", (message) => {
    if (message.type() === "error") process.stderr.write(`browser console: ${message.text()}\n`);
  });
  await page.goto(origin);
  const profileValue = await page.evaluate(async () => {
    const module = await import("/webllm.js");
    const engine = new module.MLCEngine({
      logLevel: "WARN",
      appConfig: { cacheBackend: "indexeddb", model_list: [{ model: `${location.origin}/model/`, model_id: "units", model_lib: `${location.origin}/model-lib.wasm` }] },
    });
    await engine.reload("units");
    window.engine = engine;
    window.webllmModule = module;
    return { profile: await engine.getDrowseStructuredHookProfile("units"), units: await engine.supportsDrowseUnitSites("units") };
  });
  // The adapter's own attestation: the library's descriptor against the lock-shaped model.
  const attested = engineModule.attestCompiledStructuredHookProfile(profileValue.profile, { variant, blockSpec }, "block-sites-v5");
  console.log(`descriptor attested: ${attested.descriptor.length} fields, unit sites ${attested.unitSites}, units pair ${profileValue.units}`);
  if (options.suite === "units" || options.suite === "all") {
    for (const item of cases) {
      const result = await page.evaluate(runCase, { item, textIds: gemma ? prompt.slice(1) : prompt });
      const worst = checkCase(item, result);
      console.log(`${item.name}: ${result.forwards.length} forwards, worst |Δ| ${worst.toExponential(2)}`);
    }
  }
  if (options.suite === "reads" || options.suite === "all") {
    const capabilities = engineModule.blockSitesCapabilitiesFor(attested, { variant, blockSpec, unitRefusal: null }, chatConfig.vocab_size);
    await page.evaluate(installReads, { capabilities });
    const call = async (member, ...args) => {
      const out = await page.evaluate(callRead, { member, args: args.map(toPage) });
      if ("error" in out) throw new Error(`${member}: ${out.error}`);
      return fromPage(out.ok);
    };
    const f32 = (data, shape) => ({ dtype: "f32", shape, data: Float32Array.from(data) });
    const i32 = (data, shape) => ({ dtype: "i32", shape, data: Int32Array.from(data) });
    const u32 = (data, shape) => ({ dtype: "u32", shape, data: Uint32Array.from(data) });
    const worst = await verifyReads({
      call, f32, i32, u32, weights: readWeights(), golden, decode,
      L, d, dFF: blockSpec.dFF, H: blockSpec.nHeads, KV: blockSpec.nKvHeads, hd, V: chatConfig.vocab_size,
      sandwich: gemma, eps: config.rms_norm_eps, embedScale: blockSpec.embedScale,
    });
    console.log(`reads ${options.arch} ${options.quantization}: ${Object.entries(worst).map(([k, v]) => `${k} ${v.toExponential(2)}`).join(", ")}`);
  }
  if (options.suite === "replay" || options.suite === "all") {
    const capabilities = engineModule.blockSitesCapabilitiesFor(attested, { variant, blockSpec, unitRefusal: null }, chatConfig.vocab_size);
    await page.evaluate((scales) => { window.unitScales = scales; }, Array.from(unitScalesData));
    const worstByGolden = {};
    for (const item of replayCases) {
      const result = await page.evaluate(runReplay, { item, capabilities });
      const worst = checkReplay(item, result);
      const key = item.name.split(":")[0];
      worstByGolden[key] = Math.max(worstByGolden[key] ?? 0, worst);
      console.log(`${item.name}: ${item.pieces.length} forwards, worst |Δ| ${worst.toExponential(2)}`);
    }
    console.log(`replay ${options.arch} ${options.quantization}: ${Object.entries(worstByGolden).map(([k, v]) => `${k} ${v.toExponential(2)}`).join(", ")}`);
  }
} catch (error) {
  failures += 1;
  console.error(error?.stack ?? String(error));
} finally {
  await browser.close();
  http.close();
  await vite.close();
}
process.exit(failures === 0 ? 0 : 1);

// In the page: one case as a raw greedy completion, the controls updated before each decode
// forward from the previous forward's taps (native's gates), the bundle read after each token.
async function runCase({ item, textIds }) {
  const engine = window.engine;
  const program = {};
  for (const [key, value] of Object.entries(item.program)) {
    program[key] = value && typeof value === "object" && "type" in value ? new globalThis[value.type](value.values) : value;
  }
  const active = (trigger, forward, scores) => {
    if (trigger === null) return true;
    if (forward === 0) return !trigger.gate && trigger.prompt !== false;
    if (trigger.generated === false || trigger.response === false) return false;
    if (trigger.first_n != null && forward >= trigger.first_n) return false;
    if (trigger.after_n != null && forward < trigger.after_n) return false;
    if (trigger.gate) {
      const score = scores[trigger.gate.probe];
      if (score === undefined) return false;
      const t = trigger.gate.threshold;
      return { ">": score > t, ">=": score >= t, "<": score < t, "<=": score <= t }[trigger.gate.op];
    }
    return true;
  };
  const unitActive = (forward, scores) => {
    const values = new Float32Array(4);
    item.triggers.forEach((trigger, group) => { values[group] = active(trigger, forward, scores) ? 1 : 0; });
    return values;
  };
  program.unitActive = unitActive(0, {});
  await engine.resetChat(false, "units");
  await engine.setDrowseStructuredProgram(program, "units");
  const promptText = await engine.decodeDrowseTokens(textIds, "units");
  const forwards = [];
  const stream = await engine.completions.create({
    prompt: promptText, stream: true, max_tokens: item.forwards.length, temperature: 0,
    logprobs: true, top_logprobs: 5, model: "units", ignore_eos: true,
  });
  for await (const chunk of stream) {
    const choice = chunk.choices?.[0];
    const rows = choice?.logprobs?.content ?? [];
    for (const row of rows) {
      const bundle = await engine.readDrowseMeasurementBundle("units");
      const taps = Array.from(bundle.unitTaps ?? []);
      const scores = {};
      for (const [unit, slot] of Object.entries(item.slots)) {
        const raw = taps[slot.slot];
        const scale = typeof slot.scale === "number" ? slot.scale : slot.scale[raw >= 0 ? 0 : 1];
        scores[unit] = Math.fround(Math.fround(raw) / Math.fround(scale));
      }
      forwards.push({
        token: row.token_id ?? null, tokenText: row.token, logprob: row.logprob,
        scalar: Array.from(bundle.scalar ?? []), taps, scores,
      });
      const next = forwards.length;
      if (next < item.forwards.length) {
        await engine.updateDrowseStructuredControls(program.affineActive, program.curveActive, "units", unitActive(next, scores));
      }
    }
  }
  return { forwards, promptText };
}

function checkCase(item, result) {
  let worst = 0;
  const close = (label, expected, actual, atol = item.tolerance.atol, rtol = item.tolerance.rtol) => {
    const error = Math.abs(actual - expected);
    if (!(error <= atol + rtol * Math.abs(expected))) {
      throw new Error(`${item.name} ${label}: library ${actual} vs golden ${expected} (|Δ| ${error})`);
    }
    worst = Math.max(worst, error);
  };
  if (result.forwards.length !== item.forwards.length) {
    throw new Error(`${item.name}: ${result.forwards.length} forwards, the golden has ${item.forwards.length}`);
  }
  item.forwards.forEach((expected, number) => {
    const actual = result.forwards[number];
    const label = `forward ${number}`;
    if (expected.token !== null && expected.token !== undefined && actual.token !== expected.token) {
      throw new Error(`${item.name} ${label}: greedy token ${actual.token}, the golden's next input ${expected.token}`);
    }
    if (expected.logprobs !== undefined) {
      // Greedy decoding reports no sampler logprob; the golden's own argmax is the token.
      const best = expected.logprobs.indexOf(Math.max(...expected.logprobs));
      if (actual.token !== best) {
        throw new Error(`${item.name} ${label}: greedy token ${actual.token}, the golden logits' argmax ${best}`);
      }
    }
    if (expected.out !== undefined) {
      expected.out.forEach((row, layer) => row.forEach((value, k) => close(`${label} out[L${layer}][${k}]`, value, actual.scalar[layer * 8 + k])));
    }
    if (expected.units !== undefined) {
      for (const [unit, golden] of Object.entries(expected.units)) {
        close(`${label} ${unit} raw`, golden.raw, actual.taps[item.slots[unit].slot]);
        close(`${label} ${unit} reading`, golden.value, actual.scores[unit], item.tolerance.z_atol, 0);
      }
    }
    if (expected.gate !== null && expected.gate !== undefined) {
      const unit = Object.keys(item.slots).find((name) => item.slots[name].layer === 3);
      close(`${label} gate`, expected.gate, actual.scores[unit], item.tolerance.z_atol, 0);
    }
  });
  return worst;
}

// In the page: one replay through the adapter's block-sites surface over the bridge.
async function runReplay({ item, capabilities }) {
  const engine = window.engine;
  const { webLlmBlockSitesBackend } = await import("/src/hosted/runtime/webLlmBlockSites.ts");
  const { f32Tensor } = await import("/src/hosted/runtime/modelBackend.ts");
  const backend = webLlmBlockSitesBackend(capabilities, {
    call: (request) => engine.drowseBlockSites(request, "units"),
    installProgram: (program) => engine.setDrowseStructuredProgram(program, "units"),
  });
  let program = null;
  if (item.program !== null) {
    program = {};
    for (const [key, value] of Object.entries(item.program)) {
      program[key] = value && typeof value === "object" && "type" in value ? new globalThis[value.type](value.values) : value;
    }
  }
  const L = capabilities.layerCount;
  const scales = item.request.activity === null ? null : await backend.uploadUnitScales(
    f32Tensor(Float32Array.from(window.unitScales), [L, window.unitScales.length / L]),
  );
  const replay = await backend.openReplay({
    ...item.request,
    program,
    activity: item.request.activity === null ? null : { ...item.request.activity, scales },
  });
  for (const piece of item.pieces) {
    await replay.forward({
      tokenIds: Int32Array.from(piece.tokenIds),
      controls: {
        affineActive: program?.affineActive ?? new Uint32Array(item.controls.affine),
        curveActive: program?.curveActive ?? new Uint32Array(item.controls.curve),
        unitActive: Float32Array.from(piece.unitActive),
      },
    });
  }
  const result = await replay.finish();
  const rows = [];
  if (result.patterns !== null && item.heads.length > 0) {
    for (let start = 0; start < item.heads.length; start += 16) {
      const out = await backend.attentionRows(result.patterns, item.heads.slice(start, start + 16));
      rows.push(...Array.from(out.data));
    }
  }
  if (result.patterns !== null) await backend.release(result.patterns);
  if (scales !== null) await backend.release(scales);
  return {
    anatomy: result.anatomyTaps === null ? null : Array.from(result.anatomyTaps.data),
    state: result.activity === null ? null : Array.from(result.activity.state.data),
    traces: result.activity === null ? null : Array.from(result.activity.traces.data),
    population: result.activity?.population ? Array.from(result.activity.population.data) : null,
    rows,
  };
}

/** The replay's results against the goldens (the C VM harness's checks, units_goldens.py). */
function checkReplay(item, result) {
  let worst = 0;
  const library = item.check.kind === "patterns" ? patternsGolden.tolerance.library
    : item.check.kind === "anatomy" ? anatomyGolden.tolerance.library : activityGolden.tolerance.library;
  const close = (label, expected, actual, atol = library.atol, rtol = library.rtol) => {
    const error = Math.abs(actual - expected);
    if (!(error <= atol + rtol * Math.abs(expected))) {
      throw new Error(`${item.name} ${label}: library ${actual} vs golden ${expected} (|Δ| ${error})`);
    }
    worst = Math.max(worst, error);
  };
  const d = config.hidden_size;
  const stride = 6 * d + F + H * hd;
  const offsets = { block_in: 0, mid: d, out_pre: 2 * d, out: 3 * d, attn_o: 4 * d, mlp_o: 5 * d, act: 6 * d, z: 6 * d + F };
  if (item.check.kind === "patterns") {
    const p = item.check.row.position;
    const width = result.rows.length / item.heads.length;
    item.heads.forEach(({ layer, head }, at) => {
      const expected = decode(item.check.row.layers[layer]);
      for (let j = 0; j < width; j += 1) {
        const actual = result.rows[at * width + j];
        if (j <= p) close(`L${layer}h${head}[${j}]`, expected[head * (p + 1) + j], actual, library.probability_atol, 0);
      }
    });
  } else if (item.check.kind === "anatomy") {
    const inter = item.check.item.intermediates;
    for (const name of ["block_in", "mid", "out", "act", "z"]) {
      const value = inter.taps[name];
      const expected = decode(value);
      const width = value.shape[1];
      for (let layer = 0; layer < L; layer += 1) {
        for (let k = 0; k < width; k += 1) {
          close(`${name}[L${layer}][${k}]`, expected[layer * width + k], result.anatomy[layer * stride + offsets[name] + k]);
        }
      }
    }
    const norms = decode(inter.head_write_norms);
    for (let layer = 0; layer < L; layer += 1) {
      const o = result.anatomy.slice(layer * stride + offsets.attn_o, layer * stride + offsets.attn_o + d);
      const r = gemma ? 1 / Math.sqrt(o.reduce((s, v) => s + v * v, 0) / d + config.rms_norm_eps) : 1;
      for (let head = 0; head < H; head += 1) {
        const z = result.anatomy.slice(layer * stride + offsets.z + head * hd, layer * stride + offsets.z + (head + 1) * hd);
        const gram = headGram(layer, head);
        let q = 0;
        for (let a = 0; a < hd; a += 1) for (let b = 0; b < hd; b += 1) q += z[a] * gram[a * hd + b] * z[b];
        close(`write norm L${layer}h${head}`, norms[layer * H + head], Math.sqrt(Math.max(q, 0)) * r);
      }
    }
    const record = item.check.item.envelope.measurements.anatomy;
    const p = inter.ids.length - 1;
    const width = result.rows.length / Math.max(1, item.heads.length);
    (record.patterns ?? []).forEach((pattern, at) => {
      pattern.weights.forEach((weight, j) => close(`pattern ${pattern.unit}[${j}]`, weight, result.rows[at * width + j],
        library.probability_atol + anatomyGolden.tolerance.rounded.atol, 0));
    });
  } else {
    const { item: golden, index } = item.check;
    const z = library.z_atol + activityGolden.tolerance.rounded.atol;
    const peak = decode(golden.diagnostics.peak);
    const row = int32s(golden.diagnostics.row);
    const margin = decode(golden.diagnostics.row_margin);
    const stateStride = 3 * F + 3 * H;
    const every = [...allLayers.flatMap((layer) => Array.from({ length: F }, (_, n) => ["neuron", layer, n])),
      ...allLayers.flatMap((layer) => Array.from({ length: H }, (_, h) => ["head", layer, h]))];
    const at = (kind, layer, i) => {
      const base = layer * stateStride + (kind === "neuron" ? 0 : 3 * F);
      const width = kind === "neuron" ? F : H;
      return { value: result.state[base + i], row: result.state[base + width + i], rank: result.state[base + 2 * width + i] };
    };
    every.forEach(([kind, layer, i], u) => {
      const device = at(kind, layer, i);
      const name = `${kind}/L${layer}${kind === "neuron" ? "n" : "h"}${i}`;
      if (row[u] < 0) {
        if (device.row !== -1 || device.rank !== -1) throw new Error(`${item.name} ${name}: excluded everywhere, peaked at ${device.row}`);
        return;
      }
      close(`${name} peak`, peak[u], device.value, z, 0);
      if (device.rank !== Math.abs(device.value)) throw new Error(`${item.name} ${name}: rank ${device.rank} is not |${device.value}|`);
      const forced = golden.name === "excluded-and-zero" && name === "neuron/L1n5";
      if ((forced || margin[u] >= 1e-3 * Math.max(1, Math.abs(peak[u]))) && device.row !== row[u]) {
        throw new Error(`${item.name} ${name}: peak row ${device.row}, golden ${row[u]}`);
      }
    });
    // The host ranking (native `_activity`): per kind, rank desc, flat index asc; rank −1 drops.
    const flat = allLayers.flatMap((layer) => [
      ...Array.from({ length: F }, (_, n) => ["neuron", layer, n]),
      ...Array.from({ length: H }, (_, h) => ["head", layer, h]),
    ]);
    const ranked = [];
    for (const kind of ["neuron", "head"]) {
      const items = flat.map((unit, position) => ({ unit, position, ...at(...unit) })).filter((x) => x.unit[0] === kind);
      items.sort((a, b) => b.rank - a.rank || a.position - b.position);
      ranked.push(...items.slice(0, golden.arguments.top).filter((x) => x.rank >= 0));
    }
    const listed = golden.result.top;
    if (ranked.length !== listed.length) throw new Error(`${item.name}: ${ranked.length} ranked, golden ${listed.length}`);
    const ties = activityGolden.declared_ties.find((t) => t.path === `cases[${index}].result.top`)?.groups ?? [];
    ranked.forEach((entry, position) => {
      const [kind, layer, i] = entry.unit;
      const name = `${kind}/L${layer}${kind === "neuron" ? "n" : "h"}${i}`;
      const group = ties.find((g) => g.includes(position));
      const expected = group ? listed.find((e, j) => group.includes(j) && e.unit === name) : listed[position];
      if (expected === undefined || expected.unit !== name) throw new Error(`${item.name}: top[${position}] ${name}, golden ${listed[position].unit}`);
      close(`top ${name}`, expected.peak, entry.value, z, 0);
      if (!golden.tied_peak_rows.includes(name) && entry.row !== expected.raw_index) {
        throw new Error(`${item.name}: ${name} peaks at ${entry.row}, golden ${expected.raw_index}`);
      }
    });
    (golden.arguments.units ?? []).forEach((unit, slot) => {
      const rows = golden.result.tokens.length;
      golden.result.traces[unit].forEach((value, r) => close(`trace ${unit}[${r}]`, value, result.traces[slot * rows + r], z, 0));
    });
    if (golden.arguments.raw_index !== undefined) {
      for (const layerEntry of golden.result.population.layers) {
        const width = F + H;
        const values = result.population.slice(layerEntry.layer * width, (layerEntry.layer + 1) * width);
        const expected = [...float16s(layerEntry.neurons), ...float16s(layerEntry.heads)];
        values.forEach((value, k) => {
          const rounded = toFloat16(value);
          const ulp = float16Spacing(expected[k]);
          if (!(Math.abs(rounded - expected[k]) <= ulp)) {
            throw new Error(`${item.name}: population L${layerEntry.layer}[${k}] ${value} vs ${expected[k]}`);
          }
        });
      }
    }
  }
  return worst;
}

function int32s(value) {
  const bytes = Buffer.from(value.data, "base64");
  return Array.from(new Int32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)));
}

function float16s(text) {
  const bytes = Buffer.from(text, "base64");
  const halves = new Uint16Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  return Array.from(halves, fromFloat16Bits);
}

function fromFloat16Bits(bits) {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x3ff;
  if (exponent === 0) return sign * fraction * 2 ** -24;
  if (exponent === 31) return fraction ? Number.NaN : sign * Infinity;
  return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
}

function toFloat16(value) {
  const halves = new Float16Array(1);
  halves[0] = value;
  return halves[0];
}

function float16Spacing(value) {
  const magnitude = Math.abs(value);
  if (magnitude < 2 ** -14) return 2 ** -24;
  return 2 ** (Math.floor(Math.log2(magnitude)) - 10);
}

async function loadTensorCache(directory) {
  const cache = JSON.parse(await readFile(resolve(directory, "tensor-cache.json"), "utf8"));
  const out = new Map();
  for (const shard of cache.records) {
    const data = await readFile(resolve(directory, shard.dataPath));
    for (const record of shard.records) {
      const bytes = data.subarray(record.byteOffset, record.byteOffset + record.nbytes);
      const copy = new Uint8Array(bytes).buffer;
      const values = record.dtype === "uint32" ? new Uint32Array(copy) : new Float32Array(copy);
      out.set(record.name, { shape: record.shape, values });
    }
  }
  return out;
}

function parseArguments(args) {
  const result = { headed: false, port: 0, suite: "all" };
  for (let i = 0; i < args.length; i += 1) {
    const key = args[i];
    if (key === "--headed") result.headed = true;
    else if (["--webllm", "--fixtures", "--arch", "--quantization", "--port", "--suite"].includes(key)) result[key.slice(2)] = args[++i];
    else throw new Error(`unknown argument ${key}`);
  }
  for (const key of ["webllm", "fixtures", "arch", "quantization"]) {
    if (!result[key]) throw new Error(`--${key} is required`);
  }
  if (!["all", "units", "replay", "reads"].includes(result.suite)) throw new Error("--suite is all, units, replay or reads");
  result.port = Number(result.port);
  return result;
}

// ------------------------------------------------------------------- the reads suite ---

/** Plain values crossing into the page: typed arrays travel as `{ __typed, values }`. */
function toPage(value) {
  if (ArrayBuffer.isView(value)) return { __typed: value.constructor.name, values: Array.from(value) };
  if (Array.isArray(value)) return value.map(toPage);
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, toPage(item)]));
  }
  return value;
}

function fromPage(value) {
  if (value !== null && typeof value === "object" && "__typed" in value) return new globalThis[value.__typed](value.values);
  if (Array.isArray(value)) return value.map(fromPage);
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, fromPage(item)]));
  }
  return value;
}

// In the page: the adapter's surface over the bridge, kept for the member calls.
async function installReads({ capabilities }) {
  const engine = window.engine;
  const { webLlmBlockSitesBackend } = await import("/src/hosted/runtime/webLlmBlockSites.ts");
  window.readsBackend = webLlmBlockSitesBackend(capabilities, {
    call: (request) => engine.drowseBlockSites(request, "units"),
    installProgram: (program) => engine.setDrowseStructuredProgram(program, "units"),
  });
}

async function callRead({ member, args }) {
  const revive = (value) => {
    if (value !== null && typeof value === "object" && "__typed" in value) return new globalThis[value.__typed](value.values);
    if (Array.isArray(value)) return value.map(revive);
    if (value !== null && typeof value === "object") {
      return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, revive(item)]));
    }
    return value;
  };
  const pack = (value) => {
    if (ArrayBuffer.isView(value)) return { __typed: value.constructor.name, values: Array.from(value) };
    if (Array.isArray(value)) return value.map(pack);
    if (value !== null && typeof value === "object") {
      return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, pack(item)]));
    }
    return value;
  };
  try {
    return { ok: pack(await window.readsBackend[member](...args.map(revive))) };
  } catch (error) {
    return { error: String(error?.message ?? error) };
  }
}

/** The converted tensors as the reads' formulas use them (fp64). */
function readWeights() {
  const name = (suffix) => [...tensors.keys()].find((key) =>
    [suffix, suffix.replace(".weight", ".q_weight")].some((want) => key === want || key.endsWith(`.${want}`)));
  const weight = (suffix) => {
    const found = name(suffix + ".weight");
    return dequantized(found.slice(0, found.lastIndexOf(".")));
  };
  const vector = (suffix) => tensors.get(name(suffix + ".weight")).values;
  const ones = () => new Float64Array(d).fill(1);
  const gain = (layer, which) => {
    const names = {
      ai: "input_layernorm",
      ao: gemma ? "post_attention_layernorm" : null,
      mi: gemma ? "pre_feedforward_layernorm" : "post_attention_layernorm",
      mo: gemma ? "post_feedforward_layernorm" : null,
    };
    return names[which] === null ? ones() : vector(`layers.${layer}.${names[which]}`);
  };
  const dFF = blockSpec.dFF;
  const H = blockSpec.nHeads;
  const KV = blockSpec.nKvHeads;
  const rows = (w, first, count) => Array.from({ length: count }, (_, r) => w.subarray((first + r) * w.columns, (first + r + 1) * w.columns));
  const qkv = (layer) => {
    if (name(`layers.${layer}.self_attn.c_attn.weight`) !== undefined) {
      const w = weight(`layers.${layer}.self_attn.c_attn`);
      return { q: rows(w, 0, H * hd), k: rows(w, H * hd, KV * hd), v: rows(w, (H + KV) * hd, KV * hd) };
    }
    const part = (role, count) => rows(weight(`layers.${layer}.self_attn.${role}_proj`), 0, count);
    return { q: part("q", H * hd), k: part("k", KV * hd), v: part("v", KV * hd) };
  };
  return {
    /** `[d_ff]` columns `s·γ_mo ⊙ W_down[:, n]`. */
    writes(layer) {
      const w = weight(`layers.${layer}.mlp.down_proj`);
      const g = gain(layer, "mo");
      return Array.from({ length: dFF }, (_, n) => Float64Array.from({ length: d }, (_, k) => g[k] * w[k * w.columns + n]));
    },
    reads(layer) {
      const w = weight(`layers.${layer}.mlp.gate_up_proj`);
      const g = gain(layer, "mi");
      const unit = (row) => {
        const scaled = Float64Array.from(row, (v, k) => v * g[k]);
        const norm = Math.max(Math.sqrt(scaled.reduce((s, v) => s + v * v, 0)), 1e-12);
        return scaled.map((v) => v / norm);
      };
      return [rows(w, 0, dFF).map(unit), rows(w, dFF, dFF).map(unit)];
    },
    /** `[d, hd]` row-major: `γ_ao ⊙ W_O[:, h·hd:(h+1)·hd]`. */
    headWrite(layer, head) {
      const w = weight(`layers.${layer}.self_attn.o_proj`);
      const g = gain(layer, "ao");
      return Float64Array.from({ length: d * hd }, (_, i) => g[Math.floor(i / hd)] * w[Math.floor(i / hd) * w.columns + head * hd + (i % hd)]);
    },
    /** `[3, hd, d]`: `γ_ai ⊙` the query, key and value rows. */
    headReads(layer, head) {
      const { q, k, v } = qkv(layer);
      const g = gain(layer, "ai");
      const group = Math.floor(head / (H / KV));
      const out = new Float64Array(3 * hd * d);
      [q.slice(head * hd, (head + 1) * hd), k.slice(group * hd, (group + 1) * hd), v.slice(group * hd, (group + 1) * hd)]
        .forEach((block, role) => block.forEach((row, i) => row.forEach((value, c) => { out[(role * hd + i) * d + c] = g[c] * value; })));
      return out;
    },
    gram: (layer, head) => headGram(layer, head),
    finalGain: () => vector("norm"),
    embedding: () => weight("embed_tokens"),
  };
}
