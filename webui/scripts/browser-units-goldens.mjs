#!/usr/bin/env node
// Hook ABI v5 on WebGPU: drowse's units goldens through a fixture library and the WebLLM bridge.
//
//   node scripts/browser-units-goldens.mjs --webllm WEB_LLM/lib/index.js \
//     --fixtures DIR --arch qwen3|gemma3 --quantization q0f32|q4f32_1 [--headed]
//
// DIR holds build-units-fixtures.py's model directory `units-<arch>-<quantization>` and its
// library `libs/units-<arch>-<quantization>-v5.wasm`. The bridge's schema-4 descriptor is attested
// with the adapter's own function (C0's expected vector); then every block-sites, unit-ablation and
// unit-probes case runs as a raw greedy completion under a drowse-structured-v4 program with eight
// linear probes per layer (`out[layer, last, 0:8]`): per forward the greedy token, its logprob,
// the probe readings and the live unit taps are checked against the files' library tolerance.
import { createServer as createHttpServer } from "node:http";
import { readFile } from "node:fs/promises";
import { basename, extname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { createServer as createViteServer } from "vite";
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
  if (value.encoding !== "float32-le-base64") return Float32Array.from(value.values ?? value);
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
  if (text === "prompt") return { prompt: true, generated: false };
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
  for (const item of cases) {
    const result = await page.evaluate(runCase, { item, textIds: gemma ? prompt.slice(1) : prompt });
    const worst = checkCase(item, result);
    console.log(`${item.name}: ${result.forwards.length} forwards, worst |Δ| ${worst.toExponential(2)}`);
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
  const result = { headed: false, port: 0 };
  for (let i = 0; i < args.length; i += 1) {
    const key = args[i];
    if (key === "--headed") result.headed = true;
    else if (["--webllm", "--fixtures", "--arch", "--quantization", "--port"].includes(key)) result[key.slice(2)] = args[++i];
    else throw new Error(`unknown argument ${key}`);
  }
  for (const key of ["webllm", "fixtures", "arch", "quantization"]) {
    if (!result[key]) throw new Error(`--${key} is required`);
  }
  result.port = Number(result.port);
  return result;
}
