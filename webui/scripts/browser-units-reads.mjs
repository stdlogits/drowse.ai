// Hook ABI v5's static reads on WebGPU (the `reads` suite of browser-units-goldens.mjs): every
// static-read member of the adapter (webLlmStaticReads.ts over the bridge's `read` requests) on a
// fixture library, against formulas over the converted tensors and against the goldens composed as
// the host composes them (neuron-push, unit-reads, unit-artifacts, slens, lens-reads), the same
// checks `browser-runtime/forks/static_reads.py` runs on the C TVM VM.

const SEED = 0x5eed0005;
const LENS_ORDINAL = 1000;
const LENS_LAYERS = [["R", "mid", [1, 3]], ["R", "out", [0, 1]], ["S", "mid", [0, 2]], ["S", "out", [0, 1]]];
const R_STD_FLOOR = 1e-4;
const S_STD_FLOOR = 1e-6;

const mix = (x) => {
  x = (x ^ (x >>> 16)) >>> 0;
  x = Math.imul(x, 0x7feb352d) >>> 0;
  x = (x ^ (x >>> 15)) >>> 0;
  x = Math.imul(x, 0x846ca68b) >>> 0;
  return (x ^ (x >>> 16)) >>> 0;
};

/** `browser_units.lens_matrix`: `0.75·δ_ij + ((u >> 16) mod 256 − 128)·2⁻¹⁰`. */
export function fixtureLenses(d) {
  const lenses = new Map();
  LENS_LAYERS.forEach(([family, site, layers], i) => {
    for (const layer of layers) {
      const ordinal = LENS_ORDINAL + 16 * i + layer;
      const seed = mix((SEED + Math.imul(0x9e3779b9, ordinal)) >>> 0);
      const m = new Float32Array(d * d);
      for (let k = 0; k < d * d; k += 1) {
        m[k] = (k % (d + 1) === 0 ? 0.75 : 0) + (((mix((k ^ seed) >>> 0) >>> 16) % 256) - 128) * 2 ** -10;
      }
      lenses.set(`${family}:${site}:${layer}`, { family: family.toLowerCase(), site, layer, matrix: m });
    }
  });
  return lenses;
}

/** Every reads check; `call(member, ...args)` runs an adapter member in the page. */
export async function verifyReads({ call, f32, i32, u32, weights, golden, decode, L, d, dFF, H, KV, hd, V, sandwich, eps, embedScale }) {
  const worst = {};
  const tolerance = (await golden("lens-reads")).tolerance.library;
  const check = (label, expected, actual, atol = tolerance.atol, rtol = tolerance.rtol) => {
    const e = Array.from(expected);
    const a = Array.from(actual);
    if (e.length !== a.length) throw new Error(`${label}: ${a.length} values, expected ${e.length}`);
    let most = 0;
    e.forEach((value, i) => {
      const error = Math.abs(a[i] - value);
      if (!(error <= atol + rtol * Math.abs(value))) {
        throw new Error(`${label}[${i}]: ${a[i]} vs ${value} (|Δ| ${error.toExponential(2)})`);
      }
      most = Math.max(most, error);
    });
    const key = label.split(":")[0];
    worst[key] = Math.max(worst[key] ?? 0, most);
  };
  // A golden token list against the device's: the same ids, each id's value within `atol`,
  // the device's values non-increasing, and each position holding the golden's id except inside a
  // tie (a declared group, or golden values within the tolerance of each other).
  const tokens = (label, golden, ids, values, atol, groups = []) => {
    const expect = new Map(golden.map((t) => [t.token_id, t.value]));
    const got = new Map(Array.from(ids).map((id, i) => [id, values[i]]));
    const want = [...expect.keys()].sort((a, b) => a - b).join(",");
    if ([...got.keys()].sort((a, b) => a - b).join(",") !== want) {
      throw new Error(`${label}: ids ${[...got.keys()]} are not ${want}`);
    }
    check(label, golden.map((t) => t.value), golden.map((t) => got.get(t.token_id)), atol + 2e-6, 0);
    for (let i = 0; i + 1 < values.length; i += 1) {
      if (values[i] < values[i + 1]) throw new Error(`${label}: the device's values are not non-increasing at ${i}`);
    }
    const parent = golden.map((_, i) => i);
    const find = (i) => { while (parent[i] !== i) i = parent[i] = parent[parent[i]]; return i; };
    const window = 2 * (atol + 2e-6);
    for (let i = 0; i + 1 < golden.length; i += 1) {
      if (Math.abs(golden[i].value - golden[i + 1].value) <= window) parent[find(i + 1)] = find(i);
    }
    for (const group of groups) for (const p of group.slice(1)) parent[find(p)] = find(group[0]);
    const blocks = new Map();
    golden.forEach((_, i) => blocks.set(find(i), [...(blocks.get(find(i)) ?? []), i]));
    for (const block of blocks.values()) {
      const a = block.map((p) => golden[p].token_id).sort((x, y) => x - y).join(",");
      const b = block.map((p) => ids[p]).sort((x, y) => x - y).join(",");
      if (a !== b) throw new Error(`${label}: positions ${block} hold ${b}, the golden ${a}`);
    }
  };
  const declared = (doc, path) =>
    (doc.declared_ties ?? []).filter((tie) => tie.path === path).flatMap((tie) => tie.groups);

  // Residents: the fixture lenses and the S embedding statistics.
  const lenses = fixtureLenses(d);
  const resident = new Map();
  for (const [key, lens] of lenses) {
    resident.set(key, await call("uploadLensLayer", lens.family, lens.site, lens.layer, f32(lens.matrix, [d, d])));
  }
  const slens = await golden("slens");
  const meanE = Float32Array.from(decode(slens.mean_E));
  const sigmaE = Float32Array.from(decode(slens.sigma_E));
  const mask = new Uint32Array(Math.ceil(V / 32));
  for (const v of slens.mask.kept) mask[v >> 5] |= 1 << (v & 31);
  const stats = await call("uploadEmbeddingStatistics", { meanE: f32(meanE, [d]), sigmaE: f32(sigmaE, [d, d]), mask: u32(mask, [mask.length]) });

  // ---- formulas over the converted tensors ----
  const rng = (() => { let s = 0x4b4c; return () => ((s = mix(s + 0x9e3779b9)) / 2 ** 32) * 2 - 1; })();
  const random = (n) => Float32Array.from({ length: n }, rng);
  const layers = Int32Array.of(L - 1, 0, 2 % L);
  const c = layers.length;
  const x = random(c * 2 * d);
  const writeScores = await call("layerWriteScores", f32(x, [c, 2, d]), layers);
  const readScores = await call("layerReadScores", f32(x, [c, 2, d]), layers);
  for (let j = 0; j < c; j += 1) {
    const w = weights.writes(layers[j]);
    const [a, b] = weights.reads(layers[j]);
    for (let r = 0; r < 2; r += 1) {
      const row = x.subarray((j * 2 + r) * d, (j * 2 + r + 1) * d);
      check("formula:layer_write_scores", w.map((col) => dot(col, row)), writeScores.data.subarray((j * 2 + r) * dFF, (j * 2 + r + 1) * dFF));
      check("formula:layer_read_scores", a.map((col) => dot(col, row)), readScores.data.subarray(((j * 2) * 2 + r) * dFF, ((j * 2) * 2 + r + 1) * dFF));
      check("formula:layer_read_scores", b.map((col) => dot(col, row)), readScores.data.subarray(((j * 2 + 1) * 2 + r) * dFF, ((j * 2 + 1) * 2 + r + 1) * dFF));
    }
  }
  const g = random(c * d);
  const act = random(c * dFF);
  const mlpO = random(c * d);
  const z = random(c * H * hd);
  const attnO = random(c * d);
  const neurons = await call("neuronContributions", f32(g, [c, d]), f32(act, [c, dFF]), f32(mlpO, [c, d]), layers);
  const heads = await call("headContributions", f32(g, [c, d]), f32(z, [c, H * hd]), f32(attnO, [c, d]), layers);
  const norms = await call("headWriteNorms", f32(z, [c, H * hd]), f32(attnO, [c, d]), layers);
  const rms = (v) => (sandwich ? 1 / Math.sqrt(v.reduce((s, t) => s + t * t, 0) / v.length + eps) : 1);
  for (let j = 0; j < c; j += 1) {
    const gj = g.subarray(j * d, (j + 1) * d);
    const rMo = rms(mlpO.subarray(j * d, (j + 1) * d));
    const rAo = rms(attnO.subarray(j * d, (j + 1) * d));
    const w = weights.writes(layers[j]);
    check("formula:neuron_contributions", w.map((col, n) => act[j * dFF + n] * dot(col, gj) * rMo), neurons.data.subarray(j * dFF, (j + 1) * dFF));
    const per = [];
    const nrm = [];
    for (let h = 0; h < H; h += 1) {
      const block = weights.headWrite(layers[j], h);
      const zh = z.subarray(j * H * hd + h * hd, j * H * hd + (h + 1) * hd);
      const bz = new Float64Array(d);
      for (let k = 0; k < d; k += 1) for (let i = 0; i < hd; i += 1) bz[k] += block[k * hd + i] * zh[i];
      per.push(rAo * dot(bz, gj));
      nrm.push(rAo * Math.sqrt(Math.max(0, dot(bz, bz))));
    }
    check("formula:head_contributions", per, heads.data.subarray(j * H, (j + 1) * H));
    check("formula:head_write_norms", nrm, norms.data.subarray(j * H, (j + 1) * H));
  }
  const headUnits = [{ kind: "head", layer: L - 1, index: H - 1 }, { kind: "head", layer: 0, index: 1 }];
  const headWrite = await call("unitWeightRows", "head_write", headUnits);
  const headRead = await call("unitWeightRows", "head_read", headUnits);
  const grams = await call("headWriteGrams", headUnits.map((u) => ({ layer: u.layer, head: u.index })));
  headUnits.forEach((u, i) => {
    check("formula:head_rows", weights.headWrite(u.layer, u.index), headWrite.data.subarray(i * d * hd, (i + 1) * d * hd));
    check("formula:head_rows", weights.headReads(u.layer, u.index), headRead.data.subarray(i * 3 * hd * d, (i + 1) * 3 * hd * d));
    check("formula:head_write_grams", weights.gram(u.layer, u.index), grams.data.subarray(i * hd * hd, (i + 1) * hd * hd));
  });
  const ids = Int32Array.of(0, 3, 71, V - 1);
  const hFinal = random(ids.length * d);
  const cotangent = await call("unembedCotangent", f32(hFinal, [ids.length, d]), ids);
  const gammaF = weights.finalGain();
  const E = weights.embedding();
  ids.forEach((v, i) => {
    const h = hFinal.subarray(i * d, (i + 1) * d);
    const r = 1 / Math.sqrt(h.reduce((s, t) => s + t * t, 0) / d + eps);
    check("formula:unembed_cotangent", Array.from({ length: d }, (_, k) => r * gammaF[k] * E[v * d + k]), cotangent.data.subarray(i * d, (i + 1) * d));
  });
  const rows = random(2 * d);
  const logits = rows.length && (await call("vocabTokenLogits", f32(rows, [2, d]), i32(Int32Array.from({ length: 2 * V }, (_, i) => i % V), [2, V])));
  for (let r = 0; r < 2; r += 1) {
    const row = rows.subarray(r * d, (r + 1) * d);
    const scale = 1 / Math.sqrt(row.reduce((s, t) => s + t * t, 0) / d + eps);
    const expect = Array.from({ length: V }, (_, v) => {
      let acc = 0;
      for (let k = 0; k < d; k += 1) acc += row[k] * scale * gammaF[k] * E[v * d + k];
      return acc;
    });
    check("formula:vocab_logits", expect, logits.logits.data.subarray(r * V, (r + 1) * V));
  }
  const moments = await call("vocabLogitMoments", f32(rows, [2, d]), stats);
  for (let r = 0; r < 2; r += 1) {
    const kept = slens.mask.kept.map((v) => logits.logits.data[r * V + v]);
    const mean = kept.reduce((s, t) => s + t, 0) / kept.length;
    const m = (p) => kept.reduce((s, t) => s + (t - mean) ** p, 0) / kept.length;
    const variance = Math.max(m(2), 1e-30);
    check("formula:logit_moments", [m(3) / variance ** 1.5, m(4) / variance ** 2], moments.data.subarray(2 * r, 2 * r + 2), 1e-4, 1e-3);
  }
  const big = Float32Array.from({ length: 2 * 700 }, () => Math.round(rng() * 64) / 8);
  for (const k of [1, 7, 300]) {
    const top = await call("topK", f32(big, [2, 700]), k);
    for (let r = 0; r < 2; r += 1) {
      const order = [...Array(700).keys()].sort((p, q) => big[r * 700 + q] - big[r * 700 + p] || p - q).slice(0, k);
      if (order.join(",") !== Array.from(top.indices.data.subarray(r * k, (r + 1) * k)).join(",")) {
        throw new Error(`formula:topk k=${k}: indices differ from value desc, index asc`);
      }
    }
  }

  // ---- goldens ----
  const push = await golden("neuron-push");
  for (const [layer, vector] of Object.entries(push.profile)) {
    const index = Number(/neuron\/L\d+n(\d+)/.exec(JSON.stringify(push.cases))[1]);
    const row = await call("unitWeightRows", "neuron_write", [{ kind: "neuron", layer: Number(layer), index }]);
    check("neuron-push:profile", decode(vector), row.data);
  }
  const unitReads = await golden("unit-reads");
  for (const entry of unitReads.reads) {
    const read = entry.read;
    if (read.kind !== "neuron") continue;
    const ref = [{ kind: "neuron", layer: read.layer, index: read.index }];
    const w = (await call("unitWeightRows", "neuron_write", ref)).data;
    const ab = (await call("unitWeightRows", "neuron_read", ref)).data;
    check("unit-reads:w_norm", [read.scalars.w_norm], [Math.sqrt(dot(w, w))], 2e-6 + tolerance.atol);
    check("unit-reads:cos_ab", [read.scalars.cos_ab], [dot(ab.subarray(0, d), ab.subarray(d))], 2e-6 + tolerance.atol);
  }

  const rMatrix = (layer, site) => resident.get(`R:${site}:${layer}`) ?? (site === "out" && layer === L - 1 ? null : undefined);
  const transportR = async (data, n, lens) => (lens === null ? data : (await call("lensTransport", f32(data, [1, n, d]), [lens], true)).data);
  const pullS = async (data, n, lens) => (await call("lensTransport", f32(data, [1, n, d]), [lens], false)).data;
  const allIds = (n) => i32(Int32Array.from({ length: n * V }, (_, i) => i % V), [n, V]);
  const chan = async (data, n, width) => {
    const merged = await call("chanMerge", { count: 0, mean: f32(new Float32Array(width), [width]), m2: f32(new Float32Array(width), [width]) }, f32(data, [n, width]));
    return { mean: merged.mean.data, std: merged.m2.data.map((v) => Math.sqrt(Math.max(v / n, 0))) };
  };
  const embedded = (await call("vocabRowsTimes", {
    kind: "embed", ids: Int32Array.from({ length: V }, (_, i) => i),
    factor: f32(Float32Array.from({ length: d * d }, (_, i) => (i % (d + 1) === 0 ? 1 : 0)), [d, d]), statistics: stats,
  })).data;
  const artifacts = await golden("unit-artifacts");
  for (const { layer, tensors } of artifacts.layers) {
    const units = Array.from({ length: dFF }, (_, n) => ({ kind: "neuron", layer, index: n }));
    const writes = (await call("unitWeightRows", "neuron_write", units)).data;
    const ab = (await call("unitWeightRows", "neuron_read", units)).data;
    const pole = (sign) => Float32Array.from({ length: dFF * d }, (_, i) => {
      const n = Math.floor(i / d);
      const k = i % d;
      return ab[(2 * n) * d + k] + sign * ab[(2 * n + 1) * d + k];
    });
    const populations = { writes: [writes, "out"], "fires+": [pole(1), "mid"], "fires-": [pole(-1), "mid"] };
    const pooled = [];
    for (const [name, [rowsData, site]] of Object.entries(populations)) {
      const key = `R.${name}`;
      if (tensors[`${key}.rho`] !== undefined) {
        const xR = await transportR(rowsData, dFF, rMatrix(layer, site));
        const full = (await call("vocabTokenLogits", f32(xR, [dFF, d]), allIds(dFF))).logits.data;
        const lse = (await call("logSumExp", f32(full, [dFF, V]))).data;
        const rho = Array.from({ length: dFF }, (_, n) => 1 / Math.sqrt(xR.subarray(n * d, (n + 1) * d).reduce((s, t) => s + t * t, 0) / d + eps));
        check(`unit-artifacts:${key}.rho`, decode(tensors[`${key}.rho`]), rho);
        check(`unit-artifacts:${key}.lse`, decode(tensors[`${key}.lse`]), lse);
        if (tensors[`${key}.lse_neg`] !== undefined) {
          check(`unit-artifacts:${key}.lse_neg`, decode(tensors[`${key}.lse_neg`]), (await call("logSumExp", f32(full.map((v) => -v), [dFF, V]))).data);
        }
        const { mean, std } = await chan(full, dFF, V);
        check(`unit-artifacts:${key}.mean`, decode(tensors[`${key}.mean`]), mean);
        check(`unit-artifacts:${key}.std`, decode(tensors[`${key}.std`]), std.map((v) => Math.max(v, R_STD_FLOOR)));
      }
      const sKey = `S.${name}`;
      if (tensors[`${sKey}.norm`] === undefined) continue;
      const u = await pullS(rowsData, dFF, resident.get(`S:${site}:${layer}`));
      const norm = (await call("groupedRowNorms", f32(u, [dFF, d]), d)).data;
      check(`unit-artifacts:${sKey}.norm`, decode(tensors[`${sKey}.norm`]), norm);
      const q = (await call("batchedQuadraticForm", f32(u, [dFF, d]), stats)).data;
      check(`unit-artifacts:${sKey}.footprint_std`, decode(tensors[`${sKey}.footprint_std`]), q.map((v) => Math.sqrt(Math.max(v, 1e-30))));
      const unitU = Float32Array.from(u, (v, i) => v / Math.max(norm[Math.floor(i / d)], 1e-12));
      const raw = (await call("batchedMatmul", f32(unitU, [1, dFF, d]), f32(transpose(embedded, V, d), [1, d, V]))).data;
      if (name === "writes") {
        const { mean, std } = await chan(raw, dFF, V);
        check(`unit-artifacts:${sKey}.mean`, decode(tensors[`${sKey}.mean`]), mean);
        check(`unit-artifacts:${sKey}.std`, decode(tensors[`${sKey}.std`]), std.map((v) => Math.max(v, S_STD_FLOOR)));
      } else {
        pooled.push(raw);
      }
    }
    if (pooled.length > 0) {
      const both = new Float32Array(2 * dFF * V);
      both.set(pooled[0]);
      both.set(pooled[1], dFF * V);
      const { mean, std } = await chan(both, 2 * dFF, V);
      check("unit-artifacts:S.fires.mean", decode(tensors["S.fires.mean"]), mean);
      check("unit-artifacts:S.fires.std", decode(tensors["S.fires.std"]), std.map((v) => Math.max(v, S_STD_FLOOR)));
    }
  }

  const words = new Map(slens.probe_words.map((word, i) => [word, slens.probe_word_ids[i]]));
  for (const atom of slens.atoms) {
    const name = atom.atom.split("/")[1];
    const token = name.startsWith("#") ? Number(name.slice(1)) : words.get(name);
    const layersOf = Object.keys(atom.layers).map(Number);
    const out = await call("footprintDirections", layersOf.map((l) => resident.get(`S:out:${l}`)), Int32Array.of(token), stats);
    layersOf.forEach((l, i) => check("slens:atom", decode(atom.layers[String(l)]), out.data.subarray(i * d, (i + 1) * d)));
  }
  for (const [f, footprint] of slens.footprints.entries()) {
    const u = Float32Array.from(decode(footprint.u));
    const n = u.length / d;
    const out = await call("footprintTopK", f32(u, [n, d]), footprint.k, stats);
    footprint.top.forEach((top, r) => tokens("slens:footprint", top.map((t) => ({ token_id: t.token_id, value: t.z })),
      out.ids.data.subarray(r * footprint.k, (r + 1) * footprint.k), out.z.data.subarray(r * footprint.k, (r + 1) * footprint.k), tolerance.z_atol,
      declared(slens, `footprints[${f}].top[${r}]`)));
  }

  const stats3 = await golden("unit-stats");
  const means = { out: decode(stats3.out_mean), mid: decode(stats3.mid_mean) };
  // lens.read's two halves of `n` rows at (layer, site): R probabilities and S z, or null where
  // the fixture has no lens there.
  const place = async (rowsData, n, layer, site, k, of) => {
    const rLens = rMatrix(layer, site);
    let r = null;
    if (rLens !== undefined) {
      const out = await call("vocabReadoutTopK", f32(await transportR(rowsData, n, rLens), [n, d]), k);
      r = { ids: out.ids.data, p: out.logits.data.map((v, i) => Math.exp(v - out.logSumExp.data[Math.floor(i / k)])) };
    }
    let s = null;
    const sLens = resident.get(`S:${site}:${layer}`);
    if (sLens !== undefined) {
      const centred = of === "residual"
        ? rowsData.map((v, i) => v - means[site][layer * d + (i % d)])
        : rowsData;
      const out = await call("footprintTopK", f32(await pullS(centred, n, sLens), [n, d]), k, stats);
      s = { ids: out.ids.data, p: out.z.data };
    }
    return { r, s, k };
  };
  // One row's present raw halves against a place read (hub-hidden layer_z halves are B5's).
  const checkPlace = (label, want, got, row, doc, path, where) => {
    for (const [half, atol] of [["r", tolerance.probability_atol], ["s", tolerance.z_atol]]) {
      const golden = want[half];
      if (golden === null || golden === undefined || golden.absent !== null || golden.unit === "layer_z") continue;
      if (got[half] === null) throw new Error(`${label}: a present ${half} half at ${where} has no fixture lens`);
      const m = golden.tokens.length;
      const at = row * got.k;
      tokens(`${label}:${half}`, golden.tokens, got[half].ids.subarray(at, at + m), got[half].p.subarray(at, at + m), atol,
        declared(doc, `${path}.${half}.tokens`));
    }
  };

  const lensReads = await golden("lens-reads");
  for (const [q, request] of lensReads.requests.entries()) {
    if (request.lens !== undefined && request.lens !== null) continue;
    const rowsData = Float32Array.from(decode(request.rows));
    const got = await place(rowsData, rowsData.length / d, request.layer, request.site, request.k, request.of);
    request.reads.forEach((read, i) => checkPlace("lens-reads", read, got, i, lensReads, `requests[${q}].reads[${i}]`, `L${request.layer} ${request.site}`));
  }

  // unit-reads poles: a neuron's writes (w, −w at its out) and firing directions (â ± b̂ at its
  // mid) through lens.read, and each pole's logit-lens skew and kurtosis over the kept tokens.
  for (const [c, entry] of unitReads.reads.entries()) {
    const read = entry.read;
    if (read.kind !== "neuron") continue;
    const ref = [{ kind: "neuron", layer: read.layer, index: read.index }];
    const w = (await call("unitWeightRows", "neuron_write", ref)).data;
    const ab = (await call("unitWeightRows", "neuron_read", ref)).data;
    const [a, b] = [ab.subarray(0, d), ab.subarray(d, 2 * d)];
    const moments = (await call("vocabLogitMoments", f32(w, [1, d]), stats)).data;
    const places = {
      writes: await place(Float32Array.from([...w, ...w.map((v) => -v)]), 2, read.layer, "out", entry.k, "direction"),
      fires_on: await place(Float32Array.from([...a.map((v, i) => v + b[i]), ...a.map((v, i) => v - b[i])]), 2, read.layer, "mid", entry.k, "direction"),
    };
    read.poles.forEach((pole, p) => {
      const sign = pole.pole === "+" ? 1 : -1;
      check("unit-reads:vocab_moments", [pole.vocab_skew, pole.vocab_kurt], [sign * moments[0], moments[1]], 2e-6 + tolerance.atol);
      for (const where of ["writes", "fires_on"]) {
        checkPlace(`unit-reads:${where}`, pole[where], places[where], p, unitReads, `reads[${c}].read.poles[${p}].${where}`, `L${read.layer} ${pole[where].site}`);
      }
    });
  }
  for (const handle of resident.values()) await call("release", handle);
  await call("release", stats);
  return worst;
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i += 1) s += a[i] * b[i];
  return s;
}

function transpose(data, rows, cols) {
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r += 1) for (let c = 0; c < cols; c += 1) out[c * rows + r] = data[r * cols + c];
  return out;
}
