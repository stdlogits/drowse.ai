import { compileStructuredHookProgram } from "../src/hosted/runtime/structuredHookProgram";
import { resolveStructuredHookProfile } from "../src/hosted/runtime/structuredHookProfile";

export async function verifySteeringPositions(runtime: any, manifest: any, options: { arithmeticOnly?: boolean } = {}) {
  const engine = runtime.requireEngine();
  const pipeline: any = [...engine.loadedModelIdToPipeline.values()][0];
  const originalForward = pipeline.embedAndForward.bind(pipeline);
  const hidden = manifest.hiddenSize;
  const layers = manifest.layerMap.length;
  const direction = Array.from({ length: hidden }, (_, i) => Math.sin(i + 1) / Math.sqrt(hidden / 2));
  const norm = Math.hypot(...direction);
  for (let i = 0; i < direction.length; i++) direction[i] /= norm;
  const zeros = new Array(hidden).fill(0);
  const layer = Math.floor(layers / 3);
  const structured = compileStructuredHookProgram(hidden,
    Array.from({ length: layers }, (_, i) => i === layer ? {
      affineGroups: [{ active: true, basis: [direction], neutral: zeros, target: [2], along: 1, kappa: [0] }],
    } : {}), resolveStructuredHookProfile(manifest.structuredHookProfile));
  const forced = (await runtime.tokenizeText(" clear skies and calm water beside the shore.")).slice(0, 7);
  const prompt = "The weather near the quiet lake was pleasant and the sky was";
  const run = async (program: any, positions?: any, reference?: Set<number>, referenceChunkSize = 1) => {
    const trace: { position: number; length: number; active: boolean }[] = [];
    if (reference !== undefined) {
      pipeline.embedAndForward = async (parts: number[][], length: number, useProgram = true) => {
        if (!useProgram) return originalForward(parts, length, useProgram);
        const tokens = parts.flat();
        let logits: any;
        for (let offset = 0; offset < tokens.length;) {
          const position = pipeline.filledKVCacheLength;
          const active = reference.has(position);
          let count = 1;
          while (count < referenceChunkSize && offset + count < tokens.length &&
            reference.has(position + count) === active) count++;
          await pipeline.updateDrowseStructuredControls(
            active ? program.affineActive : new Uint32Array(program.affineActive.length),
            active ? program.curveActive : new Uint32Array(program.curveActive.length),
          );
          logits?.dispose();
          logits = await originalForward([tokens.slice(offset, offset + count)], count, useProgram);
          trace.push({ position, length: count, active });
          offset += count;
        }
        return logits;
      };
    }
    const rows: { tokenId: number; logprob: number }[] = [];
    try {
      const result = await runtime.streamGeneration({
        input: { kind: "raw", prompt },
        sampling: { max_tokens: forced.length, temperature: 1, top_p: 1, top_k: 2147483647, seed: 1, return_top_k: 0 },
        replay: { forcedPrefixTokenIds: forced, scoreTokenIds: forced },
        hookProgram: program,
        ...positions,
        onRawToken(token: any) {
          if (program.controls?.affine.some((control: any) => control?.gate != null) &&
            token.hookMeasurements?.[layer * program.profile.maxProbes] !== 1) {
            throw new Error("The gate oracle requires its declared constant probe to read exactly 1");
          }
          const score = token.replayScore?.requestedLogprobs.find((row: any) => row.tokenId === token.tokenId);
          if (!score) throw new Error("Replay omitted the requested raw-model score");
          rows.push({ tokenId: token.tokenId, logprob: score.logprob });
        },
      }, () => {});
      if (rows.length !== forced.length) throw new Error(`Expected ${forced.length} forced tokens, got ${rows.length}`);
      return { rows, promptTokens: result.usage.promptTokens, trace };
    } finally {
      pipeline.embedAndForward = originalForward;
    }
  };
  const compare = (label: string, actual: any, expected: any, tolerance: number) => {
    let maxLogprobError = 0;
    for (let i = 0; i < expected.rows.length; i++) {
      if (actual.rows[i].tokenId !== expected.rows[i].tokenId) throw new Error(`${label}: token mismatch at ${i}`);
      const error = Math.abs(actual.rows[i].logprob - expected.rows[i].logprob);
      if (!Number.isFinite(error)) throw new Error(`${label}: non-finite log probabilities at ${i}: ${actual.rows[i].logprob}, ${expected.rows[i].logprob}`);
      maxLogprobError = Math.max(maxLogprobError, error);
    }
    if (maxLogprobError > tolerance) throw new Error(`${label}: max logprob error ${maxLogprobError} > ${tolerance}`);
    return { label, maxLogprobError, tolerance, tokens: actual.rows.length };
  };
  const checks = [];
  const batchingDiagnostics = [];
  const originalChunkSize = pipeline.prefillChunkSize;
  // Exercise cross-chunk boundaries with the real compiled prefill/decode kernels.
  pipeline.prefillChunkSize = Math.min(4, originalChunkSize);
  try {
    if (!options.arithmeticOnly) {
    for (const [name, program] of [["structured", structured]] as [string, any][]) {
      const control = { ...program, affineActive: new Uint32Array(program.affineActive.length), curveActive: new Uint32Array(program.curveActive.length) };
      const baseline = await run(control);
      const all = await run(program);
      const p = baseline.promptTokens;
      const spans = [[1, 3], [5, 8], [p, p + 2]];
      const indices = spans.flatMap(([start, end]) => Array.from({ length: end - start }, (_, i) => start + i));
      const expected = await run(program, undefined, new Set(indices));
      const matched = await run(program, undefined, new Set(indices), pipeline.prefillChunkSize);
      const matchedControl = await run(control, undefined, new Set(indices), pipeline.prefillChunkSize);
      const actual = await run(program, { position_spans: spans });
      checks.push(compare(`${name}: spans vs independent matched-batch reference`, actual, matched, 0));
      pipeline.prefillChunkSize = 1;
      try {
        checks.push(compare(`${name}: single-token spans vs independent per-token reference`,
          await run(program, { position_spans: spans }), expected, 0));
      } finally {
        pipeline.prefillChunkSize = Math.min(4, originalChunkSize);
      }
      // Different prefill/decode batch shapes can round differently, especially
      // in FP16. Record that separately; position parity uses matched shapes.
      const unsteeredPerToken = await run(control, undefined, new Set());
      const steeredBatching = compare(`${name}: batched vs per-token steering`, actual, expected, Infinity);
      const unsteeredBatching = compare(`${name}: batched vs per-token no injection`, baseline, unsteeredPerToken, Infinity);
      batchingDiagnostics.push({ name, steeredMaxLogprobError: steeredBatching.maxLogprobError,
        unsteeredMaxLogprobError: unsteeredBatching.maxLogprobError,
        meetsFp32ReferenceTolerance: steeredBatching.maxLogprobError <= 5e-4,
        matchedReferenceTrace: matched.trace });
      checks.push(compare(`${name}: indices vs spans`, await run(program, { position_indices: indices }), actual, 0));
      checks.push(compare(`${name}: all positions vs existing steering`, await run(program, { position_spans: [[0, 100000]] }), all, 0));
      checks.push(compare(`${name}: empty selection vs no injection`, await run(program, { position_spans: [] }), baseline, 0));
      checks.push(compare(`${name}: future selection vs no injection`, await run(program, { position_spans: [[10000, 10001]] }), baseline, 0));
      checks.push(compare(`${name}: cleared selection restores existing steering`, await run(program), all, 0));
      const effect = Math.max(...actual.rows.map((row, i) => Math.abs(row.logprob - matchedControl.rows[i].logprob)));
      if (effect < 1e-4) throw new Error(`${name}: nonzero steering had no measurable effect`);
      checks.push({ label: `${name}: measurable steering`, maxLogprobChange: effect, promptTokens: p, spans });
      console.info(`position spans: ${name} hardware checks passed`);
    }
    const chatInput = { kind: "chat", messages: [{ role: "user", content: prompt }] };
    const chat = await run(structured, { input: chatInput });
    const chatSpans = [[0, 1], [chat.promptTokens - 1, chat.promptTokens + 2]];
    const chatIndices = new Set([0, chat.promptTokens - 1, chat.promptTokens, chat.promptTokens + 1]);
    checks.push(compare("chat template/BOS positions vs independent control",
      await run(structured, { input: chatInput, position_spans: chatSpans }),
      await run(structured, { input: chatInput }, chatIndices, pipeline.prefillChunkSize), 0));
    pipeline.prefillChunkSize = 1;
    const probeProgram = compileStructuredHookProgram(hidden,
      Array.from({ length: layers }, (_, i) => i === layer ? {
        affineGroups: [{ active: true, basis: [direction], neutral: zeros, target: [2], along: 1, kappa: [0] }],
        probes: [{ kind: "linear" as const, direction: zeros, bias: 1 }],
      } : {}), resolveStructuredHookProfile(manifest.structuredHookProfile));
    probeProgram.measurementSchema = {
      layerMap: Array.from({ length: layers }, (_, i) => i), modelLayerCount: layers,
      probes: Array.from({ length: probeProgram.profile.maxProbes }, (_, i) =>
        i === 0 ? { name: "test.constant", family: "geometry" as const } : null),
      lensSource: null, saeSource: null,
    };
    const promptTokens = (await run(probeProgram)).promptTokens;
    const controlSpans = [[0, 3], [promptTokens - 1, promptTokens + 4]];
    for (const [label, phase, threshold, matches] of [
      ["prompt phase", "prompt_only", null, (p: number) => p < promptTokens],
      ["generated phase", "generated_only", null, (p: number) => p >= promptTokens],
      ["open probe gate", "both", 0.5, (p: number) => p >= promptTokens],
      ["closed probe gate", "both", 1.5, (_p: number) => false],
    ] as const) {
      const affine = Array(probeProgram.affineActive.length).fill(null);
      affine[layer * probeProgram.profile.maxAffineGroups] = { enabled: true, phase: { kind: phase },
        gate: threshold === null ? null : { slots: [{ layer, probe: 0 }], operator: ">", threshold } };
      const controlled = { ...probeProgram, controls: { affine, curve: Array(probeProgram.curveActive.length).fill(null) } };
      const expectedPositions = new Set(controlSpans.flatMap(([start, end]) =>
        Array.from({ length: end - start }, (_, i) => start + i).filter(matches)));
      checks.push(compare(`spans intersect ${label}`,
        await run(controlled, { position_spans: controlSpans }),
        await run(probeProgram, undefined, expectedPositions), 0));
    }
    }
    // Injection arithmetic on the compiled hook: a probe on the push direction at every layer
    // reads the residual after the hook, per position (a fresh prefill through each position).
    const captureTokens = await runtime.tokenizeText(prompt);
    const capturePositions = captureTokens.map((_: number, index: number) => index);
    const row = { inputIds: captureTokens, position: captureTokens.length - 1 };
    const isFp16 = manifest.quantization.includes("f16");
    const probeOn = (target: number) => compileStructuredHookProgram(hidden,
      Array.from({ length: layers }, (_, i) => ({
        probes: [{ kind: "linear" as const, direction, bias: 0 }],
        ...(i === layer
          ? { affineGroups: [{ active: true, basis: [direction], neutral: zeros, target: [target], along: 1, kappa: [0] }] }
          : {}),
      })), resolveStructuredHookProfile(manifest.structuredHookProfile));
    const unsteered = probeOn(2);
    const baseline = await runtime.measurePreparedPositions(row, capturePositions,
      { ...unsteered, affineActive: new Uint32Array(unsteered.affineActive.length) });
    const maxProbes = unsteered.profile.maxProbes;
    const reading = (values: Float32Array, p: number, l: number) => values[(p * layers + l) * maxProbes];
    // The residual's norm at the hook layer bounds the probe's rounding: the hook adds in fp32,
    // an fp16 library stores the sum in fp16 (2^-11 relative per component) before the probe
    // reads it, so the dot product on a unit direction moves by at most 2^-11·‖h + Δ‖.
    const plain = await runtime.capturePreparedPositions(row, capturePositions);
    const hookNorm = (p: number) => {
      let sum = 0;
      for (let d = 0; d < hidden; d++) sum += plain.values[(layer * capturePositions.length + p) * hidden + d] ** 2;
      return Math.sqrt(sum);
    };
    for (const sign of [1, -1]) {
      const steered = await runtime.measurePreparedPositions(row, capturePositions, probeOn(2 * sign));
      let preHook = 0;
      let injection = 0;
      let worstRatio = 0;
      for (let p = 0; p < capturePositions.length; p++) {
        for (let l = 0; l < layer; l++) {
          preHook = Math.max(preHook, Math.abs(reading(steered.values, p, l) - reading(baseline.values, p, l)));
        }
        const error = Math.abs(reading(steered.values, p, layer) - (reading(baseline.values, p, layer) + 2 * sign));
        const bound = (isFp16 ? 2 ** -11 : 2 ** -20) * 2 * (hookNorm(p) + 2) + 1e-6;
        injection = Math.max(injection, error);
        worstRatio = Math.max(worstRatio, error / bound);
      }
      if (preHook !== 0 || !(worstRatio <= 1)) {
        throw new Error(`Injection arithmetic mismatch: ${JSON.stringify({ sign, preHook, injection, worstRatio })}`);
      }
      checks.push({ label: `structured ${sign > 0 ? "positive" : "negative"} injection on the probed direction`,
        maxPreHookDifference: preHook, maxInjectionError: injection, worstErrorToBound: worstRatio,
        checkedLayers: layer + 1, positions: capturePositions.length,
        residualPrecision: isFp16 ? "fp32 intermediate, fp16 residual" : "fp32" });
    }
  } finally {
    pipeline.prefillChunkSize = originalChunkSize;
    pipeline.embedAndForward = originalForward;
    await engine.clearDrowseProgram();
  }
  return { passed: true, model: manifest.source, quantization: manifest.quantization, checks, batchingDiagnostics };
}
