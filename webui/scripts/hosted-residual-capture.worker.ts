import { contextBindingSha256, runtimeIdentitySha256 } from "../src/lib/runtime/catalog";
import type { BlockSpec } from "../src/hosted/runtime/structuredHookProfile";
import type { ModelVariant, RuntimeIdentity } from "../../drowse/webui/src/lib/runtime/contracts";
import {
  DrowseWebLlmRuntime,
  type DrowseWebLlmModule,
} from "../src/hosted/runtime/webLlmEngine";
import { compileStructuredHookProgram } from "../src/hosted/runtime/structuredHookProgram";
import { resolveStructuredHookProfile } from "../src/hosted/runtime/structuredHookProfile";
import { anatomyTapLayout } from "../src/hosted/runtime/modelBackend";
import { webLlmBlockSitesBackend } from "../src/hosted/runtime/webLlmBlockSites";
import { driveReplay } from "../src/hosted/runtime/units/nodeReplay";
import { sha256 } from "@noble/hashes/sha2.js";
import { bytesToHex } from "@noble/hashes/utils.js";

/** Capture schema 5 (browser-runtime/README.md, "production-q4 comparator"): neuron-write static
 *  reads, top 8 each, at most 64. */
const STATIC_READ_TOP = 8;
const MAXIMUM_STATIC_READS = 64;

interface CaptureRequest {
  type: "run";
  modelId: string;
  artifacts: Array<{ path: string; role: string; bytes: number; sha256: string }>;
  runtimeIdentity: RuntimeIdentity;
  structuredHookProfile: ModelVariant["structuredHookProfile"];
  /** The runtime lock's block spec (standard-v4) and unit refusal (standard-v4-residual). */
  blockSpec: BlockSpec | null;
  unitRefusal: string | null;
  prefillChunkSize: number;
  thinkingProfile: ModelVariant["thinkingProfile"];
  quantization: "q4f16_1" | "q4f32_1" | "q0f32";
  requiredFeatures: GPUFeatureName[];
  inputIds: number[];
  positions: number[];
  greedyPrompt: string;
  greedyMaxTokens: number;
  namedRoleCheck?: {
    row: {
      system: string;
      messages: Array<{
        role: "system" | "user" | "assistant";
        content: string;
        roleName?: string;
      }>;
    };
  };
  contextTokens: number;
  /** The lock's context profiles: the adapter attests the library's prefill chunk against them. */
  contextProfiles: number[];
  dumpShaders: boolean;
}

self.onmessage = async (event: MessageEvent<CaptureRequest>) => {
  if (event.data?.type !== "run") return;
  let runtime: DrowseWebLlmRuntime | null = null;
  try {
    const webLlmUrl = "/webllm.js";
    const webllm = await import(/* @vite-ignore */ webLlmUrl) as unknown as DrowseWebLlmModule;
    runtime = new DrowseWebLlmRuntime(webllm);
    const root = await navigator.storage.getDirectory();
    const directory = await root.getDirectoryHandle("drowse-residual-capture", { create: true });
    const artifacts = [];
    const revision = "a".repeat(40);
    for (const entry of event.data.artifacts) {
      const response = await fetch(
        `/model?path=${encodeURIComponent(entry.path)}`,
        { cache: "no-store" },
      );
      if (!response.ok) throw new Error(`artifact fetch failed for ${entry.path}`);
      const handle = await directory.getFileHandle(entry.path, { create: true });
      const writable = await handle.createWritable();
      await writable.write(await response.blob());
      await writable.close();
      artifacts.push({
        manifest: {
          ...entry,
          revision,
          url: `https://huggingface.co/polythetic/local/resolve/${revision}/${entry.path}`,
        },
        file: await handle.getFile(),
      });
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("no WebGPU adapter was returned");
    const shaderPosts = event.data.dumpShaders ? installShaderCapture() : [];
    const fingerprint = runtimeIdentitySha256(event.data.runtimeIdentity);
    await runtime.load({
      model: { id: event.data.modelId },
      variant: {
        id: `${event.data.modelId}-${event.data.quantization}`,
        structuredHookProfile: event.data.structuredHookProfile,
        thinkingProfile: event.data.thinkingProfile,
        runtimeIdentity: event.data.runtimeIdentity,
        runtimeIdentitySha256: fingerprint,
        contextProfiles: event.data.contextProfiles.map((contextTokens) => ({
          contextTokens,
          bindingSha256: contextBindingSha256(fingerprint, contextTokens),
        })),
        requirements: {
          features: event.data.requiredFeatures,
          limits: {
            maxStorageBufferBindingSize: 134217728,
            maxComputeWorkgroupSizeX: 256,
            maxComputeInvocationsPerWorkgroup: 256,
          },
        },
      },
      contextTokens: event.data.contextTokens,
      blockSpec: event.data.blockSpec,
      unitRefusal: event.data.unitRefusal,
      prefillChunkSize: event.data.prefillChunkSize,
      adapter,
      artifacts,
      optionalPacks: [],
      signal: new AbortController().signal,
      onDeviceLost() {},
    } as never);
    const capture = await runtime.capturePreparedPositions({
      inputIds: event.data.inputIds,
      position: event.data.positions[event.data.positions.length - 1],
    }, event.data.positions);
    // The steered capture on hook ABI v5: one structured program, probes on a fixed
    // orthonormal sketch at every layer (slot 0 the steering direction), with and without its
    // affine group, measured at every captured position.
    const steering = deterministicSteeringPrograms(
      capture.hiddenSize,
      capture.layerCount,
      event.data.structuredHookProfile,
    );
    const preparedRow = {
      inputIds: event.data.inputIds,
      position: event.data.positions[event.data.positions.length - 1],
    };
    const control = await runtime.measurePreparedPositions(
      preparedRow,
      event.data.positions,
      steering.controlProgram,
    );
    const steered = await runtime.measurePreparedPositions(
      preparedRow,
      event.data.positions,
      steering.program,
    );
    const greedyInputIds = await runtime.tokenizeText(event.data.greedyPrompt);
    const greedyTokenIds: number[] = [];
    await runtime.streamGeneration(
      {
        input: { kind: "raw", prompt: event.data.greedyPrompt },
        sampling: deterministicSampling(event.data.greedyMaxTokens),
        thinking: false,
        steeringExpression: null,
        hookProgram: null,
        onRawToken(token) {
          if (token.tokenId === null) {
            throw new Error("greedy generation returned a token without its vocabulary ID");
          }
          greedyTokenIds.push(token.tokenId);
        },
      },
      () => undefined,
    );
    const namedRolePrepared = event.data.namedRoleCheck === undefined
      ? undefined
      : (await runtime.prepareCaptureRows([event.data.namedRoleCheck.row]))[0];
    // B1 §4.3 item 2 on a units library: the block sites at every captured layer and position,
    // and neuron-write static reads; a residual library has neither.
    const units = event.data.structuredHookProfile === "standard-v4"
      ? await captureUnits(runtime, event.data.inputIds, event.data.positions, event.data.runtimeIdentity.layerMap)
      : null;
    const bytes = new Uint8Array(
      capture.values.buffer,
      capture.values.byteOffset,
      capture.values.byteLength,
    );
    const steeredBytes = new Uint8Array(
      steered.values.buffer,
      steered.values.byteOffset,
      steered.values.byteLength,
    );
    const controlBytes = new Uint8Array(
      control.values.buffer,
      control.values.byteOffset,
      control.values.byteLength,
    );
    const sitesBytes = units === null
      ? null
      : new Uint8Array(units.sites.buffer, units.sites.byteOffset, units.sites.byteLength);
    for (const [kind, body] of [
      ["unsteered", bytes],
      ["control-probes", controlBytes],
      ["steered-probes", steeredBytes],
      ...(sitesBytes === null ? [] : [["sites", sitesBytes] as const]),
    ] as const) {
      const response = await fetch(`/capture?kind=${kind}`, {
        method: "POST",
        body: body as Uint8Array<ArrayBuffer>,
      });
      if (!response.ok) throw new Error(await response.text());
    }
    await Promise.all(shaderPosts);
    await runtime.unload();
    runtime = null;
    self.postMessage({
      type: "done",
      result: {
        runtimeIdentitySha256: fingerprint,
        layerCount: capture.layerCount,
        positionCount: capture.positionCount,
        hiddenSize: capture.hiddenSize,
        positions: capture.positions,
        residualSha256: bytesToHex(sha256(bytes)),
        bytes: bytes.byteLength,
        probeCount: steered.probeCount,
        controlProbeSha256: bytesToHex(sha256(controlBytes)),
        steeredProbeSha256: bytesToHex(sha256(steeredBytes)),
        probeBytes: steeredBytes.byteLength,
        steeringProgram: steering.descriptor,
        greedyPrompt: event.data.greedyPrompt,
        greedyInputIds,
        greedyTokenIds,
        ...(namedRolePrepared === undefined
          ? {}
          : { namedRolePrepared }),
        ...(units === null || sitesBytes === null
          ? {}
          : {
              units: {
                block: units.block,
                sitesSha256: bytesToHex(sha256(sitesBytes)),
                sitesBytes: sitesBytes.byteLength,
                staticReads: units.staticReads,
              },
            }),
      },
    });
  } catch (error) {
    try { await runtime?.unload(); } catch {}
    self.postMessage({
      type: "error",
      message: error instanceof Error ? error.stack ?? error.message : String(error),
    });
  }
};

/** The units member's inputs (capture schema 5): `sites` fp32 `[layers, positions, d_ff + H·hd +
 *  d]`, each row `act | z | mid` from the block sites' anatomy taps of an unsteered replay of the
 *  input up to that position (causal, so equal to the full input's sites there); and one
 *  neuron-write static read per captured layer, `vocabReadoutTopK(unitWeightRows("neuron_write"))`. */
async function captureUnits(
  runtime: DrowseWebLlmRuntime,
  inputIds: readonly number[],
  positions: readonly number[],
  layerMap: readonly number[],
) {
  const capabilities = runtime.blockSitesCapabilities();
  if (capabilities === null || capabilities.blockSpec === null) {
    throw new Error("a standard-v4 capture needs the library's attested unit sites");
  }
  const backend = webLlmBlockSitesBackend(capabilities, runtime.blockSitesDevice());
  const spec = capabilities.blockSpec;
  const d = capabilities.hiddenSize;
  const zWidth = spec.nHeads * spec.headDim;
  const width = spec.dFF + zWidth + d;
  const layout = anatomyTapLayout(d, spec);
  const sites = new Float32Array(layerMap.length * positions.length * width);
  for (const [positionIndex, position] of positions.entries()) {
    const ids = Int32Array.from(inputIds.slice(0, position + 1));
    const replay = await driveReplay(
      backend,
      { program: null, anatomyPosition: position, activity: null, patternLayers: null },
      { ids, promptLength: ids.length, rawIds: new Int32Array(0), spans: [] },
    );
    const taps = replay.anatomyTaps;
    if (taps === null || taps.shape[0] !== capabilities.layerCount || taps.shape[1] !== layout.stride) {
      throw new Error(`the replay at position ${position} returned no anatomy taps`);
    }
    for (const [layerIndex, layer] of layerMap.entries()) {
      const row = taps.data.subarray(layer * layout.stride, (layer + 1) * layout.stride);
      const out = sites.subarray(
        (layerIndex * positions.length + positionIndex) * width,
        (layerIndex * positions.length + positionIndex + 1) * width,
      );
      out.set(row.subarray(layout.act, layout.act + spec.dFF), 0);
      out.set(row.subarray(layout.z, layout.z + zWidth), spec.dFF);
      out.set(row.subarray(layout.mid, layout.mid + d), spec.dFF + zWidth);
    }
  }
  if (sites.some((value) => !Number.isFinite(value))) {
    throw new Error("the replayed block sites are not finite");
  }
  if (layerMap.length > MAXIMUM_STATIC_READS) {
    throw new Error(`${layerMap.length} captured layers exceed ${MAXIMUM_STATIC_READS} static reads`);
  }
  // A fixed spread of neurons, one per captured layer.
  const units = layerMap.map((layer) => ({
    kind: "neuron" as const,
    layer,
    index: (layer * 7919 + 13) % spec.dFF,
  }));
  const staticReads: Array<{ layer: number; neuron: number; ids: number[]; probabilities: number[] }> = [];
  for (let start = 0; start < units.length; start += 8) {
    const batch = units.slice(start, start + 8);
    const rows = await backend.unitWeightRows("neuron_write", batch);
    const top = await backend.vocabReadoutTopK(rows, STATIC_READ_TOP);
    for (const [index, unit] of batch.entries()) {
      const ids = [...top.ids.data.subarray(index * STATIC_READ_TOP, (index + 1) * STATIC_READ_TOP)];
      const logSumExp = top.logSumExp.data[index];
      const probabilities = [...top.logits.data.subarray(index * STATIC_READ_TOP, (index + 1) * STATIC_READ_TOP)]
        .map((logit) => Math.exp(logit - logSumExp));
      staticReads.push({ layer: unit.layer, neuron: unit.index, ids, probabilities });
    }
  }
  return {
    block: { d_ff: spec.dFF, n_heads: spec.nHeads, head_dim: spec.headDim },
    sites,
    staticReads,
  };
}

/** The q4 gate's steering: a rank-1 affine group at the middle layer (`h += along·(target −
 *  kappa·(h·u))·u`, neutral 0), and linear probes on an orthonormal sketch `D` (`D[0] = u`, the
 *  rest a fixed deterministic set, Gram-Schmidt) at every layer, so each probe reads `h·D[k]` of
 *  the residual after the hook. The control is the same program with the group inactive. */
function deterministicSteeringPrograms(
  hiddenSize: number,
  layerCount: number,
  profileId: CaptureRequest["structuredHookProfile"],
) {
  const profile = resolveStructuredHookProfile(profileId);
  const enabledLayer = Math.floor(layerCount / 2);
  const component = 1 / Math.sqrt(hiddenSize);
  const directions: Float64Array[] = [
    Float64Array.from({ length: hiddenSize }, (_, index) => index % 2 === 0 ? component : -component),
  ];
  for (let row = 1; directions.length < Math.min(profile.maxProbes, hiddenSize); row += 1) {
    const candidate = Float64Array.from({ length: hiddenSize }, (_, index) => Math.sin((row + 1) * (index + 1)));
    for (const basis of directions) {
      let projection = 0;
      for (let index = 0; index < hiddenSize; index += 1) projection += candidate[index] * basis[index];
      for (let index = 0; index < hiddenSize; index += 1) candidate[index] -= projection * basis[index];
    }
    const norm = Math.hypot(...candidate);
    if (norm < 1e-6) continue;
    directions.push(candidate.map((value) => value / norm));
  }
  const probeDirections = directions.map((row) => Float32Array.from(row));
  const target = Math.fround(1.25);
  const along = Math.fround(0.4);
  const kappa = Math.fround(0.3);
  const zero = new Float32Array(hiddenSize);
  const probes = probeDirections.map((direction) => ({ kind: "linear" as const, direction, bias: 0 }));
  const program = compileStructuredHookProgram(
    hiddenSize,
    Array.from({ length: layerCount }, (_, layer) => ({
      probes,
      ...(layer === enabledLayer
        ? {
            affineGroups: [{
              active: true,
              basis: [probeDirections[0]],
              neutral: zero,
              target: [target],
              along,
              kappa: [kappa],
            }],
          }
        : {}),
    })),
    profile,
  );
  return {
    program,
    controlProgram: { ...program, affineActive: new Uint32Array(program.affineActive.length) },
    descriptor: {
      format: program.format,
      enabledLayer,
      probeDirections: probeDirections.map((direction) => [...direction]),
      target,
      along,
      kappa,
    },
  };
}

function deterministicSampling(maxTokens: number) {
  return {
    max_tokens: maxTokens,
    seed: 1,
    temperature: 0,
    top_p: 1,
    top_k: 0,
    presence_penalty: 0,
    frequency_penalty: 0,
  };
}

function installShaderCapture(): Promise<void>[] {
  const posts: Promise<void>[] = [];
  const sources = new WeakMap<GPUShaderModule, string>();
  const prototype = GPUDevice.prototype;
  const createShaderModule = prototype.createShaderModule;
  const createComputePipeline = prototype.createComputePipeline;
  const createComputePipelineAsync = prototype.createComputePipelineAsync;
  Object.defineProperty(prototype, "createShaderModule", {
    configurable: true,
    value(this: GPUDevice, descriptor: GPUShaderModuleDescriptor) {
      const module = createShaderModule.call(this, descriptor);
      sources.set(module, descriptor.code);
      return module;
    },
  });
  const record = (descriptor: GPUComputePipelineDescriptor) => {
    const code = sources.get(descriptor.compute.module);
    if (!code) return;
    const entryPoint = descriptor.compute.entryPoint ?? "main";
    posts.push(fetch(`/shader?entry=${encodeURIComponent(entryPoint)}`, {
      method: "POST",
      body: code,
    }).then((response) => {
      if (!response.ok) throw new Error(`shader capture failed for ${entryPoint}`);
    }));
  };
  Object.defineProperty(prototype, "createComputePipeline", {
    configurable: true,
    value(this: GPUDevice, descriptor: GPUComputePipelineDescriptor) {
      record(descriptor);
      return createComputePipeline.call(this, descriptor);
    },
  });
  Object.defineProperty(prototype, "createComputePipelineAsync", {
    configurable: true,
    value(this: GPUDevice, descriptor: GPUComputePipelineDescriptor) {
      record(descriptor);
      return createComputePipelineAsync.call(this, descriptor);
    },
  });
  return posts;
}
