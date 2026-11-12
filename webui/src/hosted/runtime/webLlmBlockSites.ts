// Hook ABI v6's device surface over the WebLLM bridge (B1 §2-§4, browser-runtime/HOOK_ABI.md
// §7-§9): the units programs, the residents (head grams, unit scales, lens layers, embedding
// statistics, pattern captures), the replay, the attention rows and the static reads
// (`webLlmStaticReads.ts`). The bridge takes one request per call (`drowseBlockSites`), so the
// worker engine carries the whole surface as one message kind.
import type { DrowseBlockSitesRequest } from "@drowse/web-llm";
import {
  activityStateLayout,
  anatomyTapLayout,
  BLOCK_SITES_BACKEND_MEMBERS,
  BLOCK_SITES_UNIT_MEMBERS,
  encodeReplayMeta,
  f32Tensor,
  requireTensorShape,
  unitScalesLayout,
  UnsupportedBlockError,
  type BlockSitesCapabilities,
  type BlockSitesReplayRequest,
  type BlockSitesReplayResult,
  type BlockSitesReplaySegment,
  type BlockSitesResident,
  type BlockSitesResidentKind,
  type BrowserBlockSitesBackend,
  type BrowserBlockSitesReplay,
  type F32Tensor,
} from "./modelBackend";
import {
  assertStructuredTapGramsBound,
  compileStructuredHookProgram,
  structuredUnitSectionLayout,
  type StructuredHookProgramBuffers,
} from "./structuredHookProgram";
import { webLlmStaticReads } from "./webLlmStaticReads";

/** The bridge's block-sites requests. */
export type WebLlmBlockSitesRequest = DrowseBlockSitesRequest;

interface BridgeResident {
  id: string;
  kind: BlockSitesResidentKind;
  shape: number[];
  bytes: number;
}

interface BridgeReplayForwardReadback {
  measurements: Float32Array;
  unitTaps: Float32Array;
  geometry: Float32Array | null;
  hidden: Float32Array | null;
}

interface BridgeReplayCaptures {
  anatomyTaps: Float32Array | null;
  activityState: Float32Array | null;
  traces: Float32Array | null;
  population: Float32Array | null;
  patterns: BridgeResident | null;
}

/** The loaded model's bridge, as the adapter drives it. */
export interface WebLlmBlockSitesDevice {
  call(request: WebLlmBlockSitesRequest): Promise<unknown>;
  /** Install a drowse-structured-v5 program with a unit section (the replay's steering). */
  installProgram(program: StructuredHookProgramBuffers): Promise<void>;
}

/** Members with no bridge behind them (no v5 bridge loaded) reject with this code; they are not
 *  refusals of the model. */
export const BLOCK_SITES_MEMBER_UNAVAILABLE = "BLOCK_SITES_MEMBER_UNAVAILABLE";

export function blockSitesMemberUnavailable(member: string): Error {
  return Object.assign(
    new Error(`The browser runtime does not provide ${member} for hook ABI block-sites-v6 yet`),
    { code: BLOCK_SITES_MEMBER_UNAVAILABLE },
  );
}

function resident<K extends BlockSitesResidentKind>(value: unknown, kind: K): BlockSitesResident<K> {
  const handle = value as BridgeResident | null;
  if (
    handle === null || typeof handle !== "object" || handle.kind !== kind ||
    typeof handle.id !== "string" || !Array.isArray(handle.shape) ||
    !Number.isSafeInteger(handle.bytes)
  ) {
    throw new TypeError(`The bridge returned no ${kind} resident`);
  }
  return Object.freeze({ kind, id: handle.id, bytes: handle.bytes, shape: Object.freeze([...handle.shape]) });
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) throw signal.reason ?? new DOMException("Aborted", "AbortError");
}

/** The WebLLM bridge's hook ABI v6 surface (B1 §2.2-§2.3, §3.4, §3.5, §6). A residual library
 *  answers every unit member with its lock refusal (422, as native); a units library binds
 *  drowse-structured-v5 programs, replays nodes and keeps the residents the reads need. */
export function webLlmBlockSitesBackend(
  capabilities: BlockSitesCapabilities,
  device: WebLlmBlockSitesDevice | null = null,
): BrowserBlockSitesBackend {
  const units = capabilities.profile.unitSites === 1;
  const refuse = (): Promise<never> =>
    Promise.reject(new UnsupportedBlockError(capabilities.unitRefusal ?? "unit sites unavailable"));
  const unitMembers = new Set<string>([...BLOCK_SITES_UNIT_MEMBERS, "unembedCotangent"]);
  const backend: Record<string, unknown> = {
    capabilities: () => capabilities,
  };
  for (const member of BLOCK_SITES_BACKEND_MEMBERS) {
    backend[member] = units || !unitMembers.has(member)
      ? () => Promise.reject(blockSitesMemberUnavailable(member))
      : refuse;
  }
  let headGrams: Promise<BlockSitesResident<"head_grams">> | null = null;
  const residentHeadGrams = (): Promise<BlockSitesResident<"head_grams">> => {
    headGrams ??= device!.call({ op: "headGrams" }).then((value) => resident(value, "head_grams"));
    headGrams.catch(() => { headGrams = null; });
    return headGrams;
  };
  const reads = device === null
    ? {}
    : webLlmStaticReads(capabilities, device, async () => (await residentHeadGrams()).id);
  Object.assign(backend, reads);
  backend.bindStructuredProgram = async (
    program: StructuredHookProgramBuffers,
  ): Promise<StructuredHookProgramBuffers> => {
    if (program.unitMeta === undefined) return program;
    if (!units) return refuse();
    const spec = capabilities.blockSpec!;
    const layout = structuredUnitSectionLayout(
      capabilities.profile,
      capabilities.layerCount,
      spec.headDim,
    );
    if (
      program.unitMeta.length !== layout.metaLength ||
      program.unitCoeff?.length !== layout.coeffLength ||
      program.unitActive?.length !== layout.activeLength ||
      program.tapGrams?.length !== layout.tapGramsLength
    ) {
      throw new TypeError("The program's unit section does not match the loaded model's layout");
    }
    // Head taps read `tapGrams[i] = G_h` (B1 §2.2), gathered from the resident head grams.
    const offsets = layout.meta.tapHeadOffsets.offset;
    const index = layout.meta.tapHeadIndex.offset;
    const count = program.unitMeta[offsets + capabilities.layerCount];
    if (count === 0) return program;
    if (device === null) throw blockSitesMemberUnavailable("the head write grams of a tapped head");
    const heads: { layer: number; head: number }[] = [];
    for (let layer = 0; layer < capabilities.layerCount; layer += 1) {
      for (let entry = program.unitMeta[offsets + layer]; entry < program.unitMeta[offsets + layer + 1]; entry += 1) {
        heads.push({ layer, head: program.unitMeta[index + entry] });
      }
    }
    const grams = await (backend.headWriteGrams as BrowserBlockSitesBackend["headWriteGrams"])(heads);
    const tapGrams = new Float32Array(layout.tapGramsLength);
    tapGrams.set(grams.data);
    return { ...program, tapGrams };
  };
  if (device === null) return Object.freeze(backend) as unknown as BrowserBlockSitesBackend;
  const captures = new Map<string, { layers: readonly number[]; position: number }>();
  backend.release = async (handle: BlockSitesResident) => {
    if (handle.kind === "head_grams") {
      throw new TypeError("Head grams are dropped with the model, not released");
    }
    await device.call({ op: "release", id: handle.id });
    captures.delete(handle.id);
  };
  if (!units) {
    backend.unembedCotangent = refuse;
    return Object.freeze(backend) as unknown as BrowserBlockSitesBackend;
  }

  const spec = capabilities.blockSpec!;
  const L = capabilities.layerCount;
  let open = false;

  backend.residentHeadGrams = async (signal?: AbortSignal) => {
    throwIfAborted(signal);
    return residentHeadGrams();
  };
  backend.uploadUnitScales = async (scales: F32Tensor) => {
    requireTensorShape(scales, "f32", [L, unitScalesLayout(spec).stride], "Unit scales");
    return resident(await device.call({ op: "uploadUnitScales", values: scales.data }), "unit_scales");
  };
  backend.openReplay = async (
    request: BlockSitesReplayRequest,
    signal?: AbortSignal,
  ): Promise<BrowserBlockSitesReplay> => {
    throwIfAborted(signal);
    if (open) throw Object.assign(new Error("A replay is already open"), { code: "REPLAY_BUSY" });
    const meta = encodeReplayMeta(capabilities.profile, {
      layerCount: L,
      contextTokens: capabilities.contextTokens,
      blockSpec: spec,
    }, request);
    if (request.patternLayers !== null && request.patternLayers.length > 0 && request.anatomyPosition === null) {
      throw new RangeError("Pattern capture reads queries at the anatomy position");
    }
    const tokenCount = request.tokenCount ?? capabilities.contextTokens;
    if (!Number.isSafeInteger(tokenCount) || tokenCount < 1 || tokenCount > capabilities.contextTokens) {
      throw new RangeError("A replay's token count must fit the compiled context");
    }
    const installed = replayProgram(capabilities, request.program);
    await device.installProgram(installed);
    const activity = request.activity;
    await device.call({
      op: "openReplay",
      replayMeta: meta,
      unitScales: activity?.scales.id ?? null,
      rowCount: activity?.rowCount ?? 1,
      tokenCount,
    });
    open = true;
    const anatomyAt = request.anatomyPosition;
    const patternLayers = request.patternLayers ?? [];
    const close = () => { open = false; };
    const replay: BrowserBlockSitesReplay = {
      async forward(segment: BlockSitesReplaySegment) {
        if (!(segment.tokenIds instanceof Int32Array) || segment.tokenIds.length === 0) {
          throw new TypeError("A replay segment's tokens must be a non-empty Int32Array");
        }
        const readback = await device.call({
          op: "replayForward",
          tokenIds: Array.from(segment.tokenIds),
          // An unsteered replay's segments carry no controls: nothing of its program fires.
          affineActive: segment.controls.affineActive.length === 0
            ? new Uint32Array(installed.affineActive.length)
            : segment.controls.affineActive,
          curveActive: segment.controls.curveActive.length === 0
            ? new Uint32Array(installed.curveActive.length)
            : segment.controls.curveActive,
          unitActive: segment.controls.unitActive ?? new Float32Array(capabilities.profile.maxUnitGroups),
        }) as BridgeReplayForwardReadback;
        const profile = capabilities.profile;
        return {
          measurements: f32Tensor(readback.measurements, [L, profile.maxProbes]),
          unitTaps: f32Tensor(readback.unitTaps, [profile.maxUnitTaps]),
          geometry: readback.geometry === null
            ? null
            : f32Tensor(readback.geometry, [L, profile.maxGeometryProbes, profile.geometryOutputStride]),
          hidden: readback.hidden === null ? null : f32Tensor(readback.hidden, [L, capabilities.hiddenSize]),
        };
      },
      async finish(): Promise<BlockSitesReplayResult> {
        const out = await device.call({ op: "finishReplay" }) as BridgeReplayCaptures;
        close();
        let patterns: BlockSitesResident<"pattern_capture"> | null = null;
        if (out.patterns !== null) {
          patterns = resident(out.patterns, "pattern_capture");
          captures.set(patterns.id, { layers: [...patternLayers], position: anatomyAt! });
        }
        const state = activityStateLayout(spec);
        return {
          anatomyTaps: out.anatomyTaps === null
            ? null
            : f32Tensor(out.anatomyTaps, [L, anatomyTapLayout(capabilities.hiddenSize, spec).stride]),
          activity: activity === null || out.activityState === null
            ? null
            : {
                state: f32Tensor(out.activityState, [L, state.stride]),
                traces: f32Tensor(
                  out.traces!.slice(0, activity.traces.length * activity.rowCount),
                  [activity.traces.length, activity.rowCount],
                ),
                population: activity.populationRow === null
                  ? null
                  : f32Tensor(out.population!, [L, spec.dFF + spec.nHeads]),
              },
          patterns,
        };
      },
      async cancel() {
        await device.call({ op: "cancelReplay" });
        close();
      },
    };
    return replay;
  };

  backend.attentionRows = async (
    capture: BlockSitesResident<"pattern_capture">,
    heads: readonly { layer: number; head: number }[],
  ): Promise<F32Tensor> => {
    const held = captures.get(capture.id);
    if (held === undefined) throw new TypeError("Unknown pattern capture");
    const most = capabilities.profile.maxPatternHeads;
    if (heads.length < 1 || heads.length > most) {
      throw new RangeError(`Attention rows take 1 to ${most} heads`);
    }
    const rows = new Int32Array(4 * most).fill(-1);
    heads.forEach(({ layer, head }, at) => {
      const slot = held.layers.indexOf(layer);
      if (slot < 0 || !Number.isSafeInteger(head) || head < 0 || head >= spec.nHeads) {
        throw new RangeError(`Head L${layer}h${head} is not in the pattern capture`);
      }
      rows.set([slot, layer, head, held.position], 4 * at);
    });
    const out = await device.call({ op: "attentionRows", capture: capture.id, rows }) as Float32Array;
    const width = out.length / most;
    const p = held.position + 1;
    const data = new Float32Array(heads.length * p);
    for (let at = 0; at < heads.length; at += 1) data.set(out.subarray(at * width, at * width + p), at * p);
    return f32Tensor(data, [heads.length, p]);
  };
  return Object.freeze(backend) as unknown as BrowserBlockSitesBackend;
}

/** The replay's program: the node's own (with an empty unit section when it has none, so the
 *  bridge routes through the units superset) or, for an unsteered replay, an empty one. */
export function replayProgram(
  capabilities: BlockSitesCapabilities,
  program: StructuredHookProgramBuffers | null,
): StructuredHookProgramBuffers {
  const base = program ?? compileStructuredHookProgram(
    capabilities.hiddenSize,
    Array.from({ length: capabilities.layerCount }, () => ({})),
    capabilities.profile,
  );
  if (base.unitMeta !== undefined) {
    assertStructuredTapGramsBound(base, {
      profile: capabilities.profile,
      layerCount: capabilities.layerCount,
      headDim: capabilities.blockSpec!.headDim,
    });
    return base;
  }
  const layout = structuredUnitSectionLayout(
    capabilities.profile,
    capabilities.layerCount,
    capabilities.blockSpec!.headDim,
  );
  const spec = capabilities.blockSpec!;
  return {
    ...base,
    unitShape: Object.freeze({ dFF: spec.dFF, nHeads: spec.nHeads, headDim: spec.headDim }),
    unitMeta: new Uint32Array(layout.metaLength),
    unitCoeff: new Float32Array(layout.coeffLength),
    unitActive: new Float32Array(layout.activeLength),
    tapGrams: new Float32Array(layout.tapGramsLength),
  };
}
