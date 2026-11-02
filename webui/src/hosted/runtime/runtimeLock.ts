import runtimeLock from "../../../../browser-runtime/runtime-lock.json";
import type {
  ModelThinkingProfile,
  StructuredHookProfileId,
} from "../../../../drowse/webui/src/lib/runtime/contracts";
import { validateBlockSpec, type BlockSpec } from "./structuredHookProfile";

export type BrowserRuntimeReleaseStatus = "feasibility-required" | "verified";

export const BROWSER_RUNTIME_LOCK = Object.freeze({
  status: runtimeLock.status as BrowserRuntimeReleaseStatus,
  runtimeAbi: runtimeLock.runtimeAbi,
  hookAbi: runtimeLock.hookAbi,
  models: runtimeLock.models.map((model) => Object.freeze({
    id: model.id,
    modelType: "modelType" in model && model.modelType === "base" ? "base" as const : "chat" as const,
    sourceRepository: model.sourceRepository,
    sourceRevision: model.sourceRevision,
    convertedRepository: model.convertedRepository,
    convertedRevision: model.convertedRevision,
    manifestSha256: model.manifestSha256,
    librarySha256: model.librarySha256,
    tokenizerSha256: model.tokenizerSha256,
    chatTemplateSha256: model.chatTemplateSha256,
    hiddenSize: model.hiddenSize,
    layerMap: model.layerMap,
    quantization: model.quantization,
    contextProfiles: model.contextProfiles,
    structuredHookProfile: model.structuredHookProfile as StructuredHookProfileId,
    thinkingProfile: model.thinkingProfile as ModelThinkingProfile | null,
    // A v5 entry records the block spec its library attests (standard-v4) or its unit members'
    // refusal (standard-v4-residual).
    blockSpec: "blockSpec" in model && model.blockSpec != null
      ? validateBlockSpec(model.blockSpec) as BlockSpec
      : null,
    unitRefusal: "unitRefusal" in model && typeof model.unitRefusal === "string" ? model.unitRefusal : null,
  })),
});

export const BROWSER_RUNTIME_RELEASE_VERIFIED =
  BROWSER_RUNTIME_LOCK.status === "verified";
