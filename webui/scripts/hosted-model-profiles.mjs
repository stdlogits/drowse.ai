import { structuredHookProfileForArchitecture } from "./overlay-vm-functions.mjs";

// Hook ABI v5's two capacity profiles (HOOK_ABI.md §3): units on the dense gated architectures,
// residual elsewhere. Which one an architecture attests is the overlay's own rule.
const structuredHookProfiles = new Set(["standard-v5", "standard-v5-residual"]);

export const HOOK_ABI = "block-sites-v6";
export const RUNTIME_ABI = "drowse-web-runtime-v3";
export const EXACT_READOUT_ABI = "exact-readout-v2";

export function structuredHookProfile(value) {
  if (!structuredHookProfiles.has(value)) {
    throw new Error("structuredHookProfile is not allow-listed");
  }
  return value;
}

/** The profile a library of `architecture` must attest; throws when `value` is another. */
export function architectureStructuredHookProfile(architecture, value) {
  const expected = structuredHookProfileForArchitecture(architecture);
  if (structuredHookProfile(value) !== expected) {
    throw new Error(`${architecture} libraries attest ${expected}, not ${value}`);
  }
  return expected;
}

export function thinkingProfile(value) {
  if (value === null) return null;
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error("thinkingProfile must be null or an object");
  }
  const keys = Object.keys(value).sort();
  const expected = ["end", "endTokenIds", "start", "startTokenIds", "startsInThinking"];
  if (keys.join("\0") !== expected.join("\0")) {
    throw new Error("thinkingProfile has unexpected fields");
  }
  const start = delimiter(value.start, "thinkingProfile.start");
  const end = delimiter(value.end, "thinkingProfile.end");
  if (start === end) throw new Error("thinkingProfile delimiters must differ");
  if (typeof value.startsInThinking !== "boolean") {
    throw new Error("thinkingProfile.startsInThinking must be a boolean");
  }
  const startTokenIds = tokenIds(value.startTokenIds, "thinkingProfile.startTokenIds");
  const endTokenIds = tokenIds(value.endTokenIds, "thinkingProfile.endTokenIds");
  if (
    startTokenIds.length === endTokenIds.length &&
    startTokenIds.every((tokenId, index) => tokenId === endTokenIds[index])
  ) {
    throw new Error("thinkingProfile token sequences must differ");
  }
  return {
    start,
    end,
    startsInThinking: value.startsInThinking,
    startTokenIds,
    endTokenIds,
  };
}

export function sameThinkingProfile(left, right) {
  return JSON.stringify(thinkingProfile(left)) === JSON.stringify(thinkingProfile(right));
}

export function lockedModelExecutionProfiles(lock, manifest) {
  const lockedStructuredHookProfile = structuredHookProfile(lock?.structuredHookProfile);
  const artifactStructuredHookProfile = structuredHookProfile(manifest?.structuredHookProfile);
  const lockedThinkingProfile = thinkingProfile(lock?.thinkingProfile);
  const artifactThinkingProfile = thinkingProfile(manifest?.thinkingProfile);
  const quantization = lockedQuantization(lock?.quantization);
  const artifactQuantization = lockedQuantization(manifest?.quantization);
  if (
    artifactStructuredHookProfile !== lockedStructuredHookProfile ||
    JSON.stringify(artifactThinkingProfile) !== JSON.stringify(lockedThinkingProfile) ||
    artifactQuantization !== quantization
  ) {
    throw new Error("hosted model artifacts do not match runtime-lock execution profile");
  }
  const { blockSpec, unitRefusal } = lockedBlockSites(lock, lockedStructuredHookProfile);
  const prefillChunkSize = lock?.prefillChunkSize;
  if (
    !Number.isSafeInteger(prefillChunkSize) || prefillChunkSize < 1 ||
    prefillChunkSize > Math.min(...(lock?.contextProfiles ?? [0])) ||
    (manifest?.prefillChunkSize !== undefined && manifest.prefillChunkSize !== prefillChunkSize)
  ) {
    throw new Error("hosted model prefill chunk does not match the runtime lock");
  }
  return {
    structuredHookProfile: lockedStructuredHookProfile,
    thinkingProfile: lockedThinkingProfile,
    quantization,
    requiredFeatures: quantization === "q4f16_1" ? ["shader-f16"] : [],
    // What the runtime attests a v5 library's descriptor against (HOOK_ABI.md §6): the block
    // spec of a standard-v5 library, the refusal text of a standard-v5-residual one, and the
    // prefill chunk it was compiled with.
    blockSpec,
    unitRefusal,
    prefillChunkSize,
  };
}

function lockedBlockSites(lock, profile) {
  const blockSpec = lock?.blockSpec ?? null;
  const unitRefusal = lock?.unitRefusal ?? null;
  const units = profile === "standard-v5";
  const residual = profile === "standard-v5-residual";
  if (
    units !== (blockSpec !== null) || residual !== (unitRefusal !== null) ||
    (blockSpec !== null && (typeof blockSpec !== "object" || Array.isArray(blockSpec))) ||
    (unitRefusal !== null && (typeof unitRefusal !== "string" || unitRefusal.length === 0))
  ) {
    throw new Error(
      "runtime-lock blockSpec and unitRefusal do not match the structured hook profile",
    );
  }
  return { blockSpec, unitRefusal };
}

function lockedQuantization(value) {
  if (value !== "q4f16_1" && value !== "q4f32_1" && value !== "q0f32") {
    throw new Error("hosted quantization is not allow-listed");
  }
  return value;
}

function delimiter(value, label) {
  if (typeof value !== "string" || value.length === 0 || value.length > 256 || value.includes("\0")) {
    throw new Error(`${label} must be 1-256 characters without NULs`);
  }
  return value;
}

function tokenIds(value, label) {
  if (
    !Array.isArray(value) || value.length === 0 || value.length > 32 ||
    value.some((tokenId) => !Number.isSafeInteger(tokenId) || tokenId < 0)
  ) {
    throw new Error(`${label} must contain 1-32 non-negative token IDs`);
  }
  return [...value];
}
