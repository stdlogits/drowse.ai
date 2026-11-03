import type {
  CatalogFile,
  CatalogInstrumentPack,
  ModelVariant,
  RuntimeCapabilities,
} from "../../../../drowse/webui/src/lib/runtime/contracts";
import { MEBIBYTE } from "../../../../drowse/webui/src/lib/runtime/contracts";

const FP32_BYTES = 4;
const EXACT_READOUT_TOP_K = 8;
const MAX_RUNTIME_CHUNK_BYTES = 32 * MEBIBYTE;

export const OPTIONAL_PACK_RESIDENT_BUDGET_BYTES = 1024 * MEBIBYTE;

// ---------------------------------------------------------------- pack residency ---
//
// Lens and units packs are full depth (PLAN.md): every layer of R and S at `out` and `mid`, the
// units reads and contrast statistics per layer, each layer its own file. Installing a pack
// downloads its manifests and model-wide statistics plus the live readout's R:out layers, so the
// readout runs offline after an install as it always has; every other layer file is fetched,
// verified against the signed catalog closure, and kept the first time a read needs it.

/** The live readout's R:out layers: at most eight, evenly spaced over the blocks before the
 *  last, within a 448 MiB fp32 budget (the hosted readout's rule since the first J-lens packs). */
export const LENS_READOUT_MAX_LAYERS = 8;
export const LENS_READOUT_RESIDENT_BYTES = 448 * MEBIBYTE;

export function lensReadoutLayers(layerMap: readonly number[], hiddenSize: number): number[] {
  if (
    layerMap.length < 2 || !Number.isSafeInteger(hiddenSize) || hiddenSize < 1 ||
    layerMap.some((layer, index) =>
      !Number.isSafeInteger(layer) || layer < 0 || (index > 0 && layer <= layerMap[index - 1]))
  ) {
    throw new RangeError("The lens readout needs an increasing layer map with a block before the last");
  }
  const sources = layerMap.slice(0, -1);
  const count = Math.max(1, Math.min(
    LENS_READOUT_MAX_LAYERS,
    sources.length,
    Math.floor(LENS_READOUT_RESIDENT_BYTES / (hiddenSize * hiddenSize * FP32_BYTES)),
  ));
  if (count === sources.length) return [...sources];
  if (count === 1) return [sources[sources.length - 1]];
  return Array.from({ length: count }, (_, index) =>
    sources[Math.round(index * (sources.length - 1) / (count - 1))]);
}

export type PackLayerShard =
  | { readonly kind: "lens"; readonly site: "out" | "mid"; readonly layer: number }
  | { readonly kind: "units"; readonly part: "reads" | "contrast"; readonly layer: number };

const LENS_SHARD = /^(out|mid)\.layer-(0|[1-9]\d{0,4})\.safetensors$/u;
const UNITS_SHARD = /^(reads|contrast)\.layer-(0|[1-9]\d{0,4})\.safetensors$/u;

/** The per-layer shard a pack file is, by its path: a lens pack's `<kind>/<site>.layer-<L>` or a
 *  units pack's `units/<part>.layer-<L>` (`null` for every other file and kind). */
export function packLayerShard(
  pack: Pick<CatalogInstrumentPack, "kind">,
  path: string,
): PackLayerShard | null {
  if (typeof path !== "string") return null;
  const parts = path.split("/");
  const name = parts.at(-1) ?? "";
  if (parts.length < 2 || parts.at(-2) !== pack.kind) return null;
  if (pack.kind === "jlens" || pack.kind === "slens") {
    const match = LENS_SHARD.exec(name);
    return match === null ? null : { kind: "lens", site: match[1] as "out" | "mid", layer: Number(match[2]) };
  }
  if (pack.kind === "units") {
    const match = UNITS_SHARD.exec(name);
    return match === null
      ? null
      : { kind: "units", part: match[1] as "reads" | "contrast", layer: Number(match[2]) };
  }
  return null;
}

/** `install` files download with the pack; `on_demand` files on the first read that needs them. */
export function packFileResidency(
  pack: Pick<CatalogInstrumentPack, "kind">,
  variant: Pick<ModelVariant, "runtimeIdentity">,
  path: string,
): "install" | "on_demand" {
  const shard = packLayerShard(pack, path);
  if (shard === null) return "install";
  if (pack.kind === "jlens" && shard.kind === "lens" && shard.site === "out") {
    const { layerMap, hiddenSize } = variant.runtimeIdentity;
    return lensReadoutLayers(layerMap, hiddenSize).includes(shard.layer) ? "install" : "on_demand";
  }
  return "on_demand";
}

export function packInstallFiles(
  pack: Pick<CatalogInstrumentPack, "kind" | "files">,
  variant: Pick<ModelVariant, "runtimeIdentity">,
): CatalogFile[] {
  return pack.files.filter((file) => packFileResidency(pack, variant, file.path) === "install");
}

/** Whether an install record holds a pack: every install file, and nothing outside the closure
 *  (fetched layer files join the record as reads need them). */
export function packInstallCovers(
  pack: Pick<CatalogInstrumentPack, "kind" | "files">,
  variant: Pick<ModelVariant, "runtimeIdentity">,
  objectHashes: readonly string[],
): boolean {
  const held = new Set(objectHashes);
  const closure = new Set(pack.files.map((file) => file.sha256));
  return packInstallFiles(pack, variant).every((file) => held.has(file.sha256)) &&
    [...held].every((hash) => closure.has(hash));
}

export interface OptionalPackHardwareBlock {
  code:
    | "JLENS_DEVICE_BUFFER_LIMIT"
    | "SAE_DEVICE_BUFFER_LIMIT"
    | "OPTIONAL_PACK_RESIDENT_BUDGET_EXCEEDED";
  message: string;
  requiredBytes: number;
  limitBytes: number;
  packIds: string[];
}

export function optionalPackHardwareBlock(
  pack: CatalogInstrumentPack,
  variant: ModelVariant,
  capabilities: RuntimeCapabilities,
  concurrentlySelected: readonly CatalogInstrumentPack[] = [],
): OptionalPackHardwareBlock | null {
  const available = uniqueOptionalPacks([...concurrentlySelected, pack]);
  for (const candidate of available) {
    const block = perBufferBlock(candidate, variant, capabilities);
    if (block !== null) return block;
  }
  const selected = simultaneouslyResidentPacks(available, variant);

  const requiredBytes = selected.reduce(
    (total, candidate) => total + conservativeResidentBytes(candidate, variant),
    0,
  );
  if (
    Number.isSafeInteger(requiredBytes) && requiredBytes > 0 &&
    requiredBytes <= OPTIONAL_PACK_RESIDENT_BUDGET_BYTES
  ) return null;

  const packIds = selected.map((candidate) => candidate.id);
  return {
    code: "OPTIONAL_PACK_RESIDENT_BUDGET_EXCEEDED",
    message:
      "These model tools need more graphics memory than Drowse can safely reserve at once. Remove one installed tool, or choose a smaller model or tool.",
    requiredBytes,
    limitBytes: OPTIONAL_PACK_RESIDENT_BUDGET_BYTES,
    packIds,
  };
}

function perBufferBlock(
  pack: CatalogInstrumentPack,
  variant: ModelVariant,
  capabilities: RuntimeCapabilities,
): OptionalPackHardwareBlock | null {
  if (pack.kind !== "jlens" && pack.kind !== "slens" && pack.kind !== "sae") return null;
  const hiddenSize = variant.runtimeIdentity.hiddenSize;
  const requiredBytes = pack.kind === "jlens" || pack.kind === "slens"
    ? hiddenSize * hiddenSize * FP32_BYTES
    : Math.max(
        hiddenSize * EXACT_READOUT_TOP_K * FP32_BYTES,
        Math.min(largestFileBytes(pack), MAX_RUNTIME_CHUNK_BYTES),
      );
  const limitBytes = Math.min(adapterBufferLimit(capabilities), MAX_RUNTIME_CHUNK_BYTES);
  if (
    Number.isSafeInteger(requiredBytes) &&
    requiredBytes > 0 &&
    limitBytes > 0 &&
    limitBytes >= requiredBytes
  ) return null;
  return {
    code: pack.kind === "sae" ? "SAE_DEVICE_BUFFER_LIMIT" : "JLENS_DEVICE_BUFFER_LIMIT",
    message:
      `${pack.displayName} needs a larger graphics buffer than this browser can provide. Use the model without this tool, or choose a smaller compatible model or tool.`,
    requiredBytes,
    limitBytes,
    packIds: [pack.id],
  };
}

function largestFileBytes(pack: CatalogInstrumentPack): number {
  return pack.files.reduce((largest, file) => Math.max(largest, file.bytes), 0);
}

function adapterBufferLimit(capabilities: RuntimeCapabilities): number {
  const limits = [
    capabilities.webGpu.limits.maxBufferSize,
    capabilities.webGpu.limits.maxStorageBufferBindingSize,
  ];
  if (limits.some((value) => !Number.isSafeInteger(value) || value <= 0)) return 0;
  return Math.min(...limits);
}

function conservativeResidentBytes(
  pack: CatalogInstrumentPack,
  variant: ModelVariant,
): number {
  if (pack.kind === "jlens" || pack.kind === "slens") {
    // What stays resident: the install set (the readout's R:out layers, the S statistics) and
    // one fetched layer; fetched layers beyond it are evicted as reads move on.
    const oneMatrixBytes =
      variant.runtimeIdentity.hiddenSize * variant.runtimeIdentity.hiddenSize * FP32_BYTES;
    return installBytes(pack, variant) + oneMatrixBytes;
  }
  if (pack.kind === "units") {
    const largestLayer = pack.files
      .filter((file) => packLayerShard(pack, file.path) !== null)
      .reduce((largest, file) => Math.max(largest, file.bytes), 0);
    return installBytes(pack, variant) + largestLayer;
  }
  if (pack.kind === "sae") {
    const minimumEncoderChunkBytes =
      variant.runtimeIdentity.hiddenSize * EXACT_READOUT_TOP_K * FP32_BYTES;
    return Math.max(pack.bytes, minimumEncoderChunkBytes);
  }
  return 0;
}

function installBytes(pack: CatalogInstrumentPack, variant: ModelVariant): number {
  return packInstallFiles(pack, variant).reduce((total, file) => total + file.bytes, 0);
}

function uniqueOptionalPacks(
  packs: readonly CatalogInstrumentPack[],
): CatalogInstrumentPack[] {
  const selected = new Map<string, CatalogInstrumentPack>();
  for (const pack of packs) {
    if (pack.kind !== "core") selected.set(pack.id, pack);
  }
  return [...selected.values()];
}

function simultaneouslyResidentPacks(
  packs: readonly CatalogInstrumentPack[],
  variant: ModelVariant,
): CatalogInstrumentPack[] {
  const selected = new Map<CatalogInstrumentPack["kind"], CatalogInstrumentPack>();
  for (const pack of packs) {
    const current = selected.get(pack.kind);
    if (
      current === undefined ||
      conservativeResidentBytes(pack, variant) > conservativeResidentBytes(current, variant)
    ) {
      selected.set(pack.kind, pack);
    }
  }
  return [...selected.values()];
}
