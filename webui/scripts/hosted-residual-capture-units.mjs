// Capture schema 5's `units` member (browser-runtime/README.md, "production-q4 comparator"): on a
// `standard-v4` library the block sites (`sites.f32`, fp32 `[layers, positions, d_ff + H·hd + d]`,
// act | z | mid) and 1-64 neuron-write static reads, at least one at every captured layer; on a
// `standard-v4-residual` library no member. `q4_unit_parity.py` reads what this writes.
import { createHash } from "node:crypto";

export const CAPTURE_SCHEMA_VERSION = 5;
export const STATIC_READ_TOP = 8;
export const MAXIMUM_STATIC_READS = 64;

/** What is wrong with a worker's units result against the lock and the capture input. */
export function unitsCaptureFailures({ structuredHookProfile, units, sitesBytes, lock, positions }) {
  if (structuredHookProfile === "standard-v4-residual") {
    return units === undefined && sitesBytes === undefined
      ? []
      : ["a standard-v4-residual capture carries no block sites or static reads"];
  }
  if (structuredHookProfile !== "standard-v4") {
    return [`structured hook profile ${String(structuredHookProfile)} predates capture schema 5`];
  }
  const failures = [];
  if (!isRecord(units)) return ["a standard-v4 capture's units result is missing"];
  const spec = lock.blockSpec;
  const block = units.block;
  if (
    !isRecord(block) || !isRecord(spec) ||
    block.d_ff !== spec.dFF || block.n_heads !== spec.nHeads || block.head_dim !== spec.headDim
  ) {
    failures.push(`units block ${JSON.stringify(block)} is not the lock's block spec`);
    return failures;
  }
  const width = block.d_ff + block.n_heads * block.head_dim + lock.hiddenSize;
  const expectedBytes = lock.layerMap.length * positions.length * width * 4;
  if (sitesBytes === undefined) failures.push("block sites capture body is missing");
  else {
    if (sitesBytes.length !== expectedBytes || units.sitesBytes !== expectedBytes) {
      failures.push(
        `block sites byte count is ${sitesBytes.length}, envelope declares ${String(units.sitesBytes)} (expected ${expectedBytes})`,
      );
    }
    const digest = createHash("sha256").update(sitesBytes).digest("hex");
    if (digest !== units.sitesSha256) {
      failures.push(`block sites SHA-256 is ${digest}, envelope declares ${String(units.sitesSha256)}`);
    }
    if (sitesBytes.length % 4 === 0) {
      const values = new Float32Array(sitesBytes.buffer.slice(sitesBytes.byteOffset, sitesBytes.byteOffset + sitesBytes.length));
      if (values.some((value) => !Number.isFinite(value))) failures.push("block sites are not finite");
    }
  }
  const reads = units.staticReads;
  if (!Array.isArray(reads) || reads.length < 1 || reads.length > MAXIMUM_STATIC_READS) {
    failures.push(`a standard-v4 capture needs 1 to ${MAXIMUM_STATIC_READS} static reads`);
    return failures;
  }
  const seen = new Set();
  const layers = new Set();
  for (const read of reads) {
    const key = `${read?.layer}:${read?.neuron}`;
    if (
      !isRecord(read) || Object.keys(read).sort().join(",") !== "ids,layer,neuron,probabilities" ||
      !lock.layerMap.includes(read.layer) ||
      !Number.isSafeInteger(read.neuron) || read.neuron < 0 || read.neuron >= block.d_ff ||
      seen.has(key) ||
      !Array.isArray(read.ids) || read.ids.length !== STATIC_READ_TOP ||
      new Set(read.ids).size !== STATIC_READ_TOP ||
      read.ids.some((id) => !Number.isSafeInteger(id) || id < 0) ||
      !Array.isArray(read.probabilities) || read.probabilities.length !== STATIC_READ_TOP ||
      read.probabilities.some((value) => typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1) ||
      read.probabilities.some((value, index) => index > 0 && value > read.probabilities[index - 1])
    ) {
      failures.push(`static read ${JSON.stringify(read)?.slice(0, 200)} is invalid`);
      continue;
    }
    seen.add(key);
    layers.add(read.layer);
  }
  const missing = lock.layerMap.filter((layer) => !layers.has(layer));
  if (missing.length > 0) failures.push(`no static read at captured layers ${missing.join(", ")}`);
  return failures;
}

/** `capture.json`'s `units` member for a checked standard-v4 result (`q4_unit_parity.py`
 *  `validate_units_member`). */
export function captureUnitsMember(units) {
  return {
    block: { ...units.block },
    sites_sha256: units.sitesSha256,
    static_reads: units.staticReads.map(({ layer, neuron, ids, probabilities }) => ({
      layer,
      neuron,
      ids: [...ids],
      probabilities: [...probabilities],
    })),
  };
}

function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
