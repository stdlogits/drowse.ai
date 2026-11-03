// Hook ABI v5's VM function sets, read from the MLC overlay's own definition rather than copied:
// `browser-runtime/forks/overlay_checks.py` (REQUIRED_FUNCTIONS, UNIT_FUNCTIONS,
// UNIT_ARCHITECTURES), the module every compile gate imports. A library of a unit architecture
// exports REQUIRED ∪ UNIT and attests `standard-v4`; any other architecture exports REQUIRED, no
// unit function, and attests `standard-v4-residual` (B1 §7, HOOK_ABI.md §3).

import { readFileSync } from "node:fs";

export const OVERLAY_CHECKS_PATH = "browser-runtime/forks/overlay_checks.py";

const overlayChecksUrl = new URL(`../../${OVERLAY_CHECKS_PATH}`, import.meta.url);

const NAME = /^drowse_[a-z0-9_]+$/u;
const ARCHITECTURE = /^[a-z0-9_]+$/u;

function literal(source, name, open, close) {
  const start = source.search(new RegExp(`^${name} = \\${open}`, "mu"));
  if (start < 0) throw new Error(`overlay_checks.py does not define ${name}`);
  const body = source.slice(source.indexOf(open, start) + 1);
  const end = body.indexOf(close);
  if (end < 0) throw new Error(`overlay_checks.py ${name} is not closed`);
  const values = [];
  for (const raw of body.slice(0, end).split("\n")) {
    const line = raw.replace(/#.*$/u, "").trim();
    for (const item of line.split(",")) {
      const text = item.trim();
      if (!text) continue;
      const match = /^"([^"\\]+)"$/u.exec(text);
      if (!match) throw new Error(`overlay_checks.py ${name} holds a non-literal entry ${text}`);
      values.push(match[1]);
    }
  }
  if (new Set(values).size !== values.length || values.length === 0) {
    throw new Error(`overlay_checks.py ${name} is empty or repeats an entry`);
  }
  return Object.freeze([...values].sort());
}

/** `{required, unit, unitArchitectures}` from overlay_checks.py's source text. */
export function parseOverlayVmFunctions(source) {
  const required = literal(source, "REQUIRED_FUNCTIONS", "{", "}");
  const unit = literal(source, "UNIT_FUNCTIONS", "{", "}");
  const unitArchitectures = literal(source, "UNIT_ARCHITECTURES", "(", ")");
  if ([...required, ...unit].some((name) => !NAME.test(name))) {
    throw new Error("overlay_checks.py names a VM function outside the drowse_ namespace");
  }
  if (unit.some((name) => required.includes(name))) {
    throw new Error("overlay_checks.py lists a unit function among the required ones");
  }
  if (unitArchitectures.some((name) => !ARCHITECTURE.test(name))) {
    throw new Error("overlay_checks.py names an invalid unit architecture");
  }
  return Object.freeze({ required, unit, unitArchitectures });
}

export const OVERLAY_VM_FUNCTIONS = parseOverlayVmFunctions(readFileSync(overlayChecksUrl, "utf8"));

/** The structured hook profile a library of `architecture` attests under hook ABI v5. */
export function structuredHookProfileForArchitecture(architecture, sets = OVERLAY_VM_FUNCTIONS) {
  return sets.unitArchitectures.includes(architecture) ? "standard-v4" : "standard-v4-residual";
}

/** `{present, absent}`: the VM functions a v5 library of `profile` must and must not export. */
export function vmFunctionsForProfile(profile, sets = OVERLAY_VM_FUNCTIONS) {
  if (profile === "standard-v4") {
    return { present: Object.freeze([...sets.required, ...sets.unit].sort()), absent: Object.freeze([]) };
  }
  if (profile === "standard-v4-residual") {
    return { present: sets.required, absent: sets.unit };
  }
  throw new Error(`structured hook profile ${JSON.stringify(profile)} predates hook ABI v5`);
}

/** Whether a compiled WebAssembly library's VM executable names `name` (TVM serializes each
 *  function name as a little-endian u64 length followed by its bytes). */
export function libraryNamesVmFunction(wasmBytes, name) {
  const encoded = Buffer.from(name);
  const length = Buffer.alloc(8);
  length.writeBigUInt64LE(BigInt(encoded.byteLength));
  return Buffer.from(wasmBytes).includes(Buffer.concat([length, encoded]));
}

/** Throws unless the library exports exactly the profile's v5 surface (present and absent). */
export function requireLibraryVmFunctions(wasmBytes, profile, sets = OVERLAY_VM_FUNCTIONS) {
  const { present, absent } = vmFunctionsForProfile(profile, sets);
  const missing = present.filter((name) => !libraryNamesVmFunction(wasmBytes, name));
  if (missing.length > 0) {
    throw new Error(`${profile} model library lacks required VM function ${missing.join(", ")}`);
  }
  const extra = absent.filter((name) => libraryNamesVmFunction(wasmBytes, name));
  if (extra.length > 0) {
    throw new Error(`${profile} model library exports unit VM function ${extra.join(", ")}`);
  }
}
