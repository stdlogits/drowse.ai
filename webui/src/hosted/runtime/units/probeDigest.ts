// Probe identity digests for the units and `slens/` read families, as drowse computes them:
// `sha256(repr(tuple))` over Python reprs (native `UnitsInstrument.probe_hash`, `units-readout-v2`;
// `LensInstrument.probe_hash` for an S probe, `slens-footprint-z-v3`, which also binds the
// neutral out means the read centres on and the token mask's source).
// The repr rules are the ones browserInstrumentPacks.ts' `pythonTupleDigest` follows (that
// helper is the pack registry's and only takes flat tuples).
import { sha256 } from "@noble/hashes/sha2.js";
import { bytesToHex } from "@noble/hashes/utils.js";

/** A Python value: `str`, `int`, `{float}` (repr as a float), a tuple, or `{list}`. */
export type PythonValue =
  | string
  | number
  | { readonly float: number }
  | { readonly list: readonly PythonValue[] }
  | readonly PythonValue[];

export function pythonRepr(value: PythonValue): string {
  if (typeof value === "string") return pythonStringRepr(value);
  if (typeof value === "number") {
    if (!Number.isSafeInteger(value)) throw new TypeError(`${value} is not a Python int`);
    return String(value);
  }
  if (Array.isArray(value)) {
    const items = (value as readonly PythonValue[]).map(pythonRepr);
    return `(${items.join(", ")}${items.length === 1 ? "," : ""})`;
  }
  if ("list" in (value as object)) return `[${(value as { list: readonly PythonValue[] }).list.map(pythonRepr).join(", ")}]`;
  return pythonFloatRepr((value as { float: number }).float);
}

export function pythonDigest(value: PythonValue): string {
  return bytesToHex(sha256(new TextEncoder().encode(pythonRepr(value))));
}

function pythonStringRepr(value: string): string {
  const quote = value.includes("'") && !value.includes('"') ? '"' : "'";
  let output = quote;
  for (const character of value) {
    const codePoint = character.codePointAt(0)!;
    if (character === "\\" || character === quote) output += `\\${character}`;
    else if (character === "\t") output += "\\t";
    else if (character === "\n") output += "\\n";
    else if (character === "\r") output += "\\r";
    else if (character === " " || !/[\p{Cc}\p{Cf}\p{Cs}\p{Co}\p{Cn}\p{Zl}\p{Zp}\p{Zs}]/u.test(character)) output += character;
    else if (codePoint <= 0xff) output += `\\x${codePoint.toString(16).padStart(2, "0")}`;
    else if (codePoint <= 0xffff) output += `\\u${codePoint.toString(16).padStart(4, "0")}`;
    else output += `\\U${codePoint.toString(16).padStart(8, "0")}`;
  }
  return `${output}${quote}`;
}

/** Python's `repr(float)`: the shortest round-trip digits (as JS), fixed notation for
 *  exponents in [-4, 16), else `d.ddde±XX`, and always a `.0` on an integral value. */
export function pythonFloatRepr(value: number): string {
  if (Number.isNaN(value)) return "nan";
  if (!Number.isFinite(value)) return value > 0 ? "inf" : "-inf";
  if (Object.is(value, -0)) return "-0.0";
  const negative = value < 0;
  const [digitsRaw, exponentRaw] = Math.abs(value).toExponential().split("e");
  const digits = digitsRaw.replace(".", "");
  const exponent = Number(exponentRaw);
  let body: string;
  if (exponent >= -4 && exponent < 16) {
    if (exponent < 0) body = `0.${"0".repeat(-exponent - 1)}${digits}`;
    else if (digits.length <= exponent + 1) body = `${digits}${"0".repeat(exponent + 1 - digits.length)}.0`;
    else body = `${digits.slice(0, exponent + 1)}.${digits.slice(exponent + 1)}`;
  } else {
    const mantissa = digits.length === 1 ? digits : `${digits[0]}.${digits.slice(1)}`;
    body = `${mantissa}e${exponent >= 0 ? "+" : "-"}${String(Math.abs(exponent)).padStart(2, "0")}`;
  }
  return negative ? `-${body}` : body;
}

/** Python's `round(x, ndigits)` on a float: the exact binary value rounded half to even at
 *  `ndigits` decimals, then the nearest double to that decimal. */
export function pythonRound(value: number, ndigits: number): number {
  if (!Number.isFinite(value) || value === 0) return value;
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, value);
  const bits = view.getBigUint64(0);
  const negative = bits >> 63n === 1n;
  const exponentBits = Number((bits >> 52n) & 0x7ffn);
  const fraction = bits & 0xfffffffffffffn;
  const mantissa = exponentBits === 0 ? fraction : fraction | (1n << 52n);
  const exponent = (exponentBits === 0 ? 1 : exponentBits) - 1075;
  const scale = 10n ** BigInt(ndigits);
  let n: bigint;
  if (exponent >= 0) {
    n = (mantissa << BigInt(exponent)) * scale;
  } else {
    const numerator = mantissa * scale;
    const denominator = 1n << BigInt(-exponent);
    n = numerator / denominator;
    const twice = 2n * (numerator - n * denominator);
    if (twice > denominator || (twice === denominator && n % 2n === 1n)) n += 1n;
  }
  const rounded = Number(`${n}e-${ndigits}`);
  return negative ? -rounded : rounded;
}

/** `UnitsInstrument.probe_hash`: the unit and its reading scale (detail and values, each
 *  `round(v, 9)`). */
export function unitProbeDigest(modelId: string, unit: string, detail: string, scale: readonly number[]): string {
  return pythonDigest([
    "units-readout-v2",
    modelId,
    unit,
    detail,
    { list: scale.map((value) => ({ float: pythonRound(value, 9) })) },
  ]);
}

/** `LensInstrument._centre_digest`: sha256 over `"{layer}:"` and the fp32 bytes of the neutral
 *  out mean at each layer (`means` is `[L, d]` row major). */
export function centreDigest(means: Float32Array, hiddenSize: number, layers: readonly number[]): string {
  const hash = sha256.create();
  for (const layer of layers) {
    hash.update(new TextEncoder().encode(`${layer}:`));
    const row = means.slice(layer * hiddenSize, (layer + 1) * hiddenSize);
    hash.update(new Uint8Array(littleEndianBytes(row)));
  }
  return bytesToHex(hash.digest());
}

function littleEndianBytes(row: Float32Array): ArrayBuffer {
  const out = new DataView(new ArrayBuffer(row.length * 4));
  for (let i = 0; i < row.length; i += 1) out.setFloat32(i * 4, row[i], true);
  return out.buffer;
}

/** `LensInstrument.probe_hash` for an `slens/` probe: the S lens identity, the centring
 *  means' digest and the token mask's digest (native `_mask_digest`, from the S pack). */
export function slensProbeDigest(
  modelId: string,
  word: string,
  tokenId: number,
  layers: readonly number[],
  identity: { lens: string; centre: string; mask: string },
): string {
  return pythonDigest([
    "slens-footprint-z-v3",
    modelId,
    word,
    tokenId,
    [...layers],
    [identity.lens, identity.centre, identity.mask],
  ]);
}
