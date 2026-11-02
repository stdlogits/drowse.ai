/**
 * The hosted models drowse fits natively on the browser weights (6.0's BlockSpec: Gemma 3 and
 * Qwen 3). Their R and S lenses, units packs and core neutral statistics come from
 * `browser-runtime/build_hosted_packs.py`, and their core whitener from the native neutral rows.
 * GPT-2, Pythia and Qwen3.5 base are refused natively (units answer 422); their lenses are the
 * provider's and their core whitener is fitted from rows captured in the browser.
 */
export const NATIVE_FIT_ARCHITECTURES = Object.freeze(["gemma3_text", "qwen3"]);

export function nativeFitModel(modelLock) {
  return NATIVE_FIT_ARCHITECTURES.includes(modelLock?.architecture);
}
