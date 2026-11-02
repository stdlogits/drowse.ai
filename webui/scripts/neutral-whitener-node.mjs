import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

const webuiRoot = resolve(import.meta.dirname, "..");

/**
 * The core pack's neutral whitener as the core builder fits it from `build_hosted_packs.py`'s
 * native rows (`src/hosted/fitting/neutralWhitener.ts` on the shipped fitting kernel), in Node:
 * the core validator compares a pack's whitener with it, and the instrument validator reads a
 * staged units pack through it before the core pack exists.
 */
export async function fitNativeNeutralWhitener(vite, rowsBytes, modelLock) {
  const [{ decodeNeutralRows, fitNeutralWhitener }, kernel] = await Promise.all([
    vite.ssrLoadModule("/src/hosted/fitting/neutralWhitener.ts"),
    import("../public-hosted/wasm/drowse_fitting_wasm.js"),
  ]);
  await kernel.default({
    module_or_path: await readFile(resolve(webuiRoot, "public-hosted/wasm/drowse_fitting_wasm_bg.wasm")),
  });
  const native = decodeNeutralRows(new Uint8Array(rowsBytes), modelLock.layerMap, modelLock.hiddenSize);
  return fitNeutralWhitener(kernel, native.matrices, native.rows, modelLock.hiddenSize, modelLock.layerMap).tensorBytes;
}
