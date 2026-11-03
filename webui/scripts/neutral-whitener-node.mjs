import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

const webuiRoot = resolve(import.meta.dirname, "..");

/**
 * The core pack's neutral whitener as the core builder fits it from `build_hosted_packs.py`'s
 * native rows (`src/hosted/fitting/neutralWhitener.ts` on the shipped fitting kernel), in Node:
 * the core builder and validator compare a pack's whitener (tensors and ridge) with it, and the
 * instrument validator reads a staged units pack through it before the core pack exists.
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
  const fitted = fitNeutralWhitener(kernel, native.matrices, native.rows, modelLock.hiddenSize, modelLock.layerMap);
  return { tensorBytes: fitted.tensorBytes, ridgePerLayer: fitted.ridgePerLayer };
}

/** A context's `neutral-whitener.json`, as the core builder writes it beside the tensors. */
export function neutralWhitenerSidecar({ runtimeIdentitySha256, contextBindingSha256, modelLock, tensorsSha256, ridgePerLayer }) {
  return new TextEncoder().encode(`${JSON.stringify({
    format_version: 1,
    runtime_identity_sha256: runtimeIdentitySha256,
    context_binding_sha256: contextBindingSha256,
    hidden_size: modelLock.hiddenSize,
    layer_map: modelLock.layerMap,
    tensors_sha256: tensorsSha256,
    ridge_per_layer: ridgePerLayer,
  }, null, 2)}\n`);
}
