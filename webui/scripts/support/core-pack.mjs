// A verified core-pack load request of any width and depth, for replaying drowse's goldens
// through the real structured compiler (`BrowserFeasibilityCorePackCompiler.load`): low-rank
// neutral whiteners, flat and one-node manifolds (`basis`, `nodeCoordinates` per layer) and
// curved ones (a 1-D box curve; only its fitted layers and baked Mahalanobis shares matter to
// layer placement). The archive layout is browser-core-pack.test.mjs's fixture's.
import { createHash } from "node:crypto";

const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");

function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value !== null && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function discoverNodesHash(labels, payloads, fitMode, roles) {
  const hash = createHash("sha256");
  hash.update(canonicalJson(labels));
  for (const payload of payloads) hash.update(payload);
  hash.update(canonicalJson({ fit_mode: fitMode, hyperparams: {} }));
  hash.update(canonicalJson(roles));
  hash.update(canonicalJson(roles.map(() => "abstract")));
  return hash.digest("hex");
}

const f32 = (shape, values) => ({ shape, data: Float32Array.from(values) });

// A frame's read rows dual to its write rows (`read · basisᵀ = I`): the Euclidean dual
// `(B Bᵀ)⁻¹ B` of these hand-built bases, which is `B` itself for orthonormal rows.
function dualRows(basis) {
  const r = basis.length;
  const gram = basis.map((left) => basis.map((right) => left.reduce((sum, value, i) => sum + value * right[i], 0)));
  const inverse = gram.map((_, i) => gram.map((__, j) => (i === j ? 1 : 0)));
  for (let c = 0; c < r; c += 1) {
    const pivot = gram[c][c];
    for (let j = 0; j < r; j += 1) { gram[c][j] /= pivot; inverse[c][j] /= pivot; }
    for (let i = 0; i < r; i += 1) {
      if (i === c) continue;
      const factor = gram[i][c];
      for (let j = 0; j < r; j += 1) { gram[i][j] -= factor * gram[c][j]; inverse[i][j] -= factor * inverse[c][j]; }
    }
  }
  return inverse.map((row) => basis[0].map((_, column) => row.reduce((sum, value, k) => sum + value * basis[k][column], 0)));
}

/**
 * `spec`:
 * - `hiddenSize`, `layerCount`, `structuredHookProfile` (default `standard-v5`), `modelId`;
 * - `whiteners`: `[{layer, mean, basis: [[d]…], eigenvalues, ridge}]`;
 * - `manifolds`: `[{name, labels, layers: [{layer, mean?, basis: [[d]…], nodeCoordinates: [[r]…]}],
 *   nodeCoordinates?, shares?}]` (flat), or `{name, labels, curved: true, layers: [{layer,
 *   mahalanobisShare}]}`.
 */
export async function corePackRequest(server, spec) {
  const { buildDrowseArchive, encodeFp32Safetensors, jsonBytes } = await server.ssrLoadModule("/src/hosted/artifacts/index.ts");
  const d = spec.hiddenSize;
  const fingerprint = "a".repeat(64);
  const binding = "b".repeat(64);
  const manifests = [];
  const artifacts = [];
  const add = (path, bytes) => {
    const manifest = { path, role: "core_pack", url: `https://example.test/${path}`, revision: "1".repeat(40), bytes: bytes.length, sha256: sha256(bytes) };
    manifests.push(manifest);
    artifacts.push({ manifest, file: new File([bytes], path) });
    return manifest;
  };
  for (const manifold of spec.manifolds ?? []) {
    const { name, labels } = manifold;
    const curved = manifold.curved === true;
    const fitMode = curved ? "spectral" : "pca";
    const roles = labels.map(() => null);
    const nodeFiles = Object.fromEntries(labels.map((label, index) => [
      `manifolds/local/${name}/nodes/${String(index).padStart(2, "0")}_${label}.json`,
      jsonBytes([label]),
    ]));
    const n = labels.length;
    const tensors = {};
    let fitted;
    let shares;
    if (curved) {
      // A 1-D curve through the first two axes: node t_i = i/n, mean 0, basis e0, e1.
      tensors.node_coords = f32([n, 1], labels.map((_, i) => i / n));
      fitted = manifold.layers.map((row) => row.layer);
      shares = Object.fromEntries(manifold.layers.map((row) => [String(row.layer), row.mahalanobisShare]));
      for (const layer of fitted) {
        const p = `layer_${layer}`;
        tensors[`${p}.mean`] = f32([d], new Array(d).fill(0));
        tensors[`${p}.basis`] = f32([2, d], [...Array.from({ length: d }, (_, i) => (i === 0 ? 1 : 0)), ...Array.from({ length: d }, (_, i) => (i === 1 ? 1 : 0))]);
        tensors[`${p}.read`] = tensors[`${p}.basis`];
        tensors[`${p}.node_params`] = f32([n, 1], labels.map((_, i) => i / n));
        tensors[`${p}.rbf_weights`] = f32([n, 2], new Array(2 * n).fill(0));
        tensors[`${p}.poly_coeffs`] = f32([2, 2], [0, 0, 1, 0.5]);
        tensors[`${p}.coord_offset`] = f32([1], [0]);
        tensors[`${p}.coord_scale`] = f32([1], [1]);
        tensors[`${p}.sigma_rbf_weights`] = f32([n, 1], new Array(n).fill(0));
        tensors[`${p}.sigma_poly_coeffs`] = f32([2, 1], [-20, 0]);
      }
    } else {
      tensors.node_coords = f32([n, 1], manifold.nodeCoordinates ?? labels.map((_, i) => (n === 1 ? 1 : -1 + (2 * i) / (n - 1))));
      fitted = manifold.layers.map((row) => row.layer);
      shares = Object.fromEntries(fitted.map((layer) => [String(layer), manifold.shares?.[layer] ?? 1]));
      for (const row of manifold.layers) {
        const p = `layer_${row.layer}`;
        const rank = row.basis.length;
        tensors[`${p}.mean`] = f32([d], row.mean ?? new Array(d).fill(0));
        tensors[`${p}.basis`] = f32([rank, d], row.basis.flat());
        tensors[`${p}.read`] = f32([rank, d], dualRows(row.basis).flat());
        tensors[`${p}.node_coords`] = f32([n, rank], row.nodeCoordinates.flat());
      }
    }
    const tensorBytes = encodeFp32Safetensors(tensors);
    const tensorName = "_zdGVzdC9tb2RlbA.safetensors";
    const sidecarName = tensorName.replace(".safetensors", ".json");
    const sidecar = jsonBytes({
      format_version: 10, name, method: curved ? "manifold_discover_spectral" : "manifold_discover_pca",
      drowse_version: "test", fit_mode: fitMode, hyperparams: {}, diagnostics: {},
      node_count: n, node_labels: labels, node_roles: roles, node_kinds: labels.map(() => "abstract"),
      domain: curved
        ? { type: "box", axes: [{ name: "t", periodic: false, period: 1, lo: 0, hi: 1 }] }
        : { type: "custom", embed_dim: 1, bounds: null },
      node_spread_per_layer: {}, fitted_layers: fitted, mahalanobis_share_per_layer: shares,
      origin_per_layer: curved ? Object.fromEntries(fitted.map((layer) => [String(layer), [0]])) : {},
      feature_space: "raw", nodes_sha256: discoverNodesHash(labels, Object.values(nodeFiles), fitMode, roles),
      sae_release: null, sae_revision: null, sae_fingerprint: null, sae_ids_by_layer: {}, sae_full_coverage: false,
      model_fingerprint: fingerprint, context_binding_sha256: binding, model_source_fingerprint: null,
      capture_sha256: null, capture_version: null, capture_render_sha256: null, baseline_prompts_sha256: null,
      fit_policy_version: 1, share_metric: "mahalanobis", subspace_metric: "mahalanobis",
      rbf_smoothing_per_layer: curved ? Object.fromEntries(fitted.map((layer) => [String(layer), { lambda: 0, edf: 2, gcv: 0 }])) : {},
      sigma_field_per_layer: curved ? Object.fromEntries(fitted.map((layer) => [String(layer), { sigma_mean: 0, sigma_min: 0, sigma_max: 0, lambda: 0 }])) : {},
      resolved_fit_mode: null, topology_winner: null, topology_candidates: [], components: null, bake_policy: null,
      source_model_id: null, source_model_fingerprint: null, transfer_quality_estimate: null,
    });
    const manifest = jsonBytes({
      format_version: 10, name, description: "golden replay fixture", fit_mode: fitMode, hyperparams: {},
      nodes: labels.map((label) => ({ label, role: null, kind: "abstract" })),
      files: { [tensorName]: sha256(tensorBytes), [sidecarName]: sha256(sidecar) },
      source: "local", tags: [], template_ref: null,
    });
    const archive = await buildDrowseArchive({
      primary: `manifolds/local/${name}`,
      template: null,
      producerVersion: "browser-core-test",
      source: { uri: "local", repository: null, revision: null },
      files: {
        [`manifolds/local/${name}/manifold.json`]: manifest,
        ...nodeFiles,
        [`manifolds/local/${name}/${tensorName}`]: tensorBytes,
        [`manifolds/local/${name}/${sidecarName}`]: sidecar,
      },
    });
    add(`core/${name}.drowse`, new Uint8Array(await archive.arrayBuffer()));
  }
  if (spec.whiteners?.length) {
    const tensors = {};
    for (const row of spec.whiteners) {
      const p = `layer_${row.layer}`;
      tensors[`${p}.mean`] = f32([d], row.mean);
      tensors[`${p}.basis`] = f32([row.basis.length, d], row.basis.flat());
      tensors[`${p}.eigenvalues`] = f32([row.eigenvalues.length], row.eigenvalues);
      tensors[`${p}.inverse_scales`] = f32([row.eigenvalues.length], row.eigenvalues.map((e) => 1 / Math.sqrt(e + row.ridge)));
    }
    const tensorBytes = encodeFp32Safetensors(tensors);
    const tensorManifest = add("core/neutral-whitener.safetensors", tensorBytes);
    add("core/neutral-whitener.json", jsonBytes({
      format_version: 1,
      runtime_identity_sha256: fingerprint,
      context_binding_sha256: binding,
      hidden_size: d,
      layer_map: Array.from({ length: spec.layerCount }, (_, i) => i),
      tensors_sha256: tensorManifest.sha256,
      ridge_per_layer: Object.fromEntries(spec.whiteners.map((row) => [String(row.layer), row.ridge])),
    }));
  }
  const runtimeIdentity = {
    sourceModel: "test/model", sourceRevision: "2".repeat(40), convertedManifestSha256: "3".repeat(64),
    quantization: "q4f32_1", tokenizerSha256: "4".repeat(64), chatTemplateSha256: "5".repeat(64),
    modelLibrarySha256: "6".repeat(64), runtimeAbi: "drowse-browser-v1", hookAbi: "post-block-residual-v4",
    hiddenSize: d, layerMap: Array.from({ length: spec.layerCount }, (_, i) => i),
  };
  return {
    model: { id: spec.modelId ?? "fixture-model" },
    variant: {
      structuredHookProfile: spec.structuredHookProfile ?? "standard-v5",
      runtimeIdentity,
      runtimeIdentitySha256: fingerprint,
      contextProfiles: [{ contextTokens: 2048, bindingSha256: binding }],
    },
    requiredCorePack: { files: manifests },
    artifacts,
    optionalPacks: [],
    contextTokens: 2048,
    signal: new AbortController().signal,
  };
}
