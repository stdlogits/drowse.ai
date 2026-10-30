# Architecture

The engine-wide design (extraction, composition, injection, reads, and the
dashboard) is in drowse's [ARCHITECTURE.md](drowse/ARCHITECTURE.md). This file
covers what the browser edition adds on top of it.

## Hosted browser build

The hosted Svelte entry is isolated from the Python dashboard: it has its own
Vite configuration and public assets and writes only to `webui/dist-hosted`.
It compiles the dashboard source from the `drowse` submodule; the
`@runtime-*` aliases in `webui/vite.hosted.config.ts` select the browser
implementations of the runtime seams.
`HostedController` owns compatibility, catalog, download, storage, and model
lifecycle; the workbench talks through the same `RuntimeClient` contract used by
the HTTP implementation. Browser orchestration is assigned to a dedicated
worker, and the fitting boundary targets a separate Rust/WASM worker so large
tensors do not cross the UI thread.

Hosted authoring covers manifold and template math only. SAE training and
J-lens fitting remain Python/developer-release workflows and are neither
imported by the PWA nor exposed through its worker protocol. The browser can
download, validate, activate, read, probe, gate, and steer with compatible
precomputed SAE and J-lens packs entirely on the GPU.

The runtime and distribution locks are `verified`, with pinned runtime artifacts
and provisioned catalog signing keys. The deterministic fixture backend remains
limited to tests and local UI development. Release checks reject unverified locks,
missing trust roots, and incompatible artifact closures. These release checks do
not certify numerical accuracy across every model, device, or prompt; model-specific
validation evidence remains separate. No hosted path may substitute cloud inference
or CPU-only inference when
WebGPU, the pinned Drowse MLC backend, verified artifacts, or a compatible device
is unavailable.

The browser runtime accepts a caller-owned, read-only artifact cache.
Drowse validates the MLC configuration, tokenizer, `tensor-cache.json`, exact
weight-shard closure, model library, and signed sizes before creating the
engine. WebLLM reads those verified OPFS `File` objects directly and cannot
populate a second persistent cache or request an unknown URL. Chromium consumes
an adapter after its first device request, so the compatibility worker destroys
the calibration device and acquires a fresh adapter immediately before load. It
rejects that adapter unless its hardware status, identity, features, and
bucketed limits match the checked adapter, then lets WebLLM create the production
device from it. Release builds still require the exact pinned fork package and
immutable production artifacts.

The browser owns a transport-neutral parser for the complete Python steering
syntax, including projections, ablations, manifold positions, phase windows,
probe gates, SAE selectors, and J-lens selectors. Python and TypeScript consume
the same accept/reject fixture. Parsing does not imply execution support: each
expression is reduced to an explicit hook-feature requirement set before GPU
program compilation. The `standard-v3` library profile implements
structured affine and curved programs, exact geometry reads, precomputed SAE
steering/readout, and exact full-vocabulary J-lens steering/readout. The worker
still validates the library feature manifest, runtime identity, model profile,
and required instrument packs before every program is installed; unsupported
or incompatible combinations fail before generation. Production downloads require
the verified runtime lock, immutable model closure, signed catalog, and allowed
distribution origins; the checked-in distribution lock provisions these origins
and trust keys.
