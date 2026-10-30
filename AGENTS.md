# AGENTS.md

## What this is

The source of [drowse.ai](https://drowse.ai), the browser edition of drowse: the
drowse dashboard running inference, steering, probes, SAE features, and the
J-lens on the device through WebGPU, plus the landing and content pages, the
Neuron Lens explorer (`/neuron-lens`), and the `/api/contact` Worker.

It is built on [stdlogits/drowse](https://github.com/stdlogits/drowse), pinned
as the `drowse/` submodule. That repository owns the Python engine, the CLI and
server, and the dashboard source both editions share (`drowse/webui/src`). Its
`AGENTS.md` and `ARCHITECTURE.md` describe the engine; read them before touching
anything that crosses the runtime seam.

## Layout

- `drowse/` — submodule. Never commit changes inside it from here: land them in
  drowse, then bump the pin (`git -C drowse checkout <commit>`, `git add drowse`).
  Pins must be commits on drowse's `main`.
- `package.json` — npm workspace root. `npm ci` installs everything here so the
  submodule's shared sources resolve the same `node_modules`. Scripts live in
  `webui/package.json`; run them from `webui/` (or `npm run <script> -w webui`).
- `webui/` — the hosted shell and everything site-only:
  - `hosted/` — Vite root (`index.html`, `main.ts`, shell controller).
  - `src/hosted/` — site UI, browser runtime client, artifact store, fitting
    workers. `src/drawers/LocalRuntimeDrawer.svelte` and the `src/lib/runtime/`
    files here are the browser halves of dashboard seams.
  - `vite.hosted.config.ts` — the build. Its `@runtime-*` aliases pick the
    browser implementations of the seams the shared app imports.
  - `vite.config.ts` — what ad-hoc Vite servers in `scripts/` load when they pass
    no config: the dashboard's own config from the submodule.
  - `scripts/` — unit tests, release and catalog tooling, evidence recording.
    `scripts/neurons/` builds Neuron Lens's data; see below.
  - `e2e/` — Playwright specs against the hosted dev and preview servers.
  - `contact-worker/` — separate Cloudflare Worker with its own package.
- `browser-runtime/` — WebLLM/MLC forks and compiler, the Rust/WASM fitting
  kernel, runtime and distribution locks, release inputs, evidence schemas.
  `browser-runtime/README.md` is the reference for artifacts and release.
- `tests/` — Python tests for the browser-runtime tools (`pytest`, with
  `pip install -e "./drowse[dev]"`).

## Referencing shared dashboard code

Site code reaches shared modules by their path in the submodule; there is no
alias for them.

- Source imports: relative paths, e.g. `../../../drowse/webui/src/lib/stores.svelte`.
- `ssrLoadModule` in scripts (Vite root `webui/`): `/../drowse/webui/src/...`.
- Imports evaluated in a browser page: `/@fs${absolutePath}`, since a browser
  collapses `/../` before the request reaches Vite.
- Source-reading tests: look in `webui/src` first and fall back to
  `../drowse/webui/src` when a check spans both trees.

A change that only touches shared code belongs in drowse, with its tests there.
Site tests that read shared files exist to check how the hosted shell uses them.

## Parity goldens

`drowse/tests/fixtures/browser/*` are the Python reference outputs the browser
runtime replays (`rank-one-hook-program`, `structured-hook-parity`, the Rust
topology tests, `verify-mlc-hook.py`). drowse's own tests pin them against the
engine; regenerate them there, then bump the pin and update the runtime here.
Browser-only fixtures stay in `browser-runtime/fixtures/`.

## Release provenance

Release builds record this repository's exact HEAD as the source revision
(`drowse-source-url` points at `stdlogits/drowse.ai/tree/<sha>`); the gitlink at
that commit pins the drowse source. Evidence closures may name files inside the
submodule (`drowse/drowse/core/...`); the tracked-file checks resolve those in
the submodule's own repository. A submodule bump is a change to executable
source, so evidence recorded against the previous pin no longer qualifies.
Deployment is a manual upload of `webui/dist-hosted`; see
`browser-runtime/README.md#cloudflare-pages-deployment`.

## Neuron Lens data

`/neuron-lens` reads static JSON from `webui/public-hosted/neurons-data/`, built from the private
neuron-lens catalog (`~/Work/neuron-lens/data/catalog/qwen3-8b`) and gitignored like it. Two
stages, run from `webui/` (each script's docstring has the full invocation):

1. `scripts/neurons/translate.py` machine-translates non-English tokens, label terms and example
   contexts with Claude Haiku 5.5 on the Batch API (`collect`, `pilot`, `submit --cap`, `results`,
   then `fill` for what Haiku skipped), caching them in `webui/.neuron-lens/translations.jsonl`.
   Rerun it only when the catalog's vocabulary, labels or contexts change.
2. `scripts/neurons/build-neuron-catalog.py <catalog> <tokenizer.json> public-hosted/neurons-data
   .neuron-lens/translations.jsonl` writes every neuron and head. `NEURON_LAYERS=8,21` builds a
   subset for development.

The page's types (`webui/src/hosted/ui/neurons/data.ts`) mirror the builder's output; change both
together, and the e2e spec's mocks with them.

## Commands

```bash
git submodule update --init
npm ci
cd webui
npm run dev:hosted
npm run check && npm run check:hosted && npm run test:hosted
npm run test:e2e
npm run build:hosted:release   # release gate; needs verified locks
```
