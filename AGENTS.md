# AGENTS.md

## What this is

The source of [drowse.ai](https://drowse.ai), the browser edition of drowse: the
drowse dashboard running inference, steering, probes, SAE features, and the
J-lens on the device through WebGPU, plus the landing and content pages and the
`/api/contact` Worker. (The Neuron Lens explorer that lived at `/neuron-lens` is now its own site,
`stdlogits/neuron-lens.com`; `_redirects` sends `/neuron-lens*` and `/neurons*` there with their
query strings, so keep those lines.)

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
submodule (`drowse/drowse/geometry/...`); the tracked-file checks resolve those in
the submodule's own repository. A submodule bump is a change to executable
source, so evidence recorded against the previous pin no longer qualifies.
`./deploy.sh` builds the release from a clean, pushed HEAD, uploads `webui/dist-hosted` to the
`drowse-web` Pages project with Wrangler and verifies the live site. It skips the runtime-lock and
distribution gates unless given `--full`, which changes to the runtime, its locks or the pin need;
`--dry-run` stops before the upload. The Cloudflare account is not a9's: deploying needs access
granted by its owner. The gated process behind it is in
`browser-runtime/README.md#cloudflare-pages-deployment`.

## Commands

```bash
git submodule update --init
npm ci
cd webui
npm run dev:hosted
npm run check && npm run check:hosted && npm run test:hosted
npm run test:e2e
npm run build:hosted:release   # release gate; needs verified locks
../deploy.sh                   # deploy; see Release provenance
```
