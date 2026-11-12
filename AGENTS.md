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
  - `scripts/` — unit tests, release and catalog tooling.
  - `e2e/` — Playwright specs against the hosted dev and preview servers.
  - `contact-worker/` — separate Cloudflare Worker with its own package.
- `browser-runtime/` — WebLLM/MLC forks and compiler, the Rust/WASM fitting
  kernel, runtime and distribution locks, release inputs.
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

## Parity

There are no parity goldens. The browser runtime matches the engine by reading
the pinned drowse source; when a change moves numerics, check it once against the
pinned Python engine on a tiny model, in scratch, and commit nothing from that
comparison. Tests here check invariants and quiet-regression risks. Small input
data a test needs may live in `browser-runtime/fixtures/` or beside the code that
imports it; recorded engine outputs may not.

## Release

Release builds record this repository's exact HEAD as the source revision
(`drowse-source-url` points at `stdlogits/drowse.ai/tree/<sha>`); the gitlink at
that commit pins the drowse source. `./deploy.sh` builds the release from a clean HEAD pushed to
`main` (`build:hosted:release`: the lock check, the signed catalog at the distribution lock's URL,
the build and its output checks), uploads `webui/dist-hosted` to the `drowse-web` Pages project
with Wrangler and verifies the live site; `--dry-run` stops before the upload. The Cloudflare
account is not a9's: deploying needs access granted by its owner. Publishing models, packs and a
catalog is `browser-runtime/README.md#release`.

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
