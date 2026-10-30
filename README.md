# drowse.ai

The source of [drowse.ai](https://drowse.ai): the browser edition of
[drowse](https://github.com/stdlogits/drowse), an interpretability workbench for
activation steering and trait monitoring. The site runs the drowse dashboard
with inference, steering, probes, SAE features, and the J-lens executing on the
device through WebGPU. Prompts, conversations, and fitted artifacts never leave
the browser.

## Layout

| Path | What it holds |
| --- | --- |
| `drowse/` | Submodule pinned to [stdlogits/drowse](https://github.com/stdlogits/drowse): the shared dashboard source (`drowse/webui/src`), the Python reference implementation, and the parity goldens (`drowse/tests/fixtures/browser`) the browser runtime replays. |
| `webui/` | The hosted shell (`hosted/`), site-only UI and runtime (`src/`), static assets (`public-hosted/`), build, release, and test scripts (`scripts/`), Playwright specs (`e2e/`), and the `/api/contact` Worker (`contact-worker/`). |
| `browser-runtime/` | The WebGPU runtime: WebLLM/MLC forks and compiler, the Rust/WASM fitting kernel, runtime and distribution locks, release inputs, and evidence schemas. |
| `tests/` | Python tests for the browser-runtime build and validation tools. |

The hosted Vite config swaps the dashboard's HTTP runtime for the browser one
through the `@runtime-*` seams; site code imports shared modules by their path in
the submodule. [ARCHITECTURE.md](ARCHITECTURE.md) describes the runtime;
[browser-runtime/README.md](browser-runtime/README.md) covers artifacts, locks,
and release.

## Development

Use Node 24.

```bash
git clone --recurse-submodules https://github.com/stdlogits/drowse.ai
cd drowse.ai
npm ci                    # one workspace install at the root
cd webui
npm run dev:hosted        # hosted app dev server
npm run check             # shared UI policy, landing, metadata, and WebMCP checks
npm run check:hosted      # runtime, release-tool, and lock checks
npm run test:hosted       # runtime tests and a preview build into dist-hosted/
npm run test:e2e          # Playwright, against the dev and preview servers
```

Python tooling and its tests run against the submodule:

```bash
pip install -e "./drowse[dev]"
pytest
```

### Changing shared dashboard code

Dashboard source shared with `drowse serve` lives in the drowse repository. Land
the change there, then move the pin here:

```bash
git -C drowse fetch origin && git -C drowse checkout <commit-on-main>
git add drowse && git commit -m "bump drowse to <commit>"
```

Dependabot proposes the bump weekly. To try an unmerged drowse branch, check it
out inside `drowse/`; commit only pins that exist on drowse's `main`. A pin
bump changes executable source, so release evidence recorded against the old
pin has to be recorded again.

## Browser edition status

This repository builds an isolated Svelte PWA for the on-device WebGPU
edition. The runtime and distribution locks are `verified`, enabling installation
from the signed catalog after the browser's device checks pass. These release
checks do not guarantee that every GPU remains stable under model load.
The [published model files](https://huggingface.co/logitsml/drowse-web-catalog)
are available separately; their upload alone does not establish compatibility.
Base-model file links appear on the home page without a beta badge.
Gemma PT still has a matched-weight numerical discrepancy; Qwen 3.5 still needs
matched-quantization validation. Matching core packs and a signed installation
catalog also remain required. J-lens and SAE are optional for base-model setup.
The development fixture is deterministic test data, not real inference.

The device check shows the selected GPU vendor and architecture when the browser
exposes them. Its short compute test is not a full model-load stress test. On
Windows, an Intel adapter gets conditional guidance for selecting a dedicated
GPU through the browser's Windows Graphics settings. Two recorded device losses
since the last successful load block further loads on that device profile;
changing models or context length does not bypass that block. Software fallback
adapters remain unsupported, including in remote sessions without hardware WebGPU.

Windows Chromium on an identified Intel `gen-9` adapter is conservatively blocked
before model loading following a reported GPU-process watchdog hang during
initialization. Other Intel generations retain a warning. Chrome on Windows
ignores WebGPU's `powerPreference`; Drowse cannot enumerate or select a hidden
dedicated GPU. On dual-GPU laptops, copy
`chrome://flags/#force-high-performance-gpu` into Chrome's address bar, enable it,
restart Chrome, and run the device check again. Confirm that the selected adapter
is the dedicated GPU. This changes GPU routing, not the GPU's memory capacity.

The hosted edition keeps prompts, conversations, activations, and
fitted artifacts on the device. WebGPU is mandatory for inference; it does not
fall back to cloud inference or CPU-only inference. The Python server
and bundled dashboard in [stdlogits/drowse](https://github.com/stdlogits/drowse)
remain supported alongside the hosted edition.

## Deployment

The repository builds and verifies static output; Cloudflare credentials, Pages
projects, Access policy, DNS, and publication stay outside it. See
[browser-runtime/README.md](browser-runtime/README.md#cloudflare-pages-deployment)
and [webui/hosted/README.md](webui/hosted/README.md).

## License

GNU AGPL v3 or later; see [LICENSE](LICENSE). Report security issues as
described in [SECURITY.md](SECURITY.md).
