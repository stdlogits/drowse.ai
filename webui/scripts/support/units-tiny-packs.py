"""Build the hosted R, S, core-statistics and units packs for drowse's tiny ABI v5 fixture
model (`build_hosted_packs.build`, as tests/test_hosted_packs.py builds them) into OUT, and print
what native computes for them: the build record, the runtime record and the S mask digest
(`LensInstrument._mask_digest`). scripts/units-pack-source.test.mjs reads the packs back
through the browser registry."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "browser-runtime"))
sys.path.insert(0, str(ROOT / "drowse"))

from build_hosted_packs import KINDS, NativeModel, build  # noqa: E402
from tests.support.browser_units import fixture  # noqa: E402


def main() -> None:
    arch, out = sys.argv[1], Path(sys.argv[2])
    with fixture(arch) as fx:
        model = NativeModel(fx.session)
        identity = {
            "sourceModel": "fixture/units-tiny",
            "sourceRevision": "f" * 40,
            "convertedManifestSha256": "0" * 64,
            "quantization": "q0f32",
            "tokenizerSha256": "1" * 64,
            "chatTemplateSha256": "2" * 64,
            "modelLibrarySha256": "3" * 64,
            "runtimeAbi": "drowse-web-runtime-v2",
            "hookAbi": "block-sites-v5",
            "hiddenSize": model.d_model,
            "layerMap": list(range(model.n_layers)),
        }
        runtime = {"modelId": "units-tiny", "runtimeIdentity": identity, "runtimeIdentitySha256": "a" * 64}
        record = build(model, runtime, out, KINDS, full_depth=False, words=["a", "e", "t"])
        json.dump({
            "runtime": runtime,
            "record": record,
            "maskDigest": model.session.lens._mask_digest(),
        }, sys.stdout)


if __name__ == "__main__":
    main()
