#!/usr/bin/env python3
"""Record the affine lowering of the unit-ablation `mixed` case's push.

The `mixed` golden (`0.4 deceptive_fixture + !neuron/L1n5`) records the forwards but not how its
push of the session-registered fixture direction lowers to affine groups: that lowering runs
through the fixture session's whitener. This script runs drowse's own lowering
(`tests.support.browser_units.lowered_pushes`, the one `neuron-push` records) on each fixture
session and writes it in neuron-push's case shape, bound to the model file's digest, so the C VM
and WebGPU harnesses can run `mixed` from its recorded tables plus these groups.

Usage (from the drowse.ai root, CPU only):

    python browser-runtime/forks/build-mixed-affine.py [--check]

`--check` regenerates in memory and fails if the committed file differs beyond 1e-6.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "browser-runtime" / "fixtures" / "unit-ablation-mixed-affine-v1.json"
EXPRESSION = "0.4 deceptive_fixture"
ARCHES = ("qwen3", "gemma3")


def build() -> dict:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    sys.path.insert(0, str(ROOT / "drowse"))
    from tests.support import browser_units as bu

    if f"0.4 {bu.FIXTURE_DIRECTION} + !neuron/L1n5" not in [s for _n, s, _k in bu.ABLATION_CASES]:
        raise SystemExit("the unit-ablation `mixed` case no longer pushes " + EXPRESSION)
    arches = {}
    for arch in ARCHES:
        with bu.fixture(arch, lenses=False) as fx:
            bu._register_fixture_direction(fx)
            arches[arch] = {"cases": bu.lowered_pushes(fx, bu.FIXTURE_DIRECTION, [EXPRESSION])}
    return {
        "schemaVersion": 1,
        "about": "The unit-ablation `mixed` case's push lowered to affine groups by drowse's "
                 "lowered_pushes on each fixture session (neuron-push's case shape). Regenerate "
                 "with python browser-runtime/forks/build-mixed-affine.py.",
        "golden": "unit-ablation",
        "case": "mixed",
        "fixtureModels": {"sha256": bu.model_file_sha256()},
        "arches": arches,
    }


def _close(a, b) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) <= 1e-6
    return a == b


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    doc = build()
    if args.check:
        if not _close(json.loads(OUT.read_text()), doc):
            raise SystemExit(f"{OUT.name} is stale: regenerate it")
        print(f"{OUT.name} matches")
        return
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
