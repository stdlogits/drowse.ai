#!/usr/bin/env python3
"""Release-evidence observer for hook ABI v5 on the C TVM VM (`cVmGoldens`).

Runs ``forks/verify-mlc-hook.py`` twice with the pinned, source-built MLC/TVM toolchain: the
default structured, geometry and exact-readout goldens, then ``--units-fixtures`` over the four
fixture libraries ``build-units-fixtures.py`` makes (qwen3 and gemma3 at q0f32 and q4f32_1)
against drowse's units, replay and read goldens. Prints one direct observation for
``record-hosted-evidence.mjs`` on stdout and nothing else; any failure exits non-zero with the
reason on stderr, which the recorder refuses.

    python3 browser-runtime/record-cvm-goldens.py --python TOOLCHAIN_PYTHON \\
        --mlc-repository MLC --tvm-repository TVM --units-fixtures FIXTURES
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VERIFY = ROOT / "forks" / "verify-mlc-hook.py"
FIXTURE = "drowse/tests/fixtures/browser/units-tiny-models-v1.json"
LIBRARIES = ("gemma3-q0f32", "gemma3-q4f32_1", "qwen3-q0f32", "qwen3-q4f32_1")
DEFAULT_PASSED = "compiled C/LLVM golden checks passed"
UNITS_PASSED = "passed on the C TVM VM: "


class ObservationError(RuntimeError):
    pass


def toolchain_environment(mlc: Path, tvm: Path) -> dict[str, str]:
    """The exact-source toolchain's import roots (browser-runtime/README.md), nothing inherited
    beyond the process basics."""
    keep = {key: os.environ[key] for key in ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL") if key in os.environ}
    return {
        **keep,
        "PYTHONPATH": f"{tvm / 'python'}{os.pathsep}{mlc / 'python'}",
        "TVM_LIBRARY_PATH": str(tvm / "build"),
        "TVM_DEVICE_BACKEND_AUTOLOAD": "0",
        "SKIP_LOADING_MLCLLM_SO": "1",
        "PYTHONUNBUFFERED": "1",
    }


def run(python: Path, arguments: list[str], environment: dict[str, str]) -> str:
    completed = subprocess.run(
        [str(python), str(VERIFY), *arguments],
        env=environment, cwd=ROOT.parent, capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        tail = "\n".join((completed.stdout + completed.stderr).splitlines()[-20:])
        raise ObservationError(f"verify-mlc-hook.py {' '.join(arguments[-2:])} failed:\n{tail}")
    return completed.stdout


def units_results(stdout: str) -> dict[str, dict[str, float]]:
    """The per-library worst |Δ| of every golden file from verify-mlc-hook's closing line."""
    lines = [line for line in stdout.splitlines() if UNITS_PASSED in line]
    if len(lines) != 1:
        raise ObservationError("the units goldens did not report one closing line")
    results = json.loads(lines[0].split(UNITS_PASSED, 1)[1])
    if sorted(results) != list(LIBRARIES):
        raise ObservationError(f"the units goldens covered {sorted(results)}, not {list(LIBRARIES)}")
    return {key: {suite: float(value) for suite, value in sorted(suites.items())} for key, suites in sorted(results.items())}


def observe(args: argparse.Namespace) -> dict[str, object]:
    environment = toolchain_environment(args.mlc_repository, args.tvm_repository)
    repositories = ["--mlc-repository", str(args.mlc_repository), "--tvm-repository", str(args.tvm_repository)]
    default = run(args.python, repositories, environment)
    if DEFAULT_PASSED not in default:
        raise ObservationError("the default structured, geometry and readout goldens did not report success")
    units = run(args.python, [*repositories, "--units-fixtures", str(args.units_fixtures)], environment)
    return {
        "schemaVersion": 1,
        "evidenceType": "cVmGoldens",
        "fixturePath": FIXTURE,
        "passed": True,
        "producedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "results": {
            "backend": "c-tvm-vm",
            "defaultGoldensPassed": True,
            "libraries": {key: {"suites": suites} for key, suites in units_results(units).items()},
        },
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--python", type=Path, required=True, help="the toolchain's Python")
    parser.add_argument("--mlc-repository", type=Path, required=True)
    parser.add_argument("--tvm-repository", type=Path, required=True)
    parser.add_argument("--units-fixtures", type=Path, required=True)
    args = parser.parse_args(argv)
    for name in ("python", "mlc_repository", "tvm_repository", "units_fixtures"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        observation = observe(parse_args(sys.argv[1:] if argv is None else argv))
    except (ObservationError, OSError, json.JSONDecodeError) as error:
        print(f"record-cvm-goldens: {error}", file=sys.stderr)
        return 1
    sys.stdout.write(json.dumps(observation, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
