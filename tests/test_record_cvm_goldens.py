"""The C TVM VM evidence observer: one direct observation from verify-mlc-hook's two runs."""

from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("record_cvm_goldens", ROOT / "browser-runtime" / "record-cvm-goldens.py")
observer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(observer)

SUITES = {"block-sites": 7.6e-6, "unit-ablation": 1.2e-5, "unit-probes": 9.9e-7}
UNITS_LINE = (
    "units goldens (block-sites, unit-ablation, unit-probes, attention-patterns, token-anatomy, "
    "unit-activity) passed on the C TVM VM: " + json.dumps({key: SUITES for key in observer.LIBRARIES})
)


def fake_run(outputs):
    calls = []

    def run(python, arguments, environment):
        calls.append(arguments)
        return outputs[len(calls) - 1]

    return run, calls


def args(tmp_path: Path) -> Namespace:
    for key in observer.LIBRARIES:
        arch, quantization = key.split("-", 1)
        model = tmp_path / "fixtures" / f"units-{arch}-{quantization}"
        model.mkdir(parents=True, exist_ok=True)
        (model / "tensor-cache.json").write_text(key)
    return Namespace(python=tmp_path / "python", mlc_repository=tmp_path / "mlc", tvm_repository=tmp_path / "tvm",
                     units_fixtures=tmp_path / "fixtures")


def test_observation_carries_every_library_and_suite(monkeypatch, tmp_path) -> None:
    run, calls = fake_run(["... compiled C/LLVM golden checks passed\n", f"units goldens qwen3 q0f32: ...\n{UNITS_LINE}\n"])
    monkeypatch.setattr(observer, "run", run)
    observation = observer.observe(args(tmp_path))
    assert observation["evidenceType"] == "cVmGoldens"
    assert observation["fixturePath"] == "drowse/tests/fixtures/browser/units-tiny-models-v1.json"
    import hashlib
    assert observation["results"]["libraries"] == {
        key: {"modelSha256": hashlib.sha256(key.encode()).hexdigest(), "suites": SUITES} for key in observer.LIBRARIES
    }
    assert calls[0][-1] == str(tmp_path / "tvm") and calls[1][-2:] == ["--units-fixtures", str(tmp_path / "fixtures")]
    assert sorted(observation) == ["evidenceType", "fixturePath", "passed", "producedAt", "results", "schemaVersion"]


def test_a_missing_library_or_default_run_is_refused(monkeypatch, tmp_path) -> None:
    partial = UNITS_LINE.split(": ", 1)[0] + ": " + json.dumps({"qwen3-q0f32": SUITES})
    monkeypatch.setattr(observer, "run", fake_run(["compiled C/LLVM golden checks passed", partial])[0])
    with pytest.raises(observer.ObservationError, match="covered"):
        observer.observe(args(tmp_path))
    monkeypatch.setattr(observer, "run", fake_run(["nothing", UNITS_LINE])[0])
    with pytest.raises(observer.ObservationError, match="default structured"):
        observer.observe(args(tmp_path))


def test_the_toolchain_environment_is_the_exact_source_roots(tmp_path) -> None:
    environment = observer.toolchain_environment(tmp_path / "mlc", tmp_path / "tvm")
    assert environment["PYTHONPATH"].split(":") == [str(tmp_path / "tvm" / "python"), str(tmp_path / "mlc" / "python")]
    assert environment["TVM_DEVICE_BACKEND_AUTOLOAD"] == "0" and environment["SKIP_LOADING_MLCLLM_SO"] == "1"


def test_the_toolchain_libraries_are_found_in_build_lib(tmp_path) -> None:
    (tmp_path / "tvm" / "build" / "lib").mkdir(parents=True)
    environment = observer.toolchain_environment(tmp_path / "mlc", tmp_path / "tvm")
    assert environment["TVM_LIBRARY_PATH"] == str(tmp_path / "tvm" / "build" / "lib")


def test_a_virtual_environment_python_keeps_its_path(tmp_path) -> None:
    base = tmp_path / "base" / "python3"
    base.parent.mkdir()
    base.write_text("")
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "python").symlink_to(base)
    for name in ("mlc", "tvm", "fixtures"):
        (tmp_path / name).mkdir()
    args = observer.parse_args([
        "--python", str(venv / "python"),
        "--mlc-repository", str(tmp_path / "mlc"),
        "--tvm-repository", str(tmp_path / "tvm"),
        "--units-fixtures", str(tmp_path / "fixtures"),
    ])
    assert args.python == venv / "python"

