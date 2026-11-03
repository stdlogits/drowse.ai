import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _module():
    path = Path(__file__).parents[1] / "browser-runtime" / "import-provider-sae.py"
    spec = spec_from_file_location("hosted_provider_sae", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("size,layer", [("270m", 12), ("1b", 13), ("4b", 17)])
def test_sae_import_preserves_uncentered_encoding_and_decoder_bias(tmp_path: Path, size: str, layer: int):
    module = _module()
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "model_name": f"google/gemma-3-{size}-it",
        "hf_hook_point_out": f"model.layers.{layer}.output",
        "width": 2,
    }))
    params = tmp_path / "params.safetensors"
    tensors = {
        "w_enc": torch.eye(2), "w_dec": torch.eye(2),
        "b_enc": torch.tensor([0.5, -1.0]),
        "b_dec": torch.tensor([1000.0, -1000.0]),
        "threshold": torch.tensor([1.25, 2.5]),
    }
    module.save_file(tensors, params)
    args = SimpleNamespace(
        sae_config=config, sae_params=params, sae_provider_repository=None,
        output=tmp_path / "output", runtime_fingerprint="a" * 64,
    )
    module.create_sae_pack(args, {
        "hiddenSize": 2, "sourceRepository": f"unsloth/gemma-3-{size}-it",
        "sourceRevision": "b" * 40,
    })
    pack = args.output / "packs/sae"
    manifest = json.loads((pack / "manifest.json").read_text())
    assert manifest["apply_b_dec_to_input"] is False
    assert manifest["activation"] == "jump_relu"
    with module.safe_open(pack / manifest["tensor_file"], framework="pt") as saved:
        assert torch.equal(saved.get_tensor("b_dec"), tensors["b_dec"])
        assert torch.equal(saved.get_tensor("b_enc"), tensors["b_enc"])
