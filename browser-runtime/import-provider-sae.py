#!/usr/bin/env python3
"""Package a provider JumpReLU SAE (Google's Gemma Scope 2) as a hosted SAE pack bound to one
runtime identity. The provider's parameters are copied unchanged as fp32 tensors; nothing is
trained, so a new runtime identity takes the same provider files to the same tensor bytes.

    python3 browser-runtime/import-provider-sae.py --runtime-lock browser-runtime/runtime-lock.json \
      --model-id gemma3-1b-instruct --runtime-fingerprint SHA256 \
      --sae-config config.json --sae-params params.safetensors --output PACK_DIR
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main() -> None:
    args = parse_args()
    runtime_lock = json.loads(args.runtime_lock.read_text(encoding="utf-8"))
    model = next(
        (entry for entry in runtime_lock["models"] if entry["id"] == args.model_id),
        None,
    )
    if model is None:
        raise SystemExit(f"unknown runtime-lock model {args.model_id}")
    if len(args.runtime_fingerprint) != 64:
        raise SystemExit("runtime fingerprint must be a SHA-256 digest")
    args.output.mkdir(parents=True, exist_ok=False)
    create_sae_pack(args, model)
    print(json.dumps({"sae": str(args.output)}))


def create_sae_pack(args, model: dict) -> None:
    config = json.loads(args.sae_config.read_text(encoding="utf-8"))
    layer = hook_layer(config.get("hf_hook_point_out"))
    provider_repository = args.sae_provider_repository or gemma_scope_repository(config)
    with safe_open(args.sae_params, framework="pt", device="cpu") as source:
        tensors = {
            "W_enc": source.get_tensor("w_enc").to(dtype=torch.float32).contiguous(),
            "W_dec": source.get_tensor("w_dec").to(dtype=torch.float32).contiguous(),
            "b_enc": source.get_tensor("b_enc").to(dtype=torch.float32).contiguous(),
            "b_dec": source.get_tensor("b_dec").to(dtype=torch.float32).contiguous(),
            "threshold": source.get_tensor("threshold").to(dtype=torch.float32).contiguous(),
        }
    hidden_size = model["hiddenSize"]
    feature_count = int(config["width"])
    expected = {
        "W_enc": (hidden_size, feature_count),
        "W_dec": (feature_count, hidden_size),
        "b_enc": (feature_count,),
        "b_dec": (hidden_size,),
        "threshold": (feature_count,),
    }
    for name, tensor in tensors.items():
        if tuple(tensor.shape) != expected[name] or not torch.isfinite(tensor).all():
            raise SystemExit(f"provider SAE tensor {name} has an invalid shape or value")
    if torch.any(tensors["threshold"] < 0):
        raise SystemExit("provider JumpReLU SAE contains a negative threshold")

    pack = args.output / "packs" / "sae"
    pack.mkdir(parents=True)
    tensor_path = pack / f"layer-{layer}.safetensors"
    save_file(tensors, tensor_path)
    tensor_digest = sha256(tensor_path)
    width_label = f"{feature_count // 1024}k" if feature_count % 1024 == 0 else str(feature_count)
    name = f"gemma_scope_2_l{layer}_{width_label}"
    manifest = {
        "format_version": 2,
        "kind": "local",
        "name": name,
        "release": f"local:{name}",
        "model_id": model["sourceRepository"],
        "model_fingerprint": args.runtime_fingerprint,
        "model_source_fingerprint": model["sourceRevision"],
        "layer": layer,
        "d_model": hidden_size,
        "d_sae": feature_count,
        "activation": "jump_relu",
        "apply_b_dec_to_input": False,
        "tensor_file": tensor_path.name,
        "tensor_sha256": tensor_digest,
        "corpus_spec": f"provider:{provider_repository}",
        "corpus_sha256": sha256(args.sae_params),
        "tokens_trained": 0,
        "seq_len": 0,
        "batch_size": 0,
        "learning_rate": 0.0,
        "l1_coefficient": 0.0,
        "dead_feature_threshold": 0,
    }
    write_json(pack / "manifest.json", manifest)


def gemma_scope_repository(config: dict) -> str:
    model_name = config.get("model_name")
    prefix = "google/gemma-3-"
    if not isinstance(model_name, str) or not model_name.startswith(prefix):
        raise SystemExit("--sae-provider-repository is required for this provider SAE")
    return f"google/gemma-scope-2-{model_name[len(prefix):]}"


def hook_layer(value) -> int:
    prefix = "model.layers."
    suffix = ".output"
    if not isinstance(value, str) or not value.startswith(prefix) or not value.endswith(suffix):
        raise SystemExit("provider SAE hook point is invalid")
    layer = int(value[len(prefix):-len(suffix)])
    if layer < 0:
        raise SystemExit("provider SAE layer is invalid")
    return layer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runtime-lock", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--runtime-fingerprint", required=True)
    parser.add_argument("--sae-config", required=True, type=Path)
    parser.add_argument("--sae-params", required=True, type=Path)
    parser.add_argument("--sae-provider-repository")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
