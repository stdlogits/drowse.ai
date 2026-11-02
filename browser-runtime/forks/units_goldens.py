"""C TVM VM checks of hook ABI v5 (`block-sites-v5`) against drowse's units goldens.

`verify-mlc-hook.py --units-fixtures DIR` runs these over the MLC fixture models that
`build-units-fixtures.py` writes (`DIR/units-<arch>-<quantization>`). Each model is compiled for
`llvm` through MLC's own pipeline (GPU-scheduled drowse kernels run serially, lane by lane, after
`strip_gpu_thread_bindings`), driven through MLC's paged KV cache exactly as WebLLM drives it, and
compared with the goldens' `library` tolerance:

- `block-sites`: every site (`block_in`, `z`, `attn_o`, `mid`, `act`, `mlp_o`, `out`) at every
  position of the short prompt, captured from `drowse_block` itself, and the last logits;
- `unit-ablation`: every case's prefill and decode forwards through `drowse_units_batch_*` from
  the case's own lowered tables (logits, the last row's `out` at every layer), and the site-level
  forward for the sink rows and `out_last_layer`;
- `unit-probes`: the unit tap's raw value per forward (neuron `act`, head write norm through the
  tap grams), its reading over the neutral scale, and the gate the decode loop saw.

Returns the worst |Δ| per golden so the caller can record it.
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path

import numpy as np

ARCH_FAMILY = {"gemma3": "gemma3_text", "qwen3": "qwen3"}
TERM = re.compile(
    r"^(?:(?P<alpha>[-+]?[0-9.]+)\s+)?!(?P<kind>neuron|head)/L(?P<layer>\d+)[nh](?P<index>\d+)"
    r"(?:@(?P<trigger>.+))?$"
)


def decode(value) -> np.ndarray:
    if isinstance(value, dict) and value.get("encoding") == "float32-le-base64":
        return np.frombuffer(base64.b64decode(value["data"]), "<f4").reshape(value["shape"])
    if isinstance(value, dict) and value.get("encoding") == "int32-le-base64":
        return np.frombuffer(base64.b64decode(value["data"]), "<i4").reshape(value["shape"])
    if isinstance(value, dict) and "values" in value:
        return np.asarray(value["values"], np.float32).reshape(value["shape"])
    return np.asarray(value, np.float32)


class Tolerance:
    def __init__(self, block: dict):
        self.atol = block["library"]["atol"]
        self.rtol = block["library"]["rtol"]
        self.z_atol = block["library"]["z_atol"]
        self.worst = {}

    def check(self, label: str, golden, actual, *, atol: float | None = None, rtol=None) -> None:
        golden = np.asarray(golden, np.float64)
        actual = np.asarray(actual, np.float64)
        if golden.shape != actual.shape:
            raise SystemExit(f"{label}: shape {actual.shape} is not the golden's {golden.shape}")
        a = self.atol if atol is None else atol
        r = self.rtol if rtol is None else rtol
        error = np.abs(actual - golden)
        bound = a + r * np.abs(golden)
        if os.environ.get("DROWSE_UNITS_DEBUG") and error.ndim == 3:
            for layer in range(error.shape[0]):
                print(f"  {label} L{layer}: max |Δ| {float(error[layer].max()):.3g} rows "
                      f"{[int(r) for r in np.nonzero((error[layer] > bound[layer]).any(-1))[0]]}")
        if (not np.all(np.isfinite(actual)) or np.any(error > bound)) and not os.environ.get(
            "DROWSE_UNITS_DEBUG"
        ):
            index = np.unravel_index(int(np.argmax(error - bound)), error.shape)
            raise SystemExit(
                f"{label}: |Δ| {float(error[index]):.3g} at {index} exceeds {float(bound[index]):.3g} "
                f"(golden {float(golden[index]):.6g}, library {float(actual[index]):.6g})"
            )
        key = label.split(":")[0]
        self.worst[key] = max(self.worst.get(key, 0.0), float(error.max(initial=0.0)))


class UnitProgram:
    """`unit_meta`/`unit_coeff`/`unit_active`/`tap_grams` in the HOOK_ABI.md §4 layout."""

    def __init__(self, layout_meta, layout_coeff, num_layers: int, head_dim: int, caps: dict):
        self.meta_layout, self.coeff_layout = layout_meta, layout_coeff
        self.num_layers, self.head_dim, self.caps = num_layers, head_dim, caps
        self.neurons, self.heads, self.neuron_taps, self.head_taps = [], [], [], []
        self.triggers = []

    def group(self, trigger) -> int:
        key = json.dumps(trigger, sort_keys=True)
        keys = [json.dumps(t, sort_keys=True) for t in self.triggers]
        if key not in keys:
            if len(self.triggers) >= self.caps["groups"]:
                raise SystemExit("unit program has more trigger groups than capacity")
            self.triggers.append(trigger)
            keys.append(key)
        return keys.index(key)

    def ablate(self, kind: str, layer: int, index: int, keep: float, bias, trigger) -> None:
        group = self.group(trigger)
        entry = (layer, index, group, np.float32(keep), np.asarray(bias, np.float32))
        (self.neurons if kind == "neuron" else self.heads).append(entry)

    def tap(self, kind: str, layer: int, index: int, gram=None) -> int:
        slot = len(self.neuron_taps) + len(self.head_taps)
        if kind == "neuron":
            self.neuron_taps.append((layer, index, slot))
        else:
            self.head_taps.append((layer, index, slot, gram))
        return slot

    def buffers(self, active: list[bool]):
        meta = np.zeros(self.meta_layout["__length__"][0], np.uint32)
        coeff = np.zeros(self.coeff_layout["__length__"][0], np.float32)
        grams = np.zeros((self.caps["head_taps"], self.head_dim, self.head_dim), np.float32)

        def csr(name, entries, columns):
            offsets = self.meta_layout[name + "_offsets"][0]
            ordered = sorted(enumerate(entries), key=lambda item: (item[1][0], item[0]))
            for layer in range(self.num_layers + 1):
                meta[offsets + layer] = sum(1 for _, e in ordered if e[0] < layer)
            return [e for _, e in ordered], columns

        neurons, _ = csr("neuron", self.neurons, None)
        for i, (layer, index, group, keep, bias) in enumerate(neurons):
            meta[self.meta_layout["neuron_index"][0] + i] = index
            meta[self.meta_layout["neuron_group"][0] + i] = group
            coeff[self.coeff_layout["neuron_keep"][0] + i] = keep
            coeff[self.coeff_layout["neuron_bias"][0] + i] = bias
        heads, _ = csr("head", self.heads, None)
        for i, (layer, index, group, keep, bias) in enumerate(heads):
            meta[self.meta_layout["head_index"][0] + i] = index
            meta[self.meta_layout["head_group"][0] + i] = group
            coeff[self.coeff_layout["head_keep"][0] + i] = keep
            start = self.coeff_layout["head_bias"][0] + i * self.head_dim
            coeff[start : start + self.head_dim] = bias
        taps, _ = csr("tap_neuron", self.neuron_taps, None)
        for i, (layer, index, slot) in enumerate(taps):
            meta[self.meta_layout["tap_neuron_index"][0] + i] = index
            meta[self.meta_layout["tap_neuron_slot"][0] + i] = slot
        taps, _ = csr("tap_head", self.head_taps, None)
        for i, (layer, index, slot, gram) in enumerate(taps):
            meta[self.meta_layout["tap_head_index"][0] + i] = index
            meta[self.meta_layout["tap_head_slot"][0] + i] = slot
            grams[i] = gram
        unit_active = np.asarray([1.0 if flag else 0.0 for flag in active]
                                 + [0.0] * (self.caps["groups"] - len(active)), np.float32)
        return meta, coeff, unit_active, grams


def trigger_active(trigger: dict | None, forward: int, gate_score) -> bool:
    """Native `Trigger.active` (drowse/steering/triggers.py) for a raw generation: forward 0 is the
    prefill, forward k ≥ 1 is a decode forward with `gen_step = k` (drowse/loom/replay.py: the
    forward that reads the first generated token is step 1) in the response region."""
    if trigger is None:
        return True
    if forward == 0:
        return trigger.get("gate") is None and trigger.get("prompt", True)
    if not trigger.get("generated", True) or not trigger.get("response", True):
        return False
    gen_step = forward
    if trigger.get("first_n") is not None and gen_step >= trigger["first_n"]:
        return False
    if trigger.get("after_n") is not None and gen_step < trigger["after_n"]:
        return False
    gate = trigger.get("gate")
    if gate is not None:
        if gate_score is None:
            return False
        op, threshold = gate["op"], gate["threshold"]
        return {">": gate_score > threshold, ">=": gate_score >= threshold,
                "<": gate_score < threshold, "<=": gate_score <= threshold}[op]
    return True


def parse_trigger(text: str | None) -> dict | None:
    """The browser-relevant subset of native triggers used by the units goldens."""
    if text is None:
        return None
    if text == "prompt":
        return {"prompt": True, "generated": False}
    if text == "response":
        return {"prompt": False, "generated": True, "response": True}
    if text.startswith("first:"):
        return {"prompt": False, "first_n": int(text.split(":")[1])}
    match = re.match(r"^when:(\S+)\s*(>=|<=|>|<)\s*(\S+)$", text)
    if match:
        return {"gate": {"probe": match.group(1), "op": match.group(2),
                         "threshold": float(match.group(3))}}
    raise SystemExit(f"unsupported trigger in a units golden: {text}")


class FixtureVM:
    """One fixture model compiled for llvm, its parameters, and a paged KV cache."""

    def __init__(self, env, model_dir: Path):
        self.env = env
        tvm, relax, nn = env.tvm, env.relax, env.nn
        chat = json.loads((model_dir / "mlc-chat-config.json").read_text())
        fixture = json.loads((model_dir / "drowse-units-fixture.json").read_text())
        self.arch, self.quantization = fixture["arch"], fixture["quantization"]
        config_class, model_class = env.models[chat["model_type"]]
        config = config_class.from_dict(chat["model_config"])
        quantization = env.QUANTIZATION[self.quantization]
        target = tvm.target.Target({"kind": "llvm"})
        with target:
            model, _ = env.make_quantization_functions(model_class)[quantization.kind](
                config, quantization
            )
            spec = self._spec(model, nn)
            module, named_params, ext_mods = model.export_tvm(spec=spec, allow_extern=True)
        module = strip_scheduled_kernels(module, tvm, env.strip_gpu_thread_bindings)
        self.block = model._drowse_block_spec()
        self.num_layers = model.num_hidden_layers
        self.hidden_size = model.hidden_size
        self.vocab = chat["vocab_size"]
        env.prepare_parameters(named_params)
        metadata = env.create_metadata(chat["model_type"], config, named_params, self.quantization)
        pipeline = relax.get_pipeline(
            "mlc_llm", target=target, flashinfer=False, cublas_gemm=False,
            faster_transformer=False, allreduce_strategy=env.IPCAllReduceStrategyType.NONE,
            variable_bounds={"total_seq_len": config.context_window_size,
                             "seq_len": config.prefill_chunk_size, "batch_size": 1,
                             "jlens_layers": self.num_layers, "sae_features": 16384},
            cuda_graph_symbolic_capture_hints={}, additional_tirs={}, ext_mods=ext_mods,
            metadata=metadata, debug_dump=None,
        )
        with tvm.transform.PassContext(config={"tirx.disable_cse_tir": True}):
            executable = relax.build(module, target=target, relax_pipeline=pipeline)
        self.vm = relax.VirtualMachine(executable, tvm.cpu())
        tensors = load_tensor_cache(model_dir)
        self.params = [tvm.runtime.tensor(tensors[name], device=tvm.cpu()) for name, _ in named_params]
        self.curve_zeros = {
            name: np.zeros([int(d) for d in t.shape], t.dtype)
            for name, t in env.structured_curve_program_spec(self.num_layers, self.hidden_size).items()
        }
        self.context = config.context_window_size
        self.add_sequence = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
        self.begin_forward = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
        self.end_forward = tvm.get_global_func("vm.builtin.kv_state_end_forward")

    def _spec(self, model, nn):
        env = self.env
        from mlc_llm.nn import PagedKVCache

        d = model.hidden_size
        head_dim = model._drowse_block_spec()["head_dim"]
        curved = env.structured_curve_program_spec(model.num_hidden_layers, d)
        units = env.unit_program_spec(model.num_hidden_layers, head_dim)
        packed = {"param_mode": "packed", "effect_mode": "none"}
        install_block_sites_forward(type(model), env)
        raw = {
            "embed": {"input_ids": nn.spec.Tensor(["seq_len"], "int32"), "$": packed},
            "create_paged_kv_cache": {
                "max_batch_size": int, "max_total_seq_len": int, "prefill_chunk_size": int,
                "page_size": int, "support_sliding_window": int,
                "$": {"param_mode": "none", "effect_mode": "none"},
            },
            "drowse_units_batch_prefill": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", d], "float32"),
                "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                **curved, **units, "$": packed,
            },
            "drowse_units_batch_decode": {
                "input_embeds": nn.spec.Tensor(["batch_size", 1, d], "float32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                **curved, **units, "$": packed,
            },
            "drowse_verify_block_sites": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", d], "float32"),
                "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                **units, "$": packed,
            },
        }
        return nn.spec.ModuleSpec.from_raw(raw, model)

    def new_cache(self):
        shape = self.env.tvm.runtime.ShapeTuple
        cache = self.vm["create_tir_paged_kv_cache"](
            shape([1]), shape([self.context]), shape([self.context]), shape([16]), shape([0])
        )
        self.add_sequence(cache, 0)
        return cache

    def embed(self, ids):
        tvm = self.env.tvm
        rows = self.vm["embed"](tvm.runtime.tensor(np.asarray(ids, np.int32)), self.params)
        return rows.numpy().reshape(1, len(ids), self.hidden_size).astype(np.float32)

    def forward_units(self, cache, ids, program_buffers, *, prefill: bool, affine=None):
        tvm = self.env.tvm
        meta, coeff, active, grams = program_buffers
        embeds = self.embed(ids)
        self.begin_forward(cache, tvm.runtime.ShapeTuple([0]), tvm.runtime.ShapeTuple([len(ids)]))
        curve = [tvm.runtime.tensor((affine or {}).get(name, value))
                 for name, value in self.curve_zeros.items()]
        units = [tvm.runtime.tensor(v) for v in (meta, coeff, active, grams)]
        if prefill:
            out = self.vm["drowse_units_batch_prefill"](
                tvm.runtime.tensor(embeds), tvm.runtime.tensor(np.asarray([len(ids) - 1], np.int32)),
                cache, *curve, *units, self.params,
            )
        else:
            out = self.vm["drowse_units_batch_decode"](
                tvm.runtime.tensor(embeds), cache, *curve, *units, self.params
            )
        self.end_forward(cache)
        logits = out[0].numpy().reshape(-1)[: self.vocab]
        return logits, out[4].numpy(), out[5].numpy()

    def forward_sites(self, cache, ids, program_buffers):
        tvm = self.env.tvm
        meta, coeff, active, grams = program_buffers
        embeds = self.embed(ids)
        self.begin_forward(cache, tvm.runtime.ShapeTuple([0]), tvm.runtime.ShapeTuple([len(ids)]))
        out = self.vm["drowse_verify_block_sites"](
            tvm.runtime.tensor(embeds), tvm.runtime.tensor(np.asarray([len(ids) - 1], np.int32)), cache,
            *[tvm.runtime.tensor(v) for v in (meta, coeff, active, grams)], self.params,
        )
        self.end_forward(cache)
        names = ("block_in", "z", "attn_o", "mid", "act", "mlp_o", "out")
        sites = {name: np.stack([out[1 + layer * 7 + k].numpy()[0] for layer in range(self.num_layers)])
                 for k, name in enumerate(names)}
        return out[0].numpy().reshape(-1)[: self.vocab], sites


def strip_scheduled_kernels(module, tvm, strip):
    """Serialize only the kernels that carry GPU thread bindings (the drowse kernels); the
    upstream graph's unscheduled kernels go through MLC's llvm pipeline untouched."""
    tir = tvm.tirx
    for var in list(module.get_global_vars()):
        function = module[var]
        if not isinstance(function, tir.PrimFunc):
            continue
        bound = []
        tir.stmt_functor.post_order_visit(
            function.body,
            lambda node: bound.append(node)
            if isinstance(node, tir.For) and node.kind == tir.ForKind.THREAD_BINDING else None,
        )
        if bound:
            module.update_func(var, strip(tvm.IRModule({var: function}), tvm)[var])
    return module


def install_block_sites_forward(model_class, env) -> None:
    """A verification-only entry point: one forward through `drowse_block` with unit sites that
    also capture every site tensor (never compiled into a library)."""
    if getattr(model_class, "drowse_verify_block_sites", None) is not None:
        return
    UnitSites, op = env.UnitSites, env.op

    class CaptureSites(UnitSites):
        def __init__(self, *args):
            super().__init__(*args)
            self.rows = {}

        def head(self, z, layer_id):
            z = super().head(z, layer_id)
            self.rows[(layer_id, "z")] = z
            return z

        def mid(self, h_mid, layer_id):
            self.rows[(layer_id, "mid")] = h_mid

        def neuron(self, act, layer_id):
            act = super().neuron(act, layer_id)
            self.rows[(layer_id, "act")] = act
            return act

        def tap(self, z, attn_o, act, layer_id):
            super().tap(z, attn_o, act, layer_id)
            self.rows[(layer_id, "attn_o")] = attn_o

        def mlp(self, mlp_o, layer_id):
            self.rows[(layer_id, "mlp_o")] = mlp_o

    def drowse_verify_block_sites(self, input_embeds, logit_positions, paged_kv_cache, unit_meta,
                                  unit_coeff, unit_active, tap_grams):
        spec = self._drowse_block_spec()
        b, s, _ = input_embeds.shape
        sites = CaptureSites(self.num_hidden_layers, spec["head_dim"],
                             paged_kv_cache.get_query_positions(b * s), unit_meta, unit_coeff,
                             unit_active, tap_grams, spec["sandwich"], spec["rms_eps"])
        hidden = self.model._drowse_prepare_inputs(input_embeds)
        captured = []
        for layer_id, layer in enumerate(self.model.layers):
            block_in = hidden
            hidden = layer.drowse_block(hidden, paged_kv_cache, layer_id, sites)
            captured += [block_in, sites.rows[(layer_id, "z")], sites.rows[(layer_id, "attn_o")],
                         sites.rows[(layer_id, "mid")], sites.rows[(layer_id, "act")],
                         sites.rows[(layer_id, "mlp_o")], hidden]
        last = op.take(self.model.norm(hidden), logit_positions, axis=1)
        return (self._get_logits(last), *captured)

    model_class.drowse_verify_block_sites = drowse_verify_block_sites


def load_tensor_cache(directory: Path) -> dict[str, np.ndarray]:
    cache = json.loads((directory / "tensor-cache.json").read_text())
    tensors = {}
    for shard in cache["records"]:
        data = (directory / shard["dataPath"]).read_bytes()
        for record in shard["records"]:
            chunk = data[record["byteOffset"]: record["byteOffset"] + record["nbytes"]]
            tensors[record["name"]] = np.frombuffer(chunk, dtype=record["dtype"]).reshape(
                record["shape"]).copy()
    return tensors


def head_gram(tensors: dict, block: dict, layer: int, head: int) -> np.ndarray:
    """`G_h = (γ_ao ⊙ W_O^h)ᵀ (γ_ao ⊙ W_O^h)` from the converted (dequantized) tensors, fp64."""
    hd = block["head_dim"]
    suffix = f"layers.{layer}.self_attn.o_proj.weight"
    prefix = next(name for name in tensors if name.endswith(suffix) or name.endswith(
        suffix.replace(".weight", ".q_weight")))
    prefix = prefix.rsplit(".", 1)[0]
    if prefix + ".weight" in tensors:
        w = tensors[prefix + ".weight"].astype(np.float64)
    else:
        weight, scale = tensors[prefix + ".q_weight"], tensors[prefix + ".q_scale"]
        shifts = np.arange(8, dtype=np.uint32) * 4
        codes = ((weight[:, :, None] >> shifts) & 0xF).reshape(weight.shape[0], -1).astype(np.float64)
        w = (codes - 7) * np.repeat(scale.astype(np.float64), 32, axis=1)
    columns = w[:, head * hd: (head + 1) * hd]
    if block["sandwich"]:
        gamma = tensors[prefix.replace("self_attn.o_proj", "post_attention_layernorm")
                        + ".weight"].astype(np.float64)
        columns = gamma[:, None] * columns
    return (columns.T @ columns).astype(np.float32)


def unit_stats(goldens: Path, arch: str) -> dict:
    stats = json.loads((goldens / f"unit-stats-{arch}-v1.json").read_text())
    return {
        "neuron_mean": decode(stats["neuron_mean"]),
        "head_mean": decode(stats["head_mean"]),
        "neuron_scale": decode(stats["scales"]["neuron"]),
        "head_scale": decode(stats["scales"]["head"]),
    }


def reading(kind: str, layer: int, index: int, raw: float, stats: dict) -> float:
    if kind == "neuron":
        scale = stats["neuron_scale"][layer, 0 if raw >= 0 else 1, index]
    else:
        scale = stats["head_scale"][layer, index]
    return float(np.float32(raw) / np.float32(scale))


def verify_arch(env, fixtures: Path, goldens: Path, arch: str, quantization: str) -> dict:
    model_dir = fixtures / f"units-{arch}-{quantization}"
    vm = FixtureVM(env, model_dir)
    tensors = load_tensor_cache(model_dir)
    stats = unit_stats(goldens, arch)
    block = vm.block
    caps = {"groups": env.UNIT_MAX_GROUPS, "head_taps": env.UNIT_MAX_HEAD_TAPS}

    def program():
        return UnitProgram(env.unit_meta_layout(vm.num_layers), env.unit_coeff_layout(block["head_dim"]),
                           vm.num_layers, block["head_dim"], caps)

    # block-sites: the unablated forward, every site at every position.
    sites_golden = json.loads((goldens / f"block-sites-{arch}-v1.json").read_text())
    tolerance = Tolerance(sites_golden["tolerance"])
    empty = program().buffers([])
    logits, sites = vm.forward_sites(vm.new_cache(), sites_golden["ids"], empty)
    for name, value in sites_golden["sites"].items():
        tolerance.check(f"block-sites:{name}", decode(value), sites[name])
    tolerance.check("block-sites:logits", decode(sites_golden["logits_last"]), logits)

    # The units entry points with an empty program agree with the site forward.
    unit_logits, jlens, taps = vm.forward_units(vm.new_cache(), sites_golden["ids"], empty, prefill=True)
    tolerance.check("block-sites:units-prefill-logits", decode(sites_golden["logits_last"]), unit_logits)
    tolerance.check("block-sites:units-prefill-out", decode(sites_golden["sites"]["out"])[:, -1], jlens)

    # unit-ablation: every case from its own lowered tables.
    ablation = json.loads((goldens / f"unit-ablation-{arch}-v1.json").read_text())
    push = json.loads((goldens / f"neuron-push-{arch}-v1.json").read_text())
    skipped = []
    for case in ablation["cases"]:
        affine = None
        if case["name"] == "mixed":
            # Its fixture direction lowers through the host's structured compiler, which the
            # files do not record; the WebGPU harness covers it end to end.
            skipped.append(case["name"])
            continue
        if case["name"] == "push-and-ablate":
            affine = push_affine(vm, push, "0.3 neuron/L2n7")
        unit = program()
        for table in case["tables"]:
            trigger = dict(table["trigger"])
            entries = list(table["entries"])
            if table["site"] == "act":
                for coordinate, keep, bias in zip(entries, table["keep"], table["bias"]):
                    unit.ablate("neuron", table["layer"], coordinate, keep, bias, trigger)
            else:
                hd = block["head_dim"]
                for start in range(0, len(entries), hd):
                    span = entries[start: start + hd]
                    head = span[0] // hd
                    if span != list(range(head * hd, (head + 1) * hd)):
                        raise SystemExit(f"{case['name']}: a z table is not whole heads")
                    keeps = set(table["keep"][start: start + hd])
                    if len(keeps) != 1:
                        raise SystemExit(f"{case['name']}: a head's keep is not uniform")
                    unit.ablate("head", table["layer"], head, keeps.pop(),
                                table["bias"][start: start + hd], trigger)
        gates = {t["gate"]["probe"] for t in unit.triggers if t.get("gate")}
        gate_slots = {}
        for probe in sorted(gates):
            kind, layer, index = parse_unit(probe)
            gram = head_gram(tensors, block, layer, index) if kind == "head" else None
            gate_slots[probe] = (kind, layer, index, unit.tap(kind, layer, index, gram))
        cache = vm.new_cache()
        scores = {}
        for number, forward in enumerate(case["forwards"]):
            active = [trigger_active(t, number, scores.get(((t or {}).get("gate") or {}).get("probe")))
                      for t in unit.triggers]
            buffers = unit.buffers(active)
            label = f"unit-ablation:{case['name']}[{number}]"
            if number == 0 and affine is None:
                site_logits, site = vm.forward_sites(vm.new_cache(), forward["ids"], buffers)
                tolerance.check(label + ".logits(sites)", decode(forward["logits"]), site_logits)
                golden_out = decode(forward["out"])
                tolerance.check(label + ".out", golden_out, site["out"][:, forward["out_rows"]])
                tolerance.check(label + ".out_last_layer", decode(case["out_last_layer"]),
                                site["out"][-1])
            logits, jlens, taps = vm.forward_units(cache, forward["ids"], buffers,
                                                   prefill=number == 0, affine=affine)
            tolerance.check(label + ".logits", decode(forward["logits"]), logits)
            last_row = forward["out_rows"].index(len(forward["ids"]) - 1)
            expected = decode(forward["out"])[:, last_row].astype(np.float32)
            if affine is not None:
                # `out` is the block output; the units pair returns the residual after the
                # post-block hook, which adds the push's delta at the layers it names.
                active = affine["affine_active"][:, 0] != 0
                expected = expected + np.where(active[:, None], affine["affine_along"][:, 0, None]
                                               * affine["affine_target"][:, 0, 0, None]
                                               * affine["affine_basis"][:, 0, 0], 0).astype(np.float32)
            tolerance.check(label + ".out_last", expected, jlens)
            scores = {probe: reading(kind, layer, index, float(taps[slot]), stats)
                      for probe, (kind, layer, index, slot) in gate_slots.items()}

    # unit-probes: the live tap through the units pair.
    probes = json.loads((goldens / f"unit-probes-{arch}-v1.json").read_text())
    for case in probes["cases"]:
        unit = program()
        for term in split_terms(case["steering"]):
            kind, layer, index, alpha, trigger = term
            mean = (stats["neuron_mean"][layer, index] if kind == "neuron"
                    else stats["head_mean"][layer, index])
            unit.ablate(kind, layer, index, np.float32(1 - alpha),
                        (np.float32(alpha) * np.asarray(mean, np.float32)), trigger)
        names = list(probes["probes"]) + ([probes["gate_unit"]] if case["gate_scalars"] else [])
        slots = {}
        for name in names:
            kind, layer, index = parse_unit(name)
            gram = head_gram(tensors, block, layer, index) if kind == "head" else None
            slots[name] = (kind, layer, index, unit.tap(kind, layer, index, gram))
        cache = vm.new_cache()
        ids = list(probes["ids"])
        scores = {}
        for number, forward in enumerate(case["forwards"]):
            active = [trigger_active(t, number, scores.get(((t or {}).get("gate") or {}).get("probe")))
                      for t in unit.triggers]
            step_ids = ids if number == 0 else [case["tokens"][number - 1]]
            logits, _, taps = vm.forward_units(cache, step_ids, unit.buffers(active),
                                               prefill=number == 0)
            if int(np.argmax(logits)) != case["tokens"][number]:
                raise SystemExit(f"unit-probes:{case['name']}[{number}] greedy token differs")
            label = f"unit-probes:{case['name']}[{number}]"
            scores = {}
            for name, (kind, layer, index, slot) in slots.items():
                raw = float(taps[slot])
                golden = forward["units"][name]
                tolerance.check(label + f".{name}.raw", golden["raw"], raw)
                value = reading(kind, layer, index, raw, stats)
                tolerance.check(label + f".{name}.value", golden["value"], value,
                                atol=tolerance.z_atol, rtol=0.0)
                scores[name] = value
            if case["gate_scalars"]:
                gate = probes["gate_unit"]
                tolerance.check(label + ".gate", case["gate_scalars"][number][gate], scores[gate],
                                atol=tolerance.z_atol, rtol=0.0)
    tolerance.worst["skipped"] = skipped
    return tolerance.worst


def push_affine(vm, push: dict, expression: str) -> dict:
    """The structured affine buffers of a neuron push from neuron-push's recorded lowering
    (`delta = along · target · basis`, collapse 0), in group 0 at every layer it names."""
    case = next(c for c in push["cases"] if c["expression"] == expression)
    if case["trigger"]["gate"] is not None or not case["trigger"]["prompt"]:
        raise SystemExit("push_affine handles always-on pushes only")
    affine = {name: value.copy() for name, value in vm.curve_zeros.items()}
    for layer in case["layers"]:
        index = layer["layer"]
        affine["affine_active"][index, 0] = 1.0
        affine["affine_basis"][index, 0, 0] = np.asarray(layer["basis"], np.float32)
        affine["affine_neutral"][index, 0] = np.asarray(layer["neutral"], np.float32)
        affine["affine_target"][index, 0, 0] = np.float32(layer["target"])
        affine["affine_along"][index, 0] = np.float32(layer["along"])
    return affine


def parse_unit(name: str):
    match = re.match(r"^(neuron|head)/L(\d+)[nh](\d+)$", name)
    if not match:
        raise SystemExit(f"not a unit name: {name}")
    return match.group(1), int(match.group(2)), int(match.group(3))


def split_terms(steering: str | None):
    if steering is None:
        return []
    terms = []
    for text in steering.split(" + "):
        match = TERM.match(text.strip())
        if not match:
            raise SystemExit(f"unsupported steering term in a units golden: {text}")
        alpha = float(match.group("alpha")) if match.group("alpha") else 1.0
        terms.append((match.group("kind"), int(match.group("layer")), int(match.group("index")),
                      alpha, parse_trigger(match.group("trigger"))))
    return terms


def verify(env, fixtures: Path, goldens: Path) -> dict:
    results = {}
    for arch in ("qwen3", "gemma3"):
        for quantization in ("q0f32", "q4f32_1"):
            if not (fixtures / f"units-{arch}-{quantization}").is_dir():
                raise SystemExit(f"missing fixture model units-{arch}-{quantization} in {fixtures}")
            results[f"{arch}-{quantization}"] = verify_arch(env, fixtures, goldens, arch, quantization)
            worst = results[f"{arch}-{quantization}"]
            print(f"units goldens {arch} {quantization}: "
                  + ", ".join(f"{k} {v:.2e}" for k, v in sorted(worst.items()) if k != "skipped")
                  + f"; skipped {worst['skipped']}", flush=True)
    return results
