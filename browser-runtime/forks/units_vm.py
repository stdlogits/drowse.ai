"""A units model (hook ABI v6, `block-sites-v6`) on the C TVM VM, for verify-production-library.py.

`UnitsVM` compiles a converted MLC model for `llvm` through MLC's own pipeline (GPU-scheduled
drowse kernels run serially, lane by lane, after `strip_gpu_thread_bindings`) and drives it
through MLC's paged KV cache exactly as WebLLM drives it. `UnitProgram` packs `unit_meta`,
`unit_coeff`, `unit_active` and `tap_grams` in the HOOK_ABI.md §4 layout. `Weights` dequantizes
the tensor cache to fp64 and `Device` calls the library's static reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


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


class UnitsVM:
    """One converted units model compiled for llvm, its parameters, and a paged KV cache."""

    def __init__(self, env, model_dir: Path):
        self.env = env
        tvm, relax, nn = env.tvm, env.relax, env.nn
        chat = json.loads((model_dir / "mlc-chat-config.json").read_text())
        self.arch, self.quantization = chat["model_type"], chat["quantization"]
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
        block = model._drowse_block_spec()
        replay = env.drowse_units.replay_program_spec(model.num_hidden_layers, block)
        rows = env.drowse_units.attention_rows_spec(block)
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
            "drowse_replay_batch_prefill": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", d], "float32"),
                "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                **curved, **units, **replay, "$": packed,
            },
            "drowse_attention_rows": {**rows, "$": {"param_mode": "none", "effect_mode": "none"}},
            "drowse_head_write_grams": {"$": packed},
            "drowse_verify_block_sites": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", d], "float32"),
                "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                **units, "$": packed,
            },
            **env.drowse_reads.readout_spec(model),
            **env.drowse_reads.unit_read_spec(model),
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


    # ------------------------------------------------------------------ replay (B1 §2.3, §6) ---

    def head_grams(self) -> np.ndarray:
        return self.vm["drowse_head_write_grams"](self.params).numpy()

    def replay_carry(self, rows: int, tokens: int) -> dict:
        """A fresh replay's threaded buffers: activity state (value 0, row −1, rank −1), traces
        `[32, rows]`, population and pattern keys `[8, KV, tokens, hd]` (sized to the replay, as
        the bridge sizes them)."""
        tvm, du = self.env.tvm, self.env.drowse_units
        block, L = self.block, self.num_layers
        f, h = block["d_ff"], block["n_heads"]
        state = np.zeros((L, 3 * f + 3 * h), np.float32)
        state[:, f: 3 * f] = -1
        state[:, 3 * f + h:] = -1
        arrays = {
            "state": state,
            "traces": np.zeros((du.UNIT_MAX_ACTIVITY_TRACES, rows), np.float32),
            "population": np.zeros((L, f + h), np.float32),
            "keys": np.zeros((du.UNIT_MAX_PATTERN_LAYERS, block["n_kv_heads"], tokens,
                              block["head_dim"]), np.float32),
        }
        return {name: tvm.runtime.tensor(value) for name, value in arrays.items()}

    def replay_forward(self, cache, ids, program_buffers, meta, scales, grams, carry, affine=None):
        """One `drowse_replay_batch_prefill` forward continuing `cache` (positions follow the
        cache's length); the threaded buffers come back in `carry`."""
        tvm = self.env.tvm
        embeds = self.embed(ids)
        self.begin_forward(cache, tvm.runtime.ShapeTuple([0]), tvm.runtime.ShapeTuple([len(ids)]))
        curve = [tvm.runtime.tensor((affine or {}).get(name, value))
                 for name, value in self.curve_zeros.items()]
        units = [tvm.runtime.tensor(v) for v in program_buffers]
        out = self.vm["drowse_replay_batch_prefill"](
            tvm.runtime.tensor(embeds), tvm.runtime.tensor(np.asarray([len(ids) - 1], np.int32)),
            cache, *curve, *units, tvm.runtime.tensor(meta), scales, grams,
            carry["state"], carry["traces"], carry["population"], carry["keys"], self.params,
        )
        self.end_forward(cache)
        return {
            "logits": out[0].numpy().reshape(-1)[: self.vocab],
            "measurements": out[2].numpy(),
            "jlens": out[4].numpy(),
            "taps": out[5].numpy(),
            "anatomy": out[6].numpy(),
            "pattern_q": out[11],
            "carry": {"state": out[7], "traces": out[8], "population": out[9], "keys": out[10]},
        }

    def attention_rows(self, keys, queries, rows_meta: np.ndarray) -> np.ndarray:
        tvm = self.env.tvm
        return self.vm["drowse_attention_rows"](keys, queries, tvm.runtime.tensor(rows_meta)).numpy()


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
    """Every tensor as the runtime decodes it: `f32-to-bf16` records (production q4f32_1 scales
    and norms) widen their stored bf16 bits to float32."""
    cache = json.loads((directory / "tensor-cache.json").read_text())
    tensors = {}
    for shard in cache["records"]:
        data = (directory / shard["dataPath"]).read_bytes()
        for record in shard["records"]:
            chunk = data[record["byteOffset"]: record["byteOffset"] + record["nbytes"]]
            encoding = record.get("format", "raw")
            if encoding not in ("raw", "f32-to-bf16"):
                raise SystemExit(f"{record['name']}: unsupported tensor encoding {encoding}")
            if encoding == "f32-to-bf16" and record["dtype"] == "float32":
                bits = np.frombuffer(chunk, dtype="<u2").astype(np.uint32) << 16
                value = bits.view(np.float32)
            else:  # the encoding narrows float32 records only (mlc_q4_torch.MlcTensorCache)
                value = np.frombuffer(chunk, dtype=record["dtype"]).copy()
            tensors[record["name"]] = value.reshape(record["shape"])
    return tensors


class Weights:
    """A units library's weights in fp64, dequantized from the tensor cache."""

    def __init__(self, tensors: dict, block: dict, num_layers: int):
        self.tensors, self.block, self.num_layers = tensors, block, num_layers
        self.prefix = "language_model.model." if any(
            n.startswith("language_model.") for n in tensors) else "model."

    def get(self, name: str) -> np.ndarray:
        name = self.prefix + name
        if name + ".weight" in self.tensors:
            return self.tensors[name + ".weight"].astype(np.float64)
        weight, scale = self.tensors[name + ".q_weight"], self.tensors[name + ".q_scale"]
        shifts = np.arange(8, dtype=np.uint32) * 4
        codes = ((weight[:, :, None] >> shifts) & 0xF).reshape(weight.shape[0], -1).astype(np.float64)
        return (codes - 7) * np.repeat(scale.astype(np.float64), 32, axis=1)

    def gain(self, layer: int, which: str) -> np.ndarray:
        sandwich = self.block["sandwich"]
        names = {"ai": "input_layernorm",
                 "ao": "post_attention_layernorm" if sandwich else None,
                 "mi": "pre_feedforward_layernorm" if sandwich else "post_attention_layernorm",
                 "mo": "post_feedforward_layernorm" if sandwich else None}
        if names[which] is None:
            return np.ones(self.get("embed_tokens").shape[1])
        return self.get(f"layers.{layer}.{names[which]}")

    def layer(self, layer: int) -> dict:
        b, p = self.block, f"layers.{layer}."
        hd, heads, kv, d_ff = b["head_dim"], b["n_heads"], b["n_kv_heads"], b["d_ff"]
        if self.prefix + p + "self_attn.c_attn.weight" in self.tensors or \
                self.prefix + p + "self_attn.c_attn.q_weight" in self.tensors:
            qkv = self.get(p + "self_attn.c_attn")
            q, k, v = qkv[: heads * hd], qkv[heads * hd: (heads + kv) * hd], qkv[(heads + kv) * hd:]
        else:
            q, k, v = (self.get(p + f"self_attn.{r}_proj") for r in "qkv")
        gate_up = self.get(p + "mlp.gate_up_proj")
        return {"q": q, "k": k, "v": v, "o": self.get(p + "self_attn.o_proj"),
                "gate": gate_up[:d_ff], "up": gate_up[d_ff:], "down": self.get(p + "mlp.down_proj")}

    def writes(self, layer: int) -> np.ndarray:
        """`[d_ff, d]`: `s·γ_mo ⊙ W_down[:, n]` per neuron."""
        scale = float(self.block.get("residual_scale", 1.0))
        return (scale * self.gain(layer, "mo")[:, None] * self.layer(layer)["down"]).T

    def reads(self, layer: int):
        gain = self.gain(layer, "mi")
        w = self.layer(layer)
        a, b = w["gate"] * gain, w["up"] * gain
        return (a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12),
                b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12))

    def head_write(self, layer: int, head: int) -> np.ndarray:
        hd = self.block["head_dim"]
        return self.gain(layer, "ao")[:, None] * self.layer(layer)["o"][:, head * hd:(head + 1) * hd]

    @property
    def embedding(self) -> np.ndarray:
        return self.get("embed_tokens")

    @property
    def final_gain(self) -> np.ndarray:
        return self.get("norm")


class Device:
    """Calls into a units library's static reads (fp32 in, numpy out)."""

    def __init__(self, vm):
        self.vm, self.tvm = vm, vm.env.tvm

    def __call__(self, name: str, *args, packed: bool = True):
        arrays = [self.tvm.runtime.tensor(np.ascontiguousarray(a)) for a in args]
        out = self.vm.vm[name](*arrays, self.vm.params) if packed else self.vm.vm[name](*arrays)
        if hasattr(out, "numpy"):
            return out.numpy()
        return [o.numpy() for o in out]

    def topk(self, x: np.ndarray, k: int):
        n = x.shape[0]
        values, indices = self("drowse_topk", x.astype(np.float32), np.zeros((n, k), np.float32),
                               np.zeros((n, k), np.int32), packed=False)
        return values, indices

    def chan(self, rows: np.ndarray, chunk: int = 37):
        """Mean and centred sum of squares of `rows` through repeated merges (chunked, so the
        merge rule itself is exercised)."""
        width = rows.shape[1]
        count, mean, m2 = 0, np.zeros(width, np.float32), np.zeros(width, np.float32)
        for start in range(0, rows.shape[0], chunk):
            part = rows[start:start + chunk].astype(np.float32)
            mean, m2 = self("drowse_chan_merge", np.array([count], np.float32), mean, m2, part,
                            packed=False)
            count += part.shape[0]
        return count, mean, m2
