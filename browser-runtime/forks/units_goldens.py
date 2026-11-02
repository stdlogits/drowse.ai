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
  tap grams), its reading over the neutral scale, and the gate the decode loop saw;
- replay (`drowse_replay_batch_prefill`, fed in forwards of at most 16 rows on one KV sequence):
  `attention-patterns` (every head's row at positions 2, 5, 8 through `drowse_attention_rows`),
  `token-anatomy` (the taps at the producing position, every head's write norm from the captured
  `z` and `attn_o`, the MLP write, the chosen heads' rows) and `unit-activity` (every unit's device
  peak, row and rank against the diagnostics, the host ranking, traces, the float16 population),
  with `head_grams` built by `drowse_head_write_grams`.

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
    if text in ("prompt", "before"):
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
    """One units model (a build-units-fixtures.py fixture or a converted production model)
    compiled for llvm, its parameters, and a paged KV cache."""

    def __init__(self, env, model_dir: Path):
        self.env = env
        tvm, relax, nn = env.tvm, env.relax, env.nn
        chat = json.loads((model_dir / "mlc-chat-config.json").read_text())
        fixture_path = model_dir / "drowse-units-fixture.json"
        fixture = (json.loads(fixture_path.read_text()) if fixture_path.is_file()
                   else {"arch": chat["model_type"], "quantization": chat["quantization"]})
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
            if encoding == "f32-to-bf16" and record["dtype"] == "float32":
                bits = np.frombuffer(chunk, dtype="<u2").astype(np.uint32) << 16
                value = bits.view(np.float32)
            elif encoding == "raw":
                value = np.frombuffer(chunk, dtype=record["dtype"]).copy()
            else:
                raise SystemExit(f"{record['name']}: unsupported tensor encoding {encoding}")
            tensors[record["name"]] = value.reshape(record["shape"])
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
    mixed = mixed_affine(ablation, arch)
    for case in ablation["cases"]:
        affine = None
        if case["name"] == "mixed":
            affine = push_affine(vm, mixed, case["steering"].split(" + ")[0])
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
    verify_replay(env, vm, tensors, stats, goldens, arch, program, tolerance.worst)
    if not os.environ.get("DROWSE_UNITS_SKIP_READS"):
        import static_reads

        static_reads.verify(env, vm, tensors, goldens, arch, tolerance.check)
    return tolerance.worst


MIXED_AFFINE = Path(__file__).resolve().parents[1] / "fixtures" / "unit-ablation-mixed-affine-v1.json"


def mixed_affine(ablation: dict, arch: str) -> dict:
    """The `mixed` case's push lowered to affine groups (browser-runtime/fixtures, recorded by
    build-mixed-affine.py from drowse's own lowering), bound to the goldens' model file."""
    doc = json.loads(MIXED_AFFINE.read_text())
    if doc["fixtureModels"]["sha256"] != ablation["fixtureModels"]["sha256"]:
        raise SystemExit(f"{MIXED_AFFINE.name} names another fixture model file: regenerate it")
    return doc["arches"][arch]


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
    only = os.environ.get("DROWSE_UNITS_ONLY")
    for arch in ("qwen3", "gemma3"):
        for quantization in ("q0f32", "q4f32_1"):
            if only and f"{arch}-{quantization}" not in only.split(","):
                continue
            if not (fixtures / f"units-{arch}-{quantization}").is_dir():
                raise SystemExit(f"missing fixture model units-{arch}-{quantization} in {fixtures}")
            results[f"{arch}-{quantization}"] = verify_arch(env, fixtures, goldens, arch, quantization)
            worst = results[f"{arch}-{quantization}"]
            print(f"units goldens {arch} {quantization}: "
                  + ", ".join(f"{k} {v:.2e}" for k, v in sorted(worst.items())), flush=True)
    return results


# ---------------------------------------------------------------------- replay (stage 2) ---

#: Replay forwards carry at most this many rows here, so every case also exercises the
#: KV-continuing chunk path (the bridge chunks at the library's `prefill_chunk_size`).
REPLAY_CHUNK = 16
#: Native `UnitScales` floor (drowse/units/capture.py `SCALE_FLOOR`).
SCALE_FLOOR = 1e-6
#: The raw first turn's one-token prompt (`browser_units.RAW_FIRST` through the fixture
#: tokenizer: "K" on qwen3, "=" on gemma3; no BOS, so its first row is produced at the sink).
RAW_FIRST_IDS = {"qwen3": [45], "gemma3": [32]}


def replay_meta(du, *, anatomy=None, spans=None, population_row=None, traces=(),
                pattern_layers=()) -> np.ndarray:
    """`replay_meta` as drowse.ai's `encodeReplayMeta` packs it (HOOK_ABI.md §7)."""
    lay = du.replay_meta_layout()
    meta = np.full(lay["length"], -1, np.int32)
    flags = 0
    if anatomy is not None:
        meta[lay["anatomy_position"]] = anatomy
        flags |= du.REPLAY_FLAG_ANATOMY
    meta[lay["span_count"]] = 0
    meta[lay["trace_count"]] = 0
    if spans is not None:
        meta[lay["span_count"]] = len(spans)
        for i, (first_row, rows, first_position) in enumerate(spans):
            meta[lay["spans"] + 4 * i: lay["spans"] + 4 * i + 4] = [
                first_row, rows, first_position, 1 if first_position < 0 else 0]
        meta[lay["population_row"]] = -1 if population_row is None else population_row
        meta[lay["trace_count"]] = len(traces)
        for i, (kind, layer, index) in enumerate(traces):
            meta[lay["traces"] + 3 * i: lay["traces"] + 3 * i + 3] = [
                layer, 0 if kind == "neuron" else 1, index]
        flags |= du.REPLAY_FLAG_ACTIVITY
    meta[lay["pattern_layer_count"]] = len(pattern_layers)
    if pattern_layers:
        meta[lay["pattern_layers"]: lay["pattern_layers"] + len(pattern_layers)] = sorted(pattern_layers)
        flags |= du.REPLAY_FLAG_PATTERNS
    meta[lay["flags"]] = flags
    return meta


def unit_scales(stats: dict) -> np.ndarray:
    """`[L, 2·d_ff + H]`: `S+ | S− | S_head`, floored as native's activity reads them."""
    neuron, head = stats["neuron_scale"], stats["head_scale"]
    return np.maximum(np.concatenate([neuron[:, 0], neuron[:, 1], head], axis=1),
                      SCALE_FLOOR).astype(np.float32)


def run_replay(vm, segments, *, program, meta, scales, grams, affine=None, anatomy=None, rows=1):
    """Feed `segments` (`(ids, active)` each: one control state) on one fresh sequence in
    forwards of at most `REPLAY_CHUNK` rows, threading the carried buffers. Returns the carry,
    every forward's output, and the forward holding the anatomy position (its taps and
    queries)."""
    tokens = sum(len(ids) for ids, _ in segments)
    cache, carry = vm.new_cache(), vm.replay_carry(max(1, rows), tokens)
    meta_t = meta
    position, outputs, at = 0, [], None
    for ids, active in segments:
        buffers = program.buffers(active)
        for start in range(0, len(ids), REPLAY_CHUNK):
            chunk = list(ids[start: start + REPLAY_CHUNK])
            out = vm.replay_forward(cache, chunk, buffers, meta_t, scales, grams, carry, affine)
            carry = out["carry"]
            if anatomy is not None and position <= anatomy < position + len(chunk):
                at = out
            outputs.append((position, chunk, out))
            position += len(chunk)
    return carry, outputs, at


def check_next_tokens(label: str, outputs, tokens: dict[int, int], *, distinct: bool = False) -> None:
    """Each forward's last logits pick the token the golden says was produced there (greedy, or
    `distinct` greedy over the tokens not yet generated, as the goldens' generated nodes ran)."""
    order = sorted(tokens)
    for position, chunk, out in outputs:
        last = position + len(chunk) - 1
        if last not in tokens:
            continue
        logits = out["logits"].astype(np.float64).copy()
        if distinct:
            logits[[tokens[p] for p in order if p < last]] = -np.inf
        if int(np.argmax(logits)) != tokens[last]:
            raise SystemExit(f"{label}: the replay at position {last} does not pick token {tokens[last]}")


def program_for(make_program, steering, stats, vm, push, affine_fn):
    """A unit program (and push buffers) for one golden's steering string."""
    unit, affine = make_program(), None
    if steering is None:
        return unit, affine
    terms = []
    for text in steering.split(" + "):
        text = text.strip()
        if text.startswith("0.3 neuron/"):
            affine = affine_fn(vm, push, text)
        else:
            terms.append(text)
    for kind, layer, index, alpha, trigger in split_terms(" + ".join(terms) if terms else None):
        mean = stats["neuron_mean"][layer, index] if kind == "neuron" else stats["head_mean"][layer, index]
        unit.ablate(kind, layer, index, np.float32(1 - alpha),
                    np.float32(alpha) * np.asarray(mean, np.float32), trigger)
    return unit, affine


def head_write_norms(anatomy, layout, block, grams, hidden_size):
    L = anatomy.shape[0]
    hd, H = block["head_dim"], block["n_heads"]
    out = np.zeros((L, H), np.float64)
    for layer in range(L):
        z = anatomy[layer, layout["z"]: layout["z"] + H * hd].astype(np.float64).reshape(H, hd)
        o = anatomy[layer, layout["attn_o"]: layout["attn_o"] + hidden_size].astype(np.float64)
        q = np.einsum("hi,hij,hj->h", z, grams[layer].astype(np.float64), z)
        norm = np.sqrt(np.maximum(q, 0))
        if block["sandwich"]:
            norm = norm / np.sqrt(np.mean(o * o) + block["rms_eps"])
        out[layer] = norm
    return out


def pattern_rows(vm, carry, queries, heads, slots, position):
    """`drowse_attention_rows` for `(layer, head)` pairs, in batches of `MAX_PATTERN_HEADS`."""
    du = vm.env.drowse_units
    rows = []
    for start in range(0, len(heads), du.UNIT_MAX_PATTERN_HEADS):
        batch = heads[start: start + du.UNIT_MAX_PATTERN_HEADS]
        meta = np.full((du.UNIT_MAX_PATTERN_HEADS, 4), -1, np.int32)
        for i, (layer, head) in enumerate(batch):
            meta[i] = [slots[layer], layer, head, position]
        out = vm.attention_rows(carry["keys"], queries, meta)
        rows.extend(out[: len(batch)])
        if np.any(out[len(batch):] != 0):
            raise SystemExit("attention rows: an unused row is not zero")
    return np.asarray(rows)


def verify_replay(env, vm, tensors, stats, goldens: Path, arch: str, make_program, worst: dict):
    du = vm.env.drowse_units
    block, L, d = vm.block, vm.num_layers, vm.hidden_size
    layout = du.anatomy_layout(d, block)
    scales_np = unit_scales(stats)
    grams_np = vm.head_grams()
    host_grams = np.stack([np.stack([head_gram(tensors, block, layer, h) for h in range(block["n_heads"])])
                           for layer in range(L)])
    if not np.allclose(grams_np, host_grams, rtol=1e-4, atol=1e-5):
        raise SystemExit(f"drowse_head_write_grams differs from the converted tensors' grams "
                         f"(max |Δ| {float(np.abs(grams_np - host_grams).max()):.3g})")
    tvm = env.tvm
    scales, grams = tvm.runtime.tensor(scales_np), tvm.runtime.tensor(grams_np)
    push = json.loads((goldens / f"neuron-push-{arch}-v1.json").read_text())
    all_layers = list(range(L))
    slots = {layer: i for i, layer in enumerate(all_layers)}

    # attention-patterns: every head at positions 2, 5, 8 of the short prompt, every layer.
    golden = json.loads((goldens / f"attention-patterns-{arch}-v1.json").read_text())
    tol = Tolerance(golden["tolerance"])
    prob = golden["tolerance"]["library"]["probability_atol"]
    heads = [(layer, h) for layer in all_layers for h in range(block["n_heads"])]
    for case in golden["cases"]:
        unit, affine = program_for(make_program, case["steering"], stats, vm, push, push_affine)
        for row in case["rows"]:
            p = row["position"]
            meta = replay_meta(du, anatomy=p, pattern_layers=all_layers)
            carry, outputs, at = run_replay(vm, [(golden["ids"], [True] * len(unit.triggers))],
                                            program=unit, meta=meta, scales=scales, grams=grams,
                                            affine=affine, anatomy=p)
            got = pattern_rows(vm, carry, at["pattern_q"], heads, slots, p)
            for layer in all_layers:
                expected = decode(row["layers"][layer])
                block_rows = got[layer * block["n_heads"]: (layer + 1) * block["n_heads"]]
                tol.check(f"attention-patterns:{case['name']}@{p}.L{layer}", expected,
                          block_rows[:, : p + 1], atol=prob, rtol=0.0)
                if np.any(block_rows[:, p + 1:] != 0):
                    raise SystemExit(f"attention-patterns:{case['name']}@{p}: weight past p")
    worst.update(tol.worst)

    # token-anatomy: the taps at the producing position, write norms, chosen heads' rows.
    golden = json.loads((goldens / f"token-anatomy-{arch}-v1.json").read_text())
    tol = Tolerance(golden["tolerance"])
    prob = golden["tolerance"]["library"]["probability_atol"]
    rounded = golden["tolerance"]["rounded"]["atol"]
    for case in golden["cases"]:
        inter = case["intermediates"]
        ids = inter["ids"]
        p = len(ids) - 1
        unit, affine = program_for(make_program, case["steering"], stats, vm, push, push_affine)
        meta = replay_meta(du, anatomy=p, pattern_layers=all_layers)
        carry, outputs, at = run_replay(vm, [(ids, [True] * len(unit.triggers))], program=unit,
                                        meta=meta, scales=scales, grams=grams, affine=affine,
                                        anatomy=p)
        anatomy = at["anatomy"]
        label = f"token-anatomy:{case['name']}"
        for name in ("block_in", "mid", "out", "act", "z"):
            expected = decode(inter["taps"][name])
            tol.check(f"{label}.{name}", expected,
                      anatomy[:, layout[name]: layout[name] + expected.shape[1]])
        tol.check(f"{label}.head_write_norms", decode(inter["head_write_norms"]),
                  head_write_norms(anatomy, layout, block, host_grams, d))
        # block_in[L+1] is out[L]; out_pre − mid is the MLP's write (after the post-norm on Gemma).
        tol.check(f"{label}.block_in=out", anatomy[1:, layout["block_in"]: layout["block_in"] + d],
                  anatomy[:-1, layout["out"]: layout["out"] + d])
        mlp = anatomy[:, layout["mlp_o"]: layout["mlp_o"] + d].astype(np.float64)
        if block["sandwich"]:
            gamma = np.stack([tensors[next(n for n in tensors if n.endswith(
                f"layers.{layer}.post_feedforward_layernorm.weight"))] for layer in all_layers])
            mlp = gamma * mlp / np.sqrt(np.mean(mlp * mlp, axis=1, keepdims=True) + block["rms_eps"])
        tol.check(f"{label}.mlp_write", mlp,
                  anatomy[:, layout["out_pre"]: layout["out_pre"] + d]
                  - anatomy[:, layout["mid"]: layout["mid"] + d])
        record = case["envelope"]["measurements"]["anatomy"]
        if record.get("patterns"):
            chosen = [(item["layer"], item["head"]) for item in record["patterns"]]
            got = pattern_rows(vm, carry, at["pattern_q"], chosen, slots, p)
            for item, row in zip(record["patterns"], got):
                tol.check(f"{label}.pattern.{item['unit']}", item["weights"], row[: p + 1],
                          atol=prob + rounded, rtol=0.0)
    worst.update(tol.worst)

    # unit-activity: device peaks against every unit's diagnostics, the host ranking, traces and
    # the population (B1-goldens §2.5).
    verify_activity(env, vm, goldens, arch, make_program, stats, scales, grams, worst)


def spans_from_runs(runs, rows: int):
    """Native `ReplayForward.spans()`: `(first_row, rows, producing position of the first row)`."""
    stops = [first for first, _ in runs[1:]] + [rows]
    return [(int(first), int(stop) - int(first), int(at) - 1) for (first, at), stop in zip(runs, stops)]


def rank_activity(state: np.ndarray, block: dict, top: int):
    """The host's ranking of the device state (native `_activity`): each kind's top `top` by rank
    over the flat order `[(neuron, 0), (head, 0), (neuron, 1), …]`, rank descending, flat index
    ascending on ties, rank −1 dropped."""
    f, h = block["d_ff"], block["n_heads"]
    flat = []
    for layer in range(state.shape[0]):
        row = state[layer]
        for n in range(f):
            flat.append(("neuron", layer, n, row[n], row[f + n], row[2 * f + n]))
        for k in range(h):
            flat.append(("head", layer, k, row[3 * f + k], row[3 * f + h + k], row[3 * f + 2 * h + k]))
    out = []
    for kind in ("neuron", "head"):
        items = sorted(((i, e) for i, e in enumerate(flat) if e[0] == kind),
                       key=lambda item: (-float(item[1][5]), item[0]))
        out += [e for _, e in items[:top] if e[5] >= 0]
    return out


def unit_name(kind: str, layer: int, index: int) -> str:
    return f"{kind}/L{layer}{'n' if kind == 'neuron' else 'h'}{index}"


def f16_close(golden_text: str, actual: np.ndarray) -> bool:
    golden = np.frombuffer(base64.b64decode(golden_text), "<f2").astype(np.float64)
    if golden.shape != actual.shape:
        return False
    rounded = actual.astype(np.float16).astype(np.float64)
    ulp = np.spacing(np.abs(golden.astype(np.float16))).astype(np.float64)
    return bool(np.all(np.abs(rounded - golden) <= ulp))


def verify_activity(env, vm, goldens: Path, arch: str, make_program, stats, scales, grams, worst):
    du = vm.env.drowse_units
    block, L = vm.block, vm.num_layers
    f, h = block["d_ff"], block["n_heads"]
    golden = json.loads((goldens / f"unit-activity-{arch}-v1.json").read_text())
    tol = Tolerance(golden["tolerance"])
    z_atol = tol.z_atol + golden["tolerance"]["rounded"]["atol"]
    anatomy = json.loads((goldens / f"token-anatomy-{arch}-v1.json").read_text())
    first = anatomy["cases"][0]
    prompt_length = first["envelope"]["measurements"]["anatomy"]["token"]["prompt_length"]
    chat = list(first["intermediates"]["ids"][:prompt_length])
    every = [("neuron", layer, n) for layer in range(L) for n in range(f)] + [
        ("head", layer, k) for layer in range(L) for k in range(h)]
    ties = {t["path"]: t for t in golden["declared_ties"]}
    for number, case in enumerate(golden["cases"]):
        name, args, result = case["name"], case["arguments"], case["result"]
        label = f"unit-activity:{name}"
        if "refusal" in case:
            raise SystemExit(f"{label}: refusal cases are host-only and not expected here")
        tokens = [t["token_id"] for t in result["tokens"]]
        unit = make_program()
        if name in ("generated", "segmented"):
            ids = chat + tokens[:-1]
            spans = [(0, len(tokens), prompt_length - 1)]
            if result["steering"] is not None:
                kind, layer, index, alpha, trigger = split_terms(result["steering"])[0]
                mean = stats["neuron_mean"][layer, index] if kind == "neuron" else stats["head_mean"][layer, index]
                unit.ablate(kind, layer, index, np.float32(1 - alpha),
                            np.float32(alpha) * np.asarray(mean, np.float32), trigger)
                if trigger != {"prompt": True, "generated": False}:
                    raise SystemExit(f"{label}: only a prompt-only segmented case is handled")
                segments = [(ids[:prompt_length], [True]), (ids[prompt_length:], [False])]
            else:
                segments = [(ids, [])]
            produced = {prompt_length - 1 + i: t for i, t in enumerate(tokens)}
        elif name in ("sink", "sink-population"):
            ids = RAW_FIRST_IDS[arch] + tokens[:-1]
            spans = [(0, len(tokens), 0)]
            segments = [(ids, [])]
            produced = {i: t for i, t in enumerate(tokens)}
        elif name == "authored":
            start = next(i for i in range(len(chat)) if chat[i: i + len(tokens)] == tokens)
            ids = chat[: start + len(tokens) - 1]
            spans = [(0, len(tokens), start - 1)]
            segments = [(ids, [])]
            produced = {}
        else:
            ids = list(args["ids"])
            spans = spans_from_runs(args["runs"], len(args["raw_ids"]))
            if name == "excluded-and-zero":
                unit.ablate("neuron", 1, 5, np.float32(0), np.float32(0), None)
                segments = [(ids, [True])]
            else:
                segments = [(ids, [])]
            produced = {}
        traces = [parse_unit(u) for u in args.get("units", [])]
        population_row = args.get("raw_index")
        meta = replay_meta(du, spans=spans, population_row=population_row, traces=traces)
        carry, outputs, _ = run_replay(vm, segments, program=unit, meta=meta, scales=scales,
                                       grams=grams, rows=len(tokens))
        check_next_tokens(label, outputs, produced, distinct=name in ("generated", "segmented"))
        state = carry["state"].numpy()
        # every unit's device peak against the diagnostics
        peak = decode(case["diagnostics"]["peak"])
        row = decode(case["diagnostics"]["row"])
        margin = decode(case["diagnostics"]["row_margin"])
        for i, (kind, layer, index) in enumerate(every):
            base = 0 if kind == "neuron" else 3 * f
            width = f if kind == "neuron" else h
            value = float(state[layer, base + index])
            at = int(state[layer, base + width + index])
            rank = float(state[layer, base + 2 * width + index])
            unit_label = f"{label}.{unit_name(kind, layer, index)}"
            if int(row[i]) < 0:
                if at != -1 or rank != -1:
                    raise SystemExit(f"{unit_label}: every row is excluded, the device peaked at {at}")
                continue
            tol.check(f"{label}.peak", [peak[i]], [value], atol=z_atol, rtol=0.0)
            if rank != abs(value):
                raise SystemExit(f"{unit_label}: rank {rank} is not |{value}|")
            forced = name == "excluded-and-zero" and (kind, layer, index) == ("neuron", 1, 5)
            if forced and at != int(row[i]):
                raise SystemExit(f"{unit_label}: a unit 0 everywhere peaks at row {at}, not {int(row[i])}")
            if margin[i] >= 1e-3 * max(1.0, abs(float(peak[i]))) and at != int(row[i]):
                raise SystemExit(f"{unit_label}: peak row {at}, golden {int(row[i])}")
        # the host ranking of the device state
        ranked = rank_activity(state, block, int(args["top"]))
        listed = result["top"]
        if len(ranked) != len(listed):
            raise SystemExit(f"{label}: {len(ranked)} ranked units, golden {len(listed)}")
        groups = ties.get(f"cases[{number}].result.top", {"groups": []})["groups"]
        grouped = {i: tuple(g) for g in groups for i in g}
        for i, (entry, (kind, layer, index, value, at, _rank)) in enumerate(zip(listed, ranked)):
            name_i = unit_name(kind, layer, index)
            if i in grouped:
                members = {listed[j]["unit"] for j in grouped[i]}
                if name_i not in members:
                    raise SystemExit(f"{label}: tie group {grouped[i]} lacks {name_i}")
                entry = next(e for e in listed if e["unit"] == name_i)
            elif entry["unit"] != name_i:
                raise SystemExit(f"{label}: top[{i}] is {name_i}, golden {entry['unit']}")
            tol.check(f"{label}.top", [entry["peak"]], [value], atol=z_atol, rtol=0.0)
            if name_i not in case["tied_peak_rows"] and int(at) != entry["raw_index"]:
                raise SystemExit(f"{label}: {name_i} peaks at row {int(at)}, golden {entry['raw_index']}")
        # traces (unproduced rows were never written: 0) and the population
        trace_values = carry["traces"].numpy()
        for slot, unit_text in enumerate(args.get("units", [])):
            expected = result["traces"][unit_text]
            tol.check(f"{label}.trace", expected, trace_values[slot, : len(expected)],
                      atol=z_atol, rtol=0.0)
        if population_row is not None:
            population = carry["population"].numpy()
            for entry in result["population"]["layers"]:
                layer = entry["layer"]
                if not f16_close(entry["neurons"], population[layer, :f]) or not f16_close(
                        entry["heads"], population[layer, f:]):
                    raise SystemExit(f"{label}: population at layer {layer} is not within one f16 ulp")
    worst.update(tol.worst)
