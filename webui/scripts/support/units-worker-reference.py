"""Native references for scripts/browser-units-worker.mjs: build the R, S, core-statistics and
units packs for drowse's tiny ABI v5 fixture model bound to a staged WebGPU fixture library, then
compute what drowse answers on the same model with the hosted session's lens band (the browser
keeps native's default band; the goldens read with the fixture's): greedy raw generations under a
neuron push, a unit ablation at @response and an S-lens-gated push with an attached `slens/`
probe; a generated node's token anatomy (unsteered and steered) and unit activity; and the
unit-reads golden's reads, tops and vector find. Ranked lists carry declared ties as the goldens
do (`rank_lists` one entry deeper).

With `chat`, the fixture is the instruct model its chat template makes (the hosted harness stages
the same template, `--chat`): no raw generations; chat nodes instead, rendered through drowse's
scene stitcher: the goldens' prompt (system "Be brief.", user "Hi there") unsteered and steered,
the user turn alone, a second assistant turn under it (the system folded only into the first turn
where the template folds) and a user-seat turn under it, unsteered and steered; each node's token
anatomy and unit activity read as chat (`raw=False`).

With `gated MANIFOLD_ARCHIVE` (run after the harness's browser core build): raw nodes whose push
is gated on an attached J-lens probe (`jlens/e`, the R pack's lens) and on an attached geometry
probe (`local/calm.alert`, the very fit the browser's core build wrote: drowse loads the archive's
fitted pair with its own `load_manifold` and registers it, so both sides gate on one manifold).
Each gate's threshold is the midpoint of the two largest readings drowse's decode loop scores for
it (recorded from `ReadPlane.gate_score_callback` on a run whose gate never fires), so the gate
opens mid-generation. Writes gated-reference.json.

    python3 units-worker-reference.py ARCH OUT RUNTIME_JSON [chat | gated MANIFOLD_ARCHIVE]
"""

from __future__ import annotations

import json
import tempfile
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager

import torch
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "browser-runtime"))
sys.path.insert(0, str(ROOT / "drowse"))

from build_hosted_packs import KINDS, NativeModel, build  # noqa: E402
from tests.support.browser_units import (  # noqa: E402
    DECODE_STEPS,
    SHORT_TEXT,
    FixtureError,
    _find,
    _greedy,
    fixture,
    rank_lists,
    short_prompt_ids,
)

PUSH = "0.3 neuron/L2n7"
ABLATE_RESPONSE = "!neuron/L1n5@response"
SLENS_WORD = "e"
SLENS_GATED = f"0.3 neuron/L2n7@when:slens/{SLENS_WORD} > 0"
ANATOMY_STEERING = "0.3 neuron/L2n7 + !head/L1h3"
#: Attached through every generation: the unit-probes golden's units and an S-lens probe.
PROBES = ("neuron/L1n5", "head/L2h1", f"slens/{SLENS_WORD}")
ANATOMY_INDEX = 3
ACTIVITY = {"top": 5, "units": ["neuron/L1n5", "head/L2h1"], "raw_index": 2, "layers": "0,2"}
STEPS = 5
#: The chat nodes (`chat` mode): name, steering, input, system prompt, parent node, generated seat,
#: whether unit activity is read too.
CHAT_SYSTEM = "Be brief."
CHAT_NODES = (
    ("chat", None, "Hi there", CHAT_SYSTEM, None, "assistant", True),
    ("chat-steered", ANATOMY_STEERING, "Hi there", CHAT_SYSTEM, None, "assistant", False),
    ("chat-plain", None, "Hi there", None, None, "assistant", True),
    ("chat-second", None, "Go on", CHAT_SYSTEM, "chat", "assistant", True),
    ("user-seat", None, None, CHAT_SYSTEM, "chat", "user", True),
    ("user-seat-steered", ANATOMY_STEERING, None, CHAT_SYSTEM, "chat", "user", False),
)


#: The gated nodes (`gated` mode): name, attached probe, gate key.
GATED_NODES = (
    ("jlens-gated", f"jlens/{SLENS_WORD}", f"jlens/{SLENS_WORD}"),
    ("geometry-gated", "local/calm.alert", "local/calm.alert"),
)
GATED_PUSH = "0.3 neuron/L2n7"


@contextmanager
def recorded_gate_scalars() -> Iterator[list[dict[str, float]]]:
    """Every gate scalar dict the decode loop scores, in call order."""
    from drowse.generation.read_plane import ReadPlane

    seen: list[dict[str, float]] = []
    original = ReadPlane.gate_score_callback

    def patched(self: Any) -> Any:
        score = original(self)

        def wrapped(step_id: int) -> Any:
            out = score(step_id)
            seen.append({key: float(value) for key, value in out.items()})
            return out
        return wrapped

    ReadPlane.gate_score_callback = patched  # type: ignore[method-assign]
    try:
        yield seen
    finally:
        ReadPlane.gate_score_callback = original  # type: ignore[method-assign]


def threshold(values: list[float], what: str) -> str:
    """A threshold that opens the gate for the anatomy's forward (its reading is the previous
    forward's, ``values[ANATOMY_INDEX - 1]``) and shuts it at least once elsewhere: the midpoint
    between that reading and the nearest one below it, written with enough digits for a
    probability."""
    at = values[ANATOMY_INDEX - 1]
    below = [value for value in values if value < at]
    if not below:
        raise FixtureError(f"{what}: no reading lies below the anatomy forward's ({values})")
    nearest = max(below)
    if not at - nearest > 1e-3 * max(abs(at), 1e-12):
        raise FixtureError(f"{what}: the gate's readings tie ({at} vs {nearest})")
    return f"{(at + nearest) / 2:.12f}".rstrip("0")


def register_archive_manifold(session: Any, archive: Path, key: str) -> None:
    """The fitted pair the browser's core build wrote, loaded by drowse and registered as ``key``."""
    from drowse.store.manifold_tensors import load_manifold

    folder = "manifolds/" + key + "/"
    with zipfile.ZipFile(archive) as bundle, tempfile.TemporaryDirectory() as tmp:
        tensors = [n for n in bundle.namelist() if n.startswith(folder) and n.endswith(".safetensors")]
        if len(tensors) != 1:
            raise FixtureError(f"{archive.name}: expected one fitted pair under {folder}, found {tensors}")
        for name in (tensors[0], tensors[0].removesuffix(".safetensors") + ".json"):
            bundle.extract(name, tmp)
        manifold = load_manifold(Path(tmp) / tensors[0], verify_manifest=False)
    session._registry.put_manifold(key, manifold)


def jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=lambda item: item.to_dict() if hasattr(item, "to_dict") else str(item)))


def without_underflow(value: Any) -> Any:
    """``value`` without the probability entries that underflowed to exactly 0: which of the
    vocabulary's zero-probability tokens fill a list is arbitrary (a sharp lens read leaves most
    of it at 0), so both sides drop them (browser-units-worker.mjs does the same)."""
    if isinstance(value, list):
        return [without_underflow(item) for item in value]
    if not isinstance(value, dict):
        return value
    probability = value.get("unit") == "probability"
    return {
        key: [item for item in child if not (isinstance(item, dict) and item.get("value") == 0)]
        if probability and isinstance(child, list) else without_underflow(child)
        for key, child in value.items()
    }


def ranked(call: Any, candidates: list[dict[str, int]], path: str, ties: list[dict[str, Any]]) -> Any:
    """``call(**knobs)`` at the first candidate whose ranked lists keep every boundary (no entry
    ties the best excluded one deeper, as the goldens choose their knobs), with its near-ties
    declared; returns ``{"knobs", "result"}``."""
    refusals = []
    for knobs in candidates:
        short = without_underflow(jsonable(call(**knobs)))
        long = without_underflow(jsonable(call(**{key: value + 1 for key, value in knobs.items()})))
        found: list[dict[str, Any]] = []
        try:
            rank_lists(short, long, found, path + ".result")
        except FixtureError as exc:
            refusals.append(f"{knobs}: {exc}")
            continue
        ties.extend(found)
        return {"knobs": knobs, "result": short}
    raise FixtureError(f"{path}: no knobs keep every boundary: {refusals}")


def _pursuit(fx: Any, call: Any, path: str, ties: list[dict[str, Any]]) -> Any:
    found = _find(fx, call, path, ties)
    return {"knobs": {"k": found["k"]}, "result": jsonable(found["result"])}


def around(name: str, value: int, **fixed: int) -> list[dict[str, int]]:
    return [{name: v, **fixed} for v in (value, value - 1, value + 1, value - 2, value + 2) if v > 0]


def main() -> None:
    from drowse.lens.depth import LensBand

    arch, out = sys.argv[1], Path(sys.argv[2])
    runtime = json.loads(Path(sys.argv[3]).read_text())
    mode = sys.argv[4] if len(sys.argv) > 4 else None
    chat, gated = mode == "chat", mode == "gated"
    with fixture(arch) as fx:
        session = fx.session
        model = NativeModel(session)
        if not gated:
            build(model, runtime, out / "packs", KINDS, full_depth=False, words=["a", "e", "t"])
        session.lens.band = LensBand()
        # The browser types the text and its completion prefix adds the family's BOS; the fixture
        # tokenizer adds none, so drowse reads the BOS piece as its special (the goldens' way).
        text = SHORT_TEXT[arch]
        native_text = fx.tokenizer.decode(short_prompt_ids(fx.tokenizer, arch))
        ties: list[dict[str, Any]] = []

        def generate(steering: str | None, *, steps: int = STEPS, stateless: bool = True) -> Any:
            session.tree.navigate(session.tree.root_id)
            return session.generate(native_text, raw=True, stateless=stateless, steering=steering,
                                    sampling=_greedy(steps))

        generations: dict[str, Any] = {}
        for name, steering, probes in () if chat or gated else (
            ("push", PUSH, PROBES),
            ("ablate-response", ABLATE_RESPONSE, PROBES),
            ("slens-gated", SLENS_GATED, PROBES),
        ):
            for probe in probes:
                session.add_probe(probe)
            try:
                final = generate(steering).first
                generations[name] = {
                    "steering": steering, "probes": list(probes), "tokens": [int(t) for t in final.tokens],
                    "aggregate": jsonable(final.measurements),
                }
            finally:
                for probe in probes:
                    session.remove_probe(probe)

        nodes: dict[str, Any] = {}

        def node_entry(name: str, steering: str | None, node: str, tokens: Any, *, raw: bool,
                       activity: bool, **fields: Any) -> dict[str, Any]:
            anatomy = ranked(lambda **knobs: session.token_anatomy(node, ANATOMY_INDEX, raw=raw, **knobs),
                             [{"k": k, "units": units, "patterns": 2}
                              for k in (5, 4, 6, 3, 7, 2) for units in (k, 5, 4, 6, 3)],
                             f"nodes.{name}.anatomy", ties)
            entry: dict[str, Any] = {"steering": steering, "tokens": [int(t) for t in tokens], "raw": raw,
                                     "anatomy": anatomy, **fields}
            if activity:
                entry["activity"] = ranked(
                    lambda top: session.unit_activity(node, top=top, units=ACTIVITY["units"],
                                                      raw_index=ACTIVITY["raw_index"], layers=ACTIVITY["layers"],
                                                      raw=raw),
                    around("top", ACTIVITY["top"]), f"nodes.{name}.activity", ties)
            return entry

        if chat:
            ids: dict[str, str] = {}
            for name, steering, text_in, system, parent, seat, activity in CHAT_NODES:
                if parent is None:
                    session.tree.navigate(session.tree.root_id)
                runs = session.generate(text_in, steering=steering, sampling=_greedy(DECODE_STEPS),
                                        system_prompt=system, parent_node_id=ids.get(parent) if parent else None,
                                        gen_seat=seat)
                ids[name] = str(runs.node_ids[0])
                nodes[name] = node_entry(name, steering, ids[name], runs.first.tokens, raw=False, activity=activity,
                                         probes=[], input=text_in, system=system, parent=parent, seat=seat)
        # The S-lens-gated nodes replay one decode position per forward, their gate scored from
        # the previous forward (native's gate run; the browser's replay readback) while slens/e
        # stays attached; unattached, the gate names no probe and never fires (resolve_gates), in
        # the generation and in its replays.
        slens_probe = f"slens/{SLENS_WORD}"
        if gated:
            register_archive_manifold(session, Path(sys.argv[5]), "local/calm.alert")
            for name, probe, key in GATED_NODES:
                session.add_probe(probe)
                try:
                    with recorded_gate_scalars() as seen:
                        generate(f"{GATED_PUSH}@when:{key} > 1000000", steps=DECODE_STEPS)
                    # The prefill's and the first decode forwards' scores gate the decode steps.
                    values = [step[key] for step in seen if key in step][: DECODE_STEPS - 1]
                    steering = f"{GATED_PUSH}@when:{key} > {threshold(values, name)}"
                    runs = generate(steering, steps=DECODE_STEPS, stateless=False)
                    nodes[name] = node_entry(name, steering, str(runs.node_ids[0]), runs.first.tokens, raw=True,
                                             activity=True, probes=[probe], input=text, system=None,
                                             parent=None, seat="assistant", gate_readings=values)
                finally:
                    session.remove_probe(probe)
            json.dump(jsonable({"arch": arch, "nodes": nodes, "declared_ties": ties}),
                      (out / "gated-reference.json").open("w"))
            return
        for name, steering, probes in () if chat else (
            ("unsteered", None, ()),
            ("steered", ANATOMY_STEERING, ()),
            ("slens-gated", SLENS_GATED, (slens_probe,)),
            ("slens-unattached", SLENS_GATED, ()),
        ):
            for probe in probes:
                session.add_probe(probe)
            try:
                runs = generate(steering, steps=DECODE_STEPS, stateless=False)
                # The hosted fixture is a base model: its replays render the node raw.
                nodes[name] = node_entry(name, steering, str(runs.node_ids[0]), runs.first.tokens, raw=True,
                                         activity=name != "steered", probes=list(probes), input=text,
                                         system=None, parent=None, seat="assistant")
            finally:
                for probe in probes:
                    session.remove_probe(probe)

        golden = json.loads((ROOT / "drowse/tests/fixtures/browser" / f"unit-reads-{arch}-v1.json").read_text())
        units = session.units
        reads = [
            {"unit": e["unit"], "hide_hubs": e["hide_hubs"],
             "read": ranked(lambda k, e=e: units.read(e["unit"], k=k, hide_hubs=e["hide_hubs"]).to_dict(),
                            around("k", e["k"]), f"reads[{i}]", ties)}
            for i, e in enumerate(golden["reads"])
        ]
        tops = [
            {"token": e["token"], "by": e["by"],
             "result": ranked(lambda k, e=e: units.top(e["token"], by=e["by"], layers="all", k=k).to_dict(),
                              around("k", e["k"]), f"top[{i}]", ties)}
            for i, e in enumerate(golden["top"])
        ]
        finds = []
        for e in golden["find"]:
            if e["target"] != "vector":
                continue
            from tests.support.browser_units import decode
            vector = torch.from_numpy(decode(e["vector"]).copy())
            finds.append({
                "vector": e["vector"], "layer": e["layer"],
                # A greedy pursuit: its k is chosen by the picks' margins (the golden's `_find`).
                "result": _pursuit(fx, lambda k, v=vector, e=e: units.find(v, layer=e["layer"], k=k).to_dict(),
                                   f"find[{len(finds)}]", ties),
            })

        json.dump(jsonable({
            "arch": arch, "chat": chat, "text": text, "generations": generations, "nodes": nodes,
            "reads": reads, "top": tops, "find": finds, "declared_ties": ties,
            "settings": {"push": PUSH, "ablate": ABLATE_RESPONSE, "slens": SLENS_GATED,
                         "slensWord": SLENS_WORD, "probes": list(PROBES), "anatomySteering": ANATOMY_STEERING,
                         "anatomyIndex": ANATOMY_INDEX,
                         "activity": ACTIVITY, "steps": STEPS, "decodeSteps": DECODE_STEPS},
        }), (out / "reference.json").open("w"))


if __name__ == "__main__":
    main()
