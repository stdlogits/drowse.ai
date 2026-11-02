"""Native references for scripts/browser-units-worker.mjs: build the R, S, core-statistics and
units packs for drowse's tiny ABI v5 fixture model bound to a staged WebGPU fixture library, then
compute what drowse answers on the same model with the hosted session's lens band (the browser
keeps native's default band; the goldens read with the fixture's): greedy raw generations under a
neuron push, a unit ablation at @response and an S-lens-gated push with an attached `slens/`
probe; a generated node's token anatomy (unsteered and steered) and unit activity; and the
unit-reads golden's reads, tops and vector find. Ranked lists carry declared ties as the goldens
do (`rank_lists` one entry deeper).

    python3 units-worker-reference.py ARCH OUT RUNTIME_JSON
"""

from __future__ import annotations

import json

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


def jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=lambda item: item.to_dict() if hasattr(item, "to_dict") else str(item)))


def ranked(call: Any, candidates: list[dict[str, int]], path: str, ties: list[dict[str, Any]]) -> Any:
    """``call(**knobs)`` at the first candidate whose ranked lists keep every boundary (no entry
    ties the best excluded one deeper, as the goldens choose their knobs), with its near-ties
    declared; returns ``{"knobs", "result"}``."""
    refusals = []
    for knobs in candidates:
        short = jsonable(call(**knobs))
        long = jsonable(call(**{key: value + 1 for key, value in knobs.items()}))
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
    with fixture(arch) as fx:
        session = fx.session
        model = NativeModel(session)
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
        for name, steering, probes in (
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
        for name, steering in (("unsteered", None), ("steered", ANATOMY_STEERING)):
            runs = generate(steering, steps=DECODE_STEPS, stateless=False)
            node = str(runs.node_ids[0])
            # The hosted fixture is a base model: its replays render the node raw.
            anatomy = ranked(lambda **knobs: session.token_anatomy(node, ANATOMY_INDEX, raw=True, **knobs),
                             [{"k": k, "units": k, "patterns": 2} for k in (5, 4, 6, 3, 7)],
                             f"nodes.{name}.anatomy", ties)
            entry: dict[str, Any] = {
                "steering": steering, "tokens": [int(t) for t in runs.first.tokens], "anatomy": anatomy,
            }
            if name == "unsteered":
                entry["activity"] = ranked(
                    lambda top: session.unit_activity(node, top=top, units=ACTIVITY["units"],
                                                      raw_index=ACTIVITY["raw_index"], layers=ACTIVITY["layers"],
                                                      raw=True),
                    around("top", ACTIVITY["top"]), f"nodes.{name}.activity", ties)
            nodes[name] = entry

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
            "arch": arch, "text": text, "generations": generations, "nodes": nodes,
            "reads": reads, "top": tops, "find": finds, "declared_ties": ties,
            "settings": {"push": PUSH, "ablate": ABLATE_RESPONSE, "slens": SLENS_GATED,
                         "slensWord": SLENS_WORD, "probes": list(PROBES), "anatomySteering": ANATOMY_STEERING,
                         "anatomyIndex": ANATOMY_INDEX,
                         "activity": ACTIVITY, "steps": STEPS, "decodeSteps": DECODE_STEPS},
        }), (out / "reference.json").open("w"))


if __name__ == "__main__":
    main()
