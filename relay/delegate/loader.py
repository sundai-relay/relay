"""Load real DELEGATE52 samples and build round-trip (forward, backward) pairs.

Each sample is a state graph: ``basic_state`` (the seed file) has N forward edit
prompts, each pointing to a target_state; every target_state has exactly one
backward prompt pointing back to ``basic_state``. Each (forward, backward) pair
is a ready-made round trip whose gold is the seed.

We keep only *self-contained* pairs — ones the model can reverse from its own
intermediate output. Pairs whose backward prompt requires an external helper
file the public release does NOT ship (sequence_map.json, txn_order.json,
tag_hierarchy.json, opening_moves.pgn) are dropped, since without that file the
round trip is not reversible.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Tuple

# backward prompts that depend on a helper file not derivable from the forward
# transform alone -> not reversible by a pure model round trip.
_AUX_FILE_MARKERS = ("sequence_map", "txn_order", "tag_hierarchy",
                     "opening_moves.pgn")

_SEED_FILE_SUFFIX = {"accounting": ".ledger", "chess": ".pgn"}

_DEFAULT_JSONL = os.path.join(os.path.dirname(__file__),
                              "..", "..", "delegate", "delegate52.jsonl")


@dataclass
class RoundTrip:
    name: str                  # the forward prompt_id
    forward_instruction: str
    backward_instruction: str


@dataclass
class DelegateTask:
    task_id: str
    domain: str
    sample_name: str
    seed_text: str
    round_trips: List[RoundTrip]


def _seed_text(sample: dict, domain: str) -> str:
    suffix = _SEED_FILE_SUFFIX[domain]
    for path, content in sample["files"].items():
        if path.startswith("basic_state/") and path.endswith(suffix):
            return content
    # fall back to any basic_state file
    for path, content in sample["files"].items():
        if path.startswith("basic_state/"):
            return content
    raise ValueError(f"no basic_state seed file for {sample['sample_id']}")


def _round_trips(sample: dict) -> List[RoundTrip]:
    # forward prompts live on basic_state; backward prompts live on each target.
    forwards = {}   # target_state -> (prompt_id, forward_text)
    backwards = {}  # source_state -> backward_text (target == basic_state)
    for st in sample["states"]:
        for p in st.get("prompts", []):
            if st["state_id"] == "basic_state":
                forwards[p["target_state"]] = (p["prompt_id"], p["prompt"])
            elif p["target_state"] == "basic_state":
                backwards[st["state_id"]] = p["prompt"]

    pairs: List[RoundTrip] = []
    for target_state, (pid, fwd) in forwards.items():
        bwd = backwards.get(target_state)
        if bwd is None:
            continue
        blob = (fwd + " " + bwd).lower()
        if any(m in blob for m in _AUX_FILE_MARKERS):
            continue  # not reversible without a file we don't have
        pairs.append(RoundTrip(name=pid, forward_instruction=fwd,
                               backward_instruction=bwd))
    pairs.sort(key=lambda r: r.name)  # deterministic order
    return pairs


def load_delegate_tasks(domain: str, jsonl_path: str | None = None,
                        max_samples: int | None = None) -> List[DelegateTask]:
    path = jsonl_path or _DEFAULT_JSONL
    tasks: List[DelegateTask] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            if sample.get("sample_type") != domain:
                continue
            rts = _round_trips(sample)
            if not rts:
                continue
            tasks.append(DelegateTask(
                task_id=sample["sample_id"],
                domain=domain,
                sample_name=sample.get("sample_name", sample["sample_id"]),
                seed_text=_seed_text(sample, domain),
                round_trips=rts,
            ))
            if max_samples and len(tasks) >= max_samples:
                break
    return tasks
