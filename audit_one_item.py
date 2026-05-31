#!/usr/bin/env python3
"""Audit ONE item end-to-end so a human can verify the per-step logical check.

Traces `research_sample_inventory-002` (the demo case) through the same edit
chain the runner uses, printing for every step:
  - the edit instruction,
  - the seed-vs-current INVARIANT diff (which IDs/keys/values the round trip
    must preserve and whether the editor preserved them),
  - the runtime_risk breakdown WITH the weighted arithmetic, and
  - the Conductor's intervene decision against the threshold.

Editor calls are temperature 0 -> served from the on-disk cache, so this replays
the exact documents from the live run (no new credits, fully reproducible).
"""
from __future__ import annotations

import random

from relay.roundtrip import agents
from relay.roundtrip.checksum import runtime_risk
from relay.roundtrip.jsonutil import (by_id, parse_doc, record_ids,
                                      total_stock)
from relay.roundtrip.runner import _steps
from relay.roundtrip.tasks import load_tasks
from relay.conductor import should_intervene
from relay.wandb_client import WEAK_MODEL

TASK_ID = "research_sample_inventory-002"
THRESHOLD = 0.008          # same as the live run
MODEL = "microsoft/Phi-4-mini-instruct"
NUM_ROUND_TRIPS = 3


def inv(doc_str):
    d = parse_doc(doc_str)
    return set(record_ids(d)), by_id(d), total_stock(d)


def show_risk(rep):
    print(f"      risk = {rep['risk']:.4f}   [threshold {THRESHOLD}]")
    print(f"        = 0.30*id_loss({rep['id_loss_rate']}) "
          f"+ 0.20*key_loss({rep['required_key_loss_rate']}) "
          f"+ 0.15*agg_drift({rep['aggregate_drift']}) "
          f"+ 0.25*value_drift({rep['value_drift_rate']})")
    if rep["missing_ids"]:
        print(f"        MISSING RECORD IDS (round trip dropped them): "
              f"{rep['missing_ids']}")
    if rep["value_drift"]:
        for rid, bad in list(rep["value_drift"].items())[:4]:
            print(f"        VALUE DRIFT @ {rid}: {bad}")
    if rep["numeric_drift"]:
        for rid, d in list(rep["numeric_drift"].items())[:4]:
            print(f"        STOCK DRIFT @ {rid}: {d}")


def main():
    agents.configure(use_mock=False, model=MODEL, provider="wandb")

    task = next(t for t in load_tasks(5, 42) if t.task_id == TASK_ID)
    seed_ids, _seed_by, seed_stock = inv(task.seed_doc)
    print("=" * 78)
    print(f"ITEM: {TASK_ID}   (editor = {MODEL}, threshold = {THRESHOLD})")
    print(f"SEED INVARIANTS the round trip MUST return:")
    print(f"  {len(seed_ids)} record IDs: {sorted(seed_ids)}")
    print(f"  total_stock = {seed_stock}")
    print(f"  every record must keep its id / name / category")
    print("=" * 78)

    for cond in ("naive", "adaptive"):
        print(f"\n################  CONDITION: {cond}  ################")
        rng = random.Random(f"123|{task.task_id}|{cond}")
        current = task.seed_doc
        for rt, edit_name, step_type, instruction in _steps(task, NUM_ROUND_TRIPS):
            current = agents.apply_edit(current, instruction)
            rep = runtime_risk(task.seed_doc, current)
            cur_ids, _, cur_stock = inv(current)
            print(f"\n  round {rt} {step_type:8s} edit=`{edit_name}`")
            print(f"    editor output: {len(cur_ids)}/{len(seed_ids)} IDs kept, "
                  f"total_stock={cur_stock}")
            show_risk(rep)
            decided = should_intervene(cond, rep["risk"], THRESHOLD, None, rng)
            print(f"    CONDUCTOR: risk {rep['risk']:.4f} "
                  f"{'>' if rep['risk'] > THRESHOLD else '<='} {THRESHOLD} "
                  f"-> intervene = {decided}")
            if decided:
                current = agents.repair_doc(task.seed_doc, current, instruction, rep)
                rep2 = runtime_risk(task.seed_doc, current)
                cur_ids2, _, _ = inv(current)
                print(f"    REPAIR fired -> {len(cur_ids2)}/{len(seed_ids)} IDs, "
                      f"risk_after = {rep2['risk']:.4f}")


if __name__ == "__main__":
    main()
