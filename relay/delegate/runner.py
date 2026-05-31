"""Run one real-DELEGATE52 task under one policy.

A round trip = apply the forward DELEGATE52 instruction, then its paired inverse
instruction; the result is back in the seed's representation, so the Conductor
checks it against the seed there (the intermediate is a different format, where
the structural invariants don't apply). Corruption COMPOUNDS: round trip i+1
edits the (possibly already-degraded) reconstruction from round trip i. Gold is
always the original seed.
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

from ..conductor import should_intervene
from . import agents


def run_task(domain, task, condition: str, *, threshold: float,
             random_rate, rng_seed: int, num_round_trips: int
             ) -> Tuple[List[dict], dict, dict]:
    rng = random.Random(f"{rng_seed}|{task.task_id}|{condition}")
    seed_struct = domain.parse(task.seed_text)
    current = task.seed_text
    rows: List[dict] = []
    edit_calls = 0
    repair_calls = 0
    pairs = task.round_trips

    for rt in range(num_round_trips):
        pair = pairs[rt % len(pairs)]
        # forward then inverse -> back to the seed representation.
        current = agents.apply_edit(current, pair.forward_instruction)
        current = agents.apply_edit(current, pair.backward_instruction)
        edit_calls += 2

        cur_struct = domain.parse(current)
        report = domain.runtime_risk(seed_struct, cur_struct)
        risk = report["risk"]
        intervened = should_intervene(condition, risk, threshold, random_rate, rng)

        if intervened:
            current = agents.repair_doc(task.seed_text, current,
                                        pair.backward_instruction,
                                        domain.repair_view(report))
            repair_calls += 1
            cur_struct = domain.parse(current)
            report_after = domain.runtime_risk(seed_struct, cur_struct)
        else:
            report_after = report

        rows.append({
            "task_id": task.task_id, "domain": domain.name,
            "condition": condition, "round_trip_index": rt,
            "edit_name": pair.name, "runtime_risk": risk,
            "risk_after": report_after["risk"],
            "intervened": bool(intervened),
            "cost_proxy": edit_calls + repair_calls,
            "n_seed": report.get("n_seed"), "n_cur": report.get("n_cur"),
        })

    final = domain.score(seed_struct, domain.parse(current))
    counts = {"edit_calls": edit_calls, "repair_calls": repair_calls,
              "interventions": repair_calls, "steps": len(rows),
              "cost_proxy": edit_calls + repair_calls, "final_doc": current}
    for r in rows:
        r["final_score"] = final["score"]
    return rows, final, counts
