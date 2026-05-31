"""runner.py — run one task under one policy.

Per round trip: apply the forward edit, read the runtime risk, maybe repair;
then the backward edit, read risk, maybe repair. Score the final doc against
the seed. Every edit/repair step is logged with the full Weave-friendly row.

Same tasks + edit sequence + round-trip count + temperature across conditions;
ONLY the intervention policy changes.
"""

from __future__ import annotations

import random
from typing import List, Tuple

from ..conductor import should_intervene
from ..weave_compat import op
from . import agents
from .checksum import runtime_risk
from .scorer import final_structural_score


def _steps(task, num_round_trips: int):
    """Yield (round_trip_index, edit_name, step_type, instruction)."""
    pairs = task.edit_pairs
    for rt in range(num_round_trips):
        pair = pairs[rt % len(pairs)]
        yield rt, pair.name, "forward", pair.forward_instruction
        yield rt, pair.name, "backward", pair.backward_instruction


@op()
def run_task(task, condition: str, threshold: float = 0.4,
             random_rate=None, rng_seed: int = 123, num_round_trips: int = 3
             ) -> Tuple[List[dict], dict, dict]:
    """Returns (rows, final_score, counts).

    ``rng_seed`` (an int) makes random-at-budget reproducible; the per-task RNG
    is derived from it so each task draws an independent, stable stream.
    """
    rng = random.Random(f"{rng_seed}|{task.task_id}|{condition}")
    seed = task.seed_doc
    current = seed
    rows: List[dict] = []
    edit_calls = 0
    repair_calls = 0

    for rt, edit_name, step_type, instruction in _steps(task, num_round_trips):
        current = agents.apply_edit(current, instruction)
        edit_calls += 1

        report = runtime_risk(seed, current)
        risk = report["risk"]
        intervened = should_intervene(condition, risk, threshold, random_rate, rng)

        if intervened:
            current = agents.repair_doc(seed, current, instruction, report)
            repair_calls += 1
            report_after = runtime_risk(seed, current)
        else:
            report_after = report

        cost_proxy = edit_calls + repair_calls
        rows.append({
            "task_id": task.task_id,
            "condition": condition,
            "round_trip_index": rt,
            "edit_name": edit_name,
            "step_type": step_type,
            "instruction": instruction,
            "runtime_risk": risk,
            "risk_after": report_after["risk"],
            "id_loss_rate": report["id_loss_rate"],
            "required_key_loss_rate": report["required_key_loss_rate"],
            "aggregate_drift": report["aggregate_drift"],
            "value_drift_rate": report.get("value_drift_rate", 0.0),
            "missing_ids": report["missing_ids"],
            "intervened": bool(intervened),
            "repair_call_count": repair_calls,
            "cost_proxy": cost_proxy,
            # current_doc is logged for Weave/demo; kept out of the jsonl summary
            # rows written by run_all_conditions to keep that file lean.
            "current_doc": current,
        })

    final = final_structural_score(seed, current)
    counts = {
        "edit_calls": edit_calls,
        "repair_calls": repair_calls,
        "interventions": repair_calls,
        "steps": len(rows),
        "cost_proxy": edit_calls + repair_calls,
        "final_doc": current,
    }
    # stamp the final score onto every row (handy for Weave + leaderboard joins)
    for r in rows:
        r["final_score"] = final["score"]
        r["score_components"] = final
    return rows, final, counts
