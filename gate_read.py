#!/usr/bin/env python3
"""gate_read.py -- read-only gate reader for outputs/results.jsonl.

Consumes the per-step JSONL trace written by run_all_conditions.py and prints,
in one shot:

  * always - naive mean delta (the mean final_score gap),
  * the per-task spread: how many tasks show a >= 0.15 gap, plus min / max,
  * adaptive vs random_at_budget avg_score at their observed intervention rates,
  * a count of any final_score == 0.0 (parse failures),
  * a verdict line (GREEN / YELLOW / RED).

This never runs a model and never writes a file -- it only reads the JSONL.
A final_score of exactly 0.0 means the final doc did not parse: the scorer
otherwise floors at 0.10*parse_valid, so 0.0 == a parse failure.

    python gate_read.py [path/to/results.jsonl]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from typing import Dict, List, Optional

DEFAULT_PATH = "outputs/results.jsonl"
CLEAR_GAP = 0.15  # per-task "clear gap" threshold (mirrors the GREEN gate)

# Canonical condition keys + the aliases the two writers use.
NAIVE, ALWAYS, ADAPTIVE, RANDOM = (
    "naive", "always_reground", "adaptive", "random_at_budget")
_ALIAS = {
    "naive": NAIVE,
    "always": ALWAYS, "always_reground": ALWAYS,
    "adaptive": ADAPTIVE,
    "random": RANDOM, "random_at_budget": RANDOM,
}
_ORDER = [NAIVE, ALWAYS, ADAPTIVE, RANDOM]


def _load(path: str) -> List[dict]:
    rows, bad = [], 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"[gate_read] skipped {bad} malformed line(s)")
    return rows


def _cond(row: dict) -> Optional[str]:
    return _ALIAS.get(row.get("condition", ""))


def _task(row: dict):
    return row.get("task_id", row.get("episode_id"))


def _episode_final(rows: List[dict]) -> float:
    """Final score for one (condition, task) episode.

    Round-trip rows stamp an identical ``final_score`` onto every step; older
    per-hop traces instead carry a running ``score``, so fall back to the score
    at the largest hop index.
    """
    stamped = [r["final_score"] for r in rows if "final_score" in r]
    if stamped:
        return float(stamped[-1])  # identical across the episode's rows
    last = max(rows, key=lambda r: r.get("hop", r.get("round_trip_index", 0)))
    return float(last.get("score", 0.0))


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    try:
        rows = _load(path)
    except FileNotFoundError:
        print(f"[gate_read] no results file at {path!r} -- run "
              "run_all_conditions.py first.")
        return 1
    if not rows:
        print(f"[gate_read] {path!r} is empty -- nothing to gate.")
        return 1

    # Group rows by condition, then by task/episode.
    by_cond: Dict[str, Dict[object, List[dict]]] = defaultdict(
        lambda: defaultdict(list))
    unknown = set()
    for r in rows:
        c = _cond(r)
        if c is None:
            unknown.add(r.get("condition"))
            continue
        by_cond[c][_task(r)].append(r)

    # Per condition: per-task final score, avg score, intervention rate, zeros.
    finals: Dict[str, Dict[object, float]] = {}
    avg_score: Dict[str, float] = {}
    interv_rate: Dict[str, float] = {}
    zeros: Dict[str, int] = {}
    for c, tasks in by_cond.items():
        finals[c] = {tid: _episode_final(trows) for tid, trows in tasks.items()}
        n = max(1, len(finals[c]))
        avg_score[c] = sum(finals[c].values()) / n
        flat = [r for trows in tasks.values() for r in trows]
        interv_rate[c] = (sum(1 for r in flat if r.get("intervened")) / len(flat)
                          if flat else 0.0)
        zeros[c] = sum(1 for v in finals[c].values() if v == 0.0)

    present = [c for c in _ORDER if c in by_cond]
    n_eps = sum(len(by_cond[c]) for c in by_cond)

    print(f"Relay gate_read -- {path}")
    print("=" * 62)
    print(f"rows={len(rows)}  episodes={n_eps}  "
          f"conditions={', '.join(present) or '(none recognized)'}")
    if unknown:
        labels = sorted(str(x) for x in unknown if x is not None)
        print(f"[gate_read] ignored unknown condition label(s): {labels}")
    print()

    # ---- always - naive mean delta -------------------------------------
    print("always - naive  (mean delta final_score)")
    mean_delta: Optional[float] = None
    if NAIVE in avg_score and ALWAYS in avg_score:
        print(f"  naive            avg_score = {avg_score[NAIVE]:.3f}  "
              f"(n={len(finals[NAIVE])})")
        print(f"  always_reground  avg_score = {avg_score[ALWAYS]:.3f}  "
              f"(n={len(finals[ALWAYS])})")
        mean_delta = avg_score[ALWAYS] - avg_score[NAIVE]
        print(f"  delta  always - naive      = {mean_delta:+.3f}")
    else:
        miss = [c for c in (NAIVE, ALWAYS) if c not in avg_score]
        print(f"  (need both naive and always_reground; missing: {', '.join(miss)})")
    print()

    # ---- per-task spread ----------------------------------------------
    print("per-task spread  (always - naive, paired tasks)")
    clear_gaps = 0
    if NAIVE in finals and ALWAYS in finals:
        shared = sorted(set(finals[NAIVE]) & set(finals[ALWAYS]), key=str)
        gaps = [finals[ALWAYS][t] - finals[NAIVE][t] for t in shared]
        if gaps:
            clear_gaps = sum(1 for g in gaps if g >= CLEAR_GAP)
            print(f"  paired tasks            : {len(gaps)}")
            print(f"  tasks with gap >= {CLEAR_GAP:.2f} : {clear_gaps}")
            print(f"  gap min / max           : {min(gaps):+.3f} / {max(gaps):+.3f}")
        else:
            print("  (no tasks shared between naive and always_reground)")
    else:
        print("  (need both naive and always_reground)")
    print()

    # ---- adaptive vs random_at_budget ---------------------------------
    print("adaptive vs random_at_budget  (avg_score @ intervention rate)")
    for c in (ADAPTIVE, RANDOM):
        if c in avg_score:
            print(f"  {c:<16} avg_score = {avg_score[c]:.3f}   "
                  f"interv_rate = {interv_rate[c]:.3f}  (n={len(finals[c])})")
        else:
            print(f"  {c:<16} (not in this run)")
    if ADAPTIVE in avg_score and RANDOM in avg_score:
        edge = avg_score[ADAPTIVE] - avg_score[RANDOM]
        print(f"  delta  adaptive - random   = {edge:+.3f}  "
              "(does the signal pick better moments?)")
    print()

    # ---- parse failures ------------------------------------------------
    total_zeros = sum(zeros.values())
    print(f"parse failures  (final_score == 0.0): {total_zeros} episode(s)")
    if total_zeros:
        for c in present:
            if zeros.get(c):
                print(f"  {c}: {zeros[c]}")
    print()

    # ---- verdict -------------------------------------------------------
    print("-" * 62)
    if mean_delta is None:
        print("VERDICT: N/A  (need naive + always_reground to gate)")
        return 0
    if mean_delta >= 0.15 or clear_gaps >= 3:
        verdict = "GREEN"
    elif mean_delta >= 0.05:
        verdict = "YELLOW"
    else:
        verdict = "RED"
    why = []
    if mean_delta >= 0.15:
        why.append(f"always - naive = {mean_delta:+.3f} >= 0.15")
    if clear_gaps >= 3:
        why.append(f"{clear_gaps} tasks >= {CLEAR_GAP:.2f} gap")
    if not why:
        why.append(f"always - naive = {mean_delta:+.3f}")
    print(f"VERDICT: {verdict}  ({'; '.join(why)})")
    print("  GREEN  if always - naive >= 0.15 OR >=3 tasks show a clear (>=0.15) gap")
    print("  YELLOW if 0.05 <= always - naive < 0.15")
    print("  RED    if always - naive < 0.05")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
