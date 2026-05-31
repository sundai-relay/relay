"""Aggregation, the 4-row leaderboard print, and JSONL logging."""

from __future__ import annotations

import json
import os
from typing import Dict, List

from .conductor import ADAPTIVE, ALWAYS, NAIVE, RANDOM

ORDER = [NAIVE, ALWAYS, ADAPTIVE, RANDOM]


def aggregate(results: Dict[str, List[dict]]) -> List[dict]:
    """Per condition: mean final score, avg interventions, avg token proxy."""
    summary: List[dict] = []
    for cond, rows in results.items():
        by_ep: Dict[str, dict] = {}
        for row in rows:
            e = by_ep.setdefault(
                row["episode_id"],
                {"final": 0.0, "max_hop": -1, "intervened": 0, "tokens": 0.0},
            )
            if row["hop"] > e["max_hop"]:
                e["max_hop"] = row["hop"]
                e["final"] = row["score"]  # last hop's score = final fidelity
            e["intervened"] += 1 if row["intervened"] else 0
            e["tokens"] += row["token_proxy"]

        n = max(1, len(by_ep))
        summary.append({
            "condition": cond,
            "mean_score": sum(e["final"] for e in by_ep.values()) / n,
            "avg_interventions": sum(e["intervened"] for e in by_ep.values()) / n,
            "avg_token_proxy": sum(e["tokens"] for e in by_ep.values()) / n,
            "n_episodes": len(by_ep),
        })

    summary.sort(key=lambda s: ORDER.index(s["condition"]) if s["condition"] in ORDER else 99)
    return summary


def print_leaderboard(summary: List[dict], title: str = "Relay — leaderboard") -> None:
    print()
    print(title)
    header = (f"{'condition':<18} | {'mean_score':>10} | "
              f"{'avg_interventions':>17} | {'avg_token_proxy':>15}")
    print(header)
    print("-" * len(header))
    for s in summary:
        print(f"{s['condition']:<18} | {s['mean_score']:>10.3f} | "
              f"{s['avg_interventions']:>17.2f} | {s['avg_token_proxy']:>15.1f}")
    print()


def write_jsonl(results: Dict[str, List[dict]], path: str = "outputs/results.jsonl") -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for cond in results:
            for row in results[cond]:
                f.write(json.dumps(row) + "\n")
    return path
