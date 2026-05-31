#!/usr/bin/env python3
"""Real DELEGATE52 four-condition round-trip runner (accounting + chess).

Runs naive / always_reground / adaptive / random_at_budget over the SAME real
samples, the dataset's own forward/backward edit pairs, the same round-trip
count, and one fixed editor model. Only the intervention policy changes.

The adaptive threshold is, by default, set automatically from the observed naive
risk distribution to hit a target intervention rate (quantile), so it is no
longer a hand-set magic number; pass --threshold to override.

    python run_delegate.py --domain both --live --model microsoft/Phi-4-mini-instruct \
        --num-round-trips 5 --target-rate 0.30
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np

from relay.delegate import agents, ledger, pgn
from relay.delegate.loader import load_delegate_tasks
from relay.delegate.runner import run_task
from relay.wandb_client import WEAK_MODEL

CANON = ["naive", "always_reground", "adaptive", "random_at_budget"]
DOMAINS = {"accounting": ledger, "chess": pgn}


def _aggregate(cond: str, per_task: Dict[str, tuple]) -> dict:
    scores = [v[1]["score"] for v in per_task.values()]
    interventions = [v[2]["interventions"] for v in per_task.values()]
    steps = sum(v[2]["steps"] for v in per_task.values())
    costs = [v[2]["cost_proxy"] for v in per_task.values()]
    n = max(1, len(per_task))
    total_int = sum(interventions)
    avg_score = sum(scores) / n
    avg_cost = sum(costs) / n
    return {"condition": cond, "avg_score": avg_score,
            "avg_interventions": total_int / n,
            "intervention_rate": (total_int / steps) if steps else 0.0,
            "cost_proxy": avg_cost,
            "score_per_cost": (avg_score / avg_cost) if avg_cost else 0.0,
            "n_tasks": len(per_task), "total_steps": steps}


def _run_condition(domain, cond, tasks, threshold, random_rate, rng_seed, nrt,
                   sink=None) -> Dict[str, tuple]:
    out: Dict[str, tuple] = {}
    for t in tasks:
        out[t.task_id] = run_task(domain, t, cond, threshold=threshold,
                                  random_rate=random_rate, rng_seed=rng_seed,
                                  num_round_trips=nrt)
        if sink is not None:
            sink(cond, out[t.task_id][0])
    return out


def _auto_threshold(domain, tasks, target_rate, rng_seed, nrt, sink):
    """Run naive, collect per-step risks, return (results, threshold)."""
    res = _run_condition(domain, "naive", tasks, 0.0, None, rng_seed, nrt, sink)
    risks = [r["runtime_risk"] for v in res.values() for r in v[0]]
    thr = float(np.quantile(risks, 1 - target_rate)) if risks else 0.0
    return res, thr, risks


def _run_domain(domain, tasks, args, results_path_sink) -> List[dict]:
    requested = [c for c in CANON if c in set(args.conditions)]
    results: Dict[str, Dict[str, tuple]] = {}

    # naive first (also calibrates the auto threshold).
    naive_res, auto_thr, risks = _auto_threshold(
        domain, tasks, args.target_rate, args.rng_seed, args.num_round_trips,
        results_path_sink if "naive" in requested else None)
    results["naive"] = naive_res
    threshold = args.threshold if args.threshold is not None else auto_thr
    print(f"[{domain.name}] naive per-step risk: "
          f"min={min(risks):.3f} med={np.median(risks):.3f} max={max(risks):.3f} "
          f"| adaptive threshold={threshold:.4f} "
          f"({'manual' if args.threshold is not None else f'auto @ {args.target_rate:.0%} rate'})")

    if "always_reground" in requested:
        results["always_reground"] = _run_condition(
            domain, "always_reground", tasks, threshold, None, args.rng_seed,
            args.num_round_trips, results_path_sink)

    need_adaptive = "adaptive" in requested or "random_at_budget" in requested
    if need_adaptive:
        results["adaptive"] = _run_condition(
            domain, "adaptive", tasks, threshold, None, args.rng_seed,
            args.num_round_trips,
            results_path_sink if "adaptive" in requested else None)

    if "random_at_budget" in requested:
        rate = _aggregate("adaptive", results["adaptive"])["intervention_rate"]
        print(f"[{domain.name}] adaptive intervention rate = {rate:.3f} "
              f"-> random-at-budget matches it")
        results["random_at_budget"] = _run_condition(
            domain, "random_at_budget", tasks, threshold, rate,
            args.random_seed, args.num_round_trips, results_path_sink)

    report_conds = [c for c in CANON if c in requested]
    return [_aggregate(c, results[c]) for c in report_conds], results


def _print_leaderboard(domain_name, summaries):
    print(f"\nRelay × DELEGATE52 — {domain_name} round-trip leaderboard")
    h = (f"{'condition':<18} | {'avg_score':>9} | {'avg_interv':>10} | "
         f"{'interv_rate':>11} | {'cost':>6} | {'score/cost':>10}")
    print(h); print("-" * len(h))
    for s in summaries:
        print(f"{s['condition']:<18} | {s['avg_score']:>9.3f} | "
              f"{s['avg_interventions']:>10.2f} | {s['intervention_rate']:>11.3f} | "
              f"{s['cost_proxy']:>6.1f} | {s['score_per_cost']:>10.4f}")


def _gate(domain_name, summaries):
    by = {s["condition"]: s for s in summaries}
    if "naive" in by and "always_reground" in by:
        gap = by["always_reground"]["avg_score"] - by["naive"]["avg_score"]
        verdict = "GREEN" if gap >= 0.15 else ("YELLOW" if gap >= 0.05 else "RED")
        print(f"[{domain_name} gate] always_reground - naive = {gap:+.3f} -> {verdict} "
              "(GREEN>=0.15, YELLOW 0.05-0.15, RED<0.05)")
    if "adaptive" in by and "random_at_budget" in by:
        d = by["adaptive"]["avg_score"] - by["random_at_budget"]["avg_score"]
        print(f"[{domain_name}] adaptive - random_at_budget = {d:+.3f} "
              "(the decision-value result)")


def _write_md(all_summaries, path):
    lines = ["# Relay × DELEGATE52 — round-trip leaderboards", ""]
    for dom, summaries in all_summaries.items():
        lines += [f"## {dom}", "",
                  "| condition | avg_score | avg_interventions | intervention_rate | cost_proxy | score_per_cost |",
                  "|---|---|---|---|---|---|"]
        for s in summaries:
            lines.append(f"| {s['condition']} | {s['avg_score']:.3f} | "
                         f"{s['avg_interventions']:.2f} | {s['intervention_rate']:.3f} | "
                         f"{s['cost_proxy']:.1f} | {s['score_per_cost']:.4f} |")
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Real DELEGATE52 four-condition runner")
    p.add_argument("--domain", choices=("accounting", "chess", "both"),
                   default="both")
    p.add_argument("--n", type=int, default=None, help="max samples per domain")
    p.add_argument("--num-round-trips", type=int, default=5,
                   help="depth: forward+inverse cycles per task (compounding)")
    p.add_argument("--threshold", type=float, default=None,
                   help="manual adaptive risk threshold (else auto from --target-rate)")
    p.add_argument("--target-rate", type=float, default=0.30,
                   help="auto-threshold targets this naive-risk quantile")
    p.add_argument("--rng-seed", type=int, default=42)
    p.add_argument("--random-seed", type=int, default=123)
    p.add_argument("--conditions", nargs="+", default=CANON)
    p.add_argument("--model", default=None)
    p.add_argument("--provider", choices=("wandb", "openai"), default="wandb")
    p.add_argument("--max-tokens", type=int, default=4096)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--mock", action="store_true")
    mode.add_argument("--live", action="store_true")
    p.add_argument("--out-dir", default="outputs_delegate")
    args = p.parse_args()

    model = args.model or WEAK_MODEL
    use_mock = True if args.mock else (False if args.live else None)
    use_mock = agents.configure(use_mock=use_mock, model=model,
                                provider=args.provider, max_tokens=args.max_tokens)
    backend = "MOCK" if use_mock else f"LIVE/{args.provider}:{model}"

    domains = (["accounting", "chess"] if args.domain == "both" else [args.domain])
    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, "results.jsonl")
    rf = open(results_path, "w")

    def _sink(cond, rows):
        for r in rows:
            rf.write(json.dumps(r) + "\n")
        rf.flush()

    all_summaries: Dict[str, list] = {}
    all_results: Dict[str, dict] = {}
    for dom_name in domains:
        domain = DOMAINS[dom_name]
        tasks = load_delegate_tasks(dom_name, max_samples=args.n)
        print(f"\n[{dom_name}] tasks={len(tasks)} round_trips={args.num_round_trips} "
              f"mode={backend}")
        summaries, results = _run_domain(domain, tasks, args, _sink)
        all_summaries[dom_name] = summaries
        all_results[dom_name] = results
        _print_leaderboard(dom_name, summaries)
        _gate(dom_name, summaries)

    rf.close()
    _write_md(all_summaries, os.path.join(args.out_dir, "leaderboard.md"))
    _write_demo(all_results, os.path.join(args.out_dir, "demo_case.md"))
    print(f"\n[relay] outputs -> {args.out_dir}/ (results.jsonl, leaderboard.md, demo_case.md)")


def _write_demo(all_results, path):
    lines = ["# Relay × DELEGATE52 — demo cases (adaptive recovers what naive lost)", ""]
    for dom, results in all_results.items():
        if "naive" not in results or "adaptive" not in results:
            continue
        best, gap = None, -1.0
        for tid in results["naive"]:
            na = results["naive"][tid][1]["score"]
            ad = results["adaptive"][tid][1]["score"]
            if ad - na > gap:
                gap, best = ad - na, tid
        if best is None:
            continue
        na_rows, na_final, _ = results["naive"][best]
        ad_rows, ad_final, _ = results["adaptive"][best]
        lines += [f"## {dom} — `{best}`  (adaptive − naive = {gap:+.3f})", "",
                  f"naive final score {na_final['score']:.3f} | "
                  f"adaptive final score {ad_final['score']:.3f}", "",
                  "Per-round-trip runtime risk under naive (no repair):", "",
                  "| round | edit | risk | n_seed→n_cur |", "|---|---|---|---|"]
        for r in na_rows:
            lines.append(f"| {r['round_trip_index']} | {r['edit_name']} | "
                         f"{r['runtime_risk']:.3f} | {r['n_seed']}→{r['n_cur']} |")
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
