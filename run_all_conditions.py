#!/usr/bin/env python3
"""Relay — round-trip four-condition runner (the locked-spec entry point).

Runs naive / always_reground / adaptive / random_at_budget over the SAME tasks,
edit sequence, round-trip count, and model settings (temperature 0). Only the
intervention policy changes. Adaptive runs before random-at-budget so random can
match adaptive's *observed* intervention rate.

No API key -> a deterministic mock editor/repairer runs (GREEN, zero credits).
With WANDB_API_KEY + WANDB_PROJECT -> real W&B Inference + Weave tracing.

Examples
--------
    # The naive-vs-always gate (offline, no key):
    python run_all_conditions.py --n 5 --conditions naive always_reground

    # Full four-condition run:
    python run_all_conditions.py --n 20
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

from relay import weave_compat
from relay.roundtrip import agents
from relay.roundtrip.runner import run_task
from relay.roundtrip.tasks import load_tasks
from relay.wandb_client import WEAK_MODEL

CANON = ["naive", "always_reground", "adaptive", "random_at_budget"]


# --------------------------------------------------------------------------- #
def _aggregate(cond: str, per_task: Dict[str, tuple]) -> dict:
    scores, interventions, steps, costs = [], [], 0, []
    for _tid, (rows, final, counts) in per_task.items():
        scores.append(final["score"])
        interventions.append(counts["interventions"])
        steps += counts["steps"]
        costs.append(counts["cost_proxy"])
    n = max(1, len(per_task))
    total_int = sum(interventions)
    avg_score = sum(scores) / n
    avg_cost = sum(costs) / n
    return {
        "condition": cond,
        "avg_score": avg_score,
        "avg_interventions": total_int / n,
        "intervention_rate": (total_int / steps) if steps else 0.0,
        "cost_proxy": avg_cost,
        "score_per_cost": (avg_score / avg_cost) if avg_cost else 0.0,
        "n_tasks": len(per_task),
        "total_interventions": total_int,
        "total_steps": steps,
    }


def _run_condition(cond, tasks, threshold, random_rate, rng_seed, nrt) -> Dict[str, tuple]:
    out = {}
    for t in tasks:
        rows, final, counts = run_task(t, cond, threshold=threshold,
                                       random_rate=random_rate, rng_seed=rng_seed,
                                       num_round_trips=nrt)
        out[t.task_id] = (rows, final, counts)
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="Relay round-trip four-condition runner")
    p.add_argument("--n", type=int, default=20, help="number of tasks")
    p.add_argument("--rng-seed", type=int, default=42, help="task-generation seed")
    p.add_argument("--threshold", type=float, default=0.008,
                   help="adaptive risk threshold. NOTE: the checksum risk is "
                        "small-magnitude (a single dropped record among ~17 is "
                        "~0.02), so this is much lower than a 0-1 'confidence' "
                        "threshold. Tuned for ~25-35%% interventions.")
    p.add_argument("--num-round-trips", type=int, default=4)
    p.add_argument("--random-seed", type=int, default=123,
                   help="seed for random-at-budget decisions")
    p.add_argument("--conditions", nargs="+", default=CANON,
                   help="subset of: " + " ".join(CANON))
    p.add_argument("--slip-p", type=float, default=0.6,
                   help="(mock) probability an edit loses something")
    p.add_argument("--model", default=WEAK_MODEL)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--mock", action="store_true", help="force the mock editor")
    mode.add_argument("--live", action="store_true", help="force real W&B Inference")
    p.add_argument("--out-dir", default="outputs")
    args = p.parse_args()

    use_mock = True if args.mock else (False if args.live else None)
    use_mock = agents.configure(use_mock=use_mock, model=args.model, slip_p=args.slip_p)
    weave_active = weave_compat.maybe_init()

    requested = [c for c in CANON if c in set(args.conditions)]
    unknown = set(args.conditions) - set(CANON)
    if unknown:
        print(f"[relay] ignoring unknown conditions: {sorted(unknown)}")
    if not requested:
        print(f"[relay] no valid conditions in {args.conditions}; choose from {CANON}")
        return

    tasks = load_tasks(args.n, args.rng_seed)
    print(f"[relay] roundtrip: tasks={len(tasks)} round_trips={args.num_round_trips} "
          f"threshold={args.threshold} mode={'MOCK' if use_mock else 'LIVE'} "
          f"weave={'on' if weave_active else 'off'}")

    results: Dict[str, Dict[str, tuple]] = {}

    # naive / always / adaptive in order; adaptive needed before random.
    for cond in ("naive", "always_reground", "adaptive"):
        if cond in requested or (cond == "adaptive" and "random_at_budget" in requested):
            results[cond] = _run_condition(cond, tasks, args.threshold, None,
                                           args.random_seed, args.num_round_trips)

    if "random_at_budget" in requested:
        adaptive_summary = _aggregate("adaptive", results["adaptive"])
        random_rate = adaptive_summary["intervention_rate"]
        print(f"[relay] adaptive observed intervention rate = {random_rate:.3f} "
              f"-> random-at-budget matches it")
        results["random_at_budget"] = _run_condition(
            "random_at_budget", tasks, args.threshold, random_rate,
            args.random_seed, args.num_round_trips)

    # keep only requested conditions in the report (adaptive may have been run
    # only to budget random).
    report_conds = [c for c in CANON if c in requested]
    summaries = [_aggregate(c, results[c]) for c in report_conds]

    os.makedirs(args.out_dir, exist_ok=True)
    _write_jsonl(results, report_conds, os.path.join(args.out_dir, "results.jsonl"))
    # Sidecar per-task summary carries final_doc (the lean trace strips it) so the
    # standalone `python -m relay.weave_leaderboard` re-score has a document to score.
    _write_task_summary(results, report_conds, os.path.join(args.out_dir, "task_summary.jsonl"))
    _write_leaderboard_md(summaries, os.path.join(args.out_dir, "leaderboard.md"))
    _print_leaderboard(summaries)
    _maybe_gate(summaries)
    _write_frontier(summaries, os.path.join(args.out_dir, "frontier.png"))
    _write_demo_case(results, report_conds, os.path.join(args.out_dir, "demo_case.md"))

    print(f"[relay] outputs -> {args.out_dir}/ "
          "(results.jsonl, task_summary.jsonl, leaderboard.md, frontier.*, demo_case.md)")

    # Publish a Weave Evaluation + Leaderboard from the in-memory results (full
    # final_doc fidelity, no extra model calls). A failure here never destroys the
    # run's other outputs.
    try:
        from relay.weave_leaderboard import (from_run_results, run_evaluations,
                                             publish_leaderboard)
        results_map = from_run_results(results, report_conds)
        evaluation, _ = run_evaluations(tasks, results_map)
        if evaluation is not None:
            publish_leaderboard(evaluation)
    except Exception as e:
        print(f"[relay] leaderboard step failed, run outputs intact: "
              f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
def _write_jsonl(results, conds, path):
    with open(path, "w") as f:
        for cond in conds:
            for _tid, (rows, _final, _counts) in results[cond].items():
                for r in rows:
                    lean = {k: v for k, v in r.items()
                            if k not in ("current_doc", "score_components", "instruction")}
                    f.write(json.dumps(lean) + "\n")


def _write_task_summary(results, conds, path):
    """One row per (condition, task): keeps final_doc so the leaderboard tool can
    re-score from disk without re-running the workflow."""
    with open(path, "w") as f:
        for cond in conds:
            for _tid, (_rows, final, counts) in results[cond].items():
                f.write(json.dumps({
                    "condition": cond,
                    "task_id": _tid,
                    "final_doc": counts["final_doc"],
                    "interventions": counts["interventions"],
                    "n_steps": counts["steps"],
                    "cost_proxy": counts["cost_proxy"],
                    "final_score": final["score"],
                }) + "\n")


def _print_leaderboard(summaries):
    print()
    print("Relay — round-trip leaderboard")
    h = (f"{'condition':<18} | {'avg_score':>9} | {'avg_interv':>10} | "
         f"{'interv_rate':>11} | {'cost':>6} | {'score/cost':>10}")
    print(h)
    print("-" * len(h))
    for s in summaries:
        print(f"{s['condition']:<18} | {s['avg_score']:>9.3f} | "
              f"{s['avg_interventions']:>10.2f} | {s['intervention_rate']:>11.3f} | "
              f"{s['cost_proxy']:>6.1f} | {s['score_per_cost']:>10.4f}")
    print()


def _write_leaderboard_md(summaries, path):
    lines = ["# Relay — round-trip leaderboard", "",
             "| condition | avg_score | avg_interventions | intervention_rate | cost_proxy | score_per_cost |",
             "|---|---|---|---|---|---|"]
    for s in summaries:
        lines.append(f"| {s['condition']} | {s['avg_score']:.3f} | "
                     f"{s['avg_interventions']:.2f} | {s['intervention_rate']:.3f} | "
                     f"{s['cost_proxy']:.1f} | {s['score_per_cost']:.4f} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _maybe_gate(summaries):
    by = {s["condition"]: s for s in summaries}
    if "naive" in by and "always_reground" in by:
        gap = by["always_reground"]["avg_score"] - by["naive"]["avg_score"]
        verdict = "GREEN" if gap >= 0.15 else ("YELLOW" if gap >= 0.05 else "RED")
        print(f"[gate] always_reground - naive = {gap:+.3f}  ->  {verdict} "
              "(GREEN >= 0.15, YELLOW 0.05-0.15, RED < 0.05)")
        if verdict == "RED":
            print("[gate] RED: increase --num-round-trips, raise --slip-p, or make "
                  "edits harder before building adaptive deeper.")


def _write_frontier(summaries, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        alt = path.rsplit(".", 1)[0] + ".txt"
        with open(alt, "w") as f:
            f.write("condition\tcost_proxy\tavg_score\n")
            for s in summaries:
                f.write(f"{s['condition']}\t{s['cost_proxy']:.2f}\t{s['avg_score']:.4f}\n")
        print(f"[relay] matplotlib not available -> wrote {alt} instead of frontier.png")
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for s in summaries:
        ax.scatter(s["cost_proxy"], s["avg_score"], s=90)
        ax.annotate(s["condition"], (s["cost_proxy"], s["avg_score"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("avg cost proxy (edit + repair calls)")
    ax.set_ylabel("avg structural reconstruction score")
    ax.set_title("Relay — fidelity vs cost frontier")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def _write_demo_case(results, conds, path):
    if "naive" not in results or "adaptive" not in results:
        return
    # task with the largest adaptive-over-naive gap.
    best_tid, best_gap = None, -1.0
    for tid in results["naive"]:
        na = results["naive"][tid][1]["score"]
        ad = results["adaptive"][tid][1]["score"]
        if ad - na > best_gap:
            best_gap, best_tid = ad - na, tid
    if best_tid is None:
        return

    na_rows, na_final, _ = results["naive"][best_tid]
    ad_rows, ad_final, _ = results["adaptive"][best_tid]
    al_final = results.get("always_reground", {}).get(best_tid, (None, {"score": float("nan")}, None))[1]

    # the clearest repair moment in adaptive: highest-risk step that intervened.
    spike = max((r for r in ad_rows if r["intervened"]),
                key=lambda r: r["runtime_risk"], default=None)

    lines = [
        f"# Relay demo case — `{best_tid}`", "",
        f"Adaptive beats naive on this task by **{best_gap:+.3f}** structural score.", "",
        "| condition | final score | id_f1 | scalar_fidelity |",
        "|---|---|---|---|",
        f"| naive | {na_final['score']:.3f} | {na_final['id_f1']:.3f} | {na_final['scalar_value_fidelity']:.3f} |",
        f"| adaptive | {ad_final['score']:.3f} | {ad_final['id_f1']:.3f} | {ad_final['scalar_value_fidelity']:.3f} |",
        f"| always_reground | {al_final['score']:.3f} | {al_final.get('id_f1', float('nan')):.3f} | {al_final.get('scalar_value_fidelity', float('nan')):.3f} |",
        "",
        "## Where naive silently lost fidelity",
        "",
        "Per-step runtime risk under **naive** (no repairs):", "",
        "| round | step | risk | id_loss | key_loss | missing_ids |",
        "|---|---|---|---|---|---|",
    ]
    for r in na_rows:
        lines.append(f"| {r['round_trip_index']} | {r['step_type']} | "
                     f"{r['runtime_risk']:.3f} | {r['id_loss_rate']:.3f} | "
                     f"{r['required_key_loss_rate']:.3f} | "
                     f"{','.join(r['missing_ids']) or '-'} |")
    lines += ["", "## The adaptive repair", ""]
    if spike is not None:
        lines += [
            f"Risk spiked to **{spike['runtime_risk']:.3f}** at round "
            f"{spike['round_trip_index']} ({spike['step_type']}, edit "
            f"`{spike['edit_name']}`); the Conductor fired a targeted repair "
            f"(risk_after = {spike['risk_after']:.3f}).",
        ]
    else:
        lines += ["Adaptive did not need to intervene on this task."]
    lines += ["", "_Generated by run_all_conditions.py; gold answers were used "
              "only for the final score, never by the runtime policy._", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
