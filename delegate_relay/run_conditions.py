"""The four-condition experiment (port of relay-main/spike/probe.py).

ONE question: does adaptive re-grounding beat random re-grounding at the SAME intervention
budget, on the fidelity/cost frontier?

Same chains, four conditions:
  naive            : never re-ground
  always_reground  : re-ground every boundary (upper bound)
  adaptive         : re-ground when boundary risk > threshold (tuned to ~target-rate)
  random_at_budget : re-ground at random boundaries, count matched to adaptive

Outputs: results.jsonl (per-boundary log), leaderboard.md, frontier.png, demo_case.md, and
(best-effort) a Weave Evaluation + Leaderboard with per-boundary signals as custom scores.

Run:  WANDB_API_KEY=... WANDB_PROJECT=entity/project python run_conditions.py --n 12 --depth 4
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np

from conductor import run_chain
from data import load_chains
from domain import score
from llm import BIG_MODEL, init_weave
from signals import threshold_from_quantile

CONDITIONS = ("naive", "always_reground", "adaptive", "random_at_budget")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--target-rate", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    from agents import MOCK
    init_weave()
    chains = load_chains(n=args.n, depth=args.depth, seed=args.seed)
    tag = "MOCK (synthetic corruption, offline)" if MOCK else f"editor={BIG_MODEL}"
    if MOCK:
        print("=" * 70 + "\n  RELAY_MOCK=1 — OFFLINE pipeline validation, NOT real LLM corruption\n" + "=" * 70)
    print(f"[conditions] {len(chains)} chains | depth {args.depth} | {tag}\n")

    log: list = []
    res = {c: dict(fid=[], interv=[], tok=[], final={}) for c in CONDITIONS}

    # --- Pass 1: naive (also collects boundary risks to set the adaptive threshold) ---
    all_risks: list[float] = []
    for c in chains:
        final, ni, tok, risks = run_chain(c, mode="naive", log=log, condition="naive")
        res["naive"]["fid"].append(score(final, c.seed_text))
        res["naive"]["interv"].append(ni)
        res["naive"]["tok"].append(tok)
        res["naive"]["final"][c.sample_id] = final
        all_risks += risks
    thr = threshold_from_quantile(all_risks, args.target_rate)
    print(f"[conditions] adaptive threshold @ {1-args.target_rate:.0%}ile risk = {thr:.4f}")

    # --- always_reground ---
    for c in chains:
        final, ni, tok, _ = run_chain(c, mode="always", log=log, condition="always_reground")
        res["always_reground"]["fid"].append(score(final, c.seed_text))
        res["always_reground"]["interv"].append(ni)
        res["always_reground"]["tok"].append(tok)
        res["always_reground"]["final"][c.sample_id] = final

    # --- adaptive ---
    for ci, c in enumerate(chains):
        final, ni, tok, _ = run_chain(c, mode="adaptive", threshold=thr,
                                      log=log, condition="adaptive")
        res["adaptive"]["fid"].append(score(final, c.seed_text))
        res["adaptive"]["interv"].append(ni)
        res["adaptive"]["tok"].append(tok)
        res["adaptive"]["final"][c.sample_id] = final
    adaptive_total = sum(res["adaptive"]["interv"])

    # --- random_at_budget: same total interventions, randomly placed across boundaries ---
    rng = random.Random(args.seed)
    slots = [(ci, bi) for ci, c in enumerate(chains) for bi in range(c.depth)]
    rng.shuffle(slots)
    chosen = slots[:adaptive_total]
    forced_by_chain: dict[int, set[int]] = {}
    for ci, bi in chosen:
        forced_by_chain.setdefault(ci, set()).add(bi)
    for ci, c in enumerate(chains):
        final, ni, tok, _ = run_chain(c, mode="forced",
                                      forced_boundaries=forced_by_chain.get(ci, set()),
                                      log=log, condition="random_at_budget")
        res["random_at_budget"]["fid"].append(score(final, c.seed_text))
        res["random_at_budget"]["interv"].append(ni)
        res["random_at_budget"]["tok"].append(tok)
        res["random_at_budget"]["final"][c.sample_id] = final

    # --------------------------------------------------------------- report ----
    n = len(chains)
    rows = []
    for cond in CONDITIONS:
        r = res[cond]
        rows.append((cond, np.mean(r["fid"]), np.mean(r["interv"]), np.mean(r["tok"])))

    print("\n" + "=" * 70)
    print(f"{'condition':<18}{'fidelity':>10}{'avg_interv':>12}{'avg_tokens':>14}")
    print("-" * 70)
    for name, fid, ni, tk in rows:
        print(f"{name:<18}{fid:>10.3f}{ni:>12.2f}{tk:>14.0f}")
    print("=" * 70)
    total_boundaries = n * args.depth
    print(f"adaptive intervention rate: {adaptive_total/total_boundaries:.0%}  "
          f"(budget for random = {adaptive_total} of {total_boundaries} boundaries)")
    adv = np.mean(res["adaptive"]["fid"]) - np.mean(res["random_at_budget"]["fid"])
    print(f"adaptive − random fidelity:  {adv:+.3f}  "
          f"({'adaptive wins' if adv > 0 else 'no win this run'})")

    _write_artifacts(log, rows, res, chains, thr, adaptive_total, total_boundaries, adv, args)
    try:
        _publish_weave(rows, res, chains)
    except Exception as e:  # Weave is best-effort; local artifacts are the source of truth
        print(f"[weave] leaderboard publish skipped: {type(e).__name__}: {str(e)[:120]}")


def _write_artifacts(log, rows, res, chains, thr, adaptive_total, total_boundaries, adv, args):
    with open("results.jsonl", "w") as f:
        for e in log:
            f.write(json.dumps(e) + "\n")

    with open("leaderboard.md", "w") as f:
        from domain import DOMAIN
        mock_tag = " · OFFLINE MOCK" if __import__("agents").MOCK else ""
        f.write(f"# Relay × DELEGATE52 ({DOMAIN}{mock_tag}) — four-condition leaderboard\n\n")
        f.write(f"n={len(chains)} chains · depth={args.depth} · editor={BIG_MODEL} · "
                f"threshold={thr:.4f} · target_rate={args.target_rate}\n\n")
        f.write("| Condition | Fidelity | Avg interventions | Avg tokens |\n")
        f.write("|---|---|---|---|\n")
        for name, fid, ni, tk in rows:
            f.write(f"| {name} | {fid:.3f} | {ni:.2f} | {tk:.0f} |\n")
        f.write(f"\nadaptive intervention rate: {adaptive_total/total_boundaries:.0%}  "
                f"(random matched to {adaptive_total} interventions)\n\n")
        f.write(f"**adaptive − random fidelity = {adv:+.3f}** "
                f"({'adaptive beats random at equal budget' if adv > 0 else 'no win this run — see fallback'}).\n")

    _frontier_png(rows)

    # demo case: a chain where adaptive recovered fidelity that naive lost
    demos = []
    for ci, c in enumerate(chains):
        fn = score(res["naive"]["final"][c.sample_id], c.seed_text)
        fa = score(res["adaptive"]["final"][c.sample_id], c.seed_text)
        if fa - fn > 0.05:
            demos.append((fa - fn, c.sample_id, fn, fa))
    demos.sort(reverse=True)
    with open("demo_case.md", "w") as f:
        f.write("# Demo case — adaptive recovers what naive lost\n\n")
        if demos:
            d, sid, fn, fa = demos[0]
            f.write(f"Chain `{sid}`: naive fidelity **{fn:.3f}** → adaptive **{fa:.3f}** "
                    f"(+{d:.3f}).\n\n")
            f.write("Per-boundary trace (naive vs adaptive) is in results.jsonl; filter "
                    f"`sample_id == \"{sid}\"`. Look for the boundary where naive's "
                    "invariant_deviation spikes and adaptive's `intervened == 1`.\n")
        else:
            f.write("No adaptive-beats-naive chain with margin > 0.05 this run "
                    "(see leaderboard + fallback framing).\n")
    print(f"\nwrote results.jsonl ({len(log)} rows), leaderboard.md, frontier.png, demo_case.md")
    print(f"clean adaptive-fixes-naive chains: {len(demos)}")


def _frontier_png(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[viz] frontier.png skipped: {e}")
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, fid, _ni, tk in rows:
        ax.scatter(tk, fid, s=90)
        ax.annotate(name, (tk, fid), textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax.set_xlabel("avg tokens per chain (cost)")
    ax.set_ylabel("round-trip fidelity (vs seed)")
    ax.set_title("Relay × DELEGATE52 — fidelity / cost frontier")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("frontier.png", dpi=130)
    print("wrote frontier.png")


def _publish_weave(rows, res, chains):
    """Best-effort: a Weave Evaluation per condition (scored by ledger.score on the
    already-computed final docs, so no extra LLM calls) + a published Leaderboard."""
    import os
    if not os.environ.get("WANDB_PROJECT"):
        print("[weave] set WANDB_PROJECT to publish the Evaluation/Leaderboard")
        return
    import asyncio

    import weave
    from weave.flow import leaderboard

    from agents import MOCK
    from domain import DOMAIN
    sfx = f"_{DOMAIN}" + ("_mock" if MOCK else "")  # separate domains + synthetic vs real on W&B

    dataset = [{"sample_id": c.sample_id, "seed": c.seed_text} for c in chains]

    evals = {}
    for cond in CONDITIONS:
        finals = res[cond]["final"]

        @weave.op(name=f"relay_fidelity_{cond}{sfx}")
        def fidelity(seed: str, output: str) -> dict:
            return {"fidelity": score(output, seed)}

        class Condition(weave.Model):
            condition: str

            @weave.op()
            def predict(self, sample_id: str, seed: str) -> str:
                return finals[sample_id]

        ev = weave.Evaluation(name=f"relay_{cond}{sfx}", dataset=dataset, scorers=[fidelity])
        evals[cond] = asyncio.run(ev.evaluate(Condition(condition=cond)))

    name = f"Relay-DELEGATE52-{DOMAIN}" + ("-mock" if MOCK else "")
    spec = leaderboard.Leaderboard(
        name=name,
        description=(f"[{DOMAIN}] " + ("OFFLINE MOCK (synthetic corruption) — pipeline validation. "
                     if MOCK else "") + "Round-trip fidelity: naive / always / adaptive / random-at-budget."),
        columns=[leaderboard.LeaderboardColumn(
            evaluation_object_ref=f"relay_{c}{sfx}", scorer_name=f"relay_fidelity_{c}{sfx}",
            summary_metric_path="fidelity.mean") for c in CONDITIONS],
    )
    weave.publish(spec)
    print(f"[weave] published Evaluation runs + Leaderboard '{name}'")


if __name__ == "__main__":
    main()
