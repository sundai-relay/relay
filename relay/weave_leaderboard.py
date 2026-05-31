"""Optional: publish a Weave Evaluation + Leaderboard over the four conditions.

This is the CREDENTIALS-ONLY path. It is a safe no-op unless Weave is actually
initialized (real WANDB_API_KEY + WANDB_PROJECT). The local 4-row leaderboard
in ``leaderboard.py`` is always produced regardless.

Pattern (per the W&B Inference docs): one ``weave.Evaluation`` per condition
over the same episode dataset + a reconstruction/exact-match scorer, then a
``leaderboard.Leaderboard`` with one column per condition's ``fidelity.mean``.

Everything is wrapped in try/except so a Weave/version mismatch can never break
the actual run — it just prints a skip notice.
"""

from __future__ import annotations

from typing import Dict, List

from . import weave_compat


def maybe_publish(results: Dict[str, List[dict]], substrate, episodes,
                  threshold: float = 0.4, seed: int = 0):
    if not weave_compat.is_active():
        return None
    weave = weave_compat.weave_module()
    if weave is None:
        return None

    try:
        import asyncio

        from .conductor import (ADAPTIVE, ALWAYS, NAIVE, RANDOM, AdaptivePolicy,
                                AlwaysPolicy, NaivePolicy, RandomAtBudgetPolicy)
        from .conditions import (per_episode_intervention_counts, run_condition,
                                 run_episode)

        ep_by_id = {e.id: e for e in episodes}
        dataset = [{"episode_id": e.id} for e in episodes]

        @weave.op()
        def reconstruction_scorer(episode_id: str, output: dict) -> dict:
            ep = ep_by_id[episode_id]
            return {"fidelity": float(ep.score(output["final_state"]))}

        # random-at-budget needs adaptive's observed per-episode counts.
        adaptive_rows = results.get(ADAPTIVE)
        if adaptive_rows is None:
            adaptive_rows = run_condition(substrate, episodes, ADAPTIVE,
                                          AdaptivePolicy(threshold))
        budget = per_episode_intervention_counts(adaptive_rows)

        policies = [
            (NAIVE, NaivePolicy()),
            (ALWAYS, AlwaysPolicy()),
            (ADAPTIVE, AdaptivePolicy(threshold)),
            (RANDOM, RandomAtBudgetPolicy(budget, seed=seed)),
        ]

        def make_predict(policy):
            @weave.op()
            def predict(episode_id: str) -> dict:
                ep = ep_by_id[episode_id]
                final_state, _ = run_episode(substrate, ep, policy)
                return {"final_state": final_state}
            return predict

        evals = []
        for cond, policy in policies:
            evaluation = weave.Evaluation(name=f"relay-{cond}", dataset=dataset,
                                          scorers=[reconstruction_scorer])
            asyncio.run(evaluation.evaluate(make_predict(policy)))
            evals.append((cond, evaluation))

        _publish_leaderboard(weave, evals)
        return evals
    except Exception as e:  # pragma: no cover - creds/version dependent
        print(f"[weave] Evaluation/Leaderboard publish skipped "
              f"({type(e).__name__}: {e}).")
        return None


def _publish_leaderboard(weave, evals) -> None:
    try:
        from weave.flow import leaderboard
    except Exception as e:  # pragma: no cover
        print(f"[weave] leaderboard module unavailable ({e}).")
        return

    try:
        from weave.trace.ref_util import get_ref
    except Exception:  # pragma: no cover - API moved across versions
        get_ref = None

    columns = []
    for cond, evaluation in evals:
        ref = None
        if get_ref is not None:
            try:
                ref = get_ref(evaluation).uri()
            except Exception:
                ref = None
        if ref is None:
            continue
        columns.append(leaderboard.LeaderboardColumn(
            evaluation_object_ref=ref,
            scorer_name="reconstruction_scorer",
            summary_metric_path="fidelity.mean",
        ))

    if not columns:
        print("[weave] could not resolve evaluation refs; leaderboard skipped "
              "(evaluations still logged).")
        return

    spec = leaderboard.Leaderboard(
        name="Relay — handoff-degradation (4 conditions)",
        description=("naive / always / adaptive / random_at_budget over the same "
                     "episodes. The project is adaptive vs. random-at-budget."),
        columns=columns,
    )
    weave.publish(spec)
    print("[weave] published Evaluation + Leaderboard for the 4 conditions.")
