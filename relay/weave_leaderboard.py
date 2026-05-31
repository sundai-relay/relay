"""Optional: publish a Weave Evaluation + Leaderboard over the four conditions.

This is the CREDENTIALS-ONLY path. It is a safe no-op unless Weave is actually
initialized (real WANDB_API_KEY + WANDB_PROJECT). The local 4-row leaderboard
in ``leaderboard.py`` is always produced regardless.

Pattern (per the W&B Inference docs): one ``weave.Evaluation`` per condition
over the same episode dataset + a reconstruction/exact-match scorer, then a
``leaderboard.Leaderboard`` with one column per condition's ``fidelity.mean``.

The evaluations wrap *already-computed* results — nothing is re-run. This avoids
doubling computation/API costs and sidesteps ``asyncio.run()`` issues when an
event loop is already active (e.g. inside Weave tracing or Jupyter).

Everything is wrapped in try/except so a Weave/version mismatch can never break
the actual run — it just prints a skip notice.
"""

from __future__ import annotations

from typing import Any, Dict, List

from . import weave_compat


def maybe_publish(results: Dict[str, List[dict]], substrate, episodes,
                  threshold: float = 0.4, seed: int = 0):
    """Publish pre-computed condition results as Weave Evaluations + Leaderboard.

    ``results`` is ``{condition_name: [per_hop_row_dicts, ...]}``, as returned
    by ``run_conditions()``.  Each row has at least ``episode_id`` and ``score``.
    We aggregate the last-hop score per episode and surface it as the
    ``fidelity`` metric in each Evaluation.
    """
    if not weave_compat.is_active():
        return None
    weave = weave_compat.weave_module()
    if weave is None:
        return None

    try:
        import asyncio

        from .conductor import ADAPTIVE, ALWAYS, NAIVE, RANDOM

        # Build a lookup: condition -> {episode_id: final_score}.
        # Each condition's rows are ordered by episode & hop; the *last* row per
        # episode_id carries the final per-hop score (== episode fidelity).
        def _final_scores(rows: List[dict]) -> Dict[str, float]:
            scores: Dict[str, float] = {}
            for r in rows:
                scores[r["episode_id"]] = r["score"]  # last write wins
            return scores

        ep_ids = [e.id for e in episodes]
        dataset = [{"episode_id": eid} for eid in ep_ids]

        @weave.op()
        def reconstruction_scorer(episode_id: str, output: dict) -> dict:
            return {"fidelity": output["score"]}

        canonical_order = [NAIVE, ALWAYS, ADAPTIVE, RANDOM]
        evals = []
        for cond in canonical_order:
            if cond not in results:
                continue
            finals = _final_scores(results[cond])

            def make_predict(cond_finals: Dict[str, float]):
                @weave.op()
                def predict(episode_id: str) -> dict:
                    return {"score": cond_finals.get(episode_id, 0.0)}
                return predict

            evaluation = weave.Evaluation(
                name=f"relay-{cond}",
                dataset=dataset,
                scorers=[reconstruction_scorer],
            )
            # Use get_or_create_event_loop pattern to avoid
            # "RuntimeError: This event loop is already running".
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import nest_asyncio  # type: ignore
                nest_asyncio.apply()

            asyncio.run(evaluation.evaluate(make_predict(finals)))
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
