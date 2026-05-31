"""Publish a Weave **Evaluation + Leaderboard** over the four round-trip policies.

Task A1 (Best-Use-of-Weave entry). Two entry points:

* **Standalone CLI** — ``python -m relay.weave_leaderboard`` reads the already
  written ``outputs/results.jsonl`` (+ optional ``outputs/task_summary.jsonl``)
  and publishes an Evaluation + Leaderboard. **Pure lookup + local scoring: no
  model calls, no credits.**
* **Wired into ``run_all_conditions.py``** — after a live run finishes, the
  in-memory results are published automatically (full ``final_doc`` fidelity).

Design: one ``weave.Evaluation`` (dataset = the tasks, scorers = structural
score / intervention rate / cost), evaluated with **four "models," one per
condition**, where each model is a *pure lookup* of that condition's stored
``final_doc``. Then a ``Leaderboard`` referencing that Evaluation.

Everything is import-safe **without** ``weave`` installed (the ``weave_compat``
shim turns ``@op()`` into a pass-through), and every Weave step is wrapped so a
version/credential problem can never destroy the run's other outputs — the
``outputs/leaderboard.md`` table is always written regardless.

The legacy ``maybe_publish`` at the bottom is the older substrate/episode path
still imported by ``run.py``; it is kept for backward-compatibility.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from . import weave_compat
from .weave_compat import op

# scorer + task generator live under relay.roundtrip (NOT relay.scorer / relay.tasks)
from .roundtrip.scorer import final_structural_score
from .roundtrip.tasks import load_tasks

CONDITIONS = ["naive", "always_reground", "adaptive", "random_at_budget"]


# --------------------------------------------------------------------------- #
# 2. Inputs / data contract
# --------------------------------------------------------------------------- #
@dataclass
class TaskResult:
    condition: str
    task_id: str
    final_doc: str          # final reconstructed JSON (string); "" if not stored
    interventions: int      # repair calls fired this task
    n_steps: int            # total hops where intervention was possible
    cost_proxy: float       # edit_calls + repair_calls
    final_score: float      # the score already computed by the run (scalar)


def from_run_results(results: Dict[str, Dict[str, tuple]],
                     conditions: Optional[List[str]] = None
                     ) -> "OrderedDict[Tuple[str, str], TaskResult]":
    """Adapt ``run_all_conditions``' in-memory results to the lookup map.

    ``results`` is ``{condition: {task_id: (rows, final, counts)}}``. This path
    keeps the full ``final_doc`` (counts["final_doc"]) so the Weave scorers can
    re-score with the real reconstructor.
    """
    conditions = conditions or list(results.keys())
    out: "OrderedDict[Tuple[str, str], TaskResult]" = OrderedDict()
    for cond in conditions:
        for task_id, (_rows, final, counts) in results.get(cond, {}).items():
            out[(cond, task_id)] = TaskResult(
                condition=cond,
                task_id=task_id,
                final_doc=counts.get("final_doc", ""),
                interventions=int(counts.get("interventions", 0)),
                n_steps=int(counts.get("steps", 0)),
                cost_proxy=float(counts.get("cost_proxy", 0.0)),
                final_score=float(final.get("score", 0.0)),
            )
    return out


def load_results(path: str = "outputs/results.jsonl",
                 summary_path: Optional[str] = None
                 ) -> "OrderedDict[Tuple[str, str], TaskResult]":
    """Return ``{(condition, task_id): TaskResult}`` from disk.

    Prefers a sidecar per-task summary (``outputs/task_summary.jsonl``) because
    it carries ``final_doc`` (needed to re-score). Falls back to the per-hop
    ``results.jsonl``: groups by (condition, task_id); interventions =
    sum(intervened); n_steps = hop count; cost_proxy = max cumulative cost;
    final_score = the scalar already stamped on each hop. ``final_doc`` is "" in
    the fallback because the lean trace strips the document text — the scorers
    then pass through the stored ``final_score`` (see ``structural_score``).
    """
    if summary_path is None:
        summary_path = os.path.join(os.path.dirname(path) or ".", "task_summary.jsonl")

    if os.path.exists(summary_path):
        return _load_summary(summary_path)
    return _load_per_hop(path)


def _load_summary(path: str) -> "OrderedDict[Tuple[str, str], TaskResult]":
    out: "OrderedDict[Tuple[str, str], TaskResult]" = OrderedDict()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cond, tid = d["condition"], d["task_id"]
            out[(cond, tid)] = TaskResult(
                condition=cond,
                task_id=tid,
                final_doc=d.get("final_doc", ""),
                interventions=int(d.get("interventions", 0)),
                n_steps=int(d.get("n_steps", 0)),
                cost_proxy=float(d.get("cost_proxy", 0.0)),
                final_score=float(d.get("final_score", 0.0)),
            )
    return out


def _load_per_hop(path: str) -> "OrderedDict[Tuple[str, str], TaskResult]":
    grouped: "OrderedDict[Tuple[str, str], dict]" = OrderedDict()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r["condition"], r["task_id"])
            g = grouped.setdefault(key, {
                "n_steps": 0, "interventions": 0, "cost_proxy": 0.0,
                "final_score": 0.0, "final_doc": "",
            })
            g["n_steps"] += 1
            g["interventions"] += 1 if r.get("intervened") else 0
            g["cost_proxy"] = max(g["cost_proxy"], float(r.get("cost_proxy", 0.0)))
            # final_score is stamped identically on every hop of the task.
            g["final_score"] = float(r.get("final_score", g["final_score"]))
            # current_doc is normally stripped from the lean trace, but use it if present.
            if r.get("current_doc"):
                g["final_doc"] = r["current_doc"]

    out: "OrderedDict[Tuple[str, str], TaskResult]" = OrderedDict()
    for (cond, tid), g in grouped.items():
        out[(cond, tid)] = TaskResult(
            condition=cond, task_id=tid, final_doc=g["final_doc"],
            interventions=g["interventions"], n_steps=g["n_steps"],
            cost_proxy=g["cost_proxy"], final_score=g["final_score"],
        )
    return out


# --------------------------------------------------------------------------- #
# 5. Guaranteed fallback — local summary + markdown leaderboard (always works)
# --------------------------------------------------------------------------- #
def summarize(results: "Dict[Tuple[str, str], TaskResult]") -> List[dict]:
    """Per-condition aggregates, ordered by CONDITIONS. Pure Python, no Weave."""
    by_cond: "OrderedDict[str, List[TaskResult]]" = OrderedDict()
    for (cond, _tid), tr in results.items():
        by_cond.setdefault(cond, []).append(tr)

    rows: List[dict] = []
    for cond in sorted(by_cond, key=lambda c: CONDITIONS.index(c)
                       if c in CONDITIONS else 99):
        trs = by_cond[cond]
        n = max(1, len(trs))
        total_steps = sum(t.n_steps for t in trs)
        total_int = sum(t.interventions for t in trs)
        avg_score = sum(t.final_score for t in trs) / n
        avg_cost = sum(t.cost_proxy for t in trs) / n
        rows.append({
            "condition": cond,
            "avg_score": avg_score,
            "intervention_rate": (total_int / total_steps) if total_steps else 0.0,
            "cost_proxy": avg_cost,
            "score_per_cost": (avg_score / avg_cost) if avg_cost else 0.0,
            "n_tasks": len(trs),
        })
    return rows


def write_markdown_leaderboard(summaries: List[dict],
                               path: str = "outputs/leaderboard.md") -> str:
    """The guaranteed artifact (README/demo) regardless of the Weave API."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = [
        "# Relay — fidelity vs cost by policy", "",
        "| condition | avg_score | intervention_rate | cost_proxy | score_per_cost |",
        "|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['condition']} | {s['avg_score']:.3f} | "
            f"{s['intervention_rate']:.3f} | {s['cost_proxy']:.1f} | "
            f"{s['score_per_cost']:.4f} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def print_summary(summaries: List[dict]) -> None:
    h = (f"{'condition':<18} | {'avg_score':>9} | {'interv_rate':>11} | "
         f"{'cost':>6} | {'score/cost':>10}")
    print("\nRelay — fidelity vs cost by policy")
    print(h)
    print("-" * len(h))
    for s in summaries:
        print(f"{s['condition']:<18} | {s['avg_score']:>9.3f} | "
              f"{s['intervention_rate']:>11.3f} | {s['cost_proxy']:>6.1f} | "
              f"{s['score_per_cost']:>10.4f}")
    print()


# --------------------------------------------------------------------------- #
# 4a. Scorers (@weave.op, all local, no API)
# --------------------------------------------------------------------------- #
@op()
def structural_score(seed_doc: str, output: dict) -> dict:
    # Weave 0.52.x passes the model's return value as `output` (a scorer may not
    # declare both `output` and `model_output`).
    o = output or {}
    final_doc = o.get("final_doc") or ""
    if final_doc:
        return final_structural_score(seed_doc, final_doc)
    # lean trace has no document text -> pass through the score the run computed.
    return {
        "score": float(o.get("final_score", 0.0)),
        "parse_valid": 0.0, "id_f1": 0.0, "key_path_f1": 0.0,
        "scalar_value_fidelity": 0.0, "aggregate_score": 0.0,
    }


@op()
def intervention_rate(output: dict) -> dict:
    o = output or {}
    n = o.get("n_steps") or 1
    return {"intervention_rate": o.get("interventions", 0) / n}


@op()
def cost(output: dict) -> dict:
    o = output or {}
    return {"cost_proxy": float(o.get("cost_proxy", 0.0))}


# --------------------------------------------------------------------------- #
# 4b/4c. Condition model (pure lookup) + run the evaluation for all conditions
# --------------------------------------------------------------------------- #
# Big results table lives in a module global, not a model field (model fields
# must stay lightweight/serializable). Nested {condition: {task_id: TaskResult}}
# rather than tuple keys, so Weave's op code-dep capture doesn't choke.
_RESULTS: "Dict[str, Dict[str, TaskResult]]" = {}


def _stub() -> dict:
    return {"final_doc": "", "interventions": 0, "n_steps": 1,
            "cost_proxy": 0.0, "final_score": 0.0}


def _record(condition: str, task_id: str) -> dict:
    r = _RESULTS.get(condition, {}).get(task_id)
    if r is None:                      # partial run -> 0.0-scoring stub, never crash
        return _stub()
    return {
        "final_doc": r.final_doc,
        "interventions": r.interventions,
        "n_steps": r.n_steps,
        "cost_proxy": r.cost_proxy,
        "final_score": r.final_score,
    }


def run_evaluations(tasks, results: "Dict[Tuple[str, str], TaskResult]",
                    project: Optional[str] = None, name: str = "relay-round-trip"
                    ) -> Tuple[object, dict]:
    """Evaluate the four conditions on one Evaluation. Returns (evaluation, summaries).

    Returns ``(None, {})`` (no crash) if Weave isn't installed/credentialed —
    the markdown leaderboard still ships. ``results`` keys are (condition, task_id).
    """
    # weave-facing nested table (string keys, see _RESULTS note above)
    global _RESULTS
    _RESULTS = {}
    for (cond, tid), tr in results.items():
        _RESULTS.setdefault(cond, {})[tid] = tr

    if not weave_compat.maybe_init(project):
        print("[weave] not active (no install/creds) -> skipping Evaluation; "
              "markdown leaderboard still written.")
        return None, {}

    import asyncio
    weave = weave_compat.weave_module()

    # only evaluate task_ids we actually have results for (clean partial-run UI)
    present = {tid for (_c, tid) in results}
    dataset = [{"task_id": t.task_id, "seed_doc": t.seed_doc}
               for t in tasks if t.task_id in present]
    if not dataset:
        print("[weave] no overlapping task_ids between results and tasks "
              "(check --n / --rng-seed) -> skipping Evaluation.")
        return None, {}

    class ConditionModel(weave.Model):
        condition: str

        @weave.op()
        def predict(self, task_id: str, seed_doc: str) -> dict:
            return _record(self.condition, task_id)

    evaluation = weave.Evaluation(
        dataset=dataset,
        scorers=[structural_score, intervention_rate, cost],
        name=name,
    )

    conds_present = [c for c in CONDITIONS if any(cc == c for (cc, _t) in results)]
    summaries: dict = {}
    for cond in conds_present:
        summaries[cond] = asyncio.run(
            evaluation.evaluate(ConditionModel(condition=cond)))
    # Path-discovery aid (§4c): the printed dict shows the literal metric paths
    # used by the Leaderboard columns below.
    if summaries:
        first = next(iter(summaries))
        print(f"[weave] summary[{first}] = {summaries[first]}")
    return evaluation, summaries


# --------------------------------------------------------------------------- #
# 4d. Publish the Leaderboard
# --------------------------------------------------------------------------- #
def _eval_ref_uri(weave, evaluation) -> Optional[str]:
    """Obtain the published evaluation ref URI (import path is version-sensitive)."""
    try:
        ref = weave.publish(evaluation)
        uri = ref.uri() if hasattr(ref, "uri") else None
        if uri:
            return uri
    except Exception:
        pass
    for modpath, attr in (("weave.trace.refs", "get_ref"),
                          ("weave.trace.ref_util", "get_ref")):
        try:
            mod = __import__(modpath, fromlist=[attr])
            return getattr(mod, attr)(evaluation).uri()
        except Exception:
            continue
    return None


def publish_leaderboard(evaluation):
    """Publish a Leaderboard referencing ``evaluation``. Returns the ref or None."""
    weave = weave_compat.weave_module()
    if weave is None or evaluation is None:
        return None
    try:
        from weave.flow import leaderboard
    except Exception as e:
        print(f"[weave] leaderboard module unavailable ({e}); "
              "Evaluation comparison + leaderboard.md still available.")
        return None

    eval_ref = _eval_ref_uri(weave, evaluation)
    if not eval_ref:
        print("[weave] could not resolve evaluation ref; leaderboard skipped "
              "(Evaluation still in UI).")
        return None

    # column = evaluation_ref + scorer + metric_path. Paths confirmed via the
    # §4c summary print (structural_score.score.mean, etc.).
    columns = [
        leaderboard.LeaderboardColumn(
            evaluation_object_ref=eval_ref,
            scorer_name="structural_score",
            summary_metric_path="score.mean"),
        leaderboard.LeaderboardColumn(
            evaluation_object_ref=eval_ref,
            scorer_name="intervention_rate",
            summary_metric_path="intervention_rate.mean"),
        leaderboard.LeaderboardColumn(
            evaluation_object_ref=eval_ref,
            scorer_name="cost",
            summary_metric_path="cost_proxy.mean"),
    ]
    spec = leaderboard.Leaderboard(
        name="Relay — fidelity vs cost by policy",
        description=("Round-trip reconstruction score across four re-grounding "
                     "policies at equal budget."),
        columns=columns,
    )
    ref = weave.publish(spec)
    print(f"[weave] Leaderboard: {_leaderboard_url(ref)}")
    return ref


def _leaderboard_url(ref) -> str:
    """Build the Weave UI URL for a published leaderboard ref (best-effort)."""
    for attr in ("ui_url", "url"):
        val = getattr(ref, attr, None)
        if isinstance(val, str) and val.startswith("http"):
            return val
    try:
        return (f"https://wandb.ai/{ref.entity}/{ref.project}"
                f"/weave/leaderboards/{ref.name}")
    except Exception:
        return str(ref)


# --------------------------------------------------------------------------- #
# 4e. CLI entry (standalone — no API calls, runs against existing results)
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Publish a Weave Evaluation + Leaderboard from existing results "
                    "(pure lookup; no model calls).")
    ap.add_argument("--results", default="outputs/results.jsonl")
    ap.add_argument("--summary", default=None,
                    help="per-task summary jsonl (default: task_summary.jsonl beside --results)")
    ap.add_argument("--n", type=int, default=20, help="task count the run used")
    ap.add_argument("--rng-seed", type=int, default=42, help="task seed the run used")
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT"))
    ap.add_argument("--md-out", default="outputs/leaderboard.md")
    args = ap.parse_args()

    results = load_results(args.results, args.summary)
    if not results:
        print(f"[relay] no results found in {args.results}; nothing to publish.")
        return
    summaries = summarize(results)
    write_markdown_leaderboard(summaries, args.md_out)   # guaranteed artifact
    print_summary(summaries)
    print(f"[relay] wrote {args.md_out}")

    tasks = load_tasks(n=args.n, rng_seed=args.rng_seed)  # same seed/n as the run
    try:
        evaluation, _ = run_evaluations(tasks, results, project=args.project)
        if evaluation is not None:
            publish_leaderboard(evaluation)
    except Exception as e:
        print(f"[weave] publish failed (Evaluations/markdown intact): "
              f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
# LEGACY — old substrate/episode publish path, still imported by run.py.
# Kept for backward-compatibility; the round-trip path above is task A1.
# --------------------------------------------------------------------------- #
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


if __name__ == "__main__":
    main()
