"""The per-hop run loop and the four conditions.

``run_episode`` runs one episode under one policy (the actual relay chain).
``run_condition`` (a ``@op``) runs a whole condition over the episodes.
``run_conditions`` orchestrates the four — and crucially derives
random-at-budget's per-episode budget from adaptive's *observed* interventions.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .conductor import (ADAPTIVE, ALWAYS, NAIVE, RANDOM, AdaptivePolicy,
                        AlwaysPolicy, Conductor, NaivePolicy, Policy,
                        RandomAtBudgetPolicy)
from .weave_compat import op


def run_episode(substrate, episode, policy: Policy) -> Tuple[object, List[dict]]:
    """Run one episode's relay chain under ``policy``.

    Per hop: forward (possibly lossy) pass -> read risk -> the Conductor
    decides -> if intervening, redo the hop with the grounding slice (repair).
    Returns ``(final_state, per_hop_rows)``.
    """
    conductor = Conductor(policy)
    state = episode.initial_state()
    hops = episode.hops()
    n_hops = len(hops)
    rows: List[dict] = []

    for i, hop in enumerate(hops):
        before = state
        forward = substrate.apply_hop(before, hop, grounding=None)
        r = substrate.risk(before, forward, episode)
        intervened = conductor.decide(episode, i, r, n_hops)

        grounding_used = None
        after = forward
        if intervened:
            grounding = substrate.reground(episode)
            # Repair the *degraded* state (targeted re-grounding), not a redo
            # from `before`. For the mock substrate this is identical; for the
            # round-trip substrate it is the difference between repair and reset.
            after = substrate.apply_hop(forward, hop, grounding=grounding)
            grounding_used = grounding

        state = after
        token = substrate.token_proxy(after, grounding=grounding_used)
        rows.append({
            "episode_id": episode.id,
            "condition": policy.name,
            "hop": i,
            "risk": round(float(r), 4),
            "intervened": bool(intervened),
            "score": round(float(episode.score(state)), 4),  # per-hop fidelity
            "token_proxy": round(float(token), 2),
        })

    return state, rows


@op()
def run_condition(substrate, episodes, condition_name: str, policy: Policy) -> List[dict]:
    """Run one condition across all episodes; return all per-hop rows."""
    rows: List[dict] = []
    for ep in episodes:
        _final, ep_rows = run_episode(substrate, ep, policy)
        rows.extend(ep_rows)
    return rows


def per_episode_intervention_counts(rows: List[dict]) -> Dict[str, int]:
    """episode_id -> number of hops where the policy intervened."""
    counts: Dict[str, int] = {}
    for row in rows:
        counts.setdefault(row["episode_id"], 0)
        if row["intervened"]:
            counts[row["episode_id"]] += 1
    return counts


def run_conditions(substrate, episodes, which: str = "all",
                   threshold: float = 0.4, seed: int = 0) -> Dict[str, List[dict]]:
    """Run the requested condition(s).

    ``random`` and ``all`` require adaptive first, to learn the budget that
    random-at-budget must match.
    """
    results: Dict[str, List[dict]] = {}

    if which in (NAIVE, "all"):
        results[NAIVE] = run_condition(substrate, episodes, NAIVE, NaivePolicy())
    if which in (ALWAYS, "all"):
        results[ALWAYS] = run_condition(substrate, episodes, ALWAYS, AlwaysPolicy())

    # adaptive is needed for itself, for "all", and to budget "random".
    adaptive_rows = None
    if which in (ADAPTIVE, "all", "random"):
        adaptive_rows = run_condition(substrate, episodes, ADAPTIVE, AdaptivePolicy(threshold))
        if which in (ADAPTIVE, "all"):
            results[ADAPTIVE] = adaptive_rows

    if which in ("random", "all"):
        budget = per_episode_intervention_counts(adaptive_rows or [])
        results[RANDOM] = run_condition(
            substrate, episodes, RANDOM, RandomAtBudgetPolicy(budget, seed=seed))

    return results
