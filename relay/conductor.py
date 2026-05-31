"""Conductor — the functional re-grounding rule for the four conditions.

The decision is deliberately thin: given a condition and the per-step risk,
decide whether to re-ground. The science lives in the *comparison* of the
conditions (esp. adaptive vs. random-at-budget), not in the rule itself.
"""

from __future__ import annotations

# Canonical condition names (also the leaderboard / jsonl labels).
NAIVE = "naive"
ALWAYS = "always"
ADAPTIVE = "adaptive"
RANDOM = "random_at_budget"


# Functional intervention rule used by the round-trip runner (relay.roundtrip).
def should_intervene(condition: str, risk: float, threshold: float,
                     random_rate, rng) -> bool:
    if condition in (NAIVE, "naive"):
        return False
    if condition in (ALWAYS, "always_reground", "always"):
        return True
    if condition in (ADAPTIVE, "adaptive"):
        return risk > threshold
    if condition in (RANDOM, "random_at_budget", "random"):
        return rng.random() < (random_rate or 0.0)
    raise ValueError(f"unknown condition: {condition!r}")
