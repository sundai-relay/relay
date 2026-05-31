"""Conductor + the four re-grounding policies.

The Conductor is deliberately thin: given a policy, it decides per hop whether
to re-ground. The science lives in the *comparison* of policies (esp. adaptive
vs. random-at-budget), not in the Conductor itself.
"""

from __future__ import annotations

import random
from typing import Dict, Set

# Canonical condition names (also the leaderboard / jsonl labels).
NAIVE = "naive"
ALWAYS = "always"
ADAPTIVE = "adaptive"
RANDOM = "random_at_budget"


class Policy:
    name = "policy"

    def decide(self, episode, hop_index: int, risk_value: float, n_hops: int) -> bool:
        raise NotImplementedError


class NaivePolicy(Policy):
    """Never re-ground. Proves loss exists."""

    name = NAIVE

    def decide(self, episode, hop_index, risk_value, n_hops):
        return False


class AlwaysPolicy(Policy):
    """Re-ground every hop. The fidelity upper bound (and cost ceiling)."""

    name = ALWAYS

    def decide(self, episode, hop_index, risk_value, n_hops):
        return True


class AdaptivePolicy(Policy):
    """Re-ground when the key-free risk exceeds a threshold.

    The observed intervention rate is whatever falls out of the data — we read
    it back afterwards and hand it to random-at-budget.
    """

    name = ADAPTIVE

    def __init__(self, threshold: float = 0.4):
        self.threshold = threshold

    def decide(self, episode, hop_index, risk_value, n_hops):
        return risk_value > self.threshold


class RandomAtBudgetPolicy(Policy):
    """Re-ground at random hops, matched per-episode to adaptive's observed count.

    This is the non-negotiable control: same number of interventions as
    adaptive, but at random *moments*. If adaptive beats this, the signal has
    decision value beyond "re-grounding helps."
    """

    name = RANDOM

    def __init__(self, budget: Dict[str, int], seed: int = 0):
        # budget: episode_id -> number of interventions adaptive used.
        self.budget = budget
        self.seed = seed
        self._chosen: Dict[str, Set[int]] = {}

    def _ensure(self, episode, n_hops) -> Set[int]:
        if episode.id not in self._chosen:
            k = min(self.budget.get(episode.id, 0), n_hops)
            rng = random.Random(f"{self.seed}|rand|{episode.id}")
            self._chosen[episode.id] = set(rng.sample(range(n_hops), k)) if k > 0 else set()
        return self._chosen[episode.id]

    def decide(self, episode, hop_index, risk_value, n_hops):
        return hop_index in self._ensure(episode, n_hops)


class Conductor:
    """Given a policy, decide grounding per hop."""

    def __init__(self, policy: Policy):
        self.policy = policy

    def decide(self, episode, hop_index: int, risk_value: float, n_hops: int) -> bool:
        return self.policy.decide(episode, hop_index, risk_value, n_hops)
