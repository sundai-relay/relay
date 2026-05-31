"""MockSubstrate — the no-API substrate that makes hand-off degradation visible.

State is a dict of N integer "facts" (``f0..f{N-1}`` -> value). Each hop has a
*deterministic* corruption schedule (seeded by episode+hop), so the corruption
pattern is IDENTICAL across all four conditions — conditions differ ONLY in
*when* they re-ground. That is what makes adaptive-vs-random a fair test.

Mechanic (local repair):
  - forward apply_hop  : clobber this hop's victim facts with wrong values.
  - grounded apply_hop : repair THIS hop's victims from the source (local fix);
                         earlier un-repaired damage stays — so every well-timed
                         intervention has real, local value.
  - risk()             : noisy proxy of the before/after change (the signal).
  - score()            : fraction of facts that still match the reference.

Tuned so the expected ordering is clear:
    naive  <  random_at_budget  <  adaptive  <≈  always
which validates the whole pipeline before any real API call.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from ..core import Episode, Substrate
from ..weave_compat import op


def _rng(*parts) -> random.Random:
    """Deterministic RNG seeded by the given parts (stable across processes)."""
    return random.Random("|".join(repr(p) for p in parts))


class MockEpisode(Episode):
    def __init__(self, idx: int, n_facts: int, n_hops: int, corruption_p: float,
                 corruption_size: int, seed: int):
        self.id = f"mock-{idx:03d}"
        self.n_facts = n_facts
        self.n_hops = n_hops
        self.seed = seed

        # Reference doc: stable per episode. Values in 10..99.
        rr = _rng(seed, "ref", idx)
        self._reference: Dict[str, int] = {f"f{i}": rr.randint(10, 99)
                                           for i in range(n_facts)}

        # Deterministic corruption schedule: per hop, which facts get clobbered.
        # Identical regardless of condition.
        self._schedule: List[List[str]] = []
        keys = list(self._reference)
        for h in range(n_hops):
            hr = _rng(seed, "hop", idx, h)
            if hr.random() < corruption_p:
                k = min(corruption_size, n_facts)
                self._schedule.append(hr.sample(keys, k=k))
            else:
                self._schedule.append([])

    def initial_state(self) -> Dict[str, int]:
        return dict(self._reference)

    def hops(self) -> List[Dict[str, Any]]:
        return [{"index": h, "victims": self._schedule[h]} for h in range(self.n_hops)]

    def reference(self) -> Dict[str, int]:
        return dict(self._reference)

    @op()
    def score(self, final_state: Dict[str, int]) -> float:
        if not self._reference:
            return 1.0
        ok = sum(1 for k, v in self._reference.items() if final_state.get(k) == v)
        return ok / len(self._reference)


class MockSubstrate(Substrate):
    name = "mock"

    def __init__(self, n_facts: int = 12, n_hops: int = 8, corruption_p: float = 0.5,
                 corruption_size: int = 2, noise: float = 0.25, signal: float = 0.6,
                 seed: int = 0, **_):
        self.n_facts = n_facts
        self.n_hops = n_hops
        self.corruption_p = corruption_p
        self.corruption_size = corruption_size
        self.noise = noise
        self.signal = signal
        self.seed = seed

    def load_episodes(self, n: int) -> List[MockEpisode]:
        return [
            MockEpisode(i, self.n_facts, self.n_hops, self.corruption_p,
                        self.corruption_size, self.seed)
            for i in range(n)
        ]

    @op()
    def apply_hop(self, state: Dict[str, int], hop: Dict[str, Any],
                  grounding: Optional[Dict[str, int]] = None) -> Dict[str, int]:
        victims = hop.get("victims", [])
        if grounding is not None:
            # Re-ground: repair THIS hop's victims from the source slice. Leave
            # other (possibly already-damaged) facts as they are -> local fix.
            repaired = dict(state)
            for k in victims:
                if k in grounding:
                    repaired[k] = grounding[k]
            return repaired
        # Forward (lossy) relay step: clobber this hop's victims.
        new_state = dict(state)
        for k in victims:
            cr = _rng(self.seed, "corrupt", hop.get("index"), k)
            new_state[k] = cr.randint(100, 999)  # out of valid 10..99 -> always wrong
        return new_state

    @op()
    def risk(self, state_before: Dict[str, int], state_after: Dict[str, int],
             episode: Episode) -> float:
        changed = sum(1 for k in state_after
                      if state_before.get(k) != state_after.get(k))
        base = self.signal if changed > 0 else 0.0
        # Noisy proxy: deterministic noise per transition (reproducible runs).
        nr = _rng(self.seed, "risk", episode.id,
                  tuple(sorted(state_before.items())),
                  tuple(sorted(state_after.items())))
        noisy = base + nr.uniform(-self.noise, self.noise)
        return max(0.0, min(1.0, noisy))

    @op()
    def reground(self, episode: Episode) -> Dict[str, int]:
        return episode.reference()
