"""Core substrate-agnostic abstractions for Relay.

Everything the Conductor and the four conditions touch goes through these two
interfaces, so the underlying task can be swapped via one ``--substrate`` flag.

    Episode    : one relay run (initial state, hops, reference, score).
    Substrate  : how state degrades (apply_hop), how risk is read (risk),
                 and how we re-ground (reground).

Concrete substrates decorate apply_hop / risk / reground (and the episode's
score) with ``@op()`` so every step is traced when Weave is live.
"""

from __future__ import annotations

import abc
import json
from typing import Any, List, Optional


class Episode(abc.ABC):
    """One relay run: an initial state handed hop-to-hop down a chain."""

    id: str

    @abc.abstractmethod
    def initial_state(self) -> Any:
        """The state at hop 0 (what the Explainer produces / the seed doc)."""

    @abc.abstractmethod
    def hops(self) -> List[Any]:
        """Ordered list of hop specs (one relay rewrite per hop)."""

    @abc.abstractmethod
    def reference(self) -> Any:
        """The original / ground-truth slice used to re-ground and to score."""

    @abc.abstractmethod
    def score(self, final_state: Any) -> float:
        """Task fidelity of a final state, in [0, 1]. Gold is used HERE ONLY."""


class Substrate(abc.ABC):
    """A pluggable task: how state degrades, how risk reads, how we re-ground."""

    name: str = "substrate"
    # Adaptive threshold appropriate for THIS substrate's risk scale. The mock
    # risk is ~[0,0.85]; the round-trip checksum risk is much smaller. run.py
    # uses this when --threshold is not given.
    default_threshold: float = 0.4

    @abc.abstractmethod
    def load_episodes(self, n: int) -> List[Episode]:
        ...

    @abc.abstractmethod
    def apply_hop(self, state: Any, hop: Any, grounding: Optional[Any] = None) -> Any:
        """The (possibly lossy) relay step.

        With ``grounding is None`` this is the forward pass that may degrade
        state. With ``grounding`` provided, repair / re-ground the state.
        """

    @abc.abstractmethod
    def risk(self, state_before: Any, state_after: Any, episode: Episode) -> float:
        """Key-free degradation signal in [0, 1]. Never sees gold."""

    @abc.abstractmethod
    def reground(self, episode: Episode) -> Any:
        """The grounding slice (original / relevant source) to inject."""

    def token_proxy(self, state: Any, grounding: Optional[Any] = None) -> float:
        """Cheap, substrate-agnostic cost proxy: ~chars/4 of what was processed.

        Re-grounding injects the source slice, so it costs more — this is what
        drives the fidelity/cost frontier across the four conditions.
        """
        base = _approx_tokens(state)
        if grounding is not None:
            base += _approx_tokens(grounding)
        return float(base)


def _approx_tokens(x: Any) -> int:
    if isinstance(x, str):
        s = x
    else:
        try:
            s = json.dumps(x, default=str, sort_keys=True)
        except Exception:
            s = str(x)
    return max(1, len(s) // 4)
