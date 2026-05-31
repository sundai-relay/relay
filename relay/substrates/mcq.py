"""McqSubstrate — THIN STUB (fallback task). TEAMMATE TODO.

Mirrors the team brief's RACE/QuALITY relay:
  state = a handoff memo (str); hops = relay rewrites (compress/paraphrase);
  score = exact-match of a shadow-answer to the gold MCQ letter.

Only the deterministic exact-match scaffold + a drift-based risk proxy are
implemented so the interface is callable. The model calls (Explainer first
memo, relay rewrite, shadow-answerer) are stubbed — search "TODO(teammate)".
"""

from __future__ import annotations

from typing import Any, List, Optional

from ..core import Episode, Substrate
from ..wandb_client import STRONG_MODEL, WEAK_MODEL, WandbInferenceClient
from ..weave_compat import op

# TODO(teammate): replace with QuALITY / a hard RACE subset, normalized to this
# schema: {"id", "passage", "question", "options":[..], "gold":"A".."D"}.
SAMPLE = [
    {
        "id": "demo-neg-1",
        "passage": ("All shipments left on Monday EXCEPT the one bound for Oslo, "
                    "which was delayed to Thursday."),
        "question": "Which shipment did NOT leave on Monday?",
        "options": ["Berlin", "Oslo", "Paris", "Rome"],
        "gold": "B",
    },
]


class McqEpisode(Episode):
    def __init__(self, item: dict, n_hops: int = 3):
        self.id = item["id"]
        self.item = item
        self.n_hops = n_hops

    def initial_state(self) -> str:
        # TODO(teammate): the Explainer's first memo (model call). For now, seed
        # with the passage itself so the chain is runnable.
        return self.item["passage"]

    def hops(self) -> List[dict]:
        return [{"index": i,
                 "instruction": "Rewrite in <=40 words; keep only what's needed to answer."}
                for i in range(self.n_hops)]

    def reference(self) -> str:
        return self.item["passage"]

    @op()
    def score(self, final_state: Any) -> float:
        """Exact-match of the answer derived from the final memo vs gold.

        TODO(teammate): call the shadow-answerer model to pick A/B/C/D from
        ``final_state``. The placeholder below is a deterministic keyword check
        so the harness runs key-free."""
        gold = self.item["gold"]
        gold_text = self.item["options"][ord(gold) - ord("A")].lower()
        return 1.0 if gold_text in str(final_state).lower() else 0.0


class McqSubstrate(Substrate):
    name = "mcq"

    def __init__(self, relay_model: str = WEAK_MODEL, answer_model: str = STRONG_MODEL,
                 n_hops: int = 3, seed: int = 0, **_):
        self.relay_model = relay_model
        self.answer_model = answer_model
        self.n_hops = n_hops
        self.seed = seed
        self._client: Optional[WandbInferenceClient] = None

    def client(self) -> WandbInferenceClient:
        if self._client is None:
            self._client = WandbInferenceClient()
        return self._client

    def load_episodes(self, n: int) -> List[McqEpisode]:
        items = (SAMPLE * ((n // len(SAMPLE)) + 1))[:n]
        return [McqEpisode(it, self.n_hops) for it in items]

    @op()
    def apply_hop(self, state: Any, hop: dict, grounding: Optional[Any] = None) -> Any:
        """TODO(teammate): relay rewrite via the weak model. If grounding is not
        None, inject the question-relevant source chunk and repair the memo."""
        raise NotImplementedError("McqSubstrate.apply_hop is a stub — see TODO(teammate).")

    @op()
    def risk(self, state_before: Any, state_after: Any, episode: Episode) -> float:
        """TODO(teammate): the real signal is answer-instability (shadow-answer
        flip) + question-conditioned drift. Placeholder: a length-drop proxy so
        the interface is callable end-to-end."""
        lb, la = len(str(state_before)), len(str(state_after))
        if lb == 0:
            return 0.0
        return max(0.0, min(1.0, (lb - la) / lb))

    @op()
    def reground(self, episode: Episode) -> str:
        return episode.reference()
