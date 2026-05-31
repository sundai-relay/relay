"""RoundTripSubstrate — a thin adapter exposing the relay.roundtrip build
through the substrate-agnostic Episode/Substrate interface.

The real implementation lives in ``relay/roundtrip/`` (tasks, agents, checksum,
scorer, runner) per the locked spec. This adapter lets the generic harness
(``python run.py --substrate roundtrip``) drive the same code. The dedicated,
fully-featured entry point with the gate, frontier, and demo case is
``run_all_conditions.py``.

State = a JSON document (string). Hops = the forward/backward edit instructions
of each round trip. apply_hop without grounding = apply_edit (lossy); with
grounding = targeted repair_doc against the seed. risk = checksum.runtime_risk;
score = scorer.final_structural_score; reground = the seed doc.

No key -> a deterministic mock editor/repairer runs (GREEN, no credits).
"""

from __future__ import annotations

from typing import Any, List, Optional

from ..core import Episode, Substrate
from ..roundtrip import agents
from ..roundtrip.checksum import runtime_risk
from ..roundtrip.scorer import final_structural_score
from ..roundtrip.tasks import RoundTripTask, load_tasks
from ..weave_compat import op


class RoundTripEpisode(Episode):
    def __init__(self, task: RoundTripTask, num_round_trips: int = 4):
        self.id = task.task_id
        self.task = task
        self.num_round_trips = num_round_trips

    def initial_state(self) -> str:
        return self.task.seed_doc

    def hops(self) -> List[dict]:
        pairs = self.task.edit_pairs
        hops = []
        for rt in range(self.num_round_trips):
            pair = pairs[rt % len(pairs)]
            hops.append({"round_trip_index": rt, "edit_name": pair.name,
                         "step_type": "forward", "instruction": pair.forward_instruction})
            hops.append({"round_trip_index": rt, "edit_name": pair.name,
                         "step_type": "backward", "instruction": pair.backward_instruction})
        return hops

    def reference(self) -> str:
        return self.task.seed_doc

    @op()
    def score(self, final_state: Any) -> float:
        return final_structural_score(self.task.seed_doc, final_state)["score"]


class RoundTripSubstrate(Substrate):
    name = "roundtrip"
    default_threshold = 0.008  # checksum risk is small-magnitude (see checksum.py)

    def __init__(self, num_round_trips: int = 4, seed: int = 42, slip_p: float = 0.6,
                 **_):
        self.num_round_trips = num_round_trips
        self.seed = seed
        # auto-detect mock vs live from WANDB_API_KEY.
        agents.configure(use_mock=None, slip_p=slip_p)

    def load_episodes(self, n: int) -> List[RoundTripEpisode]:
        return [RoundTripEpisode(t, self.num_round_trips)
                for t in load_tasks(n, self.seed)]

    @op()
    def apply_hop(self, state: Any, hop: dict, grounding: Optional[Any] = None) -> Any:
        if grounding is None:
            return agents.apply_edit(state, hop["instruction"])
        # targeted repair against the seed (grounding), preserving the edit.
        report = runtime_risk(grounding, state)
        return agents.repair_doc(grounding, state, hop["instruction"], report)

    @op()
    def risk(self, state_before: Any, state_after: Any, episode: Episode) -> float:
        # checksum compares the current doc to the seed (not the prior hop).
        return runtime_risk(episode.reference(), state_after)["risk"]

    @op()
    def reground(self, episode: Episode) -> Any:
        return episode.reference()
