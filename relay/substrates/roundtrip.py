"""RoundTripSubstrate — TEAMMATE TODO: flesh out the model-call edits.

State is a JSON document; hops are (forward, backward) *reversible* edit-
instruction pairs applied by a W&B Inference model. A faithful round trip
reconstructs the seed doc exactly; degradation = the model failing to apply or
undo an edit cleanly, drifting the doc.

STATUS: the interface is FULLY WIRED and the KEY-FREE / deterministic pieces
are IMPLEMENTED:
  - reference()           -> the seed doc
  - score()               -> JSON reconstruction similarity (id->value preserved
                             + numeric-total check)
  - risk()                -> checksum drift (parse the doc; fraction of original
                             ids/keys/counts still intact). Never sees gold.
  - episode / hop / round-trip construction

The ONE real TODO is ``apply_hop``: the prompt + JSON-response parsing against
W&B Inference. Search "TODO(teammate)" below. Until then ``apply_hop`` raises
NotImplementedError with a pointer, so ``--substrate roundtrip`` constructs and
loads episodes but stops at the first model call.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional

from ..core import Episode, Substrate
from ..wandb_client import WEAK_MODEL, WandbInferenceClient
from ..weave_compat import op


# --------------------------------------------------------------------------- #
# Key-free helpers: a flat checksum over a JSON doc (no gold involved).
# --------------------------------------------------------------------------- #
def _flatten(doc: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(doc, dict):
        for k, v in doc.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(doc, list):
        for i, v in enumerate(doc):
            out.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = doc
    return out


def _numeric_total(flat: Dict[str, Any]) -> float:
    tot = 0.0
    for v in flat.values():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            tot += float(v)
    return tot


def _parse_doc(state: Any) -> Optional[dict]:
    if isinstance(state, dict):
        return state
    if isinstance(state, str):
        try:
            return json.loads(state)
        except Exception:
            try:  # salvage the first {...} block
                s = state[state.index("{"): state.rindex("}") + 1]
                return json.loads(s)
            except Exception:
                return None
    return None


SEED_DOC = {
    "order_id": 4471,
    "customer": {"name": "Ada Lovelace", "tier": "gold"},
    "items": [
        {"sku": "A-1", "qty": 2, "price": 19},
        {"sku": "B-7", "qty": 1, "price": 80},
    ],
    "shipping": {"method": "express", "cost": 12},
    "total": 130,
    "notes": "do not substitute item B-7",
}

# Forward/backward reversible edit pairs. A faithful round trip == identity.
HOP_PAIRS = [
    ("Increase shipping.cost by 8.", "Decrease shipping.cost by 8."),
    ("Set customer.tier to 'platinum'.", "Set customer.tier back to 'gold'."),
    ("Add 1 to items[0].qty.", "Subtract 1 from items[0].qty."),
    ("Append ' (urgent)' to notes.", "Remove the trailing ' (urgent)' from notes."),
]


class RoundTripEpisode(Episode):
    def __init__(self, idx: int, seed_doc: dict, hop_pairs):
        self.id = f"roundtrip-{idx:03d}"
        self._seed = copy.deepcopy(seed_doc)
        self._pairs = hop_pairs

    def initial_state(self) -> dict:
        return copy.deepcopy(self._seed)

    def hops(self) -> List[dict]:
        # Forward pass then backward pass (reverse order): a clean round trip
        # returns to the seed.
        hops: List[dict] = []
        for i, (fwd, _bwd) in enumerate(self._pairs):
            hops.append({"index": i, "phase": "forward", "instruction": fwd})
        for j, i in enumerate(range(len(self._pairs) - 1, -1, -1)):
            hops.append({"index": len(self._pairs) + j, "phase": "backward",
                         "instruction": self._pairs[i][1]})
        return hops

    def reference(self) -> dict:
        return copy.deepcopy(self._seed)

    @op()
    def score(self, final_state: Any) -> float:
        """JSON reconstruction similarity: fraction of original id->value pairs
        preserved (0.8 weight) + a numeric-total check (0.2 weight)."""
        ref_flat = _flatten(self._seed)
        cur = _parse_doc(final_state)
        if cur is None:
            return 0.0
        cur_flat = _flatten(cur)
        if not ref_flat:
            return 1.0
        preserved = sum(1 for k, v in ref_flat.items() if cur_flat.get(k) == v)
        frac = preserved / len(ref_flat)
        tot_ok = 1.0 if abs(_numeric_total(ref_flat) - _numeric_total(cur_flat)) < 1e-6 else 0.0
        return 0.8 * frac + 0.2 * tot_ok


class RoundTripSubstrate(Substrate):
    name = "roundtrip"

    def __init__(self, model: str = WEAK_MODEL, seed: int = 0, **_):
        self.model = model
        self.seed = seed
        self._client: Optional[WandbInferenceClient] = None  # lazy (needs key)

    def client(self) -> WandbInferenceClient:
        if self._client is None:
            self._client = WandbInferenceClient()
        return self._client

    def load_episodes(self, n: int) -> List[RoundTripEpisode]:
        # TODO(teammate): vary the seed doc / hop pairs per episode for diversity.
        return [RoundTripEpisode(i, SEED_DOC, HOP_PAIRS) for i in range(n)]

    @op()
    def apply_hop(self, state: Any, hop: Dict[str, Any],
                  grounding: Optional[Any] = None) -> Any:
        """Apply one reversible edit via the W&B Inference model.

        TODO(teammate): THIS is the one real TODO. The prompt is drafted below;
        wire the call + parse:
          1. raw = self.client().chat(self.model, system, user, temperature=0)
          2. parsed = _parse_doc(raw)
          3. return parsed if parsed is not None else state   # fail safe
        Then delete the NotImplementedError.
        """
        doc = _parse_doc(state)
        instruction = hop["instruction"]

        system = (
            "You are a precise JSON editor. Apply EXACTLY the single instruction "
            "given. Output ONLY the resulting JSON object, no prose. Preserve "
            "every field the instruction does not mention, byte-for-byte."
        )
        ground_block = ""
        if grounding is not None:
            ground_block = (
                "\nAUTHORITATIVE SOURCE (re-ground to this if the working doc has "
                f"drifted, then apply the instruction):\n{json.dumps(grounding)}\n"
            )
        user = (
            f"WORKING DOC:\n{json.dumps(doc)}\n{ground_block}\n"
            f"INSTRUCTION: {instruction}\n\nResulting JSON:"
        )

        # --- TODO(teammate): enable the real call (creds + tested) -----------
        #   raw = self.client().chat(self.model, system, user, temperature=0)
        #   parsed = _parse_doc(raw)
        #   return parsed if parsed is not None else state
        raise NotImplementedError(
            "RoundTripSubstrate.apply_hop is a stub. The prompt is ready above — "
            "enable the self.client().chat(...) call and parse the JSON. "
            f"(system+user prompt length ~{len(system) + len(user)} chars)"
        )

    @op()
    def risk(self, state_before: Any, state_after: Any, episode: Episode) -> float:
        """Checksum drift (key-free): fraction of original ids/keys/counts no
        longer intact in the current doc. Parses the doc; never touches gold."""
        before = _parse_doc(state_before)
        after = _parse_doc(state_after)
        if after is None:
            return 1.0  # unparseable -> maximal risk
        if before is None:
            return 0.5
        fb, fa = _flatten(before), _flatten(after)
        keys_before = set(fb)
        if not keys_before:
            return 0.0
        missing = len(keys_before - set(fa)) / len(keys_before)
        shared = keys_before & set(fa)
        changed = sum(1 for k in shared if fb[k] != fa[k]) / len(keys_before)
        count_drift = abs(len(fb) - len(fa)) / max(1, len(fb))
        return max(0.0, min(1.0, 0.5 * missing + 0.4 * changed + 0.1 * count_drift))

    @op()
    def reground(self, episode: Episode) -> dict:
        """Re-ground = the seed doc (the authoritative source slice)."""
        return episode.reference()
