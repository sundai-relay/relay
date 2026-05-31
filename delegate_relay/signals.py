"""Key-free handoff-degradation signals for the DELEGATE52 accounting relay.

All signals are measured at ROUND-TRIP BOUNDARIES (after a forward+inverse pair), where
the document should equal the seed D0. None of them touch the gold SCORE — the Conductor
holds D0 (legitimate, like relay-main's Conductor holds the passage), and the score()
function is the only thing that grades fidelity.

Three signals (mirrors relay-main's drift vs answer-instability split):
  1. invariant_deviation  (PRIMARY, live) — continuous structural deviation from D0
     (transaction counts, payee/account/amount sets). Tunable to an intervention budget.
     Built in ledger.py. The document analog of relay's answer-instability.
  2. parse_health         (live backstop) — did the doc stop parsing as a ledger?
  3. embedding_drift      (SECONDARY, expected near-inert) — 1 - cos(doc, D0) via MiniLM.
     Logged for comparison; weight ~0, reproducing relay's "drift is near-inert" finding.

risk = w_inv*invariant_deviation + w_health*(1 - parse_health) + w_drift*embedding_drift
"""
from __future__ import annotations

import functools
import os

import numpy as np

from domain import invariant_deviation, parse_doc, parse_health  # domain-dispatched

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

W_INV = float(os.environ.get("RELAY_W_INV", "1.0"))
W_HEALTH = float(os.environ.get("RELAY_W_HEALTH", "1.0"))
W_DRIFT = float(os.environ.get("RELAY_W_DRIFT", "0.0"))  # inert control by default


@functools.lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(_MODEL_NAME)


@functools.lru_cache(maxsize=512)
def _embed(text: str) -> tuple:
    v = _model().encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
    return tuple(float(x) for x in v)


def embedding_drift(doc: str, seed: str) -> float:
    """1 - cosine(doc, seed). Coarse semantic displacement (near-inert at this scale)."""
    a = np.asarray(_embed(doc), dtype=np.float32)
    b = np.asarray(_embed(seed), dtype=np.float32)
    return float(1.0 - np.dot(a, b))


def boundary_signals(doc_text: str, seed_text: str) -> dict:
    """Compute all key-free signals for a doc at a round-trip boundary vs the seed D0."""
    cur = parse_doc(doc_text)
    ref = parse_doc(seed_text)
    inv = invariant_deviation(cur, ref)
    health = parse_health(doc_text)
    drift = embedding_drift(doc_text, seed_text) if W_DRIFT else 0.0
    return {
        "invariant_deviation": round(inv, 4),
        "parse_health": round(health, 4),
        "embedding_drift": round(drift, 4),
        "risk": round(risk(inv, health, drift), 4),
    }


def risk(invariant_dev: float, health: float, drift: float = 0.0) -> float:
    """Transparent, tunable combination. invariant_deviation dominates (continuous);
    parse_health adds a backstop for documents that stopped parsing; drift is the
    logged-but-near-inert control (W_DRIFT defaults to 0)."""
    return float(W_INV * invariant_dev + W_HEALTH * (1.0 - health) + W_DRIFT * drift)


def threshold_from_quantile(risks: list[float], target_rate: float) -> float:
    """Set the adaptive threshold so ~target_rate of boundaries fire (port of probe.py)."""
    if not risks:
        return 0.0
    return float(np.quantile(np.asarray(risks, dtype=np.float64), 1.0 - target_rate))


if __name__ == "__main__":
    import re
    import json

    rows = [json.loads(l) for l in open("../delegate/delegate52.jsonl")]
    a1 = next(r for r in rows if r["sample_id"] == "accounting1")
    seed = a1["files"]["basic_state/hack_club.ledger"]
    blocks = [b for b in re.split(r"\n\s*\n", seed) if b.strip()]
    corrupt = "\n\n".join(blocks[:-3])  # drop 3 txns

    print("faithful boundary (doc == seed):", boundary_signals(seed, seed))
    print("corrupted boundary (drop 3 txns):", boundary_signals(corrupt, seed))
    print("threshold @30%:", round(threshold_from_quantile([0.0, 0.02, 0.05, 0.1, 0.2], 0.30), 4))
