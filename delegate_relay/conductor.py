"""The Conductor: runs one chain under a given intervention policy and logs every boundary.

Direct port of relay-main/spike/probe.py's run_chain, with:
  - memo hops          -> round-trips (forward edit + inverse edit)
  - shadow-answerer    -> key-free boundary_signals (invariant_deviation primary)
  - chunk re-injection -> diff_blocks(boundary, D0) re-injection + repair pass
  - MCQ scoring        -> ledger.score (done by the caller after the chain finishes)

The Conductor holds the seed D0 (legitimate) and measures/intervenes ONLY at round-trip
boundaries, where the document should equal D0. Mid-transform state is never scored.

policy modes:
  naive    — never re-ground
  always   — re-ground at every boundary that actually lost data
  adaptive — re-ground when risk > threshold
  forced   — re-ground at boundary indices in forced_boundaries (random-at-budget)
"""
from __future__ import annotations

from agents import editor, reground
from data import Chain
from domain import diff_blocks
from signals import boundary_signals


def run_chain(chain: Chain, *, mode: str, threshold: float = 0.0,
              forced_boundaries: set[int] | None = None,
              log: list | None = None, condition: str = "") -> tuple[str, int, int, list[float]]:
    """Run one chain end-to-end. Returns (final_doc, n_interventions, total_tokens, risks)."""
    forced_boundaries = forced_boundaries or set()
    log = log if log is not None else []
    seed = chain.seed_text

    doc = seed
    tokens = 0
    n_int = 0
    risks: list[float] = []

    for bi, rt in enumerate(chain.steps):
        # --- one round-trip: forward edit, then inverse edit (corruption happens here) ---
        fwd = editor(doc, rt.forward_prompt)
        mid = fwd.text
        inv = editor(mid, rt.inverse_prompt)
        boundary = inv.text
        tokens += fwd.tokens + inv.tokens

        # --- BOUNDARY: doc should == D0. Score key-free risk. ---
        sig = boundary_signals(boundary, seed)
        risks.append(sig["risk"])

        if mode == "always":
            intervene = True
        elif mode == "adaptive":
            intervene = sig["risk"] > threshold
        elif mode == "forced":
            intervene = bi in forced_boundaries
        else:  # naive
            intervene = False

        repaired = 0
        sig_after = sig
        if intervene:
            # Count the DECISION (the budget unit), so adaptive vs random match exactly on
            # intervention count. Only pay tokens / repair when data was actually lost.
            n_int += 1
            missing = diff_blocks(boundary, seed)
            if missing:  # something was lost/altered -> re-inject + repair
                rg = reground(boundary, missing)
                boundary = rg.text
                tokens += rg.tokens
                repaired = 1
                sig_after = boundary_signals(boundary, seed)

        log.append(dict(
            sample_id=chain.sample_id, condition=condition, boundary=bi,
            edit=rt.label, ops=list(rt.ops),
            invariant_deviation=sig["invariant_deviation"],
            parse_health=sig["parse_health"], embedding_drift=sig["embedding_drift"],
            risk=sig["risk"], intervened=repaired,
            inv_dev_after=sig_after["invariant_deviation"],
            tokens_cum=tokens, n_missing=len(diff_blocks(boundary, seed)),
        ))
        doc = boundary  # feed the (possibly repaired) boundary into the next round-trip

    return doc, n_int, tokens, risks
