"""checksum.py — the runtime risk signal (the Conductor's warning light).

KEY-FREE: compares the current doc against the *seed* on quantities that every
legitimate edit preserves (record IDs, the stable keys id/name/category, the
total stock). It never looks at the gold/final score, so it cannot be circular.
A legitimate in-flight edit (dollars->cents, sort, nest) does NOT move these
invariants, so a high score means real corruption.

    risk = 0.35*parse_risk + 0.30*id_loss_rate
         + 0.20*required_key_loss_rate + 0.15*aggregate_drift
"""

from __future__ import annotations

from typing import Dict

from .jsonutil import (INVARIANT_KEYS, by_id, get_stock, parse_doc, record_ids,
                       total_stock)
from ..weave_compat import op


@op()
def runtime_risk(seed_doc, current_doc) -> Dict:
    seed = parse_doc(seed_doc)
    cur = parse_doc(current_doc)

    if cur is None:
        # Unparseable current doc: maximal parse risk, everything "lost".
        return {
            "risk": 1.0, "parse_risk": 1.0, "id_loss_rate": 1.0,
            "required_key_loss_rate": 1.0, "aggregate_drift": 1.0,
            "missing_ids": sorted(record_ids(seed)) if seed else [],
            "missing_keys": [], "numeric_drift": {},
        }
    if seed is None:
        return {"risk": 0.0, "parse_risk": 0.0, "id_loss_rate": 0.0,
                "required_key_loss_rate": 0.0, "aggregate_drift": 0.0,
                "missing_ids": [], "missing_keys": [], "numeric_drift": {}}

    parse_risk = 0.0

    seed_ids = set(record_ids(seed))
    cur_ids = set(record_ids(cur))
    missing_ids = sorted(seed_ids - cur_ids)
    id_loss_rate = len(missing_ids) / max(1, len(seed_ids))

    # required-key loss: over records that still exist, fraction of invariant
    # (id, key) pairs that are missing.
    cur_by_id = by_id(cur)
    total_pairs = 0
    lost_pairs = 0
    missing_keys = []
    for rid in seed_ids:
        if rid not in cur_by_id:
            continue  # counted by id_loss, not key_loss
        rec = cur_by_id[rid]
        for k in INVARIANT_KEYS:
            total_pairs += 1
            if k not in rec or rec.get(k) in (None, ""):
                lost_pairs += 1
                missing_keys.append({"id": rid, "key": k})
    required_key_loss_rate = lost_pairs / max(1, total_pairs)

    # aggregate drift on the numeric invariant (total stock).
    seed_stock = total_stock(seed)
    cur_stock = total_stock(cur)
    aggregate_drift = abs(seed_stock - cur_stock) / max(1.0, seed_stock)
    aggregate_drift = min(1.0, aggregate_drift)

    # per-id stock drift detail (for targeted repair).
    numeric_drift = {}
    seed_by_id = by_id(seed)
    for rid, rec in cur_by_id.items():
        if rid in seed_by_id:
            s, c = get_stock(seed_by_id[rid]), get_stock(rec)
            if s is not None and c is not None and abs(s - c) > 1e-9:
                numeric_drift[rid] = {"seed": s, "current": c}

    risk = (0.35 * parse_risk + 0.30 * id_loss_rate
            + 0.20 * required_key_loss_rate + 0.15 * aggregate_drift)
    return {
        "risk": round(max(0.0, min(1.0, risk)), 4),
        "parse_risk": parse_risk,
        "id_loss_rate": round(id_loss_rate, 4),
        "required_key_loss_rate": round(required_key_loss_rate, 4),
        "aggregate_drift": round(aggregate_drift, 4),
        "missing_ids": missing_ids,
        "missing_keys": missing_keys,
        "numeric_drift": numeric_drift,
    }
