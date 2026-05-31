"""scorer.py — final parser-based reconstruction score.

Used ONLY after a run; it must never drive the runtime policy. After a complete
set of round trips the final doc should equal the seed, so we compare directly.
Record paths are keyed by id (not row position), so reordering is not penalized.

    score = 0.10*parse_validity + 0.25*id_f1 + 0.20*key_path_f1
          + 0.30*scalar_value_fidelity + 0.15*aggregate_score
    (score = 0.0 if the final doc does not parse.)
"""

from __future__ import annotations

from typing import Dict

from .jsonutil import (id_keyed_flatten, parse_doc, record_ids, records,
                       total_price_dollars, total_stock)
from ..weave_compat import op


def _f1(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    inter = len(pred & gold)
    if inter == 0:
        return 0.0
    p = inter / len(pred)
    r = inter / len(gold)
    return 2 * p * r / (p + r)


def _num_eq(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) <= 1e-6 + 1e-6 * abs(float(b))
    except Exception:
        return a == b


@op()
def final_structural_score(seed_doc, final_doc) -> Dict:
    seed = parse_doc(seed_doc)
    final = parse_doc(final_doc)
    if final is None or seed is None:
        return {"score": 0.0, "parse_valid": 0.0, "id_f1": 0.0,
                "key_path_f1": 0.0, "scalar_value_fidelity": 0.0,
                "aggregate_score": 0.0}

    parse_valid = 1.0

    id_f1 = _f1(set(record_ids(final)), set(record_ids(seed)))

    seed_flat = id_keyed_flatten(seed)
    final_flat = id_keyed_flatten(final)
    key_path_f1 = _f1(set(final_flat), set(seed_flat))

    # scalar value fidelity over gold paths (missing => not faithful).
    if seed_flat:
        ok = sum(1 for k, v in seed_flat.items()
                 if k in final_flat and _num_eq(final_flat[k], v))
        scalar_value_fidelity = ok / len(seed_flat)
    else:
        scalar_value_fidelity = 1.0

    # aggregate: do the known invariants line up?
    checks = [
        len(records(final)) == len(records(seed)),
        _num_eq(total_stock(final), total_stock(seed)),
        _num_eq(total_price_dollars(final), total_price_dollars(seed)),
    ]
    aggregate_score = sum(1.0 for c in checks if c) / len(checks)

    score = (0.10 * parse_valid + 0.25 * id_f1 + 0.20 * key_path_f1
             + 0.30 * scalar_value_fidelity + 0.15 * aggregate_score)
    return {
        "score": round(score, 4),
        "parse_valid": parse_valid,
        "id_f1": round(id_f1, 4),
        "key_path_f1": round(key_path_f1, 4),
        "scalar_value_fidelity": round(scalar_value_fidelity, 4),
        "aggregate_score": round(aggregate_score, 4),
    }
