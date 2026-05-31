"""agents.py — the LLM calls (W&B Inference), Weave-traced.

Two ops:
    apply_edit(current_doc, instruction)                 -> new doc
    repair_doc(seed_doc, current_doc, instruction, report) -> repaired doc

Both run against W&B Inference when WANDB_API_KEY is set; otherwise a
deterministic MOCK editor/repairer runs so the full four-condition pipeline is
GREEN with no key and no credits. The mock editor is intentionally lossy (it
occasionally drops a record / key or perturbs stock); the mock repairer is
competent and *targeted* (it restores only the invariants the checksum flags,
preserving the in-flight edit) — exactly the behavior we want to reward.

Call ``configure(...)`` once before a run (run_all_conditions does this).
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Optional

from ..wandb_client import WEAK_MODEL, WandbInferenceClient
from ..weave_compat import op
from .jsonutil import by_id, dump, get_stock, parse_doc

_CONFIG = {
    "use_mock": True,
    "model": WEAK_MODEL,
    "slip_p": 0.5,   # mock: probability a forward/backward edit loses something
}
_CLIENT: Optional[WandbInferenceClient] = None


def configure(use_mock: Optional[bool] = None, model: str = WEAK_MODEL,
              slip_p: float = 0.5) -> bool:
    """Set the agent mode. If use_mock is None, auto-detect from WANDB_API_KEY.
    Returns the resolved use_mock value."""
    if use_mock is None:
        use_mock = not bool(os.environ.get("WANDB_API_KEY"))
    _CONFIG["use_mock"] = use_mock
    _CONFIG["model"] = model
    _CONFIG["slip_p"] = slip_p
    return use_mock


def is_mock() -> bool:
    return _CONFIG["use_mock"]


def _client() -> WandbInferenceClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = WandbInferenceClient()
    return _CLIENT


# --------------------------------------------------------------------------- #
# Public traced ops
# --------------------------------------------------------------------------- #
@op()
def apply_edit(current_doc: str, instruction: str) -> str:
    if _CONFIG["use_mock"]:
        return _mock_apply_edit(current_doc, instruction)
    return _real_apply_edit(current_doc, instruction)


@op()
def repair_doc(seed_doc, current_doc, instruction, checksum_report) -> str:
    if _CONFIG["use_mock"]:
        return _mock_repair_doc(seed_doc, current_doc, instruction, checksum_report)
    return _real_repair_doc(seed_doc, current_doc, instruction, checksum_report)


# --------------------------------------------------------------------------- #
# Real W&B Inference path
# --------------------------------------------------------------------------- #
_EDIT_SYSTEM = (
    "You are a precise JSON editor. Apply EXACTLY the single instruction given. "
    "Output ONLY valid JSON, no prose, no code fences. Preserve every field the "
    "instruction does not mention, byte-for-byte."
)


def _real_apply_edit(current_doc: str, instruction: str) -> str:
    doc = parse_doc(current_doc)
    working = dump(doc) if doc is not None else str(current_doc)
    user = f"DOCUMENT:\n{working}\n\nINSTRUCTION: {instruction}\n\nResulting JSON:"
    raw = _client().chat(_CONFIG["model"], _EDIT_SYSTEM, user,
                         temperature=0.0, max_tokens=1500)
    parsed = parse_doc(raw)
    return dump(parsed) if parsed is not None else current_doc


def _real_repair_doc(seed_doc, current_doc, instruction, checksum_report) -> str:
    system = "You repair structured JSON documents. Return valid JSON only."
    user = (
        "You are repairing a structured JSON document after a delegated edit.\n\n"
        "You are given:\n"
        "1. The original seed document.\n"
        "2. The current document after an edit.\n"
        "3. The edit instruction that should still be preserved.\n"
        "4. A checksum report listing structural problems.\n\n"
        "Your task:\n"
        "- Restore only the missing or corrupted records, fields, IDs, or numeric "
        "values identified in the checksum report.\n"
        "- Preserve the legitimate transformation requested by the edit instruction.\n"
        "- Do not simply return the original seed document.\n"
        "- Do not invent new records.\n"
        "- Return valid JSON only.\n\n"
        f"Original seed:\n{seed_doc}\n\n"
        f"Current document:\n{current_doc}\n\n"
        f"Edit instruction:\n{instruction}\n\n"
        f"Checksum report:\n{dump(_repair_view(checksum_report))}\n"
    )
    raw = _client().chat(_CONFIG["model"], system, user,
                         temperature=0.0, max_tokens=1500)
    parsed = parse_doc(raw)
    return dump(parsed) if parsed is not None else current_doc


def _repair_view(report) -> dict:
    """The compact, gold-free slice of the checksum the repairer needs."""
    return {
        "missing_ids": report.get("missing_ids", []),
        "missing_keys": report.get("missing_keys", []),
        "numeric_drift": report.get("numeric_drift", {}),
        "value_drift": report.get("value_drift", {}),
    }


# --------------------------------------------------------------------------- #
# Deterministic MOCK path (no key, no credits)
# --------------------------------------------------------------------------- #
def _rng_for(*parts) -> random.Random:
    h = hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()
    return random.Random(int(h[:12], 16))


def _detect_transform(instruction: str) -> Optional[str]:
    s = instruction.lower()
    if "price_cents" in s and "multiplying by 100" in s:
        return "to_cents"
    if "price_dollars" in s and "dividing by 100" in s:
        return "to_dollars"
    if "sort the records by category" in s:
        return "sort_cat"
    if "ascending id order" in s:
        return "sort_id"
    if "nested inventory" in s or "move stock_count" in s:
        return "nest"
    if "flatten" in s and "inventory" in s:
        return "flatten"
    return None


def _apply_transform(doc: dict, kind: str) -> None:
    recs = doc.get("records", [])
    if kind == "to_cents":
        for r in recs:
            if "price_dollars" in r:
                r["price_cents"] = int(round(r.pop("price_dollars") * 100))
    elif kind == "to_dollars":
        for r in recs:
            if "price_cents" in r:
                r["price_dollars"] = round(r.pop("price_cents") / 100.0, 2)
    elif kind == "sort_cat":
        recs.sort(key=lambda r: (str(r.get("category", "")), str(r.get("name", ""))))
    elif kind == "sort_id":
        recs.sort(key=lambda r: str(r.get("id", "")))
    elif kind == "nest":
        for r in recs:
            if "stock_count" in r or "warehouse_location" in r:
                r["inventory"] = {
                    "stock_count": r.pop("stock_count", None),
                    "warehouse_location": r.pop("warehouse_location", None),
                }
    elif kind == "flatten":
        for r in recs:
            inv = r.pop("inventory", None)
            if isinstance(inv, dict):
                r["stock_count"] = inv.get("stock_count")
                r["warehouse_location"] = inv.get("warehouse_location")
    doc["records"] = recs


def _inject_slip(doc: dict, rng: random.Random) -> None:
    recs = doc.get("records", [])
    if not recs:
        return
    choice = rng.choice(["drop_record", "drop_key", "perturb_stock"])
    if choice == "drop_record" and len(recs) > 1:
        recs.pop(rng.randrange(len(recs)))
        return
    if choice == "drop_key":
        r = rng.choice(recs)
        k = rng.choice(["name", "category"])
        r.pop(k, None)
        return
    # perturb_stock
    r = rng.choice(recs)
    if isinstance(r.get("stock_count"), (int, float)):
        r["stock_count"] = r["stock_count"] + rng.randint(5, 50)
    elif isinstance(r.get("inventory"), dict) and \
            isinstance(r["inventory"].get("stock_count"), (int, float)):
        r["inventory"]["stock_count"] = r["inventory"]["stock_count"] + rng.randint(5, 50)


def _mock_apply_edit(current_doc: str, instruction: str) -> str:
    doc = parse_doc(current_doc)
    if doc is None:
        return current_doc
    kind = _detect_transform(instruction)
    if kind:
        _apply_transform(doc, kind)
    rng = _rng_for("edit", instruction, current_doc)
    if rng.random() < _CONFIG["slip_p"]:
        _inject_slip(doc, rng)
    return dump(doc)


def _coerce_to_schema(seed_rec: dict, template: Optional[dict]) -> dict:
    """Rebuild a dropped record in the CURRENT doc's schema (preserve the edit)."""
    r = {"id": seed_rec.get("id"), "name": seed_rec.get("name"),
         "category": seed_rec.get("category")}
    if template is not None and "price_cents" in template:
        r["price_cents"] = int(round(seed_rec.get("price_dollars", 0) * 100))
    else:
        r["price_dollars"] = seed_rec.get("price_dollars")
    stock = seed_rec.get("stock_count")
    wh = seed_rec.get("warehouse_location")
    if template is not None and "inventory" in template:
        r["inventory"] = {"stock_count": stock, "warehouse_location": wh}
    else:
        r["stock_count"] = stock
        r["warehouse_location"] = wh
    return r


def _set_stock(rec: dict, value) -> None:
    if value is None:
        return
    if "stock_count" in rec:
        rec["stock_count"] = value
    elif isinstance(rec.get("inventory"), dict):
        rec["inventory"]["stock_count"] = value
    else:
        rec["stock_count"] = value


def _mock_repair_doc(seed_doc, current_doc, instruction, checksum_report) -> str:
    seed = parse_doc(seed_doc)
    cur = parse_doc(current_doc)
    if cur is None or seed is None:
        return current_doc
    seed_by = by_id(seed)
    cur_recs = cur.get("records", [])
    template = cur_recs[0] if cur_recs else None

    # 1. restore missing records (in the current schema).
    for rid in checksum_report.get("missing_ids", []):
        if rid in seed_by:
            cur_recs.append(_coerce_to_schema(seed_by[rid], template))

    cur_by = {r["id"]: r for r in cur_recs if isinstance(r, dict) and "id" in r}

    # 2. restore missing invariant keys.
    for mk in checksum_report.get("missing_keys", []):
        rid, key = mk.get("id"), mk.get("key")
        if rid in cur_by and rid in seed_by and key in seed_by[rid]:
            cur_by[rid][key] = seed_by[rid][key]

    # 3. fix numeric (stock) drift back to the seed value.
    for rid in checksum_report.get("numeric_drift", {}):
        if rid in cur_by and rid in seed_by:
            _set_stock(cur_by[rid], get_stock(seed_by[rid]))

    cur["records"] = cur_recs
    return dump(cur)
