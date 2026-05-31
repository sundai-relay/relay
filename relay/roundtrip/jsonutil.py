"""Shared JSON helpers for the round-trip substrate.

The seed documents have a small, known schema, and the three reversible edits
move fields around (price_dollars<->price_cents, flat<->nested inventory) or
reorder records. These helpers extract the *invariants* that must survive ALL
legitimate edits — record IDs, the stable keys (id/name/category), and the
total stock — so the checksum can tell real corruption apart from a legitimate
in-flight edit.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

# Keys that every record keeps through every legitimate edit.
INVARIANT_KEYS = ("id", "name", "category")


def strip_fences(s: str) -> str:
    """Strip a leading ```json / ``` fence (+ optional language tag) and trailer."""
    s = s.strip()
    if s.startswith("```"):
        s = s[3:]
        nl = s.find("\n")
        if nl != -1 and "{" not in s[:nl]:
            s = s[nl + 1:]
        end = s.rfind("```")
        if end != -1:
            s = s[:end]
    return s.strip()


def parse_doc(state: Any) -> Optional[dict]:
    """Parse a doc that may be a dict, a JSON string, or fenced/pre-amble'd text."""
    if isinstance(state, dict):
        return state
    if not isinstance(state, str):
        return None
    s = strip_fences(state)
    try:
        return json.loads(s)
    except Exception:
        try:
            return json.loads(s[s.index("{"): s.rindex("}") + 1])
        except Exception:
            return None


def dump(doc: Any) -> str:
    return json.dumps(doc, separators=(",", ":"))


def records(doc: dict) -> List[dict]:
    recs = doc.get("records", [])
    return recs if isinstance(recs, list) else []


def record_ids(doc: dict) -> List[str]:
    out = []
    for r in records(doc):
        if isinstance(r, dict) and "id" in r:
            out.append(r["id"])
    return out


def by_id(doc: dict) -> Dict[str, dict]:
    return {r["id"]: r for r in records(doc) if isinstance(r, dict) and "id" in r}


# --- schema-robust scalar extractors (work flat OR nested, dollars OR cents) --
def get_stock(rec: dict) -> Optional[float]:
    if "stock_count" in rec:
        return _num(rec["stock_count"])
    inv = rec.get("inventory")
    if isinstance(inv, dict) and "stock_count" in inv:
        return _num(inv["stock_count"])
    return None


def get_warehouse(rec: dict):
    if "warehouse_location" in rec:
        return rec["warehouse_location"]
    inv = rec.get("inventory")
    if isinstance(inv, dict):
        return inv.get("warehouse_location")
    return None


def get_price_dollars(rec: dict) -> Optional[float]:
    if "price_dollars" in rec:
        return _num(rec["price_dollars"])
    if "price_cents" in rec:
        c = _num(rec["price_cents"])
        return None if c is None else round(c / 100.0, 2)
    return None


def total_stock(doc: dict) -> float:
    return float(sum(s for s in (get_stock(r) for r in records(doc)) if s is not None))


def total_price_dollars(doc: dict) -> float:
    return float(sum(p for p in (get_price_dollars(r) for r in records(doc))
                     if p is not None))


def _num(v) -> Optional[float]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)
    except Exception:
        return None


# --- id-keyed flatten: order-independent paths keyed by record id -------------
def id_keyed_flatten(doc: dict) -> Dict[str, Any]:
    """Flatten to {path: scalar}, but key record paths by their id (not row
    index), so reordering records does not change the path set."""
    out: Dict[str, Any] = {}
    for k, v in doc.items():
        if k == "records":
            continue
        _flatten_into(v, str(k), out)
    for r in records(doc):
        rid = r.get("id") if isinstance(r, dict) else None
        prefix = f"records[{rid}]" if rid is not None else f"records[?{id(r)}]"
        _flatten_into(r, prefix, out)
    return out


def _flatten_into(node: Any, prefix: str, out: Dict[str, Any]) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            _flatten_into(v, f"{prefix}.{k}", out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _flatten_into(v, f"{prefix}[{i}]", out)
    else:
        out[prefix] = node
