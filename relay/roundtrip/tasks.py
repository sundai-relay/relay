"""Procedural seed-doc + edit-pair generation. Fixed seeds => reproducible.

A seed doc has 15-20 records with stable IDs, a category field, a numeric price,
a numeric stock count, a (nestable) warehouse location, and a summary with known
totals. Three reversible edit pairs are applied in round-robin over the round
trips. DELEGATE-52 is inspiration/citation, not a dependency.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

from .jsonutil import dump

DOMAINS = [
    "equipment_catalog",
    "product_catalog",
    "research_sample_inventory",
    "conference_schedule",
    "purchase_order_ledger",
]
CATEGORIES = ["alpha", "beta", "gamma", "delta"]
_ADJ = ["solar", "carbon", "quantum", "hydro", "thermal", "lunar", "atomic",
        "delta", "prime", "vector", "cobalt", "ember", "frost", "onyx", "zephyr"]
_NOUN = ["actuator", "module", "sensor", "ledger", "rotor", "panel", "valve",
         "beacon", "lattice", "probe", "relay", "matrix", "coil", "filter", "node"]


@dataclass
class EditPair:
    name: str
    forward_instruction: str
    backward_instruction: str


@dataclass
class RoundTripTask:
    task_id: str
    seed_doc: str          # JSON string
    edit_pairs: List[EditPair]


def make_edit_pairs() -> List[EditPair]:
    return [
        EditPair(
            name="dollars_to_cents",
            forward_instruction=(
                "Convert every price_dollars field to price_cents by multiplying "
                "by 100 and rounding to an integer. Rename price_dollars to "
                "price_cents. Preserve every item ID and all non-price fields. "
                "Return valid JSON only."
            ),
            backward_instruction=(
                "Convert every price_cents field back to price_dollars by dividing "
                "by 100. Rename price_cents back to price_dollars. Preserve every "
                "item ID and all non-price fields. Return valid JSON only."
            ),
        ),
        EditPair(
            name="sort_by_category",
            forward_instruction=(
                "Sort the records by category, then by name. Preserve every record "
                "exactly. Return valid JSON only."
            ),
            backward_instruction=(
                "Restore the records to ascending ID order. Preserve every record "
                "exactly. Return valid JSON only."
            ),
        ),
        EditPair(
            name="nest_inventory",
            forward_instruction=(
                "Move stock_count and warehouse_location into a nested inventory "
                "object for each item. Preserve all values exactly. Return valid "
                "JSON only."
            ),
            backward_instruction=(
                "Flatten inventory.stock_count back to stock_count and "
                "inventory.warehouse_location back to warehouse_location. Remove "
                "the inventory object. Preserve all values exactly. Return valid "
                "JSON only."
            ),
        ),
    ]


def generate_seed_doc(task_id: str, rng_seed: int) -> str:
    rng = random.Random(f"seed|{task_id}|{rng_seed}")
    domain = DOMAINS[rng_randrange_stable(rng, len(DOMAINS))]
    n_records = rng.randint(15, 20)

    records = []
    used_names = set()
    for i in range(n_records):
        name = f"{rng.choice(_ADJ)}-{rng.choice(_NOUN)}"
        while name in used_names:
            name = f"{rng.choice(_ADJ)}-{rng.choice(_NOUN)}-{rng.randint(2, 9)}"
        used_names.add(name)
        records.append({
            "id": f"R{i + 1:03d}",
            "name": name,
            "category": rng.choice(CATEGORIES),
            "price_dollars": round(rng.uniform(5, 500), 2),
            "stock_count": rng.randint(0, 200),
            "warehouse_location": f"{rng.choice('ABCDEF')}{rng.randint(1, 9)}",
        })

    doc = {
        "catalog_id": task_id,
        "domain": domain,
        "currency": "USD",
        "records": records,
        "summary": {
            "record_count": n_records,
            "total_price_dollars": round(sum(r["price_dollars"] for r in records), 2),
            "total_stock": sum(r["stock_count"] for r in records),
        },
    }
    return dump(doc)


def load_tasks(n: int, rng_seed: int = 42) -> List[RoundTripTask]:
    pairs = make_edit_pairs()
    tasks = []
    for i in range(n):
        task_id = f"{DOMAINS[i % len(DOMAINS)]}-{i:03d}"
        tasks.append(RoundTripTask(
            task_id=task_id,
            seed_doc=generate_seed_doc(task_id, rng_seed + i),
            edit_pairs=pairs,
        ))
    return tasks


def rng_randrange_stable(rng: random.Random, n: int) -> int:
    # random.Random seeded with a string is deterministic; randrange is fine.
    return rng.randrange(n)
