"""STEP 2 — the degradation gate (make-or-break).

Question: on passage-dependent items, does relaying the explanation through a
chain (no re-grounding) lose answer accuracy?

Levers applied (disclosed) because natural degradation was weak:
  - passage-dependent selection (source-correct AND blind-wrong items only)
  - force-compression on relay hops (RELAY_MAX_WORDS), relays don't see the question
  - weaker relay model (RELAY_SMALL_MODEL) + 3 relay hops

GREEN if relayed_acc < source_acc by a meaningful margin -> run probe.py.
RED otherwise -> reframe ("harness showing drift alone is insufficient").
"""
from __future__ import annotations

import argparse

from agents import RELAY_MAX_WORDS, answerer, explainer, relay
from data import load_items
from llm import BIG_MODEL, SMALL_MODEL
from selection import passage_dependent

N_RELAY_HOPS = 3


def answer_relayed(item, big, small):
    m = explainer(item.source, item.question, item.choices, model=big)
    for _ in range(N_RELAY_HOPS):
        m = relay(m, item.question, item.choices, model=small)
    return answerer(m, item.question, item.choices, model=big)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=15, help="target selected items")
    ap.add_argument("--pool", type=int, default=80, help="candidates to scan")
    ap.add_argument("--big", default=BIG_MODEL)
    ap.add_argument("--small", default=SMALL_MODEL)
    args = ap.parse_args()

    pool = load_items(n=args.pool)
    print(f"[sanity] scanning {len(pool)} candidates for passage-dependent items...")
    items = passage_dependent(pool, n=args.n, model=args.big)
    print(f"[sanity] selected {len(items)} items | big={args.big} small={args.small} "
          f"| relay_max_words={RELAY_MAX_WORDS} hops={N_RELAY_HOPS}\n")

    # selected set: source_acc == 1.0 and blind_acc == 0.0 by construction
    relay_ok = 0
    for it in items:
        r = answer_relayed(it, args.big, args.small) == it.gold
        relay_ok += r
        print(f"{it.item_id:>12}  relayed={'Y' if r else 'n'}")

    n = len(items)
    sa, ba, ra = 1.0, 0.0, relay_ok / n if n else 0.0
    print("\n--- accuracy (on passage-dependent items) ---")
    print(f"source : {sa:.2f}  (by selection)")
    print(f"blind  : {ba:.2f}  (by selection)")
    print(f"relayed: {ra:.2f}")

    relay_loses = ra < 0.85  # meaningful loss from a 100% answerable set
    print("\n--- GATE ---")
    print(f"relay loses on answerable items?  {relay_loses}  ({ra:.2f} < 0.85)")
    print(f"\n>>> {'GREEN — proceed to probe.py' if relay_loses else 'RED — reframe'} <<<")


if __name__ == "__main__":
    main()
