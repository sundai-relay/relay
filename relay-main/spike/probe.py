"""STEP 3 — the four-condition feasibility probe.

ONE question: does adaptive re-grounding beat random re-grounding at the same
intervention budget?

Same items, four conditions:
  naive            : never re-ground
  always_reground  : re-ground at every relay hop (upper bound)
  adaptive         : re-ground when risk > threshold (threshold tuned to ~30%)
  random_at_budget : re-ground at randomly chosen hops, matched to adaptive's count

Pipeline: Explainer -> Relay1 -> Relay2 -> Answerer.
Interventions are decided at the two relay hops (memo1, memo2); the explainer
memo is already source-grounded.

Signals per hop (gold NOT used):
  drift, drift_delta (question-conditioned, MiniLM)
  answer_instability (shadow answerer choice changed vs previous hop)
  risk = drift_delta + answer_instability

Outputs: relay_log.jsonl (per hop), demo_cases.jsonl (<=3 adaptive-fixes-naive),
and a printed 4-row table.
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np

from agents import answerer, answerer_conf, explainer, reground, relay
from data import Item, load_items
from llm import BIG_MODEL, token_proxy
from selection import passage_dependent
from signals import Retriever, risk as risk_fn

RELAY_HOPS = (1, 2, 3)  # memo indices where intervention may occur


def run_chain(item: Item, retr: Retriever, chunks, centroid, *, mode: str,
              threshold: float, forced_hops: set[int], log: list, condition: str):
    """Run one item through the chain under a given intervention policy.

    mode: 'naive' | 'always' | 'adaptive' | 'forced'
      - 'forced' intervenes exactly at hops in forced_hops (random_at_budget).
    Returns (final_answer_idx, n_interventions, tok_total, hop_records).
    """
    tok = 0
    # hop 0: explainer (sees source) — never an intervention target
    m = explainer(item.source, item.question, item.choices)
    tok += token_proxy(item.source, item.question, *item.choices, m)
    prev_drift = retr.drift(m, centroid)
    prev_shadow, prev_margin = answerer_conf(m, item.question, item.choices)
    tok += token_proxy(m, item.question, *item.choices)

    n_int = 0
    hop_records = []
    log.append(dict(item_id=item.item_id, condition=condition, hop=0, memo=m,
                    drift=prev_drift, drift_delta=0.0, shadow_answer="ABCD"[prev_shadow],
                    margin=prev_margin, margin_drop=0.0, answer_changed=0, risk=0.0,
                    intervened=0, final_answer=None, gold=item.gold_letter,
                    correct=None, token_proxy=tok))

    memo = m
    for hop in RELAY_HOPS:
        small = relay(memo, item.question, item.choices)
        tok += token_proxy(memo, item.question, *item.choices, small)
        d = retr.drift(small, centroid)
        dd = d - prev_drift
        shadow, margin = answerer_conf(small, item.question, item.choices)
        tok += token_proxy(small, item.question, *item.choices)
        changed = int(shadow != prev_shadow)
        margin_drop = prev_margin - margin            # continuous, can be negative
        r = risk_fn(margin_drop, changed)

        intervene = False
        if mode == "always":
            intervene = True
        elif mode == "adaptive":
            intervene = r > threshold
        elif mode == "forced":
            intervene = hop in forced_hops
        # naive -> never

        if intervene:
            small = reground(small, item.question, item.choices, chunks)
            tok += token_proxy(small, item.question, *item.choices, *chunks)
            n_int += 1
            # refresh signals on the repaired memo
            d = retr.drift(small, centroid)
            shadow, margin = answerer_conf(small, item.question, item.choices)
            tok += token_proxy(small, item.question, *item.choices)

        log.append(dict(item_id=item.item_id, condition=condition, hop=hop, memo=small,
                        drift=d, drift_delta=dd, shadow_answer="ABCD"[shadow],
                        margin=margin, margin_drop=margin_drop, answer_changed=changed,
                        risk=r, intervened=int(intervene), final_answer=None,
                        gold=item.gold_letter, correct=None, token_proxy=tok))
        hop_records.append(dict(hop=hop, risk=r))
        prev_drift, prev_shadow, prev_margin, memo = d, shadow, margin, small

    final = answerer(memo, item.question, item.choices)
    tok += token_proxy(memo, item.question, *item.choices)
    log[-1]["final_answer"] = "ABCD"[final]
    log[-1]["correct"] = int(final == item.gold)
    return final, n_int, tok, hop_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=15, help="target selected items")
    ap.add_argument("--pool", type=int, default=80)
    ap.add_argument("--target-rate", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    pool = load_items(n=args.pool)
    print(f"[probe] scanning {len(pool)} candidates for passage-dependent items...")
    items = passage_dependent(pool, n=args.n)
    retrievers = {}
    chunkcache = {}
    for it in items:
        r = Retriever(it.source)
        chunks, centroid = r.top_k(it.question, k=3)
        retrievers[it.item_id] = (r, chunks, centroid)
    print(f"[probe] {len(items)} items\n")

    log: list = []

    # --- Pass 1: collect risk values (naive run) to set the adaptive threshold ---
    naive_correct = 0
    all_risks = []
    for it in items:
        r, chunks, centroid = retrievers[it.item_id]
        final, _, tok, hops = run_chain(it, r, chunks, centroid, mode="naive",
                                        threshold=0.0, forced_hops=set(), log=log,
                                        condition="naive")
        naive_correct += int(final == it.gold)
        all_risks += [h["risk"] for h in hops]

    thr = float(np.quantile(all_risks, 1 - args.target_rate)) if all_risks else 0.0
    print(f"[probe] adaptive threshold @ {1-args.target_rate:.0%}ile risk = {thr:.4f}")

    # --- always_reground ---
    always_correct = always_int = always_tok = 0
    for it in items:
        r, chunks, centroid = retrievers[it.item_id]
        final, ni, tok, _ = run_chain(it, r, chunks, centroid, mode="always",
                                      threshold=thr, forced_hops=set(), log=log,
                                      condition="always_reground")
        always_correct += int(final == it.gold); always_int += ni; always_tok += tok

    # --- adaptive (records which hops fired, for the budget-matched control) ---
    adaptive_correct = adaptive_int = adaptive_tok = 0
    adaptive_answer = {}
    fired = {}  # item_id -> set of hops that intervened
    for it in items:
        r, chunks, centroid = retrievers[it.item_id]
        # replay to learn fired hops at threshold, then run
        start = len(log)
        final, ni, tok, _ = run_chain(it, r, chunks, centroid, mode="adaptive",
                                      threshold=thr, forced_hops=set(), log=log,
                                      condition="adaptive")
        fired[it.item_id] = {rec["hop"] for rec in log[start:] if rec["intervened"]}
        adaptive_correct += int(final == it.gold); adaptive_int += ni; adaptive_tok += tok
        adaptive_answer[it.item_id] = final

    # --- random_at_budget: same total interventions, randomly placed ---
    rng = random.Random(args.seed)
    candidates = [(it.item_id, h) for it in items for h in RELAY_HOPS]
    rng.shuffle(candidates)
    chosen = set(candidates[:adaptive_int])
    forced_by_item = {}
    for iid, h in chosen:
        forced_by_item.setdefault(iid, set()).add(h)

    random_correct = random_int = random_tok = 0
    for it in items:
        r, chunks, centroid = retrievers[it.item_id]
        final, ni, tok, _ = run_chain(it, r, chunks, centroid, mode="forced",
                                      threshold=thr,
                                      forced_hops=forced_by_item.get(it.item_id, set()),
                                      log=log, condition="random_at_budget")
        random_correct += int(final == it.gold); random_int += ni; random_tok += tok

    # naive token total (recompute from log)
    naive_tok = sum(e["token_proxy"] for e in log
                    if e["condition"] == "naive" and e["final_answer"])

    n = len(items)
    rows = [
        ("naive",            naive_correct,    0,            naive_tok),
        ("always_reground",  always_correct,   always_int,   always_tok),
        ("adaptive",         adaptive_correct, adaptive_int, adaptive_tok),
        ("random_at_budget", random_correct,   random_int,   random_tok),
    ]
    print("\n" + "=" * 64)
    print(f"{'condition':<18}{'accuracy':>10}{'avg_interv':>12}{'avg_tokens':>14}")
    print("-" * 64)
    for name, ok, ni, tk in rows:
        print(f"{name:<18}{ok/n:>10.2f}{ni/n:>12.2f}{tk/n:>14.1f}")
    print("=" * 64)
    print(f"adaptive observed intervention rate: "
          f"{adaptive_int/(n*len(RELAY_HOPS)):.0%}  (budget for random = {adaptive_int})")

    # demo cases: adaptive correct AND naive wrong
    naive_answer = {}
    for e in log:
        if e["condition"] == "naive" and e["final_answer"]:
            naive_answer[e["item_id"]] = e["final_answer"]
    demos = []
    for it in items:
        if (adaptive_answer.get(it.item_id) == it.gold
                and naive_answer.get(it.item_id) != it.gold_letter):
            demos.append(dict(item_id=it.item_id, question=it.question,
                              choices=it.choices, gold=it.gold_letter,
                              naive_answer=naive_answer.get(it.item_id),
                              adaptive_answer="ABCD"[adaptive_answer[it.item_id]]))
    with open("relay_log.jsonl", "w") as f:
        for e in log:
            f.write(json.dumps(e) + "\n")
    with open("demo_cases.jsonl", "w") as f:
        for d in demos[:3]:
            f.write(json.dumps(d) + "\n")
    print(f"\nwrote relay_log.jsonl ({len(log)} rows), "
          f"demo_cases.jsonl ({min(len(demos),3)} cases)")
    print(f"clean adaptive-fixes-naive cases: {len(demos)}")


if __name__ == "__main__":
    main()
