"""GO/NO-GO gate (port of relay-main/spike/relay_sanity.py).

Question: do compounding lossless round-trips actually corrupt the ledger (naive), and is
that corruption recoverable by re-grounding every boundary (always)? If so there is a real
fidelity gap for the adaptive policy to recover — build the four conditions.

Prints fid_identity (must be 1.0 — scorer sanity), fid_naive, fid_always, and per-boundary
invariant_deviation. GREEN if  fid_always - fid_naive >= 0.20.

Run:  WANDB_API_KEY=... [WANDB_PROJECT=entity/project] python run_gate.py --n 12 --depth 4
"""
from __future__ import annotations

import argparse

from conductor import run_chain
from data import load_chains
from domain import score
from llm import BIG_MODEL, init_weave

GATE_GAP = 0.20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    init_weave()
    chains = load_chains(n=args.n, depth=args.depth, seed=args.seed)
    print(f"[gate] {len(chains)} chains | depth {args.depth} | editor={BIG_MODEL}\n")

    log: list = []
    fid_naive, fid_always = [], []
    for c in chains:
        nv, _, ntok, _ = run_chain(c, mode="naive", log=log, condition="naive")
        al, ai, atok, _ = run_chain(c, mode="always", log=log, condition="always")
        fn, fa = score(nv, c.seed_text), score(al, c.seed_text)
        fid_naive.append(fn)
        fid_always.append(fa)
        print(f"{c.sample_id:14} naive={fn:.3f}  always={fa:.3f}  (always interv={ai})")

    n = len(chains)
    avg_naive = sum(fid_naive) / n
    avg_always = sum(fid_always) / n
    fid_identity = sum(score(c.seed_text, c.seed_text) for c in chains) / n
    avg_dev_naive = (sum(e["invariant_deviation"] for e in log if e["condition"] == "naive")
                     / max(1, sum(1 for e in log if e["condition"] == "naive")))

    print("\n--- gate (accounting round-trip relay) ---")
    print(f"fid_identity (score(D0,D0)) : {fid_identity:.3f}   (must be 1.000)")
    print(f"fid_naive                   : {avg_naive:.3f}")
    print(f"fid_always                  : {avg_always:.3f}")
    print(f"gap (always - naive)        : {avg_always - avg_naive:.3f}   (need >= {GATE_GAP})")
    print(f"avg invariant_deviation     : {avg_dev_naive:.3f}  (naive, per boundary)")

    green = (avg_always - avg_naive) >= GATE_GAP and fid_identity > 0.999
    print("\n>>> " + ("GREEN — build the four conditions (run_conditions.py)"
                       if green else
                       "RED — apply a disclosed lever: --depth 6, weaker editor "
                       "(RELAY_BIG_MODEL=meta-llama/Llama-3.1-8B-Instruct), or more samples") + " <<<")


if __name__ == "__main__":
    main()
