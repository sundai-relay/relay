#!/usr/bin/env python3
"""Relay — multi-agent handoff-degradation harness. CLI entry point.

Examples
--------
    # No API key needed — the acceptance run:
    python run.py --substrate mock --condition all --n 5

    # A single condition:
    python run.py --substrate mock --condition adaptive --n 10

    # Real substrates need W&B Inference creds (see README):
    export WANDB_API_KEY=...   WANDB_PROJECT=entity/project
    python run.py --substrate roundtrip --condition all --n 5
"""

from __future__ import annotations

import argparse

from relay import weave_compat
from relay.conditions import run_conditions
from relay.leaderboard import aggregate, print_leaderboard, write_jsonl
from relay.substrates import get_substrate
from relay.weave_leaderboard import maybe_publish


def main() -> None:
    p = argparse.ArgumentParser(description="Relay handoff-degradation harness")
    p.add_argument("--substrate", choices=["mock", "roundtrip", "mcq"], default="mock")
    p.add_argument("--condition",
                   choices=["naive", "always", "adaptive", "random", "all"], default="all")
    p.add_argument("--n", type=int, default=5, help="number of episodes")
    p.add_argument("--threshold", type=float, default=0.4,
                   help="adaptive risk threshold (re-ground when risk > threshold)")
    p.add_argument("--hops", type=int, default=None, help="(mock) hops per episode")
    p.add_argument("--corruption", type=float, default=None,
                   help="(mock) per-hop corruption probability")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/results.jsonl")
    args = p.parse_args()

    # Weave only initializes with real creds; safe no-op otherwise.
    weave_active = weave_compat.maybe_init()

    kw = {"seed": args.seed}
    if args.substrate == "mock":
        if args.hops is not None:
            kw["n_hops"] = args.hops
        if args.corruption is not None:
            kw["corruption_p"] = args.corruption
    substrate = get_substrate(args.substrate, **kw)

    episodes = substrate.load_episodes(args.n)
    print(f"[relay] substrate={args.substrate} condition={args.condition} "
          f"episodes={len(episodes)} threshold={args.threshold} "
          f"weave={'on' if weave_active else 'off'}")

    results = run_conditions(substrate, episodes, which=args.condition,
                             threshold=args.threshold, seed=args.seed)

    path = write_jsonl(results, args.out)
    summary = aggregate(results)
    print_leaderboard(summary, title=f"Relay — {args.substrate} leaderboard")
    print(f"[relay] per-hop log -> {path}")

    if weave_active:
        maybe_publish(results, substrate, episodes,
                      threshold=args.threshold, seed=args.seed)


if __name__ == "__main__":
    main()
