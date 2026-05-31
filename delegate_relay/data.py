"""Loader for DELEGATE52 accounting round-trip chains.

A *round-trip* = a forward edit (e.g. "split this ledger by person") followed by its
inverse (e.g. "merge these files into one ledger sorted by date"). A faithful round-trip
returns the seed exactly, so score(boundary, seed) has a true 1.0 ceiling.

A *chain* = several round-trips applied in sequence, each fed the previous boundary's
output, so corruption compounds (the paper's "25% after 20 interactions" regime). This is
the document analog of relay-main's memo-compression hops.

We keep ONLY structurally-lossless edit families (split/merge + bijective format
conversions) so every boundary is legitimately comparable to the seed. FX conversions,
website export, account-flattening, and prose rewrites are excluded — see SELECTION RULE.

The seed text (gold D0) lives here but is used ONLY by the scorer and the Conductor; the
editor agents never see it during a relay hop.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field

# SELECTION RULE (write this into the README): a forward edit is round-trip-eligible iff
# its target state is not a "cpx_" (complex multi-file) state AND its inverse edit's
# semantic_operations are a subset of the lossless set below. We keep ONLY split/merge,
# classification, and sorting — these reorganize transactions but preserve the exact ledger
# surface form, so a faithful round-trip returns the seed byte-for-byte and the scorer is fair.
# We deliberately DROP `format_knowledge` (csv/beancount conversions): translating to another
# format and back legitimately changes surface syntax (e.g. `$202.00` -> `202.00 USD`, ISO
# dates), which a surface-form scorer would mis-read as total corruption — a format-translation
# confound, not agent corruption. Also dropped: FX (numerical_reasoning), website
# (context_expansion), flatten (string_manipulation), receipts (referencing).
LOSSLESS_OPS = {"split_and_merge", "classification", "sorting"}

_DATA_PATH = os.environ.get(
    "DELEGATE_JSONL",
    os.path.join(os.path.dirname(__file__), "..", "delegate", "delegate52.jsonl"),
)


@dataclass
class RoundTrip:
    """One forward+inverse edit pair. The unit of relay (one 'hop')."""
    label: str            # e.g. "person_split"
    forward_prompt: str   # NL instruction applied to the current ledger
    inverse_prompt: str   # NL instruction that should restore the ledger
    ops: tuple[str, ...]  # semantic_operations of the inverse (for logging/slicing)


@dataclass
class Chain:
    """A seed ledger + an ordered list of round-trips to apply in sequence."""
    sample_id: str
    seed_text: str        # gold D0 — scorer/Conductor only, never shown to the editor
    steps: list[RoundTrip] = field(default_factory=list)

    @property
    def depth(self) -> int:
        return len(self.steps)


def _eligible_round_trips(sample: dict) -> list[RoundTrip]:
    start = sample["metadata"]["start_state"]
    forward = {p["target_state"]: p
               for s in sample["states"] if s["state_id"] == start
               for p in s["prompts"]}
    rts: list[RoundTrip] = []
    for s in sample["states"]:
        sid = s["state_id"]
        if sid == start or sid.startswith("cpx_"):
            continue
        ops = set(s.get("semantic_operations", []))
        if not ops or not ops.issubset(LOSSLESS_OPS):
            continue
        back = next((p for p in s["prompts"] if p["target_state"] == start), None)
        fwd = forward.get(sid)
        if back is None or fwd is None:
            continue
        rts.append(RoundTrip(
            label=sid.replace("_state", ""),
            forward_prompt=fwd["prompt"].strip(),
            inverse_prompt=back["prompt"].strip(),
            ops=tuple(sorted(ops)),
        ))
    return rts


def _round_trips_generic(sample: dict) -> list[RoundTrip]:
    """Domain-agnostic selection (used for non-accounting domains, e.g. chess): keep every
    non-complex state that has a forward edit AND an inverse back to the start state. The
    domain scorer decides what 'faithful' means (e.g. chess scores on the move sequence, so
    annotation-format round-trips are fine)."""
    start = sample["metadata"]["start_state"]
    forward = {p["target_state"]: p
               for s in sample["states"] if s["state_id"] == start
               for p in s["prompts"]}
    rts: list[RoundTrip] = []
    for s in sample["states"]:
        sid = s["state_id"]
        if sid == start or sid.startswith("cpx_"):
            continue
        back = next((p for p in s["prompts"] if p["target_state"] == start), None)
        fwd = forward.get(sid)
        if back is None or fwd is None:
            continue
        rts.append(RoundTrip(label=sid.replace("_state", ""),
                             forward_prompt=fwd["prompt"].strip(),
                             inverse_prompt=back["prompt"].strip(),
                             ops=tuple(sorted(s.get("semantic_operations", [])))))
    return rts


def _pool(sample: dict, domain: str) -> list[RoundTrip]:
    return _eligible_round_trips(sample) if domain == "accounting" else _round_trips_generic(sample)


def load_samples(domain: str | None = None) -> list[dict]:
    domain = domain or os.environ.get("RELAY_DOMAIN", "accounting")
    rows = [json.loads(l) for l in open(_DATA_PATH)]
    return [r for r in rows if r["sample_type"] == domain]


def _seed_text(sample: dict) -> str:
    start = sample["metadata"]["start_state"]
    key = next(k for k in sample["files"] if k.startswith(start + "/"))
    return sample["files"][key]


def load_chains(n: int = 12, depth: int = 4, seed: int = 13,
                domain: str | None = None) -> list[Chain]:
    """Build n round-trip chains of the given depth, spread across the domain's samples.
    When a sample's eligible pool < depth, round-trips repeat (re-running a lossless
    round-trip still compounds corruption — a valid hop)."""
    domain = domain or os.environ.get("RELAY_DOMAIN", "accounting")
    rng = random.Random(seed)
    samples = load_samples(domain)
    pools = {s["sample_id"]: _pool(s, domain) for s in samples}
    seeds = {s["sample_id"]: _seed_text(s) for s in samples}
    samples = [s for s in samples if pools[s["sample_id"]]]  # drop any with no pool

    chains: list[Chain] = []
    for i in range(n):
        sample = samples[i % len(samples)]
        sid = sample["sample_id"]
        pool = pools[sid]
        order = pool[:]
        rng.shuffle(order)
        steps = [order[j % len(order)] for j in range(depth)]  # fill, repeat if needed
        rng.shuffle(steps)
        chains.append(Chain(sample_id=f"{sid}#{i//len(samples)}",
                            seed_text=seeds[sid], steps=steps))
    return chains


if __name__ == "__main__":
    chains = load_chains(n=12, depth=4)
    print(f"loaded {len(chains)} chains | depth {chains[0].depth}")
    pools = {s["sample_id"]: len(_eligible_round_trips(s)) for s in load_samples()}
    print("eligible pool sizes:", pools)
    for c in chains[:6]:
        print(f"\n{c.sample_id}  seed={len(c.seed_text.split())}w")
        for k, rt in enumerate(c.steps):
            print(f"  hop{k+1} [{rt.label:16}] fwd: {rt.forward_prompt[:60]}")
