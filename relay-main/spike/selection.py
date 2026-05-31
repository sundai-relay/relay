"""Select the passage-dependent subpopulation.

Handoff loss can only show up on items that (a) are answerable from the source
and (b) are NOT answerable blind. Keeping only those scopes the experiment to
where the phenomenon can occur. This is selection, not cheating: gold is still
used only for scoring, and we disclose the filtering.
"""
from __future__ import annotations

from agents import answerer
from data import Item
from llm import BIG_MODEL


def passage_dependent(items: list[Item], n: int, model: str = BIG_MODEL,
                      verbose: bool = True) -> list[Item]:
    keep: list[Item] = []
    for it in items:
        src_ok = answerer(it.source, it.question, it.choices, model=model) == it.gold
        blind_ok = answerer("(no information provided)", it.question, it.choices,
                            model=model) == it.gold
        if src_ok and not blind_ok:
            keep.append(it)
            if verbose:
                print(f"  keep {it.item_id} (source=Y blind=n)  [{len(keep)}/{n}]")
        if len(keep) >= n:
            break
    return keep
