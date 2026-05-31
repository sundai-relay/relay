"""Benchmark loader for the Relay feasibility spike.

Throwaway code. Goal: get ~15-20 passage-grounded MCQ items where the correct
answer depends on a detail *in the passage* (so a lossy relay can drop it).

Primary source: RACE (reading-comprehension MCQ, passage + 4 options + gold).
We pick items with a moderately long passage so there is something to lose.
Fallback: a tiny embedded set so the signal pipeline is testable fully offline.

Gold answers live here but are used ONLY for final scoring, never fed to any
agent or signal.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Item:
    item_id: str
    source: str          # the passage
    question: str
    choices: list[str]   # 4 options, index 0..3 == A..D
    gold: int            # 0..3, used for scoring ONLY

    @property
    def gold_letter(self) -> str:
        return "ABCD"[self.gold]


def _from_race(n: int, seed: int, min_words: int, max_words: int) -> list[Item]:
    from datasets import load_dataset

    # "all" merges middle+high school; test split keeps it out of any training mix.
    # ehovy/race is the namespaced Hub mirror (legacy "race" script id no longer resolves).
    ds = load_dataset("ehovy/race", "all", split="test")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)

    items: list[Item] = []
    for i in idx:
        row = ds[i]
        wc = len(row["article"].split())
        if not (min_words <= wc <= max_words):
            continue
        if len(row["options"]) != 4:
            continue
        gold = "ABCD".index(row["answer"].strip().upper())
        items.append(
            Item(
                item_id=f"race-{i}",
                source=row["article"].strip(),
                question=row["question"].strip(),
                choices=[o.strip() for o in row["options"]],
                gold=gold,
            )
        )
        if len(items) >= n:
            break
    return items


def _embedded() -> list[Item]:
    """Small offline fallback. Each passage has one detail that decides the answer."""
    raw = [
        {
            "source": (
                "The Halworth Bridge was completed in 1923 after eleven years of "
                "construction. It was originally painted dark green, but during the "
                "1957 restoration the city council voted to repaint it a bright "
                "vermilion red to improve visibility in fog. The bridge spans the "
                "River Tame and carries both rail and pedestrian traffic. In 1984 a "
                "second pedestrian deck was added beneath the rail line. Despite "
                "several proposals, the original red colour has been retained ever "
                "since the 1957 restoration."
            ),
            "question": "What colour is the Halworth Bridge today?",
            "choices": ["Dark green", "Vermilion red", "Grey", "Blue"],
            "gold": 1,
        },
        {
            "source": (
                "Dr. Mensah's lab studied three enzymes: alpha, beta, and gamma. "
                "Alpha was active only above 40C. Beta worked across a wide range "
                "but was destroyed by light. Gamma, the focus of the 2019 paper, "
                "remained stable in darkness and at room temperature, which made it "
                "the preferred choice for the portable field kit the team eventually "
                "shipped. The kit had no temperature control and was often used "
                "outdoors at night."
            ),
            "question": "Which enzyme was chosen for the portable field kit?",
            "choices": ["Alpha", "Beta", "Gamma", "None of them"],
            "gold": 2,
        },
        {
            "source": (
                "The novel's narrator, Iris, is unreliable. Early on she claims her "
                "brother Tom died at sea. Only in the final chapter do we learn that "
                "Tom is alive and living in Lisbon, and that Iris invented his death "
                "to avoid explaining their estrangement. The sea voyage she describes "
                "in chapter two never happened."
            ),
            "question": "What is true about Tom by the end of the novel?",
            "choices": [
                "He died at sea",
                "He is alive in Lisbon",
                "He never existed",
                "He drowned in a river",
            ],
            "gold": 1,
        },
        {
            "source": (
                "The committee met four times. The first three meetings ended without "
                "a vote. At the fourth meeting, held in March, the proposal to extend "
                "the library hours passed by a single vote, 6 to 5. The chair, who "
                "had opposed the measure in earlier discussions, cast the deciding "
                "vote in favour after new survey data was presented."
            ),
            "question": "How did the proposal to extend library hours fare?",
            "choices": [
                "It failed at every meeting",
                "It passed 6 to 5 at the fourth meeting",
                "It passed unanimously",
                "It was withdrawn",
            ],
            "gold": 1,
        },
        {
            "source": (
                "Three climbers attempted the north face. Petra turned back at the "
                "second camp due to frostbite. Lars reached the summit but only after "
                "the storm cleared on the third day. Yuki, the most experienced, was "
                "forced to abandon the climb when her oxygen regulator failed below "
                "the final ridge. Only one of the three stood on the summit."
            ),
            "question": "Who reached the summit?",
            "choices": ["Petra", "Lars", "Yuki", "All three"],
            "gold": 1,
        },
        {
            "source": (
                "The recipe was passed down through four generations. The original "
                "version used honey, but Grandmother Sofia, facing wartime rationing "
                "in 1942, replaced the honey with grated apple. Every later version of "
                "the family recipe has used grated apple, and the honey was never "
                "reinstated even after rationing ended."
            ),
            "question": "What sweetener does the current family recipe use?",
            "choices": ["Honey", "Grated apple", "White sugar", "Maple syrup"],
            "gold": 1,
        },
    ]
    return [
        Item(
            item_id=f"embed-{i}",
            source=r["source"],
            question=r["question"],
            choices=r["choices"],
            gold=r["gold"],
        )
        for i, r in enumerate(raw)
    ]


def load_items(
    n: int = 18,
    seed: int = 13,
    min_words: int = 250,
    max_words: int = 600,
) -> list[Item]:
    """Try RACE; on any failure fall back to the embedded set."""
    try:
        items = _from_race(n=n, seed=seed, min_words=min_words, max_words=max_words)
        if len(items) >= max(3, n // 2):
            return items
        print(f"[data] RACE returned only {len(items)} items; using embedded fallback.")
    except Exception as e:  # offline / dataset unavailable
        print(f"[data] RACE load failed ({type(e).__name__}: {e}); using embedded fallback.")
    return _embedded()


if __name__ == "__main__":
    items = load_items()
    print(f"loaded {len(items)} items")
    it = items[0]
    print("source words:", len(it.source.split()))
    print("Q:", it.question)
    for j, c in enumerate(it.choices):
        print(f"  {'ABCD'[j]}. {c}")
    print("gold:", it.gold_letter)
