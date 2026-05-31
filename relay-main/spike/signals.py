"""Key-free handoff-degradation signals for the Relay spike.

Two signals, neither touches the gold answer:

1. question_conditioned_drift
   Chunk the source (~300 tokens/chunk, word-approx). Retrieve the top-3 chunks
   most relevant to the *question* (by MiniLM embedding cosine). Drift of a memo
   = 1 - cosine(memo_embedding, mean_of_top3_chunk_embeddings). We compare the
   memo against the question-relevant chunks, not the whole source, so honest
   compression isn't punished.

2. answer_instability
   Handled in probe.py via the shadow answerer (an LLM call), since it needs the
   model. Here we only provide the embedding/drift machinery + risk combiner.

risk = drift_delta + answer_instability   (transparent, intentionally crude)
"""
from __future__ import annotations

import functools

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@functools.lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(_MODEL_NAME)


def embed(texts: list[str]) -> np.ndarray:
    vecs = _model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    # inputs already normalized -> cosine == dot
    return float(np.dot(a, b))


def chunk_source(source: str, chunk_words: int = 300, overlap: int = 50) -> list[str]:
    words = source.split()
    if not words:
        return [source]
    chunks, start = [], 0
    step = max(1, chunk_words - overlap)
    while start < len(words):
        chunks.append(" ".join(words[start : start + chunk_words]))
        start += step
    return chunks


class Retriever:
    """Precomputes chunk embeddings for one item; retrieves top-k by question."""

    def __init__(self, source: str, chunk_words: int = 300, overlap: int = 50):
        self.chunks = chunk_source(source, chunk_words, overlap)
        self.chunk_vecs = embed(self.chunks)

    def top_k(self, question: str, k: int = 3) -> tuple[list[str], np.ndarray]:
        qv = embed([question])[0]
        sims = self.chunk_vecs @ qv
        order = np.argsort(-sims)[: min(k, len(self.chunks))]
        picked = [self.chunks[i] for i in order]
        centroid = self.chunk_vecs[order].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        return picked, centroid

    def drift(self, memo: str, question_centroid: np.ndarray) -> float:
        mv = embed([memo])[0]
        return 1.0 - _cos(mv, question_centroid)


def risk(margin_drop: float, answer_flip: int) -> float:
    """risk = answer_flip + (prev_margin - cur_margin).

    The flip term is binary (so flips always rank above non-flips), but the
    margin-drop term is CONTINUOUS, which is what makes risk thresholdable to an
    arbitrary intervention budget. It also catches the "answer didn't flip but
    confidence cratered" near-flip that the pure flip signal misses.
    Embedding drift is logged for comparison but deliberately NOT in risk: it was
    near-inert at this scale (see RESULTS.md).
    """
    return float(answer_flip) + float(margin_drop)


if __name__ == "__main__":
    src = (
        "The bridge was painted green in 1923. In 1957 it was repainted bright "
        "red for fog visibility and has stayed red ever since. It crosses the "
        "River Tame and carries rail and foot traffic. A lower deck was added in "
        "1984."
    ) * 4
    r = Retriever(src, chunk_words=40, overlap=10)
    chunks, centroid = r.top_k("What colour is the bridge now?", k=3)
    faithful = "The bridge is currently red, repainted in 1957 for fog visibility."
    lossy = "The bridge crosses a river and carries trains and pedestrians."
    print("chunks:", len(r.chunks))
    print("drift(faithful):", round(r.drift(faithful, centroid), 4))
    print("drift(lossy)   :", round(r.drift(lossy, centroid), 4))
