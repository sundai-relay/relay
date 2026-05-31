"""Thin W&B Inference client (OpenAI-compatible) for the Relay spike.

Throwaway. Sequential calls, temp 0 for answering, 429 backoff.

Auth: set WANDB_API_KEY. Optional: WANDB_PROJECT="entity/project" for trace
attribution (not required for raw inference). Endpoint base url defaults to the
W&B Inference service and is overridable via WANDB_BASE_URL.

NOTE: from some networks (e.g. proxied cloud egress) the W&B endpoint is blocked
at the edge (Cloudflare 1010) regardless of key. Run this where the endpoint is
reachable. You can point it at any OpenAI-compatible endpoint via env if needed.
"""
from __future__ import annotations

import os
import time

from openai import OpenAI

BASE_URL = os.environ.get("WANDB_BASE_URL", "https://api.inference.wandb.ai/v1")
BIG_MODEL = os.environ.get("RELAY_BIG_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
SMALL_MODEL = os.environ.get("RELAY_SMALL_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        key = os.environ.get("WANDB_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Set WANDB_API_KEY (W&B Inference) to run model calls.")
        kwargs = {"base_url": BASE_URL, "api_key": key}
        proj = os.environ.get("WANDB_PROJECT")
        if proj:
            kwargs["project"] = proj
        _client = OpenAI(**kwargs)
    return _client


def chat(
    system: str,
    user: str,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    max_retries: int = 6,
) -> str:
    model = model or BIG_MODEL
    delay = 2.0
    last_err: Exception | None = None
    for _ in range(max_retries):
        try:
            resp = client().chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:  # 429s and transient errors -> backoff
            last_err = e
            msg = str(e).lower()
            if "429" in msg or "rate" in msg or "overload" in msg or "timeout" in msg:
                time.sleep(delay)
                delay = min(delay * 2, 32)
                continue
            raise
    raise RuntimeError(f"chat failed after retries: {last_err}")


def chat_logprobs(
    system: str,
    user: str,
    model: str | None = None,
    top_logprobs: int = 8,
    max_retries: int = 6,
) -> tuple[str, dict[str, float]]:
    """One-token completion with logprobs. Returns (text, {token: prob}).

    Used to read the answerer's calibrated confidence over A/B/C/D from the
    model's own distribution, rather than self-reported confidence.
    """
    import math

    model = model or BIG_MODEL
    delay = 2.0
    last_err: Exception | None = None
    for _ in range(max_retries):
        try:
            resp = client().chat.completions.create(
                model=model,
                temperature=0.0,
                max_tokens=1,
                logprobs=True,
                top_logprobs=top_logprobs,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            choice = resp.choices[0]
            text = (choice.message.content or "").strip()
            probs: dict[str, float] = {}
            lp = choice.logprobs
            if lp and lp.content:
                for alt in lp.content[0].top_logprobs:
                    probs[alt.token.strip().upper()] = math.exp(alt.logprob)
            return text, probs
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "429" in msg or "rate" in msg or "overload" in msg or "timeout" in msg:
                time.sleep(delay)
                delay = min(delay * 2, 32)
                continue
            raise
    raise RuntimeError(f"chat_logprobs failed after retries: {last_err}")


def token_proxy(*texts: str) -> int:
    """Crude cost unit: ~word count of everything sent+received."""
    return sum(len(t.split()) for t in texts)


if __name__ == "__main__":
    print("BASE_URL:", BASE_URL)
    print("BIG_MODEL:", BIG_MODEL, "| SMALL_MODEL:", SMALL_MODEL)
    print("WANDB_API_KEY set:", bool(os.environ.get("WANDB_API_KEY")))
    try:
        out = chat("You answer in one word.", "Say: ready", max_tokens=8)
        print("live call OK ->", out)
    except Exception as e:
        print("live call FAILED ->", type(e).__name__, str(e)[:160])
