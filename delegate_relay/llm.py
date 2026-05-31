"""W&B Inference client (OpenAI-compatible) + Weave tracing for the DELEGATE52 relay.

Ported from relay-main/spike/llm.py. Changes:
  - editor/conductor model = openai/gpt-oss-120b (the locked config)
  - chat() returns ChatResult(text, tokens) with REAL usage token counts (not a word proxy)
  - @weave.op so every model call is traced; call init_weave() once at program start
  - temp-0 memoization so the shared naive prefix isn't recomputed across the 4 conditions

Auth: WANDB_API_KEY (W&B Inference). Optional WANDB_PROJECT="entity/project" enables Weave
tracing + trace attribution. Endpoint overridable via WANDB_BASE_URL.
"""
from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import dataclass

from openai import OpenAI

BASE_URL = os.environ.get("WANDB_BASE_URL", "https://api.inference.wandb.ai/v1")
# EDITOR_MODEL = the relay agent where corruption happens (fast, non-reasoning, weak so
#   loss is visible — the brief's design). BIG_MODEL = the Conductor's repair model.
# NOTE: openai/gpt-oss-120b was tested and rejected as the editor — it's a reasoning model
#   that burns the whole token budget on hidden reasoning (0 content unless throttled) and
#   takes 60-140s/call, impractical for a many-call harness. Set RELAY_EDITOR_MODEL to use it.
EDITOR_MODEL = os.environ.get("RELAY_EDITOR_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
BIG_MODEL = os.environ.get("RELAY_BIG_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
SMALL_MODEL = os.environ.get("RELAY_SMALL_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

_CACHE_ON = os.environ.get("RELAY_CACHE", "1") != "0"
_cache: dict[str, "ChatResult"] = {}
_client: OpenAI | None = None
_weave_inited = False


@dataclass
class ChatResult:
    text: str
    tokens: int  # prompt + completion (real usage)


def init_weave() -> None:
    """Initialize Weave once. Safe to call repeatedly. No-op without WANDB_PROJECT."""
    global _weave_inited
    if _weave_inited:
        return
    proj = os.environ.get("WANDB_PROJECT")
    if proj:
        import weave
        weave.init(proj)
    _weave_inited = True


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


def _key(model: str, system: str, user: str, temperature: float, max_tokens: int) -> str:
    h = hashlib.sha256()
    for part in (model, system, user, f"{temperature}", f"{max_tokens}"):
        h.update(part.encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()


# gpt-oss/reasoning models burn the whole token budget on hidden reasoning and emit no
# content unless reasoning is throttled. Auto-apply low effort for those models.
_REASONING_EFFORT = os.environ.get("RELAY_REASONING_EFFORT", "low")


def _extra_body(model: str) -> dict:
    if "gpt-oss" in model or "o1" in model or "o3" in model:
        return {"reasoning_effort": _REASONING_EFFORT}
    return {}


def _chat_uncached(system, user, model, temperature, max_tokens, max_retries):
    delay = 2.0
    last_err = None
    for _ in range(max_retries):
        try:
            resp = client().chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                extra_body=_extra_body(model),
            )
            text = (resp.choices[0].message.content or "").strip()
            usage = getattr(resp, "usage", None)
            tokens = int(getattr(usage, "total_tokens", 0)) if usage else 0
            return ChatResult(text=text, tokens=tokens)
        except Exception as e:  # 429 / transient -> backoff
            last_err = e
            msg = str(e).lower()
            if any(s in msg for s in ("429", "rate", "overload", "timeout", "503")):
                time.sleep(delay)
                delay = min(delay * 2, 32)
                continue
            raise
    raise RuntimeError(f"chat failed after retries: {last_err}")


try:
    import weave

    @weave.op()
    def chat(system: str, user: str, model: str | None = None,
             temperature: float = 0.0, max_tokens: int = 2048,
             max_retries: int = 6) -> ChatResult:
        model = model or BIG_MODEL
        if _CACHE_ON and temperature == 0.0:
            k = _key(model, system, user, temperature, max_tokens)
            if k in _cache:
                return _cache[k]
            res = _chat_uncached(system, user, model, temperature, max_tokens, max_retries)
            _cache[k] = res
            return res
        return _chat_uncached(system, user, model, temperature, max_tokens, max_retries)
except ImportError:  # weave not installed -> plain function
    def chat(system: str, user: str, model: str | None = None,
             temperature: float = 0.0, max_tokens: int = 2048,
             max_retries: int = 6) -> ChatResult:
        model = model or BIG_MODEL
        if _CACHE_ON and temperature == 0.0:
            k = _key(model, system, user, temperature, max_tokens)
            if k in _cache:
                return _cache[k]
            res = _chat_uncached(system, user, model, temperature, max_tokens, max_retries)
            _cache[k] = res
            return res
        return _chat_uncached(system, user, model, temperature, max_tokens, max_retries)


def chat_logprobs(system: str, user: str, model: str | None = None,
                  top_logprobs: int = 8, max_retries: int = 6) -> tuple[str, dict[str, float]]:
    """One-token completion with logprobs (optional secondary signal). Returns
    (text, {token: prob}). Confirmed available on the W&B Inference endpoint."""
    model = model or BIG_MODEL
    delay = 2.0
    last_err = None
    for _ in range(max_retries):
        try:
            resp = client().chat.completions.create(
                model=model, temperature=0.0, max_tokens=1,
                logprobs=True, top_logprobs=top_logprobs,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
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
            if any(s in msg for s in ("429", "rate", "overload", "timeout", "503")):
                time.sleep(delay)
                delay = min(delay * 2, 32)
                continue
            raise
    raise RuntimeError(f"chat_logprobs failed after retries: {last_err}")


if __name__ == "__main__":
    print("BASE_URL:", BASE_URL, "| BIG_MODEL:", BIG_MODEL)
    print("WANDB_API_KEY set:", bool(os.environ.get("WANDB_API_KEY")))
    init_weave()
    try:
        r = chat("You answer in one word.", "Say: ready", max_tokens=8)
        print("live call OK ->", repr(r.text), "| tokens:", r.tokens)
    except Exception as e:
        print("live call FAILED ->", type(e).__name__, str(e)[:160])
