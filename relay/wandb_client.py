"""W&B Inference client (OpenAI-compatible), traced with ``@op``.

    base_url = https://api.inference.wandb.ai/v1
    api_key  = $WANDB_API_KEY
    project  = $WANDB_PROJECT  (also sent as the "OpenAI-Project" header)

HTTP 429 = the W&B Inference concurrency limit, so we make calls sequentially
and back off exponentially. Only the LIVE round-trip path uses this; the mock
editor never touches it, so mock mode needs no key and no ``openai`` install.
"""

from __future__ import annotations

import os
import time

from .weave_compat import op

STRONG_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
WEAK_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BASE_URL = "https://api.inference.wandb.ai/v1"


class WandbInferenceClient:
    def __init__(self, api_key: str | None = None, project: str | None = None,
                 base_url: str = BASE_URL, max_retries: int = 6):
        self.api_key = api_key or os.environ.get("WANDB_API_KEY")
        self.project = project or os.environ.get("WANDB_PROJECT")
        if not self.api_key:
            raise RuntimeError(
                "WANDB_API_KEY not set. The LIVE path needs W&B Inference creds; "
                "use `--mock` for the no-key path."
            )
        try:
            import openai  # lazy: mock mode must not require openai
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("`pip install openai` to use W&B Inference.") from e

        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=self.api_key,
            project=self.project,
            default_headers={"OpenAI-Project": self.project or ""},
        )
        self.max_retries = max_retries

    @op()
    def chat(self, model: str, system: str, user: str, temperature: float = 0.0,
             max_tokens: int = 512, **kwargs) -> str:
        """One chat completion with sequential 429 backoff. Returns the text."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        delay = 2.0
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                r = self.client.chat.completions.create(
                    model=model, messages=messages, temperature=temperature,
                    max_tokens=max_tokens, **kwargs)
                return r.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001 - we classify by message
                last_err = e
                msg = str(e).lower()
                is_rate = "429" in msg or "rate" in msg or "concurren" in msg
                if attempt < self.max_retries - 1:
                    # 429 -> full backoff; other transient errors -> short backoff.
                    time.sleep(delay if is_rate else min(delay, 4.0))
                    delay *= 2
                    continue
                raise
        assert last_err is not None
        raise last_err
