"""Relay agents (domain-agnostic) for the DELEGATE52 harness.

  editor   — applies ONE natural-language edit to the current document. Where corruption
             happens. It NEVER sees the seed D0; only the current files + the instruction.
  reground — the Conductor's repair pass: given the authoritative content the document lost
             (computed by diff against D0), restore it. Analog of relay-main's reground().

Domain-specific prompts, scoring, and the offline mock live in domain.py (which dispatches to
ledger.py for accounting or chess_domain.py for chess via RELAY_DOMAIN). This module just
orchestrates the LLM call (or the mock) and threads real token cost up to the frontier.
"""
from __future__ import annotations

import os

import domain
from llm import BIG_MODEL, EDITOR_MODEL, ChatResult, chat

# Outputs run ~3-4k tokens; give headroom so a full document is never truncated.
EDIT_MAX_TOKENS = 6000
REPAIR_MAX_TOKENS = 6000

# RELAY_MOCK=1 -> deterministic local editor, NO API. Runs the whole four-condition eval in
# seconds to validate the pipeline. Corruption is SYNTHETIC (see domain.mock_edit), so mock
# numbers prove the machinery works, not real LLM corruption — drop the flag for the real run.
MOCK = os.environ.get("RELAY_MOCK", "") not in ("", "0")


def editor(doc_text: str, instruction: str, model: str = EDITOR_MODEL) -> ChatResult:
    """Apply one edit instruction to the current document. Faithful by intent, lossy in
    practice under restructuring — that is the phenomenon we measure."""
    if MOCK:
        out = domain.mock_edit(doc_text, instruction)
        return ChatResult(text=out, tokens=len(doc_text.split()) + len(out.split()))
    usr = f"CURRENT FILES:\n{doc_text}\n\nINSTRUCTION:\n{instruction}\n\nOutput the resulting file(s):"
    return chat(domain.editor_system_prompt(), usr, model=model,
                temperature=0.0, max_tokens=EDIT_MAX_TOKENS)


def reground(doc_text: str, missing_blocks: list[str], model: str = BIG_MODEL) -> ChatResult:
    """Repair the document using the authoritative content the Conductor determined is
    missing/altered (computed by diff against D0). Restore it faithfully."""
    if MOCK:
        out = domain.mock_reground(doc_text, missing_blocks)
        cost = len(doc_text.split()) + sum(len(b.split()) for b in missing_blocks)
        return ChatResult(text=out, tokens=cost)
    src = "\n\n".join(missing_blocks)
    usr = (
        f"CURRENT DOCUMENT:\n{doc_text}\n\n"
        f"{domain.repair_source_label()}:\n{src}\n\n"
        "Output the corrected document:"
    )
    return chat(domain.repair_system_prompt(), usr, model=model,
                temperature=0.0, max_tokens=REPAIR_MAX_TOKENS)


# Weave-trace the agents if weave is available (no-op otherwise).
try:
    import weave
    editor = weave.op()(editor)      # type: ignore[assignment]
    reground = weave.op()(reground)  # type: ignore[assignment]
except ImportError:
    pass


if __name__ == "__main__":
    from llm import init_weave
    init_weave()
    print(f"domain={domain.DOMAIN if hasattr(domain,'DOMAIN') else '?'} MOCK={MOCK}")
