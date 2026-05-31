"""Domain dispatcher: selects the scorer/parser/prompts/mock by RELAY_DOMAIN (default
'accounting'). Resolved at import time, so set RELAY_DOMAIN in the environment BEFORE
launching (e.g. `RELAY_DOMAIN=chess python run_conditions.py ...`).

Every domain exposes the same interface so the shared pipeline (signals/conductor/agents/
run_conditions) is domain-agnostic:
  measurement : parse_doc, invariants, invariant_deviation, parse_health, score, diff_blocks
  real prompts: editor_system_prompt, repair_system_prompt, repair_source_label
  offline mock: mock_edit, mock_reground
"""
from __future__ import annotations

import hashlib
import os
import random
import re

DOMAIN = os.environ.get("RELAY_DOMAIN", "accounting")

if DOMAIN == "chess":
    from chess_domain import (  # noqa: F401
        diff_blocks, editor_system_prompt, invariant_deviation, invariants,
        mock_edit, mock_reground, parse_doc, parse_health, repair_source_label,
        repair_system_prompt, score,
    )
else:
    # ---- accounting (ledger.py provides measurement; prompts + mock live here) ----
    from ledger import (  # noqa: F401
        diff_blocks, invariant_deviation, invariants, parse_health, score,
    )
    from ledger import parse_ledger as parse_doc  # noqa: F401

    def editor_system_prompt() -> str:
        return ("You are an assistant that edits accounting documents on a user's behalf. "
                "Apply the user's instruction to the CURRENT FILES exactly and completely. "
                "Preserve every transaction, amount, account, date, and receipt reference unless "
                "the instruction explicitly changes it — do NOT summarize, truncate, or drop data. "
                "Output ONLY the resulting file contents, no commentary. If the result is multiple "
                "files, separate each with a line of the form '=== <filename> ==='.")

    def repair_system_prompt() -> str:
        return ("You are a grounding agent that repairs an accounting ledger which lost or altered "
                "transactions during automated editing. You are given the AUTHORITATIVE source "
                "transactions (ground truth). Merge them into the current document so that every "
                "authoritative transaction appears exactly once, with its original date, payee, "
                "accounts, amounts, and receipt references. Remove duplicates and fix altered "
                "entries. Output ONLY a single corrected accounting.ledger, sorted by date.")

    def repair_source_label() -> str:
        return "AUTHORITATIVE SOURCE TRANSACTIONS (must all be present, exactly once)"

    def _blocks(text: str) -> list[str]:
        return [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]

    def mock_edit(doc_text: str, instruction: str) -> str:
        """Simulate a lossy edit: drop a per-instruction fraction (5-30%) of transaction blocks
        and occasionally zero one amount. Deterministic; drop rate varies by instruction so some
        boundaries corrupt more than others (gives the risk signal decision value)."""
        blocks = _blocks(doc_text)
        if not blocks:
            return doc_text
        h = hashlib.sha256(instruction.encode()).hexdigest()
        rng = random.Random(int(hashlib.sha256((instruction[:60] + str(len(blocks))).encode()).hexdigest()[:12], 16))
        drop = 0.05 + 0.25 * (int(h[:4], 16) / 0xFFFF)
        kept = [b for b in blocks if rng.random() > drop]
        if kept and rng.random() < 0.3:
            i = rng.randrange(len(kept))
            kept[i] = re.sub(r"\$[\d,]+\.\d{2}", "$0.00", kept[i], count=1)
        return "\n\n".join(kept)

    def mock_reground(doc_text: str, missing_blocks: list[str]) -> str:
        """Faithful local repair: re-inject the authoritative missing transaction blocks."""
        out = doc_text.rstrip()
        if missing_blocks:
            out = out + "\n\n" + "\n\n".join(missing_blocks)
        return out
