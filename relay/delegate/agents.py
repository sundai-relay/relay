"""Editor + targeted-repair agents for the real DELEGATE52 round trips.

The editor applies one DELEGATE52 NL instruction to a text document (a ledger or
a PGN, or an intermediate representation the editor itself produced) and returns
the resulting file content as plain text. The repairer is handed the seed, the
corrupted reconstruction, the instruction, and the domain checksum report, and
restores what the checksum flagged while preserving the legitimate edit.

Real calls go through W&B Inference (reusing the round-trip client + temp-0 disk
cache). A deterministic line-dropping mock keeps the pipeline runnable with no
key for smoke tests.
"""
from __future__ import annotations

import hashlib
import os
import random
import threading
from typing import Dict, Optional

from ..wandb_client import WEAK_MODEL, WandbInferenceClient

_CONFIG = {"use_mock": True, "model": WEAK_MODEL, "provider": "wandb",
           "slip_p": 0.5, "max_tokens": 4096}
_CLIENT: Optional[WandbInferenceClient] = None
_LOCK = threading.Lock()


def configure(use_mock: Optional[bool] = None, model: str = WEAK_MODEL,
              provider: str = "wandb", slip_p: float = 0.5,
              max_tokens: int = 4096) -> bool:
    if use_mock is None:
        env = "OPENAI_API_KEY" if provider == "openai" else "WANDB_API_KEY"
        use_mock = not bool(os.environ.get(env))
    _CONFIG.update(use_mock=use_mock, model=model, provider=provider,
                   slip_p=slip_p, max_tokens=max_tokens)
    return use_mock


def is_mock() -> bool:
    return _CONFIG["use_mock"]


def _client() -> WandbInferenceClient:
    global _CLIENT
    if _CLIENT is None:
        with _LOCK:
            if _CLIENT is None:
                _CLIENT = WandbInferenceClient(provider=_CONFIG["provider"])
    return _CLIENT


_EDIT_SYSTEM = (
    "You are a precise document editor for accounting ledgers and chess PGN "
    "files. Apply EXACTLY the single instruction to the document. Output ONLY "
    "the resulting file content as plain text — no prose, no explanation, no "
    "code fences. Preserve every record, transaction, move, and value the "
    "instruction does not explicitly change."
)


def apply_edit(current_doc: str, instruction: str) -> str:
    if _CONFIG["use_mock"]:
        return _mock_apply_edit(current_doc, instruction)
    user = (f"DOCUMENT:\n{current_doc}\n\nINSTRUCTION: {instruction}\n\n"
            "Resulting file content:")
    out = _client().chat(_CONFIG["model"], _EDIT_SYSTEM, user,
                         temperature=0.0, max_tokens=_CONFIG["max_tokens"])
    return _strip_fences(out) or current_doc


_REPAIR_SYSTEM = (
    "You repair a document after a delegated round-trip edit corrupted it. You "
    "are given the original seed document, the current (possibly corrupted) "
    "document, the edit instruction whose result should be preserved, and a "
    "checksum report of structural problems. Restore ONLY the missing or "
    "corrupted transactions / moves / records / values the report flags; keep "
    "the legitimate transformation. Do NOT just echo the seed. Output ONLY the "
    "repaired file content as plain text — no prose, no fences."
)


def repair_doc(seed_doc: str, current_doc: str, instruction: str,
               repair_view: Dict) -> str:
    if _CONFIG["use_mock"]:
        return _mock_repair_doc(seed_doc, current_doc, repair_view)
    import json
    user = (
        f"SEED (the correct round-trip target):\n{seed_doc}\n\n"
        f"CURRENT (corrupted):\n{current_doc}\n\n"
        f"EDIT INSTRUCTION (preserve its intent):\n{instruction}\n\n"
        f"CHECKSUM REPORT:\n{json.dumps(repair_view)}\n\n"
        "Repaired file content:")
    out = _client().chat(_CONFIG["model"], _REPAIR_SYSTEM, user,
                         temperature=0.0, max_tokens=_CONFIG["max_tokens"])
    return _strip_fences(out) or current_doc


# --------------------------------------------------------------------------- #
def _strip_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = s[3:]
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1:]
        end = s.rfind("```")
        if end != -1:
            s = s[:end]
    return s.strip()


# --- deterministic mock (no key): occasionally drop a non-empty line -------- #
def _mock_apply_edit(current_doc: str, instruction: str) -> str:
    h = hashlib.md5((instruction + current_doc[:200]).encode()).hexdigest()
    rng = random.Random(int(h[:12], 16))
    lines = current_doc.splitlines()
    if rng.random() < _CONFIG["slip_p"] and len(lines) > 4:
        idx = rng.randrange(len(lines))
        if lines[idx].strip():
            lines.pop(idx)
    return "\n".join(lines)


def _mock_repair_doc(seed_doc: str, current_doc: str, repair_view: Dict) -> str:
    # the mock repairer is competent: it restores the seed structure.
    return seed_doc
