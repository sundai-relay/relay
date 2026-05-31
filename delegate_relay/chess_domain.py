"""Chess (PGN) domain for the Relay harness — the chess analog of ledger.py.

The *content* of a chess game is its MOVE SEQUENCE; the annotations ({ [%eval] }, ?!/?? move
symbols, phase headers, time stamps) are surface form. So we parse and score on the move
sequence (robust to the annotation-format round-trips that dominate this domain), plus the key
headers and result. A faithful round-trip returns the same moves, so score(D0,D0)=1.0 and any
drop is real corruption — exactly like the ledger's transactions.

Same public interface as ledger.py so domain.py can dispatch to either:
  parse_doc, invariants, invariant_deviation, parse_health, score, diff_blocks,
  editor_system_prompt, repair_system_prompt, repair_source_label, mock_edit, mock_reground
"""
from __future__ import annotations

import difflib
import hashlib
import random
import re

_TAG = re.compile(r'\[(\w+)\s+"([^"]*)"\]')
# core SAN: castling, or [piece][disambig][capture][dest][promo], trailing !?+# stripped
_MOVE = re.compile(r"^(O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?)[+#!?]*$")
_KEY_HEADERS = ("White", "Black", "Result", "ECO")


def _strip_body(text: str) -> str:
    body = re.sub(r"^\s*\[[^\]]*\]\s*$", "", text, flags=re.M)   # drop header lines
    body = re.sub(r"\{[^}]*\}", " ", body)                        # { comments / evals }
    body = re.sub(r"\$\d+", " ", body)                            # NAGs
    body = re.sub(r"\b(1-0|0-1|1/2-1/2|\*)", " ", body)           # results
    return body


def _moves(text: str) -> list[str]:
    out: list[str] = []
    for tok in _strip_body(text).split():
        tok = re.sub(r"^\d+\.(\.\.)?", "", tok.strip())   # strip attached move number
        if not tok:
            continue
        m = _MOVE.match(tok)
        if m:
            out.append(m.group(1))
    return out


def parse_doc(text: str) -> dict:
    return {"headers": dict(_TAG.findall(text)), "moves": _moves(text)}


parse_pgn = parse_doc  # alias


def invariants(parsed: dict) -> dict:
    h, mv = parsed["headers"], parsed["moves"]
    return {
        "n_moves": len(mv),
        "moves": tuple(mv),
        "result": h.get("Result", ""),
        "key_headers": tuple((k, h.get(k, "")) for k in _KEY_HEADERS),
    }


def _seq_ratio(a: list, b: list) -> float:
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def invariant_deviation(cur: dict, ref: dict) -> float:
    """Continuous key-free corruption feature in [0,1]: move-sequence displacement (+headers)."""
    move_dev = 1.0 - _seq_ratio(cur["moves"], ref["moves"])
    hdr_dev = 1.0 - sum(1 for k in _KEY_HEADERS
                        if cur["headers"].get(k) == ref["headers"].get(k)) / len(_KEY_HEADERS)
    return float(0.8 * move_dev + 0.2 * hdr_dev)


def parse_health(text: str) -> float:
    return 1.0 if parse_doc(text)["moves"] else 0.0


def score(final_text: str, seed_text: str) -> float:
    """Round-trip fidelity in [0,1]: 0.7*move-sequence match + 0.3*key-header match."""
    f, r = parse_doc(final_text), parse_doc(seed_text)
    if not r["moves"]:
        return 0.0
    move = _seq_ratio(f["moves"], r["moves"])
    hdr = sum(1 for k in _KEY_HEADERS if f["headers"].get(k) == r["headers"].get(k)) / len(_KEY_HEADERS)
    return round(0.7 * move + 0.3 * hdr, 4)


def _clean_movetext(text: str) -> str:
    mv = _moves(text)
    parts = []
    for i, m in enumerate(mv):
        parts.append(f"{i // 2 + 1}. {m}" if i % 2 == 0 else m)
    return " ".join(parts)


def diff_blocks(cur_text: str, seed_text: str) -> list[str]:
    """Re-grounding source: if the move sequence deviates from D0, hand back the authoritative
    movetext for the Conductor to restore. (Chess's analog of the ledger's missing txn blocks.)"""
    if parse_doc(cur_text)["moves"] == parse_doc(seed_text)["moves"]:
        return []
    return [_clean_movetext(seed_text)]


# ---------------------------------------------------------- offline mock ----
def _headers_text(doc_text: str) -> str:
    return "\n".join(l for l in doc_text.splitlines() if _TAG.match(l.strip()))


def mock_edit(doc_text: str, instruction: str) -> str:
    """Simulate a lossy edit: drop a per-instruction fraction (5-25%) of moves. Deterministic."""
    moves = _moves(doc_text)
    if not moves:
        return doc_text
    rng = random.Random(int(hashlib.sha256((instruction[:60] + str(len(moves))).encode()).hexdigest()[:12], 16))
    drop = 0.05 + 0.20 * (int(hashlib.sha256(instruction.encode()).hexdigest()[:4], 16) / 0xFFFF)
    kept = [m for m in moves if rng.random() > drop]
    parts = [f"{i // 2 + 1}. {m}" if i % 2 == 0 else m for i, m in enumerate(kept)]
    return _headers_text(doc_text) + "\n\n" + " ".join(parts)


def mock_reground(doc_text: str, missing_blocks: list[str]) -> str:
    """Faithful local repair: restore the authoritative movetext the Conductor handed back."""
    movetext = "\n\n".join(missing_blocks) if missing_blocks else _clean_movetext(doc_text)
    return _headers_text(doc_text) + "\n\n" + movetext


# ---------------------------------------------------------------- prompts ----
def editor_system_prompt() -> str:
    return ("You edit chess game files (PGN) on a user's behalf. Apply the instruction exactly "
            "and completely. Preserve EVERY move, every header, and the result, in order — only "
            "change annotation format if the instruction asks. Do NOT drop or reorder moves. "
            "Output ONLY the resulting PGN file(s), no commentary.")


def repair_system_prompt() -> str:
    return ("You repair a chess PGN that lost or altered moves during automated editing. You are "
            "given the AUTHORITATIVE move sequence (ground truth). Restore the game so every move "
            "appears exactly once, in the correct order, with the original headers and result. "
            "Output ONLY the corrected PGN, no commentary.")


def repair_source_label() -> str:
    return "AUTHORITATIVE PGN MOVES (restore exactly, in order)"


if __name__ == "__main__":
    import json

    rows = [json.loads(l) for l in open("../delegate/delegate52.jsonl")]
    chess = [r for r in rows if r["sample_type"] == "chess"]
    print(f"{'sample':22}{'moves':>7}{'fid(D0,D0)':>12}{'health':>8}")
    for r in chess:
        key = next(k for k in r["files"] if k.startswith(r["metadata"]["start_state"] + "/"))
        t = r["files"][key]
        p = parse_doc(t)
        print(f"{r['sample_id']:22}{len(p['moves']):>7}{score(t, t):>12}{parse_health(t):>8}")
    # corruption + repair sanity on sample 0
    t = chess[0]["files"][next(k for k in chess[0]["files"] if "basic_state/" in k)]
    corrupt = mock_edit(t, "convert eval to symbols")
    miss = diff_blocks(corrupt, t)
    repaired = mock_reground(corrupt, miss)
    print(f"\nmock corruption: fid {score(t, t):.3f} -> {score(corrupt, t):.3f} "
          f"-> repaired {score(repaired, t):.3f}  (dev {invariant_deviation(parse_doc(corrupt), parse_doc(t)):.3f})")
