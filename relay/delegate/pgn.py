"""Chess domain: a `.pgn` parser + key-free invariants + scorer (python-chess).

Recipe (relay_delegate52_recipes.md §7):
  invariants: moves are legal from the start position; #moves; final FEN.
              An illegal move = unambiguous corruption (no embedding hand-waving).
  score     : sequence-match on the move list + final-position match + a
              legal/parse bonus -> [0, 1].

We compare on the annotation-free UCI move list (python-chess derives it by
replaying the SAN), so eval comments / `?!` symbols / header rewrites never
register as corruption — only a changed, dropped, or illegal move does.
"""
from __future__ import annotations

import io
from typing import Dict, List

import chess
import chess.pgn

name = "chess"


def parse(text: str) -> Dict:
    """Parse PGN -> {parse_ok, legal, moves:[uci], n_moves, final_fen, headers}.

    `legal` is False if python-chess hit an illegal/unreadable move while
    replaying the mainline (the decisive corruption signal for chess).
    """
    out = {"parse_ok": False, "legal": False, "moves": [], "n_moves": 0,
           "final_fen": "", "headers": {}}
    if not isinstance(text, str) or not text.strip():
        return out
    try:
        game = chess.pgn.read_game(io.StringIO(text))
    except Exception:
        return out
    if game is None:
        return out
    out["parse_ok"] = True
    out["headers"] = {k: v for k, v in game.headers.items()}
    board = game.board()
    moves: List[str] = []
    legal = True
    try:
        for mv in game.mainline_moves():
            if mv not in board.legal_moves:
                legal = False
                break
            moves.append(mv.uci())
            board.push(mv)
    except Exception:
        legal = False
    out["legal"] = legal and len(moves) > 0
    out["moves"] = moves
    out["n_moves"] = len(moves)
    out["final_fen"] = board.fen() if moves else ""
    return out


def _seq_match(a: List[str], b: List[str]) -> float:
    """Fraction of positions where the two move lists agree, normalized by the
    longer list (so both drops and substitutions cost)."""
    if not a and not b:
        return 1.0
    same = sum(1 for x, y in zip(a, b) if x == y)
    return same / max(len(a), len(b), 1)


def runtime_risk(seed: Dict, cur: Dict) -> Dict:
    """Key-free corruption signal: current game vs SEED game."""
    if not cur["parse_ok"] or cur["n_moves"] == 0:
        return {"risk": 1.0, "parse_fail": 1.0, "illegal": 1.0,
                "move_mismatch": 1.0, "fen_mismatch": 1.0,
                "move_count_drift": 1.0, "n_seed": seed["n_moves"], "n_cur": 0}
    illegal = 0.0 if cur["legal"] else 1.0
    move_mismatch = 1.0 - _seq_match(seed["moves"], cur["moves"])
    fen_mismatch = 0.0 if cur["final_fen"] == seed["final_fen"] else 1.0
    count_drift = abs(seed["n_moves"] - cur["n_moves"]) / max(1, seed["n_moves"])
    count_drift = min(1.0, count_drift)
    risk = (0.40 * illegal + 0.30 * move_mismatch + 0.20 * fen_mismatch
            + 0.10 * count_drift)
    return {
        "risk": round(max(0.0, min(1.0, risk)), 4),
        "parse_fail": 0.0,
        "illegal": illegal,
        "move_mismatch": round(move_mismatch, 4),
        "fen_mismatch": fen_mismatch,
        "move_count_drift": round(count_drift, 4),
        "n_seed": seed["n_moves"], "n_cur": cur["n_moves"],
    }


def score(seed: Dict, cur: Dict) -> Dict:
    """Gold metric: 0.5*move-seq-match + 0.3*final-FEN-match + 0.2*legal&parse."""
    if not cur["parse_ok"]:
        return {"score": 0.0, "move_match": 0.0, "fen_match": 0.0, "legal": 0.0}
    move_match = _seq_match(seed["moves"], cur["moves"])
    fen_match = 1.0 if (cur["final_fen"] and cur["final_fen"] == seed["final_fen"]) else 0.0
    legal = 1.0 if cur["legal"] else 0.0
    s = 0.5 * move_match + 0.3 * fen_match + 0.2 * legal
    return {"score": round(s, 4), "move_match": round(move_match, 4),
            "fen_match": fen_match, "legal": legal}


def repair_view(report: Dict) -> Dict:
    return {"illegal_or_unparseable": bool(report.get("illegal")
                                           or report.get("parse_fail")),
            "seed_move_count": report.get("n_seed"),
            "current_move_count": report.get("n_cur"),
            "final_position_changed": bool(report.get("fen_mismatch"))}
