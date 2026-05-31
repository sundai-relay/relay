"""llm_cache.py -- tiny stdlib on-disk cache for deterministic (temp=0) LLM calls.

A live run re-issues many *identical* prompts: every condition's first forward
edit on a given seed is the same, and naive/always repeat the same edit
sequence. At ``temperature == 0`` those completions are deterministic, so we can
replay them from a local sqlite cache instead of re-calling the API. That means
a relaunch after a crash/kill does not forfeit the calls a prior run already
paid for -- the overlapping calls come back from disk.

Only ``temperature == 0.0`` calls are cached (anything hotter is
non-deterministic and must not be replayed). The store is a single sqlite file;
set ``RELAY_NO_CACHE=1`` to disable it, ``RELAY_CACHE_DIR`` to relocate it. It
is pure stdlib, so importing it never affects the no-key mock path.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import sqlite3
import threading
from typing import Optional

_CACHE_DIR = os.environ.get("RELAY_CACHE_DIR", ".cache/relay")
_DB_PATH = os.path.join(_CACHE_DIR, "llm_cache.sqlite")
_DISABLED = os.environ.get("RELAY_NO_CACHE", "").strip().lower() in ("1", "true", "yes")

_conn: Optional[sqlite3.Connection] = None
_lock = threading.Lock()
_stats = {"hits": 0, "misses": 0, "stored": 0}
_announced = False


def _connect() -> Optional[sqlite3.Connection]:
    """Open (once) the sqlite store and confirm the cache is active."""
    global _conn, _announced
    if _DISABLED:
        return None
    if _conn is None:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        _conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")
        _conn.commit()
        if not _announced:
            print(f"[cache] disk cache ACTIVE at {_DB_PATH} "
                  "(replaying temperature==0 calls only)")
            _announced = True
    return _conn


def cacheable(temperature) -> bool:
    """We only cache deterministic calls: enabled AND temperature exactly 0.0."""
    try:
        return (not _DISABLED) and float(temperature) == 0.0
    except (TypeError, ValueError):
        return False


def key_for(model, system, user, temperature, max_tokens, **kwargs) -> str:
    payload = json.dumps(
        {"model": model, "system": system, "user": user,
         "temperature": temperature, "max_tokens": max_tokens,
         "kwargs": {k: kwargs[k] for k in sorted(kwargs)}},
        sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get(key: str) -> Optional[str]:
    conn = _connect()
    if conn is None:
        return None
    with _lock:
        row = conn.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
    if row is None:
        _stats["misses"] += 1
        return None
    _stats["hits"] += 1
    return row[0]


def put(key: str, value: str) -> None:
    conn = _connect()
    if conn is None:
        return
    with _lock:
        conn.execute("INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
                     (key, value))
        conn.commit()
    _stats["stored"] += 1


def stats() -> dict:
    return dict(_stats)


@atexit.register
def _summary() -> None:
    """Confirm what the cache actually did this run (only if it was exercised)."""
    if _stats["hits"] or _stats["misses"]:
        print(f"[cache] temp=0 disk cache: {_stats['hits']} hit(s), "
              f"{_stats['misses']} miss(es), {_stats['stored']} stored "
              f"-> {_stats['hits']} API call(s) replayed from {_DB_PATH}")
