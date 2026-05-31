"""Relay — a multi-agent handoff-degradation harness.

A workflow relays a structured document through reversible round-trip edits; a
Conductor reads a key-free degradation *risk* after each edit and selectively
performs targeted repair. We compare four conditions over the same tasks and log
everything to Weave + a local leaderboard. See ``run_all_conditions.py``.
"""

__version__ = "0.1.0"
