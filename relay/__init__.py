"""Relay — a multi-agent handoff-degradation harness.

Substrate-agnostic: a chain of agents relays a piece of state hop-to-hop; a
Conductor reads a key-free degradation *risk* per hop and selectively
re-grounds. We compare four conditions over the same episodes and log
everything to Weave + a local leaderboard.
"""

__version__ = "0.1.0"
