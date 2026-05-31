"""Substrate registry.

Lazy imports on purpose: selecting ``mock`` must never import ``openai`` or any
network client, so the no-key path stays pure-stdlib.
"""

from __future__ import annotations


def get_substrate(name: str, **kwargs):
    if name == "mock":
        from .mock import MockSubstrate
        return MockSubstrate(**kwargs)
    if name == "roundtrip":
        from .roundtrip import RoundTripSubstrate
        return RoundTripSubstrate(**kwargs)
    if name == "mcq":
        from .mcq import McqSubstrate
        return McqSubstrate(**kwargs)
    raise ValueError(f"unknown substrate: {name!r} (choose mock/roundtrip/mcq)")
