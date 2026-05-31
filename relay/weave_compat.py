"""Weave compatibility layer.

Relay must run GREEN with NO API key — and even with `weave` not installed at
all (mock mode is pure-stdlib). This module exposes a `weave`-like `op`
decorator plus init helpers so the rest of the code can decorate with
``@op()`` unconditionally.

- If ``weave`` is importable, we use the real package.
- Otherwise we install a transparent no-op shim, so ``@op()`` is a pass-through.

``maybe_init()`` only attempts ``weave.init()`` when WANDB_API_KEY + WANDB_PROJECT
are both set, so the no-key path never blocks, prompts, or raises.
"""

from __future__ import annotations

import os

try:  # real weave if available
    import weave as _weave  # type: ignore

    _HAS_WEAVE = True
except Exception:  # pragma: no cover - depends on env
    _weave = None
    _HAS_WEAVE = False

_INITIALIZED = False


def _shim_op(*dargs, **dkwargs):
    """Mimic ``weave.op`` used either as ``@op`` or ``@op(...)``."""
    # Used directly as @op (no parens): a single callable positional arg.
    if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
        return dargs[0]

    # Used as @op(...) -> return a decorator.
    def deco(fn):
        return fn

    return deco


def op(*args, **kwargs):
    """Decorator: trace with Weave when available, else a no-op."""
    if _HAS_WEAVE:
        return _weave.op(*args, **kwargs)
    return _shim_op(*args, **kwargs)


def maybe_init(project: str | None = None, verbose: bool = True) -> bool:
    """Initialize Weave iff we have credentials. Never raises.

    Returns True only when Weave tracing is actually live.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return True

    project = project or os.environ.get("WANDB_PROJECT")
    api_key = os.environ.get("WANDB_API_KEY")

    if not _HAS_WEAVE:
        if verbose:
            print("[weave] not installed -> tracing disabled (mock still runs).")
        return False
    if not (project and api_key):
        if verbose:
            print("[weave] WANDB_API_KEY / WANDB_PROJECT not set -> tracing disabled.")
        return False
    try:
        _weave.init(project)
        _INITIALIZED = True
        if verbose:
            print(f"[weave] initialized on project '{project}'.")
        return True
    except Exception as e:  # pragma: no cover - network/creds dependent
        if verbose:
            print(f"[weave] init failed ({e}) -> tracing disabled.")
        return False


def is_active() -> bool:
    return _INITIALIZED


def has_weave() -> bool:
    return _HAS_WEAVE


def weave_module():
    """Return the real ``weave`` module, or None when unavailable."""
    return _weave if _HAS_WEAVE else None
