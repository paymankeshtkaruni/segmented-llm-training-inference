"""
Operation tracer — optional, ZERO-overhead by default, measurement-only.

The forward/backward engines call `mark("label")` at each distinct operation (an
embedding-cat, a residual add, a norm, a CE vocab-slice's math, an autograd.grad, ...)
so a cost profiler can see the memory of EVERY move inside forward / backward /
validation — not just a per-phase total.

By default `_HOOK is None`, so `mark()` is a single `is None` check and a return: it
does NOT change any computation, so the segmentation_management identity tests are
unaffected (re-run them after editing the engines to confirm). A profiler installs a
hook via `set_hook(fn)`; then every `mark(label)` calls `fn(label)`, and the hook
(not this module) reads the memory counters — keeping this module dependency-free.
"""

from __future__ import annotations

from typing import Callable, Optional

_HOOK: Optional[Callable[[str], None]] = None


def set_hook(fn: Optional[Callable[[str], None]]) -> None:
    """Install (or clear, with None) the per-operation trace hook."""
    global _HOOK
    _HOOK = fn


def mark(label: str) -> None:
    """Record that operation `label` just completed (no-op unless a hook is set)."""
    if _HOOK is not None:
        _HOOK(label)
