"""Storage-neutral base for active and legacy memory-store adapters."""
from __future__ import annotations

from typing import Any


class MemoryStore:
    """Persistence-independent memory-store base.

    The concrete stores own their behavior. The fallback keeps older optional
    repository capabilities structurally available while callers migrate to
    explicit methods; it performs no persistence or database initialization.
    """

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(f"{type(self).__name__!s} has no memory-store capability {name!r}")
