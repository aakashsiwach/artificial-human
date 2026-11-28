"""Runtime memory-store contract, independent of any persistence implementation."""
from __future__ import annotations

from typing import Any, Protocol


class MemoryStore(Protocol):
    """Structural interface used by the active organism's volatile memory store.

    The protocol intentionally describes behavior rather than inheritance from
    the legacy SQLite repository. Implementations may be process-local or test
    doubles; the active organism wires the bounded in-memory implementation.
    """

    def __getattr__(self, name: str) -> Any: ...
