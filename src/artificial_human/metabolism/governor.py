from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from threading import RLock


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BudgetLimits:
    operations: int = 100
    tokens: int = 20_000
    wall_seconds: float = 60.0


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    limits: BudgetLimits
    operations_used: int
    tokens_used: int
    elapsed_seconds: float

    def as_dict(self) -> dict[str, object]:
        return {
            "limits": asdict(self.limits),
            "operations_used": self.operations_used,
            "tokens_used": self.tokens_used,
            "elapsed_seconds": self.elapsed_seconds,
        }


class ResourceGovernor:
    """Per-run declared-work budget; it is not an OS resource monitor."""

    def __init__(self, limits: BudgetLimits | None = None) -> None:
        self.limits = limits or BudgetLimits()
        self._started = time.monotonic()
        self._operations = 0
        self._tokens = 0
        self._lock = RLock()

    def consume(self, *, operations: int = 1, tokens: int = 0) -> BudgetSnapshot:
        if operations < 0 or tokens < 0:
            raise ValueError("Resource consumption cannot be negative")
        with self._lock:
            elapsed = time.monotonic() - self._started
            if elapsed > self.limits.wall_seconds:
                raise BudgetExceeded("Run time budget exceeded")
            if self._operations + operations > self.limits.operations:
                raise BudgetExceeded("Operation budget exceeded")
            if self._tokens + tokens > self.limits.tokens:
                raise BudgetExceeded("Token budget exceeded")
            self._operations += operations
            self._tokens += tokens
            return self.snapshot()

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return BudgetSnapshot(
                limits=self.limits,
                operations_used=self._operations,
                tokens_used=self._tokens,
                elapsed_seconds=round(time.monotonic() - self._started, 4),
            )
