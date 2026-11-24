from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from queue import PriorityQueue
from threading import Lock
from uuid import uuid4


class Priority(IntEnum):
    CRITICAL = 0
    HIGH = 10
    NORMAL = 20
    LOW = 30


@dataclass(order=True, frozen=True, slots=True)
class WorkItem:
    priority: int
    sequence: int
    description: str = field(compare=False)
    id: str = field(default_factory=lambda: str(uuid4()), compare=False)


class WorkScheduler:
    """Bounded in-memory priority queue for internal work descriptions."""

    def __init__(self, capacity: int = 100) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._queue: PriorityQueue[WorkItem] = PriorityQueue(maxsize=capacity)
        self._sequence = 0
        self._lock = Lock()

    def submit(self, description: str, priority: Priority = Priority.NORMAL) -> WorkItem:
        if not description.strip():
            raise ValueError("Work description is required")
        with self._lock:
            self._sequence += 1
            item = WorkItem(int(priority), self._sequence, description)
            self._queue.put_nowait(item)
            return item

    def next(self) -> WorkItem | None:
        if self._queue.empty():
            return None
        return self._queue.get_nowait()

    def depth(self) -> int:
        return self._queue.qsize()
