from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from threading import RLock


@dataclass(frozen=True, slots=True)
class WorkingMemoryEntry:
    id: str
    content: str
    kind: str
    source: str
    salience: float
    created_at: str
    access_count: int = 0


class WorkingMemory:
    """A bounded, volatile workspace inspired by human working memory.

    Items decay slightly when new material enters. Capacity is deliberately
    bounded; this is a useful software approximation, not a literal model of
    the brain's debated working-memory capacity.
    """

    def __init__(self, capacity: int = 7) -> None:
        if capacity < 1:
            raise ValueError("Working-memory capacity must be positive")
        self.capacity = capacity
        self._items: OrderedDict[str, WorkingMemoryEntry] = OrderedDict()
        self._lock = RLock()

    def remember(self, entry: WorkingMemoryEntry) -> None:
        if not entry.id.strip() or not entry.content.strip():
            raise ValueError("Working-memory entries require an id and content")
        if not 0.0 <= entry.salience <= 1.0:
            raise ValueError("Working-memory salience must be between 0 and 1")
        with self._lock:
            self._items = OrderedDict(
                (
                    item_id,
                    replace(item, salience=round(item.salience * 0.92, 4)),
                )
                for item_id, item in self._items.items()
            )
            previous = self._items.pop(entry.id, None)
            if previous is not None:
                entry = replace(entry, access_count=previous.access_count + 1)
            elif entry.access_count < 0:
                raise ValueError("Working-memory access_count cannot be negative")
            self._items[entry.id] = entry
            while len(self._items) > self.capacity:
                least_salient_id = min(
                    self._items,
                    key=lambda item_id: (
                        self._items[item_id].salience,
                        self._items[item_id].created_at,
                    ),
                )
                del self._items[least_salient_id]

    def items(self) -> tuple[WorkingMemoryEntry, ...]:
        with self._lock:
            return tuple(self._items.values())

    def context(self, *, limit: int = 5) -> tuple[WorkingMemoryEntry, ...]:
        if limit < 1:
            raise ValueError("Working-memory context limit must be positive")
        with self._lock:
            return tuple(list(self._items.values())[-limit:])

    def public_state(self) -> dict[str, object]:
        with self._lock:
            items = list(self._items.values())
        focus = max(items, key=lambda item: item.salience) if items else None
        return {
            "capacity": self.capacity,
            "item_count": len(items),
            "focus_id": focus.id if focus else None,
            "items": [
                {
                    "id": item.id,
                    "kind": item.kind,
                    "source": item.source,
                    "salience": item.salience,
                    "access_count": item.access_count,
                }
                for item in items
            ],
        }


class WorkingMemorySessions:
    """Owner-scoped, process-local working-memory sessions with an LRU bound."""

    def __init__(self, *, capacity: int = 7, max_sessions: int = 128) -> None:
        if max_sessions < 1:
            raise ValueError("Maximum working-memory sessions must be positive")
        self.capacity = capacity
        self.max_sessions = max_sessions
        self._sessions: OrderedDict[tuple[str, str], WorkingMemory] = OrderedDict()
        self._lock = RLock()

    def get(self, owner_id: str, session_id: str) -> WorkingMemory:
        if not owner_id.strip() or not session_id.strip():
            raise ValueError("Owner and session identifiers are required")
        if len(session_id) > 128:
            raise ValueError("Session identifier cannot exceed 128 characters")
        key = (owner_id, session_id)
        with self._lock:
            workspace = self._sessions.pop(key, None)
            if workspace is None:
                workspace = WorkingMemory(capacity=self.capacity)
            self._sessions[key] = workspace
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)
            return workspace

    def clear(self, owner_id: str, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop((owner_id, session_id), None) is not None

    def clear_owner(self, owner_id: str) -> int:
        with self._lock:
            keys = [key for key in self._sessions if key[0] == owner_id]
            for key in keys:
                del self._sessions[key]
            return len(keys)


def new_working_memory_entry(
    *,
    id: str,
    content: str,
    kind: str,
    source: str,
    salience: float,
) -> WorkingMemoryEntry:
    return WorkingMemoryEntry(
        id=id,
        content=content,
        kind=kind,
        source=source,
        salience=salience,
        created_at=datetime.now(UTC).isoformat(),
    )
