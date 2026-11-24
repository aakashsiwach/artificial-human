from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import UTC, datetime


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

    def remember(self, entry: WorkingMemoryEntry) -> None:
        if not entry.id.strip() or not entry.content.strip():
            raise ValueError("Working-memory entries require an id and content")
        if not 0.0 <= entry.salience <= 1.0:
            raise ValueError("Working-memory salience must be between 0 and 1")
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
                    list(self._items).index(item_id),
                ),
            )
            del self._items[least_salient_id]

    def items(self) -> tuple[WorkingMemoryEntry, ...]:
        return tuple(self._items.values())

    def context(self, *, limit: int = 5) -> tuple[WorkingMemoryEntry, ...]:
        if limit < 1:
            raise ValueError("Working-memory context limit must be positive")
        return tuple(list(self._items.values())[-limit:])

    def public_state(self) -> dict[str, object]:
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
