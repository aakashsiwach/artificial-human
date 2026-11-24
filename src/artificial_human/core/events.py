from __future__ import annotations

from collections.abc import Callable
from collections import defaultdict
from threading import RLock
from typing import Any


EventHandler = Callable[[Any], None]


class EventBus:
    """Small in-process event bus; handlers run synchronously in registration order."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._lock = RLock()

    def subscribe(self, event_name: str, handler: EventHandler) -> None:
        with self._lock:
            self._handlers[event_name].append(handler)

    def unsubscribe(self, event_name: str, handler: EventHandler) -> None:
        with self._lock:
            handlers = self._handlers.get(event_name, [])
            if handler in handlers:
                handlers.remove(handler)

    def publish(self, event_name: str, payload: Any) -> int:
        with self._lock:
            handlers = tuple(self._handlers.get(event_name, ()))
        for handler in handlers:
            handler(payload)
        return len(handlers)
