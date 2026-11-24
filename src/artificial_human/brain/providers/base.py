from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ProviderNotConfiguredError(RuntimeError):
    """Raised when no reasoning model has been configured."""


class ProviderRequestError(RuntimeError):
    """Raised when a configured model endpoint cannot complete a request."""


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class Generation:
    content: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ChatProvider(Protocol):
    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> Generation:
        """Generate one response from an ordered chat-message sequence."""
