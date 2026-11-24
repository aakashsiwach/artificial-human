from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from artificial_human.brain.providers.base import (
    ChatMessage,
    Generation,
    ProviderNotConfiguredError,
    ProviderRequestError,
)


@dataclass(frozen=True, slots=True)
class ProviderSettings:
    base_url: str = "http://127.0.0.1:11434/v1"
    model: str = ""
    api_key: str = ""
    timeout_seconds: float = 45.0
    max_output_tokens: int = 1200

    def __post_init__(self) -> None:
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("Provider base_url must be an absolute HTTP or HTTPS URL.")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Provider base_url cannot contain credentials, query, or fragment.")
        local_hosts = {"localhost", "127.0.0.1", "::1", "host.docker.internal"}
        if parsed.scheme == "http" and parsed.hostname not in local_hosts:
            raise ValueError("Remote reasoning providers must use HTTPS.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")

    @classmethod
    def from_env(cls) -> ProviderSettings:
        return cls(
            base_url=(
                os.getenv("ARTIFICIAL_HUMAN_LLM_BASE_URL", "").strip()
                or "http://127.0.0.1:11434/v1"
            ).rstrip("/"),
            model=os.getenv("ARTIFICIAL_HUMAN_LLM_MODEL", "").strip(),
            api_key=os.getenv("ARTIFICIAL_HUMAN_LLM_API_KEY", ""),
            timeout_seconds=float(
                os.getenv("ARTIFICIAL_HUMAN_LLM_TIMEOUT_SECONDS", "45")
            ),
            max_output_tokens=int(
                os.getenv("ARTIFICIAL_HUMAN_LLM_MAX_OUTPUT_TOKENS", "1200")
            ),
        )

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.model)

    def public_status(self) -> dict[str, object]:
        return {
            "provider": "openai-compatible",
            "configured": self.configured,
            "base_url": self.base_url,
            "model": self.model or None,
            "authentication_configured": bool(self.api_key),
            "timeout_seconds": self.timeout_seconds,
            "max_output_tokens": self.max_output_tokens,
        }


class OpenAICompatibleProvider:
    """Chat-completions adapter for OpenAI-compatible hosted or local endpoints."""

    def __init__(
        self,
        settings: ProviderSettings | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.settings = settings or ProviderSettings.from_env()
        self._client = httpx.Client(
            timeout=self.settings.timeout_seconds,
            transport=transport,
        )

    def status(self) -> dict[str, object]:
        return self.settings.public_status()

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> Generation:
        if not self.settings.configured:
            raise ProviderNotConfiguredError(
                "No reasoning model is configured. Set ARTIFICIAL_HUMAN_LLM_MODEL."
            )
        if not messages:
            raise ValueError("At least one chat message is required")
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")

        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"

        payload = {
            "model": self.settings.model,
            "messages": [
                {"role": item.role, "content": item.content} for item in messages
            ],
            "temperature": temperature,
            "max_tokens": max_tokens or self.settings.max_output_tokens,
        }
        try:
            response = self._client.post(
                f"{self.settings.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
        except httpx.TimeoutException as error:
            raise ProviderRequestError("The reasoning provider timed out.") from error
        except httpx.RequestError as error:
            raise ProviderRequestError("Could not reach the configured reasoning provider.") from error

        if response.is_error:
            raise ProviderRequestError(
                f"The reasoning provider returned HTTP {response.status_code}."
            )
        try:
            result = response.json()
            choice = result["choices"][0]
            content = choice["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise ProviderRequestError(
                "The reasoning provider returned an invalid chat-completion response."
            ) from error
        if not isinstance(content, str) or not content.strip():
            raise ProviderRequestError("The reasoning provider returned an empty response.")

        usage = result.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        return Generation(
            content=content.strip(),
            model=str(result.get("model") or self.settings.model),
            prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
            completion_tokens=(
                completion_tokens if isinstance(completion_tokens, int) else None
            ),
        )

    def close(self) -> None:
        self._client.close()
