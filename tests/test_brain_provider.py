import json

import httpx
import pytest

from artificial_human.brain.providers.base import (
    ChatMessage,
    ProviderNotConfiguredError,
    ProviderRequestError,
)
from artificial_human.brain.providers.openai_compatible import (
    OpenAICompatibleProvider,
    ProviderSettings,
)


def test_openai_compatible_provider_sends_chat_and_parses_result() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "local-model",
                "choices": [{"message": {"content": "A considered answer."}}],
                "usage": {"prompt_tokens": 18, "completion_tokens": 7},
            },
        )

    provider = OpenAICompatibleProvider(
        ProviderSettings(
            base_url="http://localhost:11434/v1",
            model="local-model",
            api_key="",
        ),
        transport=httpx.MockTransport(handler),
    )
    try:
        result = provider.generate(
            [ChatMessage(role="user", content="Hello")],
            temperature=0.3,
        )
    finally:
        provider.close()

    assert captured["url"] == "http://localhost:11434/v1/chat/completions"
    assert captured["authorization"] is None
    assert captured["body"]["model"] == "local-model"
    assert result.content == "A considered answer."
    assert result.prompt_tokens == 18
    assert result.completion_tokens == 7


def test_provider_requires_a_model_name() -> None:
    provider = OpenAICompatibleProvider(ProviderSettings(model=""))
    try:
        with pytest.raises(ProviderNotConfiguredError):
            provider.generate([ChatMessage(role="user", content="Hello")])
    finally:
        provider.close()


def test_provider_does_not_expose_api_key_in_status() -> None:
    provider = OpenAICompatibleProvider(
        ProviderSettings(model="model", api_key="do-not-leak")
    )
    try:
        status = provider.status()
        assert status["authentication_configured"] is True
        assert "do-not-leak" not in str(status)
    finally:
        provider.close()


def test_provider_maps_http_error_without_returning_response_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="secret detail must not escape")

    provider = OpenAICompatibleProvider(
        ProviderSettings(model="model", api_key="private"),
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(ProviderRequestError) as caught:
            provider.generate([ChatMessage(role="user", content="Hello")])
        assert "secret detail" not in str(caught.value)
    finally:
        provider.close()
