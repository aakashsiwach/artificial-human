import json

import httpx

from artificial_human.brain.providers.base import ChatMessage
from artificial_human.brain.providers.openai_compatible import (
    OpenAICompatibleProvider,
    ProviderSettings,
)


def test_remote_provider_requires_its_api_key(monkeypatch) -> None:
    monkeypatch.setenv(
        "ARTIFICIAL_HUMAN_LLM_BASE_URL", "https://api.groq.com/openai/v1"
    )
    monkeypatch.setenv("ARTIFICIAL_HUMAN_LLM_MODEL", "openai/gpt-oss-120b")
    monkeypatch.delenv("ARTIFICIAL_HUMAN_LLM_API_KEY", raising=False)

    settings = ProviderSettings.from_env()

    assert settings.configured is False
    status = settings.public_status()
    assert status["model"] == "openai/gpt-oss-120b"
    assert status["authentication_configured"] is False
    assert "api_key" not in status


def test_groq_gpt_oss_request_uses_reasoning_parameters_and_safe_output() -> None:
    captured: dict[str, object] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers.get("Authorization")
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "openai/gpt-oss-120b",
                "choices": [
                    {"message": {"role": "assistant", "content": "A considered answer."}}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            },
        )

    settings = ProviderSettings(
        base_url="https://api.groq.com/openai/v1",
        model="openai/gpt-oss-120b",
        api_key="test-secret",
        reasoning_effort="medium",
    )
    provider = OpenAICompatibleProvider(
        settings, transport=httpx.MockTransport(handle)
    )
    try:
        result = provider.generate(
            [ChatMessage(role="user", content="Think through this carefully.")],
            max_tokens=1200,
        )
    finally:
        provider.close()

    payload = captured["payload"]
    assert captured["authorization"] == "Bearer test-secret"
    assert payload["model"] == "openai/gpt-oss-120b"
    assert payload["max_completion_tokens"] == 1200
    assert payload["reasoning_effort"] == "medium"
    assert payload["include_reasoning"] is False
    assert "max_tokens" not in payload
    assert result.content == "A considered answer."
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 20
