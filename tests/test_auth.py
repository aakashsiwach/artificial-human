from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from artificial_human.auth import (
    create_credential_token,
    password_hash,
    read_credential_token,
    read_session_token,
    sign_payload,
    verify_password,
    verify_payload,
)
from artificial_human.api import create_app
from artificial_human.organism import ArtificialOrganism


def test_password_hash_uses_salted_pbkdf2_and_verifies() -> None:
    first = password_hash("Correct Horse Battery Staple 42")
    second = password_hash("Correct Horse Battery Staple 42")
    assert first != second
    assert first.startswith("pbkdf2_sha256$600000$")
    assert verify_password("Correct Horse Battery Staple 42", first)
    assert not verify_password("incorrect password", first)


def test_signed_session_and_passkey_credential_records_reject_tampering() -> None:
    secret = b"test-signing-secret-that-is-long-enough"
    token = sign_payload(
        {"purpose": "session", "sub": "aakash", "exp": 4_000_000_000},
        secret,
    )
    assert verify_payload(token, secret, purpose="session")["sub"] == "aakash"
    assert read_session_token(token, secret) == "aakash"
    with pytest.raises(ValueError):
        verify_payload(token + "x", secret, purpose="session")

    credential = create_credential_token(
        username="aakash",
        credential_id=b"credential-id",
        credential_public_key=b"public-key-bytes",
        sign_count=3,
        secret=secret,
    )
    decoded = read_credential_token(credential, secret=secret)
    assert decoded is not None
    assert decoded["credential_id"] == b"credential-id"
    assert decoded["credential_public_key"] == b"public-key-bytes"
    assert decoded["sign_count"] == 3
    assert decoded["username"] == "aakash"
    assert read_credential_token(credential + "x", secret=secret) is None


def configure_auth(monkeypatch: pytest.MonkeyPatch) -> str:
    password = "Correct Horse Battery Staple 42"
    monkeypatch.setenv("ARTIFICIAL_HUMAN_PUBLIC_DEPLOYMENT", "true")
    monkeypatch.setenv("ARTIFICIAL_HUMAN_MONITOR_ONLY", "true")
    monkeypatch.setenv("ARTIFICIAL_HUMAN_INTERACTIVE_MODE", "true")
    monkeypatch.setenv("ARTIFICIAL_HUMAN_AUTH_USERNAME", "aakash")
    monkeypatch.setenv("ARTIFICIAL_HUMAN_AUTH_PASSWORD_HASH", password_hash(password))
    monkeypatch.setenv("ARTIFICIAL_HUMAN_AUTH_SESSION_SECRET", "s" * 64)
    monkeypatch.setenv("ARTIFICIAL_HUMAN_AUTH_ORIGIN", "https://control.example")
    monkeypatch.setenv("ARTIFICIAL_HUMAN_WEBAUTHN_RP_ID", "control.example")
    monkeypatch.delenv("ARTIFICIAL_HUMAN_API_TOKEN", raising=False)
    return password


def test_password_login_restores_session_and_logout_revokes_browser_cookie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    password = configure_auth(monkeypatch)
    organism = ArtificialOrganism(storage_mode="volatile", brain_provider=None)
    client = TestClient(
        create_app(organism),
        base_url="https://control.example",
    )

    initial = client.get("/auth/session")
    assert initial.status_code == 200
    assert initial.json()["authenticated"] is False

    denied = client.get("/monitor/overview")
    assert denied.status_code == 401

    login = client.post(
        "/auth/password",
        headers={"Origin": "https://control.example"},
        json={"username": "aakash", "password": password},
    )
    assert login.status_code == 200
    assert login.json()["authenticated"] is True
    cookie = login.headers.get("set-cookie", "")
    assert "HttpOnly" in cookie
    assert "Secure" in cookie
    assert "SameSite=strict" in cookie

    restored = client.get("/auth/session")
    assert restored.status_code == 200
    assert restored.json()["authenticated"] is True
    assert restored.json()["username"] == "aakash"

    snapshot = client.get("/monitor/overview")
    assert snapshot.status_code == 200

    logout = client.post(
        "/auth/logout",
        headers={"Origin": "https://control.example"},
    )
    assert logout.status_code == 200
    assert client.get("/auth/session").json()["authenticated"] is False
    assert client.get("/monitor/overview").status_code == 401


def test_password_login_rejects_wrong_password_and_foreign_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_auth(monkeypatch)
    organism = ArtificialOrganism(storage_mode="volatile", brain_provider=None)
    client = TestClient(create_app(organism), base_url="https://control.example")

    foreign = client.post(
        "/auth/password",
        headers={"Origin": "https://attacker.example"},
        json={"username": "aakash", "password": "anything"},
    )
    assert foreign.status_code == 403

    wrong = client.post(
        "/auth/password",
        headers={"Origin": "https://control.example"},
        json={"username": "aakash", "password": "not the right password"},
    )
    assert wrong.status_code == 401


def test_authenticated_owner_can_generate_a_passkey_registration_challenge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    password = configure_auth(monkeypatch)
    organism = ArtificialOrganism(storage_mode="volatile", brain_provider=None)
    client = TestClient(create_app(organism), base_url="https://control.example")
    login = client.post(
        "/auth/password",
        headers={"Origin": "https://control.example"},
        json={"username": "aakash", "password": password},
    )
    assert login.status_code == 200

    options = client.post(
        "/auth/passkey/register/options",
        headers={"Origin": "https://control.example"},
    )
    assert options.status_code == 200
    payload = options.json()
    assert payload["rp"]["id"] == "control.example"
    assert payload["user"]["name"] == "aakash"
    assert payload["authenticatorSelection"]["userVerification"] == "required"
    assert payload["authenticatorSelection"]["residentKey"] == "required"
    assert options.headers.get("set-cookie", "").find("artificial_human_webauthn=") >= 0


def test_public_control_room_requires_cookie_or_legacy_bearer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_auth(monkeypatch)
    monkeypatch.setenv("ARTIFICIAL_HUMAN_API_TOKEN", "legacy-private-token")
    organism = ArtificialOrganism(storage_mode="volatile", brain_provider=None)
    client = TestClient(create_app(organism), base_url="https://control.example")

    missing = client.get("/monitor/overview")
    assert missing.status_code == 401

    legacy = client.get(
        "/monitor/overview",
        headers={"Authorization": "Bearer legacy-private-token"},
    )
    assert legacy.status_code == 200
