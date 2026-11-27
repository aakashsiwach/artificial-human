from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from collections import defaultdict, deque
from threading import Lock
from typing import Any


SESSION_COOKIE = "artificial_human_session"
PASSKEY_COOKIE = "artificial_human_passkey"
CHALLENGE_COOKIE = "artificial_human_webauthn"
SESSION_TTL_SECONDS = 7 * 24 * 60 * 60
PASSKEY_TTL_SECONDS = 365 * 24 * 60 * 60
CHALLENGE_TTL_SECONDS = 180
PASSWORD_ITERATIONS = 600_000


class AuthenticationConfigurationError(RuntimeError):
    pass


class InvalidSignedValue(ValueError):
    pass


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def password_hash(password: str, *, iterations: int = PASSWORD_ITERATIONS) -> str:
    """Return a salted PBKDF2-SHA256 verifier suitable for a secret environment variable."""
    if len(password) < 12:
        raise ValueError("Passwords must contain at least 12 characters")
    if iterations < 600_000:
        raise ValueError("PBKDF2 iterations must be at least 600000")
    salt = secrets.token_bytes(18)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return "pbkdf2_sha256" + "$" + str(iterations) + "$" + _b64encode(salt) + "$" + _b64encode(digest)


def verify_password(password: str, encoded_hash: str) -> bool:
    try:
        algorithm, raw_iterations, encoded_salt, encoded_digest = encoded_hash.split("$", 3)
        iterations = int(raw_iterations)
        if algorithm != "pbkdf2_sha256" or not 600_000 <= iterations <= 2_000_000:
            return False
        salt = _b64decode(encoded_salt)
        expected = _b64decode(encoded_digest)
    except (TypeError, ValueError):
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(actual, expected)


def get_signing_secret() -> bytes:
    import os

    value = os.getenv("ARTIFICIAL_HUMAN_AUTH_SESSION_SECRET", "")
    if len(value) < 32:
        raise AuthenticationConfigurationError(
            "Set ARTIFICIAL_HUMAN_AUTH_SESSION_SECRET to a random value of at least 32 characters."
        )
    return value.encode("utf-8")


def sign_payload(payload: dict[str, Any], secret: bytes) -> str:
    body = _b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    signature = hmac.new(secret, body.encode("ascii"), hashlib.sha256).digest()
    return body + "." + _b64encode(signature)


def verify_payload(token: str, secret: bytes, *, purpose: str) -> dict[str, Any]:
    try:
        body, encoded_signature = token.split(".", 1)
        supplied_signature = _b64decode(encoded_signature)
        expected_signature = hmac.new(secret, body.encode("ascii"), hashlib.sha256).digest()
        if not hmac.compare_digest(supplied_signature, expected_signature):
            raise InvalidSignedValue("Invalid signature")
        payload = json.loads(_b64decode(body))
        if not isinstance(payload, dict) or payload.get("purpose") != purpose:
            raise InvalidSignedValue("Invalid token purpose")
        if int(payload.get("exp", 0)) < int(time.time()):
            raise InvalidSignedValue("Token expired")
        return payload
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        raise InvalidSignedValue("Invalid or expired token") from error


def create_session_token(username: str, secret: bytes) -> str:
    now = int(time.time())
    return sign_payload(
        {"purpose": "session", "sub": username, "iat": now, "exp": now + SESSION_TTL_SECONDS},
        secret,
    )


def read_session_token(token: str | None, secret: bytes) -> str | None:
    if not token:
        return None
    try:
        payload = verify_payload(token, secret, purpose="session")
    except InvalidSignedValue:
        return None
    username = payload.get("sub")
    return username if isinstance(username, str) and username else None


def create_challenge_token(challenge: bytes, *, purpose: str, username: str, secret: bytes) -> str:
    now = int(time.time())
    return sign_payload(
        {
            "purpose": "webauthn_challenge",
            "ceremony": purpose,
            "challenge": _b64encode(challenge),
            "sub": username,
            "iat": now,
            "exp": now + CHALLENGE_TTL_SECONDS,
        },
        secret,
    )


def read_challenge_token(token: str | None, *, purpose: str, username: str, secret: bytes) -> bytes:
    if not token:
        raise InvalidSignedValue("Passkey ceremony expired; start again")
    payload = verify_payload(token, secret, purpose="webauthn_challenge")
    if payload.get("ceremony") != purpose or payload.get("sub") != username:
        raise InvalidSignedValue("Passkey ceremony does not match this account")
    return _b64decode(str(payload["challenge"]))


def create_credential_token(
    *,
    username: str,
    credential_id: bytes,
    credential_public_key: bytes,
    sign_count: int,
    secret: bytes,
) -> str:
    now = int(time.time())
    return sign_payload(
        {
            "purpose": "passkey_credential",
            "sub": username,
            "credential_id": _b64encode(credential_id),
            "credential_public_key": _b64encode(credential_public_key),
            "sign_count": int(sign_count),
            "iat": now,
            "exp": now + PASSKEY_TTL_SECONDS,
        },
        secret,
    )


def read_credential_token(token: str | None, *, secret: bytes) -> dict[str, Any] | None:
    if not token:
        return None
    try:
        payload = verify_payload(token, secret, purpose="passkey_credential")
        credential_id = _b64decode(str(payload["credential_id"]))
        public_key = _b64decode(str(payload["credential_public_key"]))
        sign_count = int(payload["sign_count"])
        username = str(payload["sub"])
        if not credential_id or not public_key or sign_count < 0 or not username:
            return None
        return {
            "credential_id": credential_id,
            "credential_public_key": public_key,
            "sign_count": sign_count,
            "username": username,
        }
    except (InvalidSignedValue, KeyError, TypeError, ValueError):
        return None


class LoginRateLimiter:
    """Small process-local sliding-window limiter for the single-owner login endpoint."""

    def __init__(self, *, limit: int = 5, window_seconds: int = 900) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._attempts: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            attempts = self._attempts[key]
            while attempts and now - attempts[0] > self.window_seconds:
                attempts.popleft()
            if len(attempts) >= self.limit:
                return False
            attempts.append(now)
            if len(self._attempts) > 10_000:
                expired_keys = [
                    item for item, values in self._attempts.items()
                    if not values or now - values[-1] > self.window_seconds
                ]
                for item in expired_keys:
                    self._attempts.pop(item, None)
            return True

    def reset(self, key: str) -> None:
        """Clear accumulated attempts after a successful authentication."""
        with self._lock:
            self._attempts.pop(key, None)
