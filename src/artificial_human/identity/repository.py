from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from artificial_human.identity.profile import IdentityProfile


class IdentityRepository:
    """Atomic local persistence for the owner's explicitly configured identity profile."""

    def __init__(self, owner_id: str, path: str | Path | None = None) -> None:
        if not owner_id.strip():
            raise ValueError("owner_id is required")
        self.owner_id = owner_id
        configured = os.getenv("ARTIFICIAL_HUMAN_IDENTITY_PATH")
        self.path = Path(path or configured or ".artificial-human/identity.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> IdentityProfile:
        if not self.path.exists():
            return IdentityProfile(owner_id=self.owner_id)
        try:
            data: Any = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Identity profile must contain a JSON object.")
            if data.get("owner_id") != self.owner_id:
                raise ValueError("Identity file belongs to a different owner.")
            preferences = data.get("communication_preferences", {})
            if not isinstance(preferences, dict):
                raise ValueError("communication_preferences must be an object.")
            return IdentityProfile(
                owner_id=self.owner_id,
                display_name=str(data.get("display_name", "Owner")),
                description=str(data.get("description", "")),
                principles=tuple(str(item) for item in data.get("principles", [])),
                boundaries=tuple(str(item) for item in data.get("boundaries", [])),
                communication_preferences={
                    str(key): str(value) for key, value in preferences.items()
                },
            )
        except (OSError, json.JSONDecodeError, TypeError, AttributeError) as error:
            raise ValueError("The saved identity profile is invalid or unreadable.") from error

    def save(self, profile: IdentityProfile) -> IdentityProfile:
        if profile.owner_id != self.owner_id:
            raise ValueError("Identity profile owner does not match this organism.")
        if not profile.display_name.strip():
            raise ValueError("display_name cannot be empty.")
        payload = json.dumps(asdict(profile), ensure_ascii=False, indent=2)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(payload + "\n", encoding="utf-8")
        os.replace(temporary, self.path)
        return profile
