from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class IdentityProfile:
    """Owner-authored identity configuration; not inferred psychological truth."""

    owner_id: str
    display_name: str = "Owner"
    description: str = ""
    principles: tuple[str, ...] = ()
    boundaries: tuple[str, ...] = (
        "Do not impersonate the owner in consequential situations.",
        "Do not take external actions without explicit approval.",
        "Treat inferred preferences as uncertain until confirmed.",
    )
    communication_preferences: dict[str, str] = field(default_factory=dict)
