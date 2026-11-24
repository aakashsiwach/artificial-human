from artificial_human.identity.profile import IdentityProfile
from artificial_human.identity.repository import IdentityRepository


def test_identity_profile_persists_and_round_trips(tmp_path) -> None:
    repository = IdentityRepository("owner", tmp_path / "identity.json")
    profile = IdentityProfile(
        owner_id="owner",
        display_name="Aakash",
        description="Software engineer who values clear reasoning.",
        principles=("Be honest about uncertainty.",),
        boundaries=("Do not send messages without approval.",),
        communication_preferences={"tone": "natural", "language": "English"},
    )

    repository.save(profile)
    assert repository.load() == profile


def test_identity_repository_rejects_a_different_owner(tmp_path) -> None:
    first = IdentityRepository("owner-one", tmp_path / "identity.json")
    second = IdentityRepository("owner-two", tmp_path / "identity.json")
    first.save(IdentityProfile(owner_id="owner-one", display_name="Owner One"))

    try:
        second.load()
    except ValueError as error:
        assert "different owner" in str(error)
    else:
        raise AssertionError("Identity repository accepted another owner's profile")
