import pytest

from artificial_human.core.models import MemoryKind
from artificial_human.memory.repository import ConsentRequiredError, MemoryRepository


def test_memory_requires_active_consent_and_preserves_provenance(tmp_path) -> None:
    repository = MemoryRepository(tmp_path / "memory.sqlite3")
    with pytest.raises(ConsentRequiredError):
        repository.add(
            owner_id="owner", kind=MemoryKind.EPISODIC, content="A remembered event",
            source="conversation:42", consent_id="missing",
        )
    grant = repository.grant_consent("owner")
    memory = repository.add(
        owner_id="owner", kind=MemoryKind.EPISODIC, content="A remembered event",
        source="conversation:42", consent_id=grant.id,
    )
    assert repository.list("owner")[0].id == memory.id
    assert repository.list("someone-else") == []


def test_revoking_consent_hides_memories_and_blocks_future_writes(tmp_path) -> None:
    repository = MemoryRepository(tmp_path / "memory.sqlite3")
    grant = repository.grant_consent("owner")
    repository.add(
        owner_id="owner", kind=MemoryKind.PREFERENCE, content="Prefers concise explanations",
        source="explicit user statement", consent_id=grant.id,
    )
    assert repository.revoke_consent("owner", grant.id) is True
    assert repository.list("owner") == []
    with pytest.raises(ConsentRequiredError):
        repository.add(
            owner_id="owner", kind=MemoryKind.PREFERENCE, content="Another preference",
            source="conversation", consent_id=grant.id,
        )


def test_owner_can_erase_all_memories_and_in_memory_database_is_usable() -> None:
    repository = MemoryRepository(":memory:")
    grant = repository.grant_consent("owner")
    item = repository.add(
        owner_id="owner", kind=MemoryKind.SEMANTIC, content="A fact",
        source="user-provided", consent_id=grant.id,
    )
    assert repository.delete("other-owner", item.id) is False
    assert repository.erase_owner("owner") == 1
    assert repository.list("owner") == []
