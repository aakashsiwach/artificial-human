import pytest

from artificial_human.core.models import MemoryKind
from artificial_human.memory.repository import (
    ConsentRequiredError,
    MemoryNotFoundError,
    MemoryRepository,
)


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



def test_consolidated_memory_requires_visible_episodic_sources_and_keeps_provenance() -> None:
    repository = MemoryRepository(":memory:")
    consent = repository.grant_consent("owner")
    first = repository.add(
        owner_id="owner", kind=MemoryKind.EPISODIC, content="Kickoff event",
        source="owner event log", consent_id=consent.id,
    )
    second = repository.add(
        owner_id="owner", kind=MemoryKind.EPISODIC, content="Review event",
        source="owner event log", consent_id=consent.id,
    )

    consolidated = repository.add_consolidated_memory(
        owner_id="owner",
        source_memory_ids=[first.id, second.id],
        content="A recurring workflow pattern.",
        source="owner-approved sleep consolidation:proposal-1",
        consent_id=consent.id,
        metadata={"consolidation_proposal_id": "proposal-1"},
    )

    assert consolidated.kind is MemoryKind.SEMANTIC
    assert consolidated.epistemic_status == "inferred"
    assert consolidated.metadata["consolidation_source_memory_ids"] == [first.id, second.id]
    assert consolidated.metadata["consolidation_source_count"] == 2
    assert consolidated.metadata["consolidation_review_required"] is False
    assert len(repository.list("owner")) == 3


def test_consolidated_memory_rejects_conflicting_revoked_and_non_episodic_sources() -> None:
    repository = MemoryRepository(":memory:")
    consent = repository.grant_consent("owner")
    first = repository.add(
        owner_id="owner", kind=MemoryKind.EPISODIC, content="Original plan",
        source="owner event log", consent_id=consent.id,
    )
    second = repository.add(
        owner_id="owner", kind=MemoryKind.EPISODIC, content="Alternative plan",
        source="owner event log", consent_id=consent.id,
    )
    repository.link_contradiction("owner", first.id, second.id, note="Conflicting plans")
    with pytest.raises(ValueError, match="Contradictory"):
        repository.add_consolidated_memory(
            owner_id="owner", source_memory_ids=[first.id, second.id],
            content="Do not consolidate conflicting plans.", source="review",
            consent_id=consent.id,
        )

    third = repository.add(
        owner_id="owner", kind=MemoryKind.PREFERENCE, content="A preference",
        source="owner statement", consent_id=consent.id,
    )
    with pytest.raises(ValueError, match="episodic"):
        repository.add_consolidated_memory(
            owner_id="owner", source_memory_ids=[first.id, third.id],
            content="Invalid consolidation.", source="review", consent_id=consent.id,
        )

    assert repository.revoke_consent("owner", consent.id) is True
    with pytest.raises(MemoryNotFoundError):
        repository.add_consolidated_memory(
            owner_id="owner", source_memory_ids=[first.id, second.id],
            content="Revoked data cannot be consolidated.", source="review",
            consent_id=consent.id,
        )
