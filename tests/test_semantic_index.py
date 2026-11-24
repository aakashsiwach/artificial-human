from __future__ import annotations

import logging
import sqlite3

import httpx
import pytest
from conftest import ConceptEmbedding

from artificial_human.core.models import MemoryKind
from artificial_human.memory.embeddings import (
    EmbeddingError,
    EmbeddingUnavailableError,
    FastEmbedProvider,
    OpenAICompatibleEmbeddingProvider,
    decode_vector,
    embedding_provider_from_env,
    encode_vector,
    normalise_vector,
)
from artificial_human.memory.repository import MemoryRepository, content_fingerprint
from artificial_human.memory.system import LongTermMemory

SECRET = "Private detail: my locker code story about boiling water"


def _memory(tmp_path, provider) -> tuple[LongTermMemory, str]:
    repository = MemoryRepository(tmp_path / "memory.sqlite3")
    return LongTermMemory(repository, provider), repository.grant_consent("owner").id


def _count(tmp_path, table: str) -> int:
    connection = sqlite3.connect(tmp_path / "memory.sqlite3")
    try:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    finally:
        connection.close()


def test_vectors_are_normalised_and_round_trip_as_float32() -> None:
    vector = normalise_vector([3.0, 4.0], 2)
    assert vector == pytest.approx([0.6, 0.8])
    assert list(decode_vector(encode_vector(vector))) == pytest.approx(vector, rel=1e-6)
    with pytest.raises(EmbeddingError, match="dimensions"):
        normalise_vector([1.0], 2)
    with pytest.raises(EmbeddingError, match="zero"):
        normalise_vector([0.0, 0.0], 2)
    with pytest.raises(EmbeddingError, match="non-finite"):
        normalise_vector([float("nan"), 1.0], 2)


def test_new_memory_is_indexed_incrementally_with_model_metadata(tmp_path) -> None:
    memory, consent = _memory(tmp_path, ConceptEmbedding())
    record, outcome = memory.add(
        owner_id="owner", kind=MemoryKind.SEMANTIC, content="Water boils at 100 C",
        source="textbook", consent_id=consent,
    )
    assert outcome.status == "indexed"
    stored = memory.repository.embeddings("owner", "test:test-concepts:10")
    assert stored[record.id].model_id == "test-concepts"
    assert stored[record.id].model_version == "1"
    assert stored[record.id].dimensions == 10
    assert stored[record.id].content_hash == content_fingerprint(record.content)
    status = memory.status("owner")
    assert status["retrieval_mode"] == "hybrid"
    assert status["indexed_memories"] == 1


def test_without_backend_memories_are_stored_but_not_embedded(tmp_path) -> None:
    memory, consent = _memory(tmp_path, None)
    _, outcome = memory.add(
        owner_id="owner", kind=MemoryKind.SEMANTIC, content="Fact", source="s",
        consent_id=consent,
    )
    assert outcome.status == "disabled"
    assert memory.status("owner")["retrieval_mode"] == "lexical_only"
    assert _count(tmp_path, "memory_embeddings") == 0


def test_embedding_failure_keeps_memory_intact_and_is_recorded(tmp_path, caplog) -> None:
    provider = ConceptEmbedding(fail=True)
    memory, consent = _memory(tmp_path, provider)
    with caplog.at_level(logging.DEBUG):
        record, outcome = memory.add(
            owner_id="owner", kind=MemoryKind.SEMANTIC, content=SECRET, source="s",
            consent_id=consent,
        )
    assert outcome.status == "failed"
    assert memory.repository.get("owner", record.id).content == SECRET
    status = memory.status("owner")
    assert status["failed"] == 1
    assert status["failures"][0]["error_type"] == "EmbeddingError"
    assert _count(tmp_path, "memory_embeddings") == 0
    # Lexical retrieval still works while the semantic backend is down.
    response = memory.search("owner", "locker code")
    assert [item.memory.id for item in response.results] == [record.id]
    assert response.semantic["used"] is False
    # Logs carry identifiers and error classes only, never memory content.
    assert "locker" not in caplog.text and "Private" not in caplog.text

    provider.fail = False
    report = memory.reindex("owner")
    assert (report.indexed, report.failed) == (1, 0)
    assert memory.status("owner")["failed"] == 0


def test_deleting_a_memory_removes_its_embedding_and_index_state(tmp_path) -> None:
    memory, consent = _memory(tmp_path, ConceptEmbedding())
    record, _ = memory.add(
        owner_id="owner", kind=MemoryKind.SEMANTIC, content="Water boils", source="s",
        consent_id=consent,
    )
    other, _ = memory.add(
        owner_id="owner", kind=MemoryKind.SEMANTIC, content="Tea is hot", source="s",
        consent_id=consent,
    )
    memory.repository.link_contradiction("owner", record.id, other.id)
    assert _count(tmp_path, "memory_embeddings") == 2
    assert memory.repository.delete("owner", record.id) is True
    assert _count(tmp_path, "memory_embeddings") == 1
    assert _count(tmp_path, "memory_links") == 0
    assert memory.repository.delete("owner", record.id) is False


def test_revoking_consent_deletes_embeddings_and_blocks_indexing(tmp_path) -> None:
    provider = ConceptEmbedding()
    memory, consent = _memory(tmp_path, provider)
    record, _ = memory.add(
        owner_id="owner", kind=MemoryKind.SEMANTIC, content="Water boils", source="s",
        consent_id=consent,
    )
    assert memory.repository.revoke_consent("owner", consent) is True
    assert _count(tmp_path, "memory_embeddings") == 0
    calls_before = len(provider.calls)
    assert memory.index.index_memory("owner", record.id).status == "skipped"
    report = memory.reindex("owner", full=True)
    assert report.indexed == 0
    assert len(provider.calls) == calls_before  # no further processing of revoked data
    assert memory.search("owner", "water boils").results == ()


def test_store_embedding_refuses_stale_or_ineligible_vectors(tmp_path) -> None:
    memory, consent = _memory(tmp_path, None)
    record, _ = memory.add(
        owner_id="owner", kind=MemoryKind.SEMANTIC, content="Original", source="s",
        consent_id=consent,
    )
    common = {
        "owner_id": "owner", "memory_id": record.id, "space": "x:y:2", "model_id": "y",
        "model_version": "1", "dimensions": 2, "vector": encode_vector([1.0, 0.0]),
    }
    # The text changed between embedding and storage: the vector must be discarded.
    assert memory.repository.store_embedding(
        content_hash=content_fingerprint("Something else"), **common
    ) is False
    assert memory.repository.store_embedding(
        content_hash=content_fingerprint("Original"), **{**common, "owner_id": "intruder"}
    ) is False
    assert memory.repository.store_embedding(
        content_hash=content_fingerprint("Original"), **common
    ) is True


def test_correction_replaces_the_embedding_for_new_content(tmp_path) -> None:
    provider = ConceptEmbedding()
    memory, consent = _memory(tmp_path, provider)
    record, _ = memory.add(
        owner_id="owner", kind=MemoryKind.SEMANTIC, content="I drink coffee", source="s",
        consent_id=consent,
    )
    _, outcome = memory.correct("owner", record.id, content="I drink green tea")
    assert outcome is not None and outcome.status == "indexed"
    stored = memory.repository.embeddings("owner", provider.describe().space)
    assert stored[record.id].content_hash == content_fingerprint("I drink green tea")
    _, no_reindex = memory.correct("owner", record.id, metadata={"note": "x"})
    assert no_reindex is None


def test_model_change_reindexes_and_removes_vectors_from_previous_model(tmp_path) -> None:
    memory, consent = _memory(tmp_path, ConceptEmbedding(model_id="model-v1"))
    for content in ("Water boils", "Tea is hot"):
        memory.add(
            owner_id="owner", kind=MemoryKind.SEMANTIC, content=content, source="s",
            consent_id=consent,
        )
    upgraded = LongTermMemory(memory.repository, ConceptEmbedding(model_id="model-v2"))
    before = upgraded.status("owner")
    assert before["indexed_memories"] == 0
    assert before["vectors_from_other_models"] == 2
    report = upgraded.reindex("owner")
    assert (report.indexed, report.removed_other_spaces) == (2, 2)
    after = upgraded.status("owner")
    assert (after["indexed_memories"], after["vectors_from_other_models"]) == (2, 0)


def test_remote_embedding_requires_opt_in_https_and_owner_consent(tmp_path) -> None:
    with pytest.raises(EmbeddingUnavailableError, match="disabled"):
        OpenAICompatibleEmbeddingProvider(
            base_url="https://embeddings.example.com/v1", model="m", dimensions=2
        )
    with pytest.raises(ValueError, match="HTTPS"):
        OpenAICompatibleEmbeddingProvider(
            base_url="http://embeddings.example.com/v1", model="m", dimensions=2,
            allow_remote=True,
        )
    remote = ConceptEmbedding(locality="remote")
    memory, consent = _memory(tmp_path, remote)
    _, outcome = memory.add(
        owner_id="owner", kind=MemoryKind.SEMANTIC, content="Water boils", source="s",
        consent_id=consent,
    )
    assert outcome.status == "skipped"
    assert remote.calls == []  # nothing was sent to the remote backend
    assert memory.status("owner")["retrieval_mode"] == "lexical_only_external_consent_missing"
    memory.repository.grant_consent("owner", "external_embedding")
    assert memory.reindex("owner").indexed == 1


def test_openai_compatible_provider_parses_and_validates_responses() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.content)
        seen.append(body)
        data = [
            {"index": index, "embedding": [1.0, float(index)]}
            for index, _ in enumerate(body["input"])
        ]
        return httpx.Response(200, json={"data": list(reversed(data))})

    provider = OpenAICompatibleEmbeddingProvider(
        base_url="http://127.0.0.1:11434/v1", model="local-embed", dimensions=2,
        transport=httpx.MockTransport(handler),
    )
    assert provider.describe().locality == "loopback"
    vectors = provider.embed_documents(["a", "b"])
    assert vectors[0] == pytest.approx([1.0, 0.0])
    assert seen[0]["model"] == "local-embed"

    failing = OpenAICompatibleEmbeddingProvider(
        base_url="http://localhost:11434/v1", model="m", dimensions=2,
        transport=httpx.MockTransport(lambda request: httpx.Response(500)),
    )
    with pytest.raises(EmbeddingError, match="HTTP 500"):
        failing.embed_query("secret text")


def test_embedding_backend_is_disabled_unless_configured(monkeypatch) -> None:
    monkeypatch.delenv("ARTIFICIAL_HUMAN_EMBEDDING_BACKEND", raising=False)
    assert embedding_provider_from_env() is None
    monkeypatch.setenv("ARTIFICIAL_HUMAN_EMBEDDING_BACKEND", "fastembed")
    monkeypatch.setenv("ARTIFICIAL_HUMAN_EMBEDDING_OFFLINE", "true")
    provider = embedding_provider_from_env()
    assert isinstance(provider, FastEmbedProvider) and provider.offline is True
    monkeypatch.setenv("ARTIFICIAL_HUMAN_EMBEDDING_BACKEND", "openai_compatible")
    with pytest.raises(ValueError, match="DIMENSIONS"):
        embedding_provider_from_env()
    monkeypatch.setenv("ARTIFICIAL_HUMAN_EMBEDDING_BACKEND", "cloud-magic")
    with pytest.raises(ValueError, match="Unknown"):
        embedding_provider_from_env()


def test_unloadable_local_model_degrades_to_lexical_only(tmp_path) -> None:
    memory, consent = _memory(
        tmp_path, FastEmbedProvider("not-a-real/model", cache_dir=str(tmp_path), offline=True)
    )
    record, outcome = memory.add(
        owner_id="owner", kind=MemoryKind.SEMANTIC, content="Water boils", source="s",
        consent_id=consent,
    )
    assert outcome.status == "disabled"
    status = memory.status("owner")
    assert status["retrieval_mode"] == "lexical_only_backend_unavailable"
    assert status["backend_error"] == "EmbeddingUnavailableError"
    assert [item.memory.id for item in memory.search("owner", "water").results] == [record.id]
