"""Long-term memory facade: storage, consent, indexing and retrieval together.

Every write goes to the repository first (the source of truth). Indexing runs
afterwards and can only add or remove derived vectors; an indexing failure is
reported but never rolls back or alters the stored memory.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from artificial_human.memory.embeddings import EmbeddingProvider
from artificial_human.memory.index import IndexOutcome, ReindexReport, SemanticIndex
from artificial_human.memory.repository import (
    UNSET,
    MemoryLink,
    MemoryRecord,
    MemoryRepository,
)
from artificial_human.memory.search import (
    HybridMemorySearch,
    MemorySearchResponse,
    SearchConfig,
    SearchFilters,
)


class LongTermMemory:
    def __init__(
        self,
        repository: MemoryRepository,
        provider: EmbeddingProvider | None = None,
        config: SearchConfig | None = None,
    ) -> None:
        self.repository = repository
        self.index = SemanticIndex(repository, provider)
        self.search_engine = HybridMemorySearch(repository, self.index, config)

    def add(self, **fields: Any) -> tuple[MemoryRecord, IndexOutcome]:
        record = self.repository.add(**fields)
        return record, self.index.index_memory(record.owner_id, record.id)

    def add_consolidated_memory(
        self,
        *,
        owner_id: str,
        source_memory_ids: list[str] | tuple[str, ...],
        content: str,
        source: str,
        consent_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[MemoryRecord, IndexOutcome]:
        record = self.repository.add_consolidated_memory(
            owner_id=owner_id,
            source_memory_ids=source_memory_ids,
            content=content,
            source=source,
            consent_id=consent_id,
            metadata=metadata,
        )
        return record, self.index.index_memory(owner_id, record.id)

    def add_schema_memory(
        self,
        *,
        owner_id: str,
        source_sequences: list[list[str]] | tuple[tuple[str, ...], ...],
        event_types: list[str] | tuple[str, ...],
        schema: dict[str, Any],
        content: str,
        source: str,
        consent_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[MemoryRecord, IndexOutcome]:
        record = self.repository.add_schema_memory(
            owner_id=owner_id,
            source_sequences=source_sequences,
            event_types=event_types,
            schema=schema,
            content=content,
            source=source,
            consent_id=consent_id,
            metadata=metadata,
        )
        return record, self.index.index_memory(owner_id, record.id)

    def correct(
        self, owner_id: str, memory_id: str, **changes: Any
    ) -> tuple[MemoryRecord, IndexOutcome | None]:
        before = self.repository.get(owner_id, memory_id)
        record = self.repository.correct(
            owner_id, memory_id, **{key: changes.get(key, UNSET) for key in (
                "content", "metadata", "occurred_at", "valid_from", "valid_to",
                "fact_key", "epistemic_status",
            )},
        )
        if record.content == before.content:
            return record, None
        return record, self.index.index_memory(owner_id, memory_id)

    def supersede(self, **fields: Any) -> tuple[MemoryRecord, MemoryLink, IndexOutcome]:
        record, link = self.repository.supersede(**fields)
        return record, link, self.index.index_memory(record.owner_id, record.id)

    def search(
        self,
        owner_id: str,
        query: str,
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
        config: SearchConfig | None = None,
        reference_time: datetime | None = None,
    ) -> MemorySearchResponse:
        return self.search_engine.search(
            owner_id, query, top_k=top_k, filters=filters, config=config,
            reference_time=reference_time,
        )

    def reindex(self, owner_id: str, *, full: bool = False) -> ReindexReport:
        return self.index.reindex(owner_id, full=full)

    def status(self, owner_id: str) -> dict[str, Any]:
        return {
            **self.index.status(owner_id),
            "schema_version": self.repository.schema_version(),
            "ranking": self.search_engine.config.as_dict(),
        }
