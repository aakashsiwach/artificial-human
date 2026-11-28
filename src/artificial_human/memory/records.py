"""Serialization helpers for owner-visible memory records."""
from __future__ import annotations

from typing import Any

from artificial_human.memory.repository import MemoryRecord


def memory_record_to_dict(item: MemoryRecord) -> dict[str, Any]:
    return {
        "id": item.id,
        "owner_id": item.owner_id,
        "kind": item.kind.value,
        "content": item.content,
        "source": item.source,
        "consent_id": item.consent_id,
        "created_at": item.created_at,
        "recorded_at": item.recorded_at,
        "occurred_at": item.occurred_at,
        "valid_from": item.valid_from,
        "valid_to": item.valid_to,
        "fact_key": item.fact_key,
        "epistemic_status": item.epistemic_status,
        "updated_at": item.updated_at,
        "metadata": item.metadata,
    }
