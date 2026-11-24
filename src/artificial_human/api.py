from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from artificial_human.core.models import MemoryKind
from artificial_human.memory.repository import ConsentRequiredError
from artificial_human.organism import ArtificialOrganism


class ConsentRequest(BaseModel):
    purpose: str = "personal_memory"


class MemoryRequest(BaseModel):
    kind: MemoryKind
    content: str = Field(min_length=1, max_length=20_000)
    source: str = Field(min_length=1, max_length=500)
    consent_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SignalRequest(BaseModel):
    name: str
    value: float
    source: str = "manual"


class PlanRequest(BaseModel):
    goal: str = Field(min_length=1, max_length=4_000)


def create_app(organism: ArtificialOrganism | None = None) -> FastAPI:
    app = FastAPI(
        title="Artificial Human",
        version="0.1.0",
        description="Local-first experimental artificial organism. Keep this API on loopback.",
    )
    app.state.organism = organism or ArtificialOrganism()

    def get_organism() -> ArtificialOrganism:
        return app.state.organism

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "artificial-human"}

    @app.get("/status")
    def status() -> dict[str, Any]:
        return get_organism().status()

    @app.post("/consents", status_code=201)
    def grant_consent(payload: ConsentRequest) -> dict[str, Any]:
        try:
            grant = get_organism().memory.grant_consent(get_organism().owner_id, payload.purpose)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {
            "id": grant.id, "owner_id": grant.owner_id, "purpose": grant.purpose,
            "active": grant.active, "created_at": grant.created_at,
        }

    @app.delete("/consents/{consent_id}")
    def revoke_consent(consent_id: str) -> dict[str, bool]:
        changed = get_organism().memory.revoke_consent(get_organism().owner_id, consent_id)
        if not changed:
            raise HTTPException(status_code=404, detail="Active consent not found")
        return {"revoked": True}

    @app.post("/memories", status_code=201)
    def add_memory(payload: MemoryRequest) -> dict[str, Any]:
        try:
            record = get_organism().memory.add(
                owner_id=get_organism().owner_id, kind=payload.kind,
                content=payload.content, source=payload.source,
                consent_id=payload.consent_id, metadata=payload.metadata,
            )
        except ConsentRequiredError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {
            "id": record.id, "kind": record.kind.value, "content": record.content,
            "source": record.source, "consent_id": record.consent_id,
            "created_at": record.created_at, "metadata": record.metadata,
        }

    @app.get("/memories")
    def list_memories(limit: int = 100) -> list[dict[str, Any]]:
        try:
            records = get_organism().memory.list(get_organism().owner_id, limit=limit)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return [
            {
                "id": item.id, "kind": item.kind.value, "content": item.content,
                "source": item.source, "created_at": item.created_at, "metadata": item.metadata,
            }
            for item in records
        ]

    @app.get("/memories/export")
    def export_memories() -> list[dict[str, Any]]:
        return get_organism().memory.export(get_organism().owner_id)

    @app.delete("/memories/{memory_id}")
    def delete_memory(memory_id: str) -> dict[str, bool]:
        deleted = get_organism().memory.delete(get_organism().owner_id, memory_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Memory not found")
        return {"deleted": True}

    @app.delete("/memories")
    def erase_memories() -> dict[str, int]:
        count = get_organism().memory.erase_owner(get_organism().owner_id)
        return {"deleted": count}

    @app.post("/signals/evaluate")
    def evaluate_signal(payload: SignalRequest) -> dict[str, Any]:
        return get_organism().evaluate_health(payload.name, payload.value, payload.source)

    @app.get("/incidents")
    def incidents() -> list[dict[str, Any]]:
        return [
            {
                "id": item.id, "signal": item.signal.name, "value": item.signal.value,
                "severity": item.severity.value, "summary": item.summary,
                "created_at": item.created_at.isoformat(),
            }
            for item in get_organism().immune.incidents()
        ]

    @app.get("/repairs")
    def repairs() -> list[dict[str, Any]]:
        return [
            {
                "id": item.id, "incident_id": item.incident_id, "summary": item.summary,
                "steps": item.steps, "safety_checks": item.safety_checks, "status": item.status,
            }
            for item in get_organism().regeneration.list()
        ]

    @app.post("/repairs/{proposal_id}/approve")
    def approve_repair(proposal_id: str) -> dict[str, Any]:
        try:
            item = get_organism().regeneration.approve(
                proposal_id, reviewer=get_organism().owner_id
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Proposal not found") from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"id": item.id, "status": item.status, "summary": item.summary}

    @app.post("/plans")
    def create_plan(payload: PlanRequest) -> dict[str, Any]:
        try:
            plan = get_organism().brain.plan(payload.goal)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {
            "id": plan.id, "goal": plan.goal, "note": plan.note,
            "steps": [
                {"id": step.id, "description": step.description, "requires_approval": step.requires_approval}
                for step in plan.steps
            ],
        }

    @app.post("/sleep/preview")
    def sleep_preview() -> dict[str, Any]:
        organism_ref = get_organism()
        records = organism_ref.memory.list(organism_ref.owner_id, limit=500)
        proposal = organism_ref.sleep.prepare(records)
        if proposal is None:
            return {"proposal": None, "reason": "At least two consented memories are needed."}
        return {
            "proposal": {
                "id": proposal.id, "source_memory_ids": proposal.source_memory_ids,
                "summary": proposal.summary, "status": proposal.status,
            }
        }

    return app


app = create_app()
