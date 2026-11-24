from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from artificial_human.brain.evaluation import evaluate_text_similarity
from artificial_human.brain.providers.base import (
    ProviderNotConfiguredError,
    ProviderRequestError,
)
from artificial_human.brain.service import BrainMode
from artificial_human.core.models import MemoryKind
from artificial_human.identity.profile import IdentityProfile
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


class TrainingExampleRequest(BaseModel):
    task_category: str = Field(min_length=1, max_length=120)
    input_text: str = Field(min_length=1, max_length=20_000)
    target_text: str = Field(min_length=1, max_length=20_000)
    source: str = Field(min_length=1, max_length=500)
    consent_id: str


class TrainingReviewRequest(BaseModel):
    decision: Literal["approved", "rejected"]
    note: str = Field(default="", max_length=2_000)


class SignalRequest(BaseModel):
    name: str
    value: float
    source: str = "manual"


class PlanRequest(BaseModel):
    goal: str = Field(min_length=1, max_length=4_000)


class IdentityUpdateRequest(BaseModel):
    display_name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=4_000)
    principles: list[str] = Field(default_factory=list, max_length=50)
    boundaries: list[str] = Field(default_factory=list, max_length=50)
    communication_preferences: dict[str, str] = Field(
        default_factory=dict, max_length=30
    )


class BrainRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    mode: BrainMode = "assistant"
    session_id: str | None = Field(default=None, min_length=1, max_length=128)


class BrainEvaluationRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    reference_response: str = Field(min_length=1, max_length=20_000)


def create_app(organism: ArtificialOrganism | None = None) -> FastAPI:
    app = FastAPI(
        title="Artificial Human",
        version="0.1.0",
        description="Local-first experimental artificial organism. Keep this API on loopback.",
    )
    app.state.organism = organism or ArtificialOrganism()

    def get_organism() -> ArtificialOrganism:
        return app.state.organism

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {
            "service": "Artificial Human",
            "message": "The artificial organism API is running.",
            "docs": "/docs",
            "health": "/health",
            "status": "/status",
        }

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "artificial-human"}

    @app.get("/status")
    def status() -> dict[str, Any]:
        return get_organism().status()

    @app.post("/brain/evaluate-style")
    def brain_evaluate_style(payload: BrainEvaluationRequest) -> dict[str, Any]:
        organism_ref = get_organism()
        try:
            memories = organism_ref.memory.list(organism_ref.owner_id, limit=500)
            result = organism_ref.cognitive_brain.respond(
                owner_id=organism_ref.owner_id,
                query=payload.message,
                profile=organism_ref.identity,
                memories=memories,
                mode="style_clone",
            )
        except ProviderNotConfiguredError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except ProviderRequestError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {
            "answer": result.answer,
            "model": result.model,
            "mode": result.mode,
            "reference_metrics": evaluate_text_similarity(
                payload.reference_response, result.answer
            ),
            "recalled_memories": result.recalled_memories,
            "approved_demonstrations": result.demonstrations,
            "plan_steps": result.plan_steps,
            "conversation_saved": False,
            "evaluation_note": (
                "Lexical similarity is a surface-form proxy. It does not measure semantic "
                "correctness, identity, or whether the response reflects the owner's real decisions."
            ),
        }

    @app.get("/identity")
    def get_identity() -> dict[str, Any]:
        return asdict(get_organism().identity)

    @app.put("/identity")
    def update_identity(payload: IdentityUpdateRequest) -> dict[str, Any]:
        organism_ref = get_organism()
        profile = IdentityProfile(
            owner_id=organism_ref.owner_id,
            display_name=payload.display_name,
            description=payload.description,
            principles=tuple(payload.principles),
            boundaries=tuple(payload.boundaries),
            communication_preferences=payload.communication_preferences,
        )
        try:
            organism_ref.identity_repository.save(profile)
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        organism_ref.identity = profile
        return asdict(profile)

    @app.get("/brain/cognition/status")
    def brain_cognition_status() -> dict[str, str]:
        return get_organism().cognitive_brain.architecture.status()

    @app.get("/brain/status")
    def brain_status() -> dict[str, object]:
        return get_organism().reasoning_status()

    @app.delete("/brain/sessions/{session_id}")
    def clear_brain_session(session_id: str) -> dict[str, bool]:
        cleared = get_organism().cognitive_brain.clear_session(
            get_organism().owner_id, session_id
        )
        return {"cleared": cleared}

    @app.post("/brain/respond")
    def brain_respond(payload: BrainRequest) -> dict[str, Any]:
        organism_ref = get_organism()
        try:
            memories = organism_ref.memory.list(organism_ref.owner_id, limit=500)
            result = organism_ref.cognitive_brain.respond(
                owner_id=organism_ref.owner_id,
                query=payload.message,
                profile=organism_ref.identity,
                memories=memories,
                mode=payload.mode,
                session_id=payload.session_id,
            )
        except ProviderNotConfiguredError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except ProviderRequestError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {
            "answer": result.answer,
            "model": result.model,
            "mode": result.mode,
            "recalled_memories": result.recalled_memories,
            "approved_demonstrations": result.demonstrations,
            "plan_steps": result.plan_steps,
            "cognitive_trace": result.cognitive_trace,
            "working_memory_session": result.session_id,
            "working_memory_persisted": False,
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
            },
            "conversation_saved": False,
        }

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

    @app.post("/training/examples", status_code=201)
    def add_training_example(payload: TrainingExampleRequest) -> dict[str, Any]:
        organism_ref = get_organism()
        try:
            example = organism_ref.learning.capture(
                owner_id=organism_ref.owner_id,
                task_category=payload.task_category,
                input_text=payload.input_text,
                target_text=payload.target_text,
                source=payload.source,
                consent_id=payload.consent_id,
            )
        except ConsentRequiredError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {
            "id": example.id,
            "task_category": example.task_category,
            "input_text": example.input_text,
            "target_text": example.target_text,
            "source": example.source,
            "status": example.status,
            "created_at": example.created_at,
        }

    @app.get("/training/examples")
    def list_training_examples(status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        organism_ref = get_organism()
        try:
            examples = organism_ref.learning.examples(
                organism_ref.owner_id, status=status, limit=limit
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return [
            {
                "id": item.id,
                "task_category": item.task_category,
                "input_text": item.input_text,
                "target_text": item.target_text,
                "source": item.source,
                "status": item.status,
                "reviewer": item.reviewer,
                "review_note": item.review_note,
                "created_at": item.created_at,
            }
            for item in examples
        ]

    @app.post("/training/examples/{example_id}/review")
    def review_training_example(
        example_id: str, payload: TrainingReviewRequest
    ) -> dict[str, str]:
        organism_ref = get_organism()
        try:
            organism_ref.learning.review(
                owner_id=organism_ref.owner_id,
                example_id=example_id,
                decision=payload.decision,
                reviewer=organism_ref.owner_id,
                note=payload.note,
            )
        except KeyError as error:
            raise HTTPException(
                status_code=404,
                detail="Pending example not found or its learning consent is inactive",
            ) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"id": example_id, "status": payload.decision}

    @app.delete("/training/examples/{example_id}")
    def delete_training_example(example_id: str) -> dict[str, bool]:
        organism_ref = get_organism()
        deleted = organism_ref.memory.delete_training_example(
            organism_ref.owner_id, example_id
        )
        if not deleted:
            raise HTTPException(status_code=404, detail="Training example not found")
        return {"deleted": True}

    @app.get("/training/corpus")
    def export_training_corpus() -> dict[str, Any]:
        organism_ref = get_organism()
        examples = organism_ref.learning.export_approved_corpus(organism_ref.owner_id)
        return {
            "count": len(examples),
            "format": "supervised-json-records",
            "note": "Approved examples only; exporting does not fine-tune a model.",
            "examples": examples,
        }

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
