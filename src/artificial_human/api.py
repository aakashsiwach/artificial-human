from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
import hmac
import os
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from artificial_human.brain.evaluation import evaluate_text_similarity
from artificial_human.brain.providers.base import (
    ProviderNotConfiguredError,
    ProviderRequestError,
)
from artificial_human.brain.service import BrainMode
from artificial_human.core.models import MemoryKind
from artificial_human.development.prenatal import GESTATION_DAYS, PrenatalBrain
from artificial_human.identity.profile import IdentityProfile
from artificial_human.memory.index import IndexOutcome
from artificial_human.memory.repository import (
    ConsentRequiredError,
    MemoryNotFoundError,
    MemoryRecord,
    MemoryRepository,
)
from artificial_human.memory.search import SearchFilters
from artificial_human.memory.temporal import TimeRange, parse_timestamp
from artificial_human.organism import ArtificialOrganism


class ConsentRequest(BaseModel):
    purpose: str = "personal_memory"


class MemoryRequest(BaseModel):
    kind: MemoryKind
    content: str = Field(min_length=1, max_length=20_000)
    source: str = Field(min_length=1, max_length=500)
    consent_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    occurred_at: str | None = Field(
        default=None, description="When the event happened (ISO-8601 with offset)"
    )
    valid_from: str | None = Field(default=None, description="When the fact became true")
    valid_to: str | None = Field(default=None, description="When the fact stopped being true")
    fact_key: str | None = Field(
        default=None, max_length=200,
        description="Stable subject/attribute key, for example owner.employer",
    )
    epistemic_status: Literal["asserted", "inferred"] = "asserted"


class MemoryCorrectionRequest(BaseModel):
    content: str | None = Field(default=None, min_length=1, max_length=20_000)
    metadata: dict[str, Any] | None = None
    occurred_at: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    fact_key: str | None = Field(default=None, max_length=200)
    epistemic_status: Literal["asserted", "inferred"] | None = None


class SupersedeRequest(BaseModel):
    content: str = Field(min_length=1, max_length=20_000)
    source: str = Field(min_length=1, max_length=500)
    consent_id: str
    kind: MemoryKind | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    occurred_at: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    epistemic_status: Literal["asserted", "inferred"] = "asserted"
    note: str = Field(default="", max_length=2_000)


class ContradictionRequest(BaseModel):
    other_memory_id: str
    note: str = Field(default="", max_length=2_000)


class MemorySearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4_000)
    top_k: int = Field(default=5, ge=1, le=100)
    kinds: list[MemoryKind] = Field(default_factory=list)
    start: str | None = Field(default=None, description="Time-range start (inclusive)")
    end: str | None = Field(default=None, description="Time-range end (exclusive)")
    valid_at: str | None = Field(default=None, description="Only facts valid at this time")
    known_at: str | None = Field(
        default=None, description="Only what the system had recorded by this time"
    )
    source_prefix: str | None = Field(default=None, max_length=500)
    fact_key: str | None = Field(default=None, max_length=200)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)
    include_superseded: bool = True
    infer_temporal_intent: bool = True
    include_content: bool = True
    reference_time: str | None = None


class ReindexRequest(BaseModel):
    full: bool = False


def _optional_time(value: str | None) -> datetime | None:
    return parse_timestamp(value) if value else None


def _memory_payload(record: MemoryRecord, *, include_content: bool = True) -> dict[str, Any]:
    payload = MemoryRepository.record_to_dict(record)
    payload.pop("owner_id")
    if not include_content:
        payload.pop("content")
    return payload


def _index_payload(outcome: IndexOutcome | None) -> dict[str, Any] | None:
    if outcome is None:
        return None
    return {"status": outcome.status, "reason": outcome.reason}


def _link_payload(link: Any) -> dict[str, Any]:
    return {
        "id": link.id, "relation": link.relation, "from_memory_id": link.from_memory_id,
        "to_memory_id": link.to_memory_id, "effective_at": link.effective_at,
        "created_at": link.created_at, "note": link.note,
    }


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
    public_deployment = os.getenv("ARTIFICIAL_HUMAN_PUBLIC_DEPLOYMENT", "").lower() in {
        "1", "true", "yes",
    }
    docs_enabled = not public_deployment
    app = FastAPI(
        title="Artificial Human",
        version="0.1.0",
        description="Experimental artificial organism API. Public deployments require bearer auth.",
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
    )
    configured_origins = os.getenv(
        "ARTIFICIAL_HUMAN_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    allowed_origins = [item.strip() for item in configured_origins.split(",") if item.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.middleware("http")
    async def protect_public_deployment(request: Any, call_next: Any) -> Any:
        if not public_deployment or request.url.path == "/health" or request.method == "OPTIONS":
            return await call_next(request)
        expected = os.getenv("ARTIFICIAL_HUMAN_API_TOKEN", "")
        if not expected:
            return JSONResponse(
                status_code=503,
                content={"detail": "Public API authentication is not configured."},
            )
        scheme, _, supplied = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(supplied, expected):
            return JSONResponse(
                status_code=401,
                content={"detail": "A valid bearer token is required."},
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)

    app.state.public_deployment = public_deployment
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
        payload = get_organism().status()
        payload.pop("owner_id", None)
        return payload

    @app.get("/monitor/overview")
    def monitor_overview(
        days: int = Query(default=GESTATION_DAYS, ge=0, le=GESTATION_DAYS),
        seed: int = Query(default=7, ge=0, le=2_147_483_647),
    ) -> dict[str, Any]:
        """Read-only operational snapshot; never returns prompts or memory content."""
        organism_ref = get_organism()
        organism_status = organism_ref.status()
        organism_status.pop("owner_id", None)
        prenatal = PrenatalBrain(seed=seed)
        prenatal.advance(days)
        return {
            "observed_at": datetime.now().astimezone().isoformat(),
            "organism": organism_status,
            "cognitive_systems": organism_ref.cognitive_brain.architecture.status(),
            "development": {
                "prenatal": prenatal.snapshot().as_dict(),
                "note": (
                    "Deterministic engineering snapshot, not a biological fetal-brain model. "
                    "No exposure history or stimulus content is returned."
                ),
            },
        }

    @app.post("/brain/evaluate-style")
    def brain_evaluate_style(payload: BrainEvaluationRequest) -> dict[str, Any]:
        organism_ref = get_organism()
        try:
            # Long-term memory is retrieved inside the cognitive cycle by the
            # owner-scoped, consent-filtered hybrid search.
            result = organism_ref.cognitive_brain.respond(
                owner_id=organism_ref.owner_id,
                query=payload.message,
                profile=organism_ref.identity,
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
            # Long-term memory is retrieved inside the cognitive cycle by the
            # owner-scoped, consent-filtered hybrid search.
            result = organism_ref.cognitive_brain.respond(
                owner_id=organism_ref.owner_id,
                query=payload.message,
                profile=organism_ref.identity,
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
        get_organism().cognitive_brain.clear_owner_sessions(get_organism().owner_id)
        return {"revoked": True}

    @app.post("/memories", status_code=201)
    def add_memory(payload: MemoryRequest) -> dict[str, Any]:
        organism_ref = get_organism()
        try:
            record, outcome = organism_ref.long_term_memory.add(
                owner_id=organism_ref.owner_id, kind=payload.kind,
                content=payload.content, source=payload.source,
                consent_id=payload.consent_id, metadata=payload.metadata,
                occurred_at=payload.occurred_at, valid_from=payload.valid_from,
                valid_to=payload.valid_to, fact_key=payload.fact_key,
                epistemic_status=payload.epistemic_status,
            )
        except ConsentRequiredError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {**_memory_payload(record), "index": _index_payload(outcome)}

    @app.get("/memories")
    def list_memories(limit: int = 100) -> list[dict[str, Any]]:
        try:
            records = get_organism().memory.list(get_organism().owner_id, limit=limit)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return [_memory_payload(item) for item in records]

    @app.get("/memories/export")
    def export_memories() -> list[dict[str, Any]]:
        return get_organism().memory.export(get_organism().owner_id)

    @app.post("/memories/search")
    def search_memories(payload: MemorySearchRequest) -> dict[str, Any]:
        organism_ref = get_organism()
        engine = organism_ref.long_term_memory.search_engine
        try:
            start, end = _optional_time(payload.start), _optional_time(payload.end)
            filters = SearchFilters(
                kinds=tuple(payload.kinds),
                time_range=TimeRange(start, end) if start or end else None,
                valid_at=_optional_time(payload.valid_at),
                known_at=_optional_time(payload.known_at),
                source_prefix=payload.source_prefix, fact_key=payload.fact_key,
                metadata=payload.metadata, include_superseded=payload.include_superseded,
            )
            response = engine.search(
                organism_ref.owner_id, payload.query, top_k=payload.top_k, filters=filters,
                config=replace(engine.config, use_temporal_intent=payload.infer_temporal_intent),
                reference_time=_optional_time(payload.reference_time),
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        explanation = response.explain()
        return {
            "strategy": response.strategy,
            "results": [
                {
                    "rank": item.rank,
                    "score": item.score,
                    "memory": _memory_payload(
                        item.memory, include_content=payload.include_content
                    ),
                    "explanation": item.explanation,
                }
                for item in response.results
            ],
            "temporal_query": explanation["temporal_query"],
            "filters": explanation["filters"],
            "semantic": explanation["semantic"],
            "reference_time": explanation["reference_time"],
            "candidate_count": response.candidate_count,
        }

    @app.get("/memories/timeline")
    def memory_timeline(
        start: str | None = None, end: str | None = None, valid_at: str | None = None,
        known_at: str | None = None, kind: MemoryKind | None = None, limit: int = 100,
    ) -> list[dict[str, Any]]:
        organism_ref = get_organism()
        try:
            start_time, end_time = _optional_time(start), _optional_time(end)
            entries = organism_ref.long_term_memory.search_engine.timeline(
                organism_ref.owner_id,
                filters=SearchFilters(
                    kinds=(kind,) if kind else (),
                    time_range=(
                        TimeRange(start_time, end_time) if start_time or end_time else None
                    ),
                    valid_at=_optional_time(valid_at), known_at=_optional_time(known_at),
                ),
                limit=limit,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return [
            {"memory": _memory_payload(record), "temporal": profile.as_dict()}
            for record, profile in entries
        ]

    @app.get("/memories/index/status")
    def memory_index_status() -> dict[str, Any]:
        organism_ref = get_organism()
        return organism_ref.long_term_memory.status(organism_ref.owner_id)

    @app.post("/memories/index/rebuild")
    def rebuild_memory_index(payload: ReindexRequest | None = None) -> dict[str, Any]:
        organism_ref = get_organism()
        report = organism_ref.long_term_memory.reindex(
            organism_ref.owner_id, full=bool(payload and payload.full)
        )
        return {
            **report.as_dict(),
            "status": organism_ref.long_term_memory.status(organism_ref.owner_id),
        }

    @app.get("/memories/{memory_id}")
    def inspect_memory(memory_id: str) -> dict[str, Any]:
        organism_ref = get_organism()
        repository = organism_ref.memory
        try:
            record = repository.get(organism_ref.owner_id, memory_id)
            chain = repository.version_chain(organism_ref.owner_id, memory_id)
        except MemoryNotFoundError as error:
            raise HTTPException(status_code=404, detail="Memory not found") from error
        info = organism_ref.long_term_memory.index.model_info()
        indexed = info is not None and memory_id in repository.embeddings(
            organism_ref.owner_id, info.space, [memory_id]
        )
        return {
            "memory": _memory_payload(record),
            "versions": [_memory_payload(item) for item in chain],
            "links": [
                _link_payload(link) for link in repository.links(organism_ref.owner_id)
                if memory_id in (link.from_memory_id, link.to_memory_id)
            ],
            "semantic_index": {"indexed": indexed, "space": info.space if info else None},
        }

    @app.patch("/memories/{memory_id}")
    def correct_memory(memory_id: str, payload: MemoryCorrectionRequest) -> dict[str, Any]:
        organism_ref = get_organism()
        changes = payload.model_dump(exclude_unset=True)
        for required in ("content", "epistemic_status"):
            if required in changes and changes[required] is None:
                raise HTTPException(status_code=400, detail=f"{required} cannot be null")
        try:
            record, outcome = organism_ref.long_term_memory.correct(
                organism_ref.owner_id, memory_id, **changes
            )
        except MemoryNotFoundError as error:
            raise HTTPException(status_code=404, detail="Memory not found") from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        organism_ref.cognitive_brain.clear_owner_sessions(organism_ref.owner_id)
        return {**_memory_payload(record), "index": _index_payload(outcome)}

    @app.post("/memories/{memory_id}/supersede", status_code=201)
    def supersede_memory(memory_id: str, payload: SupersedeRequest) -> dict[str, Any]:
        organism_ref = get_organism()
        try:
            record, link, outcome = organism_ref.long_term_memory.supersede(
                owner_id=organism_ref.owner_id, memory_id=memory_id, kind=payload.kind,
                content=payload.content, source=payload.source, consent_id=payload.consent_id,
                metadata=payload.metadata, occurred_at=payload.occurred_at,
                valid_from=payload.valid_from, valid_to=payload.valid_to,
                epistemic_status=payload.epistemic_status, note=payload.note,
            )
        except MemoryNotFoundError as error:
            raise HTTPException(status_code=404, detail="Memory not found") from error
        except ConsentRequiredError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        organism_ref.cognitive_brain.clear_owner_sessions(organism_ref.owner_id)
        return {
            "memory": _memory_payload(record),
            "supersedes": memory_id,
            "link": _link_payload(link),
            "index": _index_payload(outcome),
        }

    @app.post("/memories/{memory_id}/contradictions", status_code=201)
    def record_contradiction(memory_id: str, payload: ContradictionRequest) -> dict[str, Any]:
        organism_ref = get_organism()
        try:
            link = organism_ref.memory.link_contradiction(
                organism_ref.owner_id, memory_id, payload.other_memory_id, note=payload.note
            )
        except MemoryNotFoundError as error:
            raise HTTPException(status_code=404, detail="Memory not found") from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return _link_payload(link)

    @app.delete("/memories/{memory_id}")
    def delete_memory(memory_id: str) -> dict[str, bool]:
        deleted = get_organism().memory.delete(get_organism().owner_id, memory_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Memory not found")
        get_organism().cognitive_brain.clear_owner_sessions(get_organism().owner_id)
        return {"deleted": True}

    @app.delete("/memories")
    def erase_memories() -> dict[str, int]:
        organism_ref = get_organism()
        count = organism_ref.memory.erase_owner(organism_ref.owner_id)
        sessions_cleared = organism_ref.cognitive_brain.clear_owner_sessions(
            organism_ref.owner_id
        )
        return {"deleted": count, "sessions_cleared": sessions_cleared}

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
        organism_ref.cognitive_brain.clear_owner_sessions(organism_ref.owner_id)
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
