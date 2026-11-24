"""LongMemEval (Oracle) evidence-retrieval adapter.

Source: Wu et al. (2025), "LongMemEval: Benchmarking chat assistants on
long-term interactive memory", ICLR 2025 (arXiv:2410.10813). File
``longmemeval_oracle.json`` from huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
(MIT). Only the Oracle variant is supported. It contains evidence sessions
only, which are simulated user–assistant sessions; there is no ShareGPT or
UltraChat filler.

What is measured: whether long-term memory *retrieves the evidence* for a
question after the question's full history has been ingested. Answer accuracy
is **not** measured; no language model is used.

Because every Oracle history session is an evidence session, session-level
precision is 1 by construction. The informative signal is **turn-level**
retrieval of the turns labelled ``has_answer``. Session-level metrics are
reported for completeness with this caveat.

Each question is ingested into its own fresh, in-memory repository under a
unique synthetic owner, through the normal consent-gated write path. The
repository is erased and closed afterwards. Outputs contain no question text,
conversation content, or dataset answers.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from artificial_human.cognition.benchmark.external import ExternalDatasetError, sha256_file
from artificial_human.cognition.benchmark.metrics import (
    mean,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from artificial_human.core.models import MemoryKind
from artificial_human.memory.embeddings import EmbeddingProvider
from artificial_human.memory.repository import MemoryRepository
from artificial_human.memory.search import SearchConfig
from artificial_human.memory.system import LongTermMemory

EXPECTED_FILE = "longmemeval_oracle.json"
REQUIRED_FIELDS = frozenset({
    "question_id", "question_type", "question", "answer", "question_date",
    "haystack_session_ids", "haystack_dates", "haystack_sessions", "answer_session_ids",
})
QUESTION_TYPES = frozenset({
    "single-session-user", "single-session-assistant", "single-session-preference",
    "temporal-reasoning", "knowledge-update", "multi-session",
})
DATE_FORMAT = "%Y/%m/%d (%a) %H:%M"
K_VALUES = (1, 5, 10, 20)
RETRIEVAL_DEPTH = 20


@dataclass(frozen=True, slots=True)
class Turn:
    role: str
    content: str
    has_answer: bool


@dataclass(frozen=True, slots=True)
class Session:
    session_id: str
    date: datetime
    turns: tuple[Turn, ...]


@dataclass(frozen=True, slots=True)
class Question:
    question_id: str
    question_type: str
    question: str
    question_date: datetime
    sessions: tuple[Session, ...]
    answer_session_ids: frozenset[str]

    @property
    def abstention(self) -> bool:
        return self.question_id.endswith("_abs")


def parse_date(text: str) -> datetime:
    """Parse LongMemEval dates ("2023/04/10 (Mon) 23:07"). No offset is given; UTC is assumed."""
    try:
        return datetime.strptime(text, DATE_FORMAT).replace(tzinfo=UTC)
    except (TypeError, ValueError) as error:
        raise ExternalDatasetError(f"Unparseable LongMemEval date {text!r}") from error


def _validate(raw: Any, index: int) -> Question:
    where = f"item {index}"
    if not isinstance(raw, dict):
        raise ExternalDatasetError(f"{where}: expected an object")
    missing = REQUIRED_FIELDS - set(raw)
    if missing:
        raise ExternalDatasetError(f"{where}: missing fields {sorted(missing)}")
    if raw["question_type"] not in QUESTION_TYPES:
        raise ExternalDatasetError(f"{where}: unknown question_type {raw['question_type']!r}")
    for key in ("question_id", "question", "question_date"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ExternalDatasetError(f"{where}: {key} must be a non-empty string")
    ids, dates, sessions = raw["haystack_session_ids"], raw["haystack_dates"], raw["haystack_sessions"]
    if not (isinstance(ids, list) and isinstance(dates, list) and isinstance(sessions, list)):
        raise ExternalDatasetError(f"{where}: haystack fields must be lists")
    if not (len(ids) == len(dates) == len(sessions)) or not ids:
        raise ExternalDatasetError(f"{where}: haystack id/date/session lengths differ or are empty")
    if len(set(ids)) != len(ids):
        raise ExternalDatasetError(f"{where}: duplicate haystack session ids")
    answers = raw["answer_session_ids"]
    if not isinstance(answers, list) or not set(answers) <= set(ids):
        raise ExternalDatasetError(f"{where}: answer_session_ids must be a subset of haystack ids")
    parsed_sessions = []
    for session_id, date, turns in zip(ids, dates, sessions, strict=True):
        if not isinstance(turns, list) or not turns:
            raise ExternalDatasetError(f"{where}: session {session_id!r} has no turns")
        parsed_turns = []
        for turn in turns:
            if not isinstance(turn, dict) or turn.get("role") not in ("user", "assistant"):
                raise ExternalDatasetError(f"{where}: invalid turn role in {session_id!r}")
            if not isinstance(turn.get("content"), str):
                raise ExternalDatasetError(f"{where}: turn content must be a string")
            flag = turn.get("has_answer", False)
            if not isinstance(flag, bool):
                raise ExternalDatasetError(f"{where}: has_answer must be boolean")
            parsed_turns.append(Turn(turn["role"], turn["content"], flag))
        parsed_sessions.append(Session(str(session_id), parse_date(date), tuple(parsed_turns)))
    return Question(
        question_id=raw["question_id"], question_type=raw["question_type"],
        question=raw["question"], question_date=parse_date(raw["question_date"]),
        sessions=tuple(parsed_sessions), answer_session_ids=frozenset(map(str, answers)),
    )


def load_longmemeval(path: Path, *, limit: int | None = None) -> tuple[list[Question], dict[str, Any]]:
    """Load and validate the local Oracle file. Never downloads anything."""
    if not path.is_file():
        raise ExternalDatasetError(
            f"LongMemEval file not found: {path}. Download it manually "
            "(see docs/benchmarks/EXTERNAL_DATASETS.md)."
        )
    if path.name != EXPECTED_FILE:
        raise ExternalDatasetError(
            f"Only {EXPECTED_FILE} is supported; refusing {path.name!r} (the _s/_m variants "
            "contain ShareGPT/UltraChat filler sessions)."
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExternalDatasetError(f"{path.name} is not valid UTF-8 JSON") from error
    if not isinstance(raw, list) or not raw:
        raise ExternalDatasetError("LongMemEval data must be a non-empty JSON list")
    questions = [_validate(item, index) for index, item in enumerate(raw)]
    if len({question.question_id for question in questions}) != len(questions):
        raise ExternalDatasetError("duplicate question_id values")
    meta = {
        "file": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size,
        "questions_in_file": len(questions),
        "question_types": dict(sorted(Counter(q.question_type for q in questions).items())),
        "abstention_questions": sum(q.abstention for q in questions),
        "sessions": sum(len(q.sessions) for q in questions),
        "turns": sum(len(s.turns) for q in questions for s in q.sessions),
    }
    if limit is not None:
        questions = questions[:limit]
    meta["questions_evaluated"] = len(questions)
    return questions, meta


VARIANTS: dict[str, dict[str, Any]] = {
    "lexical": {"use_lexical": True, "use_semantic": False, "use_temporal_intent": False},
    "lexical_temporal": {"use_lexical": True, "use_semantic": False, "use_temporal_intent": True},
    "semantic": {"use_lexical": False, "use_semantic": True, "use_temporal_intent": False},
    "hybrid_temporal": {"use_lexical": True, "use_semantic": True, "use_temporal_intent": True},
}


def _bin(value: float, edges: Sequence[tuple[float, str]], last: str) -> str:
    for upper, label in edges:
        if value <= upper:
            return label
    return last


def evaluate_question(
    question: Question, index: int, provider: EmbeddingProvider | None, variants: Sequence[str]
) -> dict[str, Any]:
    clock = {"now": datetime(2000, 1, 1, tzinfo=UTC)}
    repository = MemoryRepository(":memory:", clock=lambda: clock["now"])
    owner = f"longmemeval-owner-{index}"
    try:
        consent = repository.grant_consent(owner).id
        turn_info: dict[str, tuple[str, int, bool]] = {}
        chronological = sorted(question.sessions, key=lambda session: session.date)
        order: list[str] = []
        for session in chronological:
            for position, turn in enumerate(session.turns):
                if not turn.content.strip():
                    continue
                clock["now"] = session.date + timedelta(seconds=position)
                record = repository.add(
                    owner_id=owner, kind=MemoryKind.EPISODIC, content=turn.content,
                    source=f"longmemeval:{session.session_id}", consent_id=consent,
                    occurred_at=session.date,
                    metadata={"session_id": session.session_id, "turn": position,
                              "role": turn.role},
                )
                turn_info[record.id] = (session.session_id, position, turn.has_answer)
                order.append(record.id)
        memory = LongTermMemory(repository, provider)
        if provider is not None:
            memory.reindex(owner)
        evidence_turns = {key for key, value in turn_info.items() if value[2]}
        evidence_sessions = {value[0] for value in turn_info.values() if value[2]}
        first_evidence = min((order.index(key) for key in evidence_turns), default=None)
        evidence_dates = [s.date for s in question.sessions if s.session_id in question.answer_session_ids]
        outcome: dict[str, Any] = {
            "question_id": question.question_id,
            "category": question.question_type,
            "abstention": question.abstention,
            "sessions": len(question.sessions),
            "turns": len(order),
            "evidence_turns": len(evidence_turns),
            "labelled_evidence_sessions": len(evidence_sessions),
            "evidence_position": (
                None if first_evidence is None
                else _bin(first_evidence / max(len(order) - 1, 1),
                          ((1 / 3, "early"), (2 / 3, "middle")), "late")
            ),
            "days_since_latest_evidence": (
                (question.question_date - max(evidence_dates)).total_seconds() / 86400
                if evidence_dates else None
            ),
            "variants": {},
        }
        base = SearchConfig()
        for name in variants:
            settings = VARIANTS[name]
            if settings["use_semantic"] and provider is None:
                continue
            response = memory.search(
                owner, question.question, top_k=RETRIEVAL_DEPTH,
                config=replace(base, **settings), reference_time=question.question_date,
            )
            ranked = [item.memory.id for item in response.results]
            leaks = sum(item.memory.owner_id != owner for item in response.results)
            sessions_ranked = list(dict.fromkeys(turn_info[key][0] for key in ranked))
            scores: dict[str, Any] = {"returned": len(ranked), "owner_leaks": leaks}
            if evidence_turns and not question.abstention:
                relevant = {key: 1 for key in evidence_turns}
                for k in K_VALUES:
                    scores[f"turn_recall@{k}"] = recall_at_k(ranked, relevant, k)
                    scores[f"turn_precision@{k}"] = precision_at_k(ranked, relevant, k)
                    scores[f"turn_ndcg@{k}"] = ndcg_at_k(ranked, relevant, k)
                scores["turn_mrr"] = reciprocal_rank(ranked, relevant, RETRIEVAL_DEPTH)
                scores["turn_recall_curve"] = [
                    recall_at_k(ranked, relevant, k) for k in range(1, RETRIEVAL_DEPTH + 1)
                ]
            if not question.abstention:
                relevant_sessions = {key: 1 for key in question.answer_session_ids}
                for k in (1, 5):
                    scores[f"session_recall@{k}"] = recall_at_k(sessions_ranked, relevant_sessions, k)
                    scores[f"session_precision@{k}"] = precision_at_k(
                        sessions_ranked, relevant_sessions, k)
                scores["session_mrr"] = reciprocal_rank(sessions_ranked, relevant_sessions, 20)
            outcome["variants"][name] = scores
        # Isolation probe: another owner must never see this question's memories.
        probe = memory.search(f"{owner}-probe", question.question, top_k=5)
        outcome["isolation_probe_results"] = len(probe.results)
        return outcome
    finally:
        repository.erase_owner(owner)
        repository.close()


DAY_BINS = ((1, "0-1 days"), (7, "2-7 days"), (30, "8-30 days"), (180, "31-180 days"))
SESSION_BINS = ((1, "1 session"), (2, "2 sessions"), (3, "3 sessions"))
TURN_BINS = ((10, "<=10 turns"), (20, "11-20 turns"), (40, "21-40 turns"))
TURN_KEYS = ("turn_recall@1", "turn_recall@5", "turn_recall@10", "turn_recall@20",
             "turn_precision@5", "turn_ndcg@10", "turn_mrr")


def _summarise(outcomes: Sequence[dict[str, Any]], variant: str) -> dict[str, Any]:
    scored = [o["variants"][variant] for o in outcomes
              if variant in o["variants"] and "turn_mrr" in o["variants"][variant]]
    summary: dict[str, Any] = {"questions": len(scored)}
    for key in TURN_KEYS:
        summary[key] = round(mean([item[key] for item in scored]), 4) if scored else None
    return summary


def aggregate(outcomes: Sequence[dict[str, Any]], variants: Sequence[str]) -> dict[str, Any]:
    present = [name for name in variants if any(name in o["variants"] for o in outcomes)]
    non_abstention = [o for o in outcomes if not o["abstention"]]
    abstention = [o for o in outcomes if o["abstention"]]
    report: dict[str, Any] = {
        "questions": len(outcomes),
        "abstention_questions": len(abstention),
        "questions_without_turn_labels": sum(
            1 for o in non_abstention if o["evidence_turns"] == 0),
        "owner_leaks": sum(v["owner_leaks"] for o in outcomes for v in o["variants"].values()),
        "isolation_probe_results": sum(o["isolation_probe_results"] for o in outcomes),
        "variants": {},
    }
    groupings: dict[str, Callable[[dict[str, Any]], str | None]] = {
        "category": lambda o: o["category"],
        "history_sessions": lambda o: _bin(o["sessions"], SESSION_BINS, "4+ sessions"),
        "history_turns": lambda o: _bin(o["turns"], TURN_BINS, ">40 turns"),
        "evidence_position": lambda o: o["evidence_position"],
        "evidence_sessions": lambda o: (
            None if o["labelled_evidence_sessions"] == 0
            else "single-session evidence" if o["labelled_evidence_sessions"] == 1
            else "multi-session evidence"
        ),
        "days_since_latest_evidence": lambda o: (
            None if o["days_since_latest_evidence"] is None
            else _bin(o["days_since_latest_evidence"], DAY_BINS, ">180 days")
        ),
    }
    for variant in present:
        entry: dict[str, Any] = {"overall": _summarise(non_abstention, variant)}
        for grouping, key_of in groupings.items():
            groups: dict[str, list[dict[str, Any]]] = {}
            for outcome in non_abstention:
                key = key_of(outcome)
                if key is not None:
                    groups.setdefault(key, []).append(outcome)
            entry[f"by_{grouping}"] = {
                key: _summarise(members, variant) for key, members in sorted(groups.items())
            }
        curves = [o["variants"][variant]["turn_recall_curve"] for o in non_abstention
                  if "turn_recall_curve" in o["variants"].get(variant, {})]
        entry["turn_recall_by_depth"] = {
            str(k + 1): round(mean([curve[k] for curve in curves]), 4)
            for k in range(RETRIEVAL_DEPTH)
        } if curves else {}
        sessions = [o["variants"][variant] for o in non_abstention if variant in o["variants"]]
        entry["session_level"] = {
            key: round(mean([s[key] for s in sessions]), 4)
            for key in ("session_recall@1", "session_recall@5", "session_precision@1",
                        "session_mrr")
        } if sessions else {}
        abstained = [o["variants"][variant] for o in abstention if variant in o["variants"]]
        entry["abstention"] = {
            "questions": len(abstained),
            "rejection_rate": round(mean([1.0 if a["returned"] == 0 else 0.0 for a in abstained]), 4)
            if abstained else None,
            "mean_results_returned": round(mean([a["returned"] for a in abstained]), 2)
            if abstained else None,
        }
        report["variants"][variant] = entry
    return report


def run_longmemeval(
    path: Path,
    *,
    provider: EmbeddingProvider | None = None,
    limit: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    questions, meta = load_longmemeval(path, limit=limit)
    variants = [name for name in VARIANTS if provider is not None or not VARIANTS[name]["use_semantic"]]
    notify = progress or (lambda _message: None)
    outcomes = []
    for index, question in enumerate(questions):
        if index and index % 100 == 0:
            notify(f"longmemeval: {index}/{len(questions)} questions")
        outcomes.append(evaluate_question(question, index, provider, variants))
    summary = aggregate(outcomes, variants)
    # Per-question failures are listed by id and category only (no text).
    primary = "hybrid_temporal" if "hybrid_temporal" in summary["variants"] else "lexical_temporal"
    failures = [
        {"question_id": o["question_id"], "category": o["category"],
         "evidence_turns": o["evidence_turns"]}
        for o in outcomes
        if not o["abstention"] and "turn_recall@20" in o["variants"].get(primary, {})
        and o["variants"][primary]["turn_recall@20"] == 0.0
    ]
    return {"dataset": meta, "summary": summary, "primary_variant": primary,
            "missed_at_depth_20": {"count": len(failures), "examples": failures[:10]}}
