"""Opt-in evaluation of schema extraction on independently authored XES logs."""
from __future__ import annotations

import random
from pathlib import Path
from statistics import fmean

from artificial_human.cognition.benchmark.common import SuiteResult, check, derive_seed
from artificial_human.cognition.episodic_replay import EpisodicEvent, EpisodicReplayMemory
from artificial_human.cognition.schema_extraction import SchemaExtractionSystem
from artificial_human.core.models import MemoryKind
from artificial_human.memory.repository import MemoryRepository

from artificial_human.cognition.benchmark.external.event_log import (
    MAX_EVENTS_PER_CASE,
    EventLogCase,
    load_xes,
)


PERMUTATION_REPEATS = 5
DEFAULT_MAX_CASES = 2_000


def _contains(sequence: tuple[str, ...], pattern: tuple[str, ...]) -> bool:
    return any(
        sequence[index:index + len(pattern)] == pattern
        for index in range(len(sequence) - len(pattern) + 1)
    )


def _contains_any(sequence: tuple[str, ...], patterns: tuple[tuple[str, ...], ...]) -> bool:
    return any(_contains(sequence, pattern) for pattern in patterns)


def run_schema_event_log_suite(
    path: Path,
    *,
    seed: int,
    attribute_keys: tuple[str, ...] = (),
    max_cases: int = DEFAULT_MAX_CASES,
) -> SuiteResult:
    """Discover on earlier cases and report recurrence on chronologically later cases.

    Event labels come from the dataset's XES concept:name field. The adapter maps
    those publisher-authored labels into safe slugs; it does not infer activities
    from free text or write approved memories.
    """
    if isinstance(max_cases, bool) or not isinstance(max_cases, int) or not 4 <= max_cases <= 10_000:
        raise ValueError("max_cases must be between 4 and 10000")
    log = load_xes(path, attribute_keys=attribute_keys)
    ordered_cases = sorted(
        log.cases,
        key=lambda case: (case.events[0].occurred_at, case.case_digest),
    )
    selected_cases = ordered_cases[-max_cases:]
    split_at = max(2, min(len(selected_cases) - 2, int(len(selected_cases) * 0.8)))
    training_cases = tuple(selected_cases[:split_at])
    evaluation_cases = tuple(selected_cases[split_at:])

    owner_id = "external-schema-eval-" + log.sha256[:16]
    repository = MemoryRepository(":memory:")
    consent = repository.grant_consent(owner_id)
    owner_digest = __import__("hashlib").sha256(owner_id.encode("utf-8")).hexdigest()[:16]
    context_prefix = f"owner-{owner_digest}.session-"
    replay = EpisodicReplayMemory(
        max_contexts=len(training_cases) + 1,
        max_episodes_per_context=1,
        max_events_per_episode=MAX_EVENTS_PER_CASE,
        max_transitions=1,
    )
    extractor = SchemaExtractionSystem(max_proposals=100)
    training_event_count = 0

    for case in training_cases:
        episode: list[EpisodicEvent] = []
        context_id = context_prefix + case.case_digest
        for event in case.events:
            record = repository.add(
                owner_id=owner_id,
                kind=MemoryKind.EPISODIC,
                content="External event-log record",
                source="external-workflow-event-log",
                consent_id=consent.id,
                occurred_at=event.occurred_at,
                metadata={"event_type": event.event_type, "attributes": event.attributes},
            )
            episode.append(EpisodicEvent(
                memory_id=record.id,
                occurred_at=record.occurred_at or record.created_at,
            ))
            training_event_count += 1
        replay.record_episode(episode, context_id=context_id)

    proposals = extractor.prepare_from_episodes(
        owner_id=owner_id,
        context_prefix=context_prefix,
        episodic_memory=replay,
        repository=repository,
    )
    patterns = tuple(proposal.event_types for proposal in proposals)

    evaluation_sequences = tuple(
        tuple(event.event_type for event in case.events) for case in evaluation_cases
    )
    per_pattern_support = [
        sum(_contains(sequence, pattern) for sequence in evaluation_sequences)
        / len(evaluation_sequences)
        for pattern in patterns
    ]
    observed_coverage = (
        sum(_contains_any(sequence, patterns) for sequence in evaluation_sequences)
        / len(evaluation_sequences)
        if patterns else 0.0
    )

    shuffled_hits = 0
    for repeat in range(PERMUTATION_REPEATS):
        for case, sequence in zip(evaluation_cases, evaluation_sequences, strict=True):
            shuffled = list(sequence)
            random.Random(derive_seed(seed, "schema-event-log", str(repeat), case.case_digest)).shuffle(
                shuffled
            )
            shuffled_hits += int(_contains_any(tuple(shuffled), patterns))
    shuffled_coverage = shuffled_hits / (PERMUTATION_REPEATS * len(evaluation_sequences))

    training_vocabulary = {
        event.event_type for case in training_cases for event in case.events
    }
    evaluation_vocabulary = {
        event.event_type for case in evaluation_cases for event in case.events
    }
    overlap = (
        len(training_vocabulary.intersection(evaluation_vocabulary)) / len(evaluation_vocabulary)
        if evaluation_vocabulary else 0.0
    )
    mean_support = fmean(per_pattern_support) if per_pattern_support else 0.0
    mean_training_support = (
        fmean(proposal.support_count for proposal in proposals) if proposals else 0.0
    )
    evaluation_event_count = sum(len(case.events) for case in evaluation_cases)
    case_ids_disjoint = not {
        case.case_digest for case in training_cases
    }.intersection(case.case_digest for case in evaluation_cases)

    result = SuiteResult(
        suite="schema-event-log",
        title="Owner-reviewed schema extraction on an external event log",
        evidence_category="independently authored event-log probe (opt-in local XES; no gold schemas)",
        splits={"train_cases": len(training_cases), "eval_cases": len(evaluation_cases)},
        systems={
            "schema_induction": {
                "training_proposal_count": len(proposals),
                "mean_training_support_cases": round(mean_training_support, 4),
                "mean_proposal_eval_case_support": round(mean_support, 4),
                "heldout_eval_case_coverage": round(observed_coverage, 4),
                "shuffled_control_eval_case_coverage": round(shuffled_coverage, 4),
                "heldout_coverage_lift_over_shuffle": round(observed_coverage - shuffled_coverage, 4),
                "train_eval_activity_vocabulary_overlap": round(overlap, 4),
            }
        },
        checks=[
            check(
                "training and evaluation cases are disjoint",
                case_ids_disjoint,
                train_cases=len(training_cases),
                eval_cases=len(evaluation_cases),
            ),
            check(
                "every proposed schema has at least two training examples",
                all(proposal.support_count >= 2 for proposal in proposals),
                proposals=len(proposals),
            ),
        ],
        findings=[
            (
                "Activity types are derived from the external XES concept:name labels. "
                "This adapter does not infer event labels from free text."
            ),
            (
                "The score measures temporal case holdout only. It is not schema precision "
                "or semantic correctness because the source log supplies no gold schema labels."
            ),
            (
                "A shuffled within-case order is reported as a simple null control; it is not "
                "a substitute for a domain-matched process-mining baseline."
            ),
            (
                "This single-log evaluation does not establish transfer to a different workflow "
                "domain. Cross-domain transfer remains an open experiment."
            ),
        ],
        extra={
            "dataset_sha256": log.sha256,
            "format": "XES",
            "source_activity_count": log.activity_count,
            "source_event_count": log.event_count,
            "selected_case_count": len(selected_cases),
            "selected_event_count": training_event_count + evaluation_event_count,
            "training_event_count": training_event_count,
            "evaluation_event_count": evaluation_event_count,
            "case_split": "chronological 80/20 by first event timestamp; most recent cases capped",
            "max_cases": max_cases,
            "attribute_keys_selected": list(attribute_keys),
            "owner_approval_required_for_runtime_storage": True,
            "external_data_downloaded_by_suite": False,
            "permutation_repeats": PERMUTATION_REPEATS,
            "gold_schema_labels_available": False,
            "source_case_ids_reported": False,
        },
    )
    return result
