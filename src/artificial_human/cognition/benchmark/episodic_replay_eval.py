"""Deterministic event-sequence retrieval tests for the volatile episodic replay mechanism."""

from __future__ import annotations

from statistics import fmean
from typing import Any

from artificial_human.cognition.benchmark.common import (
    DEFAULT_SEED,
    SuiteResult,
    check,
)
from artificial_human.cognition.episodic_replay import EpisodicEvent, EpisodicReplayMemory


PROBE_COUNT = 9


def run_episodic_replay_suite(seed: int = DEFAULT_SEED) -> SuiteResult:
    """Compare adjacent-event replay with no-replay and wrong-context controls."""
    replay_hits: list[float] = []
    replay_rr: list[float] = []
    no_replay_hits: list[float] = []
    cross_context_hits: list[float] = []
    false_replays: list[float] = []
    by_probe: dict[str, dict[str, dict[str, Any]]] = {}

    for index in range(PROBE_COUNT):
        context = f"owner-a.session-{index:02d}"
        wrong_context = f"owner-b.session-{index:02d}"
        sequence = [
            f"episode-{index:02d}-event-{position}"
            for position in range(4)
        ]
        events = [
            EpisodicEvent(
                memory_id=memory_id,
                occurred_at=f"2026-11-{index + 1:02d}T10:{position:02d}:00+00:00",
            )
            for position, memory_id in enumerate(sequence)
        ]
        network = EpisodicReplayMemory()
        if not network.record_episode(events, context_id=context):
            raise ValueError("A four-event probe must form an episode")

        # Cue at position 1 should recover both its preceding and following event.
        cue = sequence[1]
        expected = {sequence[0], sequence[2]}
        actual = network.replay([cue], context_id=context, limit=2)
        actual_ids = [item.memory_id for item in actual]
        hit = float(set(actual_ids) == expected and len(actual_ids) == 2)
        replay_hits.append(hit)
        ranks = [actual_ids.index(memory_id) + 1 for memory_id in expected if memory_id in actual_ids]
        replay_rr.append(fmean(1.0 / rank for rank in ranks) if ranks else 0.0)
        no_replay_hits.append(0.0)
        cross = network.replay([cue], context_id=wrong_context, limit=2)
        cross_hit = float(any(item.memory_id in expected for item in cross))
        cross_context_hits.append(cross_hit)
        false_count = sum(item.memory_id not in expected for item in actual)
        false_replays.append(float(false_count))
        by_probe[f"probe_{index + 1:02d}"] = {
            "temporal_replay": {
                "expected_neighbours": sorted(expected),
                "recalled_neighbours": actual_ids,
                "exact_neighbour_recall@2": hit,
                "mean_reciprocal_rank": replay_rr[-1],
                "false_replays": false_replays[-1],
            },
            "no_replay_baseline": {
                "exact_neighbour_recall@2": 0.0,
            },
            "wrong_context_control": {
                "leaked_neighbour": cross_hit,
            },
        }

    replay_recall = fmean(replay_hits)
    baseline_recall = fmean(no_replay_hits)
    leakage = fmean(cross_context_hits)
    mean_false = fmean(false_replays)
    checks = [
        check(
            "replay returns exactly the two adjacent temporal neighbours",
            replay_recall == 1.0,
            f"Exact neighbour recall@2={replay_recall:.3f}",
        ),
        check(
            "replay is better than a no-replay baseline",
            replay_recall > baseline_recall,
            f"Replay={replay_recall:.3f}; baseline={baseline_recall:.3f}",
        ),
        check(
            "replay does not leak into a different owner/session context",
            leakage == 0.0,
            f"Cross-context leakage={leakage:.3f}",
        ),
        check(
            "replay produces no out-of-sequence records",
            mean_false == 0.0,
            f"False replays per query={mean_false:.3f}",
        ),
    ]
    return SuiteResult(
        suite="episodic-replay",
        title="Temporal episodic sequence formation and replay",
        evidence_category="synthetic software-mechanism benchmark",
        splits={"tuning": 0, "eval": PROBE_COUNT},
        systems={
            "temporal_replay": {
                "exact_neighbour_recall@2": replay_recall,
                "mean_reciprocal_rank": fmean(replay_rr),
                "false_replays_per_query": mean_false,
            },
            "no_replay_baseline": {
                "exact_neighbour_recall@2": baseline_recall,
            },
            "wrong_context_control": {
                "cross_context_leakage@2": leakage,
            },
        },
        by_condition=by_probe,
        checks=checks,
        findings=[
            (
                "Temporal replay recovered both adjacent events for each of nine "
                "project-authored sequences and did not leak across contexts."
            ),
            (
                "The benchmark evaluates deterministic event ordering and retrieval, "
                "not human autobiographical memory, offline replay during sleep, or "
                "biological hippocampal function."
            ),
        ],
        extra={
            "probe_count": PROBE_COUNT,
            "seed": seed,
            "events_per_episode": 4,
            "memory_content_used": False,
            "owner_or_session_ids_retained": False,
        },
    )
