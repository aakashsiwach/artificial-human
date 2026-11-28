from __future__ import annotations

import asyncio

import pytest

from artificial_human.neural.organism_loop import Modality, SensoryEvent
from artificial_human.runtime import (
    ContinuousOrganismRuntime,
    RuntimeConfig,
    RuntimeHealth,
    RuntimeState,
)


def _event(timestamp: int = 0) -> SensoryEvent:
    return SensoryEvent(Modality.VISION, (0.2, 0.4, 0.6), timestamp, 0.9)


def test_manual_accelerated_ticks_are_deterministic_and_advance_neural_memory() -> None:
    first = ContinuousOrganismRuntime()
    second = ContinuousOrganismRuntime()

    first_trace = first.run_ticks(20)
    second_trace = second.run_ticks(20)

    assert [item.organism_steps for item in first_trace] == [
        item.organism_steps for item in second_trace
    ]
    assert first.snapshot().organism == second.snapshot().organism
    assert first.snapshot().tick_count == 20
    assert first.snapshot().organism["episodes"] == 20
    assert first.snapshot().organism["neural"]["simulation_step"] >= 40
    assert first.snapshot().organism["neural"]["total_spikes"] > 0


def test_event_queue_is_bounded_and_drop_newest_is_inspectable() -> None:
    runtime = ContinuousOrganismRuntime(
        config=RuntimeConfig(
            queue_capacity=2,
            max_events_per_tick=1,
            max_operations_per_tick=1,
        )
    )
    assert runtime.submit_event(_event(1))
    assert runtime.submit_event(_event(2))
    assert not runtime.submit_event(_event(3))

    before = runtime.snapshot()
    assert before.pending_events == 2
    assert before.dropped_events == 1
    assert before.health is RuntimeHealth.DEGRADED

    runtime.run_ticks(1)
    after = runtime.snapshot()
    assert after.processed_events == 1
    assert after.pending_events == 1


def test_tick_budget_caps_events_per_tick_and_preserves_backlog() -> None:
    runtime = ContinuousOrganismRuntime(
        config=RuntimeConfig(
            queue_capacity=8,
            max_events_per_tick=2,
            max_operations_per_tick=2,
        )
    )
    for index in range(5):
        assert runtime.submit_event(_event(index))

    first = runtime.tick()
    assert first.events_processed == 2
    assert first.pending_events == 3
    second = runtime.tick()
    assert second.events_processed == 2
    assert second.pending_events == 1
    third = runtime.tick()
    assert third.events_processed == 1
    assert third.pending_events == 0


def test_developmental_clock_is_independent_of_neural_simulation_ticks() -> None:
    runtime = ContinuousOrganismRuntime()
    before = runtime.snapshot().organism
    advanced = runtime.advance_developmental_time(postnatal_units=10)
    after_development = runtime.snapshot().organism

    assert advanced["postnatal_units"] == 510
    assert after_development["neural"]["simulation_step"] == before["neural"]["simulation_step"]
    runtime.run_ticks(1)
    after_neural = runtime.snapshot().organism
    assert after_neural["neural"]["simulation_step"] > after_development["neural"]["simulation_step"]
    assert after_neural["developmental_time"] == after_development["developmental_time"]


def test_long_run_resets_simulated_episodes_and_keeps_memory_bounded() -> None:
    runtime = ContinuousOrganismRuntime()
    runtime.run_ticks(350)
    snapshot = runtime.snapshot()

    assert snapshot.tick_count == 350
    assert snapshot.organism["episode_count"] >= 3
    assert snapshot.organism["episodes"] == 128
    assert snapshot.organism["neural"]["simulation_step"] >= 700
    assert snapshot.pending_events == 0
    assert len(snapshot.recent_trace) == 128
    assert snapshot.recent_trace[-1].tick_index == 350


def test_reset_clears_volatile_runtime_and_reinitializes_neural_program() -> None:
    runtime = ContinuousOrganismRuntime()
    runtime.submit_event(_event())
    runtime.run_ticks(5)
    runtime.reset()

    snapshot = runtime.snapshot()
    assert snapshot.state is RuntimeState.STOPPED
    assert snapshot.tick_count == 0
    assert snapshot.processed_events == 0
    assert snapshot.dropped_events == 0
    assert snapshot.pending_events == 0
    assert snapshot.organism["tick"] == 0
    assert snapshot.organism["episodes"] == 0
    assert snapshot.organism["neural"]["simulation_step"] == 0


def test_pause_resume_stop_and_async_scheduler_lifecycle() -> None:
    async def scenario() -> None:
        runtime = ContinuousOrganismRuntime(
            config=RuntimeConfig(tick_interval_seconds=0.01)
        )
        await runtime.start()
        await runtime.start()
        await asyncio.sleep(0.06)
        assert runtime.state is RuntimeState.RUNNING
        runtime.pause()
        paused_at = runtime.snapshot().tick_count
        await asyncio.sleep(0.05)
        assert runtime.snapshot().tick_count == paused_at
        runtime.resume()
        await asyncio.sleep(0.05)
        assert runtime.snapshot().tick_count > paused_at
        with pytest.raises(RuntimeError, match="stopped runtime"):
            runtime.run_ticks(1)
        await runtime.stop()
        assert runtime.state is RuntimeState.STOPPED
        await runtime.stop()
        runtime.reset()
        assert runtime.snapshot().tick_count == 0

    asyncio.run(scenario())


def test_failed_or_overloaded_runtime_has_explicit_health_state() -> None:
    runtime = ContinuousOrganismRuntime(
        config=RuntimeConfig(queue_capacity=1, max_events_per_tick=1, max_operations_per_tick=1)
    )
    assert runtime.submit_event(_event())
    assert not runtime.submit_event(_event(1))
    snapshot = runtime.snapshot()
    assert snapshot.health is RuntimeHealth.DEGRADED
    assert snapshot.overload_count == 1
    runtime.reset()
    assert runtime.snapshot().health is RuntimeHealth.STABLE


def test_runtime_config_rejects_unbounded_or_inconsistent_budgets() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        RuntimeConfig(queue_capacity=0)
    with pytest.raises(ValueError, match="cannot exceed"):
        RuntimeConfig(max_events_per_tick=9, max_operations_per_tick=8)
