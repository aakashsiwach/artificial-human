"""Continuous, bounded in-memory runtime for the simulated organism.

The runtime owns one OrganismLoop and one developmental LIF substrate. Neural
simulation time advances on deterministic ticks; developmental time advances
only through explicit commands; wall-clock scheduling is an optional adapter.
No database, LLM, device, network call or physical actuator is used here.
"""
from __future__ import annotations

import asyncio
import math
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from queue import Empty, Full, Queue
from threading import RLock
from time import monotonic

from artificial_human.metabolism.governor import (
    BudgetExceeded,
    BudgetLimits,
    BudgetSnapshot,
    ResourceGovernor,
)
from artificial_human.neural.organism_loop import OrganismLoop, SensoryEvent


class RuntimeState(StrEnum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    FAILED = "failed"


class RuntimeHealth(StrEnum):
    STABLE = "stable"
    DEGRADED = "degraded"


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    tick_interval_seconds: float = 0.05
    queue_capacity: int = 128
    max_events_per_tick: int = 8
    max_ticks_per_batch: int = 10_000
    max_operations_per_tick: int = 8
    tick_budget_seconds: float = 0.25
    max_spikes_per_tick: int = 256

    def __post_init__(self) -> None:
        if (
            isinstance(self.tick_interval_seconds, bool)
            or not math.isfinite(self.tick_interval_seconds)
            or self.tick_interval_seconds <= 0
        ):
            raise ValueError("tick_interval_seconds must be finite and positive")
        if (
            isinstance(self.tick_budget_seconds, bool)
            or not math.isfinite(self.tick_budget_seconds)
            or self.tick_budget_seconds <= 0
        ):
            raise ValueError("tick_budget_seconds must be finite and positive")
        for name in (
            "queue_capacity",
            "max_events_per_tick",
            "max_ticks_per_batch",
            "max_operations_per_tick",
            "max_spikes_per_tick",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_events_per_tick > self.max_operations_per_tick:
            raise ValueError("max_events_per_tick cannot exceed max_operations_per_tick")


@dataclass(frozen=True, slots=True)
class RuntimeTick:
    tick_index: int
    organism_steps: tuple[dict[str, object], ...]
    events_processed: int
    pending_events: int
    elapsed_seconds: float
    health: RuntimeHealth
    budget: BudgetSnapshot

    def as_dict(self) -> dict[str, object]:
        return {
            "tick_index": self.tick_index,
            "organism_steps": list(self.organism_steps),
            "events_processed": self.events_processed,
            "pending_events": self.pending_events,
            "elapsed_seconds": self.elapsed_seconds,
            "health": self.health.value,
            "budget": self.budget.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    state: RuntimeState
    health: RuntimeHealth
    tick_count: int
    processed_events: int
    dropped_events: int
    overload_count: int
    pending_events: int
    last_error: str | None
    organism: dict[str, object]
    recent_trace: tuple[RuntimeTick, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "health": self.health.value,
            "tick_count": self.tick_count,
            "processed_events": self.processed_events,
            "dropped_events": self.dropped_events,
            "overload_count": self.overload_count,
            "pending_events": self.pending_events,
            "last_error": self.last_error,
            "organism": self.organism,
            "recent_trace": [item.as_dict() for item in self.recent_trace],
        }


class ContinuousOrganismRuntime:
    """Serialises deterministic organism ticks and optional wall-clock running.

    Events use a bounded drop-newest queue. Manual `run_ticks` is deterministic
    and accelerated; `start` adds a wall-clock scheduler without changing the
    simulation or developmental clocks' units.
    """

    def __init__(
        self,
        organism: OrganismLoop | None = None,
        *,
        config: RuntimeConfig | None = None,
    ) -> None:
        self.organism = organism or OrganismLoop()
        self.config = config or RuntimeConfig()
        self._events: Queue[SensoryEvent] = Queue(maxsize=self.config.queue_capacity)
        self._lock = RLock()
        self._state = RuntimeState.STOPPED
        self._health = RuntimeHealth.STABLE
        self._task: asyncio.Task[None] | None = None
        self._tick_count = 0
        self._processed_events = 0
        self._dropped_events = 0
        self._overload_count = 0
        self._last_error: str | None = None
        self._trace: deque[RuntimeTick] = deque(maxlen=128)
        self._last_budget = BudgetSnapshot(
            BudgetLimits(
                operations=self.config.max_operations_per_tick,
                tokens=0,
                wall_seconds=self.config.tick_budget_seconds,
            ),
            0,
            0,
            0.0,
        )

    @property
    def state(self) -> RuntimeState:
        with self._lock:
            return self._state

    def submit_event(self, event: SensoryEvent) -> bool:
        """Queue an explicit synthetic event; return False on overload."""
        if not isinstance(event, SensoryEvent):
            raise TypeError("event must be a SensoryEvent")
        with self._lock:
            if self._state in {RuntimeState.STOPPING, RuntimeState.FAILED}:
                return False
            try:
                self._events.put_nowait(event)
            except Full:
                self._dropped_events += 1
                self._overload_count += 1
                self._health = RuntimeHealth.DEGRADED
                return False
            return True

    def tick(self) -> RuntimeTick:
        """Execute one bounded scheduler tick without sleeping."""
        with self._lock:
            if self._state in {
                RuntimeState.PAUSED,
                RuntimeState.STOPPING,
                RuntimeState.FAILED,
            }:
                return self._empty_tick()

            started = monotonic()
            governor = ResourceGovernor(
                BudgetLimits(
                    operations=self.config.max_operations_per_tick,
                    tokens=0,
                    wall_seconds=self.config.tick_budget_seconds,
                )
            )
            traces: list[dict[str, object]] = []
            events_processed = 0
            work_units = 0
            try:
                while work_units < self.config.max_events_per_tick:
                    governor.consume(operations=1)
                    work_units += 1
                    try:
                        event = self._events.get_nowait()
                    except Empty:
                        if not traces:
                            traces.append(self.organism.tick())
                        break
                    traces.append(self.organism.tick(event))
                    events_processed += 1
                    self._processed_events += 1
            except BudgetExceeded:
                self._overload_count += 1
                self._health = RuntimeHealth.DEGRADED
            except Exception as error:
                self._state = RuntimeState.FAILED
                self._health = RuntimeHealth.DEGRADED
                self._last_error = f"{type(error).__name__}: {error}"
                raise

            elapsed = monotonic() - started
            if elapsed > self.config.tick_budget_seconds:
                self._overload_count += 1
                self._health = RuntimeHealth.DEGRADED
            spike_count = 0
            for step in traces:
                neural = step.get("neural")
                if isinstance(neural, dict):
                    spikes = neural.get("spikes")
                    if isinstance(spikes, (tuple, list)):
                        spike_count += len(spikes)
            if spike_count > self.config.max_spikes_per_tick:
                self._overload_count += 1
                self._health = RuntimeHealth.DEGRADED
            self._tick_count += 1
            self._last_budget = governor.snapshot()
            result = RuntimeTick(
                tick_index=self._tick_count,
                organism_steps=tuple(traces),
                events_processed=events_processed,
                pending_events=self._events.qsize(),
                elapsed_seconds=round(elapsed, 6),
                health=self._health,
                budget=self._last_budget,
            )
            self._trace.append(result)
            return result

    def run_ticks(self, ticks: int) -> tuple[RuntimeTick, ...]:
        """Run deterministic accelerated ticks; no wall-clock sleep is used."""
        if isinstance(ticks, bool) or not isinstance(ticks, int):
            raise ValueError("ticks must be an integer")
        if not 0 <= ticks <= self.config.max_ticks_per_batch:
            raise ValueError(
                f"ticks must be in [0, {self.config.max_ticks_per_batch}]"
            )
        if self.state is not RuntimeState.STOPPED:
            raise RuntimeError("manual run_ticks requires a stopped runtime")
        return tuple(self.tick() for _ in range(ticks))

    def advance_developmental_time(
        self, *, prenatal_days: int = 0, postnatal_units: int = 0
    ) -> dict[str, object]:
        """Advance developmental time explicitly, independently of neural ticks."""
        if (
            isinstance(prenatal_days, bool)
            or not isinstance(prenatal_days, int)
            or prenatal_days < 0
            or isinstance(postnatal_units, bool)
            or not isinstance(postnatal_units, int)
            or postnatal_units < 0
            or (prenatal_days and postnatal_units)
        ):
            raise ValueError("supply one non-negative developmental-time increment")
        with self._lock:
            program = self.organism.neural_development
            if prenatal_days:
                snapshot = program.advance_prenatal(prenatal_days)
            elif postnatal_units:
                snapshot = program.advance_postnatal(postnatal_units)
            else:
                snapshot = program.snapshot()
            return {
                "stage": snapshot.stage.value,
                "prenatal_days": snapshot.prenatal_days,
                "postnatal_units": snapshot.postnatal_units,
            }

    async def start(self) -> None:
        with self._lock:
            if self._state is RuntimeState.RUNNING:
                return
            if self._state is RuntimeState.PAUSED:
                if self._task is not None:
                    self._state = RuntimeState.RUNNING
                    self.organism.resume()
                    return
                self._state = RuntimeState.STOPPED
            if self._state is not RuntimeState.STOPPED:
                raise RuntimeError(f"cannot start runtime from {self._state.value}")
            self._state = RuntimeState.RUNNING
            self.organism.resume()
            self._task = asyncio.create_task(
                self._run_loop(), name="artificial-human-continuous-runtime"
            )

    async def _run_loop(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time()
        try:
            while True:
                with self._lock:
                    state = self._state
                if state in {RuntimeState.STOPPING, RuntimeState.STOPPED, RuntimeState.FAILED}:
                    return
                if state is RuntimeState.RUNNING:
                    self.tick()
                deadline += self.config.tick_interval_seconds
                delay = deadline - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                else:
                    deadline = loop.time()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            with self._lock:
                self._state = RuntimeState.FAILED
                self._health = RuntimeHealth.DEGRADED
                self._last_error = f"{type(error).__name__}: {error}"

    def pause(self) -> None:
        with self._lock:
            if self._state is RuntimeState.RUNNING:
                self._state = RuntimeState.PAUSED
                self.organism.pause()

    def resume(self) -> None:
        with self._lock:
            if self._state is RuntimeState.PAUSED:
                self.organism.resume()
                self._state = (
                    RuntimeState.RUNNING if self._task is not None else RuntimeState.STOPPED
                )

    async def stop(self) -> None:
        with self._lock:
            task = self._task
            if task is None:
                if self._state is not RuntimeState.FAILED:
                    self._state = RuntimeState.STOPPED
                self.organism.resume()
                return
            self._state = RuntimeState.STOPPING
            task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        with self._lock:
            self._task = None
            self._state = RuntimeState.STOPPED
            self.organism.resume()

    def reset(self) -> None:
        with self._lock:
            if self._task is not None or self._state is RuntimeState.RUNNING:
                raise RuntimeError("stop the background runtime before reset")
            while True:
                try:
                    self._events.get_nowait()
                except Empty:
                    break
            self.organism.reset()
            self._state = RuntimeState.STOPPED
            self._health = RuntimeHealth.STABLE
            self._tick_count = 0
            self._processed_events = 0
            self._dropped_events = 0
            self._overload_count = 0
            self._last_error = None
            self._trace.clear()
            self._last_budget = BudgetSnapshot(
                BudgetLimits(
                    operations=self.config.max_operations_per_tick,
                    tokens=0,
                    wall_seconds=self.config.tick_budget_seconds,
                ),
                0,
                0,
                0.0,
            )

    def snapshot(self) -> RuntimeSnapshot:
        with self._lock:
            return RuntimeSnapshot(
                state=self._state,
                health=self._health,
                tick_count=self._tick_count,
                processed_events=self._processed_events,
                dropped_events=self._dropped_events,
                overload_count=self._overload_count,
                pending_events=self._events.qsize(),
                last_error=self._last_error,
                organism=self.organism.snapshot(),
                recent_trace=tuple(self._trace),
            )

    def _empty_tick(self) -> RuntimeTick:
        return RuntimeTick(
            tick_index=self._tick_count,
            organism_steps=(),
            events_processed=0,
            pending_events=self._events.qsize(),
            elapsed_seconds=0.0,
            health=self._health,
            budget=self._last_budget,
        )
