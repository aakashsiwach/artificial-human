"""Deterministic event-driven leaky-integrate-and-fire simulation.

This is a deliberately small computational model, not a biophysical neuron
simulator. All topology and live state are held in bounded Python structures.
Simulation time is integer-stepped and never depends on wall-clock scheduling.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field, replace
from enum import StrEnum
from collections.abc import Mapping
from typing import ClassVar


class SimulationCapacityError(RuntimeError):
    """Raised when a configured neural-network capacity would be exceeded."""


class Polarity(StrEnum):
    EXCITATORY = "excitatory"
    INHIBITORY = "inhibitory"


@dataclass(frozen=True, slots=True)
class LIFParameters:
    resting_potential_mV: float = -65.0
    reset_potential_mV: float = -65.0
    threshold_mV: float = -50.0
    membrane_time_constant_ms: float = 20.0
    membrane_resistance: float = 1.0
    refractory_period_ms: float = 2.0

    def __post_init__(self) -> None:
        values = (
            self.resting_potential_mV, self.reset_potential_mV,
            self.threshold_mV, self.membrane_time_constant_ms,
            self.membrane_resistance, self.refractory_period_ms,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("neuron parameters must be finite")
        if self.membrane_time_constant_ms <= 0:
            raise ValueError("membrane_time_constant_ms must be positive")
        if self.membrane_resistance <= 0:
            raise ValueError("membrane_resistance must be positive")
        if self.refractory_period_ms < 0:
            raise ValueError("refractory_period_ms cannot be negative")
        if self.reset_potential_mV >= self.threshold_mV:
            raise ValueError("reset potential must be below threshold")
        if self.resting_potential_mV >= self.threshold_mV:
            raise ValueError("resting potential must be below threshold")


@dataclass(frozen=True, slots=True)
class NeuronSnapshot:
    identifier: str
    membrane_potential_mV: float
    spike_count: int
    refractory_until_ms: float
    last_spike_time_ms: float | None
    input_gain: float = 1.0


@dataclass(slots=True)
class LIFNeuron:
    """Leaky integrate-and-fire neuron with a bounded software input gain."""

    MIN_INPUT_GAIN: ClassVar[float] = 0.001
    MAX_INPUT_GAIN: ClassVar[float] = 1.0

    identifier: str
    parameters: LIFParameters = field(default_factory=LIFParameters)
    membrane_potential_mV: float | None = None
    spike_count: int = 0
    refractory_until_ms: float = 0.0
    last_spike_time_ms: float | None = None
    input_gain: float = 1.0
    _initial_input_gain: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.identifier or not self.identifier.strip():
            raise ValueError("neuron identifier is required")
        if self.membrane_potential_mV is None:
            self.membrane_potential_mV = self.parameters.resting_potential_mV
        if not math.isfinite(self.potential_mV):
            raise ValueError("membrane potential must be finite")
        self.set_input_gain(self.input_gain)
        self._initial_input_gain = self.input_gain

    def set_input_gain(self, gain: float) -> None:
        """Set a bounded dimensionless input-current gain for homeostatic control."""
        if (
            isinstance(gain, bool)
            or not math.isfinite(gain)
            or not self.MIN_INPUT_GAIN <= gain <= self.MAX_INPUT_GAIN
        ):
            raise ValueError("input_gain must be finite and in [0.001, 1.0]")
        self.input_gain = gain

    def update(self, current: float, *, time_ms: float, dt_ms: float) -> bool:
        """Advance one step and return True exactly when this step emits a spike."""
        if not math.isfinite(current) or not math.isfinite(time_ms):
            raise ValueError("current and simulation time must be finite")
        if not math.isfinite(dt_ms) or dt_ms <= 0:
            raise ValueError("dt_ms must be finite and positive")
        if (
            isinstance(self.input_gain, bool)
            or not math.isfinite(self.input_gain)
            or not self.MIN_INPUT_GAIN <= self.input_gain <= self.MAX_INPUT_GAIN
        ):
            raise ValueError("input_gain is outside configured bounds")
        if time_ms + 1e-12 < self.refractory_until_ms:
            self.membrane_potential_mV = self.parameters.reset_potential_mV
            return False

        decay = math.exp(-dt_ms / self.parameters.membrane_time_constant_ms)
        potential = (
            self.parameters.resting_potential_mV
            + (self.potential_mV - self.parameters.resting_potential_mV) * decay
            + self.parameters.membrane_resistance * (current * self.input_gain) * (1.0 - decay)
        )
        if not math.isfinite(potential):
            raise ArithmeticError("neuron membrane update became non-finite")
        if potential >= self.parameters.threshold_mV:
            self.membrane_potential_mV = self.parameters.reset_potential_mV
            self.spike_count += 1
            self.last_spike_time_ms = time_ms
            self.refractory_until_ms = time_ms + self.parameters.refractory_period_ms
            return True
        self.membrane_potential_mV = potential
        return False

    @property
    def potential_mV(self) -> float:
        assert self.membrane_potential_mV is not None
        return self.membrane_potential_mV

    def snapshot(self) -> NeuronSnapshot:
        return NeuronSnapshot(
            identifier=self.identifier,
            membrane_potential_mV=round(self.potential_mV, 8),
            spike_count=self.spike_count,
            refractory_until_ms=round(self.refractory_until_ms, 8),
            last_spike_time_ms=(
                round(self.last_spike_time_ms, 8)
                if self.last_spike_time_ms is not None else None
            ),
            input_gain=round(self.input_gain, 12),
        )

    def reset(self) -> None:
        self.membrane_potential_mV = self.parameters.resting_potential_mV
        self.spike_count = 0
        self.refractory_until_ms = 0.0
        self.last_spike_time_ms = None
        self.input_gain = self._initial_input_gain


@dataclass(frozen=True, slots=True)
class Synapse:
    identifier: str
    source: str
    target: str
    weight: float
    polarity: Polarity = Polarity.EXCITATORY
    delay_steps: int = 1
    time_constant_ms: float = 5.0
    maximum_weight: float = 250.0

    def __post_init__(self) -> None:
        if not self.identifier.strip() or not self.source.strip() or not self.target.strip():
            raise ValueError("synapse, source and target identifiers are required")
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("synaptic weight must be finite and non-negative")
        if not math.isfinite(self.maximum_weight) or self.maximum_weight <= 0:
            raise ValueError("maximum_weight must be finite and positive")
        if self.weight > self.maximum_weight:
            raise ValueError("synaptic weight exceeds maximum_weight")
        if (
            isinstance(self.delay_steps, bool)
            or not isinstance(self.delay_steps, int)
            or self.delay_steps < 1
        ):
            raise ValueError("delay_steps must be an integer of at least one")
        if not isinstance(self.polarity, Polarity):
            raise ValueError("polarity must be an excitatory or inhibitory Polarity")
        if not math.isfinite(self.time_constant_ms) or self.time_constant_ms <= 0:
            raise ValueError("time_constant_ms must be finite and positive")

    @property
    def signed_weight(self) -> float:
        return self.weight if self.polarity is Polarity.EXCITATORY else -self.weight


@dataclass(frozen=True, slots=True)
class SimulationStep:
    step_index: int
    time_ms: float
    spikes: tuple[str, ...]
    membrane_potentials: tuple[tuple[str, float], ...]
    pending_events: int
    dropped_events: int


@dataclass(frozen=True, slots=True)
class NeuralSimulationState:
    """Complete bounded state snapshot for restore into the same source topology."""

    version: str
    dt_ms: float
    neuron_ids: tuple[str, ...]
    synapse_ids: tuple[str, ...]
    step_index: int
    paused: bool
    neuron_states: tuple[NeuronSnapshot, ...]
    synaptic_currents: tuple[tuple[str, float], ...]
    synaptic_weights: tuple[tuple[str, float], ...]
    pending_events: tuple[tuple[int, int, str, str], ...]
    sequence: int
    events_scheduled: int
    events_delivered: int
    events_dropped: int
    events_cancelled: int


@dataclass(frozen=True, slots=True)
class SimulationSnapshot:
    version: str
    step_index: int
    time_ms: float
    paused: bool
    neuron_count: int
    synapse_count: int
    pending_events: int
    events_scheduled: int
    events_delivered: int
    events_dropped: int
    events_cancelled: int
    queue_capacity: int
    neuron_states: tuple[NeuronSnapshot, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "step_index": self.step_index,
            "time_ms": self.time_ms,
            "paused": self.paused,
            "neuron_count": self.neuron_count,
            "synapse_count": self.synapse_count,
            "pending_events": self.pending_events,
            "events_scheduled": self.events_scheduled,
            "events_delivered": self.events_delivered,
            "events_dropped": self.events_dropped,
            "events_cancelled": self.events_cancelled,
            "queue_capacity": self.queue_capacity,
            "neuron_states": [
                {
                    "identifier": state.identifier,
                    "membrane_potential_mV": state.membrane_potential_mV,
                    "spike_count": state.spike_count,
                    "refractory_until_ms": state.refractory_until_ms,
                    "last_spike_time_ms": state.last_spike_time_ms,
                    "input_gain": state.input_gain,
                }
                for state in self.neuron_states
            ],
        }


class NeuralSimulation:
    """Bounded deterministic network with delayed spike events and synaptic currents."""

    VERSION = "lif-event-simulation-v2"

    def __init__(
        self,
        *,
        dt_ms: float = 1.0,
        max_neurons: int = 10_000,
        max_synapses: int = 100_000,
        max_pending_events: int = 100_000,
        max_abs_external_current: float = 1_000.0,
    ) -> None:
        if not math.isfinite(dt_ms) or dt_ms <= 0:
            raise ValueError("dt_ms must be finite and positive")
        if min(max_neurons, max_synapses, max_pending_events) < 1:
            raise ValueError("simulation capacities must be positive")
        if not math.isfinite(max_abs_external_current) or max_abs_external_current <= 0:
            raise ValueError("max_abs_external_current must be finite and positive")
        self.dt_ms = dt_ms
        self.max_neurons = max_neurons
        self.max_synapses = max_synapses
        self.max_pending_events = max_pending_events
        self.max_abs_external_current = max_abs_external_current
        self.max_abs_synaptic_current = 1_000.0
        self.paused = False
        self.neurons: dict[str, LIFNeuron] = {}
        self.synapses: dict[str, Synapse] = {}
        self._outgoing: dict[str, list[str]] = {}
        self._queue: list[tuple[int, int, str, str]] = []
        self._sequence = 0
        self._synaptic_current: dict[str, float] = {}
        self._initial_weights: dict[str, float] = {}
        self.step_index = 0
        self.events_scheduled = 0
        self.events_delivered = 0
        self.events_dropped = 0
        self.events_cancelled = 0

    @property
    def time_ms(self) -> float:
        return self.step_index * self.dt_ms

    def add_neuron(self, neuron: LIFNeuron) -> None:
        if neuron.identifier in self.neurons:
            raise ValueError(f"duplicate neuron: {neuron.identifier}")
        if len(self.neurons) >= self.max_neurons:
            raise SimulationCapacityError("neuron capacity reached")
        self.neurons[neuron.identifier] = neuron
        self._outgoing[neuron.identifier] = []

    def add_synapse(self, synapse: Synapse) -> None:
        if synapse.identifier in self.synapses:
            raise ValueError(f"duplicate synapse: {synapse.identifier}")
        if len(self.synapses) >= self.max_synapses:
            raise SimulationCapacityError("synapse capacity reached")
        if synapse.source not in self.neurons or synapse.target not in self.neurons:
            raise ValueError("synapse endpoints must exist before connecting")
        self.synapses[synapse.identifier] = synapse
        self._outgoing[synapse.source].append(synapse.identifier)
        self._outgoing[synapse.source].sort()
        self._synaptic_current[synapse.identifier] = 0.0
        self._initial_weights[synapse.identifier] = synapse.weight

    def update_synapse_weight(self, synapse_id: str, weight: float) -> None:
        """Set a synaptic magnitude while preserving its polarity and bounds."""
        synapse = self.synapses.get(synapse_id)
        if synapse is None:
            raise ValueError(f"unknown synapse: {synapse_id}")
        if isinstance(weight, bool) or not math.isfinite(weight):
            raise ValueError("synaptic weight must be finite")
        if not 0 <= weight <= synapse.maximum_weight:
            raise ValueError("synaptic weight is outside configured bounds")
        self.synapses[synapse_id] = replace(synapse, weight=weight)

    def update_synapse_delay(self, synapse_id: str, delay_steps: int) -> None:
        """Change delay for future spikes; already queued events keep their due time."""
        synapse = self.synapses.get(synapse_id)
        if synapse is None:
            raise ValueError(f"unknown synapse: {synapse_id}")
        if (
            isinstance(delay_steps, bool)
            or not isinstance(delay_steps, int)
            or not 1 <= delay_steps <= 10000
        ):
            raise ValueError("delay_steps must be an integer in [1, 10000]")
        self.synapses[synapse_id] = replace(synapse, delay_steps=delay_steps)

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def cancel_pending_events(
        self, *, synapse_id: str | None = None, target_id: str | None = None
    ) -> int:
        """Cancel queued events by synapse and/or target, returning the count removed.

        With no filters, all pending events are cancelled. Delivered events and
        already-applied synaptic currents are unaffected.
        """
        if synapse_id is not None and synapse_id not in self.synapses:
            raise ValueError(f"unknown synapse: {synapse_id}")
        if target_id is not None and target_id not in self.neurons:
            raise ValueError(f"unknown target neuron: {target_id}")
        retained = [
            event for event in self._queue
            if not (
                (synapse_id is None or event[2] == synapse_id)
                and (target_id is None or event[3] == target_id)
            )
        ]
        cancelled = len(self._queue) - len(retained)
        self._queue = retained
        heapq.heapify(self._queue)
        self.events_cancelled += cancelled
        return cancelled

    def step(self, external_currents: Mapping[str, float] | None = None) -> SimulationStep:
        """Advance unless paused; the simulation clock is independent of wall time."""
        if self.paused:
            return SimulationStep(
                step_index=self.step_index,
                time_ms=round(self.time_ms, 8),
                spikes=(),
                membrane_potentials=tuple(
                    (key, round(self.neurons[key].potential_mV, 8))
                    for key in sorted(self.neurons)
                ),
                pending_events=len(self._queue),
                dropped_events=self.events_dropped,
            )
        return self.single_step(external_currents)

    def single_step(self, external_currents: Mapping[str, float] | None = None) -> SimulationStep:
        """Advance exactly one step even when paused, for deterministic debugging."""
        return self._advance(external_currents)

    def _advance(self, external_currents: Mapping[str, float] | None = None) -> SimulationStep:
        """Advance one deterministic step; overload drops newest events explicitly."""
        external = dict(external_currents or {})
        for identifier, current in external.items():
            if identifier not in self.neurons:
                raise ValueError(f"unknown neuron in external current map: {identifier}")
            if (
                isinstance(current, bool)
                or not math.isfinite(current)
                or abs(current) > self.max_abs_external_current
            ):
                raise ValueError("external current is non-finite or exceeds configured bound")

        # Decay all active synaptic currents before delivering events for this step.
        for synapse_id, current in tuple(self._synaptic_current.items()):
            if current:
                tau = self.synapses[synapse_id].time_constant_ms
                self._synaptic_current[synapse_id] = current * math.exp(-self.dt_ms / tau)

        while self._queue and self._queue[0][0] <= self.step_index:
            _, _, synapse_id, target_id = heapq.heappop(self._queue)
            self._synaptic_current[synapse_id] = max(
                -self.max_abs_synaptic_current,
                min(
                    self.max_abs_synaptic_current,
                    self._synaptic_current[synapse_id]
                    + self.synapses[synapse_id].signed_weight,
                ),
            )
            self.events_delivered += 1

        total_current = {identifier: external.get(identifier, 0.0) for identifier in self.neurons}
        for synapse_id, current in self._synaptic_current.items():
            if current:
                target = self.synapses[synapse_id].target
                total_current[target] = max(
                    -self.max_abs_external_current,
                    min(self.max_abs_external_current, total_current[target] + current),
                )

        spikes: list[str] = []
        current_time = self.time_ms
        # All neurons read the same pre-step synaptic state. Newly emitted spikes
        # are delivered no earlier than the next simulation step.
        for identifier in sorted(self.neurons):
            if self.neurons[identifier].update(
                total_current[identifier], time_ms=current_time, dt_ms=self.dt_ms
            ):
                spikes.append(identifier)

        for source in spikes:
            for synapse_id in self._outgoing[source]:
                synapse = self.synapses[synapse_id]
                if len(self._queue) >= self.max_pending_events:
                    self.events_dropped += 1
                    continue
                event = (
                    self.step_index + synapse.delay_steps,
                    self._sequence,
                    synapse_id,
                    synapse.target,
                )
                self._sequence += 1
                heapq.heappush(self._queue, event)
                self.events_scheduled += 1

        result = SimulationStep(
            step_index=self.step_index,
            time_ms=round(current_time, 8),
            spikes=tuple(spikes),
            membrane_potentials=tuple(
                (identifier, round(self.neurons[identifier].potential_mV, 8))
                for identifier in sorted(self.neurons)
            ),
            pending_events=len(self._queue),
            dropped_events=self.events_dropped,
        )
        self.step_index += 1
        return result

    def run(
        self, steps: int, external_currents: Mapping[str, float] | None = None
    ) -> tuple[SimulationStep, ...]:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        if steps > 1_000_000:
            raise ValueError("steps exceeds the configured fast-forward limit")
        return tuple(self.step(external_currents) for _ in range(steps))

    def snapshot(self) -> SimulationSnapshot:
        return SimulationSnapshot(
            version=self.VERSION,
            step_index=self.step_index,
            time_ms=round(self.time_ms, 8),
            paused=self.paused,
            neuron_count=len(self.neurons),
            synapse_count=len(self.synapses),
            pending_events=len(self._queue),
            events_scheduled=self.events_scheduled,
            events_delivered=self.events_delivered,
            events_dropped=self.events_dropped,
            events_cancelled=self.events_cancelled,
            queue_capacity=self.max_pending_events,
            neuron_states=tuple(self.neurons[key].snapshot() for key in sorted(self.neurons)),
        )

    def state(self) -> NeuralSimulationState:
        """Capture complete live state without serialising it to disk."""
        return NeuralSimulationState(
            version=self.VERSION,
            dt_ms=self.dt_ms,
            neuron_ids=tuple(sorted(self.neurons)),
            synapse_ids=tuple(sorted(self.synapses)),
            step_index=self.step_index,
            paused=self.paused,
            neuron_states=tuple(self.neurons[key].snapshot() for key in sorted(self.neurons)),
            synaptic_currents=tuple(sorted(
                (key, round(value, 12)) for key, value in self._synaptic_current.items()
            )),
            synaptic_weights=tuple(sorted(
                (key, self.synapses[key].weight) for key in self.synapses
            )),
            pending_events=tuple(sorted(self._queue)),
            sequence=self._sequence,
            events_scheduled=self.events_scheduled,
            events_delivered=self.events_delivered,
            events_dropped=self.events_dropped,
            events_cancelled=self.events_cancelled,
        )

    def restore(self, state: NeuralSimulationState) -> None:
        """Restore an in-memory snapshot only into an identical fixed topology."""
        if state.version != self.VERSION or state.dt_ms != self.dt_ms:
            raise ValueError("snapshot version or simulation timestep does not match")
        if state.neuron_ids != tuple(sorted(self.neurons)):
            raise ValueError("snapshot neuron topology does not match")
        if state.synapse_ids != tuple(sorted(self.synapses)):
            raise ValueError("snapshot synapse topology does not match")
        if len(state.pending_events) > self.max_pending_events:
            raise SimulationCapacityError("snapshot exceeds pending-event capacity")
        if state.step_index < 0 or state.sequence < 0:
            raise ValueError("snapshot clock and sequence values must be non-negative")
        if any(value < 0 for value in (
            state.events_scheduled, state.events_delivered, state.events_dropped,
            state.events_cancelled,
        )):
            raise ValueError("snapshot event counters must be non-negative")
        by_id = {item.identifier: item for item in state.neuron_states}
        if set(by_id) != set(self.neurons):
            raise ValueError("snapshot neuron states do not match topology")
        for snapshot in by_id.values():
            if (
                not math.isfinite(snapshot.membrane_potential_mV)
                or not math.isfinite(snapshot.refractory_until_ms)
                or (
                    snapshot.last_spike_time_ms is not None
                    and not math.isfinite(snapshot.last_spike_time_ms)
                )
                or snapshot.spike_count < 0
                or not math.isfinite(snapshot.input_gain)
                or not LIFNeuron.MIN_INPUT_GAIN <= snapshot.input_gain <= LIFNeuron.MAX_INPUT_GAIN
            ):
                raise ValueError("snapshot contains invalid neuron state")
        weights = dict(state.synaptic_weights)
        if set(weights) != set(self.synapses) or any(
            not math.isfinite(value) or not 0 <= value <= self.synapses[key].maximum_weight
            for key, value in weights.items()
        ):
            raise ValueError("snapshot contains invalid synaptic weights")
        currents = dict(state.synaptic_currents)
        if set(currents) != set(self.synapses) or any(
            not math.isfinite(value) or abs(value) > self.max_abs_synaptic_current
            for value in currents.values()
        ):
            raise ValueError("snapshot contains invalid synaptic current state")
        for event_step, order, synapse_id, target_id in state.pending_events:
            synapse = self.synapses.get(synapse_id)
            if (
                event_step < state.step_index or order < 0 or order >= state.sequence
                or synapse is None or target_id != synapse.target
            ):
                raise ValueError("snapshot contains an invalid pending event")
        self.step_index = state.step_index
        self.paused = state.paused
        self._sequence = state.sequence
        self.events_scheduled = state.events_scheduled
        self.events_delivered = state.events_delivered
        self.events_dropped = state.events_dropped
        self.events_cancelled = state.events_cancelled
        for identifier, snapshot in by_id.items():
            neuron = self.neurons[identifier]
            neuron.membrane_potential_mV = snapshot.membrane_potential_mV
            neuron.spike_count = snapshot.spike_count
            neuron.refractory_until_ms = snapshot.refractory_until_ms
            neuron.last_spike_time_ms = snapshot.last_spike_time_ms
            neuron.input_gain = snapshot.input_gain
        self._synaptic_current = currents
        for synapse_id, weight in weights.items():
            self.update_synapse_weight(synapse_id, weight)
        self._queue = list(state.pending_events)
        heapq.heapify(self._queue)

    def reset(self) -> None:
        self.paused = False
        for neuron in self.neurons.values():
            neuron.reset()
        self._queue.clear()
        for synapse_id, weight in self._initial_weights.items():
            self.update_synapse_weight(synapse_id, weight)
        for synapse_id in self._synaptic_current:
            self._synaptic_current[synapse_id] = 0.0
        self._sequence = 0
        self.step_index = 0
        self.events_scheduled = 0
        self.events_delivered = 0
        self.events_dropped = 0
        self.events_cancelled = 0
