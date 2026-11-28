"""Bounded deterministic homeostatic control for the LIF simulation.

This is a reduced feedback controller inspired by activity homeostasis. Its
input gain is an engineering control variable, not a molecular neuron model.
All regulator state remains in RAM and is snapshotable for deterministic replay.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from artificial_human.neural.simulation import (
    LIFNeuron,
    NeuralSimulation,
    SimulationStep,
)


@dataclass(frozen=True, slots=True)
class InternalModulatorySignals:
    """Distinct inspectable channels used to derive a plasticity magnitude gate.

    These are computational signals, not neurotransmitter concentrations or
    claims about dopamine, subjective emotion, or any specific brain nucleus.
    """

    salience: float = 0.0
    reward_prediction_error: float = 0.0
    stress_load: float = 0.0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.salience)
            or not 0.0 <= self.salience <= 1.0
        ):
            raise ValueError("salience must be finite and in [0, 1]")
        if (
            not math.isfinite(self.reward_prediction_error)
            or not -1.0 <= self.reward_prediction_error <= 1.0
        ):
            raise ValueError("reward_prediction_error must be finite and in [-1, 1]")
        if (
            not math.isfinite(self.stress_load)
            or not 0.0 <= self.stress_load <= 1.0
        ):
            raise ValueError("stress_load must be finite and in [0, 1]")

    @property
    def plasticity_gate(self) -> float:
        """Return a deterministic magnitude gate without collapsing its sources."""
        return (
            self.salience
            * abs(self.reward_prediction_error)
            * (1.0 - self.stress_load)
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "salience": self.salience,
            "reward_prediction_error": self.reward_prediction_error,
            "stress_load": self.stress_load,
            "plasticity_gate": self.plasticity_gate,
        }


@dataclass(frozen=True, slots=True)
class HomeostasisParameters:
    """Parameters for a slow per-neuron rate-feedback controller.

    Rates are spikes per simulation step, not hertz. Each observation window
    contains a fixed number of simulation steps.
    """

    observation_window_steps: int = 20
    target_spike_rate: float = 0.08
    rate_ema_decay: float = 0.5
    adaptation_rate: float = 4.0
    deadband: float = 0.02
    minimum_input_gain: float = 0.01
    maximum_input_gain: float = 1.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.observation_window_steps, bool)
            or not isinstance(self.observation_window_steps, int)
            or not 1 <= self.observation_window_steps <= 100_000
        ):
            raise ValueError("observation_window_steps must be an integer in [1, 100000]")
        if (
            not math.isfinite(self.target_spike_rate)
            or not 0.0 <= self.target_spike_rate <= 1.0
        ):
            raise ValueError("target_spike_rate must be finite and in [0, 1]")
        if (
            not math.isfinite(self.rate_ema_decay)
            or not 0.0 <= self.rate_ema_decay < 1.0
        ):
            raise ValueError("rate_ema_decay must be finite and in [0, 1)")
        if (
            not math.isfinite(self.adaptation_rate)
            or not 0.0 <= self.adaptation_rate <= 10.0
        ):
            raise ValueError("adaptation_rate must be finite and in [0, 10]")
        if (
            not math.isfinite(self.deadband)
            or not 0.0 <= self.deadband < 1.0
        ):
            raise ValueError("deadband must be finite and in [0, 1)")
        if (
            not math.isfinite(self.minimum_input_gain)
            or not math.isfinite(self.maximum_input_gain)
            or self.minimum_input_gain < LIFNeuron.MIN_INPUT_GAIN
            or self.maximum_input_gain > LIFNeuron.MAX_INPUT_GAIN
            or self.minimum_input_gain > self.maximum_input_gain
        ):
            raise ValueError("input gain bounds must be within [0.001, 1.0]")


@dataclass(frozen=True, slots=True)
class NeuronHomeostaticState:
    identifier: str
    observed_spike_rate: float
    smoothed_spike_rate: float
    homeostatic_error: float
    input_gain: float
    quiescent_windows: int
    low_gain_saturation_windows: int
    high_gain_saturation_windows: int


@dataclass(frozen=True, slots=True)
class HomeostasisSnapshot:
    """Complete regulator state for replay with the same in-memory topology."""

    version: str
    neuron_ids: tuple[str, ...]
    window_steps_elapsed: int
    window_spike_counts: tuple[tuple[str, int], ...]
    last_step_index: int | None
    neuron_states: tuple[NeuronHomeostaticState, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "neuron_ids": list(self.neuron_ids),
            "window_steps_elapsed": self.window_steps_elapsed,
            "window_spike_counts": dict(self.window_spike_counts),
            "last_step_index": self.last_step_index,
            "neurons": [
                {
                    "identifier": state.identifier,
                    "observed_spike_rate": state.observed_spike_rate,
                    "smoothed_spike_rate": state.smoothed_spike_rate,
                    "homeostatic_error": state.homeostatic_error,
                    "input_gain": state.input_gain,
                    "quiescent_windows": state.quiescent_windows,
                    "low_gain_saturation_windows": state.low_gain_saturation_windows,
                    "high_gain_saturation_windows": state.high_gain_saturation_windows,
                }
                for state in self.neuron_states
            ],
        }


class HomeostaticController:
    """Deterministic per-neuron input-gain adaptation around a target spike rate."""

    VERSION = "lif-homeostasis-v1"

    def __init__(
        self,
        simulation: NeuralSimulation,
        parameters: HomeostasisParameters | None = None,
    ) -> None:
        self.simulation = simulation
        self.parameters = parameters or HomeostasisParameters()
        self._neuron_ids = tuple(sorted(simulation.neurons))
        if not self._neuron_ids:
            raise ValueError("homeostasis requires at least one neuron")
        self._initial_gains = {
            identifier: simulation.neurons[identifier].input_gain
            for identifier in self._neuron_ids
        }
        self._validate_initial_gains(self._initial_gains)
        self._window_spike_counts = {identifier: 0 for identifier in self._neuron_ids}
        self._observed_rates = {identifier: 0.0 for identifier in self._neuron_ids}
        self._smoothed_rates = {identifier: 0.0 for identifier in self._neuron_ids}
        self._homeostatic_errors = {
            identifier: self.parameters.target_spike_rate
            for identifier in self._neuron_ids
        }
        self._quiescent_windows = {identifier: 0 for identifier in self._neuron_ids}
        self._low_gain_saturation_windows = {
            identifier: 0 for identifier in self._neuron_ids
        }
        self._high_gain_saturation_windows = {
            identifier: 0 for identifier in self._neuron_ids
        }
        self._window_steps_elapsed = 0
        self._last_step_index: int | None = None

    def _validate_topology(self) -> None:
        if tuple(sorted(self.simulation.neurons)) != self._neuron_ids:
            raise ValueError("homeostasis simulation neuron topology changed")

    def _validate_initial_gains(self, gains: dict[str, float]) -> None:
        for identifier, gain in gains.items():
            if not (
                self.parameters.minimum_input_gain
                <= gain
                <= self.parameters.maximum_input_gain
            ):
                raise ValueError(
                    f"initial input gain for {identifier} is outside homeostasis bounds"
                )

    def observe_step(self, step: SimulationStep) -> bool:
        """Observe exactly one latest simulation step; return True on window update."""
        self._validate_topology()
        if step.step_index != self.simulation.step_index - 1:
            raise ValueError("homeostasis must observe the latest completed simulation step")
        if (
            self._last_step_index is not None
            and step.step_index != self._last_step_index + 1
        ):
            raise ValueError("homeostasis observations must be consecutive")
        unknown = set(step.spikes).difference(self._neuron_ids)
        if unknown:
            raise ValueError(f"homeostasis received unknown spikes: {sorted(unknown)}")

        for identifier in step.spikes:
            self._window_spike_counts[identifier] += 1
        self._window_steps_elapsed += 1
        self._last_step_index = step.step_index

        if self._window_steps_elapsed < self.parameters.observation_window_steps:
            return False

        window = self.parameters.observation_window_steps
        decay = self.parameters.rate_ema_decay
        for identifier in self._neuron_ids:
            observed = self._window_spike_counts[identifier] / window
            smoothed = (
                decay * self._smoothed_rates[identifier]
                + (1.0 - decay) * observed
            )
            error = self.parameters.target_spike_rate - smoothed
            gain = self.simulation.neurons[identifier].input_gain
            if error > self.parameters.deadband:
                gain *= math.exp(
                    self.parameters.adaptation_rate
                    * (error - self.parameters.deadband)
                )
            elif error < -self.parameters.deadband:
                gain *= math.exp(
                    self.parameters.adaptation_rate
                    * (error + self.parameters.deadband)
                )
            gain = min(
                self.parameters.maximum_input_gain,
                max(self.parameters.minimum_input_gain, gain),
            )
            self.simulation.neurons[identifier].set_input_gain(gain)

            self._observed_rates[identifier] = observed
            self._smoothed_rates[identifier] = smoothed
            self._homeostatic_errors[identifier] = error
            self._quiescent_windows[identifier] = (
                self._quiescent_windows[identifier] + 1 if observed == 0.0 else 0
            )
            self._low_gain_saturation_windows[identifier] = (
                self._low_gain_saturation_windows[identifier] + 1
                if gain <= self.parameters.minimum_input_gain + 1e-12
                else 0
            )
            self._high_gain_saturation_windows[identifier] = (
                self._high_gain_saturation_windows[identifier] + 1
                if gain >= self.parameters.maximum_input_gain - 1e-12
                else 0
            )

        self._window_spike_counts = {identifier: 0 for identifier in self._neuron_ids}
        self._window_steps_elapsed = 0
        return True

    def snapshot(self) -> HomeostasisSnapshot:
        self._validate_topology()
        states = tuple(
            NeuronHomeostaticState(
                identifier=identifier,
                observed_spike_rate=self._observed_rates[identifier],
                smoothed_spike_rate=self._smoothed_rates[identifier],
                homeostatic_error=self._homeostatic_errors[identifier],
                input_gain=self.simulation.neurons[identifier].input_gain,
                quiescent_windows=self._quiescent_windows[identifier],
                low_gain_saturation_windows=self._low_gain_saturation_windows[identifier],
                high_gain_saturation_windows=self._high_gain_saturation_windows[identifier],
            )
            for identifier in self._neuron_ids
        )
        return HomeostasisSnapshot(
            version=self.VERSION,
            neuron_ids=self._neuron_ids,
            window_steps_elapsed=self._window_steps_elapsed,
            window_spike_counts=tuple(sorted(self._window_spike_counts.items())),
            last_step_index=self._last_step_index,
            neuron_states=states,
        )

    def restore(self, snapshot: HomeostasisSnapshot) -> None:
        """Restore controller state into its original simulation topology."""
        self._validate_topology()
        if snapshot.version != self.VERSION or snapshot.neuron_ids != self._neuron_ids:
            raise ValueError("homeostasis snapshot version or topology does not match")
        if not 0 <= snapshot.window_steps_elapsed < self.parameters.observation_window_steps:
            raise ValueError("homeostasis snapshot window position is invalid")
        if (
            snapshot.last_step_index is not None
            and snapshot.last_step_index != self.simulation.step_index - 1
        ):
            raise ValueError("homeostasis snapshot clock does not match simulation state")

        counts = dict(snapshot.window_spike_counts)
        if set(counts) != set(self._neuron_ids) or any(
            isinstance(count, bool)
            or not isinstance(count, int)
            or not 0 <= count <= snapshot.window_steps_elapsed
            for count in counts.values()
        ):
            raise ValueError("homeostasis snapshot contains invalid window counts")
        states = {state.identifier: state for state in snapshot.neuron_states}
        if set(states) != set(self._neuron_ids):
            raise ValueError("homeostasis snapshot neuron states do not match topology")

        for identifier, state in states.items():
            values = (
                state.observed_spike_rate,
                state.smoothed_spike_rate,
                state.homeostatic_error,
                state.input_gain,
            )
            if (
                not all(math.isfinite(value) for value in values)
                or not 0.0 <= state.observed_spike_rate <= 1.0
                or not 0.0 <= state.smoothed_spike_rate <= 1.0
                or not (
                    self.parameters.minimum_input_gain
                    <= state.input_gain
                    <= self.parameters.maximum_input_gain
                )
                or min(
                    state.quiescent_windows,
                    state.low_gain_saturation_windows,
                    state.high_gain_saturation_windows,
                ) < 0
            ):
                raise ValueError("homeostasis snapshot contains invalid regulator state")

        # Mutate only after the entire snapshot has passed validation.
        self._window_spike_counts = counts
        self._window_steps_elapsed = snapshot.window_steps_elapsed
        self._last_step_index = snapshot.last_step_index
        for identifier, state in states.items():
            self._observed_rates[identifier] = state.observed_spike_rate
            self._smoothed_rates[identifier] = state.smoothed_spike_rate
            self._homeostatic_errors[identifier] = state.homeostatic_error
            self._quiescent_windows[identifier] = state.quiescent_windows
            self._low_gain_saturation_windows[identifier] = (
                state.low_gain_saturation_windows
            )
            self._high_gain_saturation_windows[identifier] = (
                state.high_gain_saturation_windows
            )
            self.simulation.neurons[identifier].set_input_gain(state.input_gain)

    def reset(self) -> None:
        """Clear regulator history and restore source-defined per-neuron gains."""
        self._validate_topology()
        for identifier in self._neuron_ids:
            self.simulation.neurons[identifier].set_input_gain(
                self._initial_gains[identifier]
            )
            self._window_spike_counts[identifier] = 0
            self._observed_rates[identifier] = 0.0
            self._smoothed_rates[identifier] = 0.0
            self._homeostatic_errors[identifier] = self.parameters.target_spike_rate
            self._quiescent_windows[identifier] = 0
            self._low_gain_saturation_windows[identifier] = 0
            self._high_gain_saturation_windows[identifier] = 0
        self._window_steps_elapsed = 0
        self._last_step_index = None
