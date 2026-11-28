"""Bounded local synaptic plasticity rules for the deterministic LIF simulator.

The rules here are computational hypotheses, not claims of a unique biological
learning mechanism. All traces and learned weights remain in process memory.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, cast

from artificial_human.neural.simulation import NeuralSimulation


class PlasticityRule(Protocol):
    def observe(self, simulation: NeuralSimulation, spikes: tuple[str, ...], gate: float) -> None: ...
    def reset(self) -> None: ...


@dataclass(frozen=True, slots=True)
class STDPParameters:
    """Pair-based STDP parameters in simulation-step units."""

    potentiation: float = 0.02
    depression: float = 0.025
    trace_decay: float = 0.9

    def __post_init__(self) -> None:
        if not all(math.isfinite(x) for x in (self.potentiation, self.depression, self.trace_decay)):
            raise ValueError("STDP parameters must be finite")
        if self.potentiation < 0 or self.depression < 0:
            raise ValueError("STDP amplitudes must be non-negative")
        if not 0 <= self.trace_decay < 1:
            raise ValueError("trace_decay must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class PlasticitySnapshot:
    pre_traces: tuple[tuple[str, float], ...]
    post_traces: tuple[tuple[str, float], ...]


@dataclass(slots=True)
class PairBasedSTDP:
    """All-to-all pair rule: pre-before-post potentiates; post-before-pre depresses.

    Traces decay once per call to observe(), so one call represents one
    simulation step. Simultaneous pre/post spikes do not form a zero-lag pair.
    The trace formulation is the standard all-to-all exponential pair rule.
    """

    parameters: STDPParameters = field(default_factory=STDPParameters)
    pre_traces: dict[str, float] = field(default_factory=dict)
    post_traces: dict[str, float] = field(default_factory=dict)

    def observe(self, simulation: NeuralSimulation, spikes: tuple[str, ...], gate: float = 1.0) -> None:
        _validate_gate(gate)
        spike_set = set(spikes)
        unknown = spike_set.difference(simulation.neurons)
        if unknown:
            raise ValueError(f"unknown spiking neurons: {sorted(unknown)}")
        decay = self.parameters.trace_decay
        for neuron in simulation.neurons:
            self.pre_traces[neuron] = self.pre_traces.get(neuron, 0.0) * decay
            self.post_traces[neuron] = self.post_traces.get(neuron, 0.0) * decay

        for synapse in simulation.synapses.values():
            delta = 0.0
            if synapse.source in spike_set:
                delta -= self.parameters.depression * self.post_traces[synapse.target]
            if synapse.target in spike_set:
                delta += self.parameters.potentiation * self.pre_traces[synapse.source]
            if delta:
                # Clamp magnitude to the synapse's configured range; the
                # rule may request potentiation/depression beyond a hard bound.
                weight = min(
                    synapse.maximum_weight,
                    max(0.0, synapse.weight + gate * delta),
                )
                simulation.update_synapse_weight(synapse.identifier, weight)

        for neuron in spike_set:
            self.pre_traces[neuron] += 1.0
            self.post_traces[neuron] += 1.0

    def snapshot(self) -> PlasticitySnapshot:
        return PlasticitySnapshot(tuple(sorted(self.pre_traces.items())),
                                  tuple(sorted(self.post_traces.items())))

    def restore(self, snapshot: PlasticitySnapshot, simulation: NeuralSimulation) -> None:
        expected = set(simulation.neurons)
        pre, post = dict(snapshot.pre_traces), dict(snapshot.post_traces)
        if set(pre) != expected or set(post) != expected:
            raise ValueError("plasticity snapshot neuron set does not match simulation")
        if any(not math.isfinite(v) or v < 0 for v in (*pre.values(), *post.values())):
            raise ValueError("plasticity snapshot contains invalid traces")
        self.pre_traces, self.post_traces = pre, post

    def reset(self) -> None:
        self.pre_traces.clear()
        self.post_traces.clear()


@dataclass(frozen=True, slots=True)
class HebbianParameters:
    learning_rate: float = 0.04
    trace_decay: float = 0.9

    def __post_init__(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate < 0:
            raise ValueError("learning_rate must be finite and non-negative")
        if not math.isfinite(self.trace_decay) or not 0 <= self.trace_decay < 1:
            raise ValueError("trace_decay must be in [0, 1)")


@dataclass(slots=True)
class RateBasedHebbian:
    """Exponential-rate trace correlation baseline with bounded weight updates."""

    parameters: HebbianParameters = field(default_factory=HebbianParameters)
    rates: dict[str, float] = field(default_factory=dict)

    def observe(self, simulation: NeuralSimulation, spikes: tuple[str, ...], gate: float = 1.0) -> None:
        _validate_gate(gate)
        spike_set = set(spikes)
        unknown = spike_set.difference(simulation.neurons)
        if unknown:
            raise ValueError(f"unknown spiking neurons: {sorted(unknown)}")
        decay = self.parameters.trace_decay
        for neuron in simulation.neurons:
            self.rates[neuron] = self.rates.get(neuron, 0.0) * decay
        for neuron in spike_set:
            self.rates[neuron] += 1.0
        for synapse in simulation.synapses.values():
            correlation = self.rates[synapse.source] * self.rates[synapse.target]
            if correlation:
                simulation.update_synapse_weight(
                    synapse.identifier,
                    synapse.weight + gate * self.parameters.learning_rate * correlation,
                )

    def reset(self) -> None:
        self.rates.clear()


@dataclass(slots=True)
class PlasticityController:
    simulation: NeuralSimulation
    rule: PlasticityRule

    def observe_step(self, spikes: tuple[str, ...], *, gate: float = 1.0) -> None:
        self.rule.observe(self.simulation, spikes, gate)

    def reset(self) -> None:
        self.rule.reset()
        self.simulation.reset()


def _validate_gate(gate: float) -> None:
    if isinstance(gate, bool) or not math.isfinite(gate) or not 0 <= gate <= 1:
        raise ValueError("plasticity gate must be finite and in [0, 1]")


def _association_trial(rule_name: str, trials: int = 12) -> dict[str, object]:
    """Train paired and unpaired connections using spikes from the LIF simulator."""
    from artificial_human.neural.simulation import LIFNeuron, Synapse

    network = NeuralSimulation()
    for identifier in ("pre_paired", "post_paired", "pre_unpaired", "post_unpaired"):
        network.add_neuron(LIFNeuron(identifier))
    starting_weight = 300.0
    maximum_weight = 350.0
    network.add_synapse(Synapse("paired", "pre_paired", "post_paired", starting_weight, maximum_weight=maximum_weight))
    network.add_synapse(Synapse("unpaired", "pre_unpaired", "post_unpaired", starting_weight, maximum_weight=maximum_weight))
    if rule_name in {"stdp", "stdp_gate_off"}:
        rule: PlasticityRule = PairBasedSTDP(\n            STDPParameters(potentiation=1.0, depression=0.8, trace_decay=0.9)\n        )
    elif rule_name == "rate_hebbian":
        rule = RateBasedHebbian()
    elif rule_name == "fixed":
        rule = _NoPlasticity()
    else:
        raise ValueError("unknown learning rule")

    controller = PlasticityController(network, rule)
    for _ in range(trials):
        # Paired pathway: presynaptic neuron fires, then postsynaptic neuron
        # receives an external pulse one simulation step later.
        first = network.step({"pre_paired": 400.0, "pre_unpaired": 400.0})
        gate = 0.0 if rule_name == "stdp_gate_off" else 1.0
        controller.observe_step(first.spikes, gate=gate)
        second = network.step({"post_paired": 400.0})
        controller.observe_step(second.spikes, gate=gate)
        # The unpaired control receives presynaptic activity only.
        for _ in range(30):
            quiet = network.step()
            controller.observe_step(quiet.spikes)

    paired = network.synapses["paired"].weight
    unpaired = network.synapses["unpaired"].weight

    # Test recall in a fresh network: only presynaptic cues are supplied.
    # A paired pathway must now evoke the target through its learned synapse.
    probe = NeuralSimulation()
    for identifier in ("pre_paired", "post_paired", "pre_unpaired", "post_unpaired"):
        probe.add_neuron(LIFNeuron(identifier))
    probe.add_synapse(Synapse("paired", "pre_paired", "post_paired", paired, maximum_weight=maximum_weight))
    probe.add_synapse(Synapse("unpaired", "pre_unpaired", "post_unpaired", unpaired, maximum_weight=maximum_weight))
    probe.step({"pre_paired": 400.0, "pre_unpaired": 400.0})
    recall = probe.step()
    paired_recall_spike = "post_paired" in recall.spikes
    unpaired_recall_spike = "post_unpaired" in recall.spikes
    correct = int(paired_recall_spike) + int(not unpaired_recall_spike)
    return {
        "rule": rule_name,
        "trials": trials,
        "initial_weight": starting_weight,
        "maximum_weight": maximum_weight,
        "paired_weight": round(paired, 8),
        "unpaired_weight": round(unpaired, 8),
        "paired_recall_spike": paired_recall_spike,
        "unpaired_recall_spike": unpaired_recall_spike,
        "correct": correct,
        "accuracy": correct / 2,
    }

@dataclass(slots=True)
class _NoPlasticity:
    def observe(self, simulation: NeuralSimulation, spikes: tuple[str, ...], gate: float) -> None:
        _validate_gate(gate)
        if set(spikes).difference(simulation.neurons):
            raise ValueError("unknown spiking neuron")

    def reset(self) -> None:
        return None


def run_learning_rule_comparison(*, trials: int = 12) -> dict[str, object]:
    """Compare STDP, rate-Hebbian and fixed weights on a deterministic association.

    A positive pair has pre-before-post timing; the negative control has pre
    activity without postsynaptic pairing. The task tests this chosen timing
    hypothesis only and is not evidence that it is a universal learning rule.
    """
    if isinstance(trials, bool) or not isinstance(trials, int) or not 1 <= trials <= 1_000:
        raise ValueError("trials must be an integer in [1, 1000]")
    results = [
        _association_trial(name, trials)
        for name in ("stdp", "stdp_gate_off", "rate_hebbian", "fixed")
    ]
    fixed_accuracy = next(cast(float, row["accuracy"]) for row in results if row["rule"] == "fixed")
    return {
        "task": "pre-before-post temporal association with unpaired negative control",
        "metric": "two-pathway recall accuracy from postsynaptic spikes caused by learned synapses",
        "results": results,
        "stdp_improvement_over_fixed": (
            next(cast(float, row["accuracy"]) for row in results if row["rule"] == "stdp")
            - fixed_accuracy
        ),
        "stdp_gate_off_accuracy": next(
            cast(float, row["accuracy"]) for row in results if row["rule"] == "stdp_gate_off"
        ),
        "limitations": [
            "Small deterministic mechanism test, not a biological validation.",
            "Only the selected pre-before-post timing task is evaluated.",
            "The rate-Hebbian comparator is a simple trace-correlation baseline.",
            "No claim is made that STDP is the unique or generally superior learning rule.",
        ],
    }
