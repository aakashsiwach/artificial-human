"""Small deterministic recurrent circuit used as the first integration experiment."""
from __future__ import annotations

import math

from artificial_human.neural.simulation import (
    LIFNeuron,
    LIFParameters,
    NeuralSimulation,
    Polarity,
    Synapse,
)


def build_recurrent_microcircuit(
    *, recurrent: bool = True, inhibition_enabled: bool = True, connected: bool = True,
) -> NeuralSimulation:
    """Build a compact excitatory/inhibitory circuit for causal propagation tests.

    Labels describe computational roles in this experiment only; they are not
    claims that the topology reproduces an anatomical brain region.
    """
    parameters = LIFParameters(
        resting_potential_mV=-65.0,
        reset_potential_mV=-65.0,
        threshold_mV=-50.0,
        membrane_time_constant_ms=10.0,
        membrane_resistance=1.0,
        refractory_period_ms=2.0,
    )
    simulation = NeuralSimulation(dt_ms=1.0, max_neurons=8, max_synapses=16)
    for identifier in ("sensory", "excitatory_a", "excitatory_b", "inhibitory"):
        simulation.add_neuron(LIFNeuron(identifier, parameters))

    if connected:
        simulation.add_synapse(Synapse(
            "sensory_to_a", "sensory", "excitatory_a", 170.0,
            Polarity.EXCITATORY, delay_steps=1, time_constant_ms=4.0,
        ))
        simulation.add_synapse(Synapse(
            "a_to_b", "excitatory_a", "excitatory_b", 160.0,
            Polarity.EXCITATORY, delay_steps=1, time_constant_ms=4.0,
        ))
        if inhibition_enabled:
            simulation.add_synapse(Synapse(
                "a_to_inhibitory", "excitatory_a", "inhibitory", 160.0,
                Polarity.EXCITATORY, delay_steps=1, time_constant_ms=4.0,
            ))
            simulation.add_synapse(Synapse(
                "inhibitory_to_b", "inhibitory", "excitatory_b", 220.0,
                Polarity.INHIBITORY, delay_steps=1, time_constant_ms=4.0,
            ))
        if recurrent:
            simulation.add_synapse(Synapse(
                "b_to_a_recurrent", "excitatory_b", "excitatory_a", 55.0,
                Polarity.EXCITATORY, delay_steps=2, time_constant_ms=8.0,
            ))
    return simulation


def run_microcircuit_demo(
    *, recurrent: bool = True, inhibition_enabled: bool = True, connected: bool = True,
    stimulus_steps: int = 4, total_steps: int = 40,
) -> dict[str, object]:
    if not 1 <= stimulus_steps <= total_steps <= 10_000:
        raise ValueError("require 1 <= stimulus_steps <= total_steps <= 10000")
    simulation = build_recurrent_microcircuit(
        recurrent=recurrent, inhibition_enabled=inhibition_enabled, connected=connected
    )
    trace = []
    for _ in range(total_steps):
        step = simulation.step(
            {"sensory": 180.0} if simulation.step_index < stimulus_steps else None
        )
        trace.append({
            "step": step.step_index,
            "spikes": list(step.spikes),
            "membrane_potentials_mV": dict(step.membrane_potentials),
        })
    snapshot = simulation.snapshot().as_dict()
    snapshot["trace"] = trace
    snapshot["recurrent_feedback_enabled"] = recurrent
    snapshot["inhibition_enabled"] = inhibition_enabled
    snapshot["connected"] = connected
    snapshot["note"] = (
        "Reduced LIF computational microcircuit; not an anatomical brain simulation."
    )
    return snapshot



def run_fixed_rate_reference(
    *, stimulus_steps: int = 4, total_steps: int = 40
) -> dict[str, object]:
    """Run a same-size fixed-weight rate baseline for circuit comparison.

    This control has continuous bounded activities and synchronous projections:
    it deliberately has no membrane potential, spike threshold, refractory period
    or delayed event queue. It is a simple engineering baseline, not a biological
    rate code model.
    """
    if not 1 <= stimulus_steps <= total_steps <= 10_000:
        raise ValueError("require 1 <= stimulus_steps <= total_steps <= 10000")
    identifiers = ("sensory", "excitatory_a", "excitatory_b", "inhibitory")
    edges = (
        ("sensory", "excitatory_a", 0.68),
        ("excitatory_a", "excitatory_b", 0.64),
        ("excitatory_a", "inhibitory", 0.64),
        ("inhibitory", "excitatory_b", -0.88),
        ("excitatory_b", "excitatory_a", 0.22),
    )
    activity = {identifier: 0.0 for identifier in identifiers}
    trace: list[dict[str, object]] = []
    for step in range(total_steps):
        previous = activity.copy()
        incoming = {identifier: 0.0 for identifier in identifiers}
        for source, target, weight in edges:
            incoming[target] += previous[source] * weight
        updated: dict[str, float] = {}
        for identifier in identifiers:
            drive = 0.7 if identifier == "sensory" and step < stimulus_steps else 0.0
            target_rate = max(0.0, math.tanh(1.8 * (drive + incoming[identifier])))
            updated[identifier] = min(
                1.0, max(0.0, 0.72 * previous[identifier] + 0.28 * target_rate)
            )
        activity = updated
        trace.append({"step": step, "rates": dict(activity)})
    return {
        "baseline": "fixed-weight-synchronous-rate-reference-v1",
        "neuron_count": len(identifiers),
        "projection_count": len(edges),
        "trace": trace,
        "note": "Matched-size continuous rate control; no spikes, membrane or event delays.",
    }
