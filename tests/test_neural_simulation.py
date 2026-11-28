from __future__ import annotations

import pytest

from artificial_human.neural.circuits import (
    build_recurrent_microcircuit,
    run_fixed_rate_reference,
    run_microcircuit_demo,
)
from artificial_human.neural.simulation import (
    LIFNeuron, LIFParameters, NeuralSimulation, Polarity, Synapse,
)


def test_lif_neuron_integrates_leaks_spikes_resets_and_observes_refractory_period() -> None:
    neuron = LIFNeuron("n")
    assert neuron.update(0.0, time_ms=0.0, dt_ms=1.0) is False
    neuron.membrane_potential_mV = -55.0
    assert neuron.update(0.0, time_ms=1.0, dt_ms=1.0) is False
    assert -65.0 < neuron.membrane_potential_mV < -55.0

    spiking = LIFNeuron("spike")
    assert spiking.update(400.0, time_ms=0.0, dt_ms=1.0) is True
    assert spiking.membrane_potential_mV == spiking.parameters.reset_potential_mV
    assert spiking.update(400.0, time_ms=1.0, dt_ms=1.0) is False
    assert spiking.spike_count == 1
    assert spiking.snapshot().last_spike_time_ms == 0.0
    assert spiking.update(400.0, time_ms=2.0, dt_ms=1.0) is True
    assert spiking.spike_count == 2
    assert spiking.snapshot().last_spike_time_ms == 2.0


def test_neuron_parameters_reject_invalid_time_constants_and_thresholds() -> None:
    import pytest

    with pytest.raises(ValueError, match="time_constant"):
        LIFParameters(membrane_time_constant_ms=0)
    with pytest.raises(ValueError, match="below threshold"):
        LIFParameters(reset_potential_mV=-45)


def test_excitation_propagates_after_exact_synaptic_delay() -> None:
    network = NeuralSimulation(dt_ms=1.0)
    network.add_neuron(LIFNeuron("source"))
    network.add_neuron(LIFNeuron("target"))
    network.add_synapse(Synapse(
        "s1", "source", "target", 400.0, Polarity.EXCITATORY,
        delay_steps=2, time_constant_ms=5.0, maximum_weight=500.0,
    ))

    first = network.step({"source": 400.0})
    second = network.step()
    third = network.step()

    assert first.spikes == ("source",)
    assert second.spikes == ()
    assert third.spikes == ("target",)
    assert network.snapshot().events_delivered == 1


def test_inhibitory_synapse_produces_negative_postsynaptic_drive() -> None:
    network = NeuralSimulation(dt_ms=1.0)
    network.add_neuron(LIFNeuron("source"))
    target = LIFNeuron("target")
    target.membrane_potential_mV = -55.0
    network.add_neuron(target)
    network.add_synapse(Synapse(
        "inhibitory", "source", "target", 100.0, Polarity.INHIBITORY,
        delay_steps=1,
    ))
    network.step({"source": 400.0})
    before = target.membrane_potential_mV
    network.step()
    assert target.membrane_potential_mV < before
    assert network.snapshot().events_delivered == 1


def test_event_order_is_deterministic_and_reset_clears_live_state() -> None:
    first = build_recurrent_microcircuit()
    second = build_recurrent_microcircuit()
    a = first.run(30, {"sensory": 180.0})
    b = second.run(30, {"sensory": 180.0})
    assert a == b

    first.reset()
    snapshot = first.snapshot()
    assert snapshot.step_index == 0
    assert snapshot.pending_events == 0
    assert snapshot.events_scheduled == 0
    assert all(state.spike_count == 0 for state in snapshot.neuron_states)


def test_recurrent_microcircuit_has_propagation_inhibition_and_recurrent_feedback() -> None:
    recurrent = run_microcircuit_demo(recurrent=True, stimulus_steps=4, total_steps=40)
    feed_forward = run_microcircuit_demo(recurrent=False, stimulus_steps=4, total_steps=40)
    disconnected = run_microcircuit_demo(connected=False, stimulus_steps=4, total_steps=40)
    fixed_rate = run_fixed_rate_reference(stimulus_steps=4, total_steps=40)
    recurrent_spikes = {
        spike for row in recurrent["trace"] for spike in row["spikes"]
    }
    assert {"sensory", "excitatory_a", "excitatory_b", "inhibitory"} <= recurrent_spikes
    disconnected_spikes = {
        spike for row in disconnected["trace"] for spike in row["spikes"]
    }
    assert disconnected["connected"] is False
    assert disconnected["synapse_count"] == 0
    assert disconnected_spikes == {"sensory"}
    assert fixed_rate["baseline"] == "fixed-weight-synchronous-rate-reference-v1"
    assert fixed_rate["neuron_count"] == recurrent["neuron_count"]
    assert fixed_rate["projection_count"] == recurrent["synapse_count"]
    assert fixed_rate["trace"] != recurrent["trace"]
    assert recurrent["synapse_count"] == feed_forward["synapse_count"] + 1
    assert recurrent["recurrent_feedback_enabled"] is True
    no_inhibition = run_microcircuit_demo(
        recurrent=True, inhibition_enabled=False, stimulus_steps=4, total_steps=40
    )
    recurrent_b_spikes = sum(
        "excitatory_b" in row["spikes"] for row in recurrent["trace"]
    )
    no_inhibition_b_spikes = sum(
        "excitatory_b" in row["spikes"] for row in no_inhibition["trace"]
    )
    assert no_inhibition_b_spikes > recurrent_b_spikes
    assert recurrent["inhibition_enabled"] is True
    assert no_inhibition["inhibition_enabled"] is False
    assert recurrent["events_dropped"] == 0
    assert all(
        not row["spikes"] for row in recurrent["trace"] if row["step"] >= 20
    )
    assert recurrent["trace"] != feed_forward["trace"]
    assert all(
        -300.0 <= value <= 50.0
        for row in recurrent["trace"]
        for value in row["membrane_potentials_mV"].values()
    )


def test_pause_resume_single_step_and_full_in_memory_snapshot_restore() -> None:
    simulation = build_recurrent_microcircuit()
    simulation.run(5, {"sensory": 180.0})
    saved = simulation.state()
    expected = simulation.step()

    simulation.pause()
    paused_at = simulation.step_index
    assert simulation.step().step_index == paused_at
    assert simulation.step_index == paused_at

    simulation.single_step()
    assert simulation.step_index == paused_at + 1
    simulation.restore(saved)
    assert simulation.state() == saved
    assert simulation.step() == expected

    simulation.pause()
    simulation.resume()
    assert simulation.step().step_index == saved.step_index + 1


def test_network_resource_bounds_and_queue_overflow_are_explicit() -> None:
    import pytest

    from artificial_human.neural.simulation import SimulationCapacityError

    small = NeuralSimulation(max_neurons=1)
    small.add_neuron(LIFNeuron("one"))
    with pytest.raises(SimulationCapacityError, match="neuron capacity"):
        small.add_neuron(LIFNeuron("two"))

    queued = NeuralSimulation(max_pending_events=1)
    queued.add_neuron(LIFNeuron("source"))
    queued.add_neuron(LIFNeuron("a"))
    queued.add_neuron(LIFNeuron("b"))
    queued.add_synapse(Synapse("a", "source", "a", 10.0))
    queued.add_synapse(Synapse("b", "source", "b", 10.0))
    queued.step({"source": 400.0})
    assert queued.snapshot().pending_events == 1
    assert queued.snapshot().events_dropped == 1


def test_pending_event_cancellation_is_deterministic_and_counted() -> None:
    import pytest

    network = NeuralSimulation()
    network.add_neuron(LIFNeuron("source"))
    network.add_neuron(LIFNeuron("target"))
    network.add_synapse(Synapse(
        "delayed", "source", "target", 100.0, delay_steps=3,
    ))
    step = network.step({"source": 400.0})
    assert step.spikes == ("source",)
    assert network.snapshot().pending_events == 1
    assert network.cancel_pending_events(synapse_id="delayed") == 1
    assert network.snapshot().pending_events == 0
    network.run(5)
    assert network.snapshot().events_delivered == 0
    assert network.snapshot().events_cancelled == 1
    with pytest.raises(ValueError, match="unknown synapse"):
        network.cancel_pending_events(synapse_id="missing")


def test_simultaneous_spikes_use_stable_order_and_deliver_all_events() -> None:
    def run_once():
        network = NeuralSimulation()
        for name in ("source_a", "source_b", "target"):
            network.add_neuron(LIFNeuron(name))
        network.add_synapse(Synapse("a", "source_a", "target", 30.0, delay_steps=1))
        network.add_synapse(Synapse("b", "source_b", "target", 30.0, delay_steps=1))
        first = network.step({"source_a": 400.0, "source_b": 400.0})
        second = network.step()
        return first.spikes, second, network.snapshot()

    a = run_once()
    b = run_once()
    assert a == b
    assert a[0] == ("source_a", "source_b")
    assert a[1].step_index == 1
    assert a[2].events_scheduled == 2
    assert a[2].events_delivered == 2


def test_neural_benchmark_reports_throughput_and_traced_memory() -> None:
    from artificial_human.neural.benchmark import benchmark_neural_simulation

    report = benchmark_neural_simulation(steps=100)
    assert report["network"]["neurons"] == 4
    assert report["network"]["synapses"] == 5
    assert report["simulation"]["steps"] == 100
    assert report["performance"]["steps_per_second"] > 0
    assert report["python_traced_memory"]["peak_bytes"] > 0
    assert report["events"]["dropped"] == 0



def test_synapse_delay_update_affects_future_events_not_queued_events() -> None:
    simulation = NeuralSimulation()
    simulation.add_neuron(LIFNeuron("source"))
    simulation.add_neuron(LIFNeuron("target"))
    simulation.add_synapse(
        Synapse("source_to_target", "source", "target", 10.0, delay_steps=3)
    )

    simulation.step({"source": 1000.0})
    pending_before = simulation.snapshot().pending_events
    assert pending_before > 0

    simulation.update_synapse_delay("source_to_target", 1)

    assert simulation.synapses["source_to_target"].delay_steps == 1
    assert simulation.snapshot().pending_events == pending_before


def test_synapse_delay_update_rejects_invalid_values() -> None:
    simulation = NeuralSimulation()
    simulation.add_neuron(LIFNeuron("source"))
    simulation.add_neuron(LIFNeuron("target"))
    simulation.add_synapse(Synapse("s", "source", "target", 1.0))

    with pytest.raises(ValueError, match="delay_steps"):
        simulation.update_synapse_delay("s", True)
    with pytest.raises(ValueError, match="delay_steps"):
        simulation.update_synapse_delay("s", 0)
    with pytest.raises(ValueError, match="unknown synapse"):
        simulation.update_synapse_delay("missing", 1)
