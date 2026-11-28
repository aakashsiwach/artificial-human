from __future__ import annotations

import pytest

from artificial_human.neural.homeostasis import (
    HomeostaticController,
    HomeostasisParameters,
    InternalModulatorySignals,
    run_homeostasis_comparison,
)
from artificial_human.neural.plasticity import PairBasedSTDP, PlasticityController
from artificial_human.neural.simulation import LIFNeuron, NeuralSimulation


def test_recurrent_excitation_is_reduced_against_same_fixed_gain_baseline() -> None:
    report = run_homeostasis_comparison(steps=400, stress_current=400.0)

    fixed = report["fixed_gain_baseline"]
    regulated = report["homeostatically_regulated"]
    assert isinstance(fixed, dict)
    assert isinstance(regulated, dict)
    assert regulated["mean_spike_rate_per_step"] < fixed["mean_spike_rate_per_step"] * 0.35
    assert regulated["events_dropped"] == 0
    assert fixed["events_dropped"] == 0
    gains = regulated["input_gains"]
    assert isinstance(gains, dict)
    assert gains["excitatory_a"] < 1.0
    assert gains["excitatory_b"] < 1.0
    assert report["relative_spike_rate_reduction"] > 0.5


def test_silence_increases_bounded_gain_without_fabricating_spikes() -> None:
    simulation = NeuralSimulation()
    simulation.add_neuron(LIFNeuron("quiet", input_gain=0.01))
    controller = HomeostaticController(simulation)

    for _ in range(500):
        step = simulation.step()
        assert step.spikes == ()
        controller.observe_step(step)

    state = controller.snapshot()
    neuron = state.neuron_states[0]
    assert neuron.identifier == "quiet"
    assert neuron.quiescent_windows == 25
    assert neuron.input_gain == 1.0
    assert neuron.high_gain_saturation_windows > 0
    assert simulation.neurons["quiet"].spike_count == 0

    # Once an input is presented, recovered gain makes the neuron responsive.
    spikes = []
    for _ in range(12):
        step = simulation.step({"quiet": 100.0})
        spikes.extend(step.spikes)
        controller.observe_step(step)
    assert "quiet" in spikes


def test_high_stress_reaches_minimum_gain_and_records_saturation() -> None:
    simulation = NeuralSimulation()
    simulation.add_neuron(LIFNeuron("stressed"))
    parameters = HomeostasisParameters(
        target_spike_rate=0.0,
        adaptation_rate=8.0,
        deadband=0.0,
    )
    controller = HomeostaticController(simulation, parameters)

    for _ in range(300):
        step = simulation.step({"stressed": 1_000.0})
        controller.observe_step(step)

    state = controller.snapshot().neuron_states[0]
    assert state.input_gain == parameters.minimum_input_gain
    assert state.low_gain_saturation_windows > 0
    assert 0.0 <= state.smoothed_spike_rate <= 1.0


def test_gain_recovers_after_stress_and_the_network_responds_again() -> None:
    report = run_homeostasis_comparison(steps=400, stress_current=400.0)
    # Rebuild the exact deterministic stress network via the public comparison's
    # topology so the recovery phase begins from a stressed, learned controller.
    from artificial_human.neural.homeostasis import _build_recurrent_stress_network

    simulation = _build_recurrent_stress_network()
    controller = HomeostaticController(simulation)
    for _ in range(400):
        step = simulation.step({"excitatory_a": 400.0})
        controller.observe_step(step)
    stressed_gain = simulation.neurons["excitatory_a"].input_gain
    assert stressed_gain < 0.2

    for _ in range(300):
        step = simulation.step()
        controller.observe_step(step)
    recovered_gain = simulation.neurons["excitatory_a"].input_gain
    assert recovered_gain > stressed_gain
    assert recovered_gain > 0.9
    assert controller.snapshot().neuron_states[0].quiescent_windows >= 10

    later_spikes = []
    for _ in range(20):
        step = simulation.step({"excitatory_a": 200.0})
        later_spikes.extend(step.spikes)
        controller.observe_step(step)
    assert "excitatory_a" in later_spikes
    assert report["relative_spike_rate_reduction"] > 0.5


def test_homeostasis_snapshot_restore_replays_partial_window_deterministically() -> None:
    simulation = NeuralSimulation()
    simulation.add_neuron(LIFNeuron("cell"))
    controller = HomeostaticController(simulation)

    for _ in range(13):
        step = simulation.step({"cell": 400.0})
        controller.observe_step(step)
    saved_simulation = simulation.state()
    saved_regulator = controller.snapshot()

    first_trace = []
    for _ in range(37):
        step = simulation.step({"cell": 400.0})
        controller.observe_step(step)
        first_trace.append((step, simulation.neurons["cell"].input_gain, controller.snapshot()))
    expected_simulation = simulation.state()
    expected_regulator = controller.snapshot()

    simulation.restore(saved_simulation)
    controller.restore(saved_regulator)
    replay_trace = []
    for _ in range(37):
        step = simulation.step({"cell": 400.0})
        controller.observe_step(step)
        replay_trace.append((step, simulation.neurons["cell"].input_gain, controller.snapshot()))

    assert replay_trace == first_trace
    assert simulation.state() == expected_simulation
    assert controller.snapshot() == expected_regulator


def test_homeostasis_reset_restores_source_gain_and_clears_controller_history() -> None:
    simulation = NeuralSimulation()
    simulation.add_neuron(LIFNeuron("cell", input_gain=0.5))
    controller = HomeostaticController(simulation)
    for _ in range(200):
        step = simulation.step({"cell": 800.0})
        controller.observe_step(step)
    assert simulation.neurons["cell"].input_gain < 0.5

    current_step = simulation.step_index
    controller.reset()
    state = controller.snapshot()
    assert simulation.step_index == current_step
    assert simulation.neurons["cell"].input_gain == 0.5
    assert state.window_steps_elapsed == 0
    assert state.neuron_states[0].smoothed_spike_rate == 0.0
    assert state.neuron_states[0].low_gain_saturation_windows == 0


def test_typed_modulatory_channels_remain_separate_and_gate_stdp() -> None:
    signals = InternalModulatorySignals(
        salience=0.8,
        reward_prediction_error=-0.5,
        stress_load=0.25,
    )
    assert signals.reward_prediction_error == -0.5
    assert signals.as_dict() == {
        "salience": 0.8,
        "reward_prediction_error": -0.5,
        "stress_load": 0.25,
        "plasticity_gate": pytest.approx(0.3),
    }

    simulation = NeuralSimulation()
    simulation.add_neuron(LIFNeuron("pre"))
    simulation.add_neuron(LIFNeuron("post"))
    from artificial_human.neural.simulation import Synapse

    simulation.add_synapse(Synapse("pair", "pre", "post", 0.5, maximum_weight=1.0))
    controller = PlasticityController(simulation, PairBasedSTDP())
    controller.observe_step(("pre",), modulation=signals)
    controller.observe_step(("post",), modulation=signals)
    assert 0.5 < simulation.synapses["pair"].weight < 0.52

    before = simulation.synapses["pair"].weight
    suppressed = InternalModulatorySignals(
        salience=1.0,
        reward_prediction_error=1.0,
        stress_load=1.0,
    )
    controller.observe_step(("pre",), modulation=suppressed)
    controller.observe_step(("post",), modulation=suppressed)
    assert simulation.synapses["pair"].weight == before


def test_homeostasis_rejects_invalid_parameters_and_skipped_steps() -> None:
    with pytest.raises(ValueError, match="target_spike_rate"):
        HomeostasisParameters(target_spike_rate=1.1)
    with pytest.raises(ValueError, match="input gain bounds"):
        HomeostasisParameters(minimum_input_gain=0.5, maximum_input_gain=0.1)

    simulation = NeuralSimulation()
    simulation.add_neuron(LIFNeuron("cell"))
    controller = HomeostaticController(simulation)
    first = simulation.step()
    _ = simulation.step()
    with pytest.raises(ValueError, match="latest completed"):
        controller.observe_step(first)
