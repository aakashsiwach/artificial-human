from __future__ import annotations

import pytest

from artificial_human.neural.plasticity import PairBasedSTDP, PlasticityController, RateBasedHebbian, run_learning_rule_comparison
from artificial_human.neural.simulation import LIFNeuron, NeuralSimulation, Synapse


def network() -> NeuralSimulation:
    simulation = NeuralSimulation()
    for name in ("pre", "post"):
        simulation.add_neuron(LIFNeuron(name))
    simulation.add_synapse(Synapse("connection", "pre", "post", 0.5, maximum_weight=1.0))
    return simulation


def test_stdp_direction_depends_on_spike_order_and_is_bounded() -> None:
    potentiated = network()
    rule = PairBasedSTDP()
    rule.observe(potentiated, ("pre",))
    rule.observe(potentiated, ("post",))
    assert potentiated.synapses["connection"].weight > 0.5

    depressed = network()
    rule = PairBasedSTDP()
    rule.observe(depressed, ("post",))
    rule.observe(depressed, ("pre",))
    assert depressed.synapses["connection"].weight < 0.5
    for _ in range(100):
        rule.observe(potentiated, ("pre",))
        rule.observe(potentiated, ("post",))
    assert 0 <= potentiated.synapses["connection"].weight <= 1.0


def test_gate_zero_disables_learning_and_reset_restores_initial_weights() -> None:
    simulation = network()
    controller = PlasticityController(simulation, PairBasedSTDP())
    controller.observe_step(("pre",), gate=0.0)
    controller.observe_step(("post",), gate=0.0)
    assert simulation.synapses["connection"].weight == 0.5
    controller.observe_step(("pre",))
    controller.observe_step(("post",))
    assert simulation.synapses["connection"].weight != 0.5
    controller.reset()
    assert simulation.synapses["connection"].weight == 0.5
    assert simulation.step_index == 0


def test_rate_hebbian_is_a_distinct_bounded_comparator() -> None:
    simulation = network()
    RateBasedHebbian().observe(simulation, ("pre", "post"))
    assert 0.5 < simulation.synapses["connection"].weight <= 1.0


def test_snapshot_restores_learned_weights_and_reset_forgets_them() -> None:
    simulation = network()
    rule = PairBasedSTDP()
    rule.observe(simulation, ("pre",))
    rule.observe(simulation, ("post",))
    saved = simulation.state()
    expected = simulation.synapses["connection"].weight
    simulation.update_synapse_weight("connection", 0.1)
    simulation.restore(saved)
    assert simulation.synapses["connection"].weight == expected
    simulation.reset()
    assert simulation.synapses["connection"].weight == 0.5


def test_learning_rule_comparison_reports_fixed_baseline_and_limits() -> None:
    report = run_learning_rule_comparison(trials=12)
    rows = {row["rule"]: row for row in report["results"]}
    assert rows["stdp"]["accuracy"] > rows["fixed"]["accuracy"]
    assert rows["rate_hebbian"]["rule"] == "rate_hebbian"
    assert rows["stdp_gate_off"]["accuracy"] == rows["fixed"]["accuracy"]
    assert report["stdp_gate_off_accuracy"] == rows["fixed"]["accuracy"]
    assert report["stdp_improvement_over_fixed"] > 0
    with pytest.raises(ValueError):
        run_learning_rule_comparison(trials=0)


def test_invalid_learning_inputs_are_rejected() -> None:
    simulation = network()
    with pytest.raises(ValueError, match="gate"):
        PairBasedSTDP().observe(simulation, ("pre",), gate=2)
    with pytest.raises(ValueError, match="unknown spiking"):
        PairBasedSTDP().observe(simulation, ("missing",))
