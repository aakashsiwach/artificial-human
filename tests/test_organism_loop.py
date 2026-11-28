import pytest

from artificial_human.neural.organism_loop import (
    DevelopmentProgram, DevelopmentStage, EpisodicMemory, Homeostat, Modality,
    OrganismLoop, PlasticSynapse, PopulationEncoder, SafeSelfDevelopment,
    SensoryEvent, SimulatedEnvironment, WorldModel,
    OrganismIntegrationConfig,
    run_integration_ablation_comparison,
)


def test_pair_stdp_has_timing_dependent_bounded_updates():
    synapse = PlasticSynapse()
    # Post-before-pre potentiates; pre-before-post depresses under this
    # implementation's pair-based trace update order.
    synapse.tick(False, True)
    assert synapse.tick(True, False) > 0.5
    reverse = PlasticSynapse()
    reverse.tick(True, False)
    assert reverse.tick(False, True) < 0.5


def test_homeostat_stabilizes_with_bounded_threshold():
    regulator = Homeostat()
    for _ in range(100):
        regulator.update(1.0)
    assert 0.1 <= regulator.threshold <= 2.0
    assert regulator.threshold > 0.5


def test_population_encoder_is_deterministic_and_uncertainty_aware():
    event = SensoryEvent(Modality.VISION, (0.2, 0.4, 0.6), 1, 0.7)
    encoder = PopulationEncoder(8)
    assert encoder.encode(event) == encoder.encode(event)
    assert encoder.encode(event).uncertainty == 0.3


def test_episodic_memory_is_bounded_and_recalls_partial_cues():
    memory = EpisodicMemory(2)
    memory.store(0, (1.0, 0.0), "left")
    memory.store(1, (0.0, 1.0), "right")
    assert memory.recall((0.9, 0.1)) == "left"
    memory.store(2, (0.5, 0.5), "new")
    assert len(memory.episodes) == 2


def test_world_model_prediction_error_updates_weights():
    model = WorldModel.create(2)
    before = abs(1 - model.predict((1.0, 0.0)))
    model.observe((1.0, 0.0), 1.0)
    assert abs(1 - model.predict((1.0, 0.0))) < before


def test_environment_bounds_actions_and_episode_steps():
    env = SimulatedEnvironment(max_steps=1)
    env.act(1)
    with pytest.raises(RuntimeError, match="episode step limit"):
        env.act(1)


def test_development_progresses_on_simulation_ticks():
    program = DevelopmentProgram()
    for _ in range(100):
        program.advance()
    assert program.stage is DevelopmentStage.JUVENILE
    for _ in range(200):
        program.advance()
    assert program.stage is DevelopmentStage.MATURE


def test_self_development_is_review_only_and_rejects_execution():
    guard = SafeSelfDevelopment()
    proposal = guard.propose("fix test", "+ safe change", ("pytest -q",))
    assert proposal.status == "awaiting_human_review"
    with pytest.raises(ValueError, match="prohibited execution primitive"):
        guard.propose("bad", "+ eval('x')", ("pytest -q",))


def test_integrated_loop_is_reproducible_without_llm():
    first, second = OrganismLoop(), OrganismLoop()
    assert first.run(10) == second.run(10)
    assert first.snapshot() == second.snapshot()
    assert first.snapshot()["episodes"] == 10


def test_pause_resume_and_reset():
    loop = OrganismLoop()
    loop.pause()
    assert loop.tick()["paused"]
    loop.resume()
    loop.tick()
    loop.reset()
    assert loop.snapshot()["tick"] == 0
    assert loop.snapshot()["episodes"] == 0



def test_integrated_loop_connects_attention_working_memory_action_selection_and_lif() -> None:
    loop = OrganismLoop()
    first = loop.tick()

    assert first["attended"]
    assert first["working_memory"]["item_count"] == 1
    assert first["working_memory_focus"] > 0.0
    assert first["action_selection"]["action_id"] in {
        "move_left", "hold", "move_right"
    }
    assert first["neural"]["accepted"] is True
    assert first["neural"]["simulation_steps_advanced"] == 10
    assert first["neural"]["simulation_step"] == 11
    assert ":neural=" in first["action_selection"]["state_id"]

    loop.run(12)
    snapshot = loop.snapshot()
    assert snapshot["working_memory"]["item_count"] <= 7
    assert snapshot["action_selection"]["updates"] >= 13
    assert snapshot["neural"]["simulation_step"] >= 13 * 11


def test_integrated_subsystem_ablations_are_explicit_and_observable() -> None:
    ablated = OrganismLoop(config=OrganismIntegrationConfig(
        attention_enabled=False,
        working_memory_enabled=False,
        episodic_memory_enabled=False,
        neural_substrate_enabled=False,
        learned_action_selection_enabled=False,
    ))

    result = ablated.tick()
    snapshot = ablated.snapshot()

    assert result["attended"] == ()
    assert result["working_memory"]["item_count"] == 0
    assert result["working_memory_focus"] == 0.0
    assert result["recalled"] is None
    assert result["action_selection"]["mode"] == "directional_ablation_baseline"
    assert result["neural"]["accepted"] is False
    assert snapshot["neural"]["simulation_step"] == 0
    assert snapshot["action_selection"]["updates"] == 0
    assert snapshot["episodes"] == 0


def test_learned_action_selection_never_leaves_the_simulation_allowlist() -> None:
    loop = OrganismLoop()
    allowed = {"move_left", "hold", "move_right"}

    for result in loop.run(80):
        assert result["action_selection"]["action_id"] in allowed
        assert result["action"] in {-1, 0, 1}
        assert -1.0 <= result["reward_signal"] <= 1.0

    assert loop.snapshot()["action_selection"]["updates"] == 80


def test_integrated_reset_clears_attention_workspace_and_learned_channels() -> None:
    loop = OrganismLoop()
    loop.run(10)
    assert loop.snapshot()["working_memory"]["item_count"] > 0
    assert loop.snapshot()["action_selection"]["updates"] > 0

    loop.reset()
    snapshot = loop.snapshot()
    assert snapshot["working_memory"]["item_count"] == 0
    assert snapshot["action_selection"]["updates"] == 0
    assert snapshot["episodes"] == 0
    assert snapshot["tick"] == 0



def test_integration_ablation_report_measures_major_subsystem_contributions() -> None:
    reports = {
        item["condition"]: item
        for item in run_integration_ablation_comparison(ticks=30)
    }
    full = reports["full"]

    assert len(reports) == 6
    assert full["attended_items"] > reports["attention_disabled"]["attended_items"]
    assert full["mean_attention_weight"] > reports["attention_disabled"]["mean_attention_weight"]
    assert full["working_memory_items"] > reports["working_memory_disabled"]["working_memory_items"]
    assert full["episodic_memories"] > reports["episodic_memory_disabled"]["episodic_memories"]
    assert full["action_learning_updates"] > reports["learned_action_selection_disabled"]["action_learning_updates"]
    assert full["neural_spikes"] > reports["neural_substrate_disabled"]["neural_spikes"]
    assert all(item["ticks"] == 30 for item in reports.values())
    assert all(-100.0 <= item["cumulative_reward"] <= 0.0 for item in reports.values())



def test_sensory_event_rejects_malformed_and_unbounded_inputs() -> None:
    with pytest.raises(ValueError, match="supported Modality"):
        SensoryEvent("camera", (0.1,), 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="timestamp"):
        SensoryEvent(Modality.VISION, (0.1,), True)
    with pytest.raises(ValueError, match="finite numbers"):
        SensoryEvent(Modality.VISION, (float("nan"),), 0)
    with pytest.raises(ValueError, match="confidence"):
        SensoryEvent(Modality.VISION, (0.1,), 0, float("inf"))



def test_action_selection_state_includes_the_live_neural_population_state() -> None:
    full = OrganismLoop().tick()
    neural_disabled = OrganismLoop(config=OrganismIntegrationConfig(
        neural_substrate_enabled=False,
    )).tick()

    assert full["neural"]["population_state"] in {
        "distributed", "sensory_only", "silent"
    }
    assert neural_disabled["neural"]["population_state"] == "disabled"
    assert ":neural=disabled:" in neural_disabled["action_selection"]["state_id"]
    assert (
        f":neural={full['neural']['population_state']}:"
        in full["action_selection"]["state_id"]
    )


def test_sensory_attention_weight_is_not_replaced_by_a_higher_episode_score() -> None:
    loop = OrganismLoop()
    event = SensoryEvent(Modality.PROPRIOCEPTION, (0.1,), 0, confidence=0.1)
    encoded = loop.encoder.encode(event)
    cue = tuple(value for value in loop.encoder.decode(encoded) if value is not None)
    loop.memory.store(0, cue, "recalled episode")

    result = loop.tick(event)

    assert result["attended"][0]["reference_id"] == "episode:0"
    assert result["sensory_attention_weight"] == pytest.approx(0.1)
    assert result["neural"]["input_intensity"] == pytest.approx(0.055)
