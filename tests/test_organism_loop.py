import pytest

from artificial_human.neural.organism_loop import (
    DevelopmentProgram, DevelopmentStage, EpisodicMemory, Homeostat, Modality,
    OrganismLoop, PlasticSynapse, PopulationEncoder, SafeSelfDevelopment,
    SensoryEvent, SimulatedEnvironment, WorldModel,
)


def test_pair_stdp_has_timing_dependent_bounded_updates():
    synapse = PlasticSynapse()
    synapse.tick(True, False)
    assert synapse.tick(False, True) > 0.5
    reverse = PlasticSynapse()
    reverse.tick(False, True)
    assert reverse.tick(True, False) < 0.5


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
