from __future__ import annotations

import pytest

from artificial_human.neural.world_model import (
    PredictiveWorldModel,
    WorldState,
    run_world_model_comparison,
)


def test_prediction_is_one_step_and_does_not_mutate_model() -> None:
    model = PredictiveWorldModel(2, 1)
    state = WorldState((0.2, -0.1), 4)
    before = model.clone()

    prediction = model.predict(state, (0.5,))

    assert prediction.state.timestamp == 5
    assert len(prediction.state.values) == 2
    assert prediction.standard_deviation == pytest.approx((0.5, 0.5))
    assert model.weights == before.weights
    assert model.observations == 0


def test_online_updates_reduce_prediction_error_on_repeated_transition() -> None:
    model = PredictiveWorldModel(2, 1, learning_rate=0.8)
    state = WorldState((0.2, -0.1), 0)
    action = (0.5,)
    observed = WorldState((0.4, 0.0), 1)
    initial_error = sum(
        abs(a - b)
        for a, b in zip(model.predict(state, action).state.values, observed.values, strict=True)
    )

    for _ in range(80):
        model.observe(state, action, observed)

    final_error = sum(
        abs(a - b)
        for a, b in zip(model.predict(state, action).state.values, observed.values, strict=True)
    )
    assert final_error < initial_error * 0.1
    assert model.observations == 80


def test_prediction_error_channels_are_distinct_and_update_is_optional() -> None:
    model = PredictiveWorldModel(2, 1)
    state = WorldState((0.1, 0.2), 2)
    observed = WorldState((0.7, 0.5), 3)
    before = model.clone()

    report = model.observe(
        state,
        (0.3,),
        observed,
        expected_reward=0.8,
        observed_reward=0.2,
        update=False,
    )

    assert report.model_error_mae > 0.0
    assert report.sensory_surprise == pytest.approx(0.45)
    assert report.reward_goal_error == pytest.approx(0.6)
    assert report.updated is False
    assert report.uncertainty_before == pytest.approx(report.uncertainty_after)
    assert model.weights == before.weights
    assert model.observations == 0


def test_observation_updates_residual_uncertainty_after_unexpected_transition() -> None:
    model = PredictiveWorldModel(2, 1)
    state = WorldState((0.0, 0.0), 0)
    report = model.observe(
        state,
        (0.0,),
        WorldState((0.9, -0.8), 1),
    )
    assert report.uncertainty_after > report.uncertainty_before
    assert report.updated is True


def test_multi_step_rollout_uses_each_prior_prediction_as_next_state() -> None:
    model = PredictiveWorldModel(2, 1)
    state = WorldState((0.0, 0.0), 10)
    actions = ((0.1,), (0.2,), (-0.1,))
    rollout = model.rollout(state, actions)

    assert len(rollout) == 3
    assert [item.state.timestamp for item in rollout] == [11, 12, 13]
    assert rollout[1].state.values == pytest.approx(
        model.predict(rollout[0].state, actions[1]).state.values
    )
    assert rollout[0].standard_deviation == pytest.approx((0.5, 0.5))
    assert rollout[1].standard_deviation[0] > rollout[0].standard_deviation[0]
    assert rollout[2].standard_deviation[0] > rollout[1].standard_deviation[0]


def test_model_clone_has_independent_parameters() -> None:
    model = PredictiveWorldModel(2, 1)
    clone = model.clone()
    model.observe(
        WorldState((0.2, 0.1), 0),
        (0.5,),
        WorldState((0.8, 0.6), 1),
    )
    assert clone.weights != model.weights
    assert clone.observations == 0


def test_deterministic_heldout_evaluation_beats_frozen_model_and_compares_lexical_baseline() -> None:
    first = run_world_model_comparison(training_samples=240, heldout_samples=100)
    second = run_world_model_comparison(training_samples=240, heldout_samples=100)

    assert first == second
    assert first.learned_model_mae < first.frozen_model_mae
    assert first.relative_improvement > 0.0
    assert 0.0 <= first.lexical_novelty_accuracy <= 1.0
    assert 0.0 <= first.lexical_novelty_precision <= 1.0
    assert 0.0 <= first.lexical_novelty_recall <= 1.0
    assert first.recovery.first_error > first.recovery.final_error
    assert first.recovery.uncertainty_after_shift > first.recovery.uncertainty_before_shift
    assert first.recovery.uncertainty_after_recovery < first.recovery.uncertainty_after_shift


@pytest.mark.parametrize(
    "state,action",
    [
        (WorldState((0.0,), 0), (0.0, 0.0)),
        (WorldState((0.0, 0.0), 0), (2.0,)),
    ],
)
def test_model_rejects_dimension_mismatch_and_unbounded_actions(state, action) -> None:
    model = PredictiveWorldModel(2, 1)
    with pytest.raises(ValueError):
        model.predict(state, action)


def test_model_rejects_invalid_states_timestamps_and_parameters() -> None:
    with pytest.raises(ValueError, match="world-state values"):
        WorldState((1.1,), 0)
    with pytest.raises(ValueError, match="timestamp"):
        WorldState((0.0,), -1)
    with pytest.raises(ValueError, match="learning_rate"):
        PredictiveWorldModel(1, 1, learning_rate=0.0)
    with pytest.raises(ValueError, match="residual_decay"):
        PredictiveWorldModel(1, 1, residual_decay=1.0)
    with pytest.raises(ValueError, match="state_dimensions"):
        PredictiveWorldModel(0, 1)


def test_observe_rejects_nonsequential_or_wrong_sized_observations() -> None:
    model = PredictiveWorldModel(2, 1)
    state = WorldState((0.0, 0.0), 1)
    with pytest.raises(ValueError, match="advance by one tick"):
        model.observe(state, (0.0,), WorldState((0.1, 0.2), 4))
    with pytest.raises(ValueError, match="dimension"):
        model.observe(state, (0.0,), WorldState((0.1,), 2))
