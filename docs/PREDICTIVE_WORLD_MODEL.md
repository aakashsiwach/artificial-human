# Predictive world-state model (issue #29)

## Scope

This change adds a bounded, deterministic one-step transition model and multi-step open-loop rollout for explicitly supplied normalized state/action vectors. It runs in process memory and requires no LLM, database, device, or network access. It is an engineering baseline, not a biological predictive-coding implementation or a complete physical world model.

## State and transition model

- `WorldState` stores 1..32 finite normalized coordinates in [-1, 1] and an integer simulation timestamp.
- Actions are finite normalized vectors in [-1, 1], with a configured dimension of 1..16.
- `PredictiveWorldModel` uses a linear transition function over a bias, current state and action features. Parameters are learned online with normalized LMS updates and bounded weights.
- `predict` is side-effect-free and predicts the next state before the next observation.
- `rollout` repeatedly feeds each predicted state into the next prediction to expose multi-step error accumulation.
- Residual variance is maintained per output coordinate using an exponentially weighted update. Its square root is exposed as an uncertainty proxy; it is not a calibrated Bayesian posterior or guaranteed confidence interval.

## Separate error channels

`observe` calculates all diagnostics from the prediction made before any parameter update:

1. **Model error:** mean absolute difference between predicted and observed next-state coordinates.
2. **Sensory surprise:** normalized magnitude of the observed state change from the previous state. This is deliberately an operational change-magnitude proxy, not a probabilistic surprisal score.
3. **Reward/goal error:** expected reward minus observed reward.

Only after these values are measured does the optional learning update change transition weights and residual variance. Callers can disable updates to obtain a frozen evaluation.

## Deterministic evaluation and baselines

Run:

`python -c "from artificial_human.neural import run_world_model_comparison; print(run_world_model_comparison())"`

The comparison:
- trains a transition model on a deterministic training split;
- evaluates the learned model on held-out state/action transitions against a frozen zero-update model;
- introduces controlled transition shifts and measures error recovery and uncertainty response;
- evaluates the existing `PredictionSystem` lexical novelty heuristic on deterministic event descriptions and reports accuracy, precision and recall against high numeric transition-error events.

The lexical heuristic is not a numeric dynamics model; the classification metrics compare it as an event-surprise baseline. The deterministic synthetic task does not establish generalization to real environments.

Focused tests:

`pytest -q tests/test_predictive_world_model.py tests/test_cognitive_dynamics.py`

## Research basis

The implementation is informed by system-identification principles: fit a transition predictor on observed state/action/outcome data, control model complexity, and evaluate predictive performance on held-out trajectories. Predictive-processing literature also motivates explicit predictions, prediction errors, precision/uncertainty, and iterative updating. The implementation intentionally uses a simple interpretable linear baseline rather than claiming to reproduce hierarchical Bayesian or spiking predictive-coding models.

References:
- Pillonetto, Chen & Ljung (2026), *A unified view of system identification in the learning era*: https://www.nature.com/articles/s44287-026-00345-z
- van Zwol et al. (2026), *Predictive Coding Networks and Inference Learning: Tutorial and Survey*: https://doi.org/10.1145/3797870
- N'dri et al. (2026), *Predictive coding with spiking neural networks: A survey*: https://www.sciencedirect.com/science/article/pii/S0893608025012523
- Hodson, Mehta & Smith (2024), *The empirical status of predictive coding and active inference*: https://www.sciencedirect.com/science/article/pii/S0149763423004426

## Limitations and follow-up

This is a linear, one-step-learned transition model. Multi-step rollout is recursive use of that one-step model; it is not a separately trained long-horizon predictor. Uncertainty is residual-based and approximate. The synthetic task does not cover high-dimensional observations, partially observed latent-state inference, nonlinear dynamics, model ensembles, calibrated predictive intervals, planning, or active sensing. Predictive coding is a theoretical family with multiple algorithms and limited empirical consensus; no claim of biological equivalence is made.
