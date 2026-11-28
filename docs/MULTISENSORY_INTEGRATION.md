# Multisensory integration and perceptual state estimation (issue #28)

## Scope

This module adds a deterministic synthetic scalar-state task over already encoded sensory percepts. It is an engineering model, not a biological, clinical, or validated human-perception model. No device APIs or real-world sensing are introduced.

The input to the integrator is a `SensoryEstimate`: one normalized scalar estimate decoded from a modality's `PopulationCode`, a positive variance proxy, an integer timestamp, and an explicit source/association identifier. `estimate_from_percept` derives the estimate from an `EncodedPercept`. It never stores the original raw event values.

The current synthetic shared-state demonstration uses vision and proprioception because their first feature channels are signed means in the fixture encoders. Other modalities can be represented by `SensoryEstimate` once a task defines a semantically calibrated common feature; arbitrary modality features must not be fused merely because they are numbers.

## Model

1. **Independent estimates:** each modality retains its own estimate and variance.
2. **Reliability-weighted fusion:** under a common-cause hypothesis, estimates are combined by inverse-variance precision weighting: `sum(x_i / variance_i) / sum(1 / variance_i)`.
3. **Causal arbitration:** a common-cause probability is calculated from a configured prior and simple Gaussian penalties for normalized spatial disagreement and timestamp span. Explicitly different source IDs force segregation. This is a deliberately small likelihood model, not a fitted Bayesian causal-inference model.
4. **Model averaging:** the fused estimate is blended with the selected target-modality estimate using the common-cause probability. The target-modality estimate is the independent/segregated baseline.
5. **Neural routing:** the resulting state, common-cause probability, and reliability summary are encoded as a population code and can be projected into a configured LIF network with `project_integrated_state`.

The conversion from population-code uncertainty to variance uses a documented configurable engineering mapping. It is not calibrated against empirical sensor noise and must not be interpreted as a statistically valid confidence interval.

## Missing, conflicting and asynchronous modalities

- Missing modalities are returned in `missing_modalities`; they do not receive imputed values or synthetic evidence.
- A single available modality passes through unchanged and does not claim fusion.
- Different source IDs are treated as separate causes.
- Large disagreement reduces the common-cause probability and preserves the target modality's estimate.
- Temporal span reduces common-cause probability; future-dated observations are rejected.
- Duplicate estimates for a modality, unbounded modality counts, invalid timestamps, and invalid variances are rejected.

## Baselines and deterministic task

Run:

`python -c "from artificial_human.neural import run_multisensory_comparison; print(run_multisensory_comparison())"`

The deterministic task compares:
- independent target-modality estimate,
- naive equal-weight concatenation/average,
- unconditional inverse-variance fusion,
- causal integration (reliability fusion gated by common-cause evidence).

It includes trials with missing proprioception, deliberately conflicting cues, and timestamp offsets. The report returns mean absolute error (MAE) and the number of missing/conflicting trials. The task is a controlled synthetic demonstration; its deterministic noise pattern and parameter values are not evidence of biological fidelity or general real-world performance.

Run focused tests:

`pytest -q tests/test_multisensory_integration.py tests/test_sensory_encoders.py`

## Research basis and limits

The design follows the broad computational principles of reliability-weighted cue combination and Bayesian causal inference: cue precision affects fusion, while temporal/spatial correspondence informs whether signals should be combined or segregated. The implementation uses simplified deterministic equations and fixed parameters; it does not implement full posterior distributions, learned priors, calibration, causal-structure learning, or the neural hierarchy described in the literature.

References:
- Seilheimer, Rosenberg & Angelaki, *Models and processes of multisensory cue combination* (2014): https://pmc.ncbi.nlm.nih.gov/articles/PMC3980502/
- *The neural dynamics of hierarchical Bayesian causal inference in multisensory perception* (2019): https://www.nature.com/articles/s41467-019-09664-2
- *Causal Inference in the Multisensory Brain* (2019): https://doi.org/10.1016/J.NEURON.2019.03.043

## Limitations and follow-up

The scalar task does not yet solve high-dimensional perceptual state estimation, cross-modal feature calibration, spatial reference-frame transforms, online learning of common-cause priors, or robust handling of multiple simultaneous objects. Those remain separate research and implementation work. No biological completion claim is made.
