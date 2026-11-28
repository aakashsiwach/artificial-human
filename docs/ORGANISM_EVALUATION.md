# Organism evaluation: fixed-seed baselines, ablations and limits

## Purpose

This evaluation consolidates existing, independently specified neural and
organism tasks into one reproducible offline run. It measures behaviour of
current code against explicit controls; it does not add a cognitive mechanism
or collapse different tasks into one intelligence score.

Run from the repository root:

    python scripts/evaluate_organism.py
    python scripts/evaluate_organism.py --seed 17 --seed 23 --seed 41 \
      --integration-ticks 60 --neural-steps 20000 \
      --output benchmark-output/organism-evaluation.json

The script uses fixed seeds, synthetic data and in-process computation only.
It does not call an LLM, download model weights, access a database, or contact
external services. CI runs it on Python 3.13 and uploads the JSON report as an
offline benchmark artifact.

## Evidence map

| Mechanism / task | Evidence and baseline | What is measured |
| --- | --- | --- |
| Spiking microcircuit | Recurrent LIF vs feed-forward, disconnected, inhibition-off and a matched-size fixed-rate reference | Spikes, event delivery, dropped/pending events and final membrane state |
| Developmental learning | Experience-dependent program vs frozen plasticity and mature-from-start controls | Cue-evoked prediction spikes and association weight change |
| Action selection | Learned contextual policy vs seeded random and fixed-policy baselines | Held-out accuracy and mean reward |
| Adaptive control | Learning-enabled forward model vs frozen model | RMSE, post-disturbance error, recovery and learned parameters |
| Noisy sensory/body loop | Fixed-seed episodes at standard and higher noise/delayed-feedback conditions | Goal success, final distance/velocity, sensory retention, neural activity and prediction error |
| Continuous organism integration | Full loop vs attention, working-memory, episodic-memory, LIF and learned-action ablations | Reward, prediction error, activity, attention, memory and learning updates |
| Engineering resources | Timed suites, neural throughput baseline, Python tracemalloc | Host-specific elapsed time and Python-traced memory |

The continuous organism ablation report measures the loop merged in PR #62.
The noisy embodied task is a separate, earlier sensory/controller experiment.
The runner keeps the two result sets separate rather than implying they are
the same architecture.

## Reproducibility and interpretation

The default seed set is 17, 23 and 41. The adaptive-control sensitivity grid
uses sensor-noise standard deviations 0.00, 0.02 and 0.05 with feedback
delays 0, 2 and 5 steps for each seed. Closed-loop episodes report both the
standard condition (noise 0.01, delay 2) and a stress condition (noise 0.03,
delay 5).

Every per-seed outcome remains in the JSON. Failed episodes are not discarded.
No performance threshold is imposed: throughput and traced memory depend on
the host; task-level thresholds must be justified by repeated-run variance,
task validity and the relevant baseline before becoming CI gates.

For a mechanism-specific evaluation, retain the existing reports and tests:
docs/COGNITIVE_EVALUATION.md,
docs/benchmarks/neural-simulation-baseline.md,
the developmental program comparison, action-selection comparison,
adaptive-control comparison and the body/environment closed-loop tests.
This consolidated runner is a reproducibility entry point, not a replacement
for those more detailed research records.

## Failure cases and safety boundaries

The runner validates its seed set and work limits. Existing tests cover
malformed sensory input, absent neural-substrate ablation, fixed action
allowlists, bounded state-action storage, body bounds, episode limits,
delayed feedback, noisy sensory input and reset behaviour. Reported failures
remain visible in the artifact for human interpretation.

These results use synthetic tasks and reduced computational mechanisms.
They do not establish biological fidelity, human-like cognition, intelligence,
consciousness, sentience or a complete artificial human.
