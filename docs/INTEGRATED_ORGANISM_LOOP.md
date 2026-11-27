# Integrated organism-loop research prototype

This PR adds a deterministic, bounded, process-local vertical slice across the
neural learning and organism roadmap. It is a research prototype, not proof of a
biological brain, intelligence, sentience or consciousness.

## Included mechanisms

- #24: pair-based STDP-inspired synapse updates with bounded weights and traces.
- #25: bounded homeostatic activity tracking and threshold adaptation.
- #26: deterministic distributed population encoding and uncertainty.
- #27–#28: typed synthetic vision, audition, touch, proprioception and interoception
  events. No camera, microphone or device is accessed.
- #29: bounded linear world prediction with prediction-error updates.
- #30–#31: bounded episodic RAM memory and partial-cue similarity recall.
- #32: deterministic action choice and reward-value updates in a restricted action set.
- #33: adaptive scalar forward prediction.
- #34: bounded one-dimensional simulated environment.
- #35: tick-based development stages, not clinical or human-age predictions.
- #36–#38: integrated manually ticked loop with pause, resume, reset and snapshots.
- #37: review-only self-development proposals; no execution, file writes, commit,
  merge or deployment.
- #39: deterministic unit and integrated-loop regression tests.

## Explicitly incomplete

This does not complete all acceptance criteria in #24–#40. The STDP rule is a
simplified hypothesis, not a comparison against comprehensive Hebbian/BCM and
fixed-weight behavioural baselines. The homeostat and predictor are simplified.
Sensory encoders accept synthetic numeric features, not raw media. Episodic memory
is not a full set of distinct semantic, procedural and spatial systems. Action
learning is not a robust RL comparison. The environment is one-dimensional.
Development is a software-stage scaffold. The loop is manually ticked rather than
a production continuous scheduler with backpressure. Self-development creates a
review proposal record only; isolated worktree execution and immutable evidence
bundles remain future work.

All active state is in process memory and resets on object/process recreation.
No database, LLM, network, device or physical actuator is required.

Run focused tests with `pytest -q tests/test_organism_loop.py`.
