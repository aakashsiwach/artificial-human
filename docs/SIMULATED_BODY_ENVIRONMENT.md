# Simulated body and closed-loop environment

## Goal

Connect explicit sensory events, multisensory state estimation, a recurrent
spiking microcircuit, contextual action selection, adaptive forward control,
and environment outcomes in one deterministic in-memory episode.

## Architecture and information flow

1. A bounded one-dimensional simulated body emits synthetic vision and
   proprioception events for position, plus proprioceptive velocity feedback.
2. Events pass through the permission-scoped sensory pipeline. All grants are
   explicit, synthetic-only, and ephemeral; raw events are not retained.
3. Vision and proprioceptive pose estimates are fused by the uncertainty-aware
   multisensory integrator. Velocity is decoded from a separate proprioceptive
   event. The event adapter applies the inverse of the population encoder's
   tanh normalization so decoded estimates remain on the simulator's physical
   [-1, 1] scale.
4. The fused sensory estimate drives the sensory input of the recurrent LIF
   microcircuit. Its spike trace is recorded for every simulation tick.
5. A contextual action-selection circuit chooses approach, hold, or recovery
   using the current estimated body state. The small initial policy is
   calibrated with bounded reward signals, then updated from actual task
   progress and stability.
6. The adaptive forward controller predicts the next velocity and emits a
   bounded simulated motor command. Sensory feedback is queued according to
   the configured delay and updates the forward model from prediction error.
7. The environment applies the command, returns the next body state, and
   computes progress/stability reward. The next loop consumes the resulting
   sensory events.

## Environment and evaluation

The task starts away from a fixed target and runs for a bounded number of
simulation steps. At a known midpoint, an external disturbance changes. The
body has bounded position and velocity; the command is constrained to [-1, 1].
The task succeeds only when final position and velocity are both within
specified tolerances. The report contains full action, sensory, neural-spike,
motor-command, outcome, reward, and delayed prediction-error traces.

Tests cover deterministic replay, action bounds, task completion, noisy and
delayed feedback, post-shift prediction error/adaptation, and ephemeral sensory
retention. Fixed seeds make the entire trace reproducible.

## Safety and limitations

- This is a scalar synthetic plant, not a human body, robot, or biomechanics
  simulator.
- The LIF circuit is a small recurrent computational microcircuit, not an
  anatomical sensory pathway.
- The action policy and forward controller are reduced computational models;
  their task success is not evidence of biological equivalence, general
  intelligence, consciousness, or subjective experience.
- There is no camera, microphone, physical sensor, network, robot, actuator,
  or real-world action path.
- Runtime state and learned parameters remain in bounded process memory and
  reset with the process. Sensory events are ephemeral.
- The model does not yet implement multi-joint control, muscle dynamics,
  tactile contact, body schema development, or a rich object-based world.

## Next integration work

Use this environment as the reproducible test bed for integrating the neural
kernel, sensory/world models, memory, action selection, and adaptive control
under #38. Add richer tasks only after baselines and ablations are defined.
