# Adaptive forward prediction and motor control

## Research question

Can an online forward model predict the sensory consequence of a simulated
motor command, use sensory prediction error to adapt its dynamics, and improve
closed-loop tracking after a perturbation compared with the same controller
using frozen model parameters?

## Scientific motivation and scope

Cerebellar theories include forward models that predict sensory consequences
of motor commands and use mismatches between predicted and observed feedback
to support adaptation. This is a family of computational hypotheses, not a
single settled description of all cerebellar function.

The implementation reuses the project's bounded online linear transition
model. It predicts one-dimensional velocity from current velocity and a
simulated motor command. A simple inverse step chooses a command intended to
produce the desired next velocity. Delayed sensory prediction errors update
the forward model using bounded normalized-LMS learning.

This is not a cerebellar anatomical simulation, a Purkinje-cell model, a
climbing-fibre model, or a complete motor-control architecture. It does not
yet connect to a spiking microcircuit or the simulated body/environment issue
(#34). There is no real-world actuator interface.

## Computational hypothesis

- The plant has bounded velocity dynamics with damping, motor gain, and an
  external disturbance.
- The forward model predicts next velocity from current velocity and command.
- The controller uses the learned model to invert the desired next velocity
  into a bounded motor command.
- Sensory prediction error updates the model; optional feedback delay queues
  transitions before learning.
- Model uncertainty is estimated from bounded residual variance.
- Normative command limits remain independent of learned model parameters.

## Experiment and baselines

A fixed-seed one-dimensional tracking task follows a sinusoidal target. At the
midpoint, the environment's external disturbance changes. Both conditions
receive the same seeded sensor-noise stream:

1. Frozen baseline: same inverse controller with the initial forward-model
   parameters, but no learning.
2. Adaptive condition: same controller, with bounded online model updates.

The report includes post-warm-up tracking RMSE, post-shift RMSE, early/late
recovery RMSE, and the learned damping, control gain, and disturbance bias.
Tests cover deterministic replay, bounded commands, prediction-error learning,
feedback delay, noisy feedback, perturbation recovery, and reset behaviour.

This task is intentionally small and synthetic. Performance improvement is
engineering evidence for this particular plant and controller, not evidence
of biological equivalence or general motor intelligence.

## Runtime and safety

- All state and learned parameters remain in bounded process memory.
- The feedback queue is limited to at most 32 simulation transitions.
- The command is clamped to [-1, 1].
- The controller emits only a scalar simulated command; it cannot call devices,
  tools, networks, or real-world actuators.
- Reset discards pending feedback and restores source-defined model priors.

## Failure modes and next experiments

The current task does not model multi-joint biomechanics, muscle dynamics,
sensory modality fusion, efference-copy pathways, anatomical loops, or
developmental calibration. Future tests should vary plant gain/damping,
non-stationary disturbances, partial observability, longer delays, and
multi-dimensional control. The controller should be integrated with #34 only
through an explicit simulated action interface.

## Selected research motivation

- Wolpert, Miall & Kawato (1998), *Internal models in the cerebellum*, Trends
  in Cognitive Sciences 2:338–347. DOI: 10.1016/S1364-6613(98)01221-2.
- Shadmehr & Krakauer (2008), *A computational neuroanatomy for motor control*,
  Experimental Brain Research 185:359–381. DOI: 10.1007/s00221-008-1280-5.
- Tseng et al. (2007), *Sensory prediction errors drive cerebellum-dependent
  adaptation of reaching*, Journal of Neurophysiology 98:54–62.
  DOI: 10.1152/jn.00266.2007.
