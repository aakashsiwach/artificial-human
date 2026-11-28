# Unified developmental neural substrate (experimental)

## Purpose

Issue #35 moves the prenatal scaffold onto the same deterministic LIF substrate
used by the neural simulation work. The seven functional population definitions
and ten prenatal projection definitions are instantiated as neurons and
synapses in one `NeuralSimulation`. `DevelopmentalNeuralSubstrate.from_prenatal()`
copies a `PrenatalBrain` snapshot into that network; subsequent development and
learning then occur in the LIF substrate rather than in a second population-rate
network.

This is a computational research abstraction. It is not a fetal, paediatric,
clinical, or individual human-development simulator.

## Developmental time

The API uses an integer `developmental_day` model clock, separate from the
LIF simulation's millisecond step counter and from wall-clock uptime.
`advance(days)` changes developmental stage and maturity without stepping
neurons. `run(steps)` or `step()` advances the neural clock independently;
the continuous wall-clock scheduler belongs to the later runtime integration.
The following epochs are engineering regimes chosen to exercise transitions
and are **not** a mapping from simulation time to an individual's chronological age:

| Model day | Epoch | Implemented change |
| ---: | --- | --- |
| 0–280 | Prenatal | Fixed source-defined population scaffold; gradual maturity and low-level intrinsic activity |
| 281–730 | Early postnatal | First staged connections are added; postnatal vision and proprioception inputs become available in the model |
| 731–3,650 | Childhood | Additional recurrent and inhibitory connections are added |
| 3,651–6,570 | Adolescence | Final predeclared connectivity refinement is added |
| 6,571+ | Mature model regime | Never-used staged connections are functionally silenced; base scaffold is retained |

Population maturity follows the prenatal blueprint through day 280. After that,
each population follows a bounded continuation curve derived from its source
maturation window. These curves are explicit engineering assumptions, not
developmental measurements or clinical predictions.

## Mechanisms

- **Population maturation:** a bounded maturity value adjusts each LIF neuron's
  dimensionless input gain. It does not create a count of biological cells.
- **Synaptogenesis:** a small, source-defined set of connections is added at
  postnatal model-stage boundaries. No arbitrary network growth is allowed.
- **Experience-dependent refinement:** bounded pair-based STDP-inspired weight
  updates operate on the actual LIF synapses. The plasticity gate depends on
  population maturity and a stage-specific modulation factor.
- **Functional pruning:** staged connections with no experience-driven pathway
  use are assigned zero weight in the mature model regime. The ten base
  projections are never pruned. This is a software ablation, not biological
  synapse elimination.
- **Sensory capability:** tactile, vestibular and auditory channels retain the
  prenatal onset gates. Vision, proprioception and interoception have explicit
  model availability gates. Inputs are scalar synthetic events only.
- **Intrinsic activity:** `run(steps)` supplies deterministic bounded currents
  to the same LIF network. Developmental advancement does not implicitly step
  neurons, so model age, neural time and wall-clock scheduling cannot be
  accidentally conflated. No device, database or LLM is involved.

The implementation intentionally does not claim to model myelination, glia,
cell-type-specific neurogenesis, endocrine development, detailed receptor
biology, or sleep-dependent developmental mechanisms. Those require separate
mechanistic specifications and discriminative evidence before implementation.

## Migration and compatibility

`PrenatalBrain` remains available as a compatibility/reference model so
existing callers and prenatal tests are not silently broken. Use
`DevelopmentalNeuralSubstrate.from_prenatal(prenatal)` to migrate its current
age, population maturity and projection strengths. The returned object's
ongoing state is held in the LIF network. No second prenatal rate network is
advanced by the new substrate.

The parity tests verify all seven population maturity values and all ten
projection efficacies across migration.

## Evaluation

Run:

```bash
pytest -q tests/test_prenatal_development.py tests/test_development_lifecycle.py
```

The developmental comparison runs a fixed-seed frozen-weight control and a
repeated-auditory-experience condition through the same model-time trajectory.
Both receive matched simulation-step budgets. A held-out probe runs without
plasticity and reports target-population spikes and pathway efficacy. The
comparison is a small synthetic mechanism experiment, not evidence of improved
human learning or biological fidelity.

The tests cover deterministic stage transitions, migration parity, staged
connectivity, sensory gates, bounded experience-dependent weight change,
functional pruning, reset, and reproducibility. Failure to improve the
held-out task must be reported rather than hidden by changing the metric after
seeing results.

## Limits and safety

- No user or child data are collected.
- No real camera, microphone, sensor, physical actuator or external action is used.
- Developmental days are not gestational age, chronological age or medical advice.
- The model does not establish sentience, consciousness, personhood or biological equivalence.
- Learned state is in RAM and resets with the process.
