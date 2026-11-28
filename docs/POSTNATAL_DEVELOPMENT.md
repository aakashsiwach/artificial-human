# Postnatal development on the neural simulation substrate

## Purpose

`DevelopmentalProgram` extends the coarse prenatal schedule onto the same
in-memory LIF neuron/synapse simulation used by the active neural runtime. It
does not create a second abstract population network.

## One substrate, separate clocks

- The neural simulation advances in deterministic millisecond steps.
- Prenatal time advances explicitly from 0 to 280 simulated gestational days.
- Postnatal development advances in separate, caller-supplied experience units.
  These units are engineering controls, not calendar days or individual-age
  predictions.
- Advancing development changes the neural substrate but does not itself
  advance the neural simulation clock.
- Sensory experiences are explicit, bounded events. They are not persisted.

## Developmental mechanisms

- **Synaptogenesis proxy:** source-defined candidate connections are added at
  coarse prenatal and postnatal milestones.
- **Cell maturation proxy:** membrane time constants and refractory periods
  move within bounded ranges.
- **Myelination/conduction proxy:** active synaptic delays are shortened within
  a bounded range. Updating a delay affects future spikes; already queued
  events retain their scheduled delivery time.
- **Experience-dependent plasticity:** paired sensory experience updates
  synaptic weights through an STDP-inspired rule, gated by developmental
  maturity.
- **Homeostasis:** a slow per-neuron input-gain controller limits runaway or
  quiescent activity.
- **Pruning proxy:** low-use candidate synapses can be functionally silenced
  (weight zero) during late maturation. This is functional pruning, not a claim
  to reproduce anatomical synapse elimination.

## Evaluation

`run_developmental_comparison` presents a tactile cue followed by an
auditory outcome, then probes whether the cue alone activates a prediction
neuron. It compares:
1. experience-dependent development;
2. the same developmental trajectory with plasticity frozen; and
3. a mature-from-start topology with plasticity frozen.

The report compares cue-evoked prediction spikes and the learned
association-to-prediction synaptic weight. This is a small discriminative
mechanism test, not evidence of human cognition or biological equivalence.

## Migration boundary

The older `PrenatalBrain` remains available as a compatibility/reference
implementation for its historical aggregate population model. The active
postnatal path does not instantiate it; it uses the shared LIF substrate and
reuses only the prenatal phase and sensory-onset schedule. Parity tests keep
the schedule aligned while the legacy aggregate model remains isolated pending
a later removal decision.

## Limitations and safeguards

- Developmental stages and unit thresholds are computational hypotheses and
  engineering choices, not clinical or individual biological timelines.
- Neurons are reduced LIF models; synaptic learning is an STDP-inspired rule.
- Myelination and pruning are bounded software proxies, not axonal biology.
- No fetal, clinical, developmental-disorder, consciousness, or sentience
  prediction is made.
- State, synapses, activity and learning remain in process memory and reset
  with the process. No database or external service is used.
