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


## Research anchors

These sources motivate the *families of mechanisms* represented here. They do
not validate this implementation's parameter values, milestones, or simplified
one-dimensional task.

- Bi, G.-Q. & Poo, M.-M. (1998). [Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type](https://doi.org/10.1523/JNEUROSCI.18-24-10464.1998). *Journal of Neuroscience*, 18(24), 10464–10472. Motivates the timing-dependent potentiation/depression hypothesis.
- Turrigiano, G. et al. (1998). [Activity-dependent scaling of quantal amplitude in neocortical neurons](https://doi.org/10.1038/36103). *Nature*, 391, 892–896. Motivates a homeostatic activity-regulation mechanism.
- Hensch, T. K. (2005). [Critical period plasticity in local cortical circuits](https://doi.org/10.1038/nrn1787). *Nature Reviews Neuroscience*, 6, 877–888. Motivates the hypothesis that inhibitory circuit maturation and experience can regulate periods of heightened plasticity.
- Huttenlocher, P. R. & Dabholkar, A. S. (1997). [Regional differences in synaptogenesis in human cerebral cortex](https://doi.org/10.1002/%28SICI%291096-9861%2819971020%29387%3A2%3C167%3A%3AAID-CNE1%3E3.0.CO%3B2-Z). *Journal of Comparative Neurology*, 387, 167–178. Shows why cortical developmental timing should not be represented by one universal pruning schedule.
- Fields, R. D. (2015). [A new mechanism of nervous system plasticity: activity-dependent myelination](https://doi.org/10.1038/nrn4023). *Nature Reviews Neuroscience*, 16, 756–767. Motivates treating conduction properties as potentially experience-sensitive.

The implementation intentionally uses global, coarse engineering proxies. It
does not model region-specific critical periods, glial cell biology, oligodendrocyte
lineage, extracellular matrix, receptor trafficking, dendritic morphology, or
the detailed cellular mechanisms described in these papers.
