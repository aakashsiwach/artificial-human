# Prenatal developmental substrate (experimental)

## Purpose

This milestone establishes an initial developmental substrate for Artificial Human. It starts with an abstract, stable circuit scaffold and a developmental programme. As simulated gestational time advances, population maturity and internal activity change; explicit sensory cues may modify existing connection efficacy through a bounded local plasticity rule.

This is not a biological fetal-brain simulation, a human developmental model, a neural-cell model, or evidence of learning comparable to a fetus. It is a small, falsifiable software experiment intended to prevent the project from treating a database of remembered text as its foundational learning mechanism.

## Dating and developmental sequence

The simulator uses **obstetric gestational age**: week 0 starts at the last menstrual period (LMP), approximately two weeks before conception. Forty weeks (280 days) is the conventional term reference. Biological milestones overlap and vary between sources and individuals; software phase boundaries are only coarse organisational aids.

| Approx. gestational weeks | Developmental theme | Model treatment |
| --- | --- | --- |
| 0–4 | Pre-neural embryonic development | The abstract blueprint exists; neural populations have zero maturity and sensory learning is disabled |
| 5–7 | Neural induction, neurulation and early patterning | Maturation begins in early populations |
| 8–11 | Regional patterning and progenitor expansion | Deterministic low-amplitude intrinsic activity begins |
| 12–21 | Progenitor proliferation, neuronal migration and cortical organisation | Population maturity advances; internal activity refines existing connections |
| 22–27 | Axonal/dendritic growth and early circuit formation | Early circuit efficacy becomes increasingly plastic |
| 28–35 | Sensory pathways continue to mature | Explicit tactile, vestibular and auditory inputs may influence selected paths, subject to conservative model gates |
| 36–40 | Late gestation; brain networks remain immature | The simulator reaches its term boundary without declaring the network mature |

Neural tube formation, proliferation, migration, cortical organisation, synaptogenesis and myelination overlap; they are not sequential stages with sharp biological cut-offs. Much synaptogenesis and myelination continue after birth. The milestone weeks in code are simplified reference points, not individual clinical predictions.

## Architecture

The versioned blueprint `prenatal-scaffold-v1` defines seven abstract neural populations and ten directed projections. They represent functional computational roles—sensory input, recurrent integration, association, sequence/context, action prediction, regulation and inhibitory control. They are not one-to-one anatomical regions, and capacity units are not neuron counts.

Two aspects are separated:

1. **Structural scaffold:** population identities and projection endpoints are fixed. The `structure_signature()` method makes this invariant inspectable.
2. **Development and adaptation:** each population has a maturity value; each existing projection has a bounded efficacy (`strength`) updated by low-amplitude intrinsic co-activity and explicit experience. The abstract population maturity windows intentionally extend beyond 40 gestational weeks, so no population reaches 100% maturity at the term boundary. Their exact values are model parameters, not biological measurements; postnatal development is not yet simulated.

The plasticity rule is a bounded Hebbian-style update:

~~~text
delta_weight = learning_rate * developmental_gate * pre_activity
               * post_activity * (1 - current_weight)
~~~

It is an engineering hypothesis. It does not reproduce receptor dynamics, spike-timing-dependent plasticity, cell birth/death, radial glia, neurotransmitter systems, glial development, synaptic pruning or a detailed 3-D connectome. Inhibitory polarity is recorded as a property, but there is no biophysical excitation/inhibition simulation.

## Learning without a database of experiences

`PrenatalBrain` is in-memory only. The `expose()` method accepts a modality, intensity and duration; it does not accept or save text, audio, images or a stimulus ID. After processing, the cue is discarded. Only mutable network efficacy, current population activity and aggregate counters remain in the object.

This makes the initial adaptation **channel-level**, not semantic. Repeated simulated auditory exposure can strengthen the abstract auditory path; the system cannot remember a melody, recognise a voice or infer language. A later milestone will need distributed feature representations and tasks that distinguish stimulus-specific learning from mere strengthening of a sensory channel.

The object is not wired into SQLite, long-term memory, training-example storage, ambient sensors, microphone, camera or network input. Future persistence must be a separately designed, owner-controlled checkpoint mechanism; it must not silently turn every exposure into a database record.

## Sensory gates

The conservative simulated onset gates are tactile at week 10, vestibular at week 12 and auditory at week 24. These are **simulation policy choices informed by broad developmental evidence**, not universal biological thresholds. A modality submitted earlier is not processed. The API accepts only scalar intensity and duration; no real-world sensory equipment or ambient capture is enabled. In-utero sound is transmitted through maternal tissue and amniotic fluid and differs substantially from ordinary postnatal listening.

## Reproduce the experiment

Install the development extra, then run:

~~~bash
python -m pip install -e ".[dev]"
artificial-human prenatal --days 280
artificial-human prenatal --days 280 --seed 7 \
  --exposure 25:auditory:0.7 \
  --exposure 30:auditory:0.7 \
  --exposure 34:tactile:0.4
~~~

An exposure has the form `GESTATIONAL_WEEK:MODALITY:INTENSITY`. The CLI advances to each requested week, applies the scalar cue, and continues to the selected day. No per-exposure content is written to disk. Use `--days` to stop the simulation earlier.

Automated tests verify deterministic development, maturity/phase progression, pre-onset input rejection, bounded experience-dependent strengthening, unchanged topology, no retained stimulus history, and CLI input/output validation. A no-exposure trajectory is the baseline; repeated exposures are the plasticity condition. These are software-mechanism checks, not validation against real fetal data.

## Evidence and limits

The high-level schedule is informed by developmental reviews and summaries:

1. **Fetal Brain Development: Regulating Processes and Related Malformations.** Reviews neurulation, proliferation, migration and cortical organisation. [PMC9224903](https://pmc.ncbi.nlm.nih.gov/articles/PMC9224903/)
2. **The Basics of Brain Development.** Reviews brain development before and after birth. [PMC2989000](https://pmc.ncbi.nlm.nih.gov/articles/PMC2989000/)
3. **The Effect of Musical Stimulation and Mother’s Voice on the Early Development of Musical Abilities.** Reviews prenatal auditory development and studies of fetal auditory responses. [PMC8393253](https://pmc.ncbi.nlm.nih.gov/articles/PMC8393253/)

These are background reviews, not data used to fit simulator parameters. Development windows and learning rates are explicit engineering assumptions. Biological fidelity would require primary developmental datasets, alternative plasticity rules, cell-type-specific models, realistic temporal activity and independent behavioural tasks.

## Privacy and safety

- No user or child data are collected.
- No ambient sensing or real-world input is enabled.
- This simulator does not provide pregnancy, nutrition, supplement, exercise or medical advice.
- It does not claim consciousness, sentience, personhood or biological equivalence.
- No database writes or external actions occur.


## Migration status

The original `PrenatalBrain` aggregate-population implementation is retained
as a compatibility and parity reference. The active developmental path is now
`DevelopmentalProgram`, which advances the shared in-memory LIF neuron/synapse
substrate through prenatal and postnatal stages. It reuses this document's
coarse prenatal phase and sensory-onset schedule but does not instantiate a
second aggregate network. See `docs/POSTNATAL_DEVELOPMENT.md`.
