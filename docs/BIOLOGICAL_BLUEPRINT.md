# Human-brain and organism blueprint

## Research principle

The [Research Charter](RESEARCH_CHARTER.md) is the authoritative project north star and contains the expanded neuroscience research map, evidence standards, technology-selection strategy, staged implementation sequence, bibliography, and context-recovery prompt. This blueprint is the compact biological-to-computational map; it must remain consistent with the charter and the actual implementation.

The human brain is the scientific reference for the artificial mind. We should study biological mechanisms and cognitive theories, translate selected functions into explicit computational models, and evaluate those models. A biological label must never be used to imply that ordinary software is a real organ or that an implementation reproduces the biology.

Brain functions are distributed across interacting networks. The table below is a research map, not a one-region/one-module anatomical claim.

## Cognitive architecture

## Levels of analysis and technology strategy

Research must connect cell types and synapses, local circuits, distributed network dynamics, behaviour, and causal evidence where available. A brain-region label is not a software specification. For each mechanism, distinguish an established empirical constraint, a supported but incomplete computational account, a contested theory, and a practical engineering choice.

Use modern foundation models, deep-learning runtimes, embeddings, graph and relational models, recurrent/state-space models, reinforcement learning, simulation, spiking neural networks, and neuromorphic hardware as tools for specific hypotheses. Select them through current compatibility, licence, privacy, performance, and maintenance checks. Do not assume that a newer model or a more biologically named architecture is automatically better. Define baselines, ablations, evaluation tasks, and failure criteria before adoption.

| Brain function / research area | Artificial Human subsystem | Current implementation |
| --- | --- | --- |
| Sensory pathways and perception | Perception | Explicit language percepts; multimodal sensing is future work |
| Selective attention | Attention | Bounded relevance/salience selection |
| Working-memory networks | Working memory | Capacity-limited volatile workspace |
| Hippocampal memory formation and recall | Memory | Consent-gated typed records; hybrid lexical/semantic/temporal retrieval with explanations |
| Neocortical knowledge representation | Semantic memory | Meaning-based recall through a frozen pretrained text-embedding model; structured concept learning is future work |
| Temporal organisation of episodic memory | Temporal memory | Explicit event, recorded and validity times; version links; rule-based temporal cues |
| Predictive processing | Prediction | Lexical novelty and context-overlap signal |
| Prefrontal executive functions | Executive control | Deterministic goal planning and approval gates |
| Language networks | Language interface | Model-backed language understanding and generation |
| Reinforcement and habit learning (including basal ganglia circuits) | Action selection and learning | Owner-approved demonstrations only; reinforcement mechanism is future work |
| Cerebellar prediction and error correction | Adaptive prediction | Textual novelty only; sensorimotor adaptation is future work |
| Affect and salience (including amygdala-related networks) | Appraisal and salience | Explicit salience input only; emotional inference is not implemented |
| Sleep-dependent consolidation | Sleep | Review proposals; automatic consolidation is not implemented |

## Integrated cognitive cycle

1. Perceive explicitly supplied input and retain source provenance.
2. Retrieve relevant, consent-visible memory and approved demonstrations.
3. Allocate bounded attention to competing candidates.
4. Update a limited working-memory workspace.
5. Assess mismatch/novelty against active context.
6. Prepare an executive goal plan and approval boundary.
7. Use a configurable language model as one component in the cycle.
8. Return an inspectable trace of the implemented operations.

The cycle is coordinated by `CognitiveArchitecture`. Each stage has an independent component, a documented input/output boundary, and tests. The novelty score and attention weights are engineering heuristics; they are not neural signals.

## Wider organism

| Biological inspiration | Digital subsystem | Responsibility |
| --- | --- | --- |
| Genetic inheritance (loose analogy) | Identity | Owner-authored values, preferences, and constraints |
| Neuronal communication | Nervous system | Typed internal events |
| Cardiovascular circulation | Circulation | Prioritised internal work scheduling |
| Metabolic regulation | Metabolism | Bounded compute, storage, and model budgets |
| Innate/adaptive immune functions | Immune system | Detect abnormal signals and recommend defence |
| Tissue repair / regeneration | Regeneration | Propose isolated repair work |
| Sleep | Sleep system | Prepare memory-review and consolidation proposals |

## Immune system roles

- **Dendritic / sentinel-inspired role:** collect evidence and classify unusual signals.
- **Neutrophil-inspired role:** recommend rapid containment for urgent incidents.
- **Macrophage-inspired role:** correlate evidence and prepare incident history.
- **B-cell-inspired role:** maintain reviewed patterns and known remediations.
- **T-cell-inspired role:** challenge proposed repairs against policy and regression evidence.
- **Stem-cell-inspired role:** provision a fresh, isolated worker environment.

These are software roles, not biological agents. High-impact actions remain subject to explicit approval.

## Memory systems

Memory types remain distinct because an event, a fact, a skill, a preference, a decision, and active working context are not interchangeable:

- **Episodic:** an event and its context.
- **Semantic:** facts and concepts.
- **Procedural:** how a task is performed.
- **Preference:** confirmed likes, dislikes, and interaction choices.
- **Decision:** choices together with reasons and outcomes.
- **Working:** temporary context for an active cognitive cycle.

Inferred traits must be marked as hypotheses, not facts. The owner must be able to inspect, correct, export, and delete retained information.

### Computational hypotheses for long-term memory (implemented)

Each hypothesis below is a functional analogue, measured in the [memory benchmark](benchmarks/memory-retrieval-report.md). None is a mechanistic model.

1. **Content-addressable recall** (inspired by hippocampal indexing and pattern completion). A dense embedding of a partial or paraphrased cue should retrieve the stored trace better than word overlap. *Measured:* paraphrase and conceptual categories improve over lexical-only. *Leaves out:* learned, owner-specific representations; pattern separation; attractor dynamics.
2. **Temporal organisation** (inspired by the temporal structure of episodic memory). Explicit event and validity times should let recall answer "now", "previously" and "in January" correctly. *Measured:* temporal and historical categories improve with temporal reasoning enabled. *Leaves out:* drifting temporal context and contiguity effects; the rules are hand-written.
3. **Source monitoring and belief revision.** Keeping every version with its source, and surfacing conflicts instead of overwriting, should keep historical and contradictory evidence available. *Measured:* both conflicting memories are retrieved, and superseded facts are excluded from "now" queries. *Leaves out:* learned source credibility and reconsolidation-like modification.

The memory model also leaves out consolidation over time, forgetting and interference, retrieval-dependent strengthening, emotional modulation, and multimodal binding. See [Long-term memory §11](MEMORY.md#11-comparison-with-biological-memory).

## Research requirements before claiming biological fidelity
## Target closed-loop research architecture

The current cycle is text-driven and ends in language generation. The longer-term research target is:

**Sensory and interoceptive input → perceptual state estimation → memory retrieval and association → attention and limited workspace → predictive world model → affective appraisal and homeostatic state → goal and action selection → environment interaction → outcome and prediction-error evaluation → learning, plasticity, and memory consolidation.**

This is a target architecture, not a claim that these modules are implemented. Each addition must have its own hypothesis, interface, benchmark, and safety boundary.


For each proposed subsystem, document the biological evidence, the computational hypothesis, the abstraction level, measurable predictions, test datasets, failure modes, and ethical constraints. Prefer small falsifiable models over decorative brain-region labels. Compare alternatives and record when the biological evidence is unsettled.

The project does not claim current AI is conscious, a literal human, or equivalent to a human brain.
