# Neural episodic engrams and temporal replay

## Purpose and scientific hypothesis

Issue #31 adds a deterministic circuit experiment to the existing consent-scoped
episodic store and sequence-replay network.

The computational hypothesis is that contextual binding and sparse event codes
can make similar episodes more distinguishable, while convergent cue input to a
candidate engram and recurrent output to missing-feature units can support
partial-cue completion. A shared inhibitory unit provides a simple competition
mechanism. This is a small falsifiable software abstraction, not a hippocampal
circuit replica.

Research motivation includes hippocampal pattern separation/completion and
complementary learning systems. These are inspirations for testable abstractions,
not claims that the code reproduces dentate gyrus, CA3, or human episodic memory.

## Data and runtime boundary

- The existing episodic store remains the source of owner-visible event content.
- The neural circuit stores keyed token-feature digests, a keyed context digest,
  two identity features, source ID, owner ID, consent ID, event time, salience,
  strength and access count. It does not store the event text or raw context ID.
- The feature key is random per process and is never exported in snapshots.
  Feature codes and learned strengths are volatile RAM state and reset on restart.
- Source deletion, consent revocation, owner erasure and runtime reset purge the
  neural engram index together with the other specialized memory stores.
- Recall candidates are resolved again through the existing owner-scoped
  repository before their source records enter the cognitive cycle.
- No SQL, database, vector database, external persistence, LLM call, camera,
  microphone or unapproved sensor is used.

## Mechanism

1. Encoding normalizes event tokens and maps them through a per-process HMAC
   key. Context and source-specific identity features are added to a sparse code.
2. Recall ranks owner-visible candidates with a deterministic Dice-overlap
   baseline and runs a bounded LIF simulation.
3. Query feature neurons drive candidate engram neurons through excitatory
   synapses. An active candidate excites its own missing-feature neurons.
   Candidate output also activates a shared inhibitory unit whose outputs
   suppress competing candidate activity.
4. The report exposes candidate activation, completion, baseline score,
   latency, interference and pattern-separation measurements.
5. After an engram candidate survives attention selection, its strength is
   reinforced. Temporal sequence replay remains a separate bounded mechanism:
   replay can add adjacent events, and selected replay transitions are
   strengthened by the existing replay network.

The six-step circuit, neuron parameters, synaptic weights and thresholds are
engineering choices for a small synthetic task. They are not calibrated
biophysical values.

## Evaluation and baselines

The tests use fixed synthetic episodes and deterministic cues. They measure:

- partial-cue engram activation and missing-feature completion;
- similarity/interference between overlapping events;
- reduction in pairwise code similarity after adding context/identity features;
- owner and context isolation;
- consent/source/owner deletion and capacity eviction;
- retrieval-dependent strengthening;
- integration into the cognitive cycle without an LLM.

A high interference score means the top two symbolic candidates have similar
scores. The pattern-separation index is mean semantic-feature Jaccard similarity
minus mean sparse-code Jaccard similarity for the current candidate set. These
are engineering metrics, not established cognitive or neural measures.

## Limitations and next experiments

The code uses token features rather than learned sensory embeddings, and
identity features deliberately improve code distinctiveness. It does not model
place cells, grid cells, theta/gamma timing, replay during sleep, multi-scale
hippocampal anatomy or biological consolidation. The inhibitory unit is a
minimal competition control, not a detailed interneuron population model.
Synthetic results must not be reported as evidence of human-like memory.

Future work should compare the LIF circuit with no-recurrence, no-inhibition,
identity-code ablation, symbolic-only and fixed-rate controls on held-out
episode sets before tuning its parameters further.
