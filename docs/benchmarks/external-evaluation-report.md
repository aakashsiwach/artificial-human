# Cognitive evaluation report

- Generated: 2026-11-24T18:23:26+00:00 · seed `20261124` · Python 3.12.10 (Windows-10)
- Embedding backend: `BAAI/bge-small-en-v1.5` (fastembed, in_process)
- Suites: longmemeval, digit-span

## 1. Evaluation purpose

This report measures the cognitive subsystems that exist on `main` today:
attention, working memory, the lexical prediction/novelty signal, long-term
memory recall, and the integrated multi-turn cycle. It establishes a
reproducible baseline *before* new mechanisms are introduced, as Stage 1 of the
[Research Charter](../RESEARCH_CHARTER.md) requires. No algorithm was changed to
improve these numbers.

Four kinds of evidence are kept separate:

1. **Software correctness tests** (`pytest`) show that the code does what it
   specifies.
2. **Computational capability benchmarks** (this report) measure behaviour on
   defined, synthetic tasks against baselines.
3. **Human behavioural comparisons** are qualitative here. The report notes where
   software behaviour resembles or departs from well-replicated human findings,
   but no human data were collected or fitted.
4. **Biological fidelity** is **not** assessed. Nothing in this report shows that a
   component works like a brain region or circuit.

## 2. Scientific background

The full research record (phenomena, hypotheses, competing accounts, and what
each benchmark cannot establish) is in [docs/COGNITIVE_EVALUATION.md](../COGNITIVE_EVALUATION.md).
In brief:

- **Attention.** Selection among competing representations is shaped by
  goal-relevance and stimulus salience (biased competition: Desimone & Duncan,
  1995; saliency models: Itti & Koch, 2001). Salient distractors can capture
  selection against current goals (Theeuwes, 2010), and the top-down/bottom-up
  dichotomy is itself debated (Awh, Belopolsky & Theeuwes, 2012). The software
  combines two supplied numbers linearly; it does not model competition dynamics.
- **Working memory.** Capacity is limited (Miller, 1956; about four chunks in
  Cowan, 2001). Forgetting in short-term memory is better explained by
  interference than by time-based decay (Lewandowsky, Oberauer & Brown, 2009;
  contrast Berman, Jonides & Lewis, 2009). Free recall shows primacy and recency
  (Murdock, 1962). Oberauer et al. (2018) list benchmark findings that models
  should explain. The software is a bounded store with salience decay.
- **Prediction and novelty.** Predictive-processing accounts (Rao & Ballard,
  1999; Friston, 2005) have mixed empirical support (Hodson, Mehta & Smith, 2024).
  Neural responses differ for semantic incongruity (Kutas & Hillyard, 1980), and
  hippocampal novelty/mismatch signals are an active research area (Kumaran &
  Maguire, 2007). The software computes token-set overlap only.
- **Long-term memory.** The memory benchmark addresses episodic and semantic
  recall (Tulving, 1972), temporal organisation (Howard & Kahana, 2002) and source
  monitoring (Johnson, Hashtroudi & Lindsay, 1993). See
  [docs/MEMORY.md](../MEMORY.md).

## 3. Task and dataset definitions

| Suite | Tasks | Ground truth |
| --- | --- | --- |
| Attention | Seeded candidate sets (4–64 candidates) in five conditions: clean relevance; informative salience; salient distractors; tied relevance; cross-source normalisation (memories and learning examples max-normalised separately, using the architecture's own function) | Generator-assigned usefulness grades 0/1/2; the system sees only noisy relevance and salience |
| Working memory | Delayed retention (3 targets, 0–12 distractors, capacities 3/5/7/9, distractor salience lower/equal/higher); similar vs dissimilar distractors; reactivation with updated content; seeded capacity pressure; serial position; serial order; newcomer admission; session isolation | Whether each encoded item remains in the workspace with the expected content |
| Prediction | 54 authored context/input pairs in 9 categories: exact repetition, near repetition, paraphrase, low-overlap related, unrelated, shared vocabulary with a different sense, contradiction, contextually expected, contextually unexpected | Author label `unexpected` (unrelated, other word sense, contradiction, script violation) vs expected |
| Memory | Existing synthetic memory benchmark (46 memories, two synthetic owners) plus source-attribution queries | Graded relevance per query, expected source, empty set for irrelevant queries |
| Integration | Six multi-turn scenarios through `CognitiveArchitecture` (A–F) | Scenario-specific behavioural checks |

## 4. Evaluation splits

| Suite | Tuning | Eval | What the tuning split is used for |
| --- | --- | --- | --- |
| longmemeval | 0 | 500 | none: every external question is held out |
| digit-span | 31 | 55 | fitting human reference models only (participant-level split) |

Generated tasks use RNG streams derived from (seed, suite, split, condition), so tuning and eval tasks never share a stream. Reported metrics use the eval split only.

## 5. Baselines

- **Attention.** `existing_attention` is the unmodified `AttentionSystem.select`
  (0.8 × relevance + 0.2 × salience, ties by id). Compared with relevance-only,
  salience-only, and input-order (deterministic) ranking. Ablations re-apply the
  formula with weights 0.6/0.4 and 0.5/0.5.
- **Working memory.** `existing_working_memory` is the unmodified `WorkingMemory`
  (decay × 0.92 per insertion, evict lowest salience after insertion).
  Compared with FIFO, LRU, and salience eviction without decay (ablation of decay).
- **Prediction.** `existing_prediction` is the unmodified `PredictionSystem`.
  Compared with content-word overlap (stop words removed, light stemming) and
  token Jaccard distance. With a local embedding model, an
  `embedding_distance_reference` (1 − maximum cosine to the context) is
  reported as a non-lexical reference, not as a proposed component.
- **Memory.** Lexical, semantic, hybrid and hybrid + temporal retrieval, plus
  ablations, as defined by the memory benchmark.

## 6. Metric definitions

- **Precision@K / Recall@K / MRR / nDCG@K.** Standard ranked-retrieval
  definitions, shared with the memory benchmark (`memory/benchmark/metrics.py`).
  nDCG uses exponential gain 2^grade − 1.
- **Pairwise ranking accuracy.** Share of candidate pairs with different grades
  that are ordered correctly in the full ranking.
- **Selection consistency.** Identical output on repeated runs, and an
  unchanged top-k set when the input order is shuffled.
- **Retention rate.** Share of probed items still present in the workspace.
  **Recall accuracy** also requires the expected (possibly updated) content.
- **Capacity violations.** Number of steps at which the workspace held more items
  than its capacity.
- **AUROC.** Probability that a randomly chosen unexpected input receives a
  higher novelty score than a randomly chosen expected input (ties count ½).
- **False / missed novelty rate.** False-positive and false-negative rates at the
  threshold that maximised balanced accuracy on the **tuning** split.
- **Novelty separation.** Mean novelty for unexpected inputs minus mean
  novelty for expected inputs.
- **Temporal answer accuracy / source attribution accuracy / contradiction
  recall.** Top result is a best-graded answer / top result comes from the
  expected source / all conflicting memories appear in the top 5.

## 7. Results by subsystem

## 8. Integrated multi-turn results

Not run.

## 9. Failure cases

**longmemeval** (showing 10 of 15)

- question_id: gpt4_76048e76; category: temporal-reasoning; evidence_turns: 2
- question_id: gpt4_5dcc0aab; category: temporal-reasoning; evidence_turns: 1
- question_id: gpt4_4edbafa2; category: temporal-reasoning; evidence_turns: 2
- question_id: gpt4_9a159967; category: temporal-reasoning; evidence_turns: 3
- question_id: 80ec1f4f; category: multi-session; evidence_turns: 2
- question_id: 5a7937c8; category: multi-session; evidence_turns: 3
- question_id: 10d9b85a; category: multi-session; evidence_turns: 2
- question_id: 5d3d2817; category: single-session-user; evidence_turns: 1
- question_id: 71017277; category: temporal-reasoning; evidence_turns: 1
- question_id: gpt4_d6585ce9; category: temporal-reasoning; evidence_turns: 1

## 10. Ablations

## 11. Runtime and resource use

| Suite | Wall-clock (s) | Peak traced Python memory (KiB) |
| --- | --- | --- |
| longmemeval | 3003.369 | 32322 |
| digit-span | 8.017 | 4001 |

Total: 3011.445 s. Network used by the evaluation: no.

## 12. Reproducibility

```
python -m pip install -e ".[dev]"
artificial-human cognition-eval --suite all --seed 20261124 --output-dir docs/benchmarks
# optional local semantic variants (downloads model weights once):
python -m pip install -e ".[dev,semantic]"
artificial-human cognition-eval --suite all --seed 20261124 --embedding-backend fastembed --output-dir docs/benchmarks
```

Apart from timing, memory and timestamps, results are a deterministic function of the seed, the code, and (for semantic variants) the embedding model version.

## 13. Limitations

- All datasets are synthetic, English-only and written by this project. They test
  defined behaviours of the software, not human-like cognition.
- Attention ground truth comes from the same generator that produces the
  observable scores, so results describe the formula's behaviour under stated
  input distributions. Overall averages depend on the chosen condition mix.
- Working-memory probes check presence in a software store. Humans recall by
  content and show similarity-based interference; this software does neither.
- Prediction labels are author judgements on 54 short items (18 tuning, 36 eval).
  Confidence intervals would be wide; small differences are not meaningful.
- The integration scenarios are few and hand-designed. A pass shows that one
  specified behaviour occurred, not that the behaviour generalises.
- No externally authored dataset is included. Candidates were reviewed but not
  adopted, because of non-commercial licence terms, real user conversations
  among their sources, or mixed per-source licences (see
  [docs/COGNITIVE_EVALUATION.md](../COGNITIVE_EVALUATION.md#external-datasets)).
- Runtime figures are single-run wall-clock times on the generating machine.
  Peak memory is Python allocations traced by `tracemalloc`.

## 14. Research questions for the next phase

1. Should attention normalise candidates from different sources on a common
   scale, and how should salience be gated so salient distractors cannot win?
2. Should working-memory forgetting be driven by interference (content
   similarity, overlap of features) rather than multiplicative decay, and can
   that reproduce primacy as well as recency?
3. Should retrieval use the active working-memory context (conversational state)
   so elliptical follow-ups can be resolved, and how would that be evaluated?
4. Should repeated inputs reactivate existing workspace entries (content-based
   identity) instead of creating new percepts?
5. What representation would let the prediction signal detect contradictions
   and script violations? A semantic distance alone does not (see the reference
   results), so this points towards explicit propositional or world-state models.
6. Which externally authored, permissively licensed evaluation material could
   replace or supplement the project-authored datasets?
7. These baselines are the comparison point for Stage 2 (associative memory and
   memory dynamics). Each new mechanism must improve a named metric here without
   regressing others.

## 15. Independently authored datasets (opt-in)

These suites read locally downloaded external data (see [EXTERNAL_DATASETS.md](EXTERNAL_DATASETS.md)). Raw data are never committed, and outputs contain no conversation text, question text, or participant identifiers.

```
artificial-human cognition-eval --suite longmemeval --suite digit-span \
  --longmemeval-path external-data/longmemeval/longmemeval_oracle.json \
  --digit-span-dir external-data/nemar/on003838/beh --report-stem external-evaluation \
  --output-dir docs/benchmarks
```

### LongMemEval Oracle: evidence retrieval on independently authored data

*Evidence category: computational capability benchmark (external, MIT, fully held out).*

Source file `longmemeval_oracle.json` (sha256 `821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c`, 15388478 bytes); 500 of 500 questions evaluated; 948 sessions, 10960 turns. All questions are held out; nothing was tuned on them.

**Turn-level evidence retrieval** (non-abstention questions with labelled turns)

| System | questions | turn_recall@1 | turn_recall@5 | turn_recall@10 | turn_recall@20 | turn_precision@5 | turn_ndcg@10 | turn_mrr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lexical | 470 | 0.359 | 0.755 | 0.862 | 0.916 | 0.255 | 0.694 | 0.685 |
| lexical_temporal | 470 | 0.332 | 0.728 | 0.839 | 0.887 | 0.245 | 0.664 | 0.654 |
| semantic | 470 | 0.289 | 0.704 | 0.841 | 0.891 | 0.237 | 0.629 | 0.619 |
| hybrid_temporal | 470 | 0.348 | 0.759 | 0.882 | 0.934 | 0.257 | 0.693 | 0.688 |

**Turn MRR / recall@10 by category**

| Group | n | lexical | lexical_temporal | semantic | hybrid_temporal |
| --- | --- | --- | --- | --- | --- |
| knowledge-update | 72 | 0.725 / 0.947 | 0.739 / 0.954 | 0.613 / 0.926 | 0.723 / 0.972 |
| multi-session | 121 | 0.628 / 0.749 | 0.582 / 0.707 | 0.637 / 0.741 | 0.653 / 0.757 |
| single-session-assistant | 56 | 0.767 / 0.982 | 0.703 / 0.982 | 0.555 / 0.982 | 0.673 / 1.000 |
| single-session-preference | 30 | 0.459 / 0.800 | 0.476 / 0.800 | 0.767 / 0.928 | 0.727 / 0.972 |
| single-session-user | 64 | 0.749 / 0.969 | 0.696 / 0.938 | 0.663 / 0.930 | 0.744 / 0.969 |
| temporal-reasoning | 127 | 0.702 / 0.828 | 0.673 / 0.794 | 0.575 / 0.762 | 0.671 / 0.832 |

**Turn MRR / recall@10 by history sessions**

| Group | n | lexical | lexical_temporal | semantic | hybrid_temporal |
| --- | --- | --- | --- | --- | --- |
| 1 session | 170 | 0.701 / 0.935 | 0.667 / 0.923 | 0.631 / 0.925 | 0.712 / 0.978 |
| 2 sessions | 229 | 0.712 / 0.899 | 0.695 / 0.882 | 0.642 / 0.851 | 0.727 / 0.906 |
| 3 sessions | 39 | 0.650 / 0.677 | 0.556 / 0.586 | 0.536 / 0.689 | 0.543 / 0.653 |
| 4+ sessions | 32 | 0.452 / 0.430 | 0.409 / 0.385 | 0.487 / 0.512 | 0.462 / 0.475 |

**Turn MRR / recall@10 by history turns**

| Group | n | lexical | lexical_temporal | semantic | hybrid_temporal |
| --- | --- | --- | --- | --- | --- |
| 11-20 turns | 138 | 0.665 / 0.914 | 0.645 / 0.900 | 0.649 / 0.899 | 0.721 / 0.954 |
| 21-40 turns | 255 | 0.703 / 0.863 | 0.669 / 0.838 | 0.625 / 0.828 | 0.697 / 0.872 |
| <=10 turns | 45 | 0.814 / 1.000 | 0.768 / 0.978 | 0.587 / 0.978 | 0.696 / 1.000 |
| >40 turns | 32 | 0.452 / 0.430 | 0.409 / 0.385 | 0.487 / 0.512 | 0.462 / 0.475 |

**Turn MRR / recall@10 by evidence position**

| Group | n | lexical | lexical_temporal | semantic | hybrid_temporal |
| --- | --- | --- | --- | --- | --- |
| early | 355 | 0.701 / 0.856 | 0.670 / 0.832 | 0.637 / 0.826 | 0.710 / 0.869 |
| late | 37 | 0.560 / 0.838 | 0.506 / 0.838 | 0.563 / 0.919 | 0.530 / 0.946 |
| middle | 78 | 0.672 / 0.897 | 0.651 / 0.872 | 0.561 / 0.876 | 0.664 / 0.910 |

**Turn MRR / recall@10 by evidence sessions**

| Group | n | lexical | lexical_temporal | semantic | hybrid_temporal |
| --- | --- | --- | --- | --- | --- |
| multi-session evidence | 283 | 0.706 / 0.838 | 0.676 / 0.810 | 0.635 / 0.806 | 0.705 / 0.838 |
| single-session evidence | 187 | 0.654 / 0.898 | 0.620 / 0.882 | 0.595 / 0.895 | 0.663 / 0.947 |

**Turn MRR / recall@10 by days since latest evidence**

| Group | n | lexical | lexical_temporal | semantic | hybrid_temporal |
| --- | --- | --- | --- | --- | --- |
| 0-1 days | 138 | 0.703 / 0.817 | 0.648 / 0.758 | 0.611 / 0.794 | 0.676 / 0.805 |
| 2-7 days | 206 | 0.668 / 0.881 | 0.639 / 0.870 | 0.645 / 0.875 | 0.701 / 0.912 |
| 31-180 days | 19 | 0.779 / 0.870 | 0.783 / 0.870 | 0.499 / 0.625 | 0.675 / 0.853 |
| 8-30 days | 106 | 0.675 / 0.880 | 0.664 / 0.875 | 0.596 / 0.875 | 0.677 / 0.925 |
| >180 days | 1 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |

**Turn recall by retrieval depth (hybrid_temporal)**: @1 0.348, @2 0.515, @3 0.638, @5 0.759, @10 0.882, @15 0.922, @20 0.934

**Session level and abstention**

| Variant | Session recall@1 | Session recall@5 | Session precision@1 | Abstention questions | Rejection rate | Mean results returned |
| --- | --- | --- | --- | --- | --- | --- |
| lexical | 0.648 | 0.992 | 1.000 | 30 | 0.000 | 11.770 |
| lexical_temporal | 0.639 | 0.967 | 0.981 | 30 | 0.033 | 10.900 |
| semantic | 0.633 | 0.950 | 0.979 | 30 | 0.100 | 8.770 |
| hybrid_temporal | 0.639 | 0.969 | 0.981 | 30 | 0.033 | 12.370 |

Owner leaks: 0; isolation-probe results: 0; questions without turn labels: 0.

**Checks**

- PASS: no cross-owner leakage
- PASS: isolation probe owner sees nothing

**Findings**

- lexical: turn recall@5 0.755, recall@20 0.916, MRR 0.685; abstention rejection rate 0.0.
- lexical_temporal: turn recall@5 0.728, recall@20 0.887, MRR 0.654; abstention rejection rate 0.0333.
- semantic: turn recall@5 0.704, recall@20 0.891, MRR 0.619; abstention rejection rate 0.1.
- hybrid_temporal: turn recall@5 0.759, recall@20 0.934, MRR 0.688; abstention rejection rate 0.0333.
- The primary variant (hybrid_temporal) found no evidence turn in its top 20 for 15 of 470 scored questions.
- Session-level metrics are uninformative on the Oracle file: every history session is an evidence session, so session precision is 1 by construction.
- This measures evidence retrieval only, not answer accuracy.

### Human digit-span reference (NEMAR on003838) vs software on identical sequences

*Evidence category: human behavioural comparison (external, CC0); not biological fidelity.*

86 behavioural files (manifest sha256 `d1d9524b3887864617e2c9b52af7f8a0edbd325e6c47d24373b5114448e97367`), 86 participants, 9276 trials, loads [5, 9, 13], 16 empty responses. Participant-level split: 31 tuning / 55 eval participants. Only aggregates are reported.

| Load | Humans: mean (SD) | Humans p10 / median / p90 | Humans first / middle / last third | Software: mean | Software first / middle / last third |
| --- | --- | --- | --- | --- | --- |
| 5 | 0.925 (0.059) | 0.852 / 0.933 / 0.983 | 0.979 / 0.913 / 0.905 | 1.000 | 1.000 / 1.000 / 1.000 |
| 9 | 0.460 (0.132) | 0.325 / 0.448 / 0.626 | 0.767 / 0.439 / 0.172 | 0.778 | 0.333 / 1.000 / 1.000 |
| 13 | 0.273 (0.088) | 0.162 / 0.269 / 0.396 | 0.610 / 0.202 / 0.026 | 0.538 | 0.000 / 0.600 / 1.000 |

**Held-out human serial-position curves** (mean proportion correct by position)

- Load 5: humans 0.979, 0.942, 0.902, 0.895, 0.905; software 1.000, 1.000, 1.000, 1.000, 1.000
- Load 9: humans 0.893, 0.769, 0.639, 0.541, 0.436, 0.340, 0.230, 0.133, 0.153; software 0.000, 0.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000
- Load 13: humans 0.802, 0.652, 0.533, 0.452, 0.371, 0.283, 0.167, 0.116, 0.074, 0.048, 0.027, 0.015, 0.014; software 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000

**Serial-position RMSE against held-out participants** (lower = closer)

| System | rmse |
| --- | --- |
| existing_working_memory_capacity_7 | 0.680 |
| recency_store_capacity_1_fitted_on_tuning | 0.592 |
| primacy_store_capacity_5_fitted_on_tuning | 0.284 |
| flat_per_load_mean_fitted_on_tuning | 0.232 |

**Checks**

- PASS: participants do not appear in both splits

**Findings**

- At load 13, held-out humans show a primacy gradient (first third 0.610, last third 0.026). The software workspace shows the reverse (first third 0.000, last third 1.000) because it keeps the most recent items.
- Among reference models fitted on tuning participants only, a primacy store (capacity 5) describes held-out serial-position curves better than a recency store (best capacity 1). This describes behaviour; it is not a claim about mechanism.
- Human serial recall and the software workspace are different systems. RMSE compares only serial-position curves on the same sequences; a low value would not imply a shared mechanism.
