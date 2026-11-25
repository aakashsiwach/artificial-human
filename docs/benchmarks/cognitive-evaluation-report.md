# Cognitive evaluation report

- Generated: 2026-11-24T18:24:41+00:00 · seed `20261124` · Python 3.12.10 (Windows-10)
- Embedding backend: `BAAI/bge-small-en-v1.5` (fastembed, in_process)
- Suites: attention, working-memory, prediction, memory, integration

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
| attention | 50 | 150 | reserved; nothing is tuned |
| working-memory | 10 | 40 | reserved (capacity-pressure runs); nothing is tuned |
| prediction | 18 | 36 | novelty threshold per system (balanced accuracy) |
| memory | 13 | 53 | semantic similarity threshold |
| integration | 0 | 6 | none (fixed scenarios) |

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

### Attention: bounded selection among competing candidates

*Evidence category: computational capability benchmark (synthetic inputs).*

| System | mrr | ndcg@k | pairwise_accuracy | precision@k | recall@k | tasks |
| --- | --- | --- | --- | --- | --- | --- |
| existing_attention | 0.874 | 0.793 | 0.891 | 0.728 | 0.517 | 150 |
| relevance_only | 0.855 | 0.714 | 0.815 | 0.669 | 0.464 | 150 |
| salience_only | 0.682 | 0.529 | 0.662 | 0.526 | 0.367 | 150 |
| input_order | 0.587 | 0.298 | 0.517 | 0.354 | 0.244 | 150 |

**nDCG@k by condition**

| Condition | existing_attention | relevance_only | salience_only | input_order |
| --- | --- | --- | --- | --- |
| clean_relevance | 0.966 | 0.992 | 0.246 | 0.229 |
| salience_informative | 0.812 | 0.652 | 0.993 | 0.227 |
| salient_distractors | 0.518 | 0.973 | 0.070 | 0.332 |
| tied_relevance | 0.996 | 0.280 | 0.996 | 0.271 |
| cross_source_normalisation | 0.675 | 0.675 | 0.340 | 0.429 |

**nDCG@k by candidate-set size**

| System | 4 | 8 | 16 | 32 | 64 |
| --- | --- | --- | --- | --- | --- |
| existing_attention | 0.853 | 0.761 | 0.822 | 0.744 | 0.785 |
| relevance_only | 0.748 | 0.714 | 0.730 | 0.675 | 0.705 |
| salience_only | 0.633 | 0.516 | 0.543 | 0.500 | 0.453 |
| input_order | 0.434 | 0.282 | 0.259 | 0.250 | 0.263 |

Consistency: repeat-run identical 1.000, top-k invariant to input order 1.000.

**Generalisation to held-out conditions**

Weight tuned on tuning-split tasks of clean_relevance, salience_informative (20 tasks): relevance weight **0.8**. Tested on eval-split tasks of the unseen conditions salient_distractors, cross_source_normalisation (60 tasks).

| System | In-distribution nDCG@k | Held-out nDCG@k | Gap | Salient distractors selected / task | Weak learning examples selected / task | Highly relevant missed / task |
| --- | --- | --- | --- | --- | --- | --- |
| existing_attention | 0.889 | 0.596 | 0.293 | 0.967 | 0.733 | 1.850 |
| relevance_only | 0.822 | 0.824 | -0.002 | 0.117 | 0.733 | 1.167 |
| salience_only | 0.620 | 0.205 | 0.414 | 1.383 | 0.467 | 3.200 |
| input_order | 0.228 | 0.380 | -0.152 | 0.217 | 0.300 | 2.700 |
| tuned_weight_r0.8 | 0.889 | 0.596 | 0.293 | 0.967 | 0.733 | 1.850 |

**Findings**

- Generalisation (tune on clean + informative salience, test on salient distractors + cross-source): existing 0.889 → 0.596; relevance-only 0.822 → 0.824; weight tuned in-distribution (r=0.8) 0.889 → 0.596.
- Overall nDCG@k: existing 0.793, relevance-only 0.714, salience-only 0.529, input order 0.298.
- clean_relevance: existing nDCG@k 0.966 vs relevance-only 0.992.
- salience_informative: existing nDCG@k 0.812 vs relevance-only 0.652.
- salient_distractors: existing nDCG@k 0.518 vs relevance-only 0.973.
- tied_relevance: existing nDCG@k 0.996 vs relevance-only 0.280.
- cross_source_normalisation: existing nDCG@k 0.675 vs relevance-only 0.675.
- In the cross-source condition, 43.0% of selected slots went to grade-0 learning examples (25% of candidates), because each source is max-normalised separately before attention.
- Overall averages depend on the author-chosen mix of conditions (salience is informative in two of five). Read the per-condition results, not the overall average, to judge the formula.
- 38 of 150 eval tasks selected an irrelevant candidate while excluding a highly relevant one.

### Working memory: retention, interference, reactivation and eviction

*Evidence category: software behaviour + computational capability benchmark.*

| System | capacity_violations | delayed_retention_equal_salience | delayed_retention_lower_salience_distractors | delayed_retention_salient_distractors | updated_content_recall |
| --- | --- | --- | --- | --- | --- |
| existing_working_memory | 0 | 0.385 | 0.686 | 0.385 | 0.905 |
| fifo | 0 | 0.385 | 0.385 | 0.385 | 0.667 |
| lru | 0 | 0.385 | 0.385 | 0.385 | 0.809 |
| salience_no_decay | 0 | 0.385 | 1.000 | 0.385 | 1.000 |

**Retention vs intervening items** (lower-salience distractors, mean over capacities)

| Policy | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| existing_working_memory | 1.000 | 1.000 | 1.000 | 1.000 | 0.917 | 0.917 | 0.750 | 0.750 | 0.500 | 0.500 | 0.250 | 0.250 | 0.083 |
| fifo | 1.000 | 0.917 | 0.833 | 0.667 | 0.583 | 0.417 | 0.333 | 0.167 | 0.083 | 0.000 | 0.000 | 0.000 | 0.000 |
| lru | 1.000 | 0.917 | 0.833 | 0.667 | 0.583 | 0.417 | 0.333 | 0.167 | 0.083 | 0.000 | 0.000 | 0.000 | 0.000 |
| salience_no_decay | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

**Retention vs workspace capacity** (lower-salience distractors, mean over 0–12 distractors)

| Policy | 3 | 5 | 7 | 9 |
| --- | --- | --- | --- | --- |
| existing_working_memory | 0.462 | 0.615 | 0.769 | 0.897 |
| fifo | 0.154 | 0.308 | 0.462 | 0.615 |
| lru | 0.154 | 0.308 | 0.462 | 0.615 |
| salience_no_decay | 1.000 | 1.000 | 1.000 | 1.000 |

**reactivation**

| System | reactivated_item_retention | non_reactivated_retention | updated_content_recall |
| --- | --- | --- | --- |
| existing_working_memory | 0.905 | 0.119 | 0.905 |
| fifo | 0.667 | 0.119 | 0.667 |
| lru | 0.809 | 0.071 | 0.809 |
| salience_no_decay | 1.000 | 1.000 | 1.000 |

**capacity pressure**

| System | overlap_with_highest_initial_salience | overlap_with_most_recent | capacity_violations |
| --- | --- | --- | --- |
| existing_working_memory | 0.751 | 0.623 | 0 |
| fifo | 0.404 | 1.000 | 0 |
| lru | 0.404 | 1.000 | 0 |
| salience_no_decay | 1.000 | 0.404 | 0 |

**serial order**

| System | kendall_tau_no_reactivation | kendall_tau_after_reactivation |
| --- | --- | --- |
| existing_working_memory | 1.000 | 0.467 |
| fifo | 1.000 | 1.000 |
| lru | 1.000 | 0.467 |
| salience_no_decay | 1.000 | 0.467 |

**similarity interference**

| System | similar_distractors | dissimilar_distractors | difference |
| --- | --- | --- | --- |
| existing_working_memory | 0.686 | 0.686 | 0.000 |
| fifo | 0.385 | 0.385 | 0.000 |
| lru | 0.385 | 0.385 | 0.000 |
| salience_no_decay | 1.000 | 1.000 | 0.000 |

**newcomer admission full capacity 7**

| System | salience_0.1 | salience_0.3 | salience_0.5 | salience_0.7 |
| --- | --- | --- | --- | --- |
| existing_working_memory | 0.000 | 0.000 | 0.000 | 1.000 |
| fifo | 1.000 | 1.000 | 1.000 | 1.000 |
| lru | 1.000 | 1.000 | 1.000 | 1.000 |
| salience_no_decay | 0.000 | 0.000 | 0.000 | 0.000 |

**Serial position** (12 equal-salience items into capacity 7; 1 = retained)

| Policy | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| existing_working_memory | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| fifo | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| lru | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| salience_no_decay | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

**Generalisation grid**: 216 cells × 3 seeded repetitions (capacity × intervening items × distractor-salience distribution × event ordering). Mean target retention per factor level:

*capacity*

| Policy | 2 | 4 | 7 | 10 |
| --- | --- | --- | --- | --- |
| existing_working_memory | 0.385 | 0.654 | 0.776 | 0.837 |
| fifo | 0.315 | 0.537 | 0.685 | 0.741 |
| lru | 0.315 | 0.537 | 0.685 | 0.741 |
| salience_no_decay | 0.372 | 0.658 | 0.813 | 0.870 |

*delay*

| Policy | 0 | 2 | 4 | 8 | 12 | 16 |
| --- | --- | --- | --- | --- | --- | --- |
| existing_working_memory | 0.917 | 0.830 | 0.784 | 0.583 | 0.463 | 0.401 |
| fifo | 0.917 | 0.778 | 0.667 | 0.444 | 0.306 | 0.306 |
| lru | 0.917 | 0.778 | 0.667 | 0.444 | 0.306 | 0.306 |
| salience_no_decay | 0.917 | 0.818 | 0.762 | 0.580 | 0.522 | 0.472 |

*distractor salience*

| Policy | low | matched | high |
| --- | --- | --- | --- |
| existing_working_memory | 0.833 | 0.611 | 0.545 |
| fifo | 0.569 | 0.569 | 0.569 |
| lru | 0.569 | 0.569 | 0.569 |
| salience_no_decay | 0.917 | 0.744 | 0.375 |

*ordering*

| Policy | targets_first | interleaved | targets_last |
| --- | --- | --- | --- |
| existing_working_memory | 0.526 | 0.568 | 0.895 |
| fifo | 0.375 | 0.417 | 0.917 |
| lru | 0.375 | 0.417 | 0.917 |
| salience_no_decay | 0.679 | 0.679 | 0.677 |

In 48 of 216 cells at least one baseline retains more targets than the existing policy.

**Checks**

- PASS: sessions are isolated between owners
- PASS: the same owner and session id returns the same workspace
- PASS: clearing a session empties it
- PASS: session registry is bounded (LRU)
- PASS: clear_owner removes every session of that owner
- PASS: session capacity is applied
- PASS: reactivation increments access_count
- PASS: reactivated item moves to most-recent position

**Findings**

- Across a 216-cell grid (capacity × delay × distractor salience × ordering), existing retention by ordering is targets_first 0.526, interleaved 0.568, targets_last 0.895. In 48 cells at least one baseline retains more targets.
- Retention is content-agnostic: similar and dissimilar distractors give identical retention (difference +0.000). The implementation has no similarity-based interference.
- With lower-salience distractors, mean retention (capacities 3–9) first drops below 1.0 at 4 intervening items. Multiplicative decay (×0.92 per insertion) eventually pushes older high-salience targets below fresh distractors, so decay rather than content determines forgetting.
- A low-salience input cannot enter a full workspace: it is evicted on arrival because eviction is applied after insertion.
- Serial-position retention is purely recency-based ({'1': 0.0, '2': 0.0, '3': 0.0, '4': 0.0, '5': 0.0, '6': 1.0, '7': 1.0, '8': 1.0, '9': 1.0, '10': 1.0, '11': 1.0, '12': 1.0}); there is no primacy effect.
- Capacity is never exceeded (0 violations).

### Prediction/novelty: lexical novelty against active context

*Evidence category: computational capability benchmark (authored labels).*

| System | auroc | balanced_accuracy | correlation_with_lexical_jaccard_distance | false_novelty_rate | mean_novelty_expected | mean_novelty_unexpected | missed_novelty_rate | novelty_separation | repeat_run_identical | threshold_from_tuning_split |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| existing_prediction | 0.597 | 0.550 | 0.978 | 0.650 | 0.484 | 0.612 | 0.250 | 0.128 | yes | 0.314 |
| content_word_overlap | 0.575 | 0.619 | 0.955 | 0.700 | 0.502 | 0.632 | 0.062 | 0.130 | yes | 0.125 |
| jaccard_distance | 0.586 | 0.619 | 1.000 | 0.700 | 0.551 | 0.707 | 0.062 | 0.156 | yes | 0.226 |
| embedding_distance_reference | 0.828 | 0.750 | 0.836 | 0.500 | 0.149 | 0.370 | 0.000 | 0.221 | yes | 0.067 |

**Mean novelty by category**

| Category | Label | existing_prediction | content_word_overlap | jaccard_distance | embedding_distance_reference |
| --- | --- | --- | --- | --- | --- |
| contextually_expected | expected | 0.850 | 1.000 | 0.939 | 0.325 |
| contextually_unexpected | unexpected | 0.559 | 0.592 | 0.715 | 0.314 |
| contradiction | unexpected | 0.161 | 0.217 | 0.272 | 0.131 |
| exact_repetition | expected | 0.000 | 0.000 | 0.000 | 0.000 |
| low_overlap_related | expected | 0.964 | 1.000 | 0.977 | 0.319 |
| near_repetition | expected | 0.125 | 0.100 | 0.208 | 0.016 |
| paraphrase | expected | 0.482 | 0.411 | 0.631 | 0.086 |
| shared_vocabulary_unrelated | unexpected | 0.729 | 0.721 | 0.840 | 0.500 |
| unrelated | unexpected | 1.000 | 1.000 | 1.000 | 0.536 |

Case/punctuation invariance: 1.000. Empty context comparable: no (novelty 0.000).

**Score calibration** (a logistic mapping is fitted on the tuning split, frozen, then applied to the eval split)

| System | Mapping | Brier | ECE | Cal. intercept | Cal. slope | AUROC | False novelty @0.5 | Missed novelty @0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| existing_prediction | uncalibrated | 0.325 | 0.334 | -0.294 | 0.066 | 0.597 | 0.500 | 0.312 |
| existing_prediction | logistic | 0.240 | 0.051 | 0.086 | 1.144 | 0.597 | 0.200 | 0.750 |
| existing_prediction | base rate 0.444 | 0.247 | n/a | n/a | n/a | 0.500 | n/a | n/a |
| content_word_overlap | uncalibrated | 0.347 | 0.371 | -0.255 | 0.017 | 0.575 | 0.500 | 0.250 |
| content_word_overlap | logistic | 0.241 | 0.188 | 0.064 | 1.209 | 0.575 | 0.400 | 0.750 |
| content_word_overlap | base rate 0.444 | 0.247 | n/a | n/a | n/a | 0.500 | n/a | n/a |
| jaccard_distance | uncalibrated | 0.330 | 0.326 | -0.342 | 0.065 | 0.586 | 0.550 | 0.250 |
| jaccard_distance | logistic | 0.237 | 0.170 | 0.038 | 0.950 | 0.586 | 0.400 | 0.688 |
| jaccard_distance | base rate 0.444 | 0.247 | n/a | n/a | n/a | 0.500 | n/a | n/a |
| embedding_distance_reference | uncalibrated | 0.213 | 0.199 | 1.192 | 0.979 | 0.828 | 0.000 | 0.688 |
| embedding_distance_reference | logistic | 0.179 | 0.202 | -0.387 | 0.969 | 0.828 | 0.350 | 0.250 |
| embedding_distance_reference | base rate 0.444 | 0.247 | n/a | n/a | n/a | 0.500 | n/a | n/a |

Reliability bins (existing, logistic-calibrated): [0.0, 0.2) n=0; [0.2, 0.4) n=13, predicted 0.34, observed 0.31; [0.4, 0.6) n=23, predicted 0.48, observed 0.52; [0.6, 0.8) n=0; [0.8, 1.0) n=0

Novelty is a score, not a probability. Calibration maps scores to observed frequencies of 'unexpected' labels; it does not make the score a better predictor, and it is unrelated to biological prediction error. With 18 tuning items the fitted mapping is crude.

**Findings**

- Existing novelty AUROC on the eval split is 0.597. False novelty rate 0.650, missed novelty rate 0.250 at the tuning-split threshold 0.3143.
- Contradictions look familiar: mean novelty 0.161, against 1.000 for unrelated input. Negation or a changed value barely changes token overlap.
- Paraphrases and semantically related inputs with little word overlap look novel (0.964), because the signal is lexical.
- Script violations score lower novelty than typical continuations (0.559 vs 0.850), the opposite of the intended signal, because violating sentences reuse context words.
- The score is a deterministic function of token overlap (correlation with lexical Jaccard distance 0.9783). It does not represent meaning, expectation, or future events; its above-chance separation comes from surface word differences.
- The optional local-embedding reference reaches AUROC 0.828 but also scores contradictions as familiar (0.131). Semantic similarity is not contradiction detection.

### Long-term memory recall (synthetic retrieval benchmark)

*Evidence category: computational capability benchmark (synthetic, project-authored).*

| System | contradiction_recall@5 | mrr | ndcg@5 | owner_leaks | precision@5 | recall@5 | rejection_rate | source_attribution_accuracy | temporal_answer_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lexical | 1.000 | 0.681 | 0.681 | 0 | 0.161 | 0.690 | 0.714 | 1.000 | 0.273 |
| semantic | 1.000 | 0.824 | 0.838 | 0 | 0.200 | 0.886 | 0.857 | 1.000 | 0.273 |
| hybrid | 1.000 | 0.848 | 0.861 | 0 | 0.204 | 0.908 | 0.714 | 1.000 | 0.364 |
| hybrid_temporal | 1.000 | 0.964 | 0.969 | 0 | 0.239 | 0.995 | 0.714 | 1.000 | 0.909 |
| ablation_lexical_temporal | 1.000 | 0.772 | 0.770 | 0 | 0.196 | 0.777 | 0.714 | 1.000 | 0.636 |
| ablation_semantic_temporal | 1.000 | 0.953 | 0.955 | 0 | 0.235 | 0.973 | 0.857 | 1.000 | 0.909 |
| ablation_hybrid_temporal_no_similarity_threshold | 1.000 | 0.931 | 0.939 | 0 | 0.235 | 0.973 | 0.000 | 1.000 | 0.727 |
| ablation_hybrid_temporal_no_contradiction_expansion | 1.000 | 0.967 | 0.972 | 0 | 0.239 | 0.995 | 0.714 | 1.000 | 0.909 |

Dataset `artificial-human-synthetic-memory-retrieval` v1.1.0 (sha256 `eb46c69b0819`); embedding model: `BAAI/bge-small-en-v1.5`; semantic threshold 0.570 (tuned on the tuning split).

**MRR by category** (rej = rejection rate)

| Category | lexical | semantic | hybrid | hybrid_temporal | ablation_lexical_temporal | ablation_semantic_temporal | ablation_hybrid_temporal_no_similarity_threshold | ablation_hybrid_temporal_no_contradiction_expansion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conceptual | 0.200 | 1.000 | 0.900 | 0.900 | 0.200 | 1.000 | 0.900 | 0.900 |
| contradiction | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| episodic | 0.600 | 0.900 | 0.900 | 0.900 | 0.600 | 0.900 | 0.900 | 0.900 |
| exact_keyword | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| historical_fact | 0.500 | 0.569 | 0.667 | 0.889 | 0.500 | 0.889 | 0.806 | 0.917 |
| irrelevant | rej 0.800 | rej 0.800 | rej 0.800 | rej 0.800 | rej 0.800 | rej 0.800 | rej 0.000 | rej 0.800 |
| owner_isolation | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| paraphrase | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| source_attribution | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| temporal_chronological | 0.067 | 0.200 | 0.200 | 1.000 | 0.900 | 1.000 | 0.800 | 1.000 |
| type_filter | 0.750 | 0.750 | 1.000 | 1.000 | 0.750 | 0.750 | 1.000 | 1.000 |

**Checks**

- PASS: no cross-owner leakage in any variant

**Findings**

- lexical: MRR 0.681, rejection 0.714, temporal answer accuracy 0.273, source attribution 1.000.
- semantic: MRR 0.824, rejection 0.857, temporal answer accuracy 0.273, source attribution 1.000.
- hybrid: MRR 0.848, rejection 0.714, temporal answer accuracy 0.364, source attribution 1.000.
- hybrid_temporal: MRR 0.964, rejection 0.714, temporal answer accuracy 0.909, source attribution 1.000.
- The benchmark is synthetic, English-only and project-authored. It is a regression signal, not independent validation. Multi-turn recall and context switching are measured through the full cognitive cycle in the integration suite.

## 8. Integrated multi-turn results

| Scenario | lexical_temporal | hybrid_temporal |
| --- | --- | --- |
| A_retain_and_recall — Retain and recall across distraction | 3/5 | 4/5 |
| B_temporal_knowledge — Current, previous and earlier-known facts | 6/6 | 6/6 |
| C_attention_under_distraction — Attention with high-salience distractors | 2/4 | 4/4 |
| D_working_memory_interference — Working-memory interference over turns | 4/5 | 4/5 |
| E_contradiction_awareness — Contradictions preserved through to the model | 5/5 | 5/5 |
| F_context_switching_multi_turn — Session switching and elliptical follow-up | 2/4 | 3/4 |

| Scenario | Check | lexical_temporal | hybrid_temporal |
| --- | --- | --- | --- |
| A | paraphrased question selects the stored fact | pass | pass |
| A | paraphrased question ranks the stored fact first | FAIL | pass |
| A | keyword question selects the stored fact | pass | pass |
| A | turn-1 input is still in working memory after 3 turns | FAIL | FAIL |
| A | conversation turns were not written to long-term memory | pass | pass |
| B | 'now' selects the current fact first | pass | pass |
| B | 'now' excludes the superseded fact | pass | pass |
| B | 'previously' selects the superseded fact first | pass | pass |
| B | known_at before the update returns only the old fact | pass | pass |
| B | valid_at 2024 returns only the old fact | pass | pass |
| B | both versions keep their own provenance | pass | pass |
| C | highest-weighted memory is relevant | FAIL | pass |
| C | all relevant memories are selected | pass | pass |
| C | no high-salience distractor outweighs a relevant memory | FAIL | pass |
| C | selected memory precision | pass | pass |
| D | capacity is never exceeded | pass | pass |
| D | final question retrieves the stored fact | pass | pass |
| D | re-retrieved stored fact is reactivated (access_count > 0) | pass | pass |
| D | an identical repeated input reactivates its earlier entry | FAIL | FAIL |
| D | displacement is recency-ordered (retained turns are the most recent ones) | pass | pass |
| E | both conflicting memories reach the language component | pass | pass |
| E | evidence marks the conflict | pass | pass |
| E | sources of both claims are preserved | pass | pass |
| E | no claim was promoted, rewritten, or added | pass | pass |
| E | model is told not to resolve conflicts silently | pass | pass |
| F | sessions do not share working memory | pass | pass |
| F | switching back keeps the earlier session context | pass | pass |
| F | elliptical follow-up ranks the sister's city first | FAIL | pass |
| F | the preceding turn changes what is retrieved (conversational context is used) | FAIL | FAIL |

**Findings**

- lexical_temporal: 22/29 behavioural checks passed.
- hybrid_temporal: 26/29 behavioural checks passed.
- Each cycle retrieves with the current turn only, so elliptical follow-ups cannot use the conversational context held in working memory.
- A repeated input creates a new percept entry instead of reactivating the earlier one, because percept identifiers are unique per turn.
- Retrieved memories compete with percepts for the same seven workspace slots, so the turn-1 input is displaced after a few turns that retrieve several memories.
- Lexical retrieval found nothing for 'Where does she live?' because the light stemmer maps 'lives' to 'liv' but 'live' to 'live'. This is recorded as a measured limitation and has not been changed in this phase.

## 9. Failure cases

**attention** (showing 6 of 38)

- task: attention-eval-salience_informative-32-0; condition: salience_informative; k: 5; selected_irrelevant: {'id': 'c008', 'grade': 0, 'kind': 'memory', 'relevance': 0.5532, 'salience': 0.2059}; excluded_highly_relevant: {'id': 'c019', 'grade': 2, 'kind': 'memory', 'relevance': 0.3432, 'salience': 0.8881}
- task: attention-eval-salience_informative-32-1; condition: salience_informative; k: 5; selected_irrelevant: {'id': 'c009', 'grade': 0, 'kind': 'memory', 'relevance': 0.749, 'salience': 0.1891}; excluded_highly_relevant: {'id': 'c002', 'grade': 2, 'kind': 'memory', 'relevance': 0.4759, 'salience': 0.9}
- task: attention-eval-salient_distractors-8-0; condition: salient_distractors; k: 2; selected_irrelevant: {'id': 'c001', 'grade': 0, 'kind': 'memory', 'relevance': 0.5803, 'salience': 0.9942}; excluded_highly_relevant: {'id': 'c000', 'grade': 2, 'kind': 'memory', 'relevance': 0.7507, 'salience': 0.1447}
- task: attention-eval-salient_distractors-8-1; condition: salient_distractors; k: 2; selected_irrelevant: {'id': 'c000', 'grade': 0, 'kind': 'memory', 'relevance': 0.6054, 'salience': 0.9619}; excluded_highly_relevant: {'id': 'c004', 'grade': 2, 'kind': 'memory', 'relevance': 0.7189, 'salience': 0.1712}
- task: attention-eval-cross_source_normalisation-4-0; condition: cross_source_normalisation; k: 2; selected_irrelevant: {'id': 'c000', 'grade': 0, 'kind': 'learning_example', 'relevance': 1.0, 'salience': 0.0}; excluded_highly_relevant: {'id': 'c002', 'grade': 2, 'kind': 'memory', 'relevance': 0.9218, 'salience': 0.0}
- task: attention-eval-cross_source_normalisation-4-2; condition: cross_source_normalisation; k: 2; selected_irrelevant: {'id': 'c002', 'grade': 0, 'kind': 'learning_example', 'relevance': 1.0, 'salience': 0.0}; excluded_highly_relevant: {'id': 'c001', 'grade': 2, 'kind': 'memory', 'relevance': 0.8489, 'salience': 0.0}

**working-memory** (showing 3 of 3)

- case: weak newcomer dropped on arrival; detail: With seven items at salience 0.9, a new item at salience 0.1 is evicted immediately
- case: weak newcomer dropped on arrival; detail: With seven items at salience 0.9, a new item at salience 0.3 is evicted immediately
- case: weak newcomer dropped on arrival; detail: With seven items at salience 0.9, a new item at salience 0.5 is evicted immediately

**prediction** (showing 10 of 17)

- item: prediction-near_repetition-6; category: near_repetition; error: false novelty; novelty: 0.3333; threshold: 0.3143; context: Insulin lowers blood glucose.; input: Insulin lowers the blood glucose level.
- item: prediction-paraphrase-3; category: paraphrase; error: false novelty; novelty: 0.4; threshold: 0.3143; context: The meeting starts at nine tomorrow.; input: Tomorrow's meeting begins at 9 o'clock.
- item: prediction-paraphrase-4; category: paraphrase; error: false novelty; novelty: 0.3333; threshold: 0.3143; context: Water boils at one hundred degrees Celsius.; input: At one hundred degrees Celsius, water starts to boil.
- item: prediction-paraphrase-5; category: paraphrase; error: false novelty; novelty: 0.625; threshold: 0.3143; context: The car would not start this morning.; input: This morning the engine refused to turn over.
- item: prediction-paraphrase-6; category: paraphrase; error: false novelty; novelty: 0.5714; threshold: 0.3143; context: She finished the marathon in under four hours.; input: Her marathon time was below four hours.
- item: prediction-low_overlap_related-3; category: low_overlap_related; error: false novelty; novelty: 1.0; threshold: 0.3143; context: The orchestra tuned their violins.; input: A concert is about to begin.
- item: prediction-low_overlap_related-4; category: low_overlap_related; error: false novelty; novelty: 1.0; threshold: 0.3143; context: It has rained heavily for three days.; input: The rivers are flooding.
- item: prediction-low_overlap_related-5; category: low_overlap_related; error: false novelty; novelty: 1.0; threshold: 0.3143; context: I booked a flight to Tokyo.; input: Japan travel plans are confirmed.
- item: prediction-low_overlap_related-6; category: low_overlap_related; error: false novelty; novelty: 0.8571; threshold: 0.3143; context: The chef sharpened her knives.; input: Dinner preparation is underway in the kitchen.
- item: prediction-contradiction-3; category: contradiction; error: missed novelty; novelty: 0.1667; threshold: 0.3143; context: Water boils at one hundred degrees Celsius.; input: Water boils at fifty degrees Celsius.

**memory** (showing 3 of 3)

- variant: hybrid_temporal; query: eval-type-01; category: type_filter; expected: ['ep_lisbon_arrive', 'ep_lisbon_tram', 'ep_lisbon_pasteis', 'ep_lisbon_depart']; top_results: ['ep_lisbon_tram', 'ep_lisbon_depart', 'ep_lisbon_arrive']
- variant: hybrid_temporal; query: eval-irr-02; category: irrelevant; expected: []; top_results: ['tides', 'night_owl']
- variant: hybrid_temporal; query: eval-own-04; category: owner_isolation; expected: []; top_results: ['b_pref']

**integration** (showing 10 of 10)

- scenario: A_retain_and_recall; config: lexical_temporal; check: paraphrased question ranks the stored fact first
- scenario: A_retain_and_recall; config: lexical_temporal; check: turn-1 input is still in working memory after 3 turns
- scenario: C_attention_under_distraction; config: lexical_temporal; check: highest-weighted memory is relevant
- scenario: C_attention_under_distraction; config: lexical_temporal; check: no high-salience distractor outweighs a relevant memory
- scenario: D_working_memory_interference; config: lexical_temporal; check: an identical repeated input reactivates its earlier entry
- scenario: F_context_switching_multi_turn; config: lexical_temporal; check: elliptical follow-up ranks the sister's city first
- scenario: F_context_switching_multi_turn; config: lexical_temporal; check: the preceding turn changes what is retrieved (conversational context is used)
- scenario: A_retain_and_recall; config: hybrid_temporal; check: turn-1 input is still in working memory after 3 turns
- scenario: D_working_memory_interference; config: hybrid_temporal; check: an identical repeated input reactivates its earlier entry
- scenario: F_context_switching_multi_turn; config: hybrid_temporal; check: the preceding turn changes what is retrieved (conversational context is used)

## 10. Ablations

**Attention weight sensitivity** (eval split, all conditions)

| System | ndcg@k | precision@k | recall@k | mrr | pairwise_accuracy |
| --- | --- | --- | --- | --- | --- |
| weights_r0.6_s0.4 | 0.749 | 0.695 | 0.499 | 0.825 | 0.884 |
| weights_r0.5_s0.5 | 0.729 | 0.683 | 0.486 | 0.817 | 0.868 |

Working memory: `salience_no_decay` removes the decay term; its rows appear in every working-memory table above.

Memory: the `ablation_*` rows in the memory table remove the lexical or semantic channel, the similarity threshold, or contradiction expansion.

## 11. Runtime and resource use

| Suite | Wall-clock (s) | Peak traced Python memory (KiB) |
| --- | --- | --- |
| attention | 2.231 | 1330 |
| working-memory | 1.684 | 117 |
| prediction | 2.748 | 132 |
| memory | 28.936 | 1199 |
| integration | 0.745 | 132 |

Total: 36.398 s. Network used by the evaluation: no.

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


---

## Supplemental associative-memory evaluation — 2026-11-25

This addendum records the associative-memory suite added after the baseline report above. It was generated by CI workflow [#343](https://github.com/aakashsiwach/artificial-human/actions/runs/45633697484) from commit `2baa52580e7414ed2da2d5da381a06868e51d6d8` (Python 3.13, seed `20261124`). The full suite ran alongside the existing cognitive-evaluation suites; the associative probes themselves use source-defined facts and do not use embeddings.

Nine fixed source-fact pairs were co-activated, then queried with held-out single-fact cues. All six benchmark checks passed.

| System | Target recall@5 | Target MRR | False associations / query |
| --- | ---: | ---: | ---: |
| Keyword-only | 0.000 | 0.000 | 0.000 |
| Learned associations | **1.000** | **0.481** | 0.111 |
| Random-link control | 0.085 | 0.038 | 1.100 |

The learned graph contained 9 links over 15 source facts (mean strength 0.069008; maximum 0.070400). The random-link control used 512 seeded graphs, each sampled with the same 9-link budget and mean link strength. The learned graph retrieved every target in the top five; the random-link control did so for about 8.5% of probes.

**Interpretation and limits:** this supports the narrow claim that the current co-activation graph can complete these nine authored source-fact cues better than direct lexical retrieval and this random-link control. It does not establish open-ended episodic recall, source monitoring, interference handling, human-like memory, or biological fidelity. Broader independent and held-out tasks remain necessary.
