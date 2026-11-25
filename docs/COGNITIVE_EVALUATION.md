# Cognitive evaluation framework: research record

This document is the research record for Stage 1 of the [Research Charter](RESEARCH_CHARTER.md): close the measurement gap before adding new cognitive mechanisms. It describes what each benchmark measures, why, and what it cannot show. The generated results are in [benchmarks/cognitive-evaluation-report.md](benchmarks/cognitive-evaluation-report.md). The code is in `src/artificial_human/cognition/benchmark/`.

**No cognitive algorithm was changed to improve these results.** The framework measures the code that exists on `main`.

## Evidence categories

| Category | What it can establish | Where |
| --- | --- | --- |
| Software correctness | The code behaves as specified (consent, isolation, determinism, capacity) | `tests/` |
| Computational capability | Behaviour on defined synthetic tasks, compared with baselines | This framework |
| Human behavioural comparison | Qualitative similarity or difference to replicated human findings; no human data are fitted | Notes in this document |
| Biological fidelity | Not assessed. No result here shows that software works like a brain mechanism | — |

## Architecture

```
datasets.py  ──►  *_eval.py (execute the real implementation)  ──►  metrics.py  ──►  runner.py  ──►  report.py
(seeded, split)   attention / working-memory / prediction /         (shared with       (timing,        (JSON +
                  memory (wraps memory benchmark) / integration       memory benchmark)   tracemalloc)    Markdown)
```

- **Typed tasks:** `AttentionTask`, `PredictionItem`, working-memory `Event`; results in `SuiteResult`.
- **Seeds:** `derive_seed(seed, suite, split, condition, …)` uses SHA-256, so it is platform-independent and not affected by Python's salted `hash`. Tuning and eval tasks never share an RNG stream. Default seed: `20261124`.
- **Splits:** every suite records tuning/eval sizes. Only the prediction thresholds and the memory similarity threshold are tuned, and only on tuning splits.
- **Reuse:** ranked-retrieval metrics come from `memory/benchmark/metrics.py`. The memory suite calls the existing memory benchmark rather than copying it.
- **No hidden network calls:** the default run uses no embedding model. `--embedding-backend fastembed` loads a local, in-process model (weights are downloaded once unless `--offline`).
- **Compact output:** aggregates, a few failure examples, and failing queries only.

## Benchmark records

The hypotheses below were written while the benchmarks were being built, and some after preliminary runs. They were **not preregistered**. Treat them as stated expectations that make the results interpretable, not as confirmatory tests.

### Attention

| | |
| --- | --- |
| Human phenomenon | Selective attention under competition between goal-relevant and salient stimuli. Biased competition (Desimone & Duncan, 1995); saliency maps (Itti & Koch, 2001); attentional capture by salient distractors (Theeuwes, 2010); the top-down/bottom-up dichotomy is debated (Awh, Belopolsky & Theeuwes, 2012). |
| Function measured | Whether a bounded top-k selection includes the most useful candidates given noisy relevance and explicit salience. |
| What the software implements | `weight = 0.8 × relevance + 0.2 × salience`, deduplicated by id, ties by id, top `limit`. In the cycle, memories and learning examples are max-normalised separately before selection. |
| What it cannot establish | Anything about neural competition, capacity limits of perception, or real attention. Ground truth comes from the task generator, so results describe the formula under stated input distributions. |
| Hypotheses | H1: with uninformative salience, the formula is slightly worse than relevance-only. H2: with informative salience, it beats relevance-only. H3: high-salience distractors with moderate relevance are wrongly selected. H4: per-source normalisation lets weak learning examples displace relevant memories. |
| Baselines | Relevance-only, salience-only, input order. Ablations: weights 0.6/0.4 and 0.5/0.5. |
| Metrics / failure conditions | P@k, R@k, nDCG@k, MRR, pairwise accuracy, repeat-run identity, top-k permutation invariance. A failure is a grade-0 candidate selected while a grade-2 candidate is excluded. |

### Working memory

| | |
| --- | --- |
| Human phenomenon | Limited capacity (Miller, 1956; about four chunks in Cowan, 2001); maintenance and updating; forgetting through interference rather than decay (Lewandowsky, Oberauer & Brown, 2009; for decay, Berman, Jonides & Lewis, 2009); primacy and recency in recall (Murdock, 1962); benchmark findings for models (Oberauer et al., 2018, *Psychological Bulletin* 144(9), 885–958, doi:10.1037/bul0000153). |
| Function measured | Which items remain in a bounded store after distractors, reactivation, capacity pressure and session switching. |
| What the software implements | Capacity 7 by default; each insertion multiplies existing salience by 0.92; eviction of the lowest salience *after* insertion; reactivation replaces the entry, increments `access_count`, and moves it to the end; owner-scoped LRU sessions. |
| What it cannot establish | Human-like retention, chunking, rehearsal, or binding. Presence in a store is not recall from a cue. |
| Hypotheses | H1: retention falls once decay pushes targets below fresh distractors. H2: no similarity-based interference (content is ignored). H3: retention is recency-only (no primacy). H4: a weak new input cannot enter a full workspace. H5: capacity is never exceeded and sessions are isolated. |
| Baselines | FIFO, LRU, salience eviction without decay (ablation). |
| Metrics / failure conditions | Retention rate, recall accuracy (expected content), retention against intervening items and capacity, reactivation retention, overlap with highest-salience and most-recent sets, Kendall's τ for serial order, newcomer admission, capacity violations (must be 0), session checks (must pass). |

### Prediction and novelty

| | |
| --- | --- |
| Human phenomenon | Responses to unexpected input: semantic incongruity (N400, Kutas & Hillyard, 1980), novelty/mismatch detection (Kumaran & Maguire, 2007), prediction-error accounts (Rao & Ballard, 1999; Friston, 2005), whose empirical status is mixed (Hodson, Mehta & Smith, 2024). |
| Function measured | Whether the novelty score separates inputs that are unexpected given the context from inputs that are expected. |
| What the software implements | `1 − |tokens(input) ∩ tokens(context)| / |tokens(input)|`. It is not a world model and does not forecast. It is not predictive coding, active inference, or a biological prediction-error signal. |
| What it cannot establish | Expectation, surprise, or belief. Labels are author judgements, not measured human responses. |
| Hypotheses | H1: AUROC only modestly above chance. H2: contradictions look familiar. H3: paraphrases and low-overlap related inputs look novel. H4: shared vocabulary with a different word sense looks less novel than unrelated text. |
| Baselines | Content-word overlap, Jaccard distance. Optional non-lexical reference: 1 − maximum embedding cosine (local model). |
| Metrics / failure conditions | AUROC, false/missed novelty rates at the tuning-split threshold, balanced accuracy, separation, correlation with lexical distance, repeat-run identity, case/punctuation invariance. |
| What a real predictive model would need | A state representation of the situation (entities, propositions, scripts), a generative model that predicts likely next states or inputs, a calibrated uncertainty estimate, and a learning rule that updates on error. It must then be evaluated on held-out sequences against these lexical baselines. |

### Long-term memory

The existing [memory benchmark](MEMORY.md#10-evaluation) is extended, not duplicated. Dataset v1.1.0 adds four **source attribution** queries (eval split only, so the tuned threshold is unchanged) and three task-level metrics: temporal answer accuracy, source attribution accuracy and contradiction recall@5. Written JSON results are now compact (aggregates plus failing queries). Multi-turn recall and context switching are measured through the full cycle in integration scenarios A and F.

### Associative memory

The associative-memory suite is an offline, deterministic control experiment over
nine pairs of source-defined facts. For each pair, an authored training cue is
required to retrieve exactly the source and target facts. A separate test cue is
required to retrieve only the source fact by direct keyword retrieval. The
production `AssociativeKnowledgeNetwork` then processes the paired training
inputs and is evaluated on the held-out cues.

Three systems are compared:

- **Keyword-only:** direct `retrieve_core_knowledge` output, with no graph spreading.
- **Learned associations:** the production bounded co-activation graph and its
  one-hop spreading-activation output.
- **Random-link control:** 512 seeded undirected graphs sampled uniformly over
  all possible source-fact pairs. Each random graph has the same number of edges
  and mean link strength as the trained graph; link decay is applied between
  cue queries.

Primary metrics are target recall@5 and target mean reciprocal rank (MRR).
False associations per query count top-five facts that are neither the cue source
nor the declared target. There is no tuning split: all nine authored probes are
fixed evaluation probes. CI verifies deterministic output and that the learned
graph retrieves the declared targets better than both controls on this probe set.

This is a small software-mechanism test, not an independent cognitive benchmark.
The facts and prompts are project-authored, the corpus is source-defined, and the
queries are deliberately constructed to isolate a known fact pair. The random
graph controls for link count and mean strength but does not model biological
connectivity. Results must not be interpreted as evidence of human-like,
hippocampal, or episodic memory.

**Observed CI results (25 November 2026, seed `20261124`):** across the nine probes,
learned associations achieved target recall@5 **1.000**, MRR **0.481**, and
0.111 false associations per query. Keyword-only retrieval achieved recall@5
0.000 and MRR 0.000. The 512-graph random-link control achieved recall@5 0.085,
MRR 0.038, and 1.100 false associations per query. All six suite checks passed
in [CI workflow #343](https://github.com/aakashsiwach/artificial-human/actions/runs/45633697484).
The generated results are recorded in
[the cognitive evaluation report](benchmarks/cognitive-evaluation-report.md).

### Context-scoped association and source monitoring

The runtime associative network partitions learned links by a non-content context
identifier. For a named conversation, the identifier combines a truncated hash of
the owner ID with a truncated hash of the session ID. Raw owner/session identifiers
and prompt text are not stored in the graph. Spreading activation is restricted to
links from the matching context; decay is advanced only when that context receives
an input, so activity in another session neither activates nor ages its links.

Requests without an explicit session use a random request scope that is deleted
immediately after the current input is processed. Explicit session deletion removes
both the working-memory workspace and its associative links. Owner-wide session
clearing removes all associative scopes associated with that owner. The graph keeps
one global link-count budget, so capacity pressure can evict a weak link from any
context.

The `associative-context` suite trains the same nine fixed fact-pair probes in
separate contexts. It compares cue completion inside the training context, leakage
when the same cue is issued in a different context, and a shared-global graph
control that deliberately has no context partition. Primary measures are
same-context target recall@5 and cross-context leakage@5. The suite also checks
that context cleanup deletes the learned links.

This is a narrow software source-context boundary. It does not establish the
external credibility or truth of a source, reconcile contradictory records, or
reproduce biological source-monitoring mechanisms. The facts and probes are
project-authored, and the test does not use user conversations or personal data.

### Contradiction-sensitive relational memory and interference

The cognitive cycle now includes a bounded, volatile relational graph over opaque
memory record IDs. It is separate from the source-fact associative graph. The
architecture first performs owner-scoped, consent-visible memory retrieval, then
activates one-hop links in the current owner/session context. Associated IDs are
resolved again through the same owner's supplied records or active-consent
repository before they can enter attention. Missing, deleted, revoked, or foreign
IDs are ignored.

Only memories selected by attention participate in co-activation learning. The
graph uses a bounded positive-link update, context-local decay, a global edge
budget, and retrieval-dependent strengthening after a link actually completes a
cue. The link budget intentionally creates a controlled capacity-interference
condition; it can evict the weakest association when a new relation is learned.

Explicit repository `contradicts` links are supplied to the relational network
from the current visible retrieval candidates. If a known conflict pair has a
positive association, the edge is deleted and is not used for activation or
relearned in that cycle. Both conflicting records remain ordinary retrieval
evidence; contradiction handling does not choose which record is true. Existing
hybrid search continues to expand explicit contradictions so they can be shown
beside one another.

The network stores opaque memory IDs, hashed/non-content context IDs, and scalar
weights only. It does not store memory text, source labels, owner IDs, or raw
session IDs. Requests without explicit session IDs are ephemeral and their
relational context is deleted after the cycle. Session and owner cleanup delete
relational links along with working and source-fact associative state.

The `associative-dynamics` suite contains nine deterministic project-authored
probes with two controls:
- **Contradiction suppression:** learn a positive pair, supply an explicit
  contradiction pair, and verify that the positive edge is removed and cannot
  complete the cue.
- **Capacity interference:** compare networks with identical one-edge capacity
  and distractor sequence, one with retrieval strengthening disabled and one
  with strengthening enabled. Repeated cue completion should protect the used
  target edge against a later distractor relation.

The suite reports contradiction suppression rate, contradiction recall@1,
target recall after distractors@1, and evicted-link counts. It is a test of
specified graph operations under synthetic conditions. It does not measure
human interference, factual source credibility, truth, or biological plasticity.
It uses opaque test IDs and no user memories.

### Temporal episodic sequence formation and replay

The cognitive cycle includes a bounded, volatile `EpisodicReplayMemory` over
opaque episodic memory IDs. It does not copy event text, source labels, owner IDs,
or raw session IDs. Each episode is formed only from episodic records selected by
attention. Records are ordered by `occurred_at`, falling back to `created_at`,
and the network stores bidirectional transitions between adjacent events rather
than an all-to-all association graph.

A later episodic cue can recover the immediate event before and after it from
the same hashed owner/session context. A replayed ID is resolved again through
the owner's live consent-visible repository before it can enter attention.
Deleted, revoked, foreign, non-episodic, or explicitly contradictory records
cannot be revived by sequence replay. When an item found by ordinary associative
retrieval is also supported by temporal replay, the cognitive trace records both
retrieval channels.

Transition strengths have bounded learning, retrieval-dependent strengthening,
and context-local decay. Separate hard limits bound the number of contexts,
episodes per context, events per episode, and directed transitions. A transition
evicted by the budget is no longer replayable even though an episode's event IDs
may remain in its bounded sequence record. Ephemeral requests delete their
replay context before response generation completes; session and owner cleanup
remove the corresponding episodes and transition weights.

The `episodic-replay` suite uses nine project-authored four-event sequences. It
cues each middle event and evaluates recovery of its immediate predecessor and
successor. Controls include no replay and a different owner/session context.
Primary measures are exact neighbour recall@2, MRR, false replays per query, and
cross-context leakage@2.

This is a deterministic event-sequence retrieval experiment over synthetic
opaque IDs. It does not establish human autobiographical memory, offline
sleep-stage replay, hippocampal pattern completion, or biological systems
consolidation. The runtime replay is cue-driven and occurs during an ordinary
cognitive cycle; sleep-stage-inspired offline replay and owner-reviewed
consolidation remain open research tasks.

### Integrated multi-turn scenarios

| Scenario | Question | Notes |
| --- | --- | --- |
| A Retain and recall | After unrelated input and a distractor, does a paraphrased question select the stored fact? Is the first input still in working memory? | Facts enter long-term memory only through consented writes; the cycle never stores conversation. |
| B Temporal knowledge | Does "now" select the current fact and "previously" the superseded one? Do `known_at` and `valid_at` return only what was known or valid then? | |
| C Attention under distraction | Do high-salience irrelevant memories outweigh relevant ones? | Uses explicit `salience` metadata. |
| D Working-memory interference | Over nine turns: capacity, reactivation of re-retrieved memories, handling of repeated input, recency of displacement. | |
| E Contradiction awareness | Do both conflicting memories and their sources reach the language component, unchanged and flagged? | A stub captures the prompt. Its output is never scored. |
| F Context switching and multi-turn | Are sessions isolated? Does a previous turn influence retrieval for an elliptical follow-up? | |

Each scenario runs with lexical + temporal retrieval and, when a local model is configured, with hybrid + temporal retrieval.

## Calibration of the novelty score

The prediction suite's novelty value is a **score, not a probability**. Calibration asks whether a mapping from score to probability of "unexpected" matches observed label frequencies. A logistic mapping `P = σ(a + b·score)` is fitted by ridge-regularised Newton–Raphson on the **18 tuning items only**, frozen, then applied to the 36 eval items. It is compared with the raw score read as a probability and with a constant base-rate predictor (tuning prevalence 0.444).

Reported per system: Brier score, expected calibration error (ECE), reliability bins, Cox calibration intercept and slope (logistic regression of outcomes on logit(p); ideal 0 and 1), AUROC, and false/missed novelty rates at p = 0.5.

Result (existing novelty, eval split):

| Mapping | Brier | ECE | Cal. slope | AUROC |
| --- | --- | --- | --- | --- |
| Raw score read as a probability | 0.325 | 0.334 | 0.07 | 0.597 |
| Logistic, fitted on tuning | 0.240 | 0.051 | 1.14 | 0.597 |
| Constant base rate | 0.247 | — | — | 0.5 |

Calibration removes the raw score's miscalibration, but the calibrated Brier score is barely better than predicting the base rate. The score carries little information about unexpectedness. A monotone mapping cannot change the ranking (AUROC). **Calibration is not prediction accuracy and is unrelated to a biological prediction-error signal.**

## Generalisation tests

- **Attention, held-out conditions.** The weight of the documented formula is tuned (grid 0.0–1.0) on tuning-split tasks of the *clean relevance* and *informative salience* conditions only. Systems are then tested on eval-split tasks of the unseen *salient distractors* and *cross-source normalisation* conditions. The tuned weight is 0.8, identical to the existing weight. That weight falls from nDCG@k 0.889 in-distribution to 0.596 on held-out conditions, while relevance-only stays at 0.822 → 0.824. Held-out failure categories per task: salient distractors selected 0.97 (existing) vs 0.12 (relevance-only); weak learning examples selected 0.73 for both.
- **Working memory, factorial grid.** 216 cells: capacity {2, 4, 7, 10} × intervening items {0, 2, 4, 8, 12, 16} × distractor salience {low, matched, high} × ordering {targets first, interleaved, targets last}, with 3 seeded repetitions each. Existing retention depends strongly on ordering (0.526 targets first, 0.568 interleaved, 0.895 targets last) and on distractor salience (0.833 low to 0.545 high). Salience without decay is ordering-insensitive (about 0.68). In 48 of 216 cells at least one baseline retains more targets.
- **Long-term memory, independent data.** LongMemEval Oracle breakdowns by category, number of history sessions, history length, evidence position, single- vs multi-session evidence, and days since the latest evidence (see below).

## External datasets

Full governance records (licence evidence, versions, hashes, fields used and excluded, download commands) are in [benchmarks/EXTERNAL_DATASETS.md](benchmarks/EXTERNAL_DATASETS.md).

| Dataset | Decision |
| --- | --- |
| LongMemEval **Oracle** (Wu et al., ICLR 2025), MIT | **Adopted** as the opt-in suite `longmemeval`: turn-level evidence retrieval, all questions held out |
| LongMemEval `_s` / `_m` / `custom_history` | Rejected: ShareGPT/UltraChat filler sessions (real user conversations) |
| NEMAR on003838 digit span (Pavlov et al., 2022), CC0 | **Adopted** as the opt-in suite `digit-span`: human behavioural reference, participant-level split, aggregates only |
| NEMAR on004574 cross-modal oddball, CC0 | Research candidate, not adopted: undocumented behavioural columns, mostly clinical population (health data needed), no defensible mapping to text novelty |
| LoCoMo (Maharana et al., ACL 2024) | Rejected: reported CC BY-NC 4.0, unverified at source |
| STS Benchmark (Cer et al., 2017) | Rejected: mixed per-source licences |
| Oberauer et al. (2018) working-memory benchmarks | Qualitative checklist only; no data copied |

External suites never download data. They run only with an explicit local path (`--longmemeval-path`, `--digit-span-dir`), never in CI, and their outputs contain no conversation text, question text, or participant identifiers.

### External results (full report: [benchmarks/external-evaluation-report.md](benchmarks/external-evaluation-report.md))

**LongMemEval Oracle**: 500 questions; 470 scored at turn level (the 30 abstention questions are reported separately); local `BAAI/bge-small-en-v1.5` for semantic variants; threshold 0.57 inherited from the internal tuning split, not tuned on these questions.

| Variant | Turn recall@1 | Turn recall@5 | Turn recall@10 | Turn MRR | Abstention rejection |
| --- | --- | --- | --- | --- | --- |
| lexical | 0.359 | 0.755 | 0.862 | 0.685 | 0.000 |
| lexical + temporal | 0.332 | 0.728 | 0.839 | 0.654 | 0.033 |
| semantic | 0.289 | 0.704 | 0.841 | 0.619 | 0.100 |
| hybrid + temporal | 0.348 | 0.759 | 0.882 | 0.688 | 0.033 |

- **The internal benchmark does not generalise as hoped.** On the project's synthetic memory benchmark, semantic-only beats lexical (MRR 0.824 vs 0.681). On independent data, semantic-only is worse than lexical (0.619 vs 0.685), and hybrid + temporal is roughly level with lexical (0.688). The synthetic dataset overstated the benefit of the embedding channel.
- The embedding channel helps strongly where wording diverges: **single-session-preference** turn MRR rises from 0.459 (lexical) to 0.767 (semantic) and 0.727 (hybrid). It hurts on single-session-assistant (0.767 lexical vs 0.555 semantic) and temporal-reasoning questions.
- **Rule-based temporal intent lowers evidence recall** on this data (lexical MRR 0.685 → 0.654). The English cue rules exclude or re-rank evidence for questions phrased with temporal words.
- **Longer histories are harder**: questions with more than 40 turns reach MRR 0.45–0.49 in every variant, against 0.81 (lexical) for 10 turns or fewer. Evidence late in the history is retrieved worse than early evidence (lexical 0.560 vs 0.701).
- **Abstention is not supported**: retrieval returns results for at least 90% of questions about events that never happened. Rejecting unanswerable questions needs a separate mechanism.
- Session-level metrics are uninformative here (every Oracle session is an evidence session). Owner leaks 0; isolation probes 0. Runtime: 30 s lexical; 50 min with local embeddings on CPU (about 11k turns embedded); peak traced Python memory 32 MB (ONNX runtime memory is not traced).
- This is evidence retrieval only. Answer accuracy was not evaluated.

**NEMAR on003838 digit span**: 86 participants (31 tuning / 55 held-out), 9,276 trials.

| Load | Held-out humans: mean (SD) | Humans first / last third | Software (same sequences) | Software first / last third |
| --- | --- | --- | --- | --- |
| 5 | 0.925 (0.060) | 0.979 / 0.905 | 1.000 | 1.000 / 1.000 |
| 9 | 0.460 (0.132) | 0.767 / 0.172 | 0.778 | 0.333 / 1.000 |
| 13 | 0.273 (0.088) | 0.610 / 0.026 | 0.538 | 0.000 / 1.000 |

Serial-position RMSE against held-out participants: existing software 0.680; recency store (best capacity 1, fitted on tuning) 0.592; primacy store (capacity 5, fitted on tuning) 0.284; flat per-load mean 0.232. Humans in this strict serial-recall task show a strong primacy gradient and almost no recency. The software shows pure recency. Neither the human data nor this comparison says anything about the mechanisms involved. The comparison shows only that the workspace's retention profile is the opposite of human serial-recall behaviour on identical sequences.

### Remaining Stage 1 gaps

- No independently authored material for **attention** or **prediction/novelty**. The cross-modal oddball dataset was reviewed and not adopted.
- Calibration covers only the novelty score. Attention weights and retrieval scores are uncalibrated.
- No cross-language or cross-domain generalisation, and no failure-recovery measures.
- External suites are opt-in and do not run in CI. Their committed reports are regenerated manually.

## Reproducing

```bash
python -m pip install -e ".[dev]"
artificial-human cognition-eval --suite all --output-dir docs/benchmarks           # offline
artificial-human cognition-eval --suite prediction --format json                    # one suite, JSON to stdout
# opt-in external suites (local data only; see benchmarks/EXTERNAL_DATASETS.md)
artificial-human cognition-eval --suite longmemeval --suite digit-span \
  --longmemeval-path external-data/longmemeval/longmemeval_oracle.json \
  --digit-span-dir external-data/nemar/on003838/beh \
  --report-stem external-evaluation --output-dir docs/benchmarks
python -m pip install -e ".[dev,semantic]"
artificial-human cognition-eval --embedding-backend fastembed --output-dir docs/benchmarks
```

Exit status: `0` the evaluation completed (whatever the scores), `1` execution error, `3` a safety invariant (owner isolation) failed, `2` invalid arguments.

The committed report was generated with the local `BAAI/bge-small-en-v1.5` model so semantic variants are visible. CI runs the offline configuration on Python 3.12 and 3.13.

## References

- Awh, E., Belopolsky, A. V., & Theeuwes, J. (2012). Top-down versus bottom-up attentional control: a failed theoretical dichotomy. *Trends in Cognitive Sciences*, 16(8), 437–443.
- Berman, M. G., Jonides, J., & Lewis, R. L. (2009). In search of decay in verbal short-term memory. *Journal of Experimental Psychology: Learning, Memory, and Cognition*, 35(2), 317–333.
- Cer, D., Diab, M., Agirre, E., Lopez-Gazpio, I., & Specia, L. (2017). SemEval-2017 Task 1: Semantic textual similarity. *SemEval-2017*. arXiv:1708.00055.
- Cowan, N. (2001). The magical number 4 in short-term memory. *Behavioral and Brain Sciences*, 24(1), 87–114.
- Desimone, R., & Duncan, J. (1995). Neural mechanisms of selective visual attention. *Annual Review of Neuroscience*, 18, 193–222.
- Friston, K. (2005). A theory of cortical responses. *Philosophical Transactions of the Royal Society B*, 360, 815–836.
- Hodson, R., Mehta, M., & Smith, R. (2024). The empirical status of predictive coding and active inference. *Neuroscience & Biobehavioral Reviews*. doi:10.1016/j.neubiorev.2023.105473.
- Howard, M. W., & Kahana, M. J. (2002). A distributed representation of temporal context. *Journal of Mathematical Psychology*, 46, 269–299.
- Itti, L., & Koch, C. (2001). Computational modelling of visual attention. *Nature Reviews Neuroscience*, 2, 194–203.
- Johnson, M. K., Hashtroudi, S., & Lindsay, D. S. (1993). Source monitoring. *Psychological Bulletin*, 114(1), 3–28.
- Kumaran, D., & Maguire, E. A. (2007). Which computational mechanisms operate in the hippocampus during novelty detection? *Hippocampus*, 17(9), 735–748.
- Kutas, M., & Hillyard, S. A. (1980). Reading senseless sentences: brain potentials reflect semantic incongruity. *Science*, 207, 203–205.
- Lewandowsky, S., Oberauer, K., & Brown, G. D. A. (2009). No temporal decay in verbal short-term memory. *Trends in Cognitive Sciences*, 13(3), 120–126.
- Maharana, A., et al. (2024). Evaluating very long-term conversational memory of LLM agents. *ACL 2024*. arXiv:2402.17753.
- Miller, G. A. (1956). The magical number seven, plus or minus two. *Psychological Review*, 63(2), 81–97.
- Murdock, B. B. (1962). The serial position effect of free recall. *Journal of Experimental Psychology*, 64(5), 482–488.
- Oberauer, K., et al. (2018). Benchmarks for models of short-term and working memory. *Psychological Bulletin*, 144(9), 885–958. doi:10.1037/bul0000153.
- Rao, R. P. N., & Ballard, D. H. (1999). Predictive coding in the visual cortex. *Nature Neuroscience*, 2(1), 79–87.
- Theeuwes, J. (2010). Top-down and bottom-up control of visual selection. *Acta Psychologica*, 135(2), 77–99.
- Tulving, E. (1972). Episodic and semantic memory. In E. Tulving & W. Donaldson (Eds.), *Organization of Memory* (pp. 381–403). Academic Press.
- Wu, D., et al. (2025). LongMemEval: Benchmarking chat assistants on long-term interactive memory. *ICLR 2025*. arXiv:2410.10813.

Citations are starting points for investigation. They motivate the task design; they are not claims that the software reproduces the findings.
