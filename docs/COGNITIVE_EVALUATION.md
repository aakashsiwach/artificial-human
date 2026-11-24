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

## External datasets

The brief asked for externally authored, independently reviewed material where licensing allows. Candidates reviewed on 2026-11-24:

| Dataset | Status found | Decision |
| --- | --- | --- |
| LoCoMo (Maharana et al., ACL 2024) | Reported by secondary sources as CC BY-NC 4.0. The official repository licence was not retrieved. | Not adopted: non-commercial terms are incompatible with the project's Apache-2.0 distribution, and the licence is unverified. |
| LongMemEval (Wu et al., ICLR 2025) | Repository listed as MIT. Chat histories derive from ShareGPT and UltraChat. | Not adopted: ShareGPT consists of real user conversations (personal-data risk), and the licence was not verified at the source. |
| STS Benchmark (Cer et al., 2017) | Scores CC BY-SA 4.0; source texts carry separate licences, some requiring provider agreements. | Not adopted: mixed licensing cannot be verified for redistribution. |
| Oberauer et al. (2018) working-memory benchmarks | Published list of human findings, with reference datasets of human experiments. | Used only as a qualitative checklist; no data copied. |

**No external dataset is included.** All benchmarks are synthetic and project-authored, and the reports say so. This is a known limitation, and the corresponding roadmap item remains open.

## Reproducing

```bash
python -m pip install -e ".[dev]"
artificial-human cognition-eval --suite all --output-dir docs/benchmarks           # offline
artificial-human cognition-eval --suite prediction --format json                    # one suite, JSON to stdout
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
