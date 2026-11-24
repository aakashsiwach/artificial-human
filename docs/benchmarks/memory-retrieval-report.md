# Memory retrieval benchmark report

- Benchmark: `artificial-human-synthetic-memory-retrieval` v1.1.0 (sha256 `eb46c69b0819`)
- Generated: 2026-11-24T16:56:28+00:00 with Python 3.12.10
- Reference time for temporal queries: 2026-06-15T12:00:00+00:00
- Embedding model: `BAAI/bge-small-en-v1.5` (fastembed, 384 dims, in_process, fastembed-0.9.1)
- Queries: 13 tuning, 53 eval (metrics below are eval split only)
- Semantic similarity threshold: 0.570 (selected on tuning split)

Precision@K is the standard definition (relevant hits in top K divided by K); most queries have one or two relevant memories, so its ceiling is low. Rejection rate is the share of queries with no relevant memory that returned nothing.

## Overall results (eval split)

| Variant | Recall@1 | Recall@5 | Precision@1 | Precision@5 | MRR | nDCG@5 | Rejection | Owner leaks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lexical | 0.625 | 0.690 | 0.674 | 0.161 | 0.681 | 0.681 | 0.714 | 0 |
| semantic | 0.734 | 0.886 | 0.783 | 0.200 | 0.824 | 0.838 | 0.857 | 0 |
| hybrid | 0.755 | 0.908 | 0.804 | 0.204 | 0.848 | 0.861 | 0.714 | 0 |
| hybrid_temporal | 0.859 | 0.995 | 0.935 | 0.239 | 0.964 | 0.969 | 0.714 | 0 |
| ablation_lexical_temporal | 0.685 | 0.777 | 0.761 | 0.196 | 0.772 | 0.770 | 0.714 | 0 |
| ablation_semantic_temporal | 0.859 | 0.973 | 0.935 | 0.235 | 0.953 | 0.955 | 0.857 | 0 |
| ablation_hybrid_temporal_no_similarity_threshold | 0.815 | 0.973 | 0.891 | 0.235 | 0.931 | 0.939 | 0.000 | 0 |
| ablation_hybrid_temporal_no_contradiction_expansion | 0.859 | 0.995 | 0.935 | 0.239 | 0.967 | 0.972 | 0.714 | 0 |

## Task-level accuracy (eval split)

Temporal answer accuracy: the top result is a best-graded answer for temporal or historical questions. Source attribution: the top result comes from the expected source. Contradiction recall: every conflicting memory appears in the top 5.

| Variant | Temporal answer accuracy | Source attribution accuracy | Contradiction recall@5 |
| --- | --- | --- | --- |
| lexical | 0.273 | 1.000 | 1.000 |
| semantic | 0.273 | 1.000 | 1.000 |
| hybrid | 0.364 | 1.000 | 1.000 |
| hybrid_temporal | 0.909 | 1.000 | 1.000 |
| ablation_lexical_temporal | 0.636 | 1.000 | 1.000 |
| ablation_semantic_temporal | 0.909 | 1.000 | 1.000 |
| ablation_hybrid_temporal_no_similarity_threshold | 0.727 | 1.000 | 1.000 |
| ablation_hybrid_temporal_no_contradiction_expansion | 0.909 | 1.000 | 1.000 |

## Per-category results (MRR, or rejection rate for categories without relevant items)

| Category | lexical | semantic | hybrid | hybrid_temporal |
| --- | --- | --- | --- | --- |
| conceptual | 0.200 | 1.000 | 0.900 | 0.900 |
| contradiction | 1.000 | 1.000 | 1.000 | 1.000 |
| episodic | 0.600 | 0.900 | 0.900 | 0.900 |
| exact_keyword | 1.000 | 1.000 | 1.000 | 1.000 |
| historical_fact | 0.500 | 0.569 | 0.667 | 0.889 |
| irrelevant | rej 0.800 | rej 0.800 | rej 0.800 | rej 0.800 |
| owner_isolation | 1.000 / rej 0.500 | 1.000 / rej 1.000 | 1.000 / rej 0.500 | 1.000 / rej 0.500 |
| paraphrase | 1.000 | 1.000 | 1.000 | 1.000 |
| source_attribution | 1.000 | 1.000 | 1.000 | 1.000 |
| temporal_chronological | 0.067 | 0.200 | 0.200 | 1.000 |
| type_filter | 0.750 | 0.750 | 1.000 | 1.000 |

## Measured comparison

- `semantic` vs `lexical`: MRR +0.143, nDCG@5 +0.156, Recall@5 +0.196.
- `hybrid` vs `lexical`: MRR +0.167, nDCG@5 +0.179, Recall@5 +0.217.
- `hybrid_temporal` vs `lexical`: MRR +0.283, nDCG@5 +0.287, Recall@5 +0.304.
- `hybrid` vs `semantic`: MRR +0.023, nDCG@5 +0.023.
- `hybrid_temporal` vs `hybrid`: MRR +0.116, nDCG@5 +0.108.
- These differences come from one small synthetic dataset; treat them as a regression signal, not as evidence about general retrieval quality.

## Failure cases for `hybrid_temporal`

- `eval-type-01` (type_filter) "Lisbon" — expected ep_lisbon_arrive, ep_lisbon_tram, ep_lisbon_pasteis, ep_lisbon_depart; top results: ep_lisbon_tram, ep_lisbon_depart, ep_lisbon_arrive
- `eval-irr-02` (irrelevant) "How many moons does Neptune have?" — expected (nothing); top results: tides, night_owl
- `eval-own-04` (owner_isolation) "What is my preferred code editor?" — expected (nothing); top results: b_pref

## Threshold tuning (tuning split only)

Objective: mean reciprocal rank + rejection rate (tuning split only). Ties keep the lowest threshold. Selected: 0.570.

## Reproduce

```
pip install -e ".[dev,semantic]"
artificial-human memory-eval --backend fastembed --output-dir docs/benchmarks
```
