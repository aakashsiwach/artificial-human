# Memory retrieval benchmark report

- Benchmark: `artificial-human-synthetic-memory-retrieval` v1.0.0 (sha256 `abb1ebb4f069`)
- Generated: 2026-11-24T15:56:16+00:00 with Python 3.12.10
- Reference time for temporal queries: 2026-06-15T12:00:00+00:00
- Embedding model: `BAAI/bge-small-en-v1.5` (fastembed, 384 dims, in_process, fastembed-0.9.1)
- Queries: 13 tuning, 49 eval (metrics below are eval split only)
- Semantic similarity threshold: 0.570 (selected on tuning split)

Precision@K is the standard definition (relevant hits in top K divided by K); most queries have one or two relevant memories, so its ceiling is low. Rejection rate is the share of queries with no relevant memory that returned nothing.

## Overall results (eval split)

| Variant | Recall@1 | Recall@5 | Precision@1 | Precision@5 | MRR | nDCG@5 | Rejection | Owner leaks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lexical | 0.589 | 0.661 | 0.643 | 0.157 | 0.651 | 0.651 | 0.714 | 0 |
| semantic | 0.708 | 0.875 | 0.762 | 0.200 | 0.807 | 0.822 | 0.857 | 0 |
| hybrid | 0.732 | 0.899 | 0.786 | 0.205 | 0.833 | 0.847 | 0.714 | 0 |
| hybrid_temporal | 0.845 | 0.994 | 0.929 | 0.243 | 0.960 | 0.966 | 0.714 | 0 |
| ablation_lexical_temporal | 0.655 | 0.756 | 0.738 | 0.195 | 0.750 | 0.748 | 0.714 | 0 |
| ablation_semantic_temporal | 0.845 | 0.970 | 0.929 | 0.238 | 0.948 | 0.951 | 0.857 | 0 |
| ablation_hybrid_temporal_no_similarity_threshold | 0.798 | 0.970 | 0.881 | 0.238 | 0.925 | 0.933 | 0.000 | 0 |
| ablation_hybrid_temporal_no_contradiction_expansion | 0.845 | 0.994 | 0.929 | 0.243 | 0.964 | 0.969 | 0.714 | 0 |

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
| temporal_chronological | 0.067 | 0.200 | 0.200 | 1.000 |
| type_filter | 0.750 | 0.750 | 1.000 | 1.000 |

## Measured comparison

- `semantic` vs `lexical`: MRR +0.157, nDCG@5 +0.171, Recall@5 +0.214.
- `hybrid` vs `lexical`: MRR +0.182, nDCG@5 +0.197, Recall@5 +0.238.
- `hybrid_temporal` vs `lexical`: MRR +0.309, nDCG@5 +0.315, Recall@5 +0.333.
- `hybrid` vs `semantic`: MRR +0.026, nDCG@5 +0.025.
- `hybrid_temporal` vs `hybrid`: MRR +0.127, nDCG@5 +0.118.
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
