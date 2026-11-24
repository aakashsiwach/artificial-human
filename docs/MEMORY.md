# Long-term memory: semantic, hybrid and temporal retrieval

This document describes what the long-term memory subsystem actually implements, how to configure and evaluate it, and how far it is from biological human memory. The implementation lives in `src/artificial_human/memory/`. Design decisions are recorded in [ADR 0001](adr/0001-semantic-hybrid-temporal-memory.md).

**Scope note.** Everything here is an explicit computational abstraction: SQLite records, pretrained text embeddings, ranking formulas and rules. None of it simulates neurons, synapses or hippocampal circuitry. Biological references below explain *why* a function is worth modelling, not a claim that the code reproduces the biology.

## 1. Components

| Module | Responsibility |
| --- | --- |
| `memory/repository.py` | Source of truth: consent-gated records, temporal fields, supersession/contradiction links, embedding rows, schema migrations |
| `memory/temporal.py` | UTC timestamp handling and rule-based temporal intent ("now", "previously", "in January") |
| `memory/embeddings.py` | Replaceable embedding backends behind one interface |
| `memory/index.py` | Consent-aware indexing, re-indexing, failure tracking, vector similarity |
| `memory/search.py` | Unified hybrid search: filters, lexical/semantic/temporal channels, Reciprocal Rank Fusion, explanations, timeline |
| `memory/system.py` | `LongTermMemory` facade: write first, then index |
| `memory/benchmark/` | Synthetic dataset, metrics, evaluation runner and report writer |

## 2. Semantic memory

A memory's text is converted into a dense vector by a pretrained sentence-embedding model. A query is embedded the same way, and memories are ranked by cosine similarity. This lets "What is the boiling point of H2O?" retrieve "Water boils at approximately 100 degrees Celsius…" without shared keywords.

### Embedding interface

```python
class EmbeddingProvider(Protocol):
    def describe(self) -> EmbeddingModelInfo: ...        # backend, model_id, dimensions, version, locality
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
```

The domain model (`MemoryRecord`) does not depend on any backend. Vectors are only comparable inside one *embedding space*, identified as `backend:model_id:dimensions`.

### Backends

| Backend | Locality | Notes |
| --- | --- | --- |
| `none` (default) | — | No embeddings; retrieval is lexical + temporal only |
| `fastembed` | in-process | ONNX Runtime; default model `BAAI/bge-small-en-v1.5` (384 dims, MIT licence, ~67 MB). No PyTorch. Optional extra `semantic` |
| `openai_compatible` | loopback or remote | `POST {base_url}/embeddings`, for example a local Ollama server |

Semantic indexing is **off until configured**. Loading the fastembed model downloads model *weights* once into a cache directory; no memory content leaves the process. Set `ARTIFICIAL_HUMAN_EMBEDDING_OFFLINE=true` to forbid downloads and use only a cached model.

A *remote* endpoint (anything other than `localhost`, `127.0.0.1` or `::1`) is refused unless **all** of the following hold:

1. `ARTIFICIAL_HUMAN_EMBEDDING_ALLOW_REMOTE=true` is set by the operator;
2. the URL uses HTTPS;
3. the owner holds an active consent grant with purpose `external_embedding` (in addition to `personal_memory`).

Without the owner consent, nothing is sent and the status endpoint reports `lexical_only_external_consent_missing`.

### Storage

`memory_embeddings` stores, per memory and space: model id, model version, dimensions, a SHA-256 hash of the embedded text, a little-endian float32 vector (unit length), and the indexing time. Search uses exact brute-force cosine similarity (`math.sumprod`) over the owner's eligible vectors. This is adequate for personal-scale memory (thousands to low tens of thousands of records); approximate nearest-neighbour indexing is future work.

### Indexing lifecycle

- **Create** — `LongTermMemory.add` writes the record, then embeds it. The API response includes `index.status` (`indexed`, `failed`, `skipped`, `disabled`).
- **Correct** — changing content deletes the old vector in the same transaction as the update, then re-embeds.
- **Existing / missing** — `POST /memories/index/rebuild` (or `artificial-human memory-index rebuild`) embeds every eligible memory lacking a current vector.
- **Model change** — a rebuild removes vectors from every other space and embeds with the configured model; `--full` / `{"full": true}` re-embeds everything.
- **Failure** — the memory is never modified or rolled back. Only the failure class is recorded in `memory_index_failures`; lexical retrieval keeps working; a later rebuild retries.
- **Race safety** — a vector is stored only if, at write time, the record still exists for that owner, its consent is still active, and its content still matches the hash that was embedded.

### Eligibility

Only records visible under an active `personal_memory` consent are embedded or searched. Remote backends additionally need `external_embedding` consent.

## 3. Hybrid retrieval

`HybridMemorySearch.search(owner_id, query, top_k, filters, config, reference_time)` is the single search entry point used by the API, the CLI and the cognitive cycle.

### Pipeline

1. **Candidate set** — the owner's consent-visible records only (SQL `owner_id` + active consent). Owner isolation is enforced here, before any scoring.
2. **Hard filters** — memory types, time range, `valid_at`, `known_at`, source prefix, `fact_key`, metadata equality, `include_superseded`.
3. **Temporal intent** (optional) — may exclude records that cannot answer the question (see §4).
4. **Channels**
   - *lexical*: BM25 (k1 = 1.5, b = 0.75) after stop-word removal and conservative suffix stemming; temporal cue words are removed first so "now" or "first" do not act as topics;
   - *semantic*: cosine similarity, keeping only candidates at or above `semantic_min_similarity`;
   - *temporal*: re-orders items found by a content channel by time (recency, chronology, or history). For queries with no topical words ("What happened in January?") it may retrieve on its own, events first.
5. **Fusion** — weighted Reciprocal Rank Fusion, `score(d) = Σ_c w_c / (k + rank_c(d))`, k = 60, weights lexical 1.0, semantic 1.0, temporal 0.5. BM25 scores and cosine similarities are on different scales, so they are never added directly.
6. **Contradiction-aware expansion** — if a retrieved memory has a contradicting partner (explicit link, or another record with the same `fact_key` that is valid at the same time), the partner is placed directly after it, within `top_k`.
7. **Deterministic ordering** — ties are broken by recorded time and then id; the same data, configuration and `reference_time` always yield the same ranking.

### Explanations

Every result carries an `explanation`: memory type, source, consent id, epistemic status, channels used, per-channel rank and raw evidence (BM25 score and matched terms, cosine similarity, temporal rank), RRF contributions and fused score, temporal profile (status, validity interval, superseded-by/supersedes) and conflicting memory ids. `MemorySearchResponse.explain()` returns the same information without memory content, and that form is placed in the cognitive trace.

### Lexical-only operation

With no backend, an unloadable model, a failed query embedding, or missing external consent, search silently degrades to lexical + temporal. The response's `semantic` block states why.

## 4. Temporal memory

### Representation

| Question | Field |
| --- | --- |
| When did the event happen? | `occurred_at` |
| When did the system learn it? | `recorded_at` (stored as `created_at` for backward compatibility) |
| When did the fact become true? | `valid_from` |
| When did it stop being true? | `valid_to` |
| When was the record corrected? | `updated_at` |
| What replaced it, and when? | `supersedes` link (`from` newer → `to` older) with `effective_at` |
| Which records describe the same attribute? | `fact_key`, for example `owner.employer` |
| Is it established or a guess? | `epistemic_status`: `asserted` or `inferred` |

All timestamps must include a timezone offset; they are normalised to UTC ISO-8601 with microseconds. Legacy rows without an offset are read as UTC.

### Changing facts without losing history

- `POST /memories/{id}/supersede` creates a **new** record and a `supersedes` link. The old record is not modified. Its provenance (source, consent, recorded time) remains inspectable, and `GET /memories/{id}` returns the full version chain.
- The link's `effective_at` is the new record's `valid_from`, else its `occurred_at`, else its recorded time. A superseding record without `valid_from` is treated as valid from `effective_at`.
- A record can be superseded once; later changes supersede the latest version, keeping chains linear.
- An `inferred` record cannot supersede an `asserted` one, so uncertain inferences never displace established information.
- `PATCH /memories/{id}` is for *correcting an error in the record*, not for a change in the world. It edits in place and sets `updated_at`.
- Contradictions are never resolved automatically. `POST /memories/{id}/contradictions` records an explicit link; records sharing a `fact_key` that are valid at the same time are also reported as conflicting. Both remain retrievable and are flagged in `conflicts_with`. The language model is instructed to present conflicts with their sources instead of choosing silently.
- The newest record is **not** assumed to be correct: without an explicit supersession, two concurrently valid versions are both returned as a conflict.

### Temporal status at a reference time T

`superseded` (link effective ≤ T) → `expired` (`valid_to` ≤ T) → `not_yet_valid` (start or occurrence > T) → `not_yet_recorded` (only with `known_at`) → otherwise `valid`.

### Temporal queries

| Need | How |
| --- | --- |
| Chronological recall | `GET /memories/timeline` (ordered by occurred/valid/recorded time) |
| Time range | `start`/`end` filters, or "in January", "in March 2025", "in 2024", "yesterday", "last week/month/year" in the query |
| As of a date (world time) | `valid_at` filter |
| What the system knew at a date (transaction time) | `known_at` filter: hides records and supersession links recorded later |
| Current facts | cues such as "now", "currently", "these days" exclude superseded, expired and not-yet-valid records |
| Historical facts | cues such as "previously", "used to", "in the past" exclude the current version of a changed fact and rank ended facts first |
| Recency / chronology | "most recent", "latest" / "first", "in order", "timeline" add a temporal ranking channel |

A bare month resolves to the most recent such month that has already started at the reference time. Explicit API filters override inferred intent. Set `infer_temporal_intent: false` to disable the rules.

Example (§5 of the task brief): with "The user works at Company A" (valid from 2024-09) superseded by "The user moved to Company B" (valid from 2026-03), "Where does the user work now?" returns B only, "…previously?" returns A only, and "…in January?" returns A only. Both records, their sources and the link remain stored.

## 5. Consent, deletion and privacy

- Embedding, indexing and search consider only records under active `personal_memory` consent.
- **Deleting a memory** deletes its embeddings, index-failure rows and links in the same transaction (foreign keys also cascade).
- **Revoking consent** deletes every embedding and failure row derived from records under that grant, in the same transaction. Records stay hidden as before, and no further indexing or search processes them.
- **Owner erasure** removes memories, embeddings, links, failures and learning examples, and revokes all grants.
- **Export** includes temporal fields, links and embedding *metadata* (model, dimensions, version, time). Raw vectors are derived data and are not exported.
- Logging records memory ids and error classes only; memory content, vectors and keys are never logged. Error messages from backends never include input text.
- Conversations are still not stored. Working memory remains volatile and is never written to SQLite. Supersession and correction clear volatile sessions, as deletion and revocation already did.

## 6. Cognitive-cycle integration

Perception → **long-term memory retrieval** → attention → working memory → prediction → executive control → language generation.

- Before attention, `CognitiveArchitecture` runs the hybrid search with twice the memory budget, so attention selects among lexical, semantic and temporal candidates.
- Each attention candidate carries its retrieval channels. The trace's `attention.selected[*].retrieval_channels` and `memory.retrieval` (strategy, temporal intent, filters, per-result evidence, no content) show why each memory was retrieved.
- Selected memories enter the existing volatile working memory unchanged in behaviour (capacity, decay, reactivation).
- Memory evidence sent to the model includes temporal status, validity, supersession, epistemic status and conflicts. The model is told that superseded or expired memories describe the past and that inferred memories are hypotheses.
- Callers that pass an explicit `memories` list keep the previous lexical behaviour over exactly those records (backward compatibility).
- No long-term memory is written by the cycle. Writes still require explicit API calls with active consent.

## 7. Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `ARTIFICIAL_HUMAN_EMBEDDING_BACKEND` | `none` | `none`, `fastembed`, or `openai_compatible` |
| `ARTIFICIAL_HUMAN_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` (fastembed) | Model identifier |
| `ARTIFICIAL_HUMAN_EMBEDDING_CACHE_DIR` | `.artificial-human/models` | fastembed model cache |
| `ARTIFICIAL_HUMAN_EMBEDDING_OFFLINE` | `false` | Use only an already cached model |
| `ARTIFICIAL_HUMAN_EMBEDDING_THREADS` | runtime default | ONNX Runtime threads |
| `ARTIFICIAL_HUMAN_EMBEDDING_QUERY_PREFIX` | empty | Optional instruction prefix for query embeddings |
| `ARTIFICIAL_HUMAN_EMBEDDING_BASE_URL` | `http://127.0.0.1:11434/v1` | `openai_compatible` endpoint |
| `ARTIFICIAL_HUMAN_EMBEDDING_DIMENSIONS` | — (required) | `openai_compatible` vector size |
| `ARTIFICIAL_HUMAN_EMBEDDING_API_KEY` | empty | Optional bearer token (never logged or returned) |
| `ARTIFICIAL_HUMAN_EMBEDDING_ALLOW_REMOTE` | `false` | Operator opt-in for non-loopback endpoints (HTTPS only) |
| `ARTIFICIAL_HUMAN_MEMORY_MIN_SIMILARITY` | `0.57` | Semantic acceptance threshold (tuned for bge-small-en-v1.5) |

Local setup (PowerShell):

```powershell
python -m pip install -e ".[dev,semantic]"
$env:ARTIFICIAL_HUMAN_EMBEDDING_BACKEND = "fastembed"
artificial-human memory-index rebuild     # downloads the model once, then indexes eligible memories
artificial-human memory-index status
python -m artificial_human serve
```

Local Ollama embeddings:

```powershell
$env:ARTIFICIAL_HUMAN_EMBEDDING_BACKEND = "openai_compatible"
$env:ARTIFICIAL_HUMAN_EMBEDDING_MODEL = "nomic-embed-text"
$env:ARTIFICIAL_HUMAN_EMBEDDING_DIMENSIONS = "768"
```

After changing the model, run `artificial-human memory-index rebuild` and re-tune the threshold with `artificial-human memory-eval`, because similarity distributions differ between models.

## 8. API and CLI

| Interface | Purpose |
| --- | --- |
| `POST /memories` | Now accepts `occurred_at`, `valid_from`, `valid_to`, `fact_key`, `epistemic_status`; returns `index` status (existing fields unchanged) |
| `GET /memories/{id}` | Inspect a record, its version chain, links and index state |
| `PATCH /memories/{id}` | Correct a record; re-indexes when content changes |
| `POST /memories/{id}/supersede` | Record a changed fact as a new version |
| `POST /memories/{id}/contradictions` | Link two conflicting memories |
| `POST /memories/search` | Hybrid search with filters, `top_k`, `reference_time`, `include_content`, `infer_temporal_intent` |
| `GET /memories/timeline` | Chronological recall with `start`, `end`, `valid_at`, `known_at`, `kind` |
| `GET /memories/index/status` | Retrieval mode, model identity, indexed/missing/failed counts, schema version |
| `POST /memories/index/rebuild` | Index missing vectors (`{"full": true}` re-embeds all) |
| `artificial-human memory-search "query" [--kind K] [--top-k N] [--show-content]` | CLI search; content omitted unless requested |
| `artificial-human memory-index status\|rebuild [--full]` | CLI index management |
| `artificial-human memory-eval [--backend fastembed\|none] [--offline] [--output-dir DIR]` | Run the benchmark |

`GET /memories`, `GET /memories/export`, `DELETE /memories/{id}`, `DELETE /memories` and the consent endpoints keep their previous behaviour, with additional fields only.

## 9. Database migrations

`PRAGMA user_version` records the schema version. Each migration runs in its own `BEGIN IMMEDIATE` transaction and rolls back completely on failure. A database newer than the software is refused rather than modified.

- **v1** — baseline tables from milestones 0–1 (idempotent `CREATE … IF NOT EXISTS`, so unversioned databases adopt it safely).
- **v2** — adds nullable `occurred_at`, `valid_from`, `valid_to`, `fact_key`, `updated_at` and `epistemic_status DEFAULT 'asserted'` to `memories`; creates `memory_links`, `memory_embeddings`, `memory_index_failures` and a `fact_key` index. Existing rows are untouched. Their temporal fields are `NULL`, so they behave exactly as before, with their recorded time as their only time.

Embeddings for pre-existing memories are created by an explicit rebuild, never automatically during migration.

## 10. Evaluation

```powershell
python -m pip install -e ".[dev,semantic]"
artificial-human memory-eval --backend fastembed --output-dir docs/benchmarks
artificial-human memory-eval --backend none        # offline, lexical variants only (used in CI tests)
```

Outputs: `docs/benchmarks/memory-retrieval-results.json` (machine-readable: aggregates, per-category results, and failing queries with their top 5) and `docs/benchmarks/memory-retrieval-report.md`. Full per-query rankings for every variant are available in memory from `run_benchmark`, but are not written to disk.

**Dataset.** `src/artificial_human/memory/benchmark/dataset.json` (v1.1.0) contains 46 synthetic memories for two synthetic owners and 66 queries: 13 *tuning* and 53 *eval*. It covers exact keyword, paraphrase, conceptual similarity, episodic recall, temporal/chronological recall, historical facts, contradictions, source attribution, memory-type filters, irrelevant queries and owner isolation. Version 1.1.0 added four source-attribution queries to the eval split only, so the tuned threshold is unchanged. It contains no personal data. Memories are ingested with a deterministic clock and a fixed reference time.

**Protocol.** The semantic threshold is chosen by grid search (0.40–0.85, step 0.01) maximising MRR + rejection rate **on the tuning split only**, then frozen for the eval split. RRF k = 60 and the channel weights were fixed before evaluation, not tuned. Graded relevance (0–2) is used for nDCG with exponential gain. Precision@K uses the standard denominator K. Task-level metrics: *temporal answer accuracy* (top result is a best-graded answer for temporal and historical questions), *source attribution accuracy* (top result comes from the expected source), and *contradiction recall@5* (all conflicting memories appear in the top 5).

**Results (eval split, dataset v1.1.0, `BAAI/bge-small-en-v1.5`, threshold 0.57, Python 3.12):**

| Variant | Recall@5 | Precision@1 | MRR | nDCG@5 | Rejection | Temporal answer acc. | Owner leaks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lexical | 0.690 | 0.674 | 0.681 | 0.681 | 0.714 | 0.273 | 0 |
| semantic | 0.886 | 0.783 | 0.824 | 0.838 | 0.857 | 0.273 | 0 |
| hybrid | 0.908 | 0.804 | 0.848 | 0.861 | 0.714 | 0.364 | 0 |
| hybrid + temporal | 0.995 | 0.935 | 0.964 | 0.969 | 0.714 | 0.909 | 0 |

Source attribution accuracy and contradiction recall@5 are 1.000 for every variant. These queries name their topic explicitly, so they do not discriminate between variants.

Ablations of the full system: removing the lexical channel gives MRR 0.953 and rejection 0.857; removing the semantic channel gives MRR 0.772; removing the similarity threshold gives MRR 0.931 and rejection 0.000; removing contradiction expansion gives MRR 0.967. The full report is in [docs/benchmarks/memory-retrieval-report.md](benchmarks/memory-retrieval-report.md).

**What the measurements support, and what they do not:**

- On this dataset, hybrid ranking beats both single channels on MRR and nDCG@5. The gain over semantic-only is small (+0.023 MRR); most of the improvement over lexical comes from the embedding channel.
- Temporal reasoning accounts for the largest gain on temporal and historical categories: temporal/chronological MRR rises from 0.200 to 1.000, and historical facts from 0.667 to 0.889.
- Adding the lexical channel **lowers** irrelevant-query rejection compared with semantic-only (0.714 vs 0.857). Lexical false positives such as "moons" → "Moon" in a tides fact, or "prefer" matching another preference, pass through fusion.
- Without the similarity threshold the embedding channel returns something for every query (rejection 0.000), so the threshold is necessary.
- Contradiction expansion does not improve ranking metrics here (MRR −0.003) because relevant conflicting memories were usually already retrieved. It is kept for transparency, so conflicting evidence appears beside a retrieved claim. That is a safety property, not a ranking gain.
- The dataset is small (53 eval queries), synthetic, English-only and authored by the same project that built the system. The numbers are a regression signal, not evidence of general retrieval quality. The tuning split is only 13 queries.

**Known failure cases (hybrid + temporal):** a type-filtered "Lisbon" query misses the Belém event, which never names Lisbon; "How many moons does Neptune have?" retrieves the tides fact through the stem "moon"; one owner-isolation query, which asks owner B for an editor preference B never stated, returns B's unrelated explanation preference. No result from another owner was ever returned.

The broader [cognitive evaluation](COGNITIVE_EVALUATION.md) runs this benchmark as its `memory` suite. It also measures multi-turn recall and context switching through the full cognitive cycle.

## 11. Comparison with biological memory

| Biological phenomenon (selected evidence) | What is implemented | What is missing |
| --- | --- | --- |
| Episodic vs semantic memory (Tulving, 1972) | Typed records; events carry `occurred_at` | No automatic semanticisation of repeated episodes |
| Hippocampal indexing of distributed cortical traces (Teyler & DiScenna, 1986) | Dense vectors as a content-addressable index over records | Vectors come from a frozen pretrained model, not learned from the owner's experience; no binding of multimodal features |
| Pattern completion from partial cues | Semantic similarity retrieves from paraphrased or partial cues | No pattern separation; no attractor dynamics; similar memories are not actively orthogonalised |
| Complementary learning systems and systems consolidation (McClelland, McNaughton & O'Reilly, 1995) | — (records are immediately and permanently retrievable) | No gradual transfer, replay-driven consolidation or schema integration; sleep remains proposal-only |
| Temporal context and contiguity effects (Howard & Kahana, 2002) | Explicit timestamps, chronological ordering, rule-based temporal cues | No drifting internal context; no contiguity-based cueing (recalling one event does not cue its neighbours) |
| Source monitoring (Johnson, Hashtroudi & Lindsay, 1993) | Every record and version keeps source, consent and recorded time; conflicts are surfaced | No learned source credibility; no confidence calibration |
| Reconsolidation and updating (Nader, Schafe & LeDoux, 2000) | Explicit, owner-driven supersession that preserves the old trace | No retrieval-induced modification; old versions are never altered (deliberately, for auditability) |
| Forgetting, decay and interference (e.g. Ebbinghaus; Anderson's retrieval-induced forgetting) | — | No decay, no retrieval-practice strengthening, no interference; recall does not change memory strength |
| Emotional modulation of memory | Explicit `salience` metadata only (pre-existing) | No affective appraisal (out of scope for this phase) |
| Encoding specificity (Tulving & Thomson, 1973) | Query-time context limited to text and time cues | No encoding-context representation (place, state, task) |

The project therefore implements *engineering analogues* of a few functional properties: content-addressable recall, temporal organisation, source provenance and the preservation of changed beliefs. It does not implement the mechanisms that produce them in the brain.

## 12. Open research questions

- Can retrieval-dependent strengthening and decay be added without compromising auditability or owner control?
- How should contiguity-based cueing (temporal context drift) be modelled, and does it measurably improve episodic recall?
- Can source credibility be learned from owner corrections without turning inferences into facts?
- Would owner-reviewed consolidation (episodes into semantic summaries) improve recall while keeping provenance?
- How should temporal intent be detected beyond English keyword rules, and how often do the rules misfire on real queries?
- At what memory scale does brute-force search need an ANN index, and how should deletion guarantees carry over to it?
- How well does the tuned threshold transfer to new data and other embedding models? A larger, held-out, externally authored benchmark is needed.
