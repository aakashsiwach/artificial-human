# ADR 0001: Semantic, hybrid and temporal long-term memory

- **Status:** Accepted
- **Date:** 2026-11-24
- **Context:** Milestone 1, "Add semantic/vector/hybrid and temporal memory with measured retrieval evaluations"

## Context

Long-term memory was a consent-gated SQLite table with BM25-style lexical retrieval over at most 500 records, loaded into Python per request. Paraphrased queries failed, there was no representation of event time or fact validity, updates could only overwrite or duplicate, and nothing measured retrieval quality.

Constraints: local-first and privacy-preserving; Python 3.12 and 3.13; dependency-light CI without network-dependent tests; existing consent, provenance, deletion and owner-isolation guarantees must hold; existing databases and API consumers must keep working.

## Decisions

### 1. Keep SQLite as the single store, including vectors

Vectors live in a `memory_embeddings` table next to the records they describe. They are compared with exact brute-force cosine similarity in Python (`math.sumprod` over float32 BLOBs).

- **Why:** one transactional store means deleting a memory or revoking consent removes its vectors atomically (in the same transaction, with foreign-key cascades as a backstop). Consent and owner filters are applied with the same SQL that guards records. No extra service or native extension is needed.
- **Alternatives considered:** `sqlite-vec` (native extension; packaging varies by platform, and deletion semantics must be kept in sync), FAISS/hnswlib (separate index files that must be kept consistent with SQLite, which makes guaranteed deletion harder), Chroma/Qdrant (another service or a large dependency).
- **Consequence:** search cost is O(N·d) per query. That is fine at personal scale (about 10⁴ records). Beyond that, an ANN index must be added together with tests proving that deletion and revocation also remove index entries.

### 2. Embedding backends behind a narrow interface; fastembed as the optional local default

`EmbeddingProvider` exposes `describe()`, `embed_documents()` and `embed_query()`. Backends are `fastembed` (in-process ONNX) and `openai_compatible` (loopback HTTP such as Ollama, or remote by explicit opt-in). The default backend is `none`.

- **Why fastembed:** real pretrained sentence embeddings (`BAAI/bge-small-en-v1.5`, MIT licence, 384 dimensions, about 67 MB) through ONNX Runtime, without PyTorch. It is installed as an optional extra (`.[semantic]`), so the core package and CI stay light.
- **Why off by default:** loading a model downloads weights once. That network side effect should be an explicit operator choice. Without it, retrieval is lexical + temporal and the status endpoint says so.
- **Remote services** require an operator flag, HTTPS, and an owner `external_embedding` consent, so memories are never silently sent off-machine.
- **Alternatives considered:** sentence-transformers (pulls in PyTorch, which is heavyweight for this purpose); hashing or random projections (not semantic; rejected by requirement); a cloud API by default (violates local-first).

### 3. Reciprocal Rank Fusion instead of score blending

Channel outputs are fused by weighted RRF (k = 60).

- **Why:** BM25 scores are unbounded and corpus-dependent; cosine similarity is bounded but model-dependent. A linear blend would need per-corpus normalisation and tuning. RRF uses ranks only, is deterministic, explainable as per-channel contributions, and robust (Cormack, Clarke & Büttcher, 2009).
- **Rejection of irrelevant queries** comes from channel-level gates: BM25 needs a matching non-stop-word term, and the semantic channel needs similarity ≥ a threshold tuned on a held-out tuning split.
- **Consequence:** fusion cannot express "lexical match is weak evidence". The benchmark shows that lexical false positives reduce rejection, which is a known weakness.

### 4. Bitemporal records with explicit supersession links

Records gain `occurred_at`, `valid_from`, `valid_to`, `fact_key`, `epistemic_status` and `updated_at`. `created_at` remains the transaction time (exposed as `recorded_at`). A `memory_links` table stores `supersedes` and `contradicts` relations.

- **Why:** a changed fact must not erase history, and each version keeps its own provenance. Supersession never edits the old record; validity end is derived from the link. Contradictions are recorded and surfaced, never auto-resolved. `known_at` queries reconstruct what the system knew at a time.
- **Guards:** an `inferred` record cannot supersede an `asserted` one; a record can be superseded only once (linear chains).
- **Alternatives considered:** in-place updates with a history table (blurs correction versus change, and history rows complicate deletion); event sourcing (heavier than needed).

### 5. Rule-based temporal intent

Simple, documented English cue rules ("now", "previously", "in January", "most recent", …) map a query to an intent, which can exclude records or add a temporal ranking channel. Explicit API filters override inferred intent, and inference can be disabled.

- **Why:** transparent, testable and deterministic; enough to evaluate temporal retrieval.
- **Consequence:** English-only and brittle to phrasing. A learned or LLM-assisted temporal parser is future work and must be evaluated against these rules.

### 6. Versioned migrations through `PRAGMA user_version`

Each migration runs in its own transaction, so a failure rolls back completely. Newer schemas are refused. The baseline migration is idempotent, so unversioned databases adopt versioning safely. New columns are nullable or defaulted, so existing rows are unchanged.

### 7. Write first, index second

The record is the source of truth. Indexing happens after the write and only stores a vector if the record is still visible, still consented and still has the same content hash. Failures are recorded by error class only, and never roll back or alter memories.

## Consequences

- Semantic retrieval works through a real local model when configured. CI tests the plumbing with a test-only fake and runs the lexical benchmark offline; a separate CI job runs the full model benchmark.
- New tables and columns are additive, and existing API responses gain fields only.
- Remaining limits: brute-force scale, English-only cue rules, a small synthetic benchmark, a threshold tuned for one model, and no decay or consolidation dynamics.
