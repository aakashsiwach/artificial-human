# Bounded in-memory cognitive memory (issue #30)

## Runtime boundary

The active `ArtificialOrganism` now composes a dedicated `InMemoryCognitiveMemory` owner. It is separate from the legacy SQLite `MemoryRepository` class and the active `InMemoryMemoryRepository` no longer subclasses the SQLite implementation. `LongTermMemory` and `LearningSystem` depend on the storage-neutral `MemoryStore` structural contract. The legacy SQLite implementation remains isolated for explicit compatibility/offline tools.

No SQLite connection, SQL query, vector database, file write, or external service is required to construct the active memory store or execute the cognitive memory mechanisms. Optional embedding/index integrations remain separate derived-retrieval capabilities; they are not the source of truth for the volatile cognitive stores.

Runtime experience, cognitive associations, and learned strengths are process-local. A process restart creates empty cognitive stores. `ArtificialOrganism.reset_volatile_state()` explicitly clears memory records, specialized stores, working sessions, episodic replay/associative indexes, sleep state, and schema-extraction state.

## Distinct mechanisms and rules

| Store | Representation | Retrieval / learning | Capacity and forgetting |
|---|---|---|---|
| Working | Ordered active workspace entries | Recent context; salience decays on new input | 7 entries/session by default; 128 LRU sessions; weakest salience evicted |
| Episodic | Single-event traces bound to owner, consent, context, tick, tags and salience | Partial-cue overlap recall; retrieval strengthens trace | 2,000 traces; weakest salience × strength, then least accessed/oldest evicted |
| Semantic | Concepts, typed attributes, confidence and source evidence | Concept/attribute recall; repeated evidence merges; explicit associations; retrieval strengthens | 5,000 facts; weakest confidence × strength and least accessed evicted |
| Procedural | Task-keyed ordered step sequences with success/failure counts | Task recall; outcome observations update success rate and strength | 1,000 procedures; weakest strength × success rate and least accessed evicted |
| Spatial | Labeled 3D coordinates, context, tags and explicit place relations | Nearest-neighbour and label/tag recall | 2,000 places; least-accessed place evicted |
| Relational | Context-scoped co-activation links and contradiction suppression | Association activation and retrieval reinforcement | Existing bounded `RelationalMemoryNetwork` |
| Episodic replay | Temporal transitions between episode identifiers | Sequence replay and replay-dependent strengthening | Existing bounded `EpisodicReplayMemory` |

The capacity defaults are configurable when constructing `InMemoryCognitiveMemory`. Long-term provenance records in `InMemoryMemoryRepository` use a separate hard-capacity policy: writes fail explicitly at capacity rather than silently evicting user-provided source records.

## Provenance and permission

Every specialized long-term entry retains its owner, active consent grant and source record ID. The API projects a new consent-approved memory record into its corresponding specialized store. Corrections refresh the derived representation; deletion removes all derived entries for that source; consent revocation purges entries created under that grant; owner erasure clears all owner-scoped stores. Identifier-only relational and replay indexes are reset on source deletion or consent revocation because those links do not independently carry consent metadata.

The working-memory workspace is process-local and owner/session scoped. The active `BrainService` and `CognitiveArchitecture` share the same `WorkingMemorySessions` instance owned by `InMemoryCognitiveMemory`.

## Active cognitive-cycle integration

The cognitive cycle recalls episodic, semantic, procedural and spatial candidates directly from these structures and places selected content into the bounded working-memory context. This mechanism works without an LLM; the language provider remains a separate optional deliberation interface. Existing generic long-term retrieval remains available for provenance and legacy search, but it is not used as a substitute for the specialized representations.

## Reproducible tests

`pytest -q tests/test_cognitive_memory.py tests/test_in_memory_runtime.py tests/test_memory.py`

Tests cover:
- one-trial context/time binding and partial-cue episodic recall;
- semantic evidence merging, explicit concept associations and retrieval strengthening;
- procedural step retrieval and success/failure updates;
- coordinate nearest-neighbour recall and place relations;
- capacity eviction, owner isolation, source deletion and consent purge;
- dispatch from consented records into distinct stores;
- active cognitive-cycle recall into working memory;
- owner/runtime reset and the fact that the active volatile store is not a SQLite repository subclass.

## Research basis and limits

Multiple-memory-systems research motivates distinct representations and operating rules rather than one undifferentiated text table. Episodic memory emphasizes context-bound events; semantic memory supports decontextualized concept/attribute knowledge; procedural memory captures gradual task/skill learning; spatial memory organizes locations and relations; working memory maintains a bounded active workspace. The code implements explicit computational analogues, not biological replicas. Retrieval uses deterministic token overlap and fixed strengthening/eviction rules, not a learned neural retrieval process.

References:
- Schacter & Tulving, *What are the memory systems of 1994?* (five major memory systems): https://bpb-us-e1.wpmucdn.com/sites.harvard.edu/dist/3/137/files/2022/07/1memorysystems94_2.pdf
- Robins, Aronowitz & Stolk, *Memory Structure and Cognitive Maps*: https://www.ncbi.nlm.nih.gov/books/NBK583710/
- *Distinct mechanisms and functions of episodic memory* (2024): https://pmc.ncbi.nlm.nih.gov/articles/PMC11482257/
- Finley, *Expanded taxonomies of human memory* (2025): https://www.frontiersin.org/journals/cognition/articles/10.3389/fcogn.2024.1505549/full

## Limitations

These stores are deterministic bounded software mechanisms, not biologically validated memory systems. Retrieval is lexical/attribute overlap; there is no learned distributed semantic representation here. Procedural entries are descriptions and are not automatically executed. Spatial maps are Euclidean 3D coordinate stores, not navigation policies. Episodic replay links are identifier-only and are deliberately cleared when source permissions change. Restart loss is intentional for this phase; persistent identity is maintained in source-controlled code, not in personal memory.
