# Safety and privacy model

The [Research Charter](RESEARCH_CHARTER.md) defines the scientific and ethical boundaries for the long-term artificial-mind research programme. Any work involving neural data, human participants, animals, biological tissue, neural organoids, or biohybrid systems requires appropriate institutional, legal, ethical, and domain-expert governance before experimentation.

## Default-deny posture

- Personal memory is disabled until an owner creates a purpose-specific consent grant.
- Sensor input is manual and explicit. No camera, microphone, browser, filesystem crawl, or account connector is enabled.
- The baseline does not call hosted model providers.
- Semantic memory indexing is off unless configured. The supported local backend runs in-process; downloading model weights is the only network use. A remote embedding endpoint is refused unless the operator sets `ARTIFICIAL_HUMAN_EMBEDDING_ALLOW_REMOTE=true`, the endpoint uses HTTPS, and the owner holds an active `external_embedding` consent.
- Regeneration only creates proposal objects; it does not modify files, invoke a shell, deploy, or merge.
- External or consequential actions require a separate approval workflow.

## Memory controls

Every memory stores its source and consent ID. A write without active consent is rejected. Revoking consent blocks future writes and excludes those records from standard retrieval. Deletion removes individual records; owner erasure removes all memories and revokes all grants.

Embeddings are derived personal data and follow the same rules. Only consent-visible records are embedded or searched, and search is always scoped to one owner. Deleting a memory, revoking its consent, or erasing the owner removes the associated vectors, link rows and index-failure rows in the same transaction. A vector is stored only if the record is still present, still consented and unchanged at write time. Logs contain memory ids and error classes only, never content, vectors, or keys. Changed facts are kept as linked versions (each with its own provenance) rather than overwritten, and contradictions are preserved and surfaced rather than resolved automatically.

Personal clone examples use a separate personal_learning consent grant. New examples are pending review and are excluded from the training corpus until the owner approves them. Rejected examples remain inspectable but are not exported. Revoking the associated consent excludes examples from normal retrieval and export. No model is fine-tuned by this baseline.

This release is a developer baseline, not a certified privacy product. Backups, filesystem permissions, encryption at rest, audit retention, multi-user authorization, and secure key handling require more work before sensitive deployment.

## Threat model

Consider prompt injection in imported content, malicious documents, compromised connectors, accidental disclosure, hallucinations, unauthorized actions, and runaway resource use. Treat observed content as data, never as executable instructions. Models must not expand their own permissions, read arbitrary local files, discover secrets, or approve their own actions.

## Human review

A repair approval in this release only records that a named reviewer approved the proposal. It does not run the fix. Future approval must bind to an immutable diff, test evidence, target environment, and expiry. Deployment must be a separate permission from patch preparation.


## Active runtime storage and privacy status

The active local and public `ArtificialOrganism` paths now use the same bounded in-process memory adapter. It does not initialise SQLite, create personal-memory or identity files, or connect to an external database. Memory writes still require a purpose-specific in-memory consent grant; records retain owner, source and consent provenance. Revocation removes the owner's affected volatile records and derived state; process restart clears all runtime memories and consents.

The legacy SQLite repository is retained for explicit compatibility and offline evaluation only. Its driver is lazy-loaded and normal active-runtime startup does not open a database. CLI commands use RAM by default; operators must explicitly request legacy SQLite mode.

The active memory adapter bounds records, consents, embeddings, index failures, links and training examples. Neural simulation separately bounds neurons, synapses, pending events, synaptic currents and externally applied current. Capacity exhaustion is explicit; the repository does not silently evict memory records. These controls reduce runaway state growth but are not a certified privacy or security product. Stable identity and architecture remain version-controlled source and cannot be rewritten through the public API.

## Self-development safety boundary

The long-term system may generate candidate code and regression tests for its next version, but generated content is untrusted. Candidate work must occur in a separate, resource-limited workspace with no secrets, unrestricted shell, network, or access to the live process. The organism cannot approve its own patch, modify its live executable, grant itself permissions, merge, or deploy. Human review must bind approval to an immutable diff and reproducible test evidence; deployment requires separate approval.
