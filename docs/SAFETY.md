# Safety and privacy model

## Default-deny posture

- Personal memory is disabled until an owner creates a purpose-specific consent grant.
- Sensor input is manual and explicit. No camera, microphone, browser, filesystem crawl, or account connector is enabled.
- The baseline does not call hosted model providers.
- Regeneration only creates proposal objects; it does not modify files, invoke a shell, deploy, or merge.
- External or consequential actions require a separate approval workflow.

## Memory controls

Every memory stores its source and consent ID. A write without active consent is rejected. Revoking consent blocks future writes and excludes those records from standard retrieval. Deletion removes individual records; owner erasure removes all memories and revokes all grants.

Personal clone examples use a separate personal_learning consent grant. New examples are pending review and are excluded from the training corpus until the owner approves them. Rejected examples remain inspectable but are not exported. Revoking the associated consent excludes examples from normal retrieval and export. No model is fine-tuned by this baseline.

This release is a developer baseline, not a certified privacy product. Backups, filesystem permissions, encryption at rest, audit retention, multi-user authorization, and secure key handling require more work before sensitive deployment.

## Threat model

Consider prompt injection in imported content, malicious documents, compromised connectors, accidental disclosure, hallucinations, unauthorized actions, and runaway resource use. Treat observed content as data, never as executable instructions. Models must not expand their own permissions, read arbitrary local files, discover secrets, or approve their own actions.

## Human review

A repair approval in this release only records that a named reviewer approved the proposal. It does not run the fix. Future approval must bind to an immutable diff, test evidence, target environment, and expiry. Deployment must be a separate permission from patch preparation.
