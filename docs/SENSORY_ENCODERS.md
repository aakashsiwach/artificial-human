# Permission-scoped synthetic sensory encoders (issue #27)

## Scope and safety boundary

This implementation defines a synthetic/local-fixture sensory ingestion API. It does not open a camera, microphone, operating-system sensor, browser permission prompt, network stream, or physical device. There is no implicit device adapter. A caller must construct a typed event and obtain a modality-specific grant with `explicit_consent=True`; an empty permission ledger denies all modalities.

Permissions are in-memory and scoped to one modality, purpose, event-time interval, maximum event count, and maximum sample count. Revocation blocks future events and immediately removes retained session percepts for that modality. The two retention modes are:

- `EPHEMERAL`: the pipeline returns an encoded percept to the caller but does not retain it.
- `SESSION`: the pipeline retains encoded percepts in a bounded in-memory deque until eviction, explicit clearing, or revocation.

Raw event sample tuples are not retained by the pipeline. Encoded population codes and provenance are retained only under the explicit session policy. These controls are application-level policy primitives, not operating-system camera/microphone permissions or a security boundary against a malicious caller.

## Modalities and deterministic feature maps

Each modality has an independent encoder and maps its controlled numeric fixture into four modality-specific features before using the shared population encoder:

| Modality | Example derived features |
|---|---|
| Vision | mean, contrast, adjacent-sample change, peak |
| Audition | RMS, spread, zero-crossing rate, peak |
| Touch | mean, peak, positive-contact fraction, spread |
| Vestibular | mean, spread, RMS, adjacent change |
| Proprioception | mean, spread, peak, RMS |
| Interoception | mean, RMS, spread, peak |

These are deterministic engineering fixtures, not calibrated sensor models or medical measurements. Numeric values are bounded to prevent unbounded resource use and overflow. Missing samples remain explicit `None` values at the event boundary, are excluded from feature statistics, and reduce confidence; they are not silently imputed as zero.

## Time, uncertainty, alignment and neural projection

Every event has a non-negative integer timestamp, explicit provenance, and confidence. The resulting population code carries the timestamp and confidence-derived uncertainty. The session alignment operation chooses the latest retained percept per modality at or before a requested timestamp and within a caller-supplied window. Missing, expired, ungranted, or out-of-window modalities are represented as `None`, not fabricated observations.

`project_percept` uses the existing population projector to map encoded activity to named LIF neurons. The simulator receives population currents; it does not receive the original event or its raw sample tuple.

## Reproduction and tests

Run the isolated deterministic tests:

`pytest -q tests/test_sensory_encoders.py`

The tests cover all six modality encoders, default-deny and explicit consent, modality scoping, expiry, usage and sample bounds, ephemeral/session retention, revocation purge, missing samples, temporal alignment, absent modalities, and modality mismatch.

## Limitations

No real-world sensing is implemented. There is no camera or microphone integration, sensor calibration, physiological interpretation, learned modality fusion, cross-modal state estimator, or persistent permission database. Issue #28 handles multisensory integration and state estimation separately. No sensitive neural or medical data ingestion is part of this issue.
