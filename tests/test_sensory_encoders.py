from __future__ import annotations

import pytest

from artificial_human.neural.sensory import (
    AuditionEncoder,
    InteroceptionEncoder,
    ModalityEncoder,
    ProprioceptionEncoder,
    RetentionPolicy,
    SensoryEncoderSuite,
    SensoryEvent,
    SensoryModality,
    SensoryPermissionLedger,
    SensoryPipeline,
    SensoryProvenance,
    TouchEncoder,
    VestibularEncoder,
    VisionEncoder,
    PermissionDeniedError,
)


def grant(ledger: SensoryPermissionLedger, modality: SensoryModality, *, retention: RetentionPolicy = RetentionPolicy.EPHEMERAL, max_events: int = 8) -> None:
    ledger.grant(
        modality,
        purpose="deterministic synthetic encoder test",
        granted_at=0,
        expires_at=100,
        retention=retention,
        max_events=max_events,
        max_samples=32,
        explicit_consent=True,
    )


@pytest.mark.parametrize(
    ("modality", "encoder_type"),
    [
        (SensoryModality.VISION, VisionEncoder),
        (SensoryModality.AUDITION, AuditionEncoder),
        (SensoryModality.TOUCH, TouchEncoder),
        (SensoryModality.VESTIBULAR, VestibularEncoder),
        (SensoryModality.PROPRIOCEPTION, ProprioceptionEncoder),
        (SensoryModality.INTEROCEPTION, InteroceptionEncoder),
    ],
)
def test_each_modality_has_deterministic_independent_encoder(
    modality: SensoryModality, encoder_type: type[ModalityEncoder]
) -> None:
    event = SensoryEvent(modality, (0.1, None, 0.4, -0.2), 7, 0.8)
    encoder = encoder_type()
    suite = SensoryEncoderSuite(units_per_feature=5)
    from artificial_human.neural.population import PopulationEncoder

    first = encoder.encode(event, PopulationEncoder(units_per_feature=5))
    second = encoder.encode(event, PopulationEncoder(units_per_feature=5))
    assert first == second
    assert len(first.code.activations) == 20
    assert first.timestamp == 7
    assert first.missing_samples == 1
    assert first.provenance is SensoryProvenance.SYNTHETIC
    assert first.code.uncertainty > 0.2
    assert suite.encode(event, _permission(modality)).code == first.code


def _permission(modality: SensoryModality):
    from artificial_human.neural.sensory import SensoryPermission

    return SensoryPermission(
        modality=modality,
        purpose="unit test",
        granted_at=0,
        expires_at=100,
        retention=RetentionPolicy.EPHEMERAL,
        max_events=4,
        max_samples=32,
    )


def test_permissions_are_default_deny_and_require_explicit_consent() -> None:
    ledger = SensoryPermissionLedger()
    pipeline = SensoryPipeline(ledger=ledger)
    event = SensoryEvent(SensoryModality.VISION, (0.1, 0.2), 1)
    with pytest.raises(PermissionDeniedError, match="disabled by default"):
        pipeline.ingest(event)
    with pytest.raises(PermissionDeniedError, match="explicit consent"):
        ledger.grant(
            SensoryModality.VISION,
            purpose="test",
            granted_at=0,
            expires_at=10,
            explicit_consent=False,
        )


def test_permission_is_modality_scoped_expires_and_enforces_event_budget() -> None:
    ledger = SensoryPermissionLedger()
    grant(ledger, SensoryModality.VISION, max_events=1)
    pipeline = SensoryPipeline(ledger=ledger)
    pipeline.ingest(SensoryEvent(SensoryModality.VISION, (0.1,), 1))
    with pytest.raises(PermissionDeniedError, match="budget"):
        pipeline.ingest(SensoryEvent(SensoryModality.VISION, (0.1,), 2))
    with pytest.raises(PermissionDeniedError, match="disabled"):
        pipeline.ingest(SensoryEvent(SensoryModality.AUDITION, (0.1,), 2))
    with pytest.raises(PermissionDeniedError, match="permission window"):
        pipeline.ingest(SensoryEvent(SensoryModality.VISION, (0.1,), 100))


def test_retention_is_explicit_bounded_and_revocation_purges_session_data() -> None:
    ledger = SensoryPermissionLedger()
    grant(ledger, SensoryModality.TOUCH, retention=RetentionPolicy.SESSION, max_events=8)
    pipeline = SensoryPipeline(ledger=ledger, capacity=2)
    for timestamp in (1, 2, 3):
        pipeline.ingest(SensoryEvent(SensoryModality.TOUCH, (0.1, 0.2), timestamp))
    assert pipeline.retained_count == 2
    pipeline.revoke(SensoryModality.TOUCH)
    assert pipeline.retained_count == 0
    with pytest.raises(PermissionDeniedError):
        pipeline.ingest(SensoryEvent(SensoryModality.TOUCH, (0.1,), 4))


def test_ephemeral_events_are_not_retained_and_session_clear_erases_percepts() -> None:
    ledger = SensoryPermissionLedger()
    grant(ledger, SensoryModality.AUDITION, retention=RetentionPolicy.EPHEMERAL)
    pipeline = SensoryPipeline(ledger=ledger)
    percept = pipeline.ingest(SensoryEvent(SensoryModality.AUDITION, (0.1, -0.2), 1))
    assert percept.code.timestamp == 1
    assert pipeline.retained_count == 0

    grant(ledger, SensoryModality.AUDITION, retention=RetentionPolicy.SESSION)
    pipeline.ingest(SensoryEvent(SensoryModality.AUDITION, (0.1, -0.2), 2))
    assert pipeline.retained_count == 1
    pipeline.clear_session()
    assert pipeline.retained_count == 0


def test_alignment_preserves_absent_modalities_and_temporal_window() -> None:
    ledger = SensoryPermissionLedger()
    grant(ledger, SensoryModality.VISION, retention=RetentionPolicy.SESSION)
    grant(ledger, SensoryModality.TOUCH, retention=RetentionPolicy.SESSION)
    pipeline = SensoryPipeline(ledger=ledger)
    vision = pipeline.ingest(SensoryEvent(SensoryModality.VISION, (0.1, 0.2), 8))
    pipeline.ingest(SensoryEvent(SensoryModality.TOUCH, (0.2, 0.3), 2))
    aligned = pipeline.align(10, window=3)
    assert aligned.get(SensoryModality.VISION) == vision
    assert aligned.get(SensoryModality.TOUCH) is None
    assert aligned.get(SensoryModality.AUDITION) is None


def test_missing_samples_reduce_confidence_without_zero_imputation() -> None:
    ledger = SensoryPermissionLedger()
    grant(ledger, SensoryModality.PROPRIOCEPTION)
    pipeline = SensoryPipeline(ledger=ledger)
    full = pipeline.ingest(SensoryEvent(SensoryModality.PROPRIOCEPTION, (0.1, 0.3), 1, 1.0))
    partial = pipeline.ingest(SensoryEvent(SensoryModality.PROPRIOCEPTION, (0.1, None), 2, 1.0))
    assert partial.missing_samples == 1
    assert partial.code.uncertainty > full.code.uncertainty


def test_event_and_permission_resource_bounds_are_validated() -> None:
    with pytest.raises(ValueError, match="1..256"):
        SensoryEvent(SensoryModality.VISION, (), 1)
    with pytest.raises(ValueError, match="at least one"):
        SensoryEvent(SensoryModality.VISION, (None, None), 1)
    with pytest.raises(ValueError, match="finite"):
        SensoryEvent(SensoryModality.VISION, (float("nan"),), 1)
    with pytest.raises(ValueError, match="1000000"):
        SensoryEvent(SensoryModality.VISION, (1e100,), 1)
    ledger = SensoryPermissionLedger()
    with pytest.raises(ValueError, match="max_samples"):
        ledger.grant(
            SensoryModality.VISION,
            purpose="test",
            granted_at=0,
            expires_at=10,
            max_samples=257,
            explicit_consent=True,
        )


def test_modality_encoders_reject_cross_modality_events() -> None:
    with pytest.raises(ValueError, match="another modality"):
        VisionEncoder().encode(
            SensoryEvent(SensoryModality.AUDITION, (0.1,), 1),
            SensoryEncoderSuite().population_encoder,
        )



def test_event_rejects_mutable_or_untyped_sample_payloads_and_non_boolean_consent() -> None:
    with pytest.raises(ValueError, match="must be a tuple"):
        SensoryEvent(SensoryModality.VISION, [0.1, 0.2], 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        SensoryEvent(SensoryModality.VISION, ("0.1",), 1)  # type: ignore[arg-type]

    ledger = SensoryPermissionLedger()
    with pytest.raises(PermissionDeniedError, match="explicit consent"):
        ledger.grant(
            SensoryModality.VISION,
            purpose="test",
            granted_at=0,
            expires_at=10,
            explicit_consent="yes",  # type: ignore[arg-type]
        )
