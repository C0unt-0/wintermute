"""Tests for wintermute.db.models — ORM model CRUD and relationships."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from wintermute.db.models import (
    AdversarialCycle,
    AdversarialVariant,
    Base,
    EtlRun,
    EtlRunSource,
    Model,
    Sample,
    ScanResult,
)


@pytest.fixture()
def db_session():
    """Yield an in-memory SQLite session with all tables created."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


# ------------------------------------------------------------------
# CRUD tests
# ------------------------------------------------------------------


def test_sample_crud(db_session: Session):
    """Create a Sample, commit, and read it back."""
    sample = Sample(
        sha256="a" * 64,
        family="Ramnit",
        label=1,
        source="malware_bazaar",
        opcode_count=4200,
        file_type="PE32",
        file_size_bytes=123456,
        metadata_={"tags": ["packed", "upx"]},
    )
    db_session.add(sample)
    db_session.commit()

    result = db_session.query(Sample).filter_by(sha256="a" * 64).one()
    assert result.family == "Ramnit"
    assert result.label == 1
    assert result.source == "malware_bazaar"
    assert result.opcode_count == 4200
    assert result.file_type == "PE32"
    assert result.file_size_bytes == 123456
    assert result.metadata_ == {"tags": ["packed", "upx"]}
    assert result.created_at is not None
    assert result.updated_at is not None


def test_scan_result_crud(db_session: Session):
    """Create a ScanResult with probabilities JSON and read it back."""
    probs = {"safe": 0.15, "malicious": 0.85}
    scan = ScanResult(
        sha256="b" * 64,
        filename="evil.exe",
        file_size_bytes=98765,
        predicted_family="Gatak",
        predicted_label=1,
        confidence=0.85,
        probabilities=probs,
        model_version="v1.0.0",
        execution_time_ms=42.5,
        source_ip="192.168.1.100",
    )
    db_session.add(scan)
    db_session.commit()

    result = db_session.query(ScanResult).one()
    assert result.sha256 == "b" * 64
    assert result.filename == "evil.exe"
    assert result.confidence == 0.85
    assert result.probabilities == probs
    assert result.predicted_family == "Gatak"
    assert result.execution_time_ms == 42.5
    assert result.id is not None  # UUID was auto-generated
    assert result.scanned_at is not None


def test_model_lifecycle(db_session: Session):
    """Create a Model with status='staged' and verify defaults."""
    model = Model(
        version="v1.0.0",
        weights_path="/models/v1.0.0/weights.safetensors",
        vocab_size=5000,
        num_classes=2,
        dims=128,
        best_val_macro_f1=0.92,
    )
    db_session.add(model)
    db_session.commit()

    result = db_session.query(Model).filter_by(version="v1.0.0").one()
    assert result.status == "staged"
    assert result.architecture == "WintermuteMalwareDetector"
    assert result.max_seq_length == 2048
    assert result.vocab_size == 5000
    assert result.num_classes == 2
    assert result.best_val_macro_f1 == 0.92
    assert result.created_at is not None
    assert result.promoted_at is None
    assert result.retired_at is None


def test_etl_run_with_sources(db_session: Session):
    """Create an EtlRun with EtlRunSources and verify the relationship."""
    etl_run = EtlRun(
        config_hash="c" * 64,
        config={"sources": ["malware_bazaar", "virusshare"]},
        total_samples=1500,
        vocab_size=5000,
        num_classes=9,
        output_dir="/data/processed",
    )
    source1 = EtlRunSource(
        source_name="malware_bazaar",
        samples_extracted=1000,
        samples_skipped=50,
        samples_failed=10,
        families_found={"Ramnit": 400, "Gatak": 300, "Other": 300},
    )
    source2 = EtlRunSource(
        source_name="virusshare",
        samples_extracted=500,
        samples_skipped=20,
        samples_failed=5,
        families_found={"Lollipop": 200, "Kelihos": 300},
    )
    etl_run.sources.append(source1)
    etl_run.sources.append(source2)

    db_session.add(etl_run)
    db_session.commit()

    result = db_session.query(EtlRun).one()
    assert result.total_samples == 1500
    assert result.config == {"sources": ["malware_bazaar", "virusshare"]}
    assert len(result.sources) == 2

    names = sorted(s.source_name for s in result.sources)
    assert names == ["malware_bazaar", "virusshare"]

    # Verify back-reference
    for src in result.sources:
        assert src.etl_run_id == result.id


def test_adversarial_variant(db_session: Session):
    """Create Sample + AdversarialCycle + AdversarialVariant and verify relationships."""
    sample = Sample(sha256="d" * 64, family="Ramnit", label=1)
    db_session.add(sample)
    db_session.flush()

    cycle = AdversarialCycle(
        cycle_number=1,
        episodes_played=100,
        total_evasions=15,
        evasion_rate=0.15,
    )
    db_session.add(cycle)
    db_session.flush()

    variant = AdversarialVariant(
        parent_sha256=sample.sha256,
        cycle_id=cycle.id,
        mutated_token_ids=[10, 42, 99],
        mutations_applied=["nop_insert", "register_swap"],
        mutation_count=2,
        modification_pct=3.5,
        confidence_before=0.95,
        confidence_after=0.40,
        confidence_delta=-0.55,
        achieved_evasion=True,
    )
    db_session.add(variant)
    db_session.commit()

    # Verify variant
    result = db_session.query(AdversarialVariant).one()
    assert result.parent_sha256 == "d" * 64
    assert result.cycle_id == cycle.id
    assert result.achieved_evasion is True
    assert result.used_in_retraining is False
    assert result.mutation_count == 2
    assert result.confidence_delta == -0.55
    assert result.mutations_applied == ["nop_insert", "register_swap"]

    # Verify cycle -> variants relationship
    cycle_result = db_session.query(AdversarialCycle).one()
    assert len(cycle_result.variants) == 1
    assert cycle_result.variants[0].id == result.id

    # Verify variant -> cycle relationship
    assert result.cycle is not None
    assert result.cycle.id == cycle.id
