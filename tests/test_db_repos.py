"""Tests for wintermute.db.repos — SampleRepo and ScanRepo."""

from __future__ import annotations

import struct
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from wintermute.db.models import Base
from wintermute.db.repos.samples import SampleRepo
from wintermute.db.repos.scans import ScanRepo


# ------------------------------------------------------------------
# Shared fixture
# ------------------------------------------------------------------


@pytest.fixture()
def db_session():
    """Yield an in-memory SQLite session with all tables created."""
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()
    engine.dispose()


# ==================================================================
# SampleRepo tests
# ==================================================================


class TestSampleRepo:
    def test_upsert_and_get(self, db_session: Session):
        repo = SampleRepo(db_session)
        sample = repo.upsert(
            sha256="a" * 64,
            family="Emotet",
            label=1,
            source="synthetic",
            opcode_count=500,
        )
        assert sample.sha256 == "a" * 64

        result = repo.get("a" * 64)
        assert result is not None
        assert result.family == "Emotet"

    def test_upsert_idempotent(self, db_session: Session):
        repo = SampleRepo(db_session)
        repo.upsert(sha256="a" * 64, family="Emotet", label=1, source="s1")
        repo.upsert(sha256="a" * 64, family="Emotet", label=1, source="s2")
        result = repo.get("a" * 64)
        assert result is not None  # no duplicate key error
        assert result.source == "s2"  # second upsert updated the source

    def test_exists(self, db_session: Session):
        repo = SampleRepo(db_session)
        assert repo.exists("a" * 64) is False
        repo.upsert(sha256="a" * 64, family="Emotet", label=1, source="test")
        assert repo.exists("a" * 64) is True

    def test_find_with_filters(self, db_session: Session):
        repo = SampleRepo(db_session)
        repo.upsert(
            sha256="a" * 64,
            family="Emotet",
            label=1,
            source="s1",
            opcode_count=500,
        )
        repo.upsert(
            sha256="b" * 64,
            family="AgentTesla",
            label=2,
            source="s1",
            opcode_count=200,
        )
        repo.upsert(
            sha256="c" * 64,
            family="Emotet",
            label=1,
            source="s2",
            opcode_count=800,
        )

        results = repo.find(family="Emotet")
        assert len(results) == 2

        results = repo.find(source="s1")
        assert len(results) == 2

        results = repo.find(min_opcodes=600)
        assert len(results) == 1

    def test_find_with_label_filter(self, db_session: Session):
        repo = SampleRepo(db_session)
        repo.upsert(sha256="a" * 64, family="Emotet", label=1, source="s1")
        repo.upsert(sha256="b" * 64, family="Safe", label=0, source="s1")

        results = repo.find(label=1)
        assert len(results) == 1
        assert results[0].family == "Emotet"

    def test_find_pagination(self, db_session: Session):
        repo = SampleRepo(db_session)
        for i in range(10):
            repo.upsert(
                sha256=f"{chr(97 + i)}" * 64,
                family="test",
                label=0,
                source="bulk",
            )

        page1 = repo.find(limit=3, offset=0)
        page2 = repo.find(limit=3, offset=3)
        assert len(page1) == 3
        assert len(page2) == 3
        # Pages should not overlap
        hashes1 = {s.sha256 for s in page1}
        hashes2 = {s.sha256 for s in page2}
        assert hashes1.isdisjoint(hashes2)

    def test_count_by_family(self, db_session: Session):
        repo = SampleRepo(db_session)
        repo.upsert(sha256="a" * 64, family="Emotet", label=1, source="s1")
        repo.upsert(sha256="b" * 64, family="Emotet", label=1, source="s1")
        repo.upsert(sha256="c" * 64, family="AgentTesla", label=2, source="s1")

        counts = repo.count_by_family()
        assert counts == {"Emotet": 2, "AgentTesla": 1}

    def test_count_by_source(self, db_session: Session):
        repo = SampleRepo(db_session)
        repo.upsert(sha256="a" * 64, family="Emotet", label=1, source="bazaar")
        repo.upsert(sha256="b" * 64, family="Emotet", label=1, source="bazaar")
        repo.upsert(sha256="c" * 64, family="AgentTesla", label=2, source="virusshare")

        counts = repo.count_by_source()
        assert counts == {"bazaar": 2, "virusshare": 1}

    def test_bulk_insert(self, db_session: Session):
        repo = SampleRepo(db_session)
        samples = [
            {
                "sha256": f"{chr(97 + i)}" * 64,
                "family": "test",
                "label": 0,
                "source": "bulk",
            }
            for i in range(10)
        ]
        count = repo.bulk_insert(samples)
        assert count == 10

    def test_bulk_insert_idempotent(self, db_session: Session):
        repo = SampleRepo(db_session)
        repo.upsert(sha256="a" * 64, family="existing", label=0, source="old")

        samples = [
            {"sha256": "a" * 64, "family": "test", "label": 0, "source": "bulk"},
            {"sha256": "b" * 64, "family": "test", "label": 0, "source": "bulk"},
        ]
        count = repo.bulk_insert(samples)
        # "a"*64 already existed, only "b"*64 is new
        assert count == 1

        # Existing record should not be overwritten
        existing = repo.get("a" * 64)
        assert existing is not None
        assert existing.family == "existing"

    def test_set_embedding(self, db_session: Session):
        repo = SampleRepo(db_session)
        repo.upsert(sha256="a" * 64, family="Emotet", label=1, source="test")

        embedding = [0.1, 0.2, 0.3, 0.4]
        repo.set_embedding("a" * 64, embedding)

        sample = repo.get("a" * 64)
        assert sample is not None
        assert sample.embedding is not None
        # Verify round-trip
        n = len(embedding)
        unpacked = list(struct.unpack(f"{n}f", sample.embedding))
        assert len(unpacked) == 4
        assert abs(unpacked[0] - 0.1) < 1e-6

    def test_bulk_set_embeddings(self, db_session: Session):
        repo = SampleRepo(db_session)
        for c in "abc":
            repo.upsert(sha256=c * 64, family="test", label=0, source="test")

        pairs = [
            ("a" * 64, [1.0, 2.0]),
            ("b" * 64, [3.0, 4.0]),
            ("c" * 64, [5.0, 6.0]),
        ]
        count = repo.bulk_set_embeddings(pairs)
        assert count == 3

        sample = repo.get("a" * 64)
        assert sample is not None
        assert sample.embedding is not None

    def test_set_embedding_missing_sample(self, db_session: Session):
        repo = SampleRepo(db_session)
        with pytest.raises(ValueError):
            repo.set_embedding("z" * 64, [0.1] * 256)

    def test_bulk_insert_empty(self, db_session: Session):
        repo = SampleRepo(db_session)
        count = repo.bulk_insert([])
        assert count == 0


# ==================================================================
# ScanRepo tests
# ==================================================================


class TestScanRepo:
    def test_record_and_history(self, db_session: Session):
        repo = ScanRepo(db_session)
        scan = repo.record(
            sha256="a" * 64,
            predicted_family="Emotet",
            predicted_label=1,
            confidence=0.95,
            probabilities={"safe": 0.05, "malicious": 0.95},
            model_version="v1.0.0",
        )
        assert scan.id is not None

        history = repo.history("a" * 64)
        assert len(history) == 1
        assert history[0].predicted_family == "Emotet"

    def test_history_ordering(self, db_session: Session):
        """History should return most recent first."""
        repo = ScanRepo(db_session)
        repo.record(
            sha256="a" * 64,
            predicted_family="Emotet",
            predicted_label=1,
            confidence=0.90,
            probabilities={},
            model_version="v1.0.0",
        )
        repo.record(
            sha256="a" * 64,
            predicted_family="Emotet",
            predicted_label=1,
            confidence=0.95,
            probabilities={},
            model_version="v2.0.0",
        )

        history = repo.history("a" * 64)
        assert len(history) == 2
        # Most recent should be first
        assert history[0].model_version == "v2.0.0"

    def test_recent(self, db_session: Session):
        repo = ScanRepo(db_session)
        for i in range(5):
            repo.record(
                sha256=f"{chr(97 + i)}" * 64,
                predicted_family="Emotet",
                predicted_label=1,
                confidence=0.9,
                probabilities={},
                model_version="v1.0.0",
            )
        results = repo.recent(limit=3)
        assert len(results) == 3

    def test_recent_with_since(self, db_session: Session):
        repo = ScanRepo(db_session)
        # Create a scan
        repo.record(
            sha256="a" * 64,
            predicted_family="Emotet",
            predicted_label=1,
            confidence=0.9,
            probabilities={},
            model_version="v1.0.0",
        )

        # Query with a future "since" should return nothing
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        results = repo.recent(since=future)
        assert len(results) == 0

        # Query with a past "since" should return the scan
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        results = repo.recent(since=past)
        assert len(results) == 1

    def test_by_family(self, db_session: Session):
        repo = ScanRepo(db_session)
        repo.record(
            sha256="a" * 64,
            predicted_family="Emotet",
            predicted_label=1,
            confidence=0.95,
            probabilities={},
            model_version="v1.0.0",
        )
        repo.record(
            sha256="b" * 64,
            predicted_family="AgentTesla",
            predicted_label=2,
            confidence=0.80,
            probabilities={},
            model_version="v1.0.0",
        )

        results = repo.by_family("Emotet")
        assert len(results) == 1
        assert results[0].sha256 == "a" * 64

    def test_by_family_with_min_confidence(self, db_session: Session):
        repo = ScanRepo(db_session)
        repo.record(
            sha256="a" * 64,
            predicted_family="Emotet",
            predicted_label=1,
            confidence=0.95,
            probabilities={},
            model_version="v1.0.0",
        )
        repo.record(
            sha256="b" * 64,
            predicted_family="Emotet",
            predicted_label=1,
            confidence=0.50,
            probabilities={},
            model_version="v1.0.0",
        )

        results = repo.by_family("Emotet", min_confidence=0.8)
        assert len(results) == 1
        assert results[0].confidence == 0.95

    def test_uncertain(self, db_session: Session):
        repo = ScanRepo(db_session)
        repo.record(
            sha256="a" * 64,
            predicted_family="Emotet",
            predicted_label=1,
            confidence=0.95,
            probabilities={},
            model_version="v1.0.0",
        )
        repo.record(
            sha256="b" * 64,
            predicted_family="Unknown",
            predicted_label=0,
            confidence=0.45,
            probabilities={},
            model_version="v1.0.0",
        )
        results = repo.uncertain(threshold=0.6)
        assert len(results) == 1
        assert results[0].confidence == 0.45

    def test_stats(self, db_session: Session):
        repo = ScanRepo(db_session)
        repo.record(
            sha256="a" * 64,
            predicted_family="Emotet",
            predicted_label=1,
            confidence=0.9,
            probabilities={},
            model_version="v1.0.0",
        )
        repo.record(
            sha256="b" * 64,
            predicted_family="AgentTesla",
            predicted_label=2,
            confidence=0.8,
            probabilities={},
            model_version="v1.0.0",
        )
        stats = repo.stats()
        assert stats["total_scans"] == 2
        assert "Emotet" in stats["family_distribution"]
        assert "AgentTesla" in stats["family_distribution"]
        assert abs(stats["avg_confidence"] - 0.85) < 1e-6

    def test_stats_empty(self, db_session: Session):
        repo = ScanRepo(db_session)
        stats = repo.stats()
        assert stats["total_scans"] == 0
        assert stats["family_distribution"] == {}
        assert stats["avg_confidence"] == 0.0

    def test_stats_with_since(self, db_session: Session):
        repo = ScanRepo(db_session)
        repo.record(
            sha256="a" * 64,
            predicted_family="Emotet",
            predicted_label=1,
            confidence=0.9,
            probabilities={},
            model_version="v1.0.0",
        )

        future = datetime.now(timezone.utc) + timedelta(hours=1)
        stats = repo.stats(since=future)
        assert stats["total_scans"] == 0

    def test_record_with_kwargs(self, db_session: Session):
        """Extra kwargs like filename and execution_time_ms should be stored."""
        repo = ScanRepo(db_session)
        scan = repo.record(
            sha256="a" * 64,
            predicted_family="Emotet",
            predicted_label=1,
            confidence=0.95,
            probabilities={},
            model_version="v1.0.0",
            filename="evil.exe",
            execution_time_ms=42.5,
        )
        assert scan.filename == "evil.exe"
        assert scan.execution_time_ms == 42.5
