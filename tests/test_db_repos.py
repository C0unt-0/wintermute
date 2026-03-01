"""Tests for wintermute.db.repos — SampleRepo, ScanRepo, ModelRepo, AdversarialRepo, EmbeddingRepo."""

from __future__ import annotations

import struct
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from wintermute.db.models import Base, Sample, ScanResult, TrainingRun
from wintermute.db.repos.samples import SampleRepo
from wintermute.db.repos.scans import ScanRepo
from wintermute.db.repos.models_repo import ModelRepo
from wintermute.db.repos.adversarial import AdversarialRepo
from wintermute.db.repos.embeddings import EmbeddingRepo


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


# ==================================================================
# ModelRepo tests
# ==================================================================


class TestModelRepo:
    def _register(self, repo: ModelRepo, version: str, **overrides):
        """Helper to register a model with sensible defaults."""
        defaults = {
            "version": version,
            "weights_path": f"/models/{version}.safetensors",
            "manifest_path": f"/models/{version}_manifest.json",
            "config": {"hidden_dim": 256},
            "metrics": {"best_val_macro_f1": 0.85, "best_val_accuracy": 0.90},
            "vocab_size": 5000,
            "num_classes": 2,
            "dims": 256,
        }
        defaults.update(overrides)
        return repo.register(**defaults)

    def test_register(self, db_session: Session):
        repo = ModelRepo(db_session)
        model = self._register(
            repo,
            "v1.0.0",
            metrics={
                "best_val_macro_f1": 0.88,
                "best_val_accuracy": 0.92,
                "best_val_auc_roc": 0.95,
            },
        )
        assert model.id is not None
        assert model.version == "v1.0.0"
        assert model.status == "staged"
        assert model.best_val_macro_f1 == 0.88
        assert model.best_val_accuracy == 0.92
        assert model.best_val_auc_roc == 0.95
        assert model.vocab_size == 5000
        assert model.num_classes == 2
        assert model.dims == 256

    def test_promote_retires_previous(self, db_session: Session):
        repo = ModelRepo(db_session)
        m1 = self._register(repo, "v1.0.0")
        repo.promote(m1.id)
        assert m1.status == "active"
        assert m1.promoted_at is not None

        m2 = self._register(repo, "v2.0.0")
        repo.promote(m2.id)
        assert m2.status == "active"
        # m1 should now be retired
        db_session.refresh(m1)
        assert m1.status == "retired"
        assert m1.retired_at is not None

    def test_retire(self, db_session: Session):
        repo = ModelRepo(db_session)
        m = self._register(repo, "v1.0.0")
        repo.retire(m.id)
        db_session.refresh(m)
        assert m.status == "retired"
        assert m.retired_at is not None

    def test_active(self, db_session: Session):
        repo = ModelRepo(db_session)
        m = self._register(repo, "v1.0.0")
        repo.promote(m.id)
        active = repo.active()
        assert active is not None
        assert active.id == m.id
        assert active.status == "active"

    def test_active_returns_none_when_none(self, db_session: Session):
        repo = ModelRepo(db_session)
        assert repo.active() is None

    def test_history(self, db_session: Session):
        repo = ModelRepo(db_session)
        for i in range(5):
            self._register(repo, f"v{i}.0.0")

        history = repo.history(limit=3)
        assert len(history) == 3
        # Most recent first
        assert history[0].version == "v4.0.0"

    def test_compare(self, db_session: Session):
        repo = ModelRepo(db_session)
        m1 = self._register(
            repo,
            "v1.0.0",
            metrics={"best_val_macro_f1": 0.80, "best_val_accuracy": 0.85},
        )
        m2 = self._register(
            repo,
            "v2.0.0",
            metrics={"best_val_macro_f1": 0.90, "best_val_accuracy": 0.92},
        )
        result = repo.compare(m1.id, m2.id)
        assert result["model_a"]["id"] == m1.id
        assert result["model_a"]["version"] == "v1.0.0"
        assert result["model_a"]["best_val_macro_f1"] == 0.80
        assert result["model_b"]["id"] == m2.id
        assert result["model_b"]["version"] == "v2.0.0"
        assert result["model_b"]["best_val_macro_f1"] == 0.90

    def test_promote_not_found(self, db_session: Session):
        repo = ModelRepo(db_session)
        with pytest.raises(ValueError):
            repo.promote("nonexistent-id")

    def test_retire_not_found(self, db_session: Session):
        repo = ModelRepo(db_session)
        with pytest.raises(ValueError):
            repo.retire("nonexistent-id")

    def test_compare_not_found(self, db_session: Session):
        repo = ModelRepo(db_session)
        with pytest.raises(ValueError):
            repo.compare("nonexistent-a", "nonexistent-b")


# ==================================================================
# AdversarialRepo tests
# ==================================================================


class TestAdversarialRepo:
    def _create_sample(self, db_session: Session, sha256: str, family: str = "Emotet"):
        """Helper to create a sample prerequisite."""
        sample_repo = SampleRepo(db_session)
        return sample_repo.upsert(sha256=sha256, family=family, label=1, source="test")

    def test_start_and_complete_cycle(self, db_session: Session):
        repo = AdversarialRepo(db_session)
        cycle = repo.start_cycle(cycle_number=1)
        assert cycle.id is not None
        assert cycle.started_at is not None
        assert cycle.completed_at is None

        stats = {
            "episodes_played": 100,
            "total_evasions": 30,
            "evasion_rate": 0.30,
            "mean_confidence_drop": -0.25,
            "vault_samples_used": 20,
            "defender_f1_before": 0.90,
            "defender_f1_after": 0.93,
        }
        repo.complete_cycle(cycle.id, stats)
        db_session.refresh(cycle)
        assert cycle.completed_at is not None
        assert cycle.episodes_played == 100
        assert cycle.total_evasions == 30
        assert cycle.evasion_rate == 0.30
        assert cycle.mean_confidence_drop == -0.25
        assert cycle.vault_samples_used == 20
        assert cycle.defender_f1_before == 0.90
        assert cycle.defender_f1_after == 0.93
        assert cycle.retrained is True

    def test_store_variant(self, db_session: Session):
        repo = AdversarialRepo(db_session)
        self._create_sample(db_session, "a" * 64)
        cycle = repo.start_cycle(cycle_number=1)

        variant = repo.store_variant(
            parent_sha256="a" * 64,
            cycle_id=cycle.id,
            mutated_tokens=[10, 20, 30],
            mutations=[{"type": "swap", "pos": 10}, {"type": "insert", "pos": 20}],
            confidence_before=0.95,
            confidence_after=0.30,
            modification_pct=5.0,
        )
        assert variant.id is not None
        assert variant.mutation_count == 2
        assert abs(variant.confidence_delta - 0.65) < 1e-6
        assert variant.achieved_evasion is True
        assert variant.modification_pct == 5.0
        assert variant.used_in_retraining is False

    def test_store_variant_no_evasion(self, db_session: Session):
        repo = AdversarialRepo(db_session)
        self._create_sample(db_session, "a" * 64)
        cycle = repo.start_cycle(cycle_number=1)

        variant = repo.store_variant(
            parent_sha256="a" * 64,
            cycle_id=cycle.id,
            mutated_tokens=[10],
            mutations=[{"type": "swap", "pos": 10}],
            confidence_before=0.95,
            confidence_after=0.70,
            modification_pct=1.0,
        )
        assert variant.achieved_evasion is False

    def test_get_vault_filters(self, db_session: Session):
        repo = AdversarialRepo(db_session)
        self._create_sample(db_session, "a" * 64)
        cycle = repo.start_cycle(cycle_number=1)

        # Evasive variant
        repo.store_variant(
            parent_sha256="a" * 64,
            cycle_id=cycle.id,
            mutated_tokens=[10],
            mutations=[{"type": "swap"}],
            confidence_before=0.95,
            confidence_after=0.30,
            modification_pct=3.0,
        )
        # Non-evasive variant
        repo.store_variant(
            parent_sha256="a" * 64,
            cycle_id=cycle.id,
            mutated_tokens=[20],
            mutations=[{"type": "insert"}],
            confidence_before=0.95,
            confidence_after=0.80,
            modification_pct=1.0,
        )

        # Default: evasion_only=True, unused_only=True
        vault = repo.get_vault()
        assert len(vault) == 1
        assert vault[0].achieved_evasion is True

        # Get all (including non-evasive)
        vault_all = repo.get_vault(evasion_only=False)
        assert len(vault_all) == 2

    def test_mark_retrained(self, db_session: Session):
        repo = AdversarialRepo(db_session)
        self._create_sample(db_session, "a" * 64)
        cycle = repo.start_cycle(cycle_number=1)

        v = repo.store_variant(
            parent_sha256="a" * 64,
            cycle_id=cycle.id,
            mutated_tokens=[10],
            mutations=[{"type": "swap"}],
            confidence_before=0.95,
            confidence_after=0.30,
            modification_pct=3.0,
        )

        # Create a real TrainingRun to satisfy the FK constraint
        tr = TrainingRun()
        db_session.add(tr)
        db_session.flush()

        count = repo.mark_retrained([v.id], training_run_id=tr.id)
        assert count == 1

        db_session.refresh(v)
        assert v.used_in_retraining is True
        assert v.retraining_run_id == tr.id

        # Should no longer appear in vault (unused_only=True)
        vault = repo.get_vault()
        assert len(vault) == 0

    def test_vulnerability_report(self, db_session: Session):
        repo = AdversarialRepo(db_session)
        self._create_sample(db_session, "a" * 64, family="Emotet")
        self._create_sample(db_session, "b" * 64, family="AgentTesla")
        cycle = repo.start_cycle(cycle_number=1)

        # Two Emotet variants: one evasive, one not
        repo.store_variant(
            parent_sha256="a" * 64,
            cycle_id=cycle.id,
            mutated_tokens=[10],
            mutations=[{"type": "swap"}],
            confidence_before=0.95,
            confidence_after=0.30,
            modification_pct=3.0,
        )
        repo.store_variant(
            parent_sha256="a" * 64,
            cycle_id=cycle.id,
            mutated_tokens=[20],
            mutations=[{"type": "insert"}],
            confidence_before=0.90,
            confidence_after=0.80,
            modification_pct=1.0,
        )
        # One AgentTesla variant: evasive
        repo.store_variant(
            parent_sha256="b" * 64,
            cycle_id=cycle.id,
            mutated_tokens=[30],
            mutations=[{"type": "nop"}],
            confidence_before=0.85,
            confidence_after=0.40,
            modification_pct=2.0,
        )

        report = repo.vulnerability_report()
        assert len(report) == 2

        report_by_family = {r["family"]: r for r in report}
        emotet = report_by_family["Emotet"]
        assert emotet["total_attacks"] == 2
        assert emotet["evasions"] == 1
        assert abs(emotet["evasion_rate"] - 0.5) < 1e-6

        agent = report_by_family["AgentTesla"]
        assert agent["total_attacks"] == 1
        assert agent["evasions"] == 1
        assert abs(agent["evasion_rate"] - 1.0) < 1e-6

    def test_complete_cycle_not_found(self, db_session: Session):
        repo = AdversarialRepo(db_session)
        with pytest.raises(ValueError):
            repo.complete_cycle("nonexistent", {"episodes_played": 10})

    def test_mark_retrained_empty(self, db_session: Session):
        repo = AdversarialRepo(db_session)
        count = repo.mark_retrained([], training_run_id="fake")
        assert count == 0


# ==================================================================
# EmbeddingRepo tests
# ==================================================================


class TestEmbeddingRepo:
    """Tests for EmbeddingRepo.

    All tests use the pure-Python cosine distance fallback since
    sqlite-vec is not expected to be available in CI.
    """

    def _add_samples_with_embeddings(self, db_session: Session, n: int = 5) -> None:
        """Helper: create *n* samples and set embeddings on all of them."""
        dim = 8  # small dimension for tests
        for i in range(n):
            vec = [float(i * 10 + j) for j in range(dim)]
            sample = Sample(
                sha256=f"{chr(97 + i)}" * 64,
                family="Emotet" if i < 3 else "AgentTesla",
                label=1 if i < 3 else 2,
                source="test",
            )
            sample.embedding = struct.pack(f"{dim}f", *vec)
            db_session.add(sample)
        db_session.flush()

    # -- coverage_stats -------------------------------------------------

    def test_coverage_stats_no_embeddings(self, db_session: Session):
        """coverage_stats works without vector extension."""

        for i in range(3):
            db_session.add(
                Sample(
                    sha256=f"{chr(97 + i)}" * 64,
                    family="test",
                    label=0,
                    source="test",
                )
            )
        db_session.flush()

        repo = EmbeddingRepo(db_session)
        stats = repo.coverage_stats()
        assert stats["total_samples"] == 3
        assert stats["with_embedding"] == 0
        assert stats["without_embedding"] == 3
        assert stats["pct_covered"] == 0.0

    def test_coverage_stats_with_embeddings(self, db_session: Session):

        self._add_samples_with_embeddings(db_session, n=5)
        repo = EmbeddingRepo(db_session)
        stats = repo.coverage_stats()
        assert stats["total_samples"] == 5
        assert stats["with_embedding"] == 5
        assert stats["pct_covered"] == 100.0

    def test_coverage_stats_partial(self, db_session: Session):

        self._add_samples_with_embeddings(db_session, n=3)
        # Add two samples *without* embeddings
        for c in "xy":
            db_session.add(Sample(sha256=c * 64, family="test", label=0, source="test"))
        db_session.flush()

        repo = EmbeddingRepo(db_session)
        stats = repo.coverage_stats()
        assert stats["total_samples"] == 5
        assert stats["with_embedding"] == 3
        assert stats["without_embedding"] == 2
        assert abs(stats["pct_covered"] - 60.0) < 1e-6

    def test_coverage_stats_empty_db(self, db_session: Session):

        repo = EmbeddingRepo(db_session)
        stats = repo.coverage_stats()
        assert stats["total_samples"] == 0
        assert stats["pct_covered"] == 0.0

    # -- find_nearest ---------------------------------------------------

    def test_find_nearest(self, db_session: Session):

        self._add_samples_with_embeddings(db_session, n=5)
        repo = EmbeddingRepo(db_session)

        # Query vector close to sample 0 (values [0..7])
        query = [float(j) for j in range(8)]
        results = repo.find_nearest(query, k=3)

        assert len(results) <= 3
        assert len(results) > 0
        # Results should be dicts with the expected keys
        assert "sha256" in results[0]
        assert "family" in results[0]
        assert "label" in results[0]
        assert "distance" in results[0]
        # First result should be the nearest (smallest distance)
        assert results[0]["distance"] <= results[-1]["distance"]

    def test_find_nearest_ordering(self, db_session: Session):
        """Results are sorted by ascending distance."""

        self._add_samples_with_embeddings(db_session, n=5)
        repo = EmbeddingRepo(db_session)

        query = [float(j) for j in range(8)]
        results = repo.find_nearest(query, k=5)
        distances = [r["distance"] for r in results]
        assert distances == sorted(distances)

    def test_find_nearest_with_family_filter(self, db_session: Session):

        self._add_samples_with_embeddings(db_session, n=5)
        repo = EmbeddingRepo(db_session)

        query = [float(j) for j in range(8)]
        results = repo.find_nearest(query, k=10, family="Emotet")
        # Should only return Emotet samples (at most 3 based on helper)
        assert len(results) <= 3
        for r in results:
            assert r["family"] == "Emotet"

    def test_find_nearest_with_max_distance(self, db_session: Session):

        self._add_samples_with_embeddings(db_session, n=5)
        repo = EmbeddingRepo(db_session)

        query = [float(j) for j in range(8)]
        # Use a very small max_distance to filter aggressively
        results = repo.find_nearest(query, k=10, max_distance=0.001)
        for r in results:
            assert r["distance"] <= 0.001

    def test_find_nearest_empty_db(self, db_session: Session):

        repo = EmbeddingRepo(db_session)
        results = repo.find_nearest([0.0] * 8, k=5)
        assert results == []

    def test_find_nearest_no_embeddings(self, db_session: Session):
        """Samples exist but none have embeddings."""

        db_session.add(Sample(sha256="a" * 64, family="test", label=0, source="test"))
        db_session.flush()

        repo = EmbeddingRepo(db_session)
        results = repo.find_nearest([0.0] * 8, k=5)
        assert results == []

    # -- find_nearest_with_scans ----------------------------------------

    def test_find_nearest_with_scans(self, db_session: Session):

        self._add_samples_with_embeddings(db_session, n=3)

        # Add a scan result for the first sample
        scan = ScanResult(
            sha256="a" * 64,
            predicted_family="Emotet",
            predicted_label=1,
            confidence=0.95,
            probabilities={},
            model_version="v1.0.0",
        )
        db_session.add(scan)
        db_session.flush()

        repo = EmbeddingRepo(db_session)
        query = [float(j) for j in range(8)]
        results = repo.find_nearest_with_scans(query, k=3)

        assert len(results) > 0
        # Every result should have the enriched keys
        for r in results:
            assert "sha256" in r
            assert "family" in r
            assert "distance" in r
            assert "last_scan_confidence" in r
            assert "last_scan_date" in r

        # The sample with a scan should have non-null scan fields
        scanned = [r for r in results if r["sha256"] == "a" * 64]
        if scanned:
            assert scanned[0]["last_scan_confidence"] == 0.95
            assert scanned[0]["last_scan_date"] is not None

    def test_find_nearest_with_scans_no_scans(self, db_session: Session):
        """Neighbours with no scan results should have None for scan fields."""

        self._add_samples_with_embeddings(db_session, n=2)
        repo = EmbeddingRepo(db_session)

        query = [float(j) for j in range(8)]
        results = repo.find_nearest_with_scans(query, k=2)
        for r in results:
            assert r["last_scan_confidence"] is None
            assert r["last_scan_date"] is None

    def test_find_nearest_with_scans_empty(self, db_session: Session):

        repo = EmbeddingRepo(db_session)
        results = repo.find_nearest_with_scans([0.0] * 8, k=5)
        assert results == []

    # -- cluster_family -------------------------------------------------

    def test_cluster_family(self, db_session: Session):

        self._add_samples_with_embeddings(db_session, n=5)
        repo = EmbeddingRepo(db_session)

        results = repo.cluster_family("Emotet", k=3)
        assert len(results) <= 3
        assert len(results) > 0
        # Should return dicts with sha256, family, label, distance
        for r in results:
            assert "sha256" in r
            assert "family" in r
            assert "label" in r
            assert "distance" in r

    def test_cluster_family_ordering(self, db_session: Session):
        """Results are sorted by ascending centroid distance."""

        self._add_samples_with_embeddings(db_session, n=5)
        repo = EmbeddingRepo(db_session)

        results = repo.cluster_family("Emotet", k=3)
        distances = [r["distance"] for r in results]
        assert distances == sorted(distances)

    def test_cluster_family_empty(self, db_session: Session):
        """No samples for the requested family."""

        self._add_samples_with_embeddings(db_session, n=5)
        repo = EmbeddingRepo(db_session)
        results = repo.cluster_family("NonExistentFamily", k=3)
        assert results == []

    def test_cluster_family_no_embeddings(self, db_session: Session):
        """Family exists but has no embeddings."""

        db_session.add(Sample(sha256="a" * 64, family="Emotet", label=1, source="test"))
        db_session.flush()

        repo = EmbeddingRepo(db_session)
        results = repo.cluster_family("Emotet", k=3)
        assert results == []
