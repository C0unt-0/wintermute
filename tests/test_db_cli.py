"""Tests for wintermute db CLI subcommands."""

from __future__ import annotations

import struct

import pytest
from typer.testing import CliRunner

from wintermute.cli import app
from wintermute.db.engine import create_db_engine, get_session, init_db
from wintermute.db.models import Sample
from wintermute.db.repos.adversarial import AdversarialRepo
from wintermute.db.repos.models_repo import ModelRepo
from wintermute.db.repos.samples import SampleRepo
from wintermute.db.repos.scans import ScanRepo

runner = CliRunner()


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def db_url(tmp_path):
    """Return a file-based SQLite URL in a temp directory."""
    return f"sqlite:///{tmp_path / 'test.db'}"


@pytest.fixture(autouse=True)
def _reset_engine():
    """Reset the module-level engine globals between tests."""
    import wintermute.db.engine as eng_mod

    yield
    # Dispose engine if one was created during the test
    if eng_mod._engine is not None:
        eng_mod._engine.dispose()
    eng_mod._engine = None
    eng_mod._SessionFactory = None


@pytest.fixture()
def setup_db(db_url, monkeypatch):
    """Create the engine and tables, returning the db_url for CLI usage."""
    monkeypatch.setenv("WINTERMUTE_DATABASE_URL", db_url)
    engine = create_db_engine()
    init_db(engine)
    return db_url


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _seed_samples():
    """Insert a handful of samples via the repo."""
    with get_session() as session:
        repo = SampleRepo(session)
        repo.upsert(
            sha256="a" * 64,
            family="Emotet",
            label=1,
            source="bazaar",
            opcode_count=500,
        )
        repo.upsert(
            sha256="b" * 64,
            family="Emotet",
            label=1,
            source="bazaar",
            opcode_count=200,
        )
        repo.upsert(
            sha256="c" * 64,
            family="AgentTesla",
            label=2,
            source="virusshare",
            opcode_count=800,
        )


def _seed_scans():
    """Insert some scan results."""
    with get_session() as session:
        repo = ScanRepo(session)
        repo.record(
            sha256="a" * 64,
            predicted_family="Emotet",
            predicted_label=1,
            confidence=0.95,
            probabilities={"0": 0.05, "1": 0.95},
            model_version="v1.0.0",
        )
        repo.record(
            sha256="b" * 64,
            predicted_family="Emotet",
            predicted_label=1,
            confidence=0.55,
            probabilities={"0": 0.45, "1": 0.55},
            model_version="v1.0.0",
        )


def _seed_models():
    """Register a couple of model versions and return their IDs."""
    ids = []
    with get_session() as session:
        repo = ModelRepo(session)
        m1 = repo.register(
            version="v1.0.0",
            weights_path="/models/v1.safetensors",
            manifest_path="/models/v1_manifest.json",
            config={"hidden_dim": 256},
            metrics={"best_val_macro_f1": 0.85},
            vocab_size=5000,
            num_classes=2,
            dims=256,
        )
        m2 = repo.register(
            version="v2.0.0",
            weights_path="/models/v2.safetensors",
            manifest_path="/models/v2_manifest.json",
            config={"hidden_dim": 512},
            metrics={"best_val_macro_f1": 0.92},
            vocab_size=5000,
            num_classes=2,
            dims=512,
        )
        ids.extend([m1.id, m2.id])
    return ids


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestDbInit:
    def test_db_init(self, db_url, monkeypatch):
        monkeypatch.setenv("WINTERMUTE_DATABASE_URL", db_url)
        result = runner.invoke(app, ["db", "init"])
        assert result.exit_code == 0
        assert "Database tables created successfully" in result.output


class TestDbStats:
    def test_stats_empty(self, setup_db, monkeypatch):
        result = runner.invoke(app, ["db", "stats"])
        assert result.exit_code == 0
        assert "Samples: 0" in result.output
        assert "Scans: 0" in result.output

    def test_stats_with_data(self, setup_db, monkeypatch):
        _seed_samples()
        _seed_scans()

        result = runner.invoke(app, ["db", "stats"])
        assert result.exit_code == 0
        assert "Samples: 3" in result.output
        assert "Emotet" in result.output
        assert "AgentTesla" in result.output
        assert "Scans: 2" in result.output
        assert "bazaar" in result.output
        assert "virusshare" in result.output


class TestDbSamples:
    def test_samples_empty(self, setup_db, monkeypatch):
        result = runner.invoke(app, ["db", "samples"])
        assert result.exit_code == 0
        assert "No samples found" in result.output

    def test_samples_all(self, setup_db, monkeypatch):
        _seed_samples()
        result = runner.invoke(app, ["db", "samples"])
        assert result.exit_code == 0
        assert "Found 3 sample(s)" in result.output

    def test_samples_filter_family(self, setup_db, monkeypatch):
        _seed_samples()
        result = runner.invoke(app, ["db", "samples", "--family", "Emotet"])
        assert result.exit_code == 0
        assert "Found 2 sample(s)" in result.output

    def test_samples_filter_source(self, setup_db, monkeypatch):
        _seed_samples()
        result = runner.invoke(
            app, ["db", "samples", "--source", "virusshare"]
        )
        assert result.exit_code == 0
        assert "Found 1 sample(s)" in result.output

    def test_samples_filter_min_opcodes(self, setup_db, monkeypatch):
        _seed_samples()
        result = runner.invoke(
            app, ["db", "samples", "--min-opcodes", "600"]
        )
        assert result.exit_code == 0
        assert "Found 1 sample(s)" in result.output

    def test_samples_limit(self, setup_db, monkeypatch):
        _seed_samples()
        result = runner.invoke(app, ["db", "samples", "--limit", "1"])
        assert result.exit_code == 0
        assert "Found 1 sample(s)" in result.output


class TestDbScans:
    def test_scans_empty(self, setup_db, monkeypatch):
        result = runner.invoke(app, ["db", "scans"])
        assert result.exit_code == 0
        assert "No scan results found" in result.output

    def test_scans_recent(self, setup_db, monkeypatch):
        _seed_samples()
        _seed_scans()
        result = runner.invoke(app, ["db", "scans", "--recent", "5"])
        assert result.exit_code == 0
        assert "Found 2 scan(s)" in result.output

    def test_scans_by_sha256(self, setup_db, monkeypatch):
        _seed_samples()
        _seed_scans()
        result = runner.invoke(app, ["db", "scans", "--sha256", "a" * 64])
        assert result.exit_code == 0
        assert "Found 1 scan(s)" in result.output

    def test_scans_uncertain(self, setup_db, monkeypatch):
        _seed_samples()
        _seed_scans()
        result = runner.invoke(app, ["db", "scans", "--uncertain", "0.6"])
        assert result.exit_code == 0
        assert "Found 1 scan(s)" in result.output
        assert "conf=0.55" in result.output


class TestDbModels:
    def test_models_empty(self, setup_db, monkeypatch):
        result = runner.invoke(app, ["db", "models"])
        assert result.exit_code == 0
        assert "No models registered" in result.output

    def test_models_list(self, setup_db, monkeypatch):
        _seed_models()
        result = runner.invoke(app, ["db", "models"])
        assert result.exit_code == 0
        assert "v1.0.0" in result.output
        assert "v2.0.0" in result.output
        assert "[STAGED]" in result.output

    def test_models_promote(self, setup_db, monkeypatch):
        model_ids = _seed_models()
        result = runner.invoke(
            app, ["db", "models", "--promote", model_ids[0]]
        )
        assert result.exit_code == 0
        assert "promoted to active" in result.output

        # Verify it's now active
        result = runner.invoke(app, ["db", "models"])
        assert "[ACTIVE]" in result.output


class TestDbVault:
    def test_vault_empty(self, setup_db, monkeypatch):
        result = runner.invoke(app, ["db", "vault"])
        assert result.exit_code == 0
        assert "ADVERSARIAL VAULT" in result.output
        assert "Variants: 0" in result.output

    def test_vault_with_data(self, setup_db, monkeypatch):
        _seed_samples()
        with get_session() as session:
            repo = AdversarialRepo(session)
            cycle = repo.start_cycle(cycle_number=1)
            repo.store_variant(
                parent_sha256="a" * 64,
                cycle_id=cycle.id,
                mutated_tokens=[10, 20],
                mutations=[{"type": "swap", "pos": 10}],
                confidence_before=0.95,
                confidence_after=0.30,
                modification_pct=3.0,
            )

        result = runner.invoke(app, ["db", "vault"])
        assert result.exit_code == 0
        assert "Variants: 1" in result.output
        assert "Emotet" in result.output


class TestDbEmbed:
    def test_embed_empty(self, setup_db, monkeypatch):
        result = runner.invoke(app, ["db", "embed"])
        assert result.exit_code == 0
        assert "Embedding Coverage" in result.output
        assert "Total samples: 0" in result.output
        assert "Coverage: 0.0%" in result.output

    def test_embed_with_data(self, setup_db, monkeypatch):
        _seed_samples()
        # Add an embedding to one sample
        with get_session() as session:
            sample = session.get(Sample, "a" * 64)
            dim = 8
            vec = [float(i) for i in range(dim)]
            sample.embedding = struct.pack(f"{dim}f", *vec)
            session.flush()

        result = runner.invoke(app, ["db", "embed"])
        assert result.exit_code == 0
        assert "Total samples: 3" in result.output
        assert "With embeddings: 1" in result.output


class TestDbSimilar:
    def test_similar_sample_not_found(self, setup_db, monkeypatch):
        result = runner.invoke(app, ["db", "similar", "z" * 64])
        assert result.exit_code == 1
        assert "Sample not found" in result.output

    def test_similar_no_embedding(self, setup_db, monkeypatch):
        _seed_samples()
        result = runner.invoke(app, ["db", "similar", "a" * 64])
        assert result.exit_code == 1
        assert "no embedding" in result.output

    def test_similar_with_embedding(self, setup_db, monkeypatch):
        _seed_samples()
        dim = 8

        # Add embeddings to all samples
        with get_session() as session:
            for i, sha in enumerate(["a" * 64, "b" * 64, "c" * 64]):
                sample = session.get(Sample, sha)
                vec = [float(i * 10 + j) for j in range(dim)]
                sample.embedding = struct.pack(f"{dim}f", *vec)
            session.flush()

        result = runner.invoke(app, ["db", "similar", "a" * 64, "--k", "2"])
        assert result.exit_code == 0
        assert "similar samples" in result.output
