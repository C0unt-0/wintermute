"""Tests for api/routers/db_endpoints.py — database-backed API endpoints.

Uses FastAPI TestClient with dependency overrides so no real DB engine
is required.  Each test class provides its own mock session or exercises
the 503 path where the DB is unavailable.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Stub heavy dependencies that api.main imports at module level so the test
# suite does not need Celery, Redis, or r2pipe installed.
# ---------------------------------------------------------------------------
_worker_stub = ModuleType("src.wintermute.engine.worker")
_worker_stub.analyze_binary_task = MagicMock()  # type: ignore[attr-defined]
_worker_stub.celery_app = MagicMock()  # type: ignore[attr-defined]

# Ensure parent packages exist in sys.modules
for _pkg in ("src", "src.wintermute", "src.wintermute.engine"):
    sys.modules.setdefault(_pkg, ModuleType(_pkg))
sys.modules.setdefault("src.wintermute.engine.worker", _worker_stub)
sys.modules.setdefault("src.wintermute.data.extractor", ModuleType("src.wintermute.data.extractor"))

# Also stub celery.result which is imported in main.py
sys.modules.setdefault("celery", MagicMock())
sys.modules.setdefault("celery.result", MagicMock())

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine, event  # noqa: E402
from sqlalchemy.orm import Session, sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from wintermute.db.models import Base, Model, Sample, ScanResult  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine():
    """Create a fresh in-memory SQLite engine with tables.

    Uses StaticPool and check_same_thread=False so the same in-memory
    database is accessible from the TestClient's background thread.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    def _pragmas(dbapi_conn, _rec):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON;")
        cur.close()

    event.listen(engine, "connect", _pragmas)
    Base.metadata.create_all(engine)
    return engine


def _session_factory(engine):
    return sessionmaker(bind=engine)


def _seed_sample(session: Session, sha256: str = "aabb" * 16, **kwargs) -> Sample:
    defaults = dict(
        sha256=sha256,
        family="Ramnit",
        label=1,
        source="bazaar",
        opcode_count=500,
        file_type="PE32",
        file_size_bytes=102400,
    )
    defaults.update(kwargs)
    sample = Sample(**defaults)
    session.add(sample)
    session.flush()
    return sample


def _seed_scan(session: Session, sha256: str = "aabb" * 16, **kwargs) -> ScanResult:
    defaults = dict(
        sha256=sha256,
        predicted_family="Ramnit",
        predicted_label=1,
        confidence=0.95,
        probabilities={"Ramnit": 0.95, "benign": 0.05},
        model_version="3.0.0",
    )
    defaults.update(kwargs)
    scan = ScanResult(**defaults)
    session.add(scan)
    session.flush()
    return scan


def _seed_model(session: Session, **kwargs) -> Model:
    defaults = dict(
        version="3.0.0",
        weights_path="/models/v3.safetensors",
        manifest_path="/models/v3.json",
        config={"lr": 3e-4},
        vocab_size=5000,
        num_classes=2,
        dims=128,
        status="active",
        best_val_macro_f1=0.92,
    )
    defaults.update(kwargs)
    model = Model(**defaults)
    session.add(model)
    session.flush()
    return model


# ---------------------------------------------------------------------------
# Fixture: TestClient with a live in-memory DB session
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_client():
    """TestClient backed by an in-memory SQLite database."""
    engine = _make_engine()
    SessionLocal = _session_factory(engine)

    def override_get_db():
        session = SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # Import app *after* building overrides to avoid Celery/Redis side effects
    from api.dependencies import get_db
    from api.main import app

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    yield client, SessionLocal
    app.dependency_overrides.clear()
    engine.dispose()


@pytest.fixture()
def no_db_client():
    """TestClient where get_db always raises RuntimeError (DB unavailable)."""
    from api.dependencies import get_db
    from api.main import app

    def override_no_db():
        from fastapi import HTTPException

        raise HTTPException(
            status_code=503,
            detail="Database not available. The DB engine has not been initialised.",
        )

    app.dependency_overrides[get_db] = override_no_db
    client = TestClient(app, raise_server_exceptions=False)
    yield client
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# GET /api/v1/stats
# ---------------------------------------------------------------------------


class TestStatsEndpoint:
    def test_empty_db(self, db_client):
        client, _ = db_client
        resp = client.get("/api/v1/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_samples"] == 0
        assert data["total_scans"] == 0
        assert data["total_models"] == 0
        assert data["families"] == {}

    def test_with_data(self, db_client):
        client, SessionLocal = db_client
        session = SessionLocal()
        _seed_sample(session, sha256="aa" * 32, family="Ramnit")
        _seed_sample(session, sha256="bb" * 32, family="Kelihos")
        _seed_scan(session, sha256="aa" * 32)
        _seed_model(session)
        session.commit()
        session.close()

        resp = client.get("/api/v1/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_samples"] == 2
        assert data["total_scans"] == 1
        assert data["total_models"] == 1
        assert data["families"]["Ramnit"] == 1
        assert data["families"]["Kelihos"] == 1

    def test_db_unavailable(self, no_db_client):
        resp = no_db_client.get("/api/v1/stats")
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# GET /api/v1/samples/{sha256}
# ---------------------------------------------------------------------------


class TestSampleEndpoint:
    def test_found(self, db_client):
        client, SessionLocal = db_client
        sha = "cc" * 32
        session = SessionLocal()
        _seed_sample(session, sha256=sha, family="Vundo")
        session.commit()
        session.close()

        resp = client.get(f"/api/v1/samples/{sha}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["sha256"] == sha
        assert data["family"] == "Vundo"

    def test_not_found(self, db_client):
        client, _ = db_client
        resp = client.get("/api/v1/samples/" + "00" * 32)
        assert resp.status_code == 404

    def test_db_unavailable(self, no_db_client):
        resp = no_db_client.get("/api/v1/samples/" + "00" * 32)
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# GET /api/v1/scans
# ---------------------------------------------------------------------------


class TestScansEndpoint:
    def test_empty(self, db_client):
        client, _ = db_client
        resp = client.get("/api/v1/scans")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_recent(self, db_client):
        client, SessionLocal = db_client
        session = SessionLocal()
        sha = "dd" * 32
        _seed_sample(session, sha256=sha)
        _seed_scan(session, sha256=sha, confidence=0.90)
        _seed_scan(session, sha256=sha, confidence=0.85)
        session.commit()
        session.close()

        resp = client.get("/api/v1/scans?limit=10")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    def test_filter_by_sha256(self, db_client):
        client, SessionLocal = db_client
        session = SessionLocal()
        sha_a = "ee" * 32
        sha_b = "ff" * 32
        _seed_sample(session, sha256=sha_a)
        _seed_sample(session, sha256=sha_b)
        _seed_scan(session, sha256=sha_a)
        _seed_scan(session, sha256=sha_b)
        session.commit()
        session.close()

        resp = client.get(f"/api/v1/scans?sha256={sha_a}")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["sha256"] == sha_a

    def test_uncertain_flag(self, db_client):
        client, SessionLocal = db_client
        session = SessionLocal()
        sha = "11" * 32
        _seed_sample(session, sha256=sha)
        _seed_scan(session, sha256=sha, confidence=0.30)
        _seed_scan(session, sha256=sha, confidence=0.95)
        session.commit()
        session.close()

        resp = client.get("/api/v1/scans?uncertain=true")
        assert resp.status_code == 200
        data = resp.json()
        # Only scans with confidence < 0.6 (the default threshold)
        assert len(data) == 1
        assert data[0]["confidence"] < 0.6

    def test_db_unavailable(self, no_db_client):
        resp = no_db_client.get("/api/v1/scans")
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# GET /api/v1/similar/{sha256}
# ---------------------------------------------------------------------------


class TestSimilarEndpoint:
    def test_sample_not_found(self, db_client):
        client, _ = db_client
        resp = client.get("/api/v1/similar/" + "00" * 32)
        assert resp.status_code == 404

    def test_no_embedding(self, db_client):
        client, SessionLocal = db_client
        session = SessionLocal()
        sha = "22" * 32
        _seed_sample(session, sha256=sha)
        session.commit()
        session.close()

        resp = client.get(f"/api/v1/similar/{sha}")
        assert resp.status_code == 404
        assert "no embedding" in resp.json()["detail"].lower()

    def test_with_embedding(self, db_client):
        """Seed two samples with embeddings and verify similarity search."""
        import struct

        client, SessionLocal = db_client
        session = SessionLocal()
        sha_a = "33" * 32
        sha_b = "44" * 32

        vec_a = [1.0, 0.0, 0.0, 0.0]
        vec_b = [0.9, 0.1, 0.0, 0.0]

        _seed_sample(
            session,
            sha256=sha_a,
            embedding=struct.pack(f"{len(vec_a)}f", *vec_a),
        )
        _seed_sample(
            session,
            sha256=sha_b,
            family="Kelihos",
            embedding=struct.pack(f"{len(vec_b)}f", *vec_b),
        )
        session.commit()
        session.close()

        resp = client.get(f"/api/v1/similar/{sha_a}?k=5")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1
        assert data[0]["sha256"] == sha_b

    def test_db_unavailable(self, no_db_client):
        resp = no_db_client.get("/api/v1/similar/" + "00" * 32)
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# GET /api/v1/models
# ---------------------------------------------------------------------------


class TestModelsEndpoint:
    def test_empty(self, db_client):
        client, _ = db_client
        resp = client.get("/api/v1/models")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_with_models(self, db_client):
        client, SessionLocal = db_client
        session = SessionLocal()
        _seed_model(session, version="2.0.0", status="retired")
        _seed_model(session, version="3.0.0", status="active")
        session.commit()
        session.close()

        resp = client.get("/api/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        # Each entry should have expected fields
        for entry in data:
            assert "id" in entry
            assert "version" in entry
            assert "architecture" in entry
            assert "status" in entry

    def test_db_unavailable(self, no_db_client):
        resp = no_db_client.get("/api/v1/models")
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Schema round-trip tests
# ---------------------------------------------------------------------------


class TestNewSchemas:
    def test_sample_response_round_trip(self):
        from api.schemas import SampleResponse

        obj = SampleResponse(sha256="ab" * 32, family="Ramnit", label=1, source="bazaar")
        restored = SampleResponse.model_validate_json(obj.model_dump_json())
        assert restored == obj

    def test_scan_history_item_round_trip(self):
        from api.schemas import ScanHistoryItem

        obj = ScanHistoryItem(
            id="abc",
            sha256="ab" * 32,
            predicted_family="Ramnit",
            predicted_label=1,
            confidence=0.95,
            model_version="3.0.0",
        )
        restored = ScanHistoryItem.model_validate_json(obj.model_dump_json())
        assert restored == obj

    def test_model_response_round_trip(self):
        from api.schemas import ModelResponse

        obj = ModelResponse(
            id="xyz", version="3.0.0", architecture="WintermuteMalwareDetector", status="active"
        )
        restored = ModelResponse.model_validate_json(obj.model_dump_json())
        assert restored == obj

    def test_stats_response_defaults(self):
        from api.schemas import StatsResponse

        obj = StatsResponse()
        assert obj.total_samples == 0
        assert obj.total_scans == 0
        assert obj.total_models == 0
        assert obj.families == {}

    def test_similar_sample_response_round_trip(self):
        from api.schemas import SimilarSampleResponse

        obj = SimilarSampleResponse(sha256="ab" * 32, family="Ramnit", distance=0.05)
        restored = SimilarSampleResponse.model_validate_json(obj.model_dump_json())
        assert restored == obj
