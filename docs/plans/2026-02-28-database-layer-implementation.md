# Database Layer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a unified PostgreSQL+SQLite database layer with vector search, replacing scattered flat-file state across ETL, scan, training, adversarial, and model registry subsystems.

**Architecture:** Direct repository injection — repos passed into constructors as optional parameters (like `MLflowTracker`). When `None`, DB writes are silently skipped. SQLAlchemy 2.0 ORM with Alembic migrations. sqlite-vec for local dev vector search, pgvector for production.

**Tech Stack:** SQLAlchemy 2.0, Alembic, sqlite-vec, pgvector, psycopg

---

## Phase A: Foundation

### Task 1: Dependencies and Configuration

**Files:**
- Modify: `pyproject.toml`
- Create: `configs/database.yaml`

**Step 1: Add dependencies to pyproject.toml**

Add a new `db` optional dependency group after the existing `adversarial` group:

```toml
db = [
    "sqlalchemy>=2.0.0",
    "alembic>=1.13.0",
    "sqlite-vec>=0.1.6",
]
```

Update the `all` group to include `db`:

```toml
all = [
    "wintermute[api,mlops,dev,adversarial,db]",
    "angr>=9.2.0",
]
```

**Step 2: Create database config**

```yaml
# configs/database.yaml
database:
  url: "sqlite:///data/wintermute.db"
  echo: false
  pool_size: 5
```

**Step 3: Install dependencies**

Run: `pip install -e ".[db,dev]"`

**Step 4: Commit**

```bash
git add pyproject.toml configs/database.yaml
git commit -m "chore: add database dependencies and config"
```

---

### Task 2: Engine — Connection Management

**Files:**
- Create: `src/wintermute/db/__init__.py`
- Create: `src/wintermute/db/engine.py`
- Test: `tests/test_db_engine.py`

**Step 1: Write the failing test**

```python
# tests/test_db_engine.py
"""Tests for database engine creation and session management."""
import pytest
from sqlalchemy import text


def test_create_sqlite_engine():
    """SQLite engine should be created with WAL mode and foreign keys."""
    from wintermute.db.engine import create_db_engine

    engine = create_db_engine("sqlite:///:memory:")
    assert engine is not None
    with engine.connect() as conn:
        result = conn.execute(text("PRAGMA journal_mode")).scalar()
        # In-memory SQLite doesn't persist WAL, but the pragma runs without error
        assert result in ("wal", "memory")
        fk = conn.execute(text("PRAGMA foreign_keys")).scalar()
        assert fk == 1


def test_get_session_contextmanager():
    """get_session should yield a working session and auto-close."""
    from wintermute.db.engine import create_db_engine, get_session, _SessionFactory

    engine = create_db_engine("sqlite:///:memory:")
    _SessionFactory.configure(bind=engine)

    with get_session() as session:
        result = session.execute(text("SELECT 1")).scalar()
        assert result == 1


def test_init_db_creates_tables():
    """init_db should create all tables without error."""
    from wintermute.db.engine import create_db_engine, init_db

    engine = create_db_engine("sqlite:///:memory:")
    init_db(engine)  # Should not raise

    with engine.connect() as conn:
        # Check that the samples table exists
        result = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='samples'")
        ).scalar()
        assert result == "samples"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_db_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'wintermute.db'`

**Step 3: Write engine.py**

```python
# src/wintermute/db/engine.py
"""engine.py — SQLAlchemy engine creation and session management."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import yaml
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger("wintermute.db")

_SessionFactory = sessionmaker()
_engine: Engine | None = None

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "database.yaml"


def _resolve_url(url: str | None = None) -> str:
    """Resolve database URL from argument, env var, or config file."""
    if url:
        return url

    env_url = os.environ.get("WINTERMUTE_DATABASE_URL")
    if env_url:
        return env_url

    if DEFAULT_CONFIG_PATH.exists():
        with open(DEFAULT_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
        db_cfg = cfg.get("database", {})
        return db_cfg.get("url", "sqlite:///data/wintermute.db")

    return "sqlite:///data/wintermute.db"


def create_db_engine(url: str | None = None, echo: bool = False, **kwargs) -> Engine:
    """Create SQLAlchemy engine with sensible defaults for SQLite or PostgreSQL."""
    global _engine

    url = _resolve_url(url)

    if url.startswith("sqlite"):
        engine = create_engine(
            url,
            echo=echo,
            connect_args={"check_same_thread": False},
        )

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_conn, _connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    else:
        engine = create_engine(
            url,
            echo=echo,
            pool_size=kwargs.get("pool_size", 20),
            max_overflow=kwargs.get("max_overflow", 10),
        )

    _SessionFactory.configure(bind=engine)
    _engine = engine
    return engine


def get_engine() -> Engine | None:
    """Return the current engine, or None if not initialized."""
    return _engine


def init_db(engine: Engine | None = None) -> None:
    """Create all tables. Safe to call multiple times."""
    from wintermute.db.models import Base

    eng = engine or _engine
    if eng is None:
        raise RuntimeError("No database engine. Call create_db_engine() first.")

    Base.metadata.create_all(eng)

    # Enable pgvector extension if PostgreSQL
    if "postgresql" in str(eng.url):
        with eng.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
        logger.info("pgvector extension enabled")

    logger.info("Database tables created/verified")


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session, auto-closing on exit."""
    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
```

Write the `__init__.py`:

```python
# src/wintermute/db/__init__.py
"""wintermute.db — Unified database layer."""

from wintermute.db.engine import create_db_engine, get_engine, get_session, init_db

__all__ = ["create_db_engine", "get_engine", "get_session", "init_db"]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_db_engine.py -v`
Expected: PASS (the `init_db` test will need Task 3's models.py — mark it `xfail` or create an empty Base for now)

Note: Task 2's `test_init_db_creates_tables` depends on `models.py` from Task 3. Implement both tasks before running the full test suite.

**Step 5: Commit**

```bash
git add src/wintermute/db/__init__.py src/wintermute/db/engine.py tests/test_db_engine.py
git commit -m "feat(db): add engine creation and session management"
```

---

### Task 3: ORM Models — All Seven Tables

**Files:**
- Create: `src/wintermute/db/models.py`
- Test: `tests/test_db_models.py`

**Step 1: Write the failing test**

```python
# tests/test_db_models.py
"""Tests for ORM model definitions."""
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from wintermute.db.models import (
    Base,
    Sample,
    ScanResult,
    AdversarialVariant,
    AdversarialCycle,
    Model,
    TrainingRun,
    EtlRun,
    EtlRunSource,
)


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_sample_crud(db_session):
    s = Sample(
        sha256="a" * 64,
        family="Emotet",
        label=1,
        source="synthetic",
        opcode_count=500,
    )
    db_session.add(s)
    db_session.commit()

    result = db_session.get(Sample, "a" * 64)
    assert result is not None
    assert result.family == "Emotet"
    assert result.label == 1


def test_scan_result_crud(db_session):
    sr = ScanResult(
        id=str(uuid.uuid4()),
        sha256="b" * 64,
        predicted_family="AgentTesla",
        predicted_label=1,
        confidence=0.95,
        probabilities={"AgentTesla": 0.95, "safe": 0.05},
        model_version="3.0.0",
    )
    db_session.add(sr)
    db_session.commit()

    result = db_session.get(ScanResult, sr.id)
    assert result.predicted_family == "AgentTesla"
    assert result.confidence == 0.95


def test_model_lifecycle(db_session):
    m = Model(
        id=str(uuid.uuid4()),
        version="3.0.0",
        architecture="WintermuteMalwareDetector",
        weights_path="models/v3/detector.safetensors",
        manifest_path="models/v3/manifest.json",
        vocab_size=512,
        num_classes=2,
        dims=256,
        status="staged",
    )
    db_session.add(m)
    db_session.commit()

    result = db_session.get(Model, m.id)
    assert result.status == "staged"
    assert result.version == "3.0.0"


def test_etl_run_with_sources(db_session):
    run_id = str(uuid.uuid4())
    run = EtlRun(
        id=run_id,
        config_hash="abc123",
        total_samples=500,
    )
    source = EtlRunSource(
        id=str(uuid.uuid4()),
        etl_run_id=run_id,
        source_name="synthetic",
        samples_extracted=500,
    )
    db_session.add(run)
    db_session.add(source)
    db_session.commit()

    result = db_session.get(EtlRun, run_id)
    assert result.total_samples == 500
    assert len(result.sources) == 1
    assert result.sources[0].source_name == "synthetic"


def test_adversarial_variant(db_session):
    # Create prerequisite sample
    s = Sample(sha256="c" * 64, family="Emotet", label=1, source="test")
    cycle = AdversarialCycle(
        id=str(uuid.uuid4()),
        cycle_number=1,
        episodes_played=500,
    )
    db_session.add_all([s, cycle])
    db_session.commit()

    variant = AdversarialVariant(
        id=str(uuid.uuid4()),
        parent_sha256="c" * 64,
        cycle_id=cycle.id,
        mutated_token_ids=[1, 2, 3, 4, 5],
        mutations_applied=[{"type": "nop_insert", "position": 42}],
        mutation_count=1,
        modification_pct=0.05,
        confidence_before=0.95,
        confidence_after=0.35,
        confidence_delta=0.60,
        achieved_evasion=True,
    )
    db_session.add(variant)
    db_session.commit()

    result = db_session.get(AdversarialVariant, variant.id)
    assert result.achieved_evasion is True
    assert result.confidence_delta == 0.60
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_db_models.py -v`
Expected: FAIL — `ImportError: cannot import name 'Sample' from 'wintermute.db.models'`

**Step 3: Write models.py**

```python
# src/wintermute/db/models.py
"""models.py — SQLAlchemy ORM models for all Wintermute database tables."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)
from sqlalchemy.types import JSON


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class Sample(Base):
    __tablename__ = "samples"

    sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    family: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    label: Mapped[int] = mapped_column(Integer, nullable=False)
    source: Mapped[str] = mapped_column(String(100), nullable=False)
    opcode_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    file_type: Mapped[str] = mapped_column(String(20), default="")
    file_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Embedding stored as raw bytes (SQLite) or vector(256) (PostgreSQL)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)

    # Provenance
    etl_run_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("etl_runs.id"), nullable=True
    )
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, default=dict)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    __table_args__ = (
        Index("idx_samples_family", "family"),
        Index("idx_samples_source", "source"),
        Index("idx_samples_label", "label"),
        Index("idx_samples_created", "created_at"),
        Index("idx_samples_etl_run", "etl_run_id"),
    )


class ScanResult(Base):
    __tablename__ = "scan_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), default="")
    file_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Classification output
    predicted_family: Mapped[str] = mapped_column(String(100), nullable=False)
    predicted_label: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    probabilities: Mapped[dict] = mapped_column(JSON, default=dict)

    # Model provenance
    model_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("models.id"), nullable=True
    )
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)

    # Similarity context
    nearest_neighbors: Mapped[list] = mapped_column(JSON, default=list)

    # Execution
    execution_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_ip: Mapped[str | None] = mapped_column(String(45), nullable=True)

    scanned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    __table_args__ = (
        Index("idx_scans_sha256", "sha256"),
        Index("idx_scans_predicted", "predicted_family"),
        Index("idx_scans_confidence", "confidence"),
        Index("idx_scans_scanned_at", "scanned_at"),
        Index("idx_scans_model", "model_id"),
    )


class Model(Base):
    __tablename__ = "models"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    version: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    architecture: Mapped[str] = mapped_column(
        String(100), nullable=False, default="WintermuteMalwareDetector"
    )

    # File references
    weights_path: Mapped[str] = mapped_column(Text, nullable=False)
    manifest_path: Mapped[str] = mapped_column(Text, nullable=False)
    onnx_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Model metadata
    vocab_size: Mapped[int] = mapped_column(Integer, nullable=False)
    num_classes: Mapped[int] = mapped_column(Integer, nullable=False)
    dims: Mapped[int] = mapped_column(Integer, nullable=False)
    max_seq_length: Mapped[int] = mapped_column(Integer, default=2048)
    vocab_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    config: Mapped[dict] = mapped_column(JSON, default=dict)

    # Training provenance
    training_run_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("training_runs.id"), nullable=True
    )
    parent_model_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("models.id"), nullable=True
    )
    pretrained_from: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Evaluation metrics
    best_val_macro_f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_val_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_val_auc_roc: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Lifecycle
    status: Mapped[str] = mapped_column(String(20), default="staged")
    promoted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    retired_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    __table_args__ = (
        Index("idx_models_status", "status"),
        Index("idx_models_architecture", "architecture"),
    )


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    model_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("models.id"), nullable=True
    )

    # Configuration
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    pretrained_weights: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Dataset
    dataset_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    total_samples: Mapped[int | None] = mapped_column(Integer, nullable=True)
    num_classes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    train_split_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    val_split_size: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Training progress
    epochs_completed: Mapped[int] = mapped_column(Integer, default=0)
    best_epoch: Mapped[int | None] = mapped_column(Integer, nullable=True)
    best_val_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_val_macro_f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_val_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Adversarial retraining
    is_adversarial_retrain: Mapped[bool] = mapped_column(Boolean, default=False)
    adversarial_cycle_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("adversarial_cycles.id"), nullable=True
    )
    vault_samples_mixed: Mapped[int] = mapped_column(Integer, default=0)
    trades_beta_final: Mapped[float | None] = mapped_column(Float, nullable=True)
    ewc_lambda: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Timing
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    elapsed_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)


class AdversarialCycle(Base):
    __tablename__ = "adversarial_cycles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    cycle_number: Mapped[int] = mapped_column(Integer, nullable=False)
    defender_model_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("models.id"), nullable=True
    )

    # Attacker stats
    episodes_played: Mapped[int] = mapped_column(Integer, nullable=False)
    total_evasions: Mapped[int] = mapped_column(Integer, default=0)
    evasion_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_confidence_drop: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Defender retraining
    retrained: Mapped[bool] = mapped_column(Boolean, default=False)
    vault_samples_used: Mapped[int] = mapped_column(Integer, default=0)
    defender_f1_before: Mapped[float | None] = mapped_column(Float, nullable=True)
    defender_f1_after: Mapped[float | None] = mapped_column(Float, nullable=True)

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    variants: Mapped[list[AdversarialVariant]] = relationship(back_populates="cycle")


class AdversarialVariant(Base):
    __tablename__ = "adversarial_variants"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)

    # Lineage
    parent_sha256: Mapped[str] = mapped_column(
        String(64), ForeignKey("samples.sha256"), nullable=False
    )
    cycle_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("adversarial_cycles.id"), nullable=False
    )

    # Mutation details
    mutated_token_ids: Mapped[list] = mapped_column(JSON, nullable=False)
    mutations_applied: Mapped[list] = mapped_column(JSON, default=list)
    mutation_count: Mapped[int] = mapped_column(Integer, nullable=False)
    modification_pct: Mapped[float] = mapped_column(Float, nullable=False)

    # Evasion metrics
    confidence_before: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_after: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_delta: Mapped[float] = mapped_column(Float, nullable=False)
    achieved_evasion: Mapped[bool] = mapped_column(Boolean, default=False)

    # Retraining status
    used_in_retraining: Mapped[bool] = mapped_column(Boolean, default=False)
    retraining_run_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("training_runs.id"), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    # Relationships
    cycle: Mapped[AdversarialCycle] = relationship(back_populates="variants")

    __table_args__ = (
        Index("idx_adv_parent", "parent_sha256"),
        Index("idx_adv_cycle", "cycle_id"),
        Index("idx_adv_evasion", "achieved_evasion"),
        Index(
            "idx_adv_not_retrained",
            "used_in_retraining",
            postgresql_where=text("used_in_retraining = false") if False else None,
        ),
    )


class EtlRun(Base):
    __tablename__ = "etl_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    config: Mapped[dict] = mapped_column(JSON, default=dict)

    total_samples: Mapped[int] = mapped_column(Integer, default=0)
    vocab_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    num_classes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_dir: Mapped[str | None] = mapped_column(Text, nullable=True)

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    elapsed_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Relationships
    sources: Mapped[list[EtlRunSource]] = relationship(
        back_populates="etl_run", cascade="all, delete-orphan"
    )


class EtlRunSource(Base):
    __tablename__ = "etl_run_sources"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    etl_run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("etl_runs.id", ondelete="CASCADE"), nullable=False
    )
    source_name: Mapped[str] = mapped_column(String(100), nullable=False)
    samples_extracted: Mapped[int] = mapped_column(Integer, default=0)
    samples_skipped: Mapped[int] = mapped_column(Integer, default=0)
    samples_failed: Mapped[int] = mapped_column(Integer, default=0)
    families_found: Mapped[dict] = mapped_column(JSON, default=dict)
    errors: Mapped[list] = mapped_column(JSON, default=list)
    elapsed_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Relationships
    etl_run: Mapped[EtlRun] = relationship(back_populates="sources")

    __table_args__ = (
        Index("idx_etl_sources_run", "etl_run_id"),
    )
```

Note: The `AdversarialVariant.__table_args__` partial index with `postgresql_where` won't work on SQLite. Remove the `postgresql_where` clause — use a plain index on `used_in_retraining` instead. PostgreSQL-specific partial indexes can be added in a later migration.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_db_models.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/wintermute/db/models.py tests/test_db_models.py
git commit -m "feat(db): add ORM models for all 7 tables"
```

---

### Task 4: SampleRepo

**Files:**
- Create: `src/wintermute/db/repos/__init__.py`
- Create: `src/wintermute/db/repos/samples.py`
- Test: `tests/test_db_repos.py`

**Step 1: Write the failing test**

```python
# tests/test_db_repos.py
"""Tests for repository classes."""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from wintermute.db.models import Base


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


class TestSampleRepo:
    def test_upsert_and_get(self, db_session):
        from wintermute.db.repos.samples import SampleRepo

        repo = SampleRepo(db_session)
        sample = repo.upsert(
            sha256="a" * 64, family="Emotet", label=1,
            source="synthetic", opcode_count=500
        )
        assert sample.sha256 == "a" * 64

        result = repo.get("a" * 64)
        assert result is not None
        assert result.family == "Emotet"

    def test_upsert_idempotent(self, db_session):
        from wintermute.db.repos.samples import SampleRepo

        repo = SampleRepo(db_session)
        repo.upsert(sha256="a" * 64, family="Emotet", label=1, source="s1")
        repo.upsert(sha256="a" * 64, family="Emotet", label=1, source="s2")

        result = repo.get("a" * 64)
        assert result is not None  # no duplicate key error

    def test_exists(self, db_session):
        from wintermute.db.repos.samples import SampleRepo

        repo = SampleRepo(db_session)
        assert repo.exists("a" * 64) is False
        repo.upsert(sha256="a" * 64, family="Emotet", label=1, source="test")
        assert repo.exists("a" * 64) is True

    def test_find_with_filters(self, db_session):
        from wintermute.db.repos.samples import SampleRepo

        repo = SampleRepo(db_session)
        repo.upsert(sha256="a" * 64, family="Emotet", label=1, source="s1", opcode_count=500)
        repo.upsert(sha256="b" * 64, family="AgentTesla", label=2, source="s1", opcode_count=200)
        repo.upsert(sha256="c" * 64, family="Emotet", label=1, source="s2", opcode_count=800)

        results = repo.find(family="Emotet")
        assert len(results) == 2

        results = repo.find(source="s1")
        assert len(results) == 2

        results = repo.find(min_opcodes=600)
        assert len(results) == 1

    def test_count_by_family(self, db_session):
        from wintermute.db.repos.samples import SampleRepo

        repo = SampleRepo(db_session)
        repo.upsert(sha256="a" * 64, family="Emotet", label=1, source="s1")
        repo.upsert(sha256="b" * 64, family="Emotet", label=1, source="s1")
        repo.upsert(sha256="c" * 64, family="AgentTesla", label=2, source="s1")

        counts = repo.count_by_family()
        assert counts == {"Emotet": 2, "AgentTesla": 1}

    def test_bulk_insert(self, db_session):
        from wintermute.db.repos.samples import SampleRepo

        repo = SampleRepo(db_session)
        samples = [
            {"sha256": f"{chr(97+i)}" * 64, "family": "test", "label": 0, "source": "bulk"}
            for i in range(10)
        ]
        count = repo.bulk_insert(samples)
        assert count == 10
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_db_repos.py::TestSampleRepo -v`
Expected: FAIL

**Step 3: Write SampleRepo**

Implement `src/wintermute/db/repos/samples.py` with all methods from the spec (section 5.1): `upsert`, `exists`, `get`, `find`, `count_by_family`, `count_by_source`, `bulk_insert`, `set_embedding`, `bulk_set_embeddings`.

Also create `src/wintermute/db/repos/__init__.py` that re-exports all repos.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_db_repos.py::TestSampleRepo -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/wintermute/db/repos/ tests/test_db_repos.py
git commit -m "feat(db): add SampleRepo with CRUD and bulk operations"
```

---

### Task 5: ScanRepo

**Files:**
- Create: `src/wintermute/db/repos/scans.py`
- Test: add `TestScanRepo` class to `tests/test_db_repos.py`

Follow the same TDD pattern. Implement all methods from spec section 5.2: `record`, `history`, `recent`, `by_family`, `uncertain`, `stats`.

**Commit:** `feat(db): add ScanRepo with scan history and stats`

---

### Task 6: ModelRepo

**Files:**
- Create: `src/wintermute/db/repos/models_repo.py`
- Test: add `TestModelRepo` class to `tests/test_db_repos.py`

Implement all methods from spec section 5.5: `register`, `promote`, `retire`, `active`, `history`, `compare`. The `promote` method must retire the current active model of the same architecture atomically.

**Commit:** `feat(db): add ModelRepo with registry and lifecycle management`

---

### Task 7: AdversarialRepo

**Files:**
- Create: `src/wintermute/db/repos/adversarial.py`
- Test: add `TestAdversarialRepo` class to `tests/test_db_repos.py`

Implement all methods from spec section 5.4: `store_variant`, `get_vault`, `mark_retrained`, `vulnerability_report`, `start_cycle`, `complete_cycle`.

**Commit:** `feat(db): add AdversarialRepo with vault CRUD and cycle tracking`

---

### Task 8: EmbeddingRepo

**Files:**
- Create: `src/wintermute/db/repos/embeddings.py`
- Test: add `TestEmbeddingRepo` class to `tests/test_db_repos.py`

This is the most complex repo. It must abstract vector search across SQLite (sqlite-vec) and PostgreSQL (pgvector).

Key implementation details:
- Store embeddings as `struct.pack('256f', *vec)` bytes in SQLite
- Use raw SQL for vector distance queries (sqlite-vec: `vec_distance_cosine()`, pgvector: `<=>`)
- Detect backend from the engine URL at repo initialization
- `find_nearest()` returns empty list with warning if vector extension unavailable

Implement: `find_nearest`, `find_nearest_with_scans`, `cluster_family`, `coverage_stats`.

**Commit:** `feat(db): add EmbeddingRepo with sqlite-vec and pgvector support`

---

### Task 9: ETL Pipeline Integration

**Files:**
- Modify: `src/wintermute/data/etl/pipeline.py` (lines 35-53 constructor, line 247-310 run method)
- Test: `tests/test_db_etl_integration.py`

**Step 1: Write the failing test**

```python
# tests/test_db_etl_integration.py
def test_pipeline_writes_samples_to_db(tmp_path):
    """ETL pipeline should write samples to DB when session is provided."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from wintermute.db.models import Base, Sample
    from wintermute.db.repos.samples import SampleRepo
    from wintermute.data.etl.pipeline import Pipeline

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    config = {
        "pipeline": {"out_dir": str(tmp_path), "max_seq_length": 64, "shuffle": False},
        "sources": {"synthetic": {"enabled": True, "n_samples": 10}},
    }
    pipeline = Pipeline(config=config, db_session=session)
    result = pipeline.run()

    repo = SampleRepo(session)
    assert result.total_samples == 10
    # Samples should now be in the database
    counts = repo.count_by_family()
    assert sum(counts.values()) == 10


def test_pipeline_works_without_db(tmp_path):
    """ETL pipeline should work fine when no DB session is provided."""
    from wintermute.data.etl.pipeline import Pipeline

    config = {
        "pipeline": {"out_dir": str(tmp_path), "max_seq_length": 64, "shuffle": False},
        "sources": {"synthetic": {"enabled": True, "n_samples": 5}},
    }
    pipeline = Pipeline(config=config)  # No db_session
    result = pipeline.run()
    assert result.total_samples == 5  # Still works
```

**Step 2: Modify Pipeline constructor and run method**

Add optional `db_session` parameter to `Pipeline.__init__()`. In `run()`, after `_extract()` returns samples and before `_encode()`:

1. If `db_session` is not None, create an `EtlRun` row
2. Compute SHA256 for each `RawSample` (hash of `"|".join(sample.opcodes)`)
3. Bulk insert samples via `SampleRepo.bulk_insert()`
4. After `_save()`, write `EtlRunSource` rows and update the `EtlRun` with completion stats

Wrap all DB operations in try/except — log warnings on failure, never crash the pipeline.

**Step 3: Run tests**

Run: `pytest tests/test_db_etl_integration.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/wintermute/data/etl/pipeline.py tests/test_db_etl_integration.py
git commit -m "feat(db): integrate ETL pipeline with sample database"
```

---

## Phase B: Scan History

### Task 10: CLI Scan DB Integration

**Files:**
- Modify: `src/wintermute/cli.py` (scan command, around line 61-118)

Add optional DB write after inference in the `scan` command. If DB is available (engine initialized), write a `ScanResult` row. Compute file SHA256 from the target file bytes. Wrap in try/except.

**Commit:** `feat(db): record scan results in database from CLI`

---

### Task 11: FastAPI DB Integration

**Files:**
- Create: `api/dependencies.py`
- Modify: `api/main.py` (add startup event, modify scan endpoint)
- Modify: `api/schemas.py` (add new response models)

Add `get_db()` FastAPI dependency. Add startup event to call `init_db()`. Enhance the scan response with `prior_scans` and `similar_known_samples`. Add new endpoints: `GET /api/v1/scans`, `GET /api/v1/samples/{sha256}`, `GET /api/v1/similar/{sha256}`, `GET /api/v1/models`, `GET /api/v1/stats`.

**Commit:** `feat(db): add database-backed API endpoints`

---

## Phase C: Model Registry

### Task 12: Trainer Integration

**Files:**
- Modify: `src/wintermute/engine/joint_trainer.py` (constructor line 72-94, train method line 447-504)

Add optional `db_session` parameter to `JointTrainer.__init__()`. In `train()`:
- On start: create `TrainingRun` row
- On best checkpoint save (line 497-499): update `TrainingRun` with best metrics
- After training completes: create `Model` row with `status='staged'`

Wrap all DB operations in try/except.

**Commit:** `feat(db): register models and training runs in database`

---

## Phase D: Adversarial Vault

### Task 13: Orchestrator Integration

**Files:**
- Modify: `src/wintermute/adversarial/orchestrator.py` (constructor line 37-86, run_cycle line 88-120, _collect_rollouts line 158-168)

Add optional `db_session` parameter to `AdversarialOrchestrator.__init__()`. In `run_cycle()`:
- On start: create `AdversarialCycle` row
- On vault.add() (line 160): also write `AdversarialVariant` row via `AdversarialRepo`
- On cycle end: update cycle row with stats

**Commit:** `feat(db): persist adversarial cycles and variants in database`

---

## Phase E: Production Hardening

### Task 14: CLI Database Commands

**Files:**
- Create: `src/wintermute/db/cli_db.py`
- Modify: `src/wintermute/cli.py` (add db sub-Typer registration, around line 35-39)
- Test: `tests/test_db_cli.py`

Implement all CLI commands from spec section 9.1:
- `wintermute db init` — create tables
- `wintermute db stats` — sample counts, scan counts, embedding coverage
- `wintermute db samples` — query with --family, --source, --min-opcodes filters
- `wintermute db scans` — query with --recent, --sha256, --uncertain filters
- `wintermute db models` — list models; `wintermute db models promote <version>`
- `wintermute db similar <sha256>` — k-NN search
- `wintermute db vault` — stats and unused variant queries
- `wintermute db embed` — batch encode samples, coverage stats

Register via the same `register_X_commands(typer_app)` pattern used by ETL.

**Commit:** `feat(db): add wintermute db CLI subcommands`

---

### Task 15: Alembic Migrations Setup

**Files:**
- Create: `src/wintermute/db/migrations/env.py`
- Create: `src/wintermute/db/migrations/script.py.mako`
- Create: `src/wintermute/db/migrations/versions/001_initial_schema.py`
- Create: `alembic.ini` (project root)

Set up Alembic with the initial migration matching the ORM models. Configure `env.py` to import `wintermute.db.models.Base.metadata`. The initial migration should be auto-generated from the ORM models.

Add `wintermute db migrate` to CLI (wraps `alembic upgrade head`).

**Commit:** `feat(db): add Alembic migration infrastructure with initial schema`

---

### Task 16: Docker Compose Update

**Files:**
- Modify: `docker-compose.yml`
- Modify: `Dockerfile`

Add PostgreSQL service with pgvector. Add `WINTERMUTE_DATABASE_URL` to api and worker services. Add `pgdata` volume. Update Dockerfile to install `db` extras.

```yaml
# Add to docker-compose.yml
db:
  image: pgvector/pgvector:pg17
  environment:
    POSTGRES_DB: wintermute
    POSTGRES_USER: wintermute
    POSTGRES_PASSWORD: ${DB_PASSWORD:-wintermute_dev}
  volumes:
    - pgdata:/var/lib/postgresql/data
  ports:
    - "5432:5432"
  healthcheck:
    test: ["CMD-SHELL", "pg_isready -U wintermute"]
    interval: 5s
    retries: 5

# Modify api and worker to add:
environment:
  - WINTERMUTE_DATABASE_URL=postgresql+psycopg://wintermute:${DB_PASSWORD:-wintermute_dev}@db:5432/wintermute
depends_on:
  db:
    condition: service_healthy

volumes:
  pgdata:
```

**Commit:** `feat(db): add PostgreSQL with pgvector to Docker Compose`

---

### Task 17: Final Integration Test and Cleanup

**Files:**
- Update: `src/wintermute/db/repos/__init__.py` (ensure all repos are exported)
- Update: `src/wintermute/db/__init__.py` (ensure public API is clean)
- Run: full test suite

Run: `pytest tests/ -v --tb=short`
Run: `ruff check src/ tests/ api/`
Run: `ruff format src/ tests/ api/`

Fix any issues found.

**Commit:** `chore(db): final cleanup and full test pass`

---

## Summary

| Task | Phase | Component | Commit Message |
|------|-------|-----------|----------------|
| 1 | A | Dependencies | `chore: add database dependencies and config` |
| 2 | A | Engine | `feat(db): add engine creation and session management` |
| 3 | A | ORM Models | `feat(db): add ORM models for all 7 tables` |
| 4 | A | SampleRepo | `feat(db): add SampleRepo with CRUD and bulk operations` |
| 5 | A | ScanRepo | `feat(db): add ScanRepo with scan history and stats` |
| 6 | A | ModelRepo | `feat(db): add ModelRepo with registry and lifecycle` |
| 7 | A | AdversarialRepo | `feat(db): add AdversarialRepo with vault and cycles` |
| 8 | A | EmbeddingRepo | `feat(db): add EmbeddingRepo with vector search` |
| 9 | A | ETL Integration | `feat(db): integrate ETL pipeline with sample database` |
| 10 | B | CLI Scan | `feat(db): record scan results in database from CLI` |
| 11 | B | FastAPI | `feat(db): add database-backed API endpoints` |
| 12 | C | Trainer | `feat(db): register models and training runs in database` |
| 13 | D | Adversarial | `feat(db): persist adversarial cycles and variants` |
| 14 | E | CLI Commands | `feat(db): add wintermute db CLI subcommands` |
| 15 | E | Alembic | `feat(db): add Alembic migrations with initial schema` |
| 16 | E | Docker | `feat(db): add PostgreSQL with pgvector to Docker Compose` |
| 17 | E | Cleanup | `chore(db): final cleanup and full test pass` |
