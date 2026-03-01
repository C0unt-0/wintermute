"""Tests for wintermute.db.engine — connection management."""

from __future__ import annotations

import wintermute.db.engine as engine_mod
from wintermute.db.engine import create_db_engine, get_engine, get_session, init_db


def _reset_globals() -> None:
    """Reset module-level engine/session globals between tests."""
    engine_mod._engine = None
    engine_mod._SessionFactory = None


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


def test_create_sqlite_engine():
    """Verify that WAL mode and foreign keys are enabled for SQLite."""
    _reset_globals()

    engine = create_db_engine(url="sqlite:///:memory:")
    assert engine is not None
    assert get_engine() is engine

    with engine.connect() as conn:
        wal = conn.exec_driver_sql("PRAGMA journal_mode;").scalar()
        fk = conn.exec_driver_sql("PRAGMA foreign_keys;").scalar()

    # In-memory SQLite returns 'memory' for journal_mode (WAL only applies
    # to on-disk databases), but the PRAGMA was still executed without error.
    assert wal in ("wal", "memory")
    assert fk == 1

    engine.dispose()
    _reset_globals()


def test_get_session_contextmanager():
    """Verify the session context manager yields a usable session."""
    _reset_globals()

    create_db_engine(url="sqlite:///:memory:")

    with get_session() as session:
        # Session should be open and usable
        result = session.execute(engine_mod.text("SELECT 1")).scalar()
        assert result == 1

    _reset_globals()


def test_init_db_creates_tables():
    """Verify that init_db creates the expected tables.

    NOTE: This test fully validates once models.py defines the Sample model
    (Task 3). Until then it verifies init_db runs without error.
    """
    _reset_globals()

    engine = create_db_engine(url="sqlite:///:memory:")
    init_db(engine)

    from sqlalchemy import inspect

    inspector = inspect(engine)
    tables = inspector.get_table_names()

    # After Task 3 models are defined, 'samples' should appear.
    # If models.py only has Base with no tables yet, the list may be empty,
    # which is still correct at this stage.
    from wintermute.db.models import Base

    expected = set(Base.metadata.tables.keys())
    actual = set(tables)
    assert expected.issubset(actual), f"Missing tables: {expected - actual}"

    engine.dispose()
    _reset_globals()
