"""Database engine creation and session management for Wintermute."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import yaml
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

# ---------------------------------------------------------------------------
# Module-level globals
# ---------------------------------------------------------------------------
_engine: Engine | None = None
_SessionFactory: sessionmaker | None = None

# Path to the default config file (relative to project root)
_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "database.yaml"


# ---------------------------------------------------------------------------
# URL resolution
# ---------------------------------------------------------------------------

def _resolve_url(url: str | None = None) -> str:
    """Resolve the database URL from (in priority order):

    1. Explicit *url* argument
    2. ``WINTERMUTE_DATABASE_URL`` environment variable
    3. ``configs/database.yaml`` → ``database.url``
    4. Hard-coded fallback ``sqlite:///data/wintermute.db``
    """
    if url:
        return url

    env_url = os.environ.get("WINTERMUTE_DATABASE_URL")
    if env_url:
        return env_url

    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        if cfg and "database" in cfg and "url" in cfg["database"]:
            return cfg["database"]["url"]

    return "sqlite:///data/wintermute.db"


# ---------------------------------------------------------------------------
# Engine creation
# ---------------------------------------------------------------------------

def _set_sqlite_pragmas(dbapi_conn: Any, _connection_record: Any) -> None:
    """Enable WAL journal mode and foreign-key enforcement for SQLite."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA foreign_keys=ON;")
    cursor.close()


def create_db_engine(url: str | None = None, echo: bool = False, **kwargs: Any) -> Engine:
    """Create and store a SQLAlchemy :class:`Engine`.

    * **SQLite** — sets WAL mode + foreign keys via event listener and
      ``check_same_thread=False``.
    * **PostgreSQL** — applies connection-pool settings (``pool_size``,
      ``max_overflow``).

    Returns the newly created engine (also stored in the module global).
    """
    global _engine, _SessionFactory

    resolved_url = _resolve_url(url)

    connect_args: dict[str, Any] = {}
    engine_kwargs: dict[str, Any] = {"echo": echo, **kwargs}

    if resolved_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        engine_kwargs["connect_args"] = connect_args
    elif resolved_url.startswith("postgresql"):
        engine_kwargs.setdefault("pool_size", 5)
        engine_kwargs.setdefault("max_overflow", 10)

    engine = create_engine(resolved_url, **engine_kwargs)

    if resolved_url.startswith("sqlite"):
        event.listen(engine, "connect", _set_sqlite_pragmas)

    _engine = engine
    _SessionFactory = sessionmaker(bind=engine)
    return engine


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def get_engine() -> Engine | None:
    """Return the current engine, or ``None`` if not yet created."""
    return _engine


def init_db(engine: Engine | None = None) -> None:
    """Create all tables defined on :data:`Base.metadata`.

    If the engine targets PostgreSQL, the ``vector`` extension is enabled
    first (for pgvector support).
    """
    from wintermute.db.models import Base  # local import to avoid circular dependency

    eng = engine or _engine
    if eng is None:
        raise RuntimeError("No engine available. Call create_db_engine() first.")

    url_str = str(eng.url)
    if url_str.startswith("postgresql"):
        with eng.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()

    Base.metadata.create_all(eng)


# ---------------------------------------------------------------------------
# Session context manager
# ---------------------------------------------------------------------------

@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Yield a :class:`Session` with automatic commit / rollback / close."""
    if _SessionFactory is None:
        raise RuntimeError("No session factory. Call create_db_engine() first.")

    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
