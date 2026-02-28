"""Alembic environment configuration for Wintermute."""

from __future__ import annotations

import logging
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, event, pool

from wintermute.db.engine import _resolve_url
from wintermute.db.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

logger = logging.getLogger("alembic.env")

target_metadata = Base.metadata


def _get_url() -> str:
    """Return the database URL.

    Delegates to :func:`_resolve_url` which checks (in order):
    1. ``WINTERMUTE_DATABASE_URL`` env var
    2. ``configs/database.yaml``
    3. Hard-coded fallback

    If a URL was set programmatically via
    ``Config.set_main_option("sqlalchemy.url", ...)``, that value is
    forwarded to ``_resolve_url`` as the explicit *url* argument so it
    takes highest priority.
    """
    cfg_url = config.get_main_option("sqlalchemy.url")
    # Treat the default ini placeholder as "not set".
    ini_default = "sqlite:///data/wintermute.db"
    explicit = cfg_url if cfg_url and cfg_url != ini_default else None
    return _resolve_url(explicit)


def _set_sqlite_pragmas(dbapi_conn, _connection_record):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON;")
    cursor.close()


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode -- emit SQL to stdout."""
    url = _get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode -- connect to the database."""
    url = _get_url()
    connectable = create_engine(url, poolclass=pool.NullPool)

    if url.startswith("sqlite"):
        event.listen(connectable, "connect", _set_sqlite_pragmas)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
