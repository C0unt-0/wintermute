"""Tests for Alembic migration infrastructure."""

from __future__ import annotations

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

EXPECTED_TABLES = {
    "samples",
    "scan_results",
    "models",
    "training_runs",
    "adversarial_cycles",
    "adversarial_variants",
    "etl_runs",
    "etl_run_sources",
}


def _make_alembic_cfg(db_path: str) -> Config:
    """Create an Alembic Config pointing at a file-based SQLite DB."""
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    return cfg


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


def test_migration_upgrade_creates_tables(tmp_path):
    """Alembic upgrade head should create all 8 tables."""
    db_file = tmp_path / "test.db"
    cfg = _make_alembic_cfg(str(db_file))

    command.upgrade(cfg, "head")

    engine = create_engine(f"sqlite:///{db_file}")
    tables = set(inspect(engine).get_table_names())
    # alembic_version is expected as well
    assert EXPECTED_TABLES.issubset(tables), f"Missing tables: {EXPECTED_TABLES - tables}"
    engine.dispose()


def test_migration_downgrade_drops_tables(tmp_path):
    """Alembic downgrade base should remove all tables."""
    db_file = tmp_path / "test.db"
    cfg = _make_alembic_cfg(str(db_file))

    command.upgrade(cfg, "head")
    command.downgrade(cfg, "base")

    engine = create_engine(f"sqlite:///{db_file}")
    tables = set(inspect(engine).get_table_names()) - {"alembic_version"}
    assert tables == set(), f"Tables remain after downgrade: {tables}"
    engine.dispose()


def test_cli_migrate_command(tmp_path, monkeypatch):
    """The 'wintermute db migrate' CLI command should run migrations."""
    from typer.testing import CliRunner

    from wintermute.db.cli_db import db_app

    db_file = tmp_path / "cli_test.db"
    monkeypatch.setenv("WINTERMUTE_DATABASE_URL", f"sqlite:///{db_file}")

    runner = CliRunner()
    result = runner.invoke(db_app, ["migrate"])
    assert result.exit_code == 0, result.output
    assert "Database migrations complete." in result.output

    engine = create_engine(f"sqlite:///{db_file}")
    tables = set(inspect(engine).get_table_names())
    assert EXPECTED_TABLES.issubset(tables)
    engine.dispose()
