# Database Layer Design

Date: 2026-02-28
Status: Approved
Spec: `db.md`

## Summary

Implement a unified database layer for Wintermute using SQLAlchemy ORM with PostgreSQL (production) and SQLite (development). Replaces scattered flat-file state with queryable, cross-component storage. Includes pgvector/sqlite-vec for embedding similarity search.

## Scope

All five phases from `db.md` (A through E):
- Foundation: engine, ORM models, SampleRepo, EmbeddingRepo
- Scan History: ScanRepo, CLI/API scan persistence
- Model Registry: ModelRepo, training run tracking
- Adversarial Vault: AdversarialRepo, cycle tracking
- Production Hardening: Alembic migrations, Docker Compose with PostgreSQL

## Architecture Decisions

### Integration Approach: Direct Repository Injection

Repos are passed into constructors as optional parameters (like `MLflowTracker`). When `None`, DB writes are silently skipped. This matches the existing codebase pattern and keeps DB operations explicit and testable.

### Vector Search: Full Parity

Both SQLite (sqlite-vec) and PostgreSQL (pgvector) support vector similarity search. The `EmbeddingRepo` abstracts backend differences. If sqlite-vec is not installed, vector operations gracefully return empty results with a warning.

### Migrations: Alembic From Day One

Full Alembic setup with initial migration. `wintermute db init` uses `create_all()` for quick bootstrap; `wintermute db migrate` runs Alembic for schema evolution.

## Module Structure

```
src/wintermute/db/
├── __init__.py           # Public API: get_session, init_db, get_engine
├── engine.py             # Engine creation, SQLite/PG detection, session factory
├── models.py             # All 7 SQLAlchemy ORM models
├── repos/
│   ├── __init__.py       # Re-export all repos
│   ├── samples.py        # SampleRepo
│   ├── scans.py          # ScanRepo
│   ├── adversarial.py    # AdversarialRepo
│   ├── models_repo.py    # ModelRepo (avoids name collision)
│   └── embeddings.py     # EmbeddingRepo
├── cli_db.py             # `wintermute db` subcommands
└── migrations/
    ├── env.py
    ├── script.py.mako
    └── versions/
        └── 001_initial_schema.py
```

## Schema

Seven tables as defined in `db.md` sections 3.2-3.8:
- `samples` (central, SHA256 PK, vector(256) embedding column)
- `scan_results` (UUID PK, soft FK to samples)
- `adversarial_variants` (UUID PK, FK to samples + cycles)
- `adversarial_cycles` (UUID PK, FK to models)
- `models` (UUID PK, staged/active/retired lifecycle)
- `training_runs` (UUID PK, FK to models + adversarial_cycles)
- `etl_runs` + `etl_run_sources` (UUID PKs, provenance tracking)

Key portability decisions:
- UUIDs generated in Python (`uuid.uuid4()`)
- Timestamps use Python `datetime.utcnow()`
- Arrays stored as JSON (not PostgreSQL ARRAY)
- INET stored as TEXT

## Integration Points

### ETL Pipeline (pipeline.py)
- `Pipeline.__init__` gains optional `session` parameter
- After `_extract()`, before `_encode()`: bulk insert samples with SHA256 computed from opcode sequence
- Create `etl_runs` row at start, `etl_run_sources` at end
- No-op when session is None

### Trainer (joint_trainer.py)
- Constructor gains optional `session` parameter
- Create `training_runs` row on start
- Update with best metrics during training
- Register `models` row on completion (status='staged')

### Adversarial Orchestrator (orchestrator.py)
- Constructor gains optional `session` parameter
- Create `adversarial_cycles` row per cycle
- Write `adversarial_variants` on each vault.add()
- In-memory vault continues working alongside DB

### CLI (cli.py)
- New `db` sub-Typer: init, stats, samples, scans, models, similar, vault, embed
- `scan` command writes scan_results when DB available
- `train` command passes session to trainer

### FastAPI (api/main.py)
- New `api/dependencies.py` with `get_db()` dependency
- Startup event calls `init_db()`
- Enhanced scan response with `prior_scans` + `similar_known_samples`
- New endpoints: /scans, /samples/{sha256}, /similar/{sha256}, /models, /stats

### Docker Compose
- Add `pgvector/pgvector:pg17` service
- Add `WINTERMUTE_DATABASE_URL` to api/worker
- Add `pgdata` volume

## Dependencies

New optional group in `pyproject.toml`:
```
db = ["sqlalchemy>=2.0.0", "alembic>=1.13.0", "sqlite-vec>=0.1.0"]
```
PostgreSQL extras (not in base): `pgvector`, `psycopg[binary]`

## Configuration

```yaml
# configs/database.yaml
database:
  url: "sqlite:///data/wintermute.db"
  echo: false
```
Override: `WINTERMUTE_DATABASE_URL` env var

## Testing

- All repo tests use in-memory SQLite
- Vector tests skip when sqlite-vec unavailable
- PostgreSQL tests gated behind `@pytest.mark.postgres`
- Test files: `test_db_engine.py`, `test_db_repos.py`, `test_db_cli.py`

## Error Handling

- Repo failures log warnings, never crash core pipelines
- Missing DB config defaults to `sqlite:///data/wintermute.db`
- Missing sqlite-vec degrades vector search to empty results
- Batch inserts use `on_conflict_do_nothing()` for idempotency

## Files

New (~15): db module (9 files), migrations (3), config (1), api dependency (1), tests (2)
Modified (~7): pyproject.toml, pipeline.py, cli.py, api/main.py, api/schemas.py, docker-compose.yml, Dockerfile
