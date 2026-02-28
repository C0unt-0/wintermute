"""Integration tests for ETL pipeline with database."""

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

import wintermute.data.etl.sources  # noqa: F401  # trigger source auto-registration
from wintermute.db.models import Base, EtlRun, EtlRunSource, Sample


@pytest.fixture
def db_session():
    """Create an in-memory SQLite session with foreign keys enabled."""
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    engine.dispose()


def test_pipeline_writes_samples_to_db(db_session, tmp_path):
    """ETL pipeline should write samples to DB when session is provided."""
    from wintermute.data.etl.pipeline import Pipeline
    from wintermute.db.repos.samples import SampleRepo

    config = {
        "pipeline": {
            "out_dir": str(tmp_path),
            "max_seq_length": 64,
            "shuffle": False,
        },
        "sources": {"synthetic": {"enabled": True, "n_samples": 10}},
    }
    pipeline = Pipeline(config=config, db_session=db_session)
    result = pipeline.run()

    assert result.total_samples > 0

    repo = SampleRepo(db_session)
    counts = repo.count_by_family()
    assert sum(counts.values()) > 0  # Samples were written to DB


def test_pipeline_creates_etl_run(db_session, tmp_path):
    """ETL pipeline should create EtlRun and EtlRunSource records."""
    from wintermute.data.etl.pipeline import Pipeline

    config = {
        "pipeline": {
            "out_dir": str(tmp_path),
            "max_seq_length": 64,
            "shuffle": False,
        },
        "sources": {"synthetic": {"enabled": True, "n_samples": 5}},
    }
    pipeline = Pipeline(config=config, db_session=db_session)
    pipeline.run()

    # Check EtlRun was created
    runs = db_session.query(EtlRun).all()
    assert len(runs) == 1
    assert runs[0].completed_at is not None
    assert runs[0].total_samples is not None

    # Check EtlRunSource was created
    sources = db_session.query(EtlRunSource).all()
    assert len(sources) >= 1
    assert sources[0].source_name == "synthetic"
    assert sources[0].samples_extracted > 0


def test_pipeline_etl_run_has_completion_stats(db_session, tmp_path):
    """EtlRun should have vocab_size, num_classes, and elapsed_seconds after run."""
    from wintermute.data.etl.pipeline import Pipeline

    config = {
        "pipeline": {
            "out_dir": str(tmp_path),
            "max_seq_length": 64,
            "shuffle": False,
        },
        "sources": {"synthetic": {"enabled": True, "n_samples": 10}},
    }
    pipeline = Pipeline(config=config, db_session=db_session)
    result = pipeline.run()

    run = db_session.query(EtlRun).one()
    assert run.total_samples == result.total_samples
    assert run.vocab_size == result.vocab_size
    assert run.num_classes == result.num_classes
    assert run.elapsed_seconds is not None
    assert run.elapsed_seconds > 0


def test_pipeline_samples_linked_to_etl_run(db_session, tmp_path):
    """Samples written to DB should reference the correct EtlRun."""
    from wintermute.data.etl.pipeline import Pipeline

    config = {
        "pipeline": {
            "out_dir": str(tmp_path),
            "max_seq_length": 64,
            "shuffle": False,
        },
        "sources": {"synthetic": {"enabled": True, "n_samples": 6}},
    }
    pipeline = Pipeline(config=config, db_session=db_session)
    pipeline.run()

    run = db_session.query(EtlRun).one()
    samples = db_session.query(Sample).all()
    assert len(samples) > 0
    for sample in samples:
        assert sample.etl_run_id == run.id


def test_pipeline_works_without_db(tmp_path):
    """ETL pipeline should work fine when no DB session is provided."""
    from wintermute.data.etl.pipeline import Pipeline

    config = {
        "pipeline": {
            "out_dir": str(tmp_path),
            "max_seq_length": 64,
            "shuffle": False,
        },
        "sources": {"synthetic": {"enabled": True, "n_samples": 5}},
    }
    pipeline = Pipeline(config=config)  # No db_session
    result = pipeline.run()
    assert result.total_samples > 0  # Still works without DB


def test_pipeline_config_hash_in_etl_run(db_session, tmp_path):
    """EtlRun should store a config hash for deduplication."""
    from wintermute.data.etl.pipeline import Pipeline

    config = {
        "pipeline": {
            "out_dir": str(tmp_path),
            "max_seq_length": 64,
            "shuffle": False,
        },
        "sources": {"synthetic": {"enabled": True, "n_samples": 5}},
    }
    pipeline = Pipeline(config=config, db_session=db_session)
    pipeline.run()

    run = db_session.query(EtlRun).one()
    assert run.config_hash is not None
    assert len(run.config_hash) == 64  # SHA-256 hex digest
    assert run.config is not None
