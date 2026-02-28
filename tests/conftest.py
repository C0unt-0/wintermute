import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from wintermute.db.models import Base


def _set_sqlite_pragmas(dbapi_conn, _connection_record):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON;")
    cursor.close()


@pytest.fixture()
def db_session():
    """Yield an in-memory SQLite session with all tables created."""
    engine = create_engine("sqlite:///:memory:")
    event.listen(engine, "connect", _set_sqlite_pragmas)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()
