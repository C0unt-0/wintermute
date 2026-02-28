"""api/dependencies.py -- FastAPI dependency injection for database sessions.

Provides ``get_db()``, a request-scoped SQLAlchemy session dependency.
If the DB engine has not been initialised yet, the dependency raises
HTTP 503 so individual endpoints do not need to check engine state.
"""

from __future__ import annotations

from fastapi import HTTPException
from wintermute.db.engine import get_session


def get_db():
    """Yield a SQLAlchemy session for request-scoped DB access.

    Raises :class:`HTTPException` (503) when the engine has not been
    created (``get_session`` raises ``RuntimeError``).
    """
    try:
        with get_session() as session:
            yield session
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail="Database not available. The DB engine has not been initialised.",
        )
