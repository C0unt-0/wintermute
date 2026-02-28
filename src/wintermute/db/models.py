"""ORM models for the Wintermute database layer.

Full table definitions are added in a subsequent commit.
"""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all Wintermute ORM models."""
