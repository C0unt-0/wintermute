"""Repository for Sample catalog operations."""

from __future__ import annotations

import struct

from sqlalchemy import func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from wintermute.db.models import Sample


class SampleRepo:
    """Sample catalog operations."""

    def __init__(self, session: Session) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Single-record CRUD
    # ------------------------------------------------------------------

    def upsert(
        self,
        sha256: str,
        family: str,
        label: int,
        source: str,
        opcode_count: int = 0,
        **kwargs,
    ) -> Sample:
        """Insert or update a sample.  Idempotent on *sha256*."""
        existing = self._session.get(Sample, sha256)
        if existing is not None:
            existing.family = family
            existing.label = label
            existing.source = source
            existing.opcode_count = opcode_count
            for key, value in kwargs.items():
                if hasattr(existing, key):
                    setattr(existing, key, value)
            self._session.flush()
            return existing

        sample = Sample(
            sha256=sha256,
            family=family,
            label=label,
            source=source,
            opcode_count=opcode_count,
            **kwargs,
        )
        self._session.add(sample)
        self._session.flush()
        return sample

    def exists(self, sha256: str) -> bool:
        """Check if a sample is already cataloged."""
        stmt = select(Sample.sha256).where(Sample.sha256 == sha256)
        return self._session.execute(stmt).first() is not None

    def get(self, sha256: str) -> Sample | None:
        """Retrieve by primary key."""
        return self._session.get(Sample, sha256)

    # ------------------------------------------------------------------
    # Filtered queries
    # ------------------------------------------------------------------

    def find(
        self,
        family: str | None = None,
        source: str | None = None,
        label: int | None = None,
        min_opcodes: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Sample]:
        """Filtered query with pagination."""
        stmt = select(Sample)

        if family is not None:
            stmt = stmt.where(Sample.family == family)
        if source is not None:
            stmt = stmt.where(Sample.source == source)
        if label is not None:
            stmt = stmt.where(Sample.label == label)
        if min_opcodes is not None:
            stmt = stmt.where(Sample.opcode_count >= min_opcodes)

        stmt = stmt.order_by(Sample.created_at).limit(limit).offset(offset)
        return list(self._session.execute(stmt).scalars().all())

    # ------------------------------------------------------------------
    # Aggregations
    # ------------------------------------------------------------------

    def count_by_family(self) -> dict[str, int]:
        """Aggregate sample counts per family."""
        stmt = select(Sample.family, func.count()).group_by(Sample.family)
        rows = self._session.execute(stmt).all()
        return {family: count for family, count in rows}

    def count_by_source(self) -> dict[str, int]:
        """Aggregate sample counts per ETL source."""
        stmt = select(Sample.source, func.count()).group_by(Sample.source)
        rows = self._session.execute(stmt).all()
        return {source: count for source, count in rows}

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def bulk_insert(self, samples: list[dict]) -> int:
        """Batch insert from ETL pipeline.  Returns count inserted.

        Uses SQLAlchemy's ``insert().on_conflict_do_nothing()`` for
        idempotency on the *sha256* primary key.
        """
        if not samples:
            return 0

        before_count = self._session.execute(select(func.count()).select_from(Sample)).scalar() or 0

        stmt = (
            sqlite_insert(Sample)
            .values(samples)
            .on_conflict_do_nothing(
                index_elements=["sha256"],
            )
        )
        self._session.execute(stmt)
        self._session.flush()

        after_count = self._session.execute(select(func.count()).select_from(Sample)).scalar() or 0

        return after_count - before_count

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def set_embedding(self, sha256: str, embedding: list[float]) -> None:
        """Update the embedding vector for a sample.

        Converts ``list[float]`` to bytes via ``struct.pack``.
        """
        sample = self._session.get(Sample, sha256)
        if sample is None:
            raise ValueError(f"Sample not found: {sha256}")
        sample.embedding = struct.pack(f"{len(embedding)}f", *embedding)
        self._session.flush()

    def bulk_set_embeddings(self, pairs: list[tuple[str, list[float]]]) -> int:
        """Batch update embeddings.  Returns count updated."""
        count = 0
        for sha256, vec in pairs:
            sample = self._session.get(Sample, sha256)
            if sample is not None:
                sample.embedding = struct.pack(f"{len(vec)}f", *vec)
                count += 1
        self._session.flush()
        return count
