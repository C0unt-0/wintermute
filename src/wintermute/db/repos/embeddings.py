"""Repository for vector similarity search operations."""

from __future__ import annotations

import logging
import math
import struct

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from wintermute.db.models import Sample, ScanResult

logger = logging.getLogger(__name__)


def _cosine_distance(a: list[float], b: tuple[float, ...] | list[float]) -> float:
    """Compute cosine distance between two vectors.

    Returns a value in [0, 2] where 0 means identical direction.
    """
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - dot / (norm_a * norm_b)


def _unpack_embedding(data: bytes) -> tuple[float, ...]:
    """Deserialize embedding bytes to a tuple of floats."""
    dim = len(data) // 4
    return struct.unpack(f"{dim}f", data)


class EmbeddingRepo:
    """Vector similarity search operations.

    Abstracts vector search across SQLite (sqlite-vec) and PostgreSQL (pgvector).
    If the vector extension is unavailable, search methods fall back to a
    pure-Python cosine distance implementation (slower but functional).
    """

    def __init__(self, session: Session) -> None:
        self._session = session
        self._dialect = session.bind.dialect.name  # "sqlite" or "postgresql"
        self._vec_available = self._check_vec_support()
        if not self._vec_available:
            logger.info(
                "Vector extension not available for %s — "
                "using pure-Python cosine distance fallback",
                self._dialect,
            )

    # ------------------------------------------------------------------
    # Extension detection
    # ------------------------------------------------------------------

    def _check_vec_support(self) -> bool:
        """Check if the vector extension is available.

        For SQLite: try to call ``vec_distance_cosine`` from sqlite-vec.
        For PostgreSQL: check if the pgvector extension exists.
        Returns *True* if available, *False* otherwise.
        """
        try:
            if self._dialect == "sqlite":
                self._session.execute(
                    text("SELECT vec_distance_cosine(zeroblob(4), zeroblob(4))")
                )
                return True
            elif self._dialect == "postgresql":
                row = self._session.execute(
                    text(
                        "SELECT 1 FROM pg_extension WHERE extname = 'vector' LIMIT 1"
                    )
                ).first()
                return row is not None
        except Exception:
            return False
        return False

    # ------------------------------------------------------------------
    # Nearest-neighbour search
    # ------------------------------------------------------------------

    def find_nearest(
        self,
        query_vec: list[float],
        k: int = 5,
        family: str | None = None,
        max_distance: float | None = None,
    ) -> list[dict]:
        """k-nearest neighbour search with optional filters.

        Returns ``[{sha256, family, label, distance}, ...]`` ordered by
        ascending cosine distance.

        * SQLite with sqlite-vec: uses ``vec_distance_cosine()``.
        * PostgreSQL with pgvector: uses the ``<=>`` cosine distance operator.
        * Fallback: pure-Python cosine distance over all candidate embeddings.
        """
        if self._vec_available and self._dialect == "sqlite":
            return self._find_nearest_sqlite_vec(query_vec, k, family, max_distance)
        if self._vec_available and self._dialect == "postgresql":
            return self._find_nearest_pgvector(query_vec, k, family, max_distance)
        return self._find_nearest_python(query_vec, k, family, max_distance)

    # -- sqlite-vec path --------------------------------------------------

    def _find_nearest_sqlite_vec(
        self,
        query_vec: list[float],
        k: int,
        family: str | None,
        max_distance: float | None,
    ) -> list[dict]:
        """Nearest-neighbour search using sqlite-vec ``vec_distance_cosine``.

        Uses a single SQL query instead of per-row distance calls.
        """
        dim = len(query_vec)
        query_blob = struct.pack(f"{dim}f", *query_vec)

        sql = """
            SELECT sha256, family, label,
                   vec_distance_cosine(embedding, :query_blob) AS distance
            FROM samples
            WHERE embedding IS NOT NULL
        """
        params: dict = {"query_blob": query_blob, "k": k}

        if family is not None:
            sql += " AND family = :family"
            params["family"] = family

        sql += " ORDER BY distance LIMIT :k"

        rows = self._session.execute(text(sql), params).fetchall()

        results = [
            {
                "sha256": r.sha256,
                "family": r.family,
                "label": r.label,
                "distance": float(r.distance),
            }
            for r in rows
        ]

        # Filter by max_distance in Python (SQLite alias visibility varies)
        if max_distance is not None:
            results = [r for r in results if r["distance"] <= max_distance]

        return results

    # -- pgvector path ----------------------------------------------------

    def _find_nearest_pgvector(
        self,
        query_vec: list[float],
        k: int,
        family: str | None,
        max_distance: float | None,
    ) -> list[dict]:
        """Nearest-neighbour search using pgvector ``<=>`` operator.

        Uses parameterized queries to avoid SQL injection.
        """
        # Validate all values are numeric before building the vector literal
        for i, v in enumerate(query_vec):
            if not isinstance(v, (int, float)):
                raise ValueError(
                    f"query_vec[{i}] must be int or float, got {type(v).__name__}"
                )

        vec_literal = "[" + ",".join(str(v) for v in query_vec) + "]"

        sql = """
            SELECT sha256, family, label,
                   embedding <=> :query_vec::vector AS distance
            FROM samples
            WHERE embedding IS NOT NULL
        """
        params: dict = {"query_vec": vec_literal, "k": k}

        if family is not None:
            sql += " AND family = :family"
            params["family"] = family

        if max_distance is not None:
            sql += " AND (embedding <=> :query_vec::vector) <= :max_distance"
            params["max_distance"] = max_distance

        sql += " ORDER BY distance LIMIT :k"

        rows = self._session.execute(text(sql), params).fetchall()
        return [
            {
                "sha256": row.sha256,
                "family": row.family,
                "label": row.label,
                "distance": float(row.distance),
            }
            for row in rows
        ]

    # -- pure-Python fallback ---------------------------------------------

    def _find_nearest_python(
        self,
        query_vec: list[float],
        k: int,
        family: str | None,
        max_distance: float | None,
    ) -> list[dict]:
        """Pure-Python cosine distance fallback (slow but always available)."""
        stmt = select(Sample).where(Sample.embedding.isnot(None))
        if family is not None:
            stmt = stmt.where(Sample.family == family)

        samples = self._session.execute(stmt).scalars().all()
        if not samples:
            return []

        results: list[dict] = []
        for sample in samples:
            emb = _unpack_embedding(sample.embedding)
            dist = _cosine_distance(query_vec, emb)
            if max_distance is not None and dist > max_distance:
                continue
            results.append(
                {
                    "sha256": sample.sha256,
                    "family": sample.family,
                    "label": sample.label,
                    "distance": dist,
                }
            )

        results.sort(key=lambda r: r["distance"])
        return results[:k]

    # ------------------------------------------------------------------
    # Nearest neighbours enriched with scan data
    # ------------------------------------------------------------------

    def find_nearest_with_scans(
        self,
        query_vec: list[float],
        k: int = 5,
    ) -> list[dict]:
        """Nearest neighbours enriched with most recent scan for each sample.

        Returns ``[{sha256, family, distance, last_scan_confidence,
        last_scan_date}, ...]``.
        """
        neighbours = self.find_nearest(query_vec, k=k)
        if not neighbours:
            return []

        enriched: list[dict] = []
        for nb in neighbours:
            # Fetch the most recent scan for this sample
            scan_stmt = (
                select(ScanResult.confidence, ScanResult.scanned_at)
                .where(ScanResult.sha256 == nb["sha256"])
                .order_by(ScanResult.scanned_at.desc())
                .limit(1)
            )
            scan_row = self._session.execute(scan_stmt).first()
            enriched.append(
                {
                    "sha256": nb["sha256"],
                    "family": nb["family"],
                    "distance": nb["distance"],
                    "last_scan_confidence": (
                        float(scan_row.confidence) if scan_row else None
                    ),
                    "last_scan_date": (
                        scan_row.scanned_at.isoformat() if scan_row else None
                    ),
                }
            )

        return enriched

    # ------------------------------------------------------------------
    # Family clustering
    # ------------------------------------------------------------------

    def cluster_family(self, family: str, k: int = 10) -> list[dict]:
        """Samples within a family ordered by distance to the family centroid.

        Computes the centroid (mean embedding) of all samples in the family,
        then returns the *k* samples closest to that centroid.
        """
        stmt = select(Sample).where(
            Sample.family == family,
            Sample.embedding.isnot(None),
        )
        samples = self._session.execute(stmt).scalars().all()
        if not samples:
            return []

        # Compute centroid
        dim = len(samples[0].embedding) // 4
        centroid = [0.0] * dim
        for sample in samples:
            emb = _unpack_embedding(sample.embedding)
            for i in range(dim):
                centroid[i] += emb[i]
        n = len(samples)
        centroid = [c / n for c in centroid]

        # Rank by distance to centroid
        results: list[dict] = []
        for sample in samples:
            emb = _unpack_embedding(sample.embedding)
            dist = _cosine_distance(centroid, emb)
            results.append(
                {
                    "sha256": sample.sha256,
                    "family": sample.family,
                    "label": sample.label,
                    "distance": dist,
                }
            )

        results.sort(key=lambda r: r["distance"])
        return results[:k]

    # ------------------------------------------------------------------
    # Coverage statistics
    # ------------------------------------------------------------------

    def coverage_stats(self) -> dict:
        """Embedding coverage statistics.

        Returns ``{total_samples, with_embedding, without_embedding, pct_covered}``.
        Does **not** require the vector extension.
        """
        total: int = (
            self._session.execute(
                select(func.count()).select_from(Sample)
            ).scalar()
            or 0
        )
        with_emb: int = (
            self._session.execute(
                select(func.count())
                .select_from(Sample)
                .where(Sample.embedding.isnot(None))
            ).scalar()
            or 0
        )
        without_emb = total - with_emb
        pct = (with_emb / total * 100.0) if total > 0 else 0.0

        return {
            "total_samples": total,
            "with_embedding": with_emb,
            "without_embedding": without_emb,
            "pct_covered": pct,
        }
