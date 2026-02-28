"""Repository for scan result operations."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from wintermute.db.models import ScanResult


class ScanRepo:
    """Scan result operations."""

    def __init__(self, session: Session) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Record
    # ------------------------------------------------------------------

    def record(
        self,
        sha256: str,
        predicted_family: str,
        predicted_label: int,
        confidence: float,
        probabilities: dict,
        model_version: str,
        **kwargs,
    ) -> ScanResult:
        """Record a new scan result."""
        scan = ScanResult(
            sha256=sha256,
            predicted_family=predicted_family,
            predicted_label=predicted_label,
            confidence=confidence,
            probabilities=probabilities,
            model_version=model_version,
            **kwargs,
        )
        self._session.add(scan)
        self._session.flush()
        return scan

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def history(self, sha256: str) -> list[ScanResult]:
        """All scans for a given file hash, most recent first."""
        stmt = (
            select(ScanResult)
            .where(ScanResult.sha256 == sha256)
            .order_by(ScanResult.scanned_at.desc())
        )
        return list(self._session.execute(stmt).scalars().all())

    def recent(self, limit: int = 50, since: datetime | None = None) -> list[ScanResult]:
        """Most recent scan results."""
        stmt = select(ScanResult)
        if since is not None:
            stmt = stmt.where(ScanResult.scanned_at >= since)
        stmt = stmt.order_by(ScanResult.scanned_at.desc()).limit(limit)
        return list(self._session.execute(stmt).scalars().all())

    def by_family(
        self,
        family: str,
        min_confidence: float = 0.0,
        limit: int = 100,
    ) -> list[ScanResult]:
        """Scans with a specific predicted family."""
        stmt = (
            select(ScanResult)
            .where(ScanResult.predicted_family == family)
            .where(ScanResult.confidence >= min_confidence)
            .order_by(ScanResult.scanned_at.desc())
            .limit(limit)
        )
        return list(self._session.execute(stmt).scalars().all())

    def uncertain(self, threshold: float = 0.6, limit: int = 50) -> list[ScanResult]:
        """Scans where the model was uncertain (confidence < threshold)."""
        stmt = (
            select(ScanResult)
            .where(ScanResult.confidence < threshold)
            .order_by(ScanResult.confidence.asc())
            .limit(limit)
        )
        return list(self._session.execute(stmt).scalars().all())

    # ------------------------------------------------------------------
    # Aggregations
    # ------------------------------------------------------------------

    def stats(self, since: datetime | None = None) -> dict:
        """Aggregate stats: total scans, family distribution, avg confidence."""
        filters = []
        if since is not None:
            filters.append(ScanResult.scanned_at >= since)

        count_stmt = select(func.count()).select_from(ScanResult).where(*filters)
        total: int = self._session.execute(count_stmt).scalar() or 0

        if total == 0:
            return {
                "total_scans": 0,
                "family_distribution": {},
                "avg_confidence": 0.0,
            }

        avg_stmt = select(func.avg(ScanResult.confidence)).where(*filters)
        avg_conf: float = self._session.execute(avg_stmt).scalar() or 0.0

        family_stmt = (
            select(ScanResult.predicted_family, func.count())
            .where(*filters)
            .group_by(ScanResult.predicted_family)
        )
        rows = self._session.execute(family_stmt).all()
        family_dist = {family: count for family, count in rows}

        return {
            "total_scans": total,
            "family_distribution": family_dist,
            "avg_confidence": float(avg_conf),
        }
