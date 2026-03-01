"""api/routers/db_endpoints.py -- Database-backed API endpoints.

Exposes read-only views over the Wintermute sample catalog, scan history,
model registry, and embedding similarity search.
"""

from __future__ import annotations

import logging
import struct

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.dependencies import get_db
from api.schemas import (
    ModelResponse,
    SampleResponse,
    ScanHistoryItem,
    SimilarSampleResponse,
    StatsResponse,
)

logger = logging.getLogger("wintermute.api.db")

router = APIRouter(prefix="/api/v1", tags=["database"])


# ---------------------------------------------------------------------------
# GET /api/v1/stats
# ---------------------------------------------------------------------------


@router.get("/stats", response_model=StatsResponse)
def get_stats(session: Session = Depends(get_db)) -> StatsResponse:
    """Aggregate statistics: sample count, scan count, model count, families."""
    from wintermute.db.models import Model, Sample, ScanResult

    total_samples: int = session.execute(select(func.count()).select_from(Sample)).scalar() or 0
    total_scans: int = session.execute(select(func.count()).select_from(ScanResult)).scalar() or 0
    total_models: int = session.execute(select(func.count()).select_from(Model)).scalar() or 0

    family_rows = session.execute(
        select(func.coalesce(Sample.family, ""), func.count()).group_by(Sample.family)
    ).all()
    families = {family: count for family, count in family_rows}

    return StatsResponse(
        total_samples=total_samples,
        total_scans=total_scans,
        total_models=total_models,
        families=families,
    )


# ---------------------------------------------------------------------------
# GET /api/v1/samples/{sha256}
# ---------------------------------------------------------------------------


_SHA256_PATH = Path(..., min_length=64, max_length=64, pattern=r"^[a-fA-F0-9]{64}$")


@router.get("/samples/{sha256}", response_model=SampleResponse)
def get_sample(sha256: str = _SHA256_PATH, session: Session = Depends(get_db)) -> SampleResponse:
    """Look up a single sample by its SHA-256 hash."""
    from wintermute.db.repos.samples import SampleRepo

    repo = SampleRepo(session)
    sample = repo.get(sha256)
    if sample is None:
        raise HTTPException(status_code=404, detail=f"Sample '{sha256}' not found.")

    return SampleResponse(
        sha256=sample.sha256,
        family=sample.family,
        label=sample.label,
        source=sample.source,
        file_type=sample.file_type,
        file_size_bytes=sample.file_size_bytes,
        opcode_count=sample.opcode_count,
        created_at=sample.created_at,
    )


# ---------------------------------------------------------------------------
# GET /api/v1/scans
# ---------------------------------------------------------------------------


@router.get("/scans", response_model=list[ScanHistoryItem])
def list_scans(
    limit: int = Query(default=20, ge=1, le=200),
    sha256: str | None = Query(default=None),
    uncertain: bool = Query(default=False),
    session: Session = Depends(get_db),
) -> list[ScanHistoryItem]:
    """Recent scan history with optional filters."""
    from wintermute.db.repos.scans import ScanRepo

    repo = ScanRepo(session)

    if sha256 is not None:
        rows = repo.history(sha256, limit=limit)
    elif uncertain:
        rows = repo.uncertain(limit=limit)
    else:
        rows = repo.recent(limit=limit)

    return [
        ScanHistoryItem(
            id=r.id,
            sha256=r.sha256,
            filename=r.filename,
            predicted_family=r.predicted_family,
            predicted_label=r.predicted_label,
            confidence=r.confidence,
            model_version=r.model_version,
            scanned_at=r.scanned_at,
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# GET /api/v1/similar/{sha256}
# ---------------------------------------------------------------------------


@router.get("/similar/{sha256}", response_model=list[SimilarSampleResponse])
def find_similar(
    sha256: str = _SHA256_PATH,
    k: int = Query(default=5, ge=1, le=50),
    session: Session = Depends(get_db),
) -> list[SimilarSampleResponse]:
    """k-NN search for samples similar to the given SHA-256."""
    from wintermute.db.repos.embeddings import EmbeddingRepo
    from wintermute.db.repos.samples import SampleRepo

    sample_repo = SampleRepo(session)
    sample = sample_repo.get(sha256)
    if sample is None:
        raise HTTPException(status_code=404, detail=f"Sample '{sha256}' not found.")

    if sample.embedding is None:
        raise HTTPException(
            status_code=404,
            detail=f"Sample '{sha256}' has no embedding vector.",
        )

    if len(sample.embedding) % 4 != 0:
        raise HTTPException(
            status_code=500,
            detail=f"Sample '{sha256}' has a corrupt embedding.",
        )

    dim = len(sample.embedding) // 4
    query_vec = list(struct.unpack(f"{dim}f", sample.embedding))

    emb_repo = EmbeddingRepo(session)
    neighbours = emb_repo.find_nearest(query_vec, k=k + 1)

    # Exclude the query sample itself from results
    results = [n for n in neighbours if n["sha256"] != sha256][:k]

    return [
        SimilarSampleResponse(
            sha256=n["sha256"],
            family=n["family"],
            distance=n["distance"],
        )
        for n in results
    ]


# ---------------------------------------------------------------------------
# GET /api/v1/models
# ---------------------------------------------------------------------------


@router.get("/models", response_model=list[ModelResponse])
def list_models(session: Session = Depends(get_db)) -> list[ModelResponse]:
    """List all registered model versions, newest first."""
    from wintermute.db.repos.models_repo import ModelRepo

    repo = ModelRepo(session)
    rows = repo.history(limit=100)

    return [
        ModelResponse(
            id=m.id,
            version=m.version,
            architecture=m.architecture,
            status=m.status,
            best_val_macro_f1=m.best_val_macro_f1,
            created_at=m.created_at,
            promoted_at=m.promoted_at,
        )
        for m in rows
    ]
