"""api/routers/dashboard.py — Dashboard summary endpoint.

Returns the latest model metrics and system status by reading
the malware_detector_manifest.json on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter

from api.schemas import DashboardResponse

router = APIRouter(prefix="/api/v1", tags=["dashboard"])

MANIFEST_PATH = Path("malware_detector_manifest.json")


@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard() -> DashboardResponse:
    """Return current model metrics and system status."""
    if not MANIFEST_PATH.exists():
        return DashboardResponse()

    try:
        data = json.loads(MANIFEST_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return DashboardResponse()

    return DashboardResponse(
        model_version=data.get("version", "3.0.0"),
        f1=data.get("best_val_macro_f1", 0.0),
        accuracy=data.get("accuracy", 0.0),
    )
