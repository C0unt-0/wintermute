"""api/routers/vault.py — Adversarial sample vault endpoints.

Lists and inspects adversarial samples collected during adversarial
training cycles.  The in-memory store is populated by the adversarial
router's WebSocket callback when ``vault_sample_added`` events fire.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException

from api.schemas import VaultSample, VaultSampleDetail

router = APIRouter(prefix="/api/v1/vault", tags=["vault"])

# In-memory vault store (populated by adversarial events) ---------------------
_vault_samples: list[dict] = []


@router.get("/samples", response_model=list[VaultSample])
async def list_vault_samples() -> list[VaultSample]:
    """List all adversarial samples in the vault."""
    results: list[VaultSample] = []
    for sample in _vault_samples:
        results.append(
            VaultSample(
                id=sample.get("id", str(uuid.uuid4())),
                family=sample.get("family", "unknown"),
                confidence=sample.get("evasion_confidence", sample.get("confidence", 0.0)),
                mutations=sample.get("n_mutations", sample.get("mutations", 0)),
                cycle=sample.get("epoch", sample.get("cycle", 0)),
            )
        )
    return results


@router.get("/samples/{sample_id}", response_model=VaultSampleDetail)
async def get_vault_sample(sample_id: str) -> VaultSampleDetail:
    """Get detailed view of a vault sample."""
    for sample in _vault_samples:
        sid = sample.get("id", "")
        if sid == sample_id:
            return VaultSampleDetail(
                id=sid,
                family=sample.get("family", "unknown"),
                confidence=sample.get("evasion_confidence", sample.get("confidence", 0.0)),
                mutations=sample.get("n_mutations", sample.get("mutations", 0)),
                cycle=sample.get("epoch", sample.get("cycle", 0)),
                original_bytes=sample.get("original_bytes", ""),
                mutated_bytes=sample.get("mutated_bytes", ""),
                diff=sample.get("diff", ""),
            )

    raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found.")
