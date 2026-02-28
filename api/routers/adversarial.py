"""api/routers/adversarial.py — Adversarial training job endpoints.

Start, poll, and cancel AdversarialOrchestrator runs.  Each run
executes in a background thread and pushes live cycle/episode events
over the WebSocket.
"""

from __future__ import annotations

import asyncio
import threading
import uuid

from fastapi import APIRouter, HTTPException

from api.schemas import AdversarialRequest, AdversarialStatus, JobResponse
from api.ws import ws_manager

router = APIRouter(prefix="/api/v1/adversarial", tags=["adversarial"])

# In-memory job store ---------------------------------------------------------
_jobs: dict[str, dict] = {}


# -- helpers ------------------------------------------------------------------


def _run_adversarial(
    job_id: str,
    config: AdversarialRequest,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Execute the adversarial loop inside a background thread."""
    from wintermute.engine.hooks import AdversarialHook

    def _callback(event: dict) -> None:
        asyncio.run_coroutine_threadsafe(ws_manager.broadcast(event), loop)
        # Update polling store with latest metrics from cycle-end events
        if event.get("type") == "adversarial_cycle_end":
            metrics = event.get("metrics", {})
            _jobs[job_id].update(
                {
                    "cycle": event.get("cycle", _jobs[job_id].get("cycle", 0)),
                    "evasion_rate": metrics.get(
                        "evasion_rate", _jobs[job_id].get("evasion_rate", 0.0)
                    ),
                    "vault_size": metrics.get("vault_size", _jobs[job_id].get("vault_size", 0)),
                }
            )
        # Store vault samples for the vault router
        if event.get("type") == "vault_sample_added":
            from api.routers.vault import _vault_samples

            sample = event.get("sample", {})
            if sample:
                _vault_samples.append(sample)

    hook = AdversarialHook(callback=_callback)
    _jobs[job_id]["hook"] = hook

    try:
        import json
        from pathlib import Path

        import numpy as np

        from wintermute.adversarial.orchestrator import AdversarialOrchestrator
        from wintermute.models.fusion import DetectorConfig, WintermuteMalwareDetector

        data_dir = Path("data/processed")
        manifest_path = Path("malware_detector_manifest.json")

        with open(data_dir / "vocab.json") as f:
            vocab = json.load(f)

        manifest_data = json.loads(manifest_path.read_text())
        detector_cfg = DetectorConfig(
            vocab_size=manifest_data.get("vocab_size", len(vocab)),
            num_classes=manifest_data.get("num_classes", 2),
        )
        detector = WintermuteMalwareDetector(detector_cfg)
        detector.load_weights("malware_detector.safetensors")

        # Build sample pool (malicious samples only)
        x_data = np.load(data_dir / "x_data.npy")
        y_data = np.load(data_dir / "y_data.npy")
        pool = [
            (x_data[i], int(y_data[i]), "unknown") for i in range(len(y_data)) if y_data[i] == 1
        ]

        orch = AdversarialOrchestrator(
            model=detector,
            vocab=vocab,
            sample_pool=pool,
            trades_beta=config.trades_beta,
            hook=hook,
        )

        for _cycle in range(config.cycles):
            if hook.cancelled:
                break
            orch.run_cycle(n_episodes=config.episodes_per_cycle)

        if hook.cancelled:
            _jobs[job_id]["status"] = "CANCELLED"
        else:
            _jobs[job_id]["status"] = "COMPLETE"
    except Exception as exc:
        _jobs[job_id]["status"] = "FAILED"
        _jobs[job_id]["error"] = str(exc)


# -- endpoints ----------------------------------------------------------------


@router.post("/start", response_model=JobResponse, status_code=202)
async def start_adversarial(config: AdversarialRequest) -> JobResponse:
    """Start an adversarial training job in a background thread."""
    job_id = str(uuid.uuid4())
    loop = asyncio.get_event_loop()

    _jobs[job_id] = {
        "status": "RUNNING",
        "cycle": 0,
        "evasion_rate": 0.0,
        "adv_tpr": 0.0,
        "vault_size": 0,
        "error": None,
    }

    thread = threading.Thread(
        target=_run_adversarial,
        args=(job_id, config, loop),
        daemon=True,
    )
    thread.start()

    return JobResponse(job_id=job_id, poll_url=f"/api/v1/adversarial/{job_id}/status")


@router.get("/{job_id}/status", response_model=AdversarialStatus)
async def get_adversarial_status(job_id: str) -> AdversarialStatus:
    """Poll adversarial job status."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    return AdversarialStatus(
        job_id=job_id,
        status=job["status"],
        cycle=job.get("cycle", 0),
        evasion_rate=job.get("evasion_rate", 0.0),
        adv_tpr=job.get("adv_tpr", 0.0),
        vault_size=job.get("vault_size", 0),
    )


@router.post("/{job_id}/cancel")
async def cancel_adversarial(job_id: str) -> dict:
    """Cancel an adversarial job via hook.cancel()."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    hook = job.get("hook")
    if hook is not None:
        hook.cancel()

    return {"job_id": job_id, "message": "Cancel signal sent."}
