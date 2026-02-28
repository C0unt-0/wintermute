"""api/routers/training.py — Training job management endpoints.

Start, poll, and cancel JointTrainer runs.  Each run executes in a
background thread and pushes live epoch events over the WebSocket.
"""

from __future__ import annotations

import asyncio
import threading
import uuid

from fastapi import APIRouter, HTTPException

from api.schemas import JobResponse, TrainingRequest, TrainingStatus
from api.ws import ws_manager

router = APIRouter(prefix="/api/v1/training", tags=["training"])

# In-memory job store ---------------------------------------------------------
_jobs: dict[str, dict] = {}


# -- helpers ------------------------------------------------------------------


def _run_training(job_id: str, config: TrainingRequest, loop: asyncio.AbstractEventLoop) -> None:
    """Execute the training loop inside a background thread."""
    from wintermute.engine.hooks import TrainingHook

    def _callback(event: dict) -> None:
        asyncio.run_coroutine_threadsafe(ws_manager.broadcast(event), loop)
        # Mirror key fields into the job store for REST polling
        _jobs[job_id].update(
            {
                "epoch": event.get("epoch", _jobs[job_id].get("epoch", 0)),
                "phase": event.get("phase", _jobs[job_id].get("phase", "")),
                "loss": event.get("loss", _jobs[job_id].get("loss", 0.0)),
                "train_acc": event.get("train_acc", _jobs[job_id].get("train_acc", 0.0)),
                "val_acc": event.get("val_acc", _jobs[job_id].get("val_acc", 0.0)),
                "f1": event.get("f1", _jobs[job_id].get("f1", 0.0)),
            }
        )

    hook = TrainingHook(callback=_callback)
    _jobs[job_id]["hook"] = hook

    try:
        from wintermute.engine.joint_trainer import JointTrainer
        from wintermute.models.fusion import DetectorConfig

        import json
        from pathlib import Path

        data_dir = Path("data/processed")
        vocab = json.loads((data_dir / "vocab.json").read_text())

        overrides = {
            "epochs_phase_a": config.epochs_phase_a,
            "epochs_phase_b": config.epochs_phase_b,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
        }

        detector_config = DetectorConfig(
            vocab_size=len(vocab),
            num_classes=config.num_classes,
            max_seq_length=config.max_seq_length,
        )

        trainer = JointTrainer(
            detector_config,
            data_dir,
            overrides=overrides,
            hook=hook,
        )
        trainer.train()

        if hook.cancelled:
            _jobs[job_id]["status"] = "CANCELLED"
        else:
            _jobs[job_id]["status"] = "COMPLETE"
    except Exception as exc:
        _jobs[job_id]["status"] = "FAILED"
        _jobs[job_id]["error"] = str(exc)


# -- endpoints ----------------------------------------------------------------


@router.post("/start", response_model=JobResponse, status_code=202)
async def start_training(config: TrainingRequest) -> JobResponse:
    """Start a training job in a background thread."""
    job_id = str(uuid.uuid4())
    loop = asyncio.get_event_loop()

    _jobs[job_id] = {
        "status": "RUNNING",
        "epoch": 0,
        "phase": "",
        "loss": 0.0,
        "train_acc": 0.0,
        "val_acc": 0.0,
        "f1": 0.0,
        "error": None,
    }

    thread = threading.Thread(
        target=_run_training,
        args=(job_id, config, loop),
        daemon=True,
    )
    thread.start()

    return JobResponse(job_id=job_id, poll_url=f"/api/v1/training/{job_id}/status")


@router.get("/{job_id}/status", response_model=TrainingStatus)
async def get_training_status(job_id: str) -> TrainingStatus:
    """Poll training job status."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    return TrainingStatus(
        job_id=job_id,
        status=job["status"],
        epoch=job.get("epoch", 0),
        phase=job.get("phase", ""),
        loss=job.get("loss", 0.0),
        train_acc=job.get("train_acc", 0.0),
        val_acc=job.get("val_acc", 0.0),
        f1=job.get("f1", 0.0),
    )


@router.post("/{job_id}/cancel")
async def cancel_training(job_id: str) -> dict:
    """Cancel a training job via hook.cancel()."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    hook = job.get("hook")
    if hook is not None:
        hook.cancel()

    return {"job_id": job_id, "message": "Cancel signal sent."}
