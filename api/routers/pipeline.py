"""api/routers/pipeline.py — Data pipeline operation endpoints.

Start, poll, and cancel pipeline operations (build / synthetic / pretrain).
Each operation runs in a background thread and pushes progress events
over the WebSocket.
"""

from __future__ import annotations

import asyncio
import threading
import uuid

from fastapi import APIRouter, HTTPException

from api.schemas import JobResponse, PipelineRequest, PipelineStatus
from api.ws import ws_manager

router = APIRouter(prefix="/api/v1/pipeline", tags=["pipeline"])

# In-memory job store ---------------------------------------------------------
_jobs: dict[str, dict] = {}

_VALID_OPERATIONS = {"build", "synthetic", "pretrain"}


# -- background runners -------------------------------------------------------


def _run_synthetic(
    job_id: str,
    config: PipelineRequest,
    hook: object,
) -> None:
    """Run synthetic data generation."""
    from wintermute.data.augment import SyntheticGenerator

    gen = SyntheticGenerator(
        n_samples=config.n_samples,
        max_seq_length=config.max_seq_length,
        seed=config.seed,
    )
    gen.generate_dataset(out_dir=config.output_dir)


def _run_pretrain(
    job_id: str,
    config: PipelineRequest,
    hook: object,
) -> None:
    """Run MalBERT MLM pre-training."""
    from wintermute.engine.pretrain import MLMPretrainer

    overrides = {
        "pretrain": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "mask_prob": config.mask_prob,
        },
    }
    pretrainer = MLMPretrainer(overrides=overrides, hook=hook)
    pretrainer.pretrain(data_dir=config.output_dir)


def _run_build(
    job_id: str,
    config: PipelineRequest,
    hook: object,
) -> None:
    """Run the tokenizer build pipeline."""
    from pathlib import Path

    import json
    import numpy as np

    from wintermute.data.tokenizer import (
        build_vocabulary,
        collect_pe_files,
        encode_sequence,
        extract_opcodes_pe,
    )

    data_path = Path(config.data_dir)
    out_dir = data_path / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    filepaths, labels = collect_pe_files(data_path)
    if not filepaths:
        raise RuntimeError("No PE files found in data/raw/safe or data/raw/malicious.")

    total = len(filepaths)
    all_opcodes: list[list[str]] = []
    for i, fp in enumerate(filepaths, 1):
        ops = extract_opcodes_pe(fp)
        all_opcodes.append(ops)
        if hasattr(hook, "on_progress"):
            hook.on_progress("build", i / total, f"Extracted {i}/{total}")

    stoi = build_vocabulary(all_opcodes)
    x_data = np.stack([encode_sequence(ops, stoi, config.max_seq_length) for ops in all_opcodes])
    y_data = np.array(labels, dtype=np.int32)

    np.save(out_dir / "x_data.npy", x_data)
    np.save(out_dir / "y_data.npy", y_data)
    with open(out_dir / "vocab.json", "w") as f:
        json.dump(stoi, f, indent=2)


_RUNNERS = {
    "synthetic": _run_synthetic,
    "pretrain": _run_pretrain,
    "build": _run_build,
}


def _run_pipeline(
    job_id: str,
    operation: str,
    config: PipelineRequest,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Execute a pipeline operation inside a background thread."""
    from wintermute.engine.hooks import PipelineHook

    def _callback(event: dict) -> None:
        asyncio.run_coroutine_threadsafe(ws_manager.broadcast(event), loop)
        _jobs[job_id].update(
            {
                "operation": event.get("operation", _jobs[job_id].get("operation", "")),
                "progress": event.get("progress", _jobs[job_id].get("progress", 0.0)),
                "message": event.get("message", _jobs[job_id].get("message", "")),
            }
        )

    hook = PipelineHook(callback=_callback)
    _jobs[job_id]["hook"] = hook

    try:
        runner = _RUNNERS[operation]
        runner(job_id, config, hook)

        if hook.cancelled:
            _jobs[job_id]["status"] = "CANCELLED"
        else:
            _jobs[job_id]["status"] = "COMPLETE"
    except Exception as exc:
        _jobs[job_id]["status"] = "FAILED"
        _jobs[job_id]["error"] = str(exc)


# -- endpoints ----------------------------------------------------------------


@router.post("/{operation}", response_model=JobResponse, status_code=202)
async def start_pipeline(operation: str, config: PipelineRequest) -> JobResponse:
    """Start a pipeline operation (build/synthetic/pretrain)."""
    if operation not in _VALID_OPERATIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid operation '{operation}'. Must be one of: {', '.join(sorted(_VALID_OPERATIONS))}",
        )

    job_id = str(uuid.uuid4())
    loop = asyncio.get_event_loop()

    _jobs[job_id] = {
        "status": "RUNNING",
        "operation": operation,
        "progress": 0.0,
        "message": "",
        "error": None,
    }

    thread = threading.Thread(
        target=_run_pipeline,
        args=(job_id, operation, config, loop),
        daemon=True,
    )
    thread.start()

    return JobResponse(job_id=job_id, poll_url=f"/api/v1/pipeline/{job_id}/status")


@router.get("/{job_id}/status", response_model=PipelineStatus)
async def get_pipeline_status(job_id: str) -> PipelineStatus:
    """Poll pipeline job status."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    return PipelineStatus(
        job_id=job_id,
        status=job["status"],
        operation=job.get("operation", ""),
        progress=job.get("progress", 0.0),
        message=job.get("message", ""),
    )


@router.post("/{job_id}/cancel")
async def cancel_pipeline(job_id: str) -> dict:
    """Cancel a pipeline job via hook.cancel()."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    hook = job.get("hook")
    if hook is not None:
        hook.cancel()

    return {"job_id": job_id, "message": "Cancel signal sent."}
