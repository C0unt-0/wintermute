"""
api/main.py — Wintermute Threat Intelligence FastAPI Server

Acts purely as a fast traffic cop:
  POST /api/v1/scan         — Accept a raw binary, save it, enqueue analysis,
                              return a job_id in milliseconds (HTTP 202)
  GET  /api/v1/status/{id}  — Poll for the verdict of a queued or completed job

The heavy lifting is done asynchronously by the Celery worker in
src/wintermute/engine/worker.py.
"""

import os
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from celery.result import AsyncResult

from src.wintermute.engine.worker import analyze_binary_task, celery_app

app = FastAPI(
    title="Wintermute Threat Intelligence API",
    description=(
        "Asynchronous malware analysis pipeline. "
        "Submit a raw .exe or .elf and poll for the AI verdict."
    ),
    version="2.0.0",
)

UPLOAD_DIR = "/tmp/wintermute_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── Health Check ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
async def health_check():
    """Returns 200 OK when the API is running."""
    return {"status": "ok"}


# ── Scan Endpoint ─────────────────────────────────────────────────────────────

@app.post("/api/v1/scan", status_code=202, tags=["analysis"])
async def analyze_file(file: UploadFile = File(...)):
    """
    Upload a raw executable (.exe / .elf) for AI-powered threat analysis.

    The file is saved to a temporary directory and dispatched to a GPU-enabled
    Celery worker. The response contains a **job_id** you can poll.
    """
    job_id = str(uuid.uuid4())
    safe_filepath = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")

    # Persist the upload to disk so the worker can read it
    with open(safe_filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Dispatch to the Celery background worker — returns immediately
    task = analyze_binary_task.apply_async(args=[safe_filepath], task_id=job_id)

    return JSONResponse(
        status_code=202,
        content={
            "message": "File queued for analysis",
            "job_id": task.id,
            "poll_url": f"/api/v1/status/{task.id}",
        },
    )


# ── Status / Result Endpoint ──────────────────────────────────────────────────

@app.get("/api/v1/status/{job_id}", tags=["analysis"])
async def get_status(job_id: str):
    """
    Poll the status of a previously submitted analysis job.

    Possible states:
    - **PENDING_IN_QUEUE** — job waiting to be picked up by a worker
    - **DISASSEMBLING**    — Radare2 is reverse-engineering the binary
    - **INFERENCE**        — deep learning models are running
    - **COMPLETED**        — verdict is ready (see `result` field)
    - **FAILED**           — analysis encountered an error
    """
    task_result = AsyncResult(job_id, app=celery_app)

    if task_result.state == "PENDING":
        return {"job_id": job_id, "status": "PENDING_IN_QUEUE"}

    elif task_result.state in ("DISASSEMBLING", "INFERENCE"):
        return {
            "job_id": job_id,
            "status": task_result.state,
            "details": task_result.info,
        }

    elif task_result.state == "SUCCESS":
        return {"job_id": job_id, "result": task_result.result}

    elif task_result.state == "FAILURE":
        return {
            "job_id": job_id,
            "status": "FAILED",
            "error": str(task_result.info),
        }

    else:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
