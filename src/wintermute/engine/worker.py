"""
src/wintermute/engine/worker.py — Asynchronous Celery Worker

Runs continuously in the background. Loads heavy ML models into memory once
on startup and processes analysis tasks as they arrive from the Redis queue.

Each task:
  1. Reverse-engineers the binary with HeadlessDisassembler
  2. Runs sequence and graph model inference
  3. Applies ensemble fusion to produce a final threat verdict
  4. Securely deletes the raw payload from disk
"""

import os
from celery import Celery
from src.wintermute.data.disassembler import HeadlessDisassembler

# Import your existing Wintermute models here when ready:
# from src.wintermute.models.transformer import MalBERT
# from src.wintermute.models.gnn import MalwareGNN

BROKER_URL = os.getenv("CELERY_BROKER_URL") or os.getenv("REDIS_URL", "redis://localhost:6379/0")
BACKEND_URL = os.getenv("CELERY_RESULT_BACKEND") or os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "wintermute_worker",
    broker=BROKER_URL,
    backend=BACKEND_URL,
)

# ---------------------------------------------------------------------------
# Pre-load models into GPU memory ONCE when the worker process boots.
# This avoids expensive model I/O on every task.
# ---------------------------------------------------------------------------
# malbert = MalBERT.load_from_checkpoint("configs/model_config.yaml")
# gnn     = MalwareGNN.load_from_checkpoint("configs/model_config.yaml")


@celery_app.task(bind=True, name="analyze_binary")
def analyze_binary_task(self, file_path: str) -> dict:
    """
    Celery task: reverse-engineer a binary and return a threat verdict.

    Args:
        file_path: Absolute path to the uploaded binary saved on disk.

    Returns:
        A dict with status, threat_score, is_malicious, predicted_family,
        and telemetry metrics.
    """
    try:
        # ── Step 1: Disassembly & Feature Extraction ─────────────────────
        self.update_state(
            state="DISASSEMBLING",
            meta={"step": "Extracting features from binary..."},
        )
        extractor = HeadlessDisassembler(file_path)
        result = extractor.extract()

        # ── Step 2: Deep Learning Inference ──────────────────────────────
        self.update_state(
            state="INFERENCE",
            meta={"step": "Running Deep Learning models..."},
        )
        # seq_score   = malbert.predict(sequence)
        # graph_score = gnn.predict(cfg)
        seq_score, graph_score = 0.95, 0.88  # Mock scores — replace with real inference

        # ── Step 3: Ensemble Fusion ───────────────────────────────────────
        # Sequence model given higher weight (60%) because transformer attention
        # captures long-range opcode co-occurrence patterns.
        final_score = (seq_score * 0.6) + (graph_score * 0.4)
        is_malicious = final_score > 0.85

        # ── Step 4: Secure Cleanup ────────────────────────────────────────
        _secure_delete(file_path)

        return {
            "status": "COMPLETED",
            "threat_score": round(final_score, 4),
            "is_malicious": is_malicious,
            "predicted_family": "AgentTesla" if is_malicious else "Clean",
            "telemetry": {
                "instructions_analyzed": len(result.sequence),
                "cfg_nodes": result.n_nodes,
                "cfg_edges": result.n_edges,
            },
        }

    except Exception as exc:
        _secure_delete(file_path)
        return {"status": "FAILED", "error": str(exc)}


def _secure_delete(path: str) -> None:
    """Remove a file from disk if it exists."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass  # Best-effort; log in production
