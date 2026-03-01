"""api/schemas.py — Pydantic Request/Response Models for the Wintermute web API.

Covers all endpoints: scan, training, adversarial, pipeline, vault, and
the dashboard summary.  Defaults match the TUI config-drawer defaults
so callers can POST ``{}`` and get sensible behaviour.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Job lifecycle
# ---------------------------------------------------------------------------


class JobResponse(BaseModel):
    """Returned when a long-running task is accepted (HTTP 202)."""

    job_id: str
    poll_url: str


class JobStatus(BaseModel):
    """Generic polling response for any async job."""

    job_id: str
    status: str
    error: str | None = None


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


class DashboardResponse(BaseModel):
    """Aggregate metrics shown on the web-UI dashboard."""

    model_version: str = "3.0.0"
    f1: float = 0.0
    accuracy: float = 0.0
    vault_size: int = 0
    family_counts: dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------


class ScanResponse(BaseModel):
    """Status / result of a binary scan job."""

    job_id: str
    status: str
    result: dict | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class TrainingRequest(BaseModel):
    """Parameters for a training run (phase-A warm-up + phase-B full)."""

    epochs_phase_a: int = 5
    epochs_phase_b: int = 20
    learning_rate: float = 3e-4
    batch_size: int = 8
    max_seq_length: int = 2048
    num_classes: int = 2
    mlflow: bool = False
    experiment_name: str = "default"


class TrainingStatus(BaseModel):
    """Live progress of a training job."""

    job_id: str
    status: str
    epoch: int = 0
    phase: str = ""
    loss: float = 0.0
    train_acc: float = 0.0
    val_acc: float = 0.0
    f1: float = 0.0


# ---------------------------------------------------------------------------
# Adversarial
# ---------------------------------------------------------------------------


class AdversarialRequest(BaseModel):
    """Parameters for an adversarial-training loop."""

    cycles: int = 10
    episodes_per_cycle: int = 500
    trades_beta: float = 1.0
    ewc_lambda: float = 0.4
    ppo_lr: float = 3e-4
    ppo_epochs: int = 4


class AdversarialStatus(BaseModel):
    """Live progress of an adversarial-training job."""

    job_id: str
    status: str
    cycle: int = 0
    evasion_rate: float = 0.0
    adv_tpr: float = 0.0
    vault_size: int = 0


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class PipelineRequest(BaseModel):
    """Parameters for a data-pipeline operation (build / synthetic / pretrain)."""

    data_dir: str = "data"
    max_seq_length: int = 2048
    vocab_size: int | None = None
    n_samples: int = 500
    output_dir: str = "data/processed"
    seed: int = 42
    epochs: int = 50
    learning_rate: float = 3e-4
    batch_size: int = 8
    mask_prob: float = 0.15


class PipelineStatus(BaseModel):
    """Live progress of a pipeline operation."""

    job_id: str
    status: str
    operation: str = ""
    progress: float = 0.0
    message: str = ""


# ---------------------------------------------------------------------------
# Vault
# ---------------------------------------------------------------------------


class VaultSample(BaseModel):
    """Summary record for one adversarial sample in the vault."""

    id: str
    family: str
    confidence: float
    mutations: int
    cycle: int


class VaultSampleDetail(VaultSample):
    """Full detail view including raw byte diffs."""

    original_bytes: str = ""
    mutated_bytes: str = ""
    diff: str = ""


# ---------------------------------------------------------------------------
# Database-backed responses
# ---------------------------------------------------------------------------


class SampleResponse(BaseModel):
    """A cataloged malware sample."""

    sha256: str
    family: str
    label: int
    source: str
    file_type: str | None = None
    file_size_bytes: int | None = None
    opcode_count: int | None = None
    created_at: datetime | None = None


class ScanHistoryItem(BaseModel):
    """One row from the scan history."""

    id: str
    sha256: str
    filename: str | None = None
    predicted_family: str
    predicted_label: int
    confidence: float
    model_version: str
    scanned_at: datetime | None = None


class ModelResponse(BaseModel):
    """A registered model version."""

    id: str
    version: str
    architecture: str
    status: str
    best_val_macro_f1: float | None = None
    created_at: datetime | None = None
    promoted_at: datetime | None = None


class StatsResponse(BaseModel):
    """Aggregate system statistics."""

    total_samples: int = 0
    total_scans: int = 0
    total_models: int = 0
    families: dict[str, int] = Field(default_factory=dict)


class SimilarSampleResponse(BaseModel):
    """A sample returned from k-NN similarity search."""

    sha256: str
    family: str
    distance: float
