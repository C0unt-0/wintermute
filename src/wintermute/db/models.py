"""ORM models for the Wintermute database layer.

Defines all seven tables:
  samples, scan_results, models, training_runs,
  adversarial_cycles, adversarial_variants,
  etl_runs, etl_run_sources
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    LargeBinary,
    String,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """Base class for all Wintermute ORM models."""


# ---------------------------------------------------------------------------
# 1. Sample
# ---------------------------------------------------------------------------

class Sample(Base):
    __tablename__ = "samples"

    sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    family: Mapped[str | None] = mapped_column(String(100), nullable=True)
    label: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source: Mapped[str | None] = mapped_column(String(50), nullable=True)
    opcode_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    file_type: Mapped[str | None] = mapped_column(String(20), nullable=True)
    file_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    etl_run_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("etl_runs.id"), nullable=True
    )
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)

    __table_args__ = (
        Index("ix_samples_family", "family"),
        Index("ix_samples_source", "source"),
        Index("ix_samples_label", "label"),
        Index("ix_samples_created_at", "created_at"),
        Index("ix_samples_etl_run_id", "etl_run_id"),
    )


# ---------------------------------------------------------------------------
# 2. ScanResult
# ---------------------------------------------------------------------------

class ScanResult(Base):
    __tablename__ = "scan_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    file_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    predicted_family: Mapped[str | None] = mapped_column(String(100), nullable=True)
    predicted_label: Mapped[int | None] = mapped_column(Integer, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    probabilities: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    model_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("models.id"), nullable=True
    )
    model_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    nearest_neighbors: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    execution_time_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    source_ip: Mapped[str | None] = mapped_column(String(45), nullable=True)
    scanned_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    __table_args__ = (
        Index("ix_scan_results_sha256", "sha256"),
        Index("ix_scan_results_predicted_family", "predicted_family"),
        Index("ix_scan_results_confidence", "confidence"),
        Index("ix_scan_results_scanned_at", "scanned_at"),
        Index("ix_scan_results_model_id", "model_id"),
    )


# ---------------------------------------------------------------------------
# 3. Model
# ---------------------------------------------------------------------------

class Model(Base):
    __tablename__ = "models"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    version: Mapped[str] = mapped_column(String(50), unique=True)
    architecture: Mapped[str] = mapped_column(
        String(100), default="WintermuteMalwareDetector"
    )
    weights_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    manifest_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    onnx_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    vocab_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    num_classes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dims: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_seq_length: Mapped[int] = mapped_column(Integer, default=2048)
    vocab_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    training_run_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("training_runs.id"), nullable=True
    )
    parent_model_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("models.id"), nullable=True
    )
    pretrained_from: Mapped[str | None] = mapped_column(String(200), nullable=True)
    best_val_macro_f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_val_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_val_auc_roc: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="staged")
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    retired_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    __table_args__ = (
        Index("ix_models_status", "status"),
        Index("ix_models_architecture", "architecture"),
    )


# ---------------------------------------------------------------------------
# 4. TrainingRun
# ---------------------------------------------------------------------------

class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    model_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("models.id"), nullable=True
    )
    config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    pretrained_weights: Mapped[str | None] = mapped_column(String(500), nullable=True)
    dataset_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    total_samples: Mapped[int | None] = mapped_column(Integer, nullable=True)
    num_classes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    train_split_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    val_split_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    epochs_completed: Mapped[int] = mapped_column(Integer, default=0)
    best_epoch: Mapped[int | None] = mapped_column(Integer, nullable=True)
    best_val_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_val_macro_f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_val_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_adversarial_retrain: Mapped[bool] = mapped_column(Boolean, default=False)
    adversarial_cycle_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("adversarial_cycles.id"), nullable=True
    )
    vault_samples_mixed: Mapped[int] = mapped_column(Integer, default=0)
    trades_beta_final: Mapped[float | None] = mapped_column(Float, nullable=True)
    ewc_lambda: Mapped[float | None] = mapped_column(Float, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    elapsed_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)


# ---------------------------------------------------------------------------
# 5. AdversarialCycle
# ---------------------------------------------------------------------------

class AdversarialCycle(Base):
    __tablename__ = "adversarial_cycles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    cycle_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    defender_model_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("models.id"), nullable=True
    )
    episodes_played: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_evasions: Mapped[int] = mapped_column(Integer, default=0)
    evasion_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_confidence_drop: Mapped[float | None] = mapped_column(Float, nullable=True)
    retrained: Mapped[bool] = mapped_column(Boolean, default=False)
    vault_samples_used: Mapped[int] = mapped_column(Integer, default=0)
    defender_f1_before: Mapped[float | None] = mapped_column(Float, nullable=True)
    defender_f1_after: Mapped[float | None] = mapped_column(Float, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    variants: Mapped[list[AdversarialVariant]] = relationship(
        "AdversarialVariant", back_populates="cycle"
    )


# ---------------------------------------------------------------------------
# 6. AdversarialVariant
# ---------------------------------------------------------------------------

class AdversarialVariant(Base):
    __tablename__ = "adversarial_variants"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    parent_sha256: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("samples.sha256"), nullable=True
    )
    cycle_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("adversarial_cycles.id"), nullable=True
    )
    mutated_token_ids: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    mutations_applied: Mapped[list | None] = mapped_column(JSON, default=list)
    mutation_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    modification_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_before: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_after: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_delta: Mapped[float | None] = mapped_column(Float, nullable=True)
    achieved_evasion: Mapped[bool] = mapped_column(Boolean, default=False)
    used_in_retraining: Mapped[bool] = mapped_column(Boolean, default=False)
    retraining_run_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("training_runs.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    cycle: Mapped[AdversarialCycle | None] = relationship(
        "AdversarialCycle", back_populates="variants"
    )

    __table_args__ = (
        Index("ix_adversarial_variants_parent_sha256", "parent_sha256"),
        Index("ix_adversarial_variants_cycle_id", "cycle_id"),
        Index("ix_adversarial_variants_achieved_evasion", "achieved_evasion"),
        Index("ix_adversarial_variants_used_in_retraining", "used_in_retraining"),
    )


# ---------------------------------------------------------------------------
# 7. EtlRun + EtlRunSource
# ---------------------------------------------------------------------------

class EtlRun(Base):
    __tablename__ = "etl_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    config_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    total_samples: Mapped[int | None] = mapped_column(Integer, nullable=True)
    vocab_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    num_classes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_dir: Mapped[str | None] = mapped_column(String(500), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    elapsed_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    sources: Mapped[list[EtlRunSource]] = relationship(
        "EtlRunSource",
        back_populates="etl_run",
        cascade="all, delete-orphan",
    )


class EtlRunSource(Base):
    __tablename__ = "etl_run_sources"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    etl_run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("etl_runs.id", ondelete="CASCADE")
    )
    source_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    samples_extracted: Mapped[int | None] = mapped_column(Integer, nullable=True)
    samples_skipped: Mapped[int | None] = mapped_column(Integer, nullable=True)
    samples_failed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    families_found: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    errors: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    elapsed_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    etl_run: Mapped[EtlRun] = relationship("EtlRun", back_populates="sources")

    __table_args__ = (
        Index("ix_etl_run_sources_etl_run_id", "etl_run_id"),
    )
