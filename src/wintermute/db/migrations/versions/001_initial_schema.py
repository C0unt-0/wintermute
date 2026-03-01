"""Initial schema — all eight Wintermute tables.

Revision ID: 001
Revises:
Create Date: 2026-03-01
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # 1. etl_runs (no FK dependencies)
    # ------------------------------------------------------------------
    op.create_table(
        "etl_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("config_hash", sa.String(64), nullable=True),
        sa.Column("config", sa.JSON, nullable=True),
        sa.Column("total_samples", sa.Integer, nullable=True),
        sa.Column("vocab_size", sa.Integer, nullable=True),
        sa.Column("num_classes", sa.Integer, nullable=True),
        sa.Column("output_dir", sa.String(500), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("elapsed_seconds", sa.Float, nullable=True),
    )

    # ------------------------------------------------------------------
    # 2. etl_run_sources (FK -> etl_runs)
    # ------------------------------------------------------------------
    op.create_table(
        "etl_run_sources",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "etl_run_id",
            sa.String(36),
            sa.ForeignKey("etl_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("source_name", sa.String(100), nullable=False),
        sa.Column("samples_extracted", sa.Integer, nullable=False, server_default="0"),
        sa.Column("samples_skipped", sa.Integer, nullable=False, server_default="0"),
        sa.Column("samples_failed", sa.Integer, nullable=False, server_default="0"),
        sa.Column("families_found", sa.JSON, nullable=True),
        sa.Column("errors", sa.JSON, nullable=True),
        sa.Column("elapsed_seconds", sa.Float, nullable=True),
    )
    op.create_index("ix_etl_run_sources_etl_run_id", "etl_run_sources", ["etl_run_id"])

    # ------------------------------------------------------------------
    # 3. samples (FK -> etl_runs)
    # ------------------------------------------------------------------
    op.create_table(
        "samples",
        sa.Column("sha256", sa.String(64), primary_key=True),
        sa.Column("family", sa.String(100), nullable=False, server_default=""),
        sa.Column("label", sa.Integer, nullable=False),
        sa.Column("source", sa.String(100), nullable=False),
        sa.Column("opcode_count", sa.Integer, nullable=True),
        sa.Column("file_type", sa.String(20), nullable=True),
        sa.Column("file_size_bytes", sa.Integer, nullable=True),
        sa.Column("embedding", sa.LargeBinary, nullable=True),
        sa.Column(
            "etl_run_id",
            sa.String(36),
            sa.ForeignKey("etl_runs.id"),
            nullable=True,
        ),
        sa.Column("metadata", sa.JSON, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_samples_family", "samples", ["family"])
    op.create_index("ix_samples_source", "samples", ["source"])
    op.create_index("ix_samples_label", "samples", ["label"])
    op.create_index("ix_samples_created_at", "samples", ["created_at"])
    op.create_index("ix_samples_etl_run_id", "samples", ["etl_run_id"])

    # ------------------------------------------------------------------
    # 4. adversarial_cycles (FK -> models, but models not created yet;
    #    we create cycles first with a deferred FK so that models can
    #    reference training_runs which references cycles)
    # ------------------------------------------------------------------
    # Note: models and training_runs have circular FKs (models.training_run_id
    # -> training_runs.id, training_runs.model_id -> models.id).
    # adversarial_cycles.defender_model_id -> models.id.
    # We create tables first without cross-FKs, then add them.
    # ------------------------------------------------------------------

    # 4a. adversarial_cycles (create without defender_model_id FK initially)
    op.create_table(
        "adversarial_cycles",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("cycle_number", sa.Integer, nullable=True),
        sa.Column("defender_model_id", sa.String(36), nullable=True),
        sa.Column("episodes_played", sa.Integer, nullable=True),
        sa.Column("total_evasions", sa.Integer, nullable=False, server_default="0"),
        sa.Column("evasion_rate", sa.Float, nullable=True),
        sa.Column("mean_confidence_drop", sa.Float, nullable=True),
        sa.Column("retrained", sa.Boolean, nullable=False, server_default="0"),
        sa.Column("vault_samples_used", sa.Integer, nullable=False, server_default="0"),
        sa.Column("defender_f1_before", sa.Float, nullable=True),
        sa.Column("defender_f1_after", sa.Float, nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )

    # ------------------------------------------------------------------
    # 5. training_runs (FK -> adversarial_cycles; model_id FK deferred)
    # ------------------------------------------------------------------
    op.create_table(
        "training_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("model_id", sa.String(36), nullable=True),
        sa.Column("config", sa.JSON, nullable=True),
        sa.Column("pretrained_weights", sa.String(500), nullable=True),
        sa.Column("dataset_sha256", sa.String(64), nullable=True),
        sa.Column("total_samples", sa.Integer, nullable=True),
        sa.Column("num_classes", sa.Integer, nullable=True),
        sa.Column("train_split_size", sa.Integer, nullable=True),
        sa.Column("val_split_size", sa.Integer, nullable=True),
        sa.Column("epochs_completed", sa.Integer, nullable=False, server_default="0"),
        sa.Column("best_epoch", sa.Integer, nullable=True),
        sa.Column("best_val_loss", sa.Float, nullable=True),
        sa.Column("best_val_macro_f1", sa.Float, nullable=True),
        sa.Column("best_val_accuracy", sa.Float, nullable=True),
        sa.Column("is_adversarial_retrain", sa.Boolean, nullable=False, server_default="0"),
        sa.Column(
            "adversarial_cycle_id",
            sa.String(36),
            sa.ForeignKey("adversarial_cycles.id"),
            nullable=True,
        ),
        sa.Column("vault_samples_mixed", sa.Integer, nullable=False, server_default="0"),
        sa.Column("trades_beta_final", sa.Float, nullable=True),
        sa.Column("ewc_lambda", sa.Float, nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("elapsed_seconds", sa.Float, nullable=True),
    )

    # ------------------------------------------------------------------
    # 6. models (FK -> training_runs, self-FK -> models)
    # ------------------------------------------------------------------
    op.create_table(
        "models",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("version", sa.String(50), unique=True, nullable=False),
        sa.Column(
            "architecture",
            sa.String(100),
            nullable=False,
            server_default="WintermuteMalwareDetector",
        ),
        sa.Column("weights_path", sa.String(500), nullable=False),
        sa.Column("manifest_path", sa.String(500), nullable=False, server_default=""),
        sa.Column("onnx_path", sa.String(500), nullable=True),
        sa.Column("vocab_size", sa.Integer, nullable=False),
        sa.Column("num_classes", sa.Integer, nullable=False),
        sa.Column("dims", sa.Integer, nullable=False),
        sa.Column("max_seq_length", sa.Integer, nullable=False, server_default="2048"),
        sa.Column("vocab_sha256", sa.String(64), nullable=True),
        sa.Column("config", sa.JSON, nullable=True),
        sa.Column(
            "training_run_id",
            sa.String(36),
            sa.ForeignKey("training_runs.id"),
            nullable=True,
        ),
        sa.Column(
            "parent_model_id",
            sa.String(36),
            sa.ForeignKey("models.id"),
            nullable=True,
        ),
        sa.Column("pretrained_from", sa.String(200), nullable=True),
        sa.Column("best_val_macro_f1", sa.Float, nullable=True),
        sa.Column("best_val_accuracy", sa.Float, nullable=True),
        sa.Column("best_val_auc_roc", sa.Float, nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="staged"),
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("retired_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_models_status", "models", ["status"])
    op.create_index("ix_models_architecture", "models", ["architecture"])

    # ------------------------------------------------------------------
    # 7. Add deferred FKs now that models exists
    # ------------------------------------------------------------------
    # training_runs.model_id -> models.id
    with op.batch_alter_table("training_runs") as batch_op:
        batch_op.create_foreign_key("fk_training_runs_model_id", "models", ["model_id"], ["id"])

    # adversarial_cycles.defender_model_id -> models.id
    with op.batch_alter_table("adversarial_cycles") as batch_op:
        batch_op.create_foreign_key(
            "fk_adversarial_cycles_defender_model_id",
            "models",
            ["defender_model_id"],
            ["id"],
        )

    # ------------------------------------------------------------------
    # 8. scan_results (FK -> models)
    # ------------------------------------------------------------------
    op.create_table(
        "scan_results",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("filename", sa.String(255), nullable=True),
        sa.Column("file_size_bytes", sa.Integer, nullable=True),
        sa.Column("predicted_family", sa.String(100), nullable=False),
        sa.Column("predicted_label", sa.Integer, nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("probabilities", sa.JSON, nullable=False),
        sa.Column(
            "model_id",
            sa.String(36),
            sa.ForeignKey("models.id"),
            nullable=True,
        ),
        sa.Column("model_version", sa.String(50), nullable=False),
        sa.Column("nearest_neighbors", sa.JSON, nullable=True),
        sa.Column("execution_time_ms", sa.Float, nullable=True),
        sa.Column("source_ip", sa.String(45), nullable=True),
        sa.Column("scanned_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_scan_results_sha256", "scan_results", ["sha256"])
    op.create_index("ix_scan_results_predicted_family", "scan_results", ["predicted_family"])
    op.create_index("ix_scan_results_confidence", "scan_results", ["confidence"])
    op.create_index("ix_scan_results_scanned_at", "scan_results", ["scanned_at"])
    op.create_index("ix_scan_results_model_id", "scan_results", ["model_id"])

    # ------------------------------------------------------------------
    # 9. adversarial_variants (FK -> samples, adversarial_cycles, training_runs)
    # ------------------------------------------------------------------
    op.create_table(
        "adversarial_variants",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "parent_sha256",
            sa.String(64),
            sa.ForeignKey("samples.sha256"),
            nullable=False,
        ),
        sa.Column(
            "cycle_id",
            sa.String(36),
            sa.ForeignKey("adversarial_cycles.id"),
            nullable=False,
        ),
        sa.Column("mutated_token_ids", sa.JSON, nullable=False),
        sa.Column("mutations_applied", sa.JSON, nullable=True),
        sa.Column("mutation_count", sa.Integer, nullable=False),
        sa.Column("modification_pct", sa.Float, nullable=False),
        sa.Column("confidence_before", sa.Float, nullable=False),
        sa.Column("confidence_after", sa.Float, nullable=False),
        sa.Column("confidence_delta", sa.Float, nullable=False),
        sa.Column("achieved_evasion", sa.Boolean, nullable=False, server_default="0"),
        sa.Column("used_in_retraining", sa.Boolean, nullable=False, server_default="0"),
        sa.Column(
            "retraining_run_id",
            sa.String(36),
            sa.ForeignKey("training_runs.id"),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_adversarial_variants_parent_sha256",
        "adversarial_variants",
        ["parent_sha256"],
    )
    op.create_index("ix_adversarial_variants_cycle_id", "adversarial_variants", ["cycle_id"])
    op.create_index(
        "ix_adversarial_variants_achieved_evasion",
        "adversarial_variants",
        ["achieved_evasion"],
    )
    op.create_index(
        "ix_adversarial_variants_used_in_retraining",
        "adversarial_variants",
        ["used_in_retraining"],
    )


def downgrade() -> None:
    # Drop in reverse dependency order.
    op.drop_table("adversarial_variants")
    op.drop_table("scan_results")

    # Remove deferred FKs before dropping models.
    with op.batch_alter_table("adversarial_cycles") as batch_op:
        batch_op.drop_constraint("fk_adversarial_cycles_defender_model_id", type_="foreignkey")
    with op.batch_alter_table("training_runs") as batch_op:
        batch_op.drop_constraint("fk_training_runs_model_id", type_="foreignkey")

    op.drop_table("models")
    op.drop_table("training_runs")
    op.drop_table("adversarial_cycles")
    op.drop_table("samples")
    op.drop_table("etl_run_sources")
    op.drop_table("etl_runs")
