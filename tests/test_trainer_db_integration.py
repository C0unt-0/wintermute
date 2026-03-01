"""Tests for JointTrainer database integration.

Verifies that the JointTrainer correctly persists TrainingRun and Model
rows when a db_session is provided, and that training works without
errors when db_session is None or when the DB raises exceptions.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from sqlalchemy import select
from sqlalchemy.orm import Session

from wintermute.db.models import Model, TrainingRun


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_trainer(tmp_dir, db_session=None, epochs_b=2):
    """Create a JointTrainer with minimal synthetic data."""
    from wintermute.data.augment import SyntheticGenerator
    from wintermute.engine.joint_trainer import JointTrainer
    from wintermute.models.fusion import DetectorConfig

    tmp = Path(tmp_dir)
    SyntheticGenerator(n_samples=40, max_seq_length=32, seed=0).generate_dataset(str(tmp))
    (tmp / "graphs").mkdir(exist_ok=True)
    (tmp / "graph_index.json").write_text("{}")

    vocab = json.loads((tmp / "vocab.json").read_text())
    cfg = DetectorConfig(
        vocab_size=len(vocab),
        dims=32,
        num_heads=2,
        num_layers=1,
        mlp_dims=64,
        dropout=0.0,
        gat_layers=1,
        gat_heads=2,
        num_fusion_heads=2,
        num_classes=2,
        max_seq_length=32,
    )
    overrides = {
        "epochs_phase_a": 0,
        "epochs_phase_b": epochs_b,
        "batch_size": 8,
        "learning_rate": 3e-3,
        "val_ratio": 0.2,
        "mixup_prob": 0.0,
        "augment_prob": 0.0,
        "save_path": str(tmp / "model.safetensors"),
        "manifest_path": str(tmp / "manifest.json"),
    }
    return JointTrainer(cfg, tmp, overrides=overrides, db_session=db_session)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestTrainerDBIntegration:
    def test_trainer_creates_training_run(self, db_session: Session):
        """Verify that a TrainingRun row is created when db_session is provided."""
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _make_trainer(tmp, db_session=db_session, epochs_b=1)
            trainer.train()

        rows = db_session.execute(select(TrainingRun)).scalars().all()
        assert len(rows) == 1
        run = rows[0]
        assert run.id is not None
        assert run.num_classes == 2
        assert run.total_samples > 0
        assert run.train_split_size > 0
        assert run.val_split_size > 0

    def test_trainer_creates_model_after_training(self, db_session: Session):
        """Full flow: both TrainingRun and Model rows should exist after training."""
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _make_trainer(tmp, db_session=db_session, epochs_b=1)
            trainer.train()

        runs = db_session.execute(select(TrainingRun)).scalars().all()
        models = db_session.execute(select(Model)).scalars().all()

        assert len(runs) == 1
        assert len(models) == 1

        run = runs[0]
        model = models[0]

        # Model should reference the training run
        assert model.training_run_id == run.id
        assert model.architecture == "WintermuteMalwareDetector"
        assert model.status == "staged"
        assert model.vocab_size > 0
        assert model.num_classes == 2
        assert model.dims == 32
        assert model.max_seq_length == 32

        # Training run should link back to the model
        assert run.model_id == model.id
        assert run.completed_at is not None
        assert run.epochs_completed > 0

    def test_trainer_updates_best_metrics(self, db_session: Session):
        """Verify best_epoch and best_val_macro_f1 are updated during training."""
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _make_trainer(tmp, db_session=db_session, epochs_b=2)
            best_f1 = trainer.train()

        run = db_session.execute(select(TrainingRun)).scalars().first()
        assert run is not None
        assert run.best_epoch is not None
        assert run.best_epoch >= 1
        assert run.best_val_macro_f1 is not None
        assert run.best_val_macro_f1 > 0
        assert run.best_val_loss is None

        model = db_session.execute(select(Model)).scalars().first()
        assert model is not None
        assert model.best_val_macro_f1 == best_f1

    def test_trainer_works_without_db(self):
        """Pass db_session=None: training should complete without errors."""
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _make_trainer(tmp, db_session=None, epochs_b=1)
            f1 = trainer.train()

        assert f1 >= 0.0
        assert trainer._training_run_id is None

    def test_trainer_cancelled_no_model(self, db_session: Session):
        """Cancelled training should create TrainingRun but no Model row."""

        class CancellingHook:
            """Hook that cancels training after the first epoch."""

            def __init__(self):
                self.cancelled = False

            def on_epoch(self, epoch, phase, loss, val_loss, f1, val_f1, elapsed):
                self.cancelled = True

            def on_log(self, msg, level):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            trainer = _make_trainer(tmp, db_session=db_session, epochs_b=2)
            trainer._hook = CancellingHook()
            trainer.train()

        runs = db_session.execute(select(TrainingRun)).scalars().all()
        models = db_session.execute(select(Model)).scalars().all()

        assert len(runs) == 1
        assert len(models) == 0, "No Model should be created for a cancelled run"

        run = runs[0]
        assert run.completed_at is not None, "completed_at should be set even for cancelled runs"

    def test_trainer_db_error_no_crash(self):
        """Mock db_session to raise on add/flush: training should still complete."""
        mock_session = MagicMock()
        mock_session.flush.side_effect = RuntimeError("DB unavailable")
        mock_session.add.side_effect = RuntimeError("DB unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            trainer = _make_trainer(tmp, db_session=mock_session, epochs_b=1)
            f1 = trainer.train()

        # Training should still succeed despite DB errors
        assert f1 >= 0.0
        assert trainer._training_run_id is None
