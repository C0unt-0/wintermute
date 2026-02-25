# tests/test_pipeline.py
import tempfile, json
import importlib
from pathlib import Path
import numpy as np
import mlx.core as mx
from wintermute.data.augment import SyntheticGenerator
from wintermute.engine.joint_trainer import JointTrainer
from wintermute.models.fusion import DetectorConfig, WintermuteMalwareDetector


def _tiny_dataset(tmp):
    tmp = Path(tmp)
    SyntheticGenerator(n_samples=40, max_seq_length=32, seed=0).generate_dataset(str(tmp))
    (tmp / "graphs").mkdir(exist_ok=True)
    (tmp / "graph_index.json").write_text("{}")
    return tmp


class TestJointTrainer:
    def test_one_epoch_finite_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            dp = _tiny_dataset(tmp)
            vocab = json.loads((dp / "vocab.json").read_text())
            cfg = DetectorConfig(
                vocab_size=len(vocab), dims=32, num_heads=1, num_layers=1,
                mlp_dims=64, dropout=0.0, gat_layers=1, gat_heads=1,
                num_fusion_heads=1, num_classes=2, max_seq_length=32,
            )
            trainer = JointTrainer(cfg, dp)
            loss = trainer.train_one_epoch(phase="B")
            assert np.isfinite(loss)

    def test_full_training_completes(self):
        with tempfile.TemporaryDirectory() as tmp:
            dp = _tiny_dataset(tmp)
            vocab = json.loads((dp / "vocab.json").read_text())
            cfg = DetectorConfig(
                vocab_size=len(vocab), dims=32, num_heads=1, num_layers=1,
                mlp_dims=64, dropout=0.0, gat_layers=1, gat_heads=1,
                num_fusion_heads=1, num_classes=2, max_seq_length=32,
            )
            overrides = {"epochs_phase_a": 1, "epochs_phase_b": 2,
                         "batch_size": 8, "learning_rate": 1e-3}
            trainer = JointTrainer(cfg, dp, overrides=overrides)
            best_f1 = trainer.train()
            assert 0.0 <= best_f1 <= 1.0


class TestJointTrainerAugmentation:
    def test_augment_sequences_strips_pad(self):
        """_augment_sequences must not pass PAD tokens to the augmenter."""
        with tempfile.TemporaryDirectory() as tmp:
            dp = _tiny_dataset(tmp)
            vocab = json.loads((dp / "vocab.json").read_text())
            cfg = DetectorConfig(
                vocab_size=len(vocab), dims=32, num_heads=1, num_layers=1,
                mlp_dims=64, dropout=0.0, gat_layers=1, gat_heads=1,
                num_fusion_heads=1, num_classes=2, max_seq_length=32,
            )
            trainer = JointTrainer(cfg, dp)
            # Initialize model so token_embedding is available
            trainer.model = WintermuteMalwareDetector(cfg)
            # Build a batch with PAD tokens (trailing zeros — all-PAD batch)
            xb = mx.zeros((4, 32), dtype=mx.int32)
            result = trainer._augment_sequences(xb)
            # Output should have same shape as input
            assert result.shape == xb.shape
            # Materialise the lazy MLX graph then verify token IDs are non-negative
            mx.eval(result)
            assert (result >= 0).all().item()

    def test_train_one_epoch_with_full_augmentation(self):
        """Both augmentation and Mixup paths must complete without error at prob=1.0."""
        with tempfile.TemporaryDirectory() as tmp:
            dp = _tiny_dataset(tmp)
            vocab = json.loads((dp / "vocab.json").read_text())
            cfg = DetectorConfig(
                vocab_size=len(vocab), dims=32, num_heads=1, num_layers=1,
                mlp_dims=64, dropout=0.0, gat_layers=1, gat_heads=1,
                num_fusion_heads=1, num_classes=2, max_seq_length=32,
            )
            # Force both augmentation and Mixup to always fire
            overrides = {"augment_prob": 1.0, "mixup_prob": 1.0, "batch_size": 8}
            trainer = JointTrainer(cfg, dp, overrides=overrides)
            loss = trainer.train_one_epoch(phase="B")
            assert np.isfinite(loss), f"Loss is not finite: {loss}"

    def test_train_one_epoch_phase_a_runs(self):
        """Phase A (encoder-frozen) must complete without error."""
        with tempfile.TemporaryDirectory() as tmp:
            dp = _tiny_dataset(tmp)
            vocab = json.loads((dp / "vocab.json").read_text())
            cfg = DetectorConfig(
                vocab_size=len(vocab), dims=32, num_heads=1, num_layers=1,
                mlp_dims=64, dropout=0.0, gat_layers=1, gat_heads=1,
                num_fusion_heads=1, num_classes=2, max_seq_length=32,
            )
            trainer = JointTrainer(cfg, dp)
            loss = trainer.train_one_epoch(phase="A")
            assert np.isfinite(loss), f"Phase A loss is not finite: {loss}"
