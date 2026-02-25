# tests/test_pipeline.py
import tempfile, json
from pathlib import Path
import numpy as np
from wintermute.data.augment import SyntheticGenerator
from wintermute.engine.joint_trainer import JointTrainer
from wintermute.models.fusion import DetectorConfig


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
