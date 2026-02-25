# tests/test_learning.py
"""Verifies the model actually learns (not just correct tensor shapes)."""
import tempfile, json
from pathlib import Path
from wintermute.data.augment import SyntheticGenerator
from wintermute.engine.joint_trainer import JointTrainer
from wintermute.models.fusion import DetectorConfig


def _trainer(tmp, epochs_b=5):
    tmp = Path(tmp)
    SyntheticGenerator(n_samples=200, max_seq_length=64, seed=0).generate_dataset(str(tmp))
    (tmp / "graphs").mkdir(exist_ok=True)
    (tmp / "graph_index.json").write_text("{}")
    vocab = json.loads((tmp / "vocab.json").read_text())
    cfg = DetectorConfig(vocab_size=len(vocab), dims=64, num_heads=2, num_layers=2,
                         mlp_dims=128, dropout=0.0, gat_layers=1, gat_heads=2,
                         num_fusion_heads=2, num_classes=2, max_seq_length=64)
    ov = {"epochs_phase_a": 0, "epochs_phase_b": epochs_b,
          "batch_size": 16, "learning_rate": 3e-3, "val_ratio": 0.2,
          "mixup_prob": 0.0, "augment_prob": 0.0}
    return JointTrainer(cfg, tmp, overrides=ov)


class TestModelLearns:
    def test_beats_chance_after_5_epochs(self):
        """Macro F1 > 0.55 (chance = 0.5) proves gradient flow is working."""
        with tempfile.TemporaryDirectory() as tmp:
            f1 = _trainer(tmp, epochs_b=5).train()
        assert f1 > 0.55, f"Expected F1 > 0.55, got {f1:.3f}"
