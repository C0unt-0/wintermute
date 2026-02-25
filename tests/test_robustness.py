# tests/test_robustness.py
"""NOP insertion should not flip >20% of model predictions."""
import tempfile, json
from pathlib import Path
import mlx.core as mx
import numpy as np
from wintermute.data.augment import HeuristicAugmenter, SyntheticGenerator
from wintermute.data.tokenizer import encode_sequence
from wintermute.models.fusion import DetectorConfig, WintermuteMalwareDetector


class TestAugmentationRobustness:
    def test_nop_flip_rate_below_20pct(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            SyntheticGenerator(n_samples=100, max_seq_length=64, seed=1).generate_dataset(str(tmp))
            stoi = json.loads((tmp / "vocab.json").read_text())
            x_np = np.load(tmp / "x_data.npy")

        vocab_size = len(stoi)
        cfg = DetectorConfig(vocab_size=vocab_size, dims=32, num_heads=1, num_layers=1,
                             mlp_dims=64, dropout=0.0, gat_layers=1, gat_heads=1,
                             num_fusion_heads=1, num_classes=2, max_seq_length=64)
        mx.random.seed(42)
        model = WintermuteMalwareDetector(cfg)
        aug = HeuristicAugmenter(seed=42)
        itos = {v: k for k, v in stoi.items()}
        pad_id = stoi.get("<PAD>", 0)

        flips = 0
        n = min(50, len(x_np))
        for i in range(n):
            x = mx.array(x_np[i : i + 1])
            orig_pred = int(mx.argmax(model(x), axis=1).item())
            opcodes = [itos.get(int(t), "<UNK>") for t in x_np[i] if t != pad_id]
            aug_ids = encode_sequence(aug.augment_sequence(opcodes, ["nop"]), stoi, 64)
            aug_pred = int(mx.argmax(model(mx.array(aug_ids[np.newaxis, :])), axis=1).item())
            if orig_pred != aug_pred:
                flips += 1

        assert flips / n < 0.2, f"NOP flipped {flips/n:.0%} of predictions"
