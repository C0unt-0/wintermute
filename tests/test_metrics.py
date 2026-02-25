# tests/test_metrics.py
import numpy as np
from wintermute.engine.metrics import compute_macro_f1, compute_auc_roc, fpr_at_fnr_threshold
from wintermute.models.fusion import DetectorConfig, WintermuteMalwareDetector
import mlx.core as mx


def _model():
    cfg = DetectorConfig(vocab_size=16, dims=16, num_heads=1, num_layers=1,
                         mlp_dims=32, dropout=0.0, gat_layers=1, gat_heads=1,
                         num_fusion_heads=1, num_classes=2, max_seq_length=8)
    return WintermuteMalwareDetector(cfg)


class TestMacroF1:
    def test_returns_valid_float(self):
        m = _model()
        x = mx.zeros((10, 8), dtype=mx.int32)
        y = mx.array([0,1]*5)
        f1 = compute_macro_f1(m, x, y, batch_size=4, num_classes=2)
        assert 0.0 <= f1 <= 1.0

class TestAucRoc:
    def test_perfect(self):
        auc = compute_auc_roc(np.array([0.9,0.8,0.1,0.2]), np.array([1,1,0,0]))
        assert abs(auc - 1.0) < 1e-6

    def test_random_near_half(self):
        rng = np.random.default_rng(42)
        auc = compute_auc_roc(rng.random(1000), rng.integers(0,2,1000))
        assert 0.4 < auc < 0.6

class TestFprAtFnr:
    def test_zero_fnr_high_fpr(self):
        fpr = fpr_at_fnr_threshold(np.array([0.9,0.8,0.3,0.2]), np.array([1,1,0,0]), 0.0)
        assert fpr == 1.0

    def test_full_fnr_zero_fpr(self):
        fpr = fpr_at_fnr_threshold(np.array([0.9,0.8,0.3,0.2]), np.array([1,1,0,0]), 1.0)
        assert fpr == 0.0
