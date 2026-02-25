# tests/test_fusion.py
import json, tempfile
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from wintermute.models.fusion import DetectorConfig, WintermuteMalwareDetector


def _cfg(num_classes=2):
    return DetectorConfig(
        vocab_size=32, dims=64, num_heads=2, num_layers=2,
        mlp_dims=128, dropout=0.0, gat_layers=2, gat_heads=2,
        num_fusion_heads=2, num_classes=num_classes, max_seq_length=16,
    )


class TestWintermuteMalwareDetector:
    def test_forward_with_graphs(self):
        model = WintermuteMalwareDetector(_cfg())
        seq = mx.zeros((2, 16), dtype=mx.int32)
        nf = mx.zeros((2, 3, 64))
        mask = mx.array([[True, True, True], [True, True, False]])
        out = model(seq, node_features=nf, node_mask=mask)
        mx.eval(out)
        assert out.shape == (2, 2)

    def test_forward_no_graphs(self):
        model = WintermuteMalwareDetector(_cfg())
        seq = mx.zeros((3, 16), dtype=mx.int32)
        out = model(seq)
        mx.eval(out)
        assert out.shape == (3, 2)

    def test_9_class(self):
        model = WintermuteMalwareDetector(_cfg(num_classes=9))
        out = model(mx.zeros((4, 16), dtype=mx.int32))
        mx.eval(out)
        assert out.shape == (4, 9)

    def test_bfloat16(self):
        model = WintermuteMalwareDetector(_cfg())
        WintermuteMalwareDetector.cast_to_bf16(model)
        out = model(mx.zeros((2, 16), dtype=mx.int32))
        mx.eval(out)
        assert out.dtype == mx.bfloat16

    def test_manifest_roundtrip(self):
        model = WintermuteMalwareDetector(_cfg())
        with tempfile.TemporaryDirectory() as tmp:
            wp = Path(tmp) / "model.safetensors"
            mp = Path(tmp) / "manifest.json"
            model.save_weights(str(wp))
            model.save_manifest(str(mp), vocab_sha256="abc", best_val_macro_f1=0.9)
            m = json.loads(mp.read_text())
        assert m["arch"] == "WintermuteMalwareDetector"
        assert m["vocab_sha256"] == "abc"

    def test_vocab_mismatch_raises(self):
        model = WintermuteMalwareDetector(_cfg())
        with tempfile.TemporaryDirectory() as tmp:
            wp = Path(tmp) / "model.safetensors"
            mp = Path(tmp) / "manifest.json"
            model.save_weights(str(wp))
            model.save_manifest(str(mp), vocab_sha256="correct")
            try:
                WintermuteMalwareDetector.load(str(wp), str(mp), vocab_sha256="wrong")
                assert False, "Should raise"
            except ValueError as e:
                assert "vocab" in str(e).lower()
