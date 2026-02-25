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


def _sparse_graph(B: int, D: int):
    """Returns (node_embs, edge_src, edge_dst, batch_idx) for a tiny 2-node-per-graph batch."""
    N = B * 2  # 2 nodes per graph
    node_embs = mx.random.normal((N, D))
    edge_src = mx.array([2 * i for i in range(B)], dtype=mx.int32)
    edge_dst = mx.array([2 * i + 1 for i in range(B)], dtype=mx.int32)
    batch_idx = mx.array([i for i in range(B) for _ in range(2)], dtype=mx.int32)
    return node_embs, edge_src, edge_dst, batch_idx


class TestWintermuteMalwareDetector:
    def test_forward_with_graphs(self):
        model = WintermuteMalwareDetector(_cfg())
        seq = mx.zeros((2, 16), dtype=mx.int32)
        nf, es, ed, bi = _sparse_graph(2, 64)
        out = model(seq, node_embs=nf, edge_src=es, edge_dst=ed, batch_idx=bi, n_graphs=2)
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
        assert m["num_fusion_heads"] == 2   # non-default field is preserved

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

    def test_gat_is_executed(self):
        """Verify GAT weights actually receive gradients when graph inputs are provided."""
        model = WintermuteMalwareDetector(_cfg())
        seq = mx.zeros((2, 16), dtype=mx.int32)
        nf, es, ed, bi = _sparse_graph(2, 64)

        def loss_fn(m, seq, nf, es, ed, bi):
            return mx.mean(m(seq, node_embs=nf, edge_src=es, edge_dst=ed, batch_idx=bi, n_graphs=2))

        _, grads = nn.value_and_grad(model, loss_fn)(model, seq, nf, es, ed, bi)
        mx.eval(grads)
        # If GAT were bypassed, gat_encoder weights would have zero gradients
        gat_grad = grads["gat_encoder"]["layers"][0]["W"]["weight"]
        assert not mx.all(gat_grad == 0).item(), "GAT encoder received no gradients — it was not called"
