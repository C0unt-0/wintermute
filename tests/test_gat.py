# tests/test_gat.py
import mlx.core as mx
import mlx.nn as nn
from wintermute.models.gat import GATLayer, GATEncoder


class TestGATLayer:
    def test_output_shape(self):
        layer = GATLayer(in_dims=32, out_dims=32, num_heads=2)
        h = mx.random.normal((4, 32))
        src, dst = mx.array([0, 1, 2, 3]), mx.array([1, 2, 3, 0])
        out = layer(h, src, dst)
        mx.eval(out)
        assert out.shape == (4, 32)

    def test_isolated_node_no_crash(self):
        layer = GATLayer(in_dims=16, out_dims=16, num_heads=1)
        h = mx.random.normal((4, 16))
        src, dst = mx.array([0, 1]), mx.array([1, 2])  # node 3 isolated
        out = layer(h, src, dst)
        mx.eval(out)
        assert out.shape == (4, 16)
        assert not mx.any(mx.isnan(out)).item()

    def test_gradient_flows_to_weight(self):
        """Critical: verifies the old GCN bug (raw mx.array weights) is fixed."""
        layer = GATLayer(in_dims=8, out_dims=8, num_heads=1)
        h = mx.random.normal((3, 8))
        src = mx.array([0, 1, 2])
        dst = mx.array([1, 2, 0])

        def loss_fn(model, h, src, dst):
            return mx.mean(model(h, src, dst))

        _, grads = nn.value_and_grad(layer, loss_fn)(layer, h, src, dst)
        mx.eval(grads)
        w_grad = grads["W"]["weight"]
        assert w_grad is not None
        assert not mx.all(w_grad == 0).item()


class TestGATEncoder:
    def test_single_graph(self):
        enc = GATEncoder(in_dims=32, hidden_dims=64, num_layers=2, num_heads=2)
        h = mx.random.normal((5, 32))
        src, dst = mx.array([0, 1, 2, 3]), mx.array([1, 2, 3, 4])
        batch_idx = mx.zeros((5,), dtype=mx.int32)
        out = enc(h, src, dst, batch_idx, n_graphs=1)
        mx.eval(out)
        assert out.shape == (1, 64)

    def test_two_graphs_disjoint_batch(self):
        enc = GATEncoder(in_dims=16, hidden_dims=32, num_layers=1, num_heads=1)
        h = mx.random.normal((5, 16))
        src = mx.array([0, 1, 3])       # graph 0: 0->1, graph 1: 3->4
        dst = mx.array([1, 2, 4])
        batch_idx = mx.array([0, 0, 0, 1, 1], dtype=mx.int32)
        out = enc(h, src, dst, batch_idx, n_graphs=2)
        mx.eval(out)
        assert out.shape == (2, 32)

    def test_no_edges(self):
        enc = GATEncoder(in_dims=8, hidden_dims=8, num_layers=1, num_heads=1)
        h = mx.random.normal((3, 8))
        src = mx.array([], dtype=mx.int32)
        dst = mx.array([], dtype=mx.int32)
        batch_idx = mx.zeros((3,), dtype=mx.int32)
        out = enc(h, src, dst, batch_idx, n_graphs=1)
        mx.eval(out)
        assert out.shape == (1, 8)

    def test_encoder_gradients_flow(self):
        enc = GATEncoder(in_dims=16, hidden_dims=16, num_layers=2, num_heads=2)
        h = mx.random.normal((4, 16))
        src, dst = mx.array([0, 1, 2, 3]), mx.array([1, 2, 3, 0])
        batch_idx = mx.zeros((4,), dtype=mx.int32)

        def loss_fn(model, h, src, dst, batch_idx):
            return mx.mean(model(h, src, dst, batch_idx, n_graphs=1))

        _, grads = nn.value_and_grad(enc, loss_fn)(enc, h, src, dst, batch_idx)
        mx.eval(grads)
        w_grad = grads["layers"][0]["W"]["weight"]
        assert not mx.all(w_grad == 0).item()
