# src/wintermute/models/gat.py
"""
gat.py -- Graph Attention Network encoder.

Replaces the broken GCNLayer/MalwareGNN (gnn.py).
All learnable params use nn.Linear so MLX tracks gradients correctly.
Uses sparse edge-index format instead of O(N^2) dense adjacency.
Grouped softmax via MLX .at[].add() scatter operations.
"""
from __future__ import annotations
import mlx.core as mx
import mlx.nn as nn


class GATLayer(nn.Module):
    """
    Single GAT layer.

    h_i' = ELU( sum_{j in N(i)} alpha_ij * W * h_j )
    alpha_ij = softmax_j( LeakyReLU( a_src(Wh_i) + a_dst(Wh_j) ) )
    """

    def __init__(self, in_dims: int, out_dims: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert out_dims % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = out_dims // num_heads
        self.out_dims = out_dims
        self.W = nn.Linear(in_dims, out_dims, bias=False)
        self.attn_src = nn.Linear(self.head_dim, 1, bias=False)
        self.attn_dst = nn.Linear(self.head_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, h: mx.array, src_idx: mx.array, dst_idx: mx.array) -> mx.array:
        N, H, D = h.shape[0], self.num_heads, self.head_dim
        h_proj = self.W(h).reshape(N, H, D)             # [N, H, D]
        E = src_idx.shape[0]
        if E == 0:
            return nn.elu(h_proj.reshape(N, self.out_dims))

        h_src = h_proj[src_idx]                          # [E, H, D]
        h_dst = h_proj[dst_idx]

        attn_s = self.attn_src(h_src.reshape(E * H, D)).reshape(E, H)
        attn_d = self.attn_dst(h_dst.reshape(E * H, D)).reshape(E, H)
        e = nn.leaky_relu(attn_s + attn_d, negative_slope=0.2)

        # Grouped softmax. Global max subtraction is shift-invariant within each
        # destination group (numerator and denominator shift by the same constant),
        # so attention weights are identical to per-node max. MLX has no scatter-max,
        # so global max is the correct pragmatic choice here.
        exp_e = mx.exp(e - mx.max(e, axis=0, keepdims=True))
        sum_exp = mx.zeros((N, H)).at[dst_idx].add(exp_e)
        alpha = self.dropout(exp_e / (sum_exp[dst_idx] + 1e-16))

        output = mx.zeros((N, H, D)).at[dst_idx].add(alpha[:, :, None] * h_src)
        # Residual: nodes with no incoming edges preserve their projected features
        # instead of being zeroed out by the scatter initialisation.
        return nn.elu(output.reshape(N, self.out_dims) + h_proj.reshape(N, self.out_dims))


class GATEncoder(nn.Module):
    """Stack of GAT layers with per-graph global mean pool."""

    def __init__(self, in_dims: int, hidden_dims: int = 128, num_layers: int = 3,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.layers = [GATLayer(in_dims, hidden_dims, num_heads, dropout)]
        for _ in range(num_layers - 1):
            self.layers.append(GATLayer(hidden_dims, hidden_dims, num_heads, dropout))
        self.norm = nn.LayerNorm(hidden_dims)

    def __call__(self, h: mx.array, src_idx: mx.array, dst_idx: mx.array,
                 batch_idx: mx.array, n_graphs: int) -> mx.array:
        for layer in self.layers:
            h = layer(h, src_idx, dst_idx)
        h = self.norm(h)
        # Scatter mean pool per graph
        out = mx.zeros((n_graphs, h.shape[-1])).at[batch_idx].add(h)
        counts = mx.zeros((n_graphs, 1)).at[batch_idx].add(mx.ones((h.shape[0], 1)))
        return out / mx.maximum(counts, 1.0)
