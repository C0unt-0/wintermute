"""
gnn.py — Wintermute Graph Neural Network (GNN) Architecture

Implements GCN (Graph Convolutional Network) layers directly in MLX to
analyze malware represented as Control Flow Graphs (CFG).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class GCNLayer(nn.Module):
    """
    A Graph Convolutional Network (GCN) layer.
    
    Computes: H' = sigma(D^-1/2 * A_tilde * D^-1/2 * H * W)
    Wait, in a non-sparse framework like MLX, we typically use:
    H' = activation(Normalized_Adjacency @ H @ W)
    """

    def __init__(self, in_dims: int, out_dims: int, bias: bool = True):
        super().__init__()
        self.weight = mx.random.normal((in_dims, out_dims)) * (1.0 / np.sqrt(in_dims))
        if bias:
            self.bias = mx.zeros((out_dims,))
        else:
            self.bias = None

    def __call__(self, h: mx.array, adj: mx.array) -> mx.array:
        """
        Forward pass.
        
        Args:
            h: Node features [N, in_dims]
            adj: Normalized adjacency matrix [N, N]
        """
        # Linear transform: H @ W
        support = h @ self.weight
        
        # Message passing: Adj @ support
        output = adj @ support
        
        if self.bias is not None:
            output += self.bias
            
        return output


class MalwareGNN(nn.Module):
    """
    Graph Neural Network for malware classification.
    
    Architecture:
    1. Node Embedding (average pooling of opcode embeddings per block)
    2. Multiple GCN layers
    3. Global Mean Pooling
    4. Classification Head
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dims: int = 128,
        num_layers: int = 3,
        num_classes: int = 2,
        embedding_dims: int = 64
    ):
        super().__init__()
        
        # 1. Opcode Embedding (used to build node features from basic blocks)
        self.embedding = nn.Embedding(vocab_size, embedding_dims)
        
        # 2. Sequential GCN Layers
        self.gnn_layers = []
        # First layer maps embedding_dims -> hidden_dims
        self.gnn_layers.append(GCNLayer(embedding_dims, hidden_dims))
        
        for _ in range(num_layers - 1):
            self.gnn_layers.append(GCNLayer(hidden_dims, hidden_dims))
            
        # 3. Final classifier
        self.classifier = nn.Linear(hidden_dims, num_classes)
        
        self.activation = nn.relu

    def __call__(self, x: mx.array, adj: mx.array) -> mx.array:
        """
        Forward pass.
        
        Args:
            x: Node features [N, hidden_dims] - precomputed node features.
            adj: [N, N] adjacency matrix.
        """
        h = x
        
        for layer in self.gnn_layers:
            h = layer(h, adj)
            h = self.activation(h)
            
        # Global Mean Pooling
        # In a single-graph pass, this is just mean over N
        pooled = mx.mean(h, axis=0, keepdims=True) # [1, hidden_dims]
        
        # Classify
        logits = self.classifier(pooled) # [1, num_classes]
        
        return logits

    def embed_nodes(self, node_opcodes: list[list[int]]) -> mx.array:
        """
        Convert lists of opcode IDs per node into a [N, embedding_dims] tensor.
        Uses mean pooling of embeddings within each basic block.
        """
        node_features = []
        unk_token_id = 1 # Consistent with tokenizer.py

        for ops in node_opcodes:
            if not ops:
                # Empty block (e.g. padding/auxiliary node)
                node_features.append(mx.zeros((self.embedding.weight.shape[1],)))
                continue
            
            # Embed opcodes [len(ops), embedding_dims]
            op_indices = mx.array(ops)
            embeddings = self.embedding(op_indices)
            
            # Mean pool [embedding_dims]
            pooled = mx.mean(embeddings, axis=0)
            node_features.append(pooled)
            
        return mx.stack(node_features)

    @staticmethod
    def normalize_adjacency(adj: mx.array) -> mx.array:
        """
        Compute Symmetric Normalized Adjacency: D^-1/2 * A_tilde * D^-1/2
        """
        # Add self-loops (A_tilde = A + I)
        n = adj.shape[0]
        a_tilde = adj + mx.eye(n)
        
        # Degree matrix D
        degrees = mx.sum(a_tilde, axis=1)
        
        # D^-1/2
        d_inv_sqrt = mx.power(degrees, -0.5)
        # Handle division by zero
        d_inv_sqrt = mx.where(mx.isinf(d_inv_sqrt), 0.0, d_inv_sqrt)
        
        d_mat = mx.diag(d_inv_sqrt)
        
        # Normalized Adj
        return d_mat @ a_tilde @ d_mat

    @staticmethod
    def cast_to_bf16(model: MalwareGNN):
        """Cast all model parameters to bfloat16 for efficiency."""
        model.apply(lambda x: x.astype(mx.bfloat16))
        return model

import numpy as np 
