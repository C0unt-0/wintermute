"""
gnn_trainer.py — Wintermute GNN Training Engine

Handles loading graph pickles, constructing adjacency matrices,
and running the GNN training loop.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from omegaconf import OmegaConf

from wintermute.models.gnn import MalwareGNN


class GNNTrainer:
    """
    MLX training engine for MalwareGNN.
    """

    DEFAULTS = {
        "model": {
            "hidden_dims": 128,
            "num_layers": 3,
            "num_classes": 2,
            "embedding_dims": 64,
        },
        "training": {
            "epochs": 20,
            "batch_size": 1,  # Single graph per batch for now
            "learning_rate": 3e-4,
            "weight_decay": 0.01,
            "val_ratio": 0.2,
            "seed": 42,
            "precision": "bfloat16",
            "save_path": "malware_gnn.safetensors",
        }
    }

    def __init__(
        self,
        config_path: str | Path | None = None,
        overrides: dict | None = None,
    ):
        cfg = OmegaConf.create(self.DEFAULTS)
        if config_path and Path(config_path).exists():
            file_cfg = OmegaConf.load(config_path)
            cfg = OmegaConf.merge(cfg, file_cfg)
        if overrides:
            cfg = OmegaConf.merge(cfg, overrides)
        self.cfg = cfg

    def load_graph_dataset(self, graphs_dir: Path, vocab: dict[str, int]) -> list[dict]:
        """Load all .pkl files from directory."""
        datasets = []
        stoi = vocab
        
        for pkl_file in sorted(graphs_dir.glob("*.pkl")):
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
                
            graph = data["graph"]
            
            # Map opcodes to IDs
            node_opcode_ids = []
            for ops in graph["nodes"]:
                ids = [stoi.get(op, 1) for op in ops] # 1 = <UNK>
                node_opcode_ids.append(ids)
            
            # Prepare adjacency matrix
            n_nodes = len(graph["nodes"])
            adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
            for u, v in graph["edges"]:
                adj[u, v] = 1.0
                adj[v, u] = 1.0 # Undirected CFG for simplicity
                
            datasets.append({
                "node_ids": node_opcode_ids,
                "adj": mx.array(adj),
                "label": data["label"]
            })
            
        return datasets

    def train(self, graphs_dir: str | Path, vocab_path: str | Path) -> float:
        """Run Graph Neural Network training."""
        graphs_dir = Path(graphs_dir)
        vocab_path = Path(vocab_path)
        
        with open(vocab_path) as f:
            vocab = json.load(f)
        vocab_size = len(vocab)
        
        print("Loading graph dataset …")
        dataset = self.load_graph_dataset(graphs_dir, vocab)
        
        # Split
        n = len(dataset)
        rng = np.random.default_rng(self.cfg.training.seed)
        indices = rng.permutation(n)
        split = int(n * (1 - self.cfg.training.val_ratio))
        
        train_data = [dataset[i] for i in indices[:split]]
        val_data = [dataset[i] for i in indices[split:]]
        
        print(f"  Graphs: {n}  |  Train: {len(train_data)}  |  Val: {len(val_data)}")
        
        # Build model
        mcfg = self.cfg.model
        tcfg = self.cfg.training
        
        model = MalwareGNN(
            vocab_size=vocab_size,
            hidden_dims=mcfg.hidden_dims,
            num_layers=mcfg.num_layers,
            num_classes=mcfg.num_classes,
            embedding_dims=mcfg.embedding_dims
        )
        
        if tcfg.precision == "bfloat16":
            MalwareGNN.cast_to_bf16(model)
            
        # Optimizer
        total_steps = tcfg.epochs * len(train_data)
        lr_schedule = optim.cosine_decay(tcfg.learning_rate, total_steps)
        optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=tcfg.weight_decay)
        
        def loss_fn(model, node_ids, adj, y):
            # 1. Embed nodes [N, embedding_dims]
            x = model.embed_nodes(node_ids)
            # 2. Normalize Adj
            adj_norm = model.normalize_adjacency(adj)
            # 3. Forward
            logits = model(x, adj_norm)
            # 4. Cross-entropy
            return mx.mean(nn.losses.cross_entropy(logits, mx.array([y])))

        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        
        print(f"\nTraining GNN for {tcfg.epochs} epochs …\n")
        best_val_acc = 0.0
        
        for epoch in range(1, tcfg.epochs + 1):
            t0 = time.perf_counter()
            epoch_loss = 0.0
            
            # Shuffle train
            rng.shuffle(train_data)
            
            for sample in train_data:
                loss, grads = loss_and_grad_fn(
                    model, sample["node_ids"], sample["adj"], sample["label"]
                )
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(train_data)
            
            # Eval
            train_acc = self._compute_accuracy(model, train_data)
            val_acc = self._compute_accuracy(model, val_data)
            elapsed = time.perf_counter() - t0
            
            print(f"Epoch {epoch:2d} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.1%} | Val Acc: {val_acc:.1%} | {elapsed:.1f}s")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model.save_weights(tcfg.save_path)
                print(f"  ↑ Saved best model")
                
        return best_val_acc

    def _compute_accuracy(self, model, dataset) -> float:
        if not dataset: return 0.0
        correct = 0
        for sample in dataset:
            x = model.embed_nodes(sample["node_ids"])
            adj_norm = model.normalize_adjacency(sample["adj"])
            logits = model(x, adj_norm)
            pred = mx.argmax(logits, axis=1).item()
            if pred == sample["label"]:
                correct += 1
        return correct / len(dataset)
