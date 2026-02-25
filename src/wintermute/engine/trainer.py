"""
trainer.py — Wintermute Training Engine

Class-based trainer refactored from 03_train.py.
Loads config from YAML, runs the MLX training loop with AdamW + cosine decay.
Integrates with MLflow for experiment tracking (Phase 2).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from omegaconf import OmegaConf

from wintermute.models.sequence import MalwareClassifier


# ---------------------------------------------------------------------------
# Shared utilities (used by metrics.py too via import)
# ---------------------------------------------------------------------------
def batch_iterate(
    x: mx.array, y: mx.array, batch_size: int, shuffle: bool = True
):
    """Yield mini-batches from the dataset."""
    n = x.shape[0]
    if shuffle:
        indices = mx.array(np.random.permutation(n))
        x = x[indices]
        y = y[indices]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield x[start:end], y[start:end]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer:
    """
    MLX training engine for MalwareClassifier.

    Parameters
    ----------
    config_path : str | Path | None
        Path to model_config.yaml.  Falls back to defaults.
    overrides : dict | None
        Runtime overrides (e.g. from CLI flags).
    """

    # Defaults matching spec.md
    DEFAULTS = {
        "model": {
            "dims": 128,
            "num_heads": 4,
            "num_layers": 4,
            "mlp_dims": 512,
            "num_classes": 2,
            "max_seq_length": 2048,
        },
        "training": {
            "epochs": 20,
            "batch_size": 8,
            "learning_rate": 3e-4,
            "weight_decay": 0.01,
            "val_ratio": 0.2,
            "seed": 42,
            "precision": "bfloat16",
            "save_path": "malware_model.safetensors",
        },
        "tracking": {
            "enabled": False,
            "experiment": "wintermute",
            "tracking_uri": "mlruns",
            "run_name": None,
        },
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

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    @staticmethod
    def load_dataset(data_dir: Path) -> tuple[mx.array, mx.array, dict]:
        """Load x_data.npy, y_data.npy, vocab.json and return MLX arrays."""
        x_np = np.load(data_dir / "x_data.npy")
        y_np = np.load(data_dir / "y_data.npy")
        with open(data_dir / "vocab.json") as f:
            vocab = json.load(f)
        x = mx.array(x_np)
        y = mx.array(y_np)
        return x, y, vocab

    @staticmethod
    def train_val_split(
        x: mx.array,
        y: mx.array,
        val_ratio: float = 0.2,
        seed: int = 42,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """Shuffle and split into train / validation sets."""
        n = x.shape[0]
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n)
        split = int(n * (1 - val_ratio))
        train_idx = mx.array(indices[:split])
        val_idx = mx.array(indices[split:])
        return x[train_idx], y[train_idx], x[val_idx], y[val_idx]

    # ------------------------------------------------------------------
    # Core training loop
    # ------------------------------------------------------------------
    def train(self, data_dir: str | Path = "data/processed") -> float:
        """
        Run the full training loop.

        Returns the best validation accuracy achieved.
        """
        data_dir = Path(data_dir)
        mcfg = self.cfg.model
        tcfg = self.cfg.training

        # 1. Load data ----------------------------------------------------------
        print("Loading dataset …")
        x, y, vocab = self.load_dataset(data_dir)
        vocab_size = len(vocab)
        print(f"  Samples: {x.shape[0]}  |  Vocab: {vocab_size}  "
              f"|  Seq length: {x.shape[1]}  |  Classes: {mcfg.num_classes}")

        x_train, y_train, x_val, y_val = self.train_val_split(
            x, y, val_ratio=tcfg.val_ratio, seed=tcfg.seed
        )
        print(f"  Train: {x_train.shape[0]}  |  Val: {x_val.shape[0]}")

        # 2. Build model --------------------------------------------------------
        print("Building model …")
        model = MalwareClassifier(
            vocab_size=vocab_size,
            max_seq_length=mcfg.max_seq_length,
            dims=mcfg.dims,
            num_heads=mcfg.num_heads,
            num_layers=mcfg.num_layers,
            mlp_dims=mcfg.mlp_dims,
            num_classes=mcfg.num_classes,
        )

        # Cast to bfloat16
        if tcfg.precision == "bfloat16":
            MalwareClassifier.cast_to_bf16(model)

        # Count parameters
        import mlx.utils
        flat = mlx.utils.tree_flatten(model.parameters())
        n_params = sum(v.size for _, v in flat)
        print(f"  Parameters: {n_params:,}  ({tcfg.precision})")

        # ── MLflow tracking setup ──────────────────────────────────────
        from wintermute.engine.tracking import MLflowTracker

        trcfg = self.cfg.tracking
        tracker = MLflowTracker(
            experiment=trcfg.experiment,
            tracking_uri=trcfg.tracking_uri,
            enabled=trcfg.enabled,
        )
        tracker.start_run(run_name=trcfg.get("run_name"))
        tracker.log_params(OmegaConf.to_container(self.cfg, resolve=True))
        tracker.log_model_summary(n_params, tcfg.precision)

        # 3. Optimizer + LR schedule --------------------------------------------
        n_batches = (x_train.shape[0] + tcfg.batch_size - 1) // tcfg.batch_size
        total_steps = tcfg.epochs * n_batches

        lr_schedule = optim.cosine_decay(tcfg.learning_rate, total_steps)
        optimizer = optim.AdamW(
            learning_rate=lr_schedule, weight_decay=tcfg.weight_decay
        )

        # 4. Loss function ------------------------------------------------------
        def loss_fn(model, xb, yb):
            logits = model(xb)
            return mx.mean(nn.losses.cross_entropy(logits, yb))

        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

        # 5. Training loop ------------------------------------------------------
        best_val_acc = 0.0
        print(f"\nTraining for {tcfg.epochs} epochs ({total_steps} steps) …\n")
        print(f"{'Epoch':>5}  {'Loss':>10}  {'Train Acc':>10}  "
              f"{'Val Acc':>10}  {'Time':>8}")
        print("─" * 52)

        for epoch in range(1, tcfg.epochs + 1):
            t0 = time.perf_counter()
            epoch_loss = 0.0
            n_batches_actual = 0

            for xb, yb in batch_iterate(x_train, y_train, tcfg.batch_size):
                loss, grads = loss_and_grad_fn(model, xb, yb)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)
                epoch_loss += loss.item()
                n_batches_actual += 1

            avg_loss = epoch_loss / max(n_batches_actual, 1)
            train_acc = self._compute_accuracy(
                model, x_train, y_train, tcfg.batch_size
            )
            val_acc = self._compute_accuracy(
                model, x_val, y_val, tcfg.batch_size
            )
            elapsed = time.perf_counter() - t0

            print(f"{epoch:5d}  {avg_loss:10.4f}  {train_acc:9.1%}  "
                  f"{val_acc:9.1%}  {elapsed:7.1f}s")

            # Log to MLflow
            tracker.log_metrics({
                "loss": avg_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "epoch_time_s": elapsed,
            }, step=epoch)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model.save_weights(tcfg.save_path)
                print(f"       ↑ new best — saved to {tcfg.save_path}")

        # ── Post-training ──────────────────────────────────────────────
        final_metrics = {
            "best_val_acc": best_val_acc,
            "final_loss": avg_loss,
            "final_train_acc": train_acc,
            "epochs_completed": tcfg.epochs,
        }
        tracker.save_metrics_json(final_metrics, "metrics.json")
        tracker.log_artifact(tcfg.save_path)
        tracker.log_artifact(str(data_dir / "vocab.json"))
        tracker.end_run()

        print(f"\n✅  Training complete.  Best val accuracy: {best_val_acc:.1%}")
        return best_val_acc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_accuracy(
        model: MalwareClassifier,
        x: mx.array,
        y: mx.array,
        batch_size: int,
    ) -> float:
        """Compute classification accuracy over the full dataset."""
        correct = 0
        total = 0
        for xb, yb in batch_iterate(x, y, batch_size, shuffle=False):
            logits = model(xb)
            preds = mx.argmax(logits, axis=1)
            correct += mx.sum(preds == yb).item()
            total += yb.shape[0]
        return correct / total if total > 0 else 0.0
