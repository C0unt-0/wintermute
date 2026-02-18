#!/usr/bin/env python3
"""
03_train.py — Wintermute Training Loop

Loads the preprocessed .npy dataset, instantiates the MalwareClassifier,
and trains it using MLX's value_and_grad with AdamW + cosine LR decay.

Usage:
    python src/03_train.py                       # defaults (20 epochs, 2 classes)
    python src/03_train.py --num-classes 9        # MS malware families
    python src/03_train.py --epochs 50 --lr 1e-4
    python src/03_train.py --data-dir data/processed --batch-size 16
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

# Python module names can't start with a digit, so we dynamically import
# 02_model.py via importlib.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "model", str(Path(__file__).resolve().parent / "02_model.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MalwareClassifier = _mod.MalwareClassifier



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_dataset(data_dir: Path):
    """Load x_data.npy, y_data.npy, vocab.json and return MLX arrays."""
    x_np = np.load(data_dir / "x_data.npy")
    y_np = np.load(data_dir / "y_data.npy")

    with open(data_dir / "vocab.json") as f:
        vocab = json.load(f)

    x = mx.array(x_np)
    y = mx.array(y_np)
    return x, y, vocab


def train_val_split(x: mx.array, y: mx.array, val_ratio: float = 0.2,
                    seed: int = 42):
    """Shuffle and split into train / validation sets."""
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    split = int(n * (1 - val_ratio))
    train_idx = mx.array(indices[:split])
    val_idx = mx.array(indices[split:])

    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def batch_iterate(x: mx.array, y: mx.array, batch_size: int, shuffle: bool = True):
    """Yield mini-batches from the dataset."""
    n = x.shape[0]
    if shuffle:
        indices = mx.array(np.random.permutation(n))
        x = x[indices]
        y = y[indices]

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield x[start:end], y[start:end]


def compute_accuracy(model: MalwareClassifier, x: mx.array, y: mx.array,
                     batch_size: int) -> float:
    """Compute classification accuracy over the full dataset."""
    correct = 0
    total = 0
    for xb, yb in batch_iterate(x, y, batch_size, shuffle=False):
        logits = model(xb)
        preds = mx.argmax(logits, axis=1)
        correct += mx.sum(preds == yb).item()
        total += yb.shape[0]
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wintermute — MLX training loop")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Directory containing x_data.npy, y_data.npy, vocab.json")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Peak learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="AdamW weight decay.")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                        help="Sequence length (must match dataset).")
    parser.add_argument("--num-classes", type=int, default=2,
                        help="Number of output classes (2=binary, 9=MS malware families).")
    parser.add_argument("--save-path", type=str, default="malware_model.safetensors",
                        help="Path to save the best model weights.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # 1. Load data --------------------------------------------------------------
    print("Loading dataset …")
    x, y, vocab = load_dataset(data_dir)
    vocab_size = len(vocab)
    num_classes = args.num_classes
    print(f"  Samples: {x.shape[0]}  |  Vocab: {vocab_size}  |  Seq length: {x.shape[1]}  |  Classes: {num_classes}")

    x_train, y_train, x_val, y_val = train_val_split(x, y)
    print(f"  Train: {x_train.shape[0]}  |  Val: {x_val.shape[0]}")

    # 2. Build model ------------------------------------------------------------
    print("Building model …")
    model = MalwareClassifier(
        vocab_size=vocab_size,
        max_seq_length=args.max_seq_length,
        num_classes=num_classes,
    )

    # Cast to bfloat16
    MalwareClassifier.cast_to_bf16(model)

    # Count parameters
    import mlx.utils
    flat = mlx.utils.tree_flatten(model.parameters())
    n_params = sum(v.size for _, v in flat)
    print(f"  Parameters: {n_params:,}  (bfloat16)")

    # 3. Optimizer + LR schedule ------------------------------------------------
    n_batches = (x_train.shape[0] + args.batch_size - 1) // args.batch_size
    total_steps = args.epochs * n_batches

    lr_schedule = optim.cosine_decay(args.lr, total_steps)
    optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=args.weight_decay)

    # 4. Loss function ----------------------------------------------------------
    def loss_fn(model, xb, yb):
        logits = model(xb)
        return mx.mean(nn.losses.cross_entropy(logits, yb))

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # 5. Training loop ----------------------------------------------------------
    best_val_acc = 0.0
    print(f"\nTraining for {args.epochs} epochs ({total_steps} steps) …\n")
    print(f"{'Epoch':>5}  {'Loss':>10}  {'Train Acc':>10}  {'Val Acc':>10}  {'Time':>8}")
    print("─" * 52)

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        epoch_loss = 0.0
        n_batches_actual = 0

        for xb, yb in batch_iterate(x_train, y_train, args.batch_size):
            loss, grads = loss_and_grad_fn(model, xb, yb)
            optimizer.update(model, grads)
            # Force evaluation so MLX doesn't build an infinitely deep graph
            mx.eval(model.parameters(), optimizer.state)
            epoch_loss += loss.item()
            n_batches_actual += 1

        avg_loss = epoch_loss / max(n_batches_actual, 1)
        train_acc = compute_accuracy(model, x_train, y_train, args.batch_size)
        val_acc = compute_accuracy(model, x_val, y_val, args.batch_size)
        elapsed = time.perf_counter() - t0

        print(f"{epoch:5d}  {avg_loss:10.4f}  {train_acc:9.1%}  {val_acc:9.1%}  {elapsed:7.1f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_weights(args.save_path)
            print(f"       ↑ new best — saved to {args.save_path}")

    print(f"\n✅  Training complete.  Best val accuracy: {best_val_acc:.1%}")


if __name__ == "__main__":
    main()
