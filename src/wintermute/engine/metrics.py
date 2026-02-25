"""
metrics.py — Wintermute Evaluation Metrics

Provides classification metrics for model evaluation.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def compute_accuracy(
    model: nn.Module,
    x: mx.array,
    y: mx.array,
    batch_size: int,
) -> float:
    """Compute classification accuracy over the full dataset."""
    from wintermute.engine.trainer import batch_iterate

    correct = 0
    total = 0
    for xb, yb in batch_iterate(x, y, batch_size, shuffle=False):
        logits = model(xb)
        preds = mx.argmax(logits, axis=1)
        correct += mx.sum(preds == yb).item()
        total += yb.shape[0]
    return correct / total if total > 0 else 0.0


def compute_f1(
    model: nn.Module,
    x: mx.array,
    y: mx.array,
    batch_size: int,
    num_classes: int = 2,
) -> dict[str, float]:
    """
    Compute per-class and macro F1-score.

    Returns a dict with 'per_class' (list) and 'macro' (float).
    """
    from wintermute.engine.trainer import batch_iterate
    import numpy as np

    all_preds = []
    all_labels = []

    for xb, yb in batch_iterate(x, y, batch_size, shuffle=False):
        logits = model(xb)
        preds = mx.argmax(logits, axis=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(yb.tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    per_class_f1 = []
    for c in range(num_classes):
        tp = np.sum((all_preds == c) & (all_labels == c))
        fp = np.sum((all_preds == c) & (all_labels != c))
        fn = np.sum((all_preds != c) & (all_labels == c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
               if (precision + recall) > 0 else 0.0)
        per_class_f1.append(f1)

    macro_f1 = float(np.mean(per_class_f1))
    return {"per_class": per_class_f1, "macro": macro_f1}


def confusion_matrix(
    model: nn.Module,
    x: mx.array,
    y: mx.array,
    batch_size: int,
    num_classes: int = 2,
) -> list[list[int]]:
    """
    Compute a confusion matrix.

    Returns a num_classes × num_classes list of lists.
    matrix[true_label][predicted_label] = count.
    """
    from wintermute.engine.trainer import batch_iterate
    import numpy as np

    all_preds = []
    all_labels = []

    for xb, yb in batch_iterate(x, y, batch_size, shuffle=False):
        logits = model(xb)
        preds = mx.argmax(logits, axis=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(yb.tolist())

    matrix = [[0] * num_classes for _ in range(num_classes)]
    for true, pred in zip(all_labels, all_preds):
        matrix[true][pred] += 1

    return matrix
