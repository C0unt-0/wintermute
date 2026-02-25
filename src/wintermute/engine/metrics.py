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

    Returns a num_classes x num_classes list of lists.
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


def compute_macro_f1(model, x, y, batch_size: int, num_classes: int) -> float:
    """Macro-averaged F1 over all classes. Uses model in inference mode."""
    import numpy as np
    from wintermute.engine.trainer import batch_iterate

    preds, labels = [], []
    for xb, yb in batch_iterate(x, y, batch_size, shuffle=False):
        p = mx.argmax(model(xb), axis=1)
        # Materialise the lazy MLX array before converting to Python list
        mx.synchronize()
        preds.extend(p.tolist())
        labels.extend(yb.tolist())
    p_arr, l_arr = np.array(preds), np.array(labels)
    f1s = []
    for c in range(num_classes):
        tp = np.sum((p_arr == c) & (l_arr == c))
        fp = np.sum((p_arr == c) & (l_arr != c))
        fn = np.sum((p_arr != c) & (l_arr == c))
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1s.append(2 * prec * rec / (prec + rec + 1e-9))
    return float(np.mean(f1s))


def compute_auc_roc(scores: "np.ndarray", labels: "np.ndarray") -> float:
    """Binary AUC-ROC via trapezoidal rule."""
    import numpy as np
    idx = np.argsort(-scores)
    ls = labels[idx]
    n_pos, n_neg = np.sum(labels == 1), np.sum(labels == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tpr_curve = np.concatenate([[0.0], np.cumsum(ls) / n_pos])
    fpr_curve = np.concatenate([[0.0], np.cumsum(1 - ls) / n_neg])
    # np.trapz was removed in NumPy 2.0; use np.trapezoid when available.
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    return float(_trapz(tpr_curve, fpr_curve))


def fpr_at_fnr_threshold(
    scores: "np.ndarray", labels: "np.ndarray", target_fnr: float = 0.01
) -> float:
    """FPR when threshold is set to achieve target_fnr (miss rate).

    Sweeps thresholds in descending order and returns the FPR at the last
    threshold where FNR >= target_fnr.  Descending sweep means thresholds run
    from the most conservative (highest, fewest positives flagged) down to the
    most permissive (lowest, all positives flagged).  Returning the last match
    gives the lowest threshold — and therefore the highest sensitivity — that
    still satisfies the target miss rate.  If no threshold achieves
    FNR >= target_fnr (e.g. target_fnr=1.0 when positives always appear in the
    score range), returns 0.0.
    """
    import numpy as np
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.0
    result = None
    for thresh in np.sort(np.unique(scores))[::-1]:
        pos_pred = scores >= thresh
        fnr = np.sum((~pos_pred) & (labels == 1)) / n_pos
        fpr = np.sum(pos_pred & (labels == 0)) / n_neg
        if fnr >= target_fnr:
            result = float(fpr)
    return result if result is not None else 0.0
