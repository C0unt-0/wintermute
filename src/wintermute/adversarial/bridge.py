"""
bridge.py — Connects the Gymnasium env (numpy) to the MLX defender model.

Single conversion point: np.ndarray → mx.array → model forward → float
"""

import mlx.core as mx
import numpy as np
from wintermute.models.fusion import WintermuteMalwareDetector


class DefenderBridge:
    """
    Wraps WintermuteMalwareDetector for use by the RL environment.

    Call signature matches what AsmMutationEnv expects:
        confidence: float = bridge(tokens: np.ndarray)
    """

    def __init__(self, model: WintermuteMalwareDetector):
        self.model = model
        self.model.eval()

    def __call__(self, tokens: np.ndarray) -> float:
        """
        Run inference on a single token sequence.

        Args:
            tokens: [T] numpy int32 array of vocab IDs

        Returns:
            float — P(malicious), between 0.0 and 1.0
        """
        x = mx.array(tokens[np.newaxis, :], dtype=mx.int32)  # [1, T]
        logits = self.model(x)                                 # [1, num_classes]
        probs = mx.softmax(logits, axis=-1)
        mx.eval(probs)

        if probs.shape[1] == 2:
            # Binary: index 1 = malicious
            return float(probs[0, 1].item())
        else:
            # Multi-class: 1 - P(benign), assuming class 0 = benign
            return float(1.0 - probs[0, 0].item())
