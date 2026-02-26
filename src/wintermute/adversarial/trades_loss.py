"""
trades_loss.py — TRADES loss in pure MLX.

L = cross_entropy(f(x_clean), y) + beta * KL(softmax(f(x_clean)) || softmax(f(x_adv)))

beta warms up from 0 to target over the first 30% of training.
"""

import mlx.core as mx
import mlx.nn as nn


class TRADESLoss:
    def __init__(self, beta: float = 1.0, warmup_fraction: float = 0.3):
        self.beta = beta
        self.warmup_fraction = warmup_fraction

    def __call__(self, model, x_clean, labels, x_adv, epoch: int, max_epochs: int):
        clean_logits = model(x_clean)
        L_det = mx.mean(nn.losses.cross_entropy(clean_logits, labels))

        adv_logits = model(x_adv)
        clean_p = mx.softmax(clean_logits, axis=-1)
        adv_p = mx.softmax(adv_logits, axis=-1)
        adv_log_p = mx.log(adv_p + 1e-8)
        kl = mx.sum(clean_p * (mx.log(clean_p + 1e-8) - adv_log_p), axis=-1)
        L_kl = mx.mean(kl)

        beta_t = self.beta * min(1.0, epoch / max(max_epochs * self.warmup_fraction, 1))
        return L_det + beta_t * L_kl
