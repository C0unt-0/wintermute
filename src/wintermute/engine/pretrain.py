"""
pretrain.py — MalBERT Masked Language Model Pre-training

Implements the standard BERT MLM recipe:
    - Randomly mask 15% of input tokens
    - Of those: 80% → <MASK>, 10% → random token, 10% → unchanged
    - Train to predict the original token at masked positions
    - Save encoder weights for fine-tuning on classification
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

from wintermute.engine.trainer import batch_iterate
from wintermute.models.transformer import MalBERT, MalBERTConfig


# ---------------------------------------------------------------------------
# Masking strategy
# ---------------------------------------------------------------------------
def apply_mlm_masking(
    x: mx.array,
    config: MalBERTConfig,
    mask_prob: float = 0.15,
    rng: np.random.Generator | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Apply BERT-style masking to input sequences.

    Args:
        x: [B, T] input token IDs
        config: MalBERTConfig with special token IDs
        mask_prob: probability of masking each token
        rng: numpy random generator

    Returns:
        masked_x: [B, T] tokens with masking applied
        labels: [B, T] original token IDs at masked positions, -100 elsewhere
        mask_positions: [B, T] boolean mask of which positions were masked
    """
    if rng is None:
        rng = np.random.default_rng()

    x_np = np.array(x)
    B, T = x_np.shape

    # Don't mask special tokens (PAD, CLS, SEP) or existing MASK tokens
    special_ids = {config.pad_id, config.cls_id, config.sep_id, config.mask_id}
    maskable = np.ones_like(x_np, dtype=bool)
    for sid in special_ids:
        maskable &= (x_np != sid)

    # Randomly select tokens to mask
    rand = rng.random((B, T))
    mask_positions = maskable & (rand < mask_prob)

    # Labels: original token at masked positions, -100 elsewhere (ignore)
    labels = np.full_like(x_np, -100)
    labels[mask_positions] = x_np[mask_positions]

    # Apply masking strategy: 80% MASK, 10% random, 10% unchanged
    masked_x = x_np.copy()
    mask_indices = np.where(mask_positions)

    n_masked = mask_indices[0].shape[0]
    if n_masked > 0:
        strategy = rng.random(n_masked)

        # 80% → <MASK>
        mask_token_positions = strategy < 0.8
        masked_x[mask_indices[0][mask_token_positions],
                 mask_indices[1][mask_token_positions]] = config.mask_id

        # 10% → random token (avoid special tokens)
        random_positions = (strategy >= 0.8) & (strategy < 0.9)
        n_random = random_positions.sum()
        if n_random > 0:
            # Random token from vocab, excluding special tokens (0-4)
            random_tokens = rng.integers(5, config.vocab_size, size=n_random)
            masked_x[mask_indices[0][random_positions],
                     mask_indices[1][random_positions]] = random_tokens

        # 10% → unchanged (already copied from x_np)

    return mx.array(masked_x), mx.array(labels), mx.array(mask_positions)


# ---------------------------------------------------------------------------
# Pre-training loop
# ---------------------------------------------------------------------------
class MLMPretrainer:
    """
    MLM pre-training engine for MalBERT.

    Usage
    -----
    >>> pretrainer = MLMPretrainer(config_path="configs/malbert_config.yaml")
    >>> pretrainer.pretrain(data_dir="data/processed")
    """

    DEFAULTS = {
        "malbert": {
            "vocab_size": 256,
            "dims": 256,
            "num_heads": 8,
            "num_layers": 6,
            "mlp_dims": 1024,
            "dropout": 0.1,
            "max_seq_length": 2048,
            "num_classes": 9,
        },
        "pretrain": {
            "mask_prob": 0.15,
            "epochs": 50,
            "batch_size": 8,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "seed": 42,
            "save_path": "malbert_pretrained.safetensors",
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

    def pretrain(self, data_dir: str | Path = "data/processed") -> float:
        """
        Run MLM pre-training.

        Returns the final MLM loss.
        """
        data_dir = Path(data_dir)
        mcfg = self.cfg.malbert
        pcfg = self.cfg.pretrain

        # 1. Load data (we only need x_data for unsupervised pre-training)
        print("Loading data for MLM pre-training …")
        x_np = np.load(data_dir / "x_data.npy")
        with open(data_dir / "vocab.json") as f:
            vocab = json.load(f)
        vocab_size = len(vocab)

        x = mx.array(x_np)
        # Dummy labels (not used for MLM, but needed for batch_iterate)
        y_dummy = mx.zeros(x.shape[0], dtype=mx.int32)

        print(f"  Samples: {x.shape[0]}  |  Vocab: {vocab_size}  "
              f"|  Seq length: {x.shape[1]}")

        # 2. Build MalBERT config + model
        config = MalBERTConfig(
            vocab_size=vocab_size,
            max_seq_length=mcfg.max_seq_length,
            dims=mcfg.dims,
            num_heads=mcfg.num_heads,
            num_layers=mcfg.num_layers,
            mlp_dims=mcfg.mlp_dims,
            dropout=mcfg.dropout,
            num_classes=mcfg.num_classes,
        )

        model = MalBERT(config)
        MalBERT.cast_to_bf16(model)

        import mlx.utils
        flat = mlx.utils.tree_flatten(model.parameters())
        n_params = sum(v.size for _, v in flat)
        print(f"  MalBERT parameters: {n_params:,}  (bfloat16)")

        # 3. Optimizer
        n_batches = (x.shape[0] + pcfg.batch_size - 1) // pcfg.batch_size
        total_steps = pcfg.epochs * n_batches

        lr_schedule = optim.cosine_decay(pcfg.learning_rate, total_steps)
        optimizer = optim.AdamW(
            learning_rate=lr_schedule,
            weight_decay=pcfg.get("weight_decay", 0.01),
        )

        # 4. Loss function
        rng = np.random.default_rng(pcfg.seed)

        def mlm_loss_fn(model, xb, _yb):
            """Compute MLM loss: mask tokens, predict, cross-entropy on masked only."""
            masked_x, labels, mask_pos = apply_mlm_masking(
                xb, config, mask_prob=pcfg.mask_prob, rng=rng
            )
            # Forward in MLM mode
            logits = model(masked_x, mode="mlm")       # [B, T+2, V]

            # We only compute loss on the original T positions (skip [CLS]/[SEP])
            # labels are for positions 0..T-1 of the input
            # logits include [CLS] at 0 and [SEP] at T+1
            # So the real token logits are at positions 1..T
            token_logits = logits[:, 1:-1, :]           # [B, T, V]

            # Flatten everything
            B, T, V = token_logits.shape
            flat_logits = token_logits.reshape(-1, V)   # [B*T, V]
            flat_labels = labels.reshape(-1)            # [B*T]

            # Only compute loss where labels != -100
            valid = flat_labels != -100
            if mx.sum(valid).item() == 0:
                return mx.array(0.0)

            # Cross-entropy on valid positions
            valid_logits = flat_logits[valid]
            valid_labels = flat_labels[valid]
            loss = mx.mean(nn.losses.cross_entropy(valid_logits, valid_labels))
            return loss

        loss_and_grad_fn = nn.value_and_grad(model, mlm_loss_fn)

        # 5. Training loop
        print(f"\nMLM Pre-training for {pcfg.epochs} epochs ({total_steps} steps) …\n")
        print(f"{'Epoch':>5}  {'MLM Loss':>10}  {'Time':>8}")
        print("─" * 30)

        best_loss = float("inf")
        for epoch in range(1, pcfg.epochs + 1):
            t0 = time.perf_counter()
            epoch_loss = 0.0
            n_actual = 0

            for xb, yb in batch_iterate(x, y_dummy, pcfg.batch_size):
                loss, grads = loss_and_grad_fn(model, xb, yb)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)
                epoch_loss += loss.item()
                n_actual += 1

            avg_loss = epoch_loss / max(n_actual, 1)
            elapsed = time.perf_counter() - t0

            print(f"{epoch:5d}  {avg_loss:10.4f}  {elapsed:7.1f}s")

            if avg_loss < best_loss:
                best_loss = avg_loss
                model.save_weights(pcfg.save_path)
                print(f"       ↑ new best — saved to {pcfg.save_path}")

        print(f"\n✅  MLM pre-training complete.  Best loss: {best_loss:.4f}")
        print(f"    Encoder weights saved to: {pcfg.save_path}")
        print(f"    Use these for fine-tuning: wintermute train --pretrained {pcfg.save_path}")
        return best_loss
