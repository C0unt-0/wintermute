"""
transformer.py — MalBERT Architecture

A purpose-built malware language model that extends the base Transformer
encoder with BERT-style features optimised for opcode classification.

Key improvements over the base MalwareClassifier:
    - Padding-aware attention masks (ignores PAD tokens)
    - [CLS] token pooling for classification
    - Dropout regularisation (embeddings + attention + FFN)
    - Optional MLM pre-training mode
    - Scaled-up capacity (256-D, 6 layers, 8 heads)

Architecture
------------
 Input  [B, T]  int tokens  →  prepend [CLS], append [SEP]
   ↓  Token Embedding (V → 256)
   ↓  + Positional Embedding (T+2 → 256)
   ↓  Dropout(0.1)
   ↓  × 6 Pre-Norm Encoder Blocks
   │     ├ LN → MHA(8 heads, mask=pad_mask) + Dropout + Residual
   │     └ LN → FFN(256→1024→256 GELU) + Dropout + Residual
   ↓  Final LayerNorm
   ↓  [CLS] token → Linear(256, C)   or   All → Linear(256, V) for MLM
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn

# Re-export the base model for backward compatibility
from wintermute.models.sequence import MalwareClassifier  # noqa: F401


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class MalBERTConfig:
    """All hyperparameters for MalBERT."""

    vocab_size: int = 256
    max_seq_length: int = 2048       # before [CLS]/[SEP] are prepended
    dims: int = 256
    num_heads: int = 8
    num_layers: int = 6
    mlp_dims: int = 1024
    dropout: float = 0.1
    num_classes: int = 9

    # Special token IDs (must match the tokenizer)
    pad_id: int = 0
    cls_id: int = 2
    sep_id: int = 3
    mask_id: int = 4


# ═══════════════════════════════════════════════════════════════════════════
# Building blocks
# ═══════════════════════════════════════════════════════════════════════════
class MalBERTFeedForward(nn.Module):
    """FFN with GELU activation and dropout."""

    def __init__(self, dims: int, hidden_dims: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(dims, hidden_dims)
        self.linear2 = nn.Linear(hidden_dims, dims)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class MalBERTEncoderBlock(nn.Module):
    """
    Pre-LayerNorm Transformer Encoder block with dropout and mask support.

    Unlike the base EncoderBlock, this version:
    - Accepts an attention mask to ignore PAD tokens
    - Applies dropout after attention and FFN
    """

    def __init__(
        self, dims: int, num_heads: int, mlp_dims: int, dropout: float = 0.1
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(dims)
        self.attention = nn.MultiHeadAttention(dims, num_heads, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(dims)
        self.ffn = MalBERTFeedForward(dims, mlp_dims, dropout)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        # Self-attention with pre-norm + residual + dropout
        h = self.ln1(x)
        h = self.attention(h, h, h, mask=mask)
        h = self.attn_dropout(h)
        x = x + h

        # FFN with pre-norm + residual (FFN has internal dropout)
        h = self.ln2(x)
        h = self.ffn(h)
        x = x + h
        return x


# ═══════════════════════════════════════════════════════════════════════════
# MalBERT Encoder
# ═══════════════════════════════════════════════════════════════════════════
class MalBERTEncoder(nn.Module):
    """
    Stack of MalBERT encoder blocks with embeddings and attention masking.

    Handles:
    - Token + positional embeddings
    - Padding-aware attention mask generation
    - Post-encoder layer normalisation
    """

    def __init__(self, config: MalBERTConfig):
        super().__init__()
        self.config = config

        # +2 for [CLS] and [SEP] tokens
        effective_length = config.max_seq_length + 2

        self.token_embedding = nn.Embedding(config.vocab_size, config.dims)
        self.position_embedding = nn.Embedding(effective_length, config.dims)
        self.embed_dropout = nn.Dropout(config.dropout)

        self.layers = [
            MalBERTEncoderBlock(
                config.dims, config.num_heads, config.mlp_dims, config.dropout
            )
            for _ in range(config.num_layers)
        ]

        self.final_norm = nn.LayerNorm(config.dims)

    def _make_pad_mask(self, x: mx.array) -> mx.array:
        """
        Create an additive attention mask that blocks PAD positions.

        Args:
            x: [B, T] integer token IDs

        Returns:
            mask: [B, 1, 1, T] with 0.0 for real tokens, -inf for PAD
        """
        # Infer dtype from model parameters to match bfloat16 models
        dtype = self.token_embedding.weight.dtype

        # True where token is NOT pad
        not_pad = (x != self.config.pad_id)               # [B, T]
        # Convert: True→0.0, False→large negative value
        mask = mx.where(
            not_pad,
            mx.zeros(not_pad.shape, dtype=dtype),
            mx.full(not_pad.shape, -1e9, dtype=dtype),
        )                                                  # [B, T]
        # Reshape for broadcasting: [B, 1, 1, T]
        mask = mask[:, None, None, :]
        return mask

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass through the encoder.

        Args:
            x: [B, T] integer token IDs (with [CLS] and [SEP] already prepended/appended)

        Returns:
            h: [B, T, D] hidden states
        """
        B, T = x.shape

        # Embeddings
        tok_emb = self.token_embedding(x)                  # [B, T, D]
        positions = mx.arange(T)                           # [T]
        pos_emb = self.position_embedding(positions)       # [T, D]
        h = self.embed_dropout(tok_emb + pos_emb)

        # Attention mask
        mask = self._make_pad_mask(x)                      # [B, 1, 1, T]

        # Encoder stack
        for layer in self.layers:
            h = layer(h, mask=mask)

        # Final norm
        h = self.final_norm(h)
        return h


# ═══════════════════════════════════════════════════════════════════════════
# MalBERT (Classification + MLM)
# ═══════════════════════════════════════════════════════════════════════════
class MalBERT(nn.Module):
    """
    MalBERT — purpose-built malware language model.

    Supports two modes:
    - **classify**: [CLS] token → Linear → class logits  (default)
    - **mlm**: all tokens → Linear → vocab logits  (pre-training)

    Parameters
    ----------
    config : MalBERTConfig
        All architecture hyperparameters.
    """

    def __init__(self, config: MalBERTConfig):
        super().__init__()
        self.config = config
        self.encoder = MalBERTEncoder(config)

        # Classification head: [CLS] → logits
        self.classifier = nn.Linear(config.dims, config.num_classes)

        # MLM head: all tokens → vocab logits (shared or separate from embeddings)
        self.mlm_head = nn.Linear(config.dims, config.vocab_size)

    def _prepend_cls_append_sep(self, x: mx.array) -> mx.array:
        """
        Prepend [CLS] and append [SEP] to input sequences.

        Input:  [B, T]
        Output: [B, T+2]
        """
        B, T = x.shape
        cls_col = mx.full((B, 1), self.config.cls_id, dtype=x.dtype)
        sep_col = mx.full((B, 1), self.config.sep_id, dtype=x.dtype)
        return mx.concatenate([cls_col, x, sep_col], axis=1)

    def __call__(
        self, x: mx.array, mode: str = "classify"
    ) -> mx.array:
        """
        Forward pass.

        Args:
            x: [B, T] raw token IDs (without [CLS]/[SEP])
            mode: 'classify' or 'mlm'

        Returns:
            - classify mode: [B, num_classes] logits
            - mlm mode: [B, T+2, vocab_size] logits
        """
        # Add special tokens
        x_with_special = self._prepend_cls_append_sep(x)   # [B, T+2]

        # Encode
        hidden = self.encoder(x_with_special)               # [B, T+2, D]

        if mode == "classify":
            # Use [CLS] token (position 0) for classification
            cls_hidden = hidden[:, 0, :]                    # [B, D]
            logits = self.classifier(cls_hidden)            # [B, C]
            return logits
        elif mode == "mlm":
            # All positions → vocab logits for masked language modeling
            logits = self.mlm_head(hidden)                  # [B, T+2, V]
            return logits
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'classify' or 'mlm'.")

    def encode(self, x: mx.array) -> mx.array:
        """
        Get the [CLS] representation without the classification head.

        Useful for downstream tasks or embedding extraction.
        """
        x_with_special = self._prepend_cls_append_sep(x)
        hidden = self.encoder(x_with_special)
        return hidden[:, 0, :]  # [B, D]

    @staticmethod
    def cast_to_bf16(model: nn.Module) -> None:
        """Convert all floating-point parameters to bfloat16 in-place."""
        model.apply(lambda x: x.astype(mx.bfloat16))

    @staticmethod
    def from_config(config: MalBERTConfig) -> "MalBERT":
        """Factory method to create MalBERT from a config."""
        return MalBERT(config)


# ═══════════════════════════════════════════════════════════════════════════
# MalBERT for MLM Pre-training (convenience wrapper)
# ═══════════════════════════════════════════════════════════════════════════
class MalBERTForMLM(nn.Module):
    """
    Convenience wrapper for MLM pre-training.

    Wraps a MalBERT model and forces `mode='mlm'` on forward pass.
    After pre-training, extract the encoder weights with `get_encoder_weights()`
    and load them into a fresh MalBERT for fine-tuning.
    """

    def __init__(self, config: MalBERTConfig):
        super().__init__()
        self.malbert = MalBERT(config)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass in MLM mode → [B, T+2, vocab_size]."""
        return self.malbert(x, mode="mlm")

    def get_encoder_weights(self) -> dict:
        """Extract encoder weights for transfer to classification model."""
        return self.malbert.encoder.parameters()
