"""
sequence.py — Wintermute Bidirectional Transformer Encoder

Relocated from 02_model.py with no architectural changes.

A classification-only Transformer that reads tokenised opcode sequences
and outputs an N-class logit vector.

Architecture
------------
 Input  [B, 2048]  int tokens
   ↓  Token Embedding   (vocab → 128-D)
   ↓  + Positional Embedding (2048 → 128-D)
   ↓  × 4 Pre-Norm Encoder Blocks
   │     ├ LayerNorm → MultiHeadAttention (4 heads, no causal mask)
   │     │ + residual
   │     └ LayerNorm → FFN 128→512→128 GELU
   │       + residual
   ↓  Global Mean Pooling  → [B, 128]
   ↓  Linear(128, C)       → [B, C] logits
"""

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    """Position-wise FFN:  Linear → GELU → Linear."""

    def __init__(self, dims: int, hidden_dims: int):
        super().__init__()
        self.linear1 = nn.Linear(dims, hidden_dims)
        self.linear2 = nn.Linear(hidden_dims, dims)
        self.gelu = nn.GELU()

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear2(self.gelu(self.linear1(x)))


# ---------------------------------------------------------------------------
# Encoder Block (Pre-LN variant)
# ---------------------------------------------------------------------------
class EncoderBlock(nn.Module):
    """
    Pre-LayerNorm Transformer Encoder block.

    Pre-LN applies LayerNorm *before* the sub-layer (attention / FFN)
    rather than after, which improves training stability for small models.

    Causal masking is intentionally **disabled** — this is a bidirectional
    encoder, not a GPT-style decoder.
    """

    def __init__(self, dims: int, num_heads: int, mlp_dims: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dims)
        self.attention = nn.MultiHeadAttention(dims, num_heads, bias=True)
        self.ln2 = nn.LayerNorm(dims)
        self.ffn = FeedForward(dims, mlp_dims)

    def __call__(self, x: mx.array) -> mx.array:
        # Self-attention with pre-norm + residual
        h = self.ln1(x)
        h = self.attention(h, h, h, mask=None)   # bidirectional — no mask
        x = x + h

        # FFN with pre-norm + residual
        h = self.ln2(x)
        h = self.ffn(h)
        x = x + h
        return x


# ---------------------------------------------------------------------------
# Full Classifier
# ---------------------------------------------------------------------------
class MalwareClassifier(nn.Module):
    """
    Bidirectional Transformer Encoder for malware classification.

    Parameters
    ----------
    vocab_size : int
        Number of tokens in the opcode vocabulary (incl. PAD & UNK).
    max_seq_length : int
        Maximum number of opcodes per sample (default 2048).
    dims : int
        Embedding / hidden dimension (default 128).
    num_heads : int
        Number of attention heads (default 4 → 32-D per head).
    num_layers : int
        Number of stacked encoder blocks (default 4).
    mlp_dims : int
        Hidden width of the position-wise FFN (default 512).
    num_classes : int
        Output classes (default 2 — safe / malicious).
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_length: int = 2048,
        dims: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        mlp_dims: int = 512,
        num_classes: int = 2,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dims)
        self.position_embedding = nn.Embedding(max_seq_length, dims)

        self.encoder_blocks = [
            EncoderBlock(dims, num_heads, mlp_dims)
            for _ in range(num_layers)
        ]

        self.final_norm = nn.LayerNorm(dims)
        self.classifier = nn.Linear(dims, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.

        Parameters
        ----------
        x : mx.array, shape [B, T]
            Integer token IDs.

        Returns
        -------
        logits : mx.array, shape [B, num_classes]
        """
        B, T = x.shape

        # Embeddings  -----------------------------------------------------------
        tok_emb = self.token_embedding(x)                      # [B, T, D]
        positions = mx.arange(T)                               # [T]
        pos_emb = self.position_embedding(positions)           # [T, D]
        h = tok_emb + pos_emb                                  # broadcast

        # Encoder stack ---------------------------------------------------------
        for block in self.encoder_blocks:
            h = block(h)                                       # [B, T, D]

        # Post-encoder norm
        h = self.final_norm(h)

        # Global Mean Pooling ---------------------------------------------------
        # Average across the sequence dimension → [B, D]
        h = mx.mean(h, axis=1)

        # Classification head ---------------------------------------------------
        logits = self.classifier(h)                            # [B, C]
        return logits

    # ------------------------------------------------------------------
    # Utility: cast entire model to bfloat16
    # ------------------------------------------------------------------
    @staticmethod
    def cast_to_bf16(model: nn.Module) -> None:
        """
        Convert all floating-point parameters to ``bfloat16`` in-place.

        This halves the memory footprint and utilises Apple Silicon's
        dedicated 16-bit hardware accelerators.
        """
        model.apply(lambda x: x.astype(mx.bfloat16))
