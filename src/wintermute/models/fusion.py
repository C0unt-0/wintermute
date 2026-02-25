# src/wintermute/models/fusion.py
"""
fusion.py — WintermuteMalwareDetector

Unified malware classifier:
  sequence -> shared embedding -> MalBERT encoder -> [CLS] [B, D]
  CFG nodes -> shared embedding (mean pool per block) -> GAT encoder -> [B, D]
  cross-attention fusion -> classifier -> [B, num_classes]
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from wintermute.models.transformer import MalBERTConfig, MalBERTEncoder
from wintermute.models.gat import GATEncoder


@dataclass
class DetectorConfig:
    vocab_size: int = 512
    dims: int = 256
    num_heads: int = 8
    num_layers: int = 6
    mlp_dims: int = 1024
    dropout: float = 0.1
    max_seq_length: int = 2048
    gat_layers: int = 3
    gat_heads: int = 4
    num_fusion_heads: int = 4
    num_classes: int = 2
    pad_id: int = 0
    cls_id: int = 2
    sep_id: int = 3
    mask_id: int = 4


class WintermuteMalwareDetector(nn.Module):
    VERSION = "3.0.0"

    def __init__(self, config: DetectorConfig):
        super().__init__()
        self.config = config
        D = config.dims

        # Shared token embedding — used by both MalBERT and GAT node features
        self.token_embedding = nn.Embedding(config.vocab_size, D)

        # MalBERT encoder (receives the shared embedding)
        malbert_cfg = MalBERTConfig(
            vocab_size=config.vocab_size, max_seq_length=config.max_seq_length,
            dims=D, num_heads=config.num_heads, num_layers=config.num_layers,
            mlp_dims=config.mlp_dims, dropout=config.dropout,
            num_classes=config.num_classes,
            pad_id=config.pad_id, cls_id=config.cls_id,
            sep_id=config.sep_id, mask_id=config.mask_id,
        )
        self.malbert_encoder = MalBERTEncoder(malbert_cfg, self.token_embedding)

        # GAT encoder — expects pre-embedded node features [N, D]
        self.gat_encoder = GATEncoder(
            in_dims=D, hidden_dims=D,
            num_layers=config.gat_layers, num_heads=config.gat_heads,
            dropout=config.dropout,
        )

        # nn.Embedding ensures the fallback is a tracked learnable parameter
        self.no_graph_embedding = nn.Embedding(1, D)

        # Cross-attention: seq [CLS] queries over graph node features
        self.cross_attn = nn.MultiHeadAttention(D, config.num_fusion_heads, bias=True)
        self.fusion_norm = nn.LayerNorm(D)

        # Final fusion projection + classifier
        self.fusion_proj = nn.Linear(D * 2, D)
        self.classifier = nn.Linear(D, config.num_classes)

    def _prepend_cls_append_sep(self, x: mx.array) -> mx.array:
        """Prepend [CLS] and append [SEP] to input sequences.

        Input:  [B, T]
        Output: [B, T+2]
        """
        B = x.shape[0]
        cls_col = mx.full((B, 1), self.config.cls_id, dtype=x.dtype)
        sep_col = mx.full((B, 1), self.config.sep_id, dtype=x.dtype)
        return mx.concatenate([cls_col, x, sep_col], axis=1)

    def __call__(
        self,
        sequence: mx.array | None,                 # [B, T] token IDs  (may be None when token_embeddings provided)
        node_embs: mx.array | None = None,         # [N_total, D] flat pre-embedded node features
        edge_src: mx.array | None = None,          # [E] sparse COO source indices
        edge_dst: mx.array | None = None,          # [E] sparse COO destination indices
        batch_idx: mx.array | None = None,         # [N_total] node-to-graph membership
        n_graphs: int | None = None,               # equals B when graphs are provided
        token_embeddings: mx.array | None = None,  # [B, T+2, D] pre-computed (e.g. from Mixup)
    ) -> mx.array:
        # Derive batch size from whichever input is available
        if sequence is not None:
            B = sequence.shape[0]
        else:
            if token_embeddings is None:
                raise ValueError(
                    "Either 'sequence' or 'token_embeddings' must be provided."
                )
            B = token_embeddings.shape[0]
        D = self.config.dims

        # Sequence encoding: prepend [CLS] / append [SEP], then encode
        if sequence is not None:
            x_with_special = self._prepend_cls_append_sep(sequence)   # [B, T+2]
        else:
            # Construct a placeholder token-ID tensor filled with a non-PAD
            # token (cls_id) so that the pad mask inside the encoder does NOT
            # mask any position.  The actual embedding values are already
            # pre-computed and will override the internal lookup via token_embs.
            T_plus2 = token_embeddings.shape[1]
            x_with_special = mx.full(
                (B, T_plus2), self.config.cls_id, dtype=mx.int32
            )
        hidden = self.malbert_encoder(
            x_with_special,
            token_embs=token_embeddings,          # None on normal path
        )                                          # [B, T+2, D]
        seq_cls = hidden[:, 0, :]                                 # [B, D]

        # Graph representation
        if node_embs is None or edge_src is None:
            # NO_GRAPH fallback: learnable single-token representation per sample
            graph_repr = mx.broadcast_to(
                self.no_graph_embedding.weight,    # [1, D]
                (B, D)
            )
        else:
            # Run GAT: sparse [N_total, D] -> per-graph pooled [B, D]
            graph_repr = self.gat_encoder(
                node_embs, edge_src, edge_dst, batch_idx, n_graphs or B
            )

        # Cross-attention fusion: seq_cls queries over graph_repr
        query = seq_cls[:, None, :]                # [B, 1, D]
        kv = graph_repr[:, None, :]               # [B, 1, D]
        fused_graph = self.cross_attn(query, kv, kv)
        fused_graph = self.fusion_norm(fused_graph[:, 0, :])   # [B, D]

        # Final projection + classify
        fused = nn.gelu(self.fusion_proj(
            mx.concatenate([seq_cls, fused_graph], axis=-1)    # [B, 2D]
        ))
        return self.classifier(fused)                          # [B, C]

    def save_manifest(self, path: str, vocab_sha256: str = "",
                      best_val_macro_f1: float = 0.0,
                      trained_with_pretrained_encoder: bool = False) -> None:
        c = self.config
        Path(path).write_text(json.dumps({
            "arch": "WintermuteMalwareDetector", "version": self.VERSION,
            "vocab_size": c.vocab_size, "num_classes": c.num_classes,
            "dims": c.dims, "num_heads": c.num_heads, "num_layers": c.num_layers,
            "mlp_dims": c.mlp_dims, "dropout": c.dropout,
            "max_seq_length": c.max_seq_length,
            "gat_layers": c.gat_layers, "gat_heads": c.gat_heads,
            "num_fusion_heads": c.num_fusion_heads,
            "pad_id": c.pad_id, "cls_id": c.cls_id,
            "sep_id": c.sep_id, "mask_id": c.mask_id,
            "vocab_sha256": vocab_sha256,
            "best_val_macro_f1": best_val_macro_f1,
            "trained_with_pretrained_encoder": trained_with_pretrained_encoder,
        }, indent=2))

    @classmethod
    def load(cls, weights_path: str, manifest_path: str,
             vocab_sha256: str = "") -> "WintermuteMalwareDetector":
        m = json.loads(Path(manifest_path).read_text())
        if vocab_sha256 and m.get("vocab_sha256") != vocab_sha256:
            raise ValueError(
                f"Vocab SHA256 mismatch: manifest='{m['vocab_sha256']}' "
                f"provided='{vocab_sha256}'. Model and vocab.json must match."
            )
        cfg = DetectorConfig(
            vocab_size=m["vocab_size"], num_classes=m["num_classes"],
            dims=m["dims"], num_heads=m["num_heads"], num_layers=m["num_layers"],
            mlp_dims=m.get("mlp_dims", 1024), dropout=m.get("dropout", 0.1),
            max_seq_length=m.get("max_seq_length", 2048),
            gat_layers=m["gat_layers"], gat_heads=m.get("gat_heads", 4),
            num_fusion_heads=m.get("num_fusion_heads", 4),
            pad_id=m.get("pad_id", 0), cls_id=m.get("cls_id", 2),
            sep_id=m.get("sep_id", 3), mask_id=m.get("mask_id", 4),
        )
        model = cls(cfg)
        model.load_weights(weights_path)
        return model

    @staticmethod
    def cast_to_bf16(model: "WintermuteMalwareDetector") -> None:
        model.apply(lambda x: x.astype(mx.bfloat16))
