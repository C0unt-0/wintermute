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

        # Learnable NO_GRAPH fallback: [1, 1, D] broadcast to [B, 1, D]
        self.no_graph_embedding = mx.zeros((1, 1, D))

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
        sequence: mx.array,                      # [B, T]
        node_features: mx.array | None = None,   # [B, max_N, D] padded
        node_mask: mx.array | None = None,        # [B, max_N] bool
    ) -> mx.array:
        B = sequence.shape[0]
        D = self.config.dims

        # Sequence encoding: prepend [CLS] / append [SEP], then encode
        x_with_special = self._prepend_cls_append_sep(sequence)  # [B, T+2]
        hidden = self.malbert_encoder(x_with_special)            # [B, T+2, D]
        seq_cls = hidden[:, 0, :]                                 # [B, D]

        # Graph representation via cross-attention
        if node_features is None:
            # All NO_GRAPH — broadcast fallback embedding
            graph_kv = mx.broadcast_to(self.no_graph_embedding, (B, 1, D))
            attn_mask = None
        else:
            graph_kv = node_features             # [B, max_N, D]
            if node_mask is not None:
                dtype = self.token_embedding.weight.dtype
                attn_mask = mx.where(
                    node_mask,
                    mx.zeros(node_mask.shape, dtype=dtype),
                    mx.full(node_mask.shape, -1e9, dtype=dtype),
                )[:, None, None, :]
            else:
                attn_mask = None

        # Cross-attention: query [B, 1, D], key/value [B, max_N, D]
        query = seq_cls[:, None, :]
        graph_repr = self.cross_attn(query, graph_kv, graph_kv, mask=attn_mask)
        graph_repr = self.fusion_norm(graph_repr[:, 0, :])   # [B, D]

        # Fuse + classify
        fused = nn.gelu(self.fusion_proj(
            mx.concatenate([seq_cls, graph_repr], axis=-1)   # [B, 2D]
        ))
        return self.classifier(fused)                        # [B, C]

    def save_manifest(self, path: str, vocab_sha256: str = "",
                      best_val_macro_f1: float = 0.0,
                      trained_with_pretrained_encoder: bool = False) -> None:
        c = self.config
        Path(path).write_text(json.dumps({
            "arch": "WintermuteMalwareDetector", "version": self.VERSION,
            "vocab_size": c.vocab_size, "num_classes": c.num_classes,
            "dims": c.dims, "num_heads": c.num_heads, "num_layers": c.num_layers,
            "gat_layers": c.gat_layers, "vocab_sha256": vocab_sha256,
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
            dims=m["dims"], num_heads=m["num_heads"],
            num_layers=m["num_layers"], gat_layers=m["gat_layers"],
        )
        model = cls(cfg)
        model.load_weights(weights_path)
        return model

    @staticmethod
    def cast_to_bf16(model: "WintermuteMalwareDetector") -> None:
        model.apply(lambda x: x.astype(mx.bfloat16))
