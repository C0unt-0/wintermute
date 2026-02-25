"""
test_malbert.py — Tests for MalBERT architecture and augmentation

Tests for:
    - MalBERTConfig defaults
    - MalBERT forward pass in classify mode
    - MalBERT forward pass in MLM mode
    - Padding mask generation
    - [CLS] token encoding
    - MalBERTForMLM wrapper
    - SMOTEAugmenter
    - HeuristicAugmenter
    - MLM masking function
"""

import numpy as np

import mlx.core as mx

from wintermute.models.transformer import (
    MalBERT,
    MalBERTConfig,
    MalBERTForMLM,
)
from wintermute.data.augment import HeuristicAugmenter, SMOTEAugmenter
from wintermute.engine.pretrain import apply_mlm_masking


# ═══════════════════════════════════════════════════════════════════════════
# MalBERT Architecture Tests
# ═══════════════════════════════════════════════════════════════════════════
class TestMalBERTConfig:
    def test_defaults(self):
        cfg = MalBERTConfig()
        assert cfg.dims == 256
        assert cfg.num_heads == 8
        assert cfg.num_layers == 6
        assert cfg.mlp_dims == 1024
        assert cfg.dropout == 0.1
        assert cfg.num_classes == 9
        assert cfg.pad_id == 0
        assert cfg.cls_id == 2
        assert cfg.sep_id == 3
        assert cfg.mask_id == 4


class TestMalBERT:
    """Test MalBERT model forward passes."""

    def _make_model(self, vocab_size=64, seq_len=32, num_classes=9):
        cfg = MalBERTConfig(
            vocab_size=vocab_size,
            max_seq_length=seq_len,
            dims=64,        # small for tests
            num_heads=4,
            num_layers=2,
            mlp_dims=128,
            dropout=0.0,    # no dropout for deterministic tests
            num_classes=num_classes,
        )
        return MalBERT(cfg), cfg

    def test_classify_shape(self):
        """Classify mode should output [B, num_classes]."""
        model, cfg = self._make_model(num_classes=9)
        x = mx.zeros((4, 32), dtype=mx.int32)
        logits = model(x, mode="classify")
        mx.eval(logits)
        assert logits.shape == (4, 9)

    def test_mlm_shape(self):
        """MLM mode should output [B, T+2, vocab_size]."""
        model, cfg = self._make_model(vocab_size=64)
        x = mx.zeros((4, 32), dtype=mx.int32)
        logits = model(x, mode="mlm")
        mx.eval(logits)
        # T+2 because of [CLS] and [SEP]
        assert logits.shape == (4, 34, 64)

    def test_cls_prepend(self):
        """Should prepend CLS and append SEP."""
        model, cfg = self._make_model()
        x = mx.ones((2, 10), dtype=mx.int32) * 5
        x_special = model._prepend_cls_append_sep(x)
        mx.eval(x_special)
        assert x_special.shape == (2, 12)
        assert x_special[0, 0].item() == cfg.cls_id
        assert x_special[0, -1].item() == cfg.sep_id

    def test_encode(self):
        """encode() should return [B, D] embeddings."""
        model, cfg = self._make_model()
        x = mx.zeros((3, 32), dtype=mx.int32)
        emb = model.encode(x)
        mx.eval(emb)
        assert emb.shape == (3, 64)  # dims=64

    def test_bfloat16_cast(self):
        model, _ = self._make_model()
        MalBERT.cast_to_bf16(model)
        x = mx.zeros((2, 32), dtype=mx.int32)
        logits = model(x)
        mx.eval(logits)
        assert logits.dtype == mx.bfloat16

    def test_padding_mask(self):
        """PAD tokens should get -inf in attention mask."""
        model, cfg = self._make_model()
        # All zeros = all PAD
        x = mx.zeros((1, 10), dtype=mx.int32)
        mask = model.encoder._make_pad_mask(x)
        mx.eval(mask)
        # All positions should be masked (close to -inf)
        assert mask[0, 0, 0, 0].item() < -1e8

        # Non-PAD tokens should have 0
        x_real = mx.ones((1, 10), dtype=mx.int32) * 5
        mask_real = model.encoder._make_pad_mask(x_real)
        mx.eval(mask_real)
        assert mask_real[0, 0, 0, 0].item() == 0.0

    def test_invalid_mode(self):
        model, _ = self._make_model()
        x = mx.zeros((1, 10), dtype=mx.int32)
        try:
            model(x, mode="invalid")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestMalBERTForMLM:
    def test_forward(self):
        cfg = MalBERTConfig(
            vocab_size=32, max_seq_length=16,
            dims=32, num_heads=2, num_layers=1, mlp_dims=64,
            dropout=0.0, num_classes=2,
        )
        wrapper = MalBERTForMLM(cfg)
        x = mx.zeros((2, 16), dtype=mx.int32)
        logits = wrapper(x)
        mx.eval(logits)
        assert logits.shape == (2, 18, 32)  # T+2, vocab_size


# ═══════════════════════════════════════════════════════════════════════════
# MLM Masking Tests
# ═══════════════════════════════════════════════════════════════════════════
class TestMLMMasking:
    def test_masking_shape(self):
        cfg = MalBERTConfig(vocab_size=64, pad_id=0, mask_id=4)
        x = mx.ones((4, 32), dtype=mx.int32) * 10
        masked, labels, mask_pos = apply_mlm_masking(x, cfg, mask_prob=0.15)
        assert masked.shape == x.shape
        assert labels.shape == x.shape
        assert mask_pos.shape == x.shape

    def test_pad_not_masked(self):
        """PAD tokens should never be masked."""
        cfg = MalBERTConfig(vocab_size=64, pad_id=0, mask_id=4)
        x = mx.zeros((4, 32), dtype=mx.int32)  # all PAD
        _, labels, _ = apply_mlm_masking(x, cfg, mask_prob=0.5)
        # All labels should be -100 (no masking)
        assert mx.all(labels == -100).item()

    def test_some_tokens_masked(self):
        """With enough tokens, some should be masked."""
        cfg = MalBERTConfig(vocab_size=64, pad_id=0, mask_id=4)
        rng = np.random.default_rng(42)
        x = mx.array(rng.integers(5, 60, size=(8, 100)))
        _, labels, _ = apply_mlm_masking(x, cfg, mask_prob=0.15, rng=rng)
        # Should have some non -100 labels
        n_masked = mx.sum(labels != -100).item()
        assert n_masked > 0


# ═══════════════════════════════════════════════════════════════════════════
# Augmentation Tests
# ═══════════════════════════════════════════════════════════════════════════
class TestSMOTEAugmenter:
    def test_balanced_output(self):
        rng = np.random.default_rng(42)
        x = np.vstack([
            rng.integers(0, 50, size=(20, 32)),   # 20 samples class 0
            rng.integers(0, 50, size=(5, 32)),     # 5 samples class 1
        ]).astype(np.int32)
        y = np.array([0]*20 + [1]*5, dtype=np.int32)

        aug = SMOTEAugmenter(k_neighbors=3, seed=42)
        x_aug, y_aug = aug.augment(x, y, target_ratio=1.0)

        # Should have more samples than original
        assert len(y_aug) > len(y)
        # Class 1 should now have ~20 samples
        assert np.sum(y_aug == 1) >= 20

    def test_already_balanced(self):
        """If already balanced, should return unchanged."""
        x = np.ones((10, 16), dtype=np.int32)
        y = np.array([0]*5 + [1]*5, dtype=np.int32)
        aug = SMOTEAugmenter(seed=42)
        x_aug, y_aug = aug.augment(x, y)
        assert len(y_aug) == len(y)


class TestHeuristicAugmenter:
    def test_nop_insertion(self):
        aug = HeuristicAugmenter(seed=42)
        ops = ["mov", "push", "call", "ret"]
        result = aug.augment_sequence(ops, techniques=["nop"])
        assert len(result) > len(ops)
        assert "nop" in result

    def test_dead_code_insertion(self):
        aug = HeuristicAugmenter(seed=42)
        ops = ["mov", "push", "call", "ret"]
        result = aug.augment_sequence(ops, techniques=["dead_code"])
        assert len(result) > len(ops)

    def test_reorder(self):
        """Reordering should preserve all instructions."""
        aug = HeuristicAugmenter(seed=42)
        ops = ["mov", "push", "add", "sub", "xor", "nop"] * 5
        result = aug.augment_sequence(ops, techniques=["reorder"])
        # Same instructions, possibly different order
        assert sorted(result) == sorted(ops)

    def test_augment_dataset(self):
        """augment_dataset should produce larger dataset."""
        stoi = {"<PAD>": 0, "<UNK>": 1, "mov": 2, "push": 3, "nop": 4}
        x = np.array([[2, 3, 2, 3, 0, 0, 0, 0]] * 10, dtype=np.int32)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)

        aug = HeuristicAugmenter(seed=42)
        x_aug, y_aug = aug.augment_dataset(
            x, y, stoi, augment_ratio=0.5, max_seq_length=8,
        )
        assert len(y_aug) > len(y)
