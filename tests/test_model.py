"""
test_model.py — Tests for wintermute.models.sequence
"""

import mlx.core as mx
import pytest

from wintermute.models.sequence import MalwareClassifier


class TestMalwareClassifier:
    def test_forward_shape_binary(self):
        """Test output shape for binary classification (2 classes)."""
        model = MalwareClassifier(vocab_size=40, max_seq_length=64, num_classes=2)
        x = mx.random.randint(0, 40, shape=(4, 64))
        logits = model(x)
        mx.eval(logits)
        assert logits.shape == (4, 2)

    def test_forward_shape_multiclass(self):
        """Test output shape for 9-class family detection."""
        model = MalwareClassifier(vocab_size=40, max_seq_length=64, num_classes=9)
        x = mx.random.randint(0, 40, shape=(4, 64))
        logits = model(x)
        mx.eval(logits)
        assert logits.shape == (4, 9)

    def test_single_sample(self):
        """Test with batch size 1."""
        model = MalwareClassifier(vocab_size=40, max_seq_length=32, num_classes=2)
        x = mx.random.randint(0, 40, shape=(1, 32))
        logits = model(x)
        mx.eval(logits)
        assert logits.shape == (1, 2)

    def test_bfloat16_cast(self):
        """Test bfloat16 casting doesn't break forward pass."""
        model = MalwareClassifier(vocab_size=40, max_seq_length=32, num_classes=2)
        MalwareClassifier.cast_to_bf16(model)
        x = mx.random.randint(0, 40, shape=(2, 32))
        logits = model(x)
        mx.eval(logits)
        assert logits.shape == (2, 2)

    def test_parameter_count(self):
        """Test that model has a reasonable parameter count."""
        import mlx.utils

        model = MalwareClassifier(
            vocab_size=100, max_seq_length=2048, dims=128,
            num_heads=4, num_layers=4, mlp_dims=512,
        )
        flat = mlx.utils.tree_flatten(model.parameters())
        n_params = sum(v.size for _, v in flat)
        # Should be around ~1M params
        assert 500_000 < n_params < 2_000_000, (
            f"Expected ~1M params, got {n_params:,}"
        )
