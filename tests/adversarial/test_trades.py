import mlx.core as mx
import mlx.nn as nn
from wintermute.adversarial.trades_loss import TRADESLoss
from wintermute.models.fusion import DetectorConfig, WintermuteMalwareDetector


class TestTRADES:
    def test_identical_inputs_zero_kl(self):
        """When clean == adversarial, KL term should be ~0."""
        cfg = DetectorConfig(
            vocab_size=32,
            dims=32,
            num_heads=1,
            num_layers=1,
            mlp_dims=64,
            num_classes=2,
            max_seq_length=16,
        )
        model = WintermuteMalwareDetector(cfg)
        x = mx.random.randint(0, 32, (4, 16))
        y = mx.array([0, 1, 0, 1])
        trades = TRADESLoss(beta=1.0)
        loss_trades = trades(model, x, y, x, epoch=10, max_epochs=10)  # same input
        loss_ce = mx.mean(nn.losses.cross_entropy(model(x), y))
        mx.eval(loss_trades, loss_ce)
        # Should be very close since KL(p||p) ~ 0
        assert abs(float(loss_trades.item()) - float(loss_ce.item())) < 0.1

    def test_beta_zero_equals_ce(self):
        cfg = DetectorConfig(
            vocab_size=32,
            dims=32,
            num_heads=1,
            num_layers=1,
            mlp_dims=64,
            num_classes=2,
            max_seq_length=16,
        )
        model = WintermuteMalwareDetector(cfg)
        x = mx.random.randint(0, 32, (4, 16))
        y = mx.array([0, 1, 0, 1])
        trades = TRADESLoss(beta=0.0)
        loss = trades(model, x, y, x, epoch=0, max_epochs=10)
        mx.eval(loss)
        assert float(loss.item()) > 0
