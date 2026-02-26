import mlx.core as mx
import numpy as np
from wintermute.adversarial.ppo import ActorCritic, PPOConfig, PPOTrainer

class TestActorCritic:
    def test_output_shapes(self):
        cfg = PPOConfig(obs_dim=64, n_actions=4, max_position=16, hidden_dim=32)
        model = ActorCritic(cfg)
        obs = mx.random.normal((4, 64))
        act_logits, pos_logits, values = model(obs)
        assert act_logits.shape == (4, 4)
        assert pos_logits.shape == (4, 16)
        assert values.shape == (4,)

    def test_evaluate_actions_shapes(self):
        cfg = PPOConfig(obs_dim=64, n_actions=4, max_position=16, hidden_dim=32)
        model = ActorCritic(cfg)
        obs = mx.random.normal((4, 64))
        actions = mx.array([0, 1, 2, 3])
        positions = mx.array([0, 5, 10, 15])
        lp, ent, val = model.evaluate_actions(obs, actions, positions)
        assert lp.shape == (4,)
        assert ent.shape == (4,)
        assert val.shape == (4,)

class TestPPOTrainer:
    def test_update_reduces_loss(self):
        cfg = PPOConfig(obs_dim=20, n_actions=4, max_position=8,
                        hidden_dim=16, n_update_epochs=2, minibatch_size=8)
        trainer = PPOTrainer(cfg)
        T = 32
        rollout = {
            "obs": np.random.randn(T, 20).astype(np.float32),
            "actions": np.random.randint(0, 4, T).astype(np.int32),
            "positions": np.random.randint(0, 8, T).astype(np.int32),
            "log_probs": np.random.randn(T).astype(np.float32) * 0.1,
            "advantages": np.random.randn(T).astype(np.float32),
            "returns": np.random.randn(T).astype(np.float32),
        }
        result = trainer.update(rollout)
        assert np.isfinite(result["loss"])
