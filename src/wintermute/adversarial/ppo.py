"""
ppo.py — Proximal Policy Optimization in pure MLX.

Patterns copied from joint_trainer.py:
  - nn.value_and_grad(model, loss_fn) for backward pass
  - mlx.utils.tree_flatten(grads) for gradient clipping
  - optim.AdamW for parameter updates
  - mx.eval(model.parameters(), optimizer.state) to materialize
"""

from __future__ import annotations
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils
import numpy as np
from dataclasses import dataclass


@dataclass
class PPOConfig:
    obs_dim: int = 4100          # will be set from env.obs_dim
    n_actions: int = 4           # number of action types
    max_position: int = 2048     # max target position
    hidden_dim: int = 256
    gamma: float = 0.5
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 1e-4
    n_update_epochs: int = 4
    minibatch_size: int = 64


class ActorCritic(nn.Module):
    """Shared-backbone actor-critic. Outputs action logits, position logits, value."""

    def __init__(self, cfg: PPOConfig):
        super().__init__()
        D = cfg.hidden_dim
        self.backbone = nn.Sequential(
            nn.Linear(cfg.obs_dim, D), nn.GELU(),
            nn.Linear(D, D), nn.GELU(),
        )
        self.action_head = nn.Linear(D, cfg.n_actions)
        self.position_head = nn.Linear(D, cfg.max_position)
        self.value_head = nn.Sequential(
            nn.Linear(D, D // 2), nn.GELU(), nn.Linear(D // 2, 1),
        )

    def __call__(self, obs: mx.array):
        """obs: [B, obs_dim] -> (action_logits [B, A], pos_logits [B, P], values [B])"""
        h = self.backbone(obs)
        return self.action_head(h), self.position_head(h), self.value_head(h).squeeze(-1)

    def evaluate_actions(self, obs, actions, positions):
        """Given obs and taken actions, return log_probs, entropy, values."""
        act_logits, pos_logits, values = self(obs)

        act_probs = mx.softmax(act_logits, axis=-1)
        act_lp = mx.log(act_probs[mx.arange(actions.shape[0]), actions] + 1e-8)

        pos_probs = mx.softmax(pos_logits, axis=-1)
        pos_lp = mx.log(pos_probs[mx.arange(positions.shape[0]), positions] + 1e-8)

        log_prob = act_lp + pos_lp

        act_ent = -mx.sum(act_probs * mx.log(act_probs + 1e-8), axis=-1)
        pos_ent = -mx.sum(pos_probs * mx.log(pos_probs + 1e-8), axis=-1)
        entropy = act_ent + pos_ent

        return log_prob, entropy, values


class PPOTrainer:
    """PPO training loop. Uses the same grad clip + AdamW pattern as JointTrainer."""

    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        self.model = ActorCritic(cfg)
        self.optimizer = optim.AdamW(learning_rate=cfg.learning_rate)

    def sample_action(self, obs_np: np.ndarray) -> tuple[int, int, float, float]:
        """
        Given a numpy observation from the env, sample an action.

        Args:
            obs_np: [obs_dim] numpy array

        Returns:
            (action_type: int, position: int, log_prob: float, value: float)
        """
        obs_mx = mx.array(obs_np[np.newaxis, :], dtype=mx.float32)
        act_logits, pos_logits, values = self.model(obs_mx)

        action = mx.random.categorical(act_logits)
        position = mx.random.categorical(pos_logits)

        act_probs = mx.softmax(act_logits, axis=-1)
        pos_probs = mx.softmax(pos_logits, axis=-1)
        log_prob = (
            mx.log(act_probs[0, action[0]] + 1e-8) +
            mx.log(pos_probs[0, position[0]] + 1e-8)
        )

        mx.eval(action, position, log_prob, values)
        return (
            int(action.item()),
            int(position.item()),
            float(log_prob.item()),
            float(values[0].item()),
        )

    def compute_gae(self, rewards, values, dones) -> tuple[np.ndarray, np.ndarray]:
        """Generalized Advantage Estimation. Runs in numpy (sequential)."""
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_v = 0.0 if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.cfg.gamma * next_v * (1 - dones[t]) - values[t]
            last_gae = delta + self.cfg.gamma * self.cfg.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        returns = advantages + np.array(values, dtype=np.float32)
        return advantages, returns

    def update(self, rollout: dict) -> dict:
        """
        PPO clipped surrogate update on collected rollout data.

        Args:
            rollout: dict with numpy arrays — obs, actions, positions,
                     log_probs, advantages, returns

        Returns:
            dict with 'loss' metric
        """
        # Convert numpy rollout -> MLX
        obs = mx.array(rollout["obs"], dtype=mx.float32)
        actions = mx.array(rollout["actions"], dtype=mx.int32)
        positions = mx.array(rollout["positions"], dtype=mx.int32)
        old_lp = mx.array(rollout["log_probs"], dtype=mx.float32)
        advantages = mx.array(rollout["advantages"], dtype=mx.float32)
        returns = mx.array(rollout["returns"], dtype=mx.float32)

        # Normalize advantages
        adv_std = mx.sqrt(mx.var(advantages) + 1e-8)
        advantages = (advantages - mx.mean(advantages)) / adv_std

        total_loss = 0.0
        n_updates = 0
        T = obs.shape[0]

        for _ in range(self.cfg.n_update_epochs):
            idx = np.random.permutation(T)
            for start in range(0, T, self.cfg.minibatch_size):
                end = min(start + self.cfg.minibatch_size, T)
                mb = mx.array(idx[start:end])

                def ppo_loss(model):
                    new_lp, entropy, new_val = model.evaluate_actions(
                        obs[mb], actions[mb], positions[mb]
                    )
                    ratio = mx.exp(new_lp - old_lp[mb])
                    surr1 = ratio * advantages[mb]
                    surr2 = mx.clip(ratio, 1 - self.cfg.clip_range,
                                    1 + self.cfg.clip_range) * advantages[mb]
                    policy_loss = -mx.mean(mx.minimum(surr1, surr2))
                    value_loss = mx.mean((new_val - returns[mb]) ** 2)
                    entropy_loss = -mx.mean(entropy)
                    return (policy_loss
                            + self.cfg.value_coef * value_loss
                            + self.cfg.entropy_coef * entropy_loss)

                loss, grads = nn.value_and_grad(self.model, ppo_loss)(self.model)

                # Gradient clipping — same as joint_trainer.py
                flat = [v for _, v in mlx.utils.tree_flatten(grads)
                        if isinstance(v, mx.array)]
                if flat:
                    norm = mx.sqrt(sum(mx.sum(g * g) for g in flat))
                    mx.eval(norm)
                    if float(norm.item()) > self.cfg.max_grad_norm:
                        scale = self.cfg.max_grad_norm / (float(norm.item()) + 1e-6)
                        grads = mlx.utils.tree_map(
                            lambda g: g * scale if isinstance(g, mx.array) else g,
                            grads,
                        )

                self.optimizer.update(self.model, grads)
                mx.eval(self.model.parameters(), self.optimizer.state)

                total_loss += float(loss.item())
                n_updates += 1

        return {"loss": total_loss / max(n_updates, 1)}
