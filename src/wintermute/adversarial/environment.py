"""
environment.py — Gymnasium environment for adversarial malware mutation.

Framework-agnostic: all observations and actions are numpy arrays.
The defender model is accessed through an injected callable (DefenderBridge).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dataclasses import dataclass
from wintermute.adversarial.actions.code_actions import apply_action
from wintermute.adversarial.reward import compute_reward, RewardConfig


@dataclass
class EnvConfig:
    max_steps: int = 25
    modification_budget: float = 0.15
    vocab_size: int = 287
    max_seq_length: int = 2048
    n_action_types: int = 4       # start with 4 code-level actions
    pad_id: int = 0


class AsmMutationEnv(gym.Env):
    """
    RL environment: mutate malware token sequences to evade detection.

    Action space: MultiDiscrete([n_action_types, max_seq_length])
    Observation space: Box — flattened (normalized_tokens + mod_mask + meta)
    """
    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig, defender_fn, oracle_fn,
                 sample_pool: list, vocab: dict,
                 reward_config: RewardConfig | None = None):
        """
        Args:
            config: environment parameters
            defender_fn: callable(np.ndarray) -> float (confidence)
            oracle_fn: callable(np.ndarray, np.ndarray) -> OracleResult
            sample_pool: list of (tokens_np, label_int, family_str)
            vocab: dict mapping opcode strings to int IDs
            reward_config: reward shaping parameters
        """
        super().__init__()
        self.cfg = config
        self.defender_fn = defender_fn
        self.oracle_fn = oracle_fn
        self.sample_pool = sample_pool
        self.vocab = vocab
        self.reward_config = reward_config or RewardConfig()

        T = config.max_seq_length
        # Observation: normalized tokens + modification mask + 4 meta floats
        obs_dim = T + T + 4
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete(
            [config.n_action_types, config.max_seq_length]
        )

        # Internal state (set in reset)
        self.original_tokens = None
        self.current_tokens = None
        self.mod_mask = None
        self.step_count = 0
        self.n_mutations = 0
        self.last_confidence = 0.0
        self.initial_confidence = 0.0
        self.budget_remaining = 1.0

    def _make_obs(self) -> np.ndarray:
        T = self.cfg.max_seq_length
        norm_tokens = self.current_tokens.astype(np.float32) / max(self.cfg.vocab_size, 1)
        meta = np.array([
            self.budget_remaining,
            self.last_confidence,
            self.step_count / max(self.cfg.max_steps, 1),
            self.n_mutations / max(self.cfg.max_steps, 1),
        ], dtype=np.float32)
        return np.concatenate([norm_tokens, self.mod_mask, meta])

    @property
    def obs_dim(self) -> int:
        return self.cfg.max_seq_length * 2 + 4

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        idx = self.np_random.integers(len(self.sample_pool))
        tokens, label, family = self.sample_pool[idx]

        self.original_tokens = tokens.copy()
        self.current_tokens = tokens.copy()
        self.mod_mask = np.zeros(self.cfg.max_seq_length, dtype=np.float32)
        self.step_count = 0
        self.n_mutations = 0
        self.budget_remaining = 1.0

        self.initial_confidence = self.defender_fn(self.current_tokens)
        self.last_confidence = self.initial_confidence

        return self._make_obs(), {"family": family}

    def step(self, action):
        action_type, target_pos = int(action[0]), int(action[1])
        self.step_count += 1

        # Apply mutation
        mutated, success = apply_action(
            self.current_tokens, action_type, target_pos, self.vocab
        )

        if not success:
            # Action couldn't be applied (e.g., no substitution available)
            reward = -0.1
            obs = self._make_obs()
            truncated = self.step_count >= self.cfg.max_steps
            return obs, reward, False, truncated, {"action_failed": True}

        # Oracle check
        oracle_result = self.oracle_fn(self.original_tokens, mutated)
        if not oracle_result.valid:
            reward = compute_reward(
                self.last_confidence, self.last_confidence,
                oracle_valid=False, budget_remaining=self.budget_remaining,
                config=self.reward_config,
            )
            obs = self._make_obs()
            return obs, reward, True, False, {"oracle_rejected": True}

        # Update state
        self.current_tokens = mutated
        self.mod_mask[target_pos] = 1.0
        self.n_mutations += 1
        changed = np.sum(self.current_tokens != self.original_tokens)
        self.budget_remaining = 1.0 - changed / len(self.current_tokens)

        # Query defender
        new_confidence = self.defender_fn(self.current_tokens)
        reward = compute_reward(
            self.last_confidence, new_confidence,
            oracle_valid=True, budget_remaining=self.budget_remaining,
            config=self.reward_config,
        )
        self.last_confidence = new_confidence

        terminated = self.budget_remaining <= 0
        truncated = self.step_count >= self.cfg.max_steps
        evasion = new_confidence < 0.5

        info = {
            "confidence": new_confidence,
            "evasion": evasion,
            "n_mutations": self.n_mutations,
            "budget": self.budget_remaining,
        }
        return self._make_obs(), reward, terminated, truncated, info
