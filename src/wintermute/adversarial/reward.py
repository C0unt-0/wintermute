"""
reward.py — Shaped reward for the RL attacker agent.

Returns a float reward given the defender's confidence before and after mutation.
Replaces the original sparse {-10, -1, +10, +100} with a smooth gradient signal.
"""

from dataclasses import dataclass


@dataclass
class RewardConfig:
    evasion_bonus: float = 10.0          # defender predicts benign (conf < 0.5)
    confidence_drop_scale: float = 5.0    # reward proportional to confidence drop
    detection_penalty: float = -0.5       # detected with high confidence
    oracle_rejection: float = -5.0        # mutation broke functionality
    step_penalty: float = -0.05           # small per-step cost
    budget_penalty_scale: float = 0.5     # penalize large modifications


def compute_reward(
    confidence_before: float,
    confidence_after: float,
    oracle_valid: bool,
    budget_remaining: float,
    config: RewardConfig | None = None,
) -> float:
    """Compute shaped reward for one mutation step."""
    cfg = config or RewardConfig()

    if not oracle_valid:
        return cfg.oracle_rejection

    # Total evasion: defender thinks it's benign
    if confidence_after < 0.5:
        return cfg.evasion_bonus

    # Confidence dropped meaningfully
    delta = confidence_before - confidence_after
    if delta > 0.01:
        reward = cfg.confidence_drop_scale * delta
        # Budget penalty: penalize mutations that change too much
        budget_used = 1.0 - budget_remaining
        reward -= cfg.budget_penalty_scale * budget_used
        return reward + cfg.step_penalty

    # No progress
    return cfg.detection_penalty + cfg.step_penalty
