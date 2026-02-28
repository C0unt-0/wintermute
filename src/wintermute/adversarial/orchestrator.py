"""
orchestrator.py — Coordinates the adversarial training loop.

1. Collect rollouts from AsmMutationEnv (numpy)
2. Update PPO agent (MLX)
3. Store evasive samples in vault (numpy)
4. Retrain defender with TRADES loss + vault replay (MLX)
"""

from __future__ import annotations
import hashlib
import logging
import numpy as np

from wintermute.adversarial.ppo import PPOTrainer, PPOConfig
from wintermute.adversarial.environment import AsmMutationEnv, EnvConfig
from wintermute.adversarial.oracle import TieredOracle
from wintermute.adversarial.bridge import DefenderBridge
from wintermute.adversarial.vault import AdversarialVault, VaultConfig, VaultEntry
from wintermute.adversarial.trades_loss import TRADESLoss
from wintermute.models.fusion import WintermuteMalwareDetector

_db_log = logging.getLogger("wintermute.db")


class AdversarialOrchestrator:
    """
    Main entry point for adversarial training.

    Usage:
        orch = AdversarialOrchestrator(model, vocab, sample_pool)
        for cycle in range(n_cycles):
            metrics = orch.run_cycle(n_episodes=500)
            print(metrics)
    """

    def __init__(
        self,
        model: WintermuteMalwareDetector,
        vocab: dict,
        sample_pool: list[tuple[np.ndarray, int, str]],
        env_config: EnvConfig | None = None,
        ppo_config: PPOConfig | None = None,
        vault_config: VaultConfig | None = None,
        trades_beta: float = 1.0,
        hook=None,
        db_session=None,
    ):
        self.model = model
        self.vocab = vocab

        # Defender bridge (MLX model <-> numpy env)
        self.bridge = DefenderBridge(model)

        # Oracle
        self.oracle = TieredOracle(vocab_size=len(vocab))

        # Environment
        env_cfg = env_config or EnvConfig(
            vocab_size=len(vocab),
            max_seq_length=sample_pool[0][0].shape[0] if sample_pool else 2048,
            n_action_types=4,
        )
        self.env = AsmMutationEnv(
            config=env_cfg,
            defender_fn=self.bridge,
            oracle_fn=lambda orig, mut: self.oracle.validate(orig, mut),
            sample_pool=sample_pool,
            vocab=vocab,
        )

        # PPO agent
        ppo_cfg = ppo_config or PPOConfig(
            obs_dim=self.env.obs_dim,
            n_actions=env_cfg.n_action_types,
            max_position=env_cfg.max_seq_length,
        )
        self.ppo = PPOTrainer(ppo_cfg)

        # Vault
        self.vault = AdversarialVault(vault_config or VaultConfig())

        # TRADES loss for defender retraining
        self.trades = TRADESLoss(beta=trades_beta)

        self._cycle_count = 0
        self._hook = hook
        self._db_session = db_session
        self._current_cycle_id = None

    def run_cycle(self, n_episodes: int = 500) -> dict:
        """Run one full adversarial cycle: attack -> PPO update -> store evasions."""
        self._cycle_count += 1

        # --- DB: start cycle row ---
        self._current_cycle_id = None
        if self._db_session is not None:
            try:
                from wintermute.db.repos.adversarial import AdversarialRepo

                cycle = AdversarialRepo(self._db_session).start_cycle(
                    cycle_number=self._cycle_count,
                )
                self._current_cycle_id = cycle.id
            except Exception:
                _db_log.warning("Failed to create AdversarialCycle row", exc_info=True)

        print(f"\n{'=' * 60}")
        print(f" Adversarial Cycle {self._cycle_count}")
        print(f"{'=' * 60}")

        # Step 1: Collect rollouts
        print(f"\nCollecting {n_episodes} episodes...")
        rollout_data, episode_stats = self._collect_rollouts(n_episodes)

        # Step 2: PPO update
        print("Updating PPO agent...")
        ppo_metrics = self.ppo.update(rollout_data)

        # Step 3: Log stats
        evasion_rate = episode_stats["evasions"] / max(n_episodes, 1)
        print(f"   Episodes: {n_episodes}")
        print(f"   Evasions: {episode_stats['evasions']} ({evasion_rate:.1%})")
        print(f"   PPO loss: {ppo_metrics['loss']:.4f}")
        print(f"   Vault size: {len(self.vault)}")

        metrics = {
            "cycle": self._cycle_count,
            "evasion_rate": evasion_rate,
            "ppo_loss": ppo_metrics["loss"],
            "vault_size": len(self.vault),
            "mean_confidence": episode_stats["mean_final_confidence"],
            "mean_mutations": episode_stats["mean_mutations"],
        }
        # --- DB: complete cycle row ---
        if self._db_session is not None and self._current_cycle_id is not None:
            try:
                from wintermute.db.repos.adversarial import AdversarialRepo

                AdversarialRepo(self._db_session).complete_cycle(
                    self._current_cycle_id,
                    stats={
                        "episodes_played": n_episodes,
                        "total_evasions": episode_stats["evasions"],
                        "evasion_rate": evasion_rate,
                        "mean_confidence_drop": 1.0 - episode_stats["mean_final_confidence"],
                    },
                )
            except Exception:
                _db_log.warning("Failed to complete AdversarialCycle", exc_info=True)

        if self._hook:
            self._hook.on_cycle_end(self._cycle_count, metrics)
        return metrics

    def _collect_rollouts(self, n_episodes: int) -> tuple[dict, dict]:
        """Collect experience from env. Numpy throughout, convert to MLX in PPO.update()."""
        all_obs, all_actions, all_positions = [], [], []
        all_log_probs, all_values, all_rewards, all_dones = [], [], [], []

        n_evasions = 0
        final_confidences = []
        mutations_per_ep = []

        for _ in range(n_episodes):
            obs, info = self.env.reset()
            ep_rewards = []
            ep_values = []
            done = False

            while not done:
                action, position, log_prob, value = self.ppo.sample_action(obs)

                next_obs, reward, terminated, truncated, step_info = self.env.step(
                    np.array([action, position])
                )
                done = terminated or truncated

                all_obs.append(obs)
                all_actions.append(action)
                all_positions.append(position)
                all_log_probs.append(log_prob)
                all_values.append(value)
                all_rewards.append(reward)
                all_dones.append(float(done))

                ep_rewards.append(reward)
                ep_values.append(value)
                obs = next_obs

                # Track evasions and store in vault
                if step_info.get("evasion", False):
                    n_evasions += 1
                    self.vault.add(
                        VaultEntry(
                            mutated_tokens=self.env.current_tokens.copy(),
                            label=self.env.current_label
                            if hasattr(self.env, "current_label")
                            else 1,
                            family=info.get("family", "unknown"),
                            evasion_confidence=step_info["confidence"],
                            action_types_used=[action],
                            n_mutations=step_info.get("n_mutations", 0),
                            epoch=self._cycle_count,
                        )
                    )

                    # --- DB: store adversarial variant ---
                    if self._db_session is not None and self._current_cycle_id is not None:
                        try:
                            from wintermute.db.repos.adversarial import AdversarialRepo

                            parent_sha = hashlib.sha256(
                                self.env.original_tokens.tobytes()
                            ).hexdigest()
                            tokens = self.env.current_tokens
                            # Use a savepoint so FK violations don't poison
                            # the session for subsequent operations.
                            nested = self._db_session.begin_nested()
                            try:
                                AdversarialRepo(self._db_session).store_variant(
                                    parent_sha256=parent_sha,
                                    cycle_id=self._current_cycle_id,
                                    mutated_tokens=tokens.tolist()
                                    if hasattr(tokens, "tolist")
                                    else list(tokens),
                                    mutations=[{"action": int(action)}],
                                    confidence_before=self.env.initial_confidence,
                                    confidence_after=float(step_info["confidence"]),
                                    modification_pct=step_info.get("n_mutations", 0)
                                    / max(len(self.env.current_tokens), 1)
                                    * 100,
                                )
                            except Exception:
                                nested.rollback()
                                raise
                        except Exception:
                            _db_log.warning("Failed to store adversarial variant", exc_info=True)

            final_confidences.append(self.env.last_confidence)
            mutations_per_ep.append(self.env.n_mutations)

        # Compute GAE
        advantages, returns = self.ppo.compute_gae(all_rewards, all_values, all_dones)

        rollout = {
            "obs": np.stack(all_obs).astype(np.float32),
            "actions": np.array(all_actions, dtype=np.int32),
            "positions": np.array(all_positions, dtype=np.int32),
            "log_probs": np.array(all_log_probs, dtype=np.float32),
            "advantages": advantages,
            "returns": returns,
        }

        stats = {
            "evasions": n_evasions,
            "mean_final_confidence": float(np.mean(final_confidences))
            if final_confidences
            else 0.0,
            "mean_mutations": float(np.mean(mutations_per_ep)) if mutations_per_ep else 0.0,
        }

        return rollout, stats
