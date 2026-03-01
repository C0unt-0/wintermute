"""Tests for AdversarialOrchestrator database integration.

Verifies that the orchestrator correctly persists AdversarialCycle and
AdversarialVariant rows when a db_session is provided, and that the
orchestrator works without errors when db_session is None or when the
DB raises exceptions.

Since the full orchestrator requires PPO, environment, and MLX models,
we mock the heavy components and test the DB integration paths directly.
"""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock, patch

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from wintermute.db.models import (
    AdversarialCycle,
    AdversarialVariant,
)
from wintermute.db.repos.samples import SampleRepo


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_orchestrator(db_session=None):
    """Create an AdversarialOrchestrator with all heavy components mocked.

    Returns (orchestrator, mock_env, mock_ppo, mock_vault) so tests can
    control the behaviour of the environment and PPO trainer.
    """
    from wintermute.adversarial.orchestrator import AdversarialOrchestrator

    # We need to patch __init__ to avoid constructing real PPO/env/etc.
    with patch.object(AdversarialOrchestrator, "__init__", lambda self, *a, **kw: None):
        orch = AdversarialOrchestrator.__new__(AdversarialOrchestrator)

    # Manually set the attributes that __init__ would set
    orch.model = MagicMock()
    orch.vocab = {"nop": 0, "mov": 1, "push": 2, "pop": 3}
    orch.bridge = MagicMock()
    orch.oracle = MagicMock()
    orch.env = MagicMock()
    orch.ppo = MagicMock()
    orch.vault = MagicMock()
    orch.vault.__len__ = MagicMock(return_value=0)
    orch.trades = MagicMock()
    orch._cycle_count = 0
    orch._hook = None
    orch._db_session = db_session
    orch._current_cycle_id = None

    return orch


def _setup_env_for_evasion(orch, n_episodes=1, evasion=True):
    """Configure the mock environment to simulate episodes with evasions.

    Sets up env.reset() and env.step() to produce one step per episode,
    with the evasion flag controllable.
    """
    original_tokens = np.array([1, 2, 3, 4], dtype=np.int32)
    mutated_tokens = np.array([1, 2, 5, 4], dtype=np.int32)
    obs = np.zeros(10, dtype=np.float32)

    # env.reset() returns (obs, info_dict)
    orch.env.reset.return_value = (obs, {"family": "Emotet"})

    # env.step() returns (obs, reward, terminated, truncated, step_info)
    step_info = {
        "confidence": 0.3 if evasion else 0.7,
        "evasion": evasion,
        "n_mutations": 1,
        "budget": 0.9,
    }
    orch.env.step.return_value = (obs, 1.0, True, False, step_info)

    # Environment state accessed during vault/DB storage
    orch.env.current_tokens = mutated_tokens
    orch.env.original_tokens = original_tokens
    orch.env.initial_confidence = 0.95
    orch.env.last_confidence = step_info["confidence"]
    orch.env.n_mutations = 1
    orch.env.current_label = 1

    # PPO sample_action returns (action, position, log_prob, value)
    orch.ppo.sample_action.return_value = (1, 2, -0.5, 0.8)

    # PPO update returns metrics
    orch.ppo.update.return_value = {"loss": 0.1}

    # PPO compute_gae returns (advantages, returns)
    orch.ppo.compute_gae.return_value = (
        np.array([0.5], dtype=np.float32),
        np.array([1.5], dtype=np.float32),
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestOrchestratorDBIntegration:
    def test_orchestrator_creates_cycle(self, db_session: Session):
        """Verify that an AdversarialCycle row is created when db_session is provided."""
        orch = _make_orchestrator(db_session=db_session)
        _setup_env_for_evasion(orch, evasion=False)

        orch.run_cycle(n_episodes=1)

        rows = db_session.execute(select(AdversarialCycle)).scalars().all()
        assert len(rows) == 1
        cycle = rows[0]
        assert cycle.id is not None
        assert cycle.cycle_number == 1
        assert cycle.started_at is not None

    def test_orchestrator_completes_cycle(self, db_session: Session):
        """Verify cycle gets completed_at and stats after run_cycle finishes."""
        orch = _make_orchestrator(db_session=db_session)
        _setup_env_for_evasion(orch, evasion=True)

        orch.run_cycle(n_episodes=1)

        rows = db_session.execute(select(AdversarialCycle)).scalars().all()
        assert len(rows) == 1
        cycle = rows[0]
        assert cycle.completed_at is not None
        assert cycle.episodes_played == 1
        assert cycle.total_evasions == 1
        assert cycle.evasion_rate == 1.0
        assert cycle.mean_confidence_drop is not None
        assert cycle.mean_confidence_drop > 0

    def test_orchestrator_stores_variants(self, db_session: Session):
        """Verify AdversarialVariant rows are created when evasions occur.

        Since store_variant has a FK constraint on parent_sha256 -> samples,
        we need to insert the parent sample first.
        """
        orch = _make_orchestrator(db_session=db_session)
        _setup_env_for_evasion(orch, evasion=True)

        # Pre-create the sample that matches the hash of original_tokens
        original_tokens = np.array([1, 2, 3, 4], dtype=np.int32)
        parent_sha = hashlib.sha256(original_tokens.tobytes()).hexdigest()
        SampleRepo(db_session).upsert(
            sha256=parent_sha,
            family="Emotet",
            label=1,
            source="test",
        )

        orch.run_cycle(n_episodes=1)

        db_session.expire_all()
        variants = db_session.execute(select(AdversarialVariant)).scalars().all()
        assert len(variants) == 1
        v = variants[0]
        assert v.parent_sha256 == parent_sha
        assert v.confidence_before == 0.95
        assert v.confidence_after == 0.3
        assert v.achieved_evasion is True
        assert v.mutation_count == 1

    def test_orchestrator_works_without_db(self):
        """Pass db_session=None: run_cycle should complete without errors."""
        orch = _make_orchestrator(db_session=None)
        _setup_env_for_evasion(orch, evasion=True)

        metrics = orch.run_cycle(n_episodes=1)

        assert metrics["cycle"] == 1
        assert metrics["evasion_rate"] == 1.0
        assert orch._current_cycle_id is None

    def test_orchestrator_db_error_no_crash(self):
        """Mock session to fail: verify no crash and run_cycle completes."""
        mock_session = MagicMock()
        mock_session.flush.side_effect = RuntimeError("DB unavailable")
        mock_session.add.side_effect = RuntimeError("DB unavailable")

        orch = _make_orchestrator(db_session=mock_session)
        _setup_env_for_evasion(orch, evasion=True)

        # Should not raise despite DB errors
        metrics = orch.run_cycle(n_episodes=1)

        assert metrics["cycle"] == 1
        assert metrics["evasion_rate"] == 1.0
        # Cycle ID should be None because start_cycle failed
        assert orch._current_cycle_id is None

    def test_orchestrator_variant_fk_failure_no_crash(self, db_session: Session):
        """When parent sample is missing from DB (FK violation), variant write
        fails gracefully and the cycle still completes."""
        orch = _make_orchestrator(db_session=db_session)
        _setup_env_for_evasion(orch, evasion=True)

        # Don't create the parent sample — FK violation will occur
        metrics = orch.run_cycle(n_episodes=1)

        # Cycle should still complete
        assert metrics["cycle"] == 1
        assert metrics["evasion_rate"] == 1.0

        # Cycle row should exist and be completed
        cycles = db_session.execute(select(AdversarialCycle)).scalars().all()
        assert len(cycles) == 1
        assert cycles[0].completed_at is not None

        # No variant rows (FK violation was caught)
        db_session.expire_all()
        variants = db_session.execute(select(AdversarialVariant)).scalars().all()
        assert len(variants) == 0

    def test_orchestrator_multiple_cycles(self, db_session: Session):
        """Multiple run_cycle calls create separate cycle rows."""
        orch = _make_orchestrator(db_session=db_session)
        _setup_env_for_evasion(orch, evasion=False)

        orch.run_cycle(n_episodes=1)
        orch.run_cycle(n_episodes=1)

        cycles = db_session.execute(select(AdversarialCycle)).scalars().all()
        assert len(cycles) == 2
        assert cycles[0].cycle_number == 1
        assert cycles[1].cycle_number == 2

    def test_vault_add_still_called_with_db(self, db_session: Session):
        """The in-memory vault.add() should still be called even when DB is active."""
        orch = _make_orchestrator(db_session=db_session)
        _setup_env_for_evasion(orch, evasion=True)

        orch.run_cycle(n_episodes=1)

        orch.vault.add.assert_called_once()
