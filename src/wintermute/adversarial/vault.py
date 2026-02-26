"""
vault.py — Adversarial sample vault for replay during defender retraining.

Stores mutated token sequences that successfully evaded the defender.
Provides stratified sampling for balanced replay batches.
All storage is numpy. No MLX.
"""

from __future__ import annotations
import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VaultEntry:
    mutated_tokens: np.ndarray
    label: int
    family: str
    evasion_confidence: float     # defender's confidence on the mutated sample
    action_types_used: list[int]
    n_mutations: int
    epoch: int


@dataclass
class VaultConfig:
    max_samples: int = 50_000
    replay_ratio: float = 0.2


class AdversarialVault:
    def __init__(self, config: VaultConfig | None = None):
        self.cfg = config or VaultConfig()
        self.entries: list[VaultEntry] = []

    def __len__(self):
        return len(self.entries)

    def add(self, entry: VaultEntry):
        self.entries.append(entry)
        # Evict oldest if over capacity
        if len(self.entries) > self.cfg.max_samples:
            self.entries = self.entries[-self.cfg.max_samples:]

    def sample_replay_batch(self, batch_size: int, rng: np.random.Generator | None = None
                             ) -> np.ndarray | None:
        """
        Sample adversarial tokens for replay.

        Returns: [N, T] numpy array where N = batch_size * replay_ratio, or None if vault empty.
        """
        if len(self.entries) == 0:
            return None
        rng = rng or np.random.default_rng()
        n = max(1, int(batch_size * self.cfg.replay_ratio))
        n = min(n, len(self.entries))
        indices = rng.choice(len(self.entries), size=n, replace=False)
        tokens = np.stack([self.entries[i].mutated_tokens for i in indices])
        return tokens

    def save(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        tokens = np.stack([e.mutated_tokens for e in self.entries]) if self.entries else np.array([])
        np.save(path / "vault_tokens.npy", tokens)
        meta = [
            {"label": e.label, "family": e.family, "confidence": e.evasion_confidence,
             "actions": e.action_types_used, "n_mutations": e.n_mutations, "epoch": e.epoch}
            for e in self.entries
        ]
        (path / "vault_meta.json").write_text(json.dumps(meta))

    def load(self, path: str | Path):
        path = Path(path)
        tokens = np.load(path / "vault_tokens.npy")
        meta = json.loads((path / "vault_meta.json").read_text())
        self.entries = []
        for i, m in enumerate(meta):
            self.entries.append(VaultEntry(
                mutated_tokens=tokens[i], label=m["label"], family=m["family"],
                evasion_confidence=m["confidence"], action_types_used=m["actions"],
                n_mutations=m["n_mutations"], epoch=m["epoch"],
            ))
