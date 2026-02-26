"""
oracle.py — Tiered verification of mutated samples.

Tier 1 (this task): token-level structural checks.
  - All token IDs within vocab bounds
  - Sequence length unchanged
  - At least 10% non-PAD tokens remain
  - Modification budget not exceeded

Tier 2 (future): CFG diff via r2pipe
Tier 3 (future): CAPE sandbox execution
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class OracleResult:
    valid: bool
    tier: int
    reason: str = ""


class TieredOracle:
    def __init__(self, vocab_size: int, pad_id: int = 0,
                 max_modification_ratio: float = 0.15):
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.max_mod_ratio = max_modification_ratio

    def validate(self, original_tokens: np.ndarray,
                 mutated_tokens: np.ndarray) -> OracleResult:
        """Run Tier 1 structural validation."""
        # Check 1: same shape
        if original_tokens.shape != mutated_tokens.shape:
            return OracleResult(False, 1, "shape_mismatch")

        # Check 2: all token IDs in vocab range
        if np.any(mutated_tokens < 0) or np.any(mutated_tokens >= self.vocab_size):
            return OracleResult(False, 1, "token_out_of_range")

        # Check 3: not all padding
        non_pad = np.sum(mutated_tokens != self.pad_id)
        total = len(mutated_tokens)
        if non_pad < total * 0.10:
            return OracleResult(False, 1, "too_few_tokens")

        # Check 4: modification budget
        changed = np.sum(original_tokens != mutated_tokens)
        if changed / max(total, 1) > self.max_mod_ratio:
            return OracleResult(False, 1, "budget_exceeded")

        return OracleResult(True, 1)
