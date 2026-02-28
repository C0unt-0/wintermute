"""synthetic.py — Fake opcode sequences for pipeline testing and CI."""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

from wintermute.data.augment import BENIGN_OPCODES, MALICIOUS_OPCODES, SHARED_OPCODES
from wintermute.data.etl.base import DataSource, RawSample
from wintermute.data.etl.registry import register_source

logger = logging.getLogger("wintermute.data.etl")


@register_source("synthetic")
class SyntheticSource(DataSource):
    """Generate fake opcode sequences with label-dependent distributions.

    Safe samples draw from a benign-biased pool, malicious from a
    suspicious-biased pool. Both pools share common opcodes to make
    classification non-trivial.
    """

    name = "synthetic"

    def extract(self) -> Iterable[RawSample]:
        n_samples = self.get("n_samples", 500)
        max_seq_length = self.get("max_seq_length", 2048)
        seed = self.get("seed", 42)

        rng = np.random.default_rng(seed)
        n_safe = n_samples // 2
        n_mal = n_samples - n_safe
        min_len = min(50, max_seq_length)
        safe_pool = BENIGN_OPCODES + SHARED_OPCODES
        mal_pool = MALICIOUS_OPCODES + SHARED_OPCODES

        for i in range(n_safe):
            seq_len = int(rng.integers(min_len, max_seq_length + 1))
            opcodes = [safe_pool[j] for j in rng.integers(0, len(safe_pool), size=seq_len)]
            yield RawSample(
                opcodes=opcodes,
                label=0,
                family="safe",
                source_id=f"synthetic_safe_{i}",
            )

        for i in range(n_mal):
            seq_len = int(rng.integers(min_len, max_seq_length + 1))
            opcodes = [mal_pool[j] for j in rng.integers(0, len(mal_pool), size=seq_len)]
            yield RawSample(
                opcodes=opcodes,
                label=1,
                family="malicious",
                source_id=f"synthetic_mal_{i}",
            )
