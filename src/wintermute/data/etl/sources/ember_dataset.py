"""ember_dataset.py — Offline EMBER dataset ETL source.

Cross-references EMBER JSONL metadata with cached .asm files from other
sources (MalwareBazaar, MalShare, URLhaus, etc.).  Purely filesystem-based:
no HTTP calls are made.

EMBER (Endgame Malware BEnchmark for Research) distributes metadata as JSONL
files where each line is a JSON object containing ``sha256``, ``label``
(0=benign, 1=malicious, -1=unlabeled), and ``avclass`` (family name).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from wintermute.data.etl.base import DataSource, RawSample
from wintermute.data.etl.registry import register_source

logger = logging.getLogger("wintermute.data.etl")


@register_source("ember_dataset")
class EmberDatasetSource(DataSource):
    """Offline EMBER dataset source that yields samples for hashes with cached .asm files.

    Reads EMBER JSONL metadata files, filters by label, and cross-references
    SHA-256 hashes against one or more cache directories.  Only produces
    ``RawSample`` objects for hashes that already have cached ``.asm`` files
    from other ETL sources.

    Config keys:
        data_dir:       Path to the EMBER data directory containing .jsonl files.
        cache_dirs:     List of directories to search for cached .asm files.
        min_opcodes:    Minimum number of opcodes to accept a sample (default: 10).
        max_samples:    Maximum number of samples to yield (null = no limit).
        include_benign: Whether to include label=0 (benign) samples (default: true).
    """

    name = "ember_dataset"

    def validate_config(self) -> list[str]:
        """Check that ``data_dir`` exists and contains at least one ``.jsonl`` file."""
        errors: list[str] = []
        data_dir = self.get("data_dir", "data/ember")
        d = Path(data_dir)

        if not d.is_dir():
            errors.append(f"EMBER data_dir does not exist: {data_dir}")
            return errors

        jsonl_files = list(d.glob("*.jsonl"))
        if not jsonl_files:
            errors.append(f"EMBER data_dir contains no .jsonl files: {data_dir}")

        return errors

    def extract(self) -> Iterable[RawSample]:
        from wintermute.data.tokenizer import read_bazaar_asm

        data_dir = Path(self.get("data_dir", "data/ember"))
        cache_dirs: list[str] = self.get("cache_dirs", [])
        min_opcodes: int = self.get("min_opcodes", 10)
        max_samples: int | None = self.get("max_samples", None)
        include_benign: bool = self.get("include_benign", True)

        # Collect all .jsonl files
        jsonl_files = sorted(data_dir.glob("*.jsonl"))
        if not jsonl_files:
            logger.warning("EMBER: no .jsonl files found in %s", data_dir)
            return

        logger.info(
            "EMBER: reading %d JSONL files from %s (cache_dirs=%d, include_benign=%s)",
            len(jsonl_files),
            data_dir,
            len(cache_dirs),
            include_benign,
        )

        yielded = 0

        for jsonl_path in jsonl_files:
            for line_no, line in enumerate(jsonl_path.open("r", errors="ignore"), start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug(
                        "EMBER: invalid JSON at %s:%d — skipping",
                        jsonl_path.name,
                        line_no,
                    )
                    continue

                label = entry.get("label")
                sha256 = (entry.get("sha256") or "").strip().lower()
                avclass = (entry.get("avclass") or "").strip()

                # Skip unlabeled entries
                if label == -1:
                    continue

                # Skip benign if disabled
                if label == 0 and not include_benign:
                    continue

                # Must have a valid sha256
                if not sha256 or len(sha256) != 64:
                    continue

                # Search cache_dirs for matching .asm file
                opcodes = self._find_cached_asm(sha256, cache_dirs, read_bazaar_asm)
                if opcodes is None:
                    logger.debug("EMBER: no cached .asm for %s — skipping", sha256[:16])
                    continue

                # Filter by min_opcodes
                if len(opcodes) < min_opcodes:
                    logger.debug(
                        "EMBER: %s has %d opcodes < %d — skipping",
                        sha256[:16],
                        len(opcodes),
                        min_opcodes,
                    )
                    continue

                # Normalize label to int (EMBER uses 0/1)
                sample_label = int(label)

                yield RawSample(
                    opcodes=opcodes,
                    label=sample_label,
                    family=avclass,
                    source_id=sha256,
                    metadata={"dataset": "ember"},
                )

                yielded += 1
                if max_samples is not None and yielded >= max_samples:
                    logger.info("EMBER: reached max_samples limit (%d)", max_samples)
                    return

        logger.info("EMBER: yielded %d samples total", yielded)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _find_cached_asm(
        sha256: str,
        cache_dirs: list[str],
        read_fn,
    ) -> list[str] | None:
        """Search cache directories for a matching .asm file.

        Checks both flat layout (``dir/<sha256>.asm``) and nested layout
        (``dir/*/<sha256>.asm``).

        Returns the opcodes list if found, or None otherwise.
        """
        sha_lower = sha256.lower()
        for dir_path in cache_dirs:
            d = Path(dir_path)
            if not d.is_dir():
                continue
            # Check flat layout: dir/<sha256>.asm
            candidate = d / f"{sha_lower}.asm"
            if candidate.is_file():
                opcodes = read_fn(str(candidate))
                if opcodes:
                    return opcodes
            # Check nested layout: dir/*/<sha256>.asm
            for asm_file in d.rglob(f"{sha_lower}.asm"):
                opcodes = read_fn(str(asm_file))
                if opcodes:
                    return opcodes
        return None
