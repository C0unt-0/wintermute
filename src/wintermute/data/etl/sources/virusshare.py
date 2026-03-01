"""virusshare.py — VirusShare hash-list cross-referencing ETL source.

Cross-references VirusShare hash lists with cached .asm files from other
sources (MalwareBazaar, MalShare, URLhaus, etc.).  Primarily filesystem-based:
no HTTP calls unless an optional API key is configured.

VirusShare is the largest free malware repository (~48M samples).  It
distributes text-file hash lists (one MD5 or SHA-256 hash per line).  For our
ETL pipeline the primary mode is hash-list cross-referencing: read local hash
files, look up matching .asm files in cache directories, and yield them as
``RawSample`` objects.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Iterable

from wintermute.data.etl.base import DataSource, RawSample
from wintermute.data.etl.registry import register_source

logger = logging.getLogger("wintermute.data.etl")

# Regex patterns for hash validation
_MD5_RE = re.compile(r"^[0-9a-fA-F]{32}$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


@register_source("virusshare")
class VirusShareSource(DataSource):
    """Offline VirusShare hash-list cross-referencing source.

    Reads hash lists (one hash per line) from ``hash_dir``, then searches
    ``cache_dirs`` for matching ``.asm`` files.  All matched samples are
    labelled malicious (label=1).

    Accepts both MD5 (32-char) and SHA-256 (64-char) hashes.  MD5 hashes
    are matched against cache filenames that contain the MD5 substring;
    SHA-256 hashes are matched directly by filename.

    Config keys:
        hash_dir:     Path to directory containing .md5 / .txt hash files.
        cache_dirs:   List of directories to search for cached .asm files.
        api_key:      Optional VirusShare API key (env ``VIRUSSHARE_API_KEY``).
        delay:        Seconds between API calls (unused unless api_key set).
        min_opcodes:  Minimum number of opcodes to accept a sample (default: 10).
        max_samples:  Maximum number of samples to yield (null = no limit).
    """

    name = "virusshare"

    def validate_config(self) -> list[str]:
        """Check that ``hash_dir`` exists."""
        errors: list[str] = []
        hash_dir = self.get("hash_dir", "data/virusshare")
        d = Path(hash_dir)

        if not d.is_dir():
            errors.append(f"VirusShare hash_dir does not exist: {hash_dir}")

        return errors

    def extract(self) -> Iterable[RawSample]:
        hash_dir = Path(self.get("hash_dir", "data/virusshare"))
        cache_dirs: list[str] = self.get("cache_dirs", [])
        api_key: str = self.get("api_key", "") or os.environ.get("VIRUSSHARE_API_KEY", "")
        min_opcodes: int = self.get("min_opcodes", 10)
        max_samples: int | None = self.get("max_samples", None)

        # Load unique hashes from hash files
        hashes = self._load_hashes(hash_dir, max_samples)
        if not hashes:
            logger.warning("VirusShare: no hashes found in %s", hash_dir)
            return

        logger.info(
            "VirusShare: loaded %d hashes from %s (cache_dirs=%d)",
            len(hashes),
            hash_dir,
            len(cache_dirs),
        )

        yielded = 0

        for idx, hash_value in enumerate(hashes):
            # Search cache_dirs for matching .asm file
            opcodes = self._find_cached_asm(hash_value, cache_dirs, min_opcodes)

            if opcodes:
                logger.debug(
                    "[%d/%d] Cache hit: %s (%d opcodes)",
                    idx + 1,
                    len(hashes),
                    hash_value[:16],
                    len(opcodes),
                )
                yield RawSample(
                    opcodes=opcodes,
                    label=1,
                    family="",
                    source_id=hash_value,
                    metadata={"dataset": "virusshare"},
                )
                yielded += 1
                if max_samples is not None and yielded >= max_samples:
                    logger.info("VirusShare: reached max_samples limit (%d)", max_samples)
                    return
                continue

            # Not cached — if API key is set, log that download would happen
            if api_key:
                logger.debug(
                    "[%d/%d] Would download %s via VirusShare API (not implemented)",
                    idx + 1,
                    len(hashes),
                    hash_value[:16],
                )
            else:
                logger.debug(
                    "[%d/%d] No cached .asm for %s — skipping",
                    idx + 1,
                    len(hashes),
                    hash_value[:16],
                )

        logger.info("VirusShare: yielded %d samples total", yielded)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _load_hashes(hash_dir: Path, max_samples: int | None) -> list[str]:
        """Read all ``.md5`` and ``.txt`` files in *hash_dir*.

        Returns a list of unique, lowercased hashes (MD5 or SHA-256).
        Lines starting with ``#`` are treated as comments and skipped.
        """
        hashes: list[str] = []
        seen: set[str] = set()

        hash_files = sorted(list(hash_dir.glob("*.md5")) + list(hash_dir.glob("*.txt")))

        for hf in hash_files:
            for line in hf.read_text(errors="ignore").splitlines():
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                h = line.lower()

                # Accept both MD5 (32 chars) and SHA-256 (64 chars)
                if not (_MD5_RE.match(h) or _SHA256_RE.match(h)):
                    continue

                if h in seen:
                    continue

                hashes.append(h)
                seen.add(h)

                if max_samples is not None and len(hashes) >= max_samples:
                    return hashes

        return hashes

    @staticmethod
    def _find_cached_asm(
        hash_value: str,
        cache_dirs: list[str],
        min_opcodes: int,
    ) -> list[str] | None:
        """Search cache directories for a matching .asm file.

        Checks both flat layout (``dir/<hash>.asm``) and nested layout
        (``dir/*/<hash>.asm``).

        Returns the opcodes list if found and meets *min_opcodes* threshold,
        or None otherwise.
        """
        h_lower = hash_value.lower()
        for dir_path in cache_dirs:
            d = Path(dir_path)
            if not d.is_dir():
                continue
            # Check flat layout: dir/<hash>.asm
            candidate = d / f"{h_lower}.asm"
            if candidate.is_file():
                text = candidate.read_text()
                opcodes = [line for line in text.splitlines() if line.strip()]
                if len(opcodes) >= min_opcodes:
                    return opcodes
            # Check nested layout: dir/*/<hash>.asm
            for asm_file in d.rglob(f"{h_lower}.asm"):
                text = asm_file.read_text()
                opcodes = [line for line in text.splitlines() if line.strip()]
                if len(opcodes) >= min_opcodes:
                    return opcodes
        return None
