"""virustotal.py — VirusTotal API v3 dual-mode ETL data source (enrich + download)."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Iterable

import requests

from wintermute.data.etl.base import DataSource, RawSample
from wintermute.data.etl.pe_utils import PEProcessor, RateLimiter
from wintermute.data.etl.registry import register_source

logger = logging.getLogger("wintermute.data.etl")

VT_API_BASE = "https://www.virustotal.com/api/v3"

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


@register_source("virustotal")
class VirusTotalSource(DataSource):
    """Dual-mode VirusTotal data source: enrich existing .asm caches or download PE binaries.

    **enrich** mode (free tier):
        Reads SHA-256 hashes from a ``hash_file`` or scans ``cache_dirs`` for
        existing ``.asm`` files.  Queries VT ``/files/{id}`` for AV metadata,
        filters by ``min_detection_ratio``, and yields enriched ``RawSample``
        objects with family labels from ``popular_threat_name``.

    **download** mode (Premium API):
        Same as enrich, but also downloads PE binaries via ``/files/{id}/download``
        for hashes that are not already cached.  Downloaded binaries are processed
        through ``PEProcessor.process_pe_binary()``.

    Authentication: API key required (config ``api_key`` or env ``VT_API_KEY``).
    Free-tier limits: 500 lookups/day, 4 requests/minute (default delay=15.0s).
    """

    name = "virustotal"

    def validate_config(self) -> list[str]:
        api_key = self.get("api_key", "") or os.environ.get("VT_API_KEY", "")
        if not api_key:
            return ["VirusTotal API key required. Set config 'api_key' or env var VT_API_KEY."]
        return []

    def extract(self) -> Iterable[RawSample]:
        api_key: str = self.get("api_key", "") or os.environ.get("VT_API_KEY", "")
        cache_dir = Path(self.get("cache_dir", "data/virustotal"))
        mode: str = self.get("mode", "enrich")
        delay: float = self.get("delay", 15.0)
        min_opcodes: int = self.get("min_opcodes", 10)
        max_samples: int = self.get("max_samples", 500)
        hash_file: str = self.get("hash_file", "")
        cache_dirs: list[str] = self.get("cache_dirs", [])
        min_detection_ratio: float = self.get("min_detection_ratio", 0.5)

        processor = PEProcessor(cache_dir=cache_dir, min_opcodes=min_opcodes)
        limiter = RateLimiter(delay=delay)

        # Collect hashes from hash_file or cache_dirs
        hashes = self._get_hashes(hash_file, cache_dirs, max_samples)
        if not hashes:
            logger.warning(
                "VirusTotal: no hashes to process (hash_file=%r, cache_dirs=%r)",
                hash_file,
                cache_dirs,
            )
            return

        logger.info(
            "VirusTotal: processing %d hashes (mode=%s, min_detection_ratio=%.2f)",
            len(hashes),
            mode,
            min_detection_ratio,
        )

        for idx, sha256 in enumerate(hashes):
            limiter.wait()

            # Query VT file report
            report = self._query_file_report(api_key, sha256)
            if report is None:
                logger.warning("[%d/%d] VT report failed: %s", idx + 1, len(hashes), sha256[:16])
                continue

            # Extract detection stats
            attrs = report.get("data", {}).get("attributes", {})
            stats = attrs.get("last_analysis_stats", {})
            malicious = stats.get("malicious", 0)
            undetected = stats.get("undetected", 0)
            total = malicious + undetected
            if total == 0:
                logger.debug(
                    "[%d/%d] No analysis stats for %s — skipping",
                    idx + 1,
                    len(hashes),
                    sha256[:16],
                )
                continue

            detection_ratio = malicious / total

            # Filter by detection ratio
            if detection_ratio < min_detection_ratio:
                logger.debug(
                    "[%d/%d] Detection ratio %.2f < %.2f for %s — skipping",
                    idx + 1,
                    len(hashes),
                    detection_ratio,
                    min_detection_ratio,
                    sha256[:16],
                )
                continue

            # Determine family name and label
            family = attrs.get("popular_threat_name", "") or ""
            label = 1 if detection_ratio >= min_detection_ratio else 0

            # Try to read opcodes from cache directories
            opcodes = self._find_cached_asm(sha256, cache_dirs)

            # Also check the processor's own cache_dir
            if not opcodes and processor.is_cached(sha256):
                opcodes = processor.read_cached_asm(sha256)

            if not opcodes:
                if mode == "download":
                    # Premium: download PE binary from VT
                    pe_bytes = self._download_from_vt(api_key, sha256)
                    if pe_bytes is None:
                        logger.warning(
                            "[%d/%d] VT download failed: %s",
                            idx + 1,
                            len(hashes),
                            sha256[:16],
                        )
                        continue
                    opcodes = processor.process_pe_binary(pe_bytes, sha256)
                    if opcodes is None:
                        continue
                else:
                    # Enrich mode: skip if no cached .asm
                    logger.debug(
                        "[%d/%d] No cached .asm for %s (enrich mode) — skipping",
                        idx + 1,
                        len(hashes),
                        sha256[:16],
                    )
                    continue

            logger.debug(
                "[%d/%d] Enriched: %s (ratio=%.2f, family=%s, %d opcodes)",
                idx + 1,
                len(hashes),
                sha256[:16],
                detection_ratio,
                family or "(unknown)",
                len(opcodes),
            )
            yield RawSample(
                opcodes=opcodes,
                label=label,
                family=family,
                source_id=sha256,
                metadata={"detection_ratio": detection_ratio},
            )

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _get_hashes(hash_file: str, cache_dirs: list[str], max_samples: int) -> list[str]:
        """Collect SHA-256 hashes from a hash file or by scanning cache directories.

        Returns up to *max_samples* unique, valid SHA-256 hashes.
        """
        hashes: list[str] = []
        seen: set[str] = set()

        # Read from hash_file (one hash per line)
        if hash_file:
            path = Path(hash_file)
            if path.is_file():
                for line in path.read_text().splitlines():
                    h = line.strip().lower()
                    if _SHA256_RE.match(h) and h not in seen:
                        hashes.append(h)
                        seen.add(h)
                        if len(hashes) >= max_samples:
                            return hashes

        # Scan cache_dirs for *.asm files with SHA-256 filenames
        if not hashes:
            for dir_path in cache_dirs:
                d = Path(dir_path)
                if not d.is_dir():
                    continue
                for asm_file in d.rglob("*.asm"):
                    stem = asm_file.stem.lower()
                    if _SHA256_RE.match(stem) and stem not in seen:
                        hashes.append(stem)
                        seen.add(stem)
                        if len(hashes) >= max_samples:
                            return hashes

        return hashes

    @staticmethod
    def _query_file_report(api_key: str, sha256: str) -> dict | None:
        """GET ``/files/{sha256}`` — return parsed JSON response or None."""
        url = f"{VT_API_BASE}/files/{sha256}"
        try:
            resp = requests.get(
                url,
                headers={"x-apikey": api_key},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            logger.warning("VT file report failed for %s: %s", sha256[:16], exc)
            return None

    @staticmethod
    def _find_cached_asm(sha256: str, cache_dirs: list[str]) -> list[str] | None:
        """Scan multiple cache directories for a matching .asm file.

        Returns the opcodes list, or None if not found.
        """
        sha_lower = sha256.lower()
        for dir_path in cache_dirs:
            d = Path(dir_path)
            if not d.is_dir():
                continue
            # Check flat layout: dir/<sha256>.asm
            candidate = d / f"{sha_lower}.asm"
            if candidate.is_file():
                text = candidate.read_text()
                opcodes = [line for line in text.splitlines() if line.strip()]
                if opcodes:
                    return opcodes
            # Check nested layout: dir/<family>/<sha256>.asm
            for asm_file in d.rglob(f"{sha_lower}.asm"):
                text = asm_file.read_text()
                opcodes = [line for line in text.splitlines() if line.strip()]
                if opcodes:
                    return opcodes
        return None

    @staticmethod
    def _download_from_vt(api_key: str, sha256: str) -> bytes | None:
        """GET ``/files/{sha256}/download`` — Premium API only.

        Returns the raw PE binary bytes, or None on failure.
        """
        url = f"{VT_API_BASE}/files/{sha256}/download"
        try:
            resp = requests.get(
                url,
                headers={"x-apikey": api_key},
                timeout=60,
            )
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as exc:
            logger.warning("VT download failed for %s: %s", sha256[:16], exc)
            return None
