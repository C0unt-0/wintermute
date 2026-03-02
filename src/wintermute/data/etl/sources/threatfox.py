"""threatfox.py — ThreatFox IOC hash discovery + family labeling source.

Cross-references IOC hashes from ThreatFox with cached .asm files from
other sources (MalwareBazaar, URLhaus, MalShare) and optionally downloads
missing samples via the MalwareBazaar API.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

import requests

from wintermute.data.etl.base import DataSource, RawSample
from wintermute.data.etl.pe_utils import PEProcessor, RateLimiter
from wintermute.data.etl.registry import register_source

logger = logging.getLogger("wintermute.data.etl")

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")

THREATFOX_API_URL = "https://threatfox-api.abuse.ch/api/v1/"
MALWARE_BAZAAR_API_URL = "https://mb-api.abuse.ch/api/v1/"


@register_source("threatfox")
class ThreatFoxSource(DataSource):
    """Hash discovery + family labeling from ThreatFox IOC feed.

    Queries ThreatFox for recent IOCs, filters to SHA-256 hashes, then
    cross-references with cached ``.asm`` files from other ETL sources.
    When ``download_missing`` is True, attempts to download uncached samples
    via the MalwareBazaar API.

    No API key required for basic usage.
    """

    name = "threatfox"

    def validate_config(self) -> list[str]:
        """No API key needed for ThreatFox basic usage."""
        return []

    def extract(self) -> Iterable[RawSample]:
        cache_dirs: list[str] = self.get("cache_dirs", [])
        delay: float = self.get("delay", 1.0)
        min_opcodes: int = self.get("min_opcodes", 10)
        max_samples: int = self.get("max_samples", 500)
        days: int = self.get("days", 7)
        download_missing: bool = self.get("download_missing", True)
        malware_families: list[str] = self.get("malware_families", [])

        processor = PEProcessor(
            cache_dir=Path(cache_dirs[0] if cache_dirs else "data/threatfox"),
            min_opcodes=min_opcodes,
        )
        limiter = RateLimiter(delay=delay)

        # Query ThreatFox for recent IOCs
        iocs = self._query_iocs(days)
        if not iocs:
            logger.warning("ThreatFox: no IOCs returned for the last %d days", days)
            return

        # Filter to sha256_hash IOCs only
        hash_iocs = [ioc for ioc in iocs if ioc.get("ioc_type") == "sha256_hash"]

        # Optionally filter to specific malware families
        if malware_families:
            families_lower = [f.lower() for f in malware_families]
            hash_iocs = [
                ioc for ioc in hash_iocs if (ioc.get("malware") or "").lower() in families_lower
            ]

        # Limit to max_samples
        hash_iocs = hash_iocs[:max_samples]

        if not hash_iocs:
            logger.warning(
                "ThreatFox: no sha256_hash IOCs after filtering (%d total IOCs, families=%r)",
                len(iocs),
                malware_families,
            )
            return

        logger.info(
            "ThreatFox: processing %d hash IOCs (days=%d, families=%r)",
            len(hash_iocs),
            days,
            malware_families or "all",
        )

        for idx, ioc in enumerate(hash_iocs):
            limiter.wait()

            sha256 = ioc["ioc_value"].strip().lower()
            if not _SHA256_RE.match(sha256):
                logger.warning("Invalid SHA-256 in ThreatFox IOC: %s — skipping", sha256[:20])
                continue
            malware_printable = ioc.get("malware_printable", "") or ""
            threat_type = ioc.get("threat_type", "") or ""
            tags = ioc.get("tags") or []

            # Try to find cached .asm in cache_dirs + processor's own cache_dir
            search_dirs = list(cache_dirs)
            if str(processor.cache_dir) not in search_dirs:
                search_dirs.append(str(processor.cache_dir))
            opcodes = self._find_cached_asm(sha256, search_dirs, min_opcodes)

            if opcodes:
                logger.debug(
                    "[%d/%d] Cache hit: %s (family=%s, %d opcodes)",
                    idx + 1,
                    len(hash_iocs),
                    sha256[:16],
                    malware_printable or "(unknown)",
                    len(opcodes),
                )
                yield RawSample(
                    opcodes=opcodes,
                    label=1,
                    family=malware_printable,
                    source_id=sha256,
                    metadata={"threat_type": threat_type, "tags": tags},
                )
                continue

            # Not cached — optionally download from MalwareBazaar
            if download_missing:
                opcodes = self._download_from_bazaar(sha256, processor)
                if opcodes:
                    logger.debug(
                        "[%d/%d] Downloaded: %s (family=%s, %d opcodes)",
                        idx + 1,
                        len(hash_iocs),
                        sha256[:16],
                        malware_printable or "(unknown)",
                        len(opcodes),
                    )
                    yield RawSample(
                        opcodes=opcodes,
                        label=1,
                        family=malware_printable,
                        source_id=sha256,
                        metadata={"threat_type": threat_type, "tags": tags},
                    )
                    continue
                else:
                    logger.debug(
                        "[%d/%d] Download failed: %s",
                        idx + 1,
                        len(hash_iocs),
                        sha256[:16],
                    )
            else:
                logger.debug(
                    "[%d/%d] No cached .asm for %s (download_missing=false) — skipping",
                    idx + 1,
                    len(hash_iocs),
                    sha256[:16],
                )

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _query_iocs(days: int) -> list[dict]:
        """POST to ThreatFox API to get recent IOCs.

        Returns a list of IOC dicts, or an empty list on failure.
        """
        try:
            resp = requests.post(
                THREATFOX_API_URL,
                json={"query": "get_iocs", "days": days},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            logger.error("ThreatFox IOC query failed: %s", exc)
            return []

        if data.get("query_status") != "ok":
            logger.warning(
                "ThreatFox query returned status: %s",
                data.get("query_status", "unknown"),
            )
            return []

        return data.get("data") or []

    @staticmethod
    def _find_cached_asm(sha256: str, cache_dirs: list[str], min_opcodes: int) -> list[str] | None:
        """Scan cache directories for a matching .asm file.

        Checks both flat layout (``dir/<sha256>.asm``) and nested layout
        (``dir/<family>/<sha256>.asm``).

        Returns the opcodes list if found and meets min_opcodes threshold,
        or None otherwise.
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
                if len(opcodes) >= min_opcodes:
                    return opcodes
            # Check nested layout: dir/*/<sha256>.asm
            for asm_file in d.rglob(f"{sha_lower}.asm"):
                text = asm_file.read_text()
                opcodes = [line for line in text.splitlines() if line.strip()]
                if len(opcodes) >= min_opcodes:
                    return opcodes
        return None

    @staticmethod
    def _download_from_bazaar(sha256: str, processor: PEProcessor) -> list[str] | None:
        """Attempt to download a sample from MalwareBazaar and process it.

        POSTs to the MalwareBazaar API with the SHA-256 hash. The response
        is an AES-encrypted ZIP; the PE binary is extracted, disassembled,
        and cached.

        Returns the opcodes list, or None on failure.
        """
        try:
            resp = requests.post(
                MALWARE_BAZAAR_API_URL,
                data={"query": "get_file", "sha256_hash": sha256},
                timeout=60,
            )
            resp.raise_for_status()

            # MalwareBazaar returns the encrypted ZIP directly as bytes
            zip_bytes = resp.content
            if not zip_bytes or len(zip_bytes) < 100:
                logger.debug("MalwareBazaar: empty/small response for %s", sha256[:16])
                return None

            # Unzip (password: "infected")
            pe_bytes = processor.unzip_encrypted(zip_bytes, password="infected")
            if pe_bytes is None:
                logger.debug("MalwareBazaar: unzip failed for %s", sha256[:16])
                return None

            # Process: disassemble -> filter -> cache
            return processor.process_pe_binary(pe_bytes, sha256)

        except requests.RequestException as exc:
            logger.warning("MalwareBazaar download failed for %s: %s", sha256[:16], exc)
            return None
