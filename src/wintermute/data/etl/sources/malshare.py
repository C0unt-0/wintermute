"""malshare.py — MalShare REST API download + disassemble."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable

import requests

from wintermute.data.etl.base import DataSource, RawSample
from wintermute.data.etl.pe_utils import PEProcessor, RateLimiter
from wintermute.data.etl.registry import register_source

logger = logging.getLogger("wintermute.data.etl")

BASE_URL = "https://malshare.com/api.php"


@register_source("malshare")
class MalShareSource(DataSource):
    """Download PE samples from MalShare, disassemble with Capstone, cache .asm files.

    MalShare requires a free API key (register at https://malshare.com).
    All samples are labelled as malicious (label=1) since MalShare is a
    malware repository.
    """

    name = "malshare"

    def validate_config(self) -> list[str]:
        api_key = self.get("api_key", "") or os.environ.get("MALSHARE_API_KEY", "")
        if not api_key:
            return ["MalShare API key required. Set config 'api_key' or env var MALSHARE_API_KEY."]
        return []

    def extract(self) -> Iterable[RawSample]:
        api_key = self.get("api_key", "") or os.environ.get("MALSHARE_API_KEY", "")
        cache_dir = Path(self.get("cache_dir", "data/malshare"))
        delay = self.get("delay", 1.0)
        min_opcodes = self.get("min_opcodes", 10)
        max_samples = self.get("max_samples", 500)
        file_type = self.get("file_type", "PE32")

        processor = PEProcessor(cache_dir=cache_dir, min_opcodes=min_opcodes)
        limiter = RateLimiter(delay=delay)

        # Query MalShare for PE sample hashes
        hashes = self._get_sample_hashes(api_key, file_type, max_samples)
        if not hashes:
            logger.warning("No sample hashes returned from MalShare for type '%s'", file_type)
            return

        logger.info(
            "MalShare: processing %d/%d samples (type=%s)",
            len(hashes),
            max_samples,
            file_type,
        )

        for idx, sha256 in enumerate(hashes):
            limiter.wait()

            # Cache hit — skip download
            if processor.is_cached(sha256):
                opcodes = processor.read_cached_asm(sha256)
                if opcodes:
                    logger.debug("[%d/%d] Cache hit: %s", idx + 1, len(hashes), sha256[:16])
                    yield RawSample(
                        opcodes=opcodes,
                        label=1,
                        family="",
                        source_id=sha256,
                    )
                    continue

            # Download raw PE binary
            pe_bytes = self._download_sample(api_key, sha256)
            if pe_bytes is None:
                logger.warning("[%d/%d] Download failed: %s", idx + 1, len(hashes), sha256[:16])
                continue

            # Process: disassemble -> filter -> cache
            opcodes = processor.process_pe_binary(pe_bytes, sha256)
            if opcodes is None:
                continue

            logger.debug(
                "[%d/%d] Processed: %s (%d opcodes)",
                idx + 1,
                len(hashes),
                sha256[:16],
                len(opcodes),
            )
            yield RawSample(
                opcodes=opcodes,
                label=1,
                family="",
                source_id=sha256,
            )

    # ------------------------------------------------------------------
    # MalShare API helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_sample_hashes(api_key: str, file_type: str, max_samples: int) -> list[str]:
        """Query ``action=type`` to get a list of sample hashes filtered by file type.

        Returns up to *max_samples* SHA-256 hashes.
        """
        try:
            resp = requests.get(
                BASE_URL,
                params={"api_key": api_key, "action": "type", "type": file_type},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            logger.error("MalShare type query failed: %s", exc)
            return []

        if not isinstance(data, list):
            logger.error("Unexpected MalShare response type: %s", type(data).__name__)
            return []

        hashes: list[str] = []
        for entry in data:
            sha256 = entry.get("sha256") or entry.get("md5", "")
            if sha256:
                hashes.append(sha256)
            if len(hashes) >= max_samples:
                break

        return hashes

    @staticmethod
    def _download_sample(api_key: str, sha256: str) -> bytes | None:
        """Download a raw PE binary via ``action=getfile``."""
        try:
            resp = requests.get(
                BASE_URL,
                params={"api_key": api_key, "action": "getfile", "hash": sha256},
                timeout=60,
            )
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as exc:
            logger.warning("MalShare download failed for %s: %s", sha256[:16], exc)
            return None
