"""urlhaus.py — URLhaus (abuse.ch) PE payload download + disassemble."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import requests

from wintermute.data.etl.base import DataSource, RawSample
from wintermute.data.etl.pe_utils import PEProcessor, RateLimiter
from wintermute.data.etl.registry import register_source

logger = logging.getLogger("wintermute.data.etl")

PAYLOADS_RECENT_URL = "https://urlhaus-api.abuse.ch/v1/payloads/recent/"
DOWNLOAD_URL = "https://urlhaus-api.abuse.ch/v1/download/{sha256}/"


@register_source("urlhaus")
class URLhausSource(DataSource):
    """Download PE payloads from URLhaus, disassemble with Capstone, cache .asm files.

    URLhaus is a public service by abuse.ch — no API key is required.
    Payloads are delivered as AES-encrypted ZIPs (password ``infected``).
    All samples are labelled as malicious (label=1).
    """

    name = "urlhaus"

    def validate_config(self) -> list[str]:
        # No API key needed for URLhaus.
        return []

    def extract(self) -> Iterable[RawSample]:
        cache_dir = Path(self.get("cache_dir", "data/urlhaus"))
        delay = self.get("delay", 1.0)
        min_opcodes = self.get("min_opcodes", 10)
        max_samples = self.get("max_samples", 500)
        recent_limit = self.get("recent_limit", 1000)
        file_types: list[str] = self.get("file_types", ["exe", "dll"])

        processor = PEProcessor(cache_dir=cache_dir, min_opcodes=min_opcodes)
        limiter = RateLimiter(delay=delay)

        # Fetch recent payloads
        payloads = self._fetch_recent_payloads(recent_limit)
        if not payloads:
            logger.warning("No payloads returned from URLhaus recent endpoint")
            return

        # Filter to PE file types only
        pe_payloads = [p for p in payloads if p.get("file_type", "").lower() in file_types]

        # Limit to max_samples
        pe_payloads = pe_payloads[:max_samples]

        if not pe_payloads:
            logger.warning(
                "No PE payloads found among %d recent payloads (filter: %s)",
                len(payloads),
                file_types,
            )
            return

        logger.info(
            "URLhaus: processing %d PE payloads (from %d total, limit=%d)",
            len(pe_payloads),
            len(payloads),
            recent_limit,
        )

        for idx, payload in enumerate(pe_payloads):
            sha256 = payload.get("sha256_hash", "")
            signature = payload.get("signature") or ""
            if not sha256:
                continue

            limiter.wait()

            # Cache hit — skip download
            if processor.is_cached(sha256):
                opcodes = processor.read_cached_asm(sha256)
                if opcodes:
                    logger.debug(
                        "[%d/%d] Cache hit: %s",
                        idx + 1,
                        len(pe_payloads),
                        sha256[:16],
                    )
                    yield RawSample(
                        opcodes=opcodes,
                        label=1,
                        family=signature,
                        source_id=sha256,
                    )
                    continue

            # Download AES-encrypted ZIP
            zip_bytes = self._download_payload(sha256)
            if zip_bytes is None:
                logger.warning(
                    "[%d/%d] Download failed: %s",
                    idx + 1,
                    len(pe_payloads),
                    sha256[:16],
                )
                continue

            # Decrypt ZIP -> raw PE bytes
            pe_bytes = processor.unzip_encrypted(zip_bytes)
            if pe_bytes is None:
                logger.warning(
                    "[%d/%d] Unzip failed: %s",
                    idx + 1,
                    len(pe_payloads),
                    sha256[:16],
                )
                continue

            # Disassemble -> filter -> cache
            opcodes = processor.process_pe_binary(pe_bytes, sha256)
            if opcodes is None:
                continue

            logger.debug(
                "[%d/%d] Processed: %s (%d opcodes, family=%s)",
                idx + 1,
                len(pe_payloads),
                sha256[:16],
                len(opcodes),
                signature or "(unknown)",
            )
            yield RawSample(
                opcodes=opcodes,
                label=1,
                family=signature,
                source_id=sha256,
            )

    # ------------------------------------------------------------------
    # URLhaus API helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_recent_payloads(limit: int) -> list[dict]:
        """POST to ``payloads/recent/`` and return the payloads list."""
        try:
            resp = requests.post(
                PAYLOADS_RECENT_URL,
                data={"limit": limit},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            logger.error("URLhaus recent payloads query failed: %s", exc)
            return []

        if not isinstance(data, dict):
            logger.error("Unexpected URLhaus response type: %s", type(data).__name__)
            return []

        return data.get("payloads", []) or []

    @staticmethod
    def _download_payload(sha256: str) -> bytes | None:
        """GET ``download/{sha256}/`` to fetch the AES-encrypted ZIP."""
        url = DOWNLOAD_URL.format(sha256=sha256)
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as exc:
            logger.warning("URLhaus download failed for %s: %s", sha256[:16], exc)
            return None
