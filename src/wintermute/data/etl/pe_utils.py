"""pe_utils.py — Shared PE processing infrastructure for ETL data sources.

Extracted from MalwareBazaarDownloader to avoid duplicating PE disassembly,
ZIP extraction, caching, and rate-limiting logic across multiple sources.
"""

from __future__ import annotations

import io
import logging
import os
import re
import tempfile
import time
from pathlib import Path

import pyzipper
import requests

from wintermute.data.tokenizer import extract_opcodes_pe

logger = logging.getLogger("wintermute.data.etl")

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------
class RateLimiter:
    """Simple rate limiter that enforces a minimum delay between calls."""

    def __init__(self, delay: float = 1.0, rpm: int | None = None) -> None:
        if rpm is not None:
            self.delay = 60.0 / rpm
        else:
            self.delay = delay
        self._last_call_time: float = 0.0

    def wait(self) -> None:
        """Sleep for the remaining delay since the last call."""
        now = time.monotonic()
        elapsed = now - self._last_call_time
        remaining = self.delay - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_call_time = time.monotonic()


# ---------------------------------------------------------------------------
# PEProcessor
# ---------------------------------------------------------------------------
class PEProcessor:
    """Shared PE processing utilities: disassembly, caching, download, ZIP extraction."""

    def __init__(
        self,
        cache_dir: Path,
        min_opcodes: int = 10,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.min_opcodes = min_opcodes

    # --- Cache helpers ---

    def _asm_path(self, sha256: str, family: str = "") -> Path:
        """Return the Path for a cached .asm file.

        Raises ValueError if *sha256* is not a valid 64-character hex string.
        """
        if not _SHA256_RE.match(sha256):
            raise ValueError(f"Invalid SHA-256 hash: {sha256!r}")
        if family:
            return self.cache_dir / family / f"{sha256}.asm"
        return self.cache_dir / f"{sha256}.asm"

    def is_cached(self, sha256: str, family: str = "") -> bool:
        """Check if a cached .asm file exists."""
        return self._asm_path(sha256, family).exists()

    def get_cached_path(self, sha256: str, family: str = "") -> Path:
        """Return the Path for a cached .asm file."""
        return self._asm_path(sha256, family)

    def save_asm_cache(self, opcodes: list[str], sha256: str, family: str = "") -> None:
        """Write opcodes to a .asm cache file (one opcode per line)."""
        dest = self._asm_path(sha256, family)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w") as f:
            f.write("\n".join(opcodes))

    def read_cached_asm(self, sha256: str, family: str = "") -> list[str]:
        """Read a cached .asm file and return the list of opcodes.

        Returns an empty list if the file does not exist or cannot be read.
        """
        path = self._asm_path(sha256, family)
        try:
            text = path.read_text()
            return [line for line in text.splitlines() if line.strip()]
        except OSError as exc:
            logger.warning("Failed to read cached asm %s: %s", path, exc)
            return []

    # --- Disassembly ---

    @staticmethod
    def disassemble_pe_bytes(pe_bytes: bytes) -> list[str]:
        """Write PE bytes to a temp file, disassemble with Capstone, return opcodes.

        The temp file is deleted immediately after disassembly.
        """
        tmp = None
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".exe", delete=False)
            tmp.write(pe_bytes)
            tmp.close()
            opcodes = extract_opcodes_pe(tmp.name)
            return opcodes
        except Exception as exc:
            logger.warning("Disassembly failed: %s", exc)
            return []
        finally:
            if tmp and os.path.exists(tmp.name):
                os.unlink(tmp.name)

    # --- ZIP extraction ---

    @staticmethod
    def unzip_encrypted(zip_bytes: bytes, password: str = "infected") -> bytes | None:
        """Extract the first file from an AES-encrypted ZIP.

        Returns the raw file bytes, or None on failure.
        """
        try:
            with pyzipper.AESZipFile(io.BytesIO(zip_bytes)) as zf:
                names = zf.namelist()
                if not names:
                    return None
                return zf.read(names[0], pwd=password.encode())
        except Exception as exc:
            logger.warning("Unzip failed: %s", exc)
            return None

    # --- Download ---

    @staticmethod
    def download_file(
        url: str,
        headers: dict[str, str] | None = None,
        timeout: int = 60,
        method: str = "GET",
        data: dict | bytes | str | None = None,
    ) -> bytes | None:
        """Download a file via HTTP. Returns bytes or None on failure."""
        try:
            if method.upper() == "POST":
                resp = requests.post(url, headers=headers, data=data, timeout=timeout)
            else:
                resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as exc:
            logger.warning("Download failed for %s: %s", url, exc)
            return None

    # --- Full pipeline ---

    def process_pe_binary(
        self,
        pe_bytes: bytes,
        sha256: str,
        family: str = "",
    ) -> list[str] | None:
        """Full processing pipeline: cache check -> disassemble -> filter -> cache -> return.

        Returns the opcodes list, or None if the sample has fewer than
        min_opcodes instructions.
        """
        # Cache hit
        if self.is_cached(sha256, family):
            opcodes = self.read_cached_asm(sha256, family)
            if opcodes:
                return opcodes

        # Cache miss — disassemble
        opcodes = self.disassemble_pe_bytes(pe_bytes)
        if len(opcodes) < self.min_opcodes:
            logger.info(
                "Sample %s has %d opcodes (min %d) — skipping",
                sha256[:16],
                len(opcodes),
                self.min_opcodes,
            )
            return None

        # Cache and return
        self.save_asm_cache(opcodes, sha256, family)
        return opcodes
