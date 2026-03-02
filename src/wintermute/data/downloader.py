"""
downloader.py — Wintermute Data Downloaders

Consolidated from:
    - 05_download_malwarebazaar.py  (MalwareBazaar API)
    - download_ms_dataset.py        (Microsoft Malware Classification dataset)
"""

from __future__ import annotations

import time
from pathlib import Path

import requests

from wintermute.data.etl.pe_utils import PEProcessor


# ---------------------------------------------------------------------------
# MalwareBazaar Downloader  (from 05_download_malwarebazaar.py)
# ---------------------------------------------------------------------------
BAZAAR_API_URL = "https://mb-api.abuse.ch/api/v1/"
PE_FILE_TYPES = {"exe", "dll", "sys", "ocx", "scr"}


class MalwareBazaarDownloader:
    """Download, disassemble, and save malware samples from MalwareBazaar."""

    def __init__(
        self,
        api_key: str = "",
        out_dir: str | Path = "data/bazaar",
        delay: float = 1.0,
    ):
        self.api_key = api_key
        self.out_dir = Path(out_dir)
        self.delay = delay

    def query_family(self, signature: str, limit: int = 200) -> list[dict]:
        """
        Query MalwareBazaar for recent PE samples matching a family signature.

        Returns a list of sample metadata dicts (sha256_hash, file_type, etc.).
        """
        headers = {"API-KEY": self.api_key} if self.api_key else {}
        params = {
            "query": "get_siginfo",
            "signature": signature,
            "limit": limit,
        }

        try:
            resp = requests.post(BAZAAR_API_URL, data=params, headers=headers, timeout=30)
            resp.raise_for_status()
            result = resp.json()
        except (requests.RequestException, ValueError) as exc:
            print(f"  [ERROR] API query failed for '{signature}': {exc}")
            return []

        if result.get("query_status") != "ok":
            print(f"  [WARN] No results for signature '{signature}': {result.get('query_status')}")
            return []

        samples = result.get("data", [])
        # Filter to PE file types only
        pe_samples = [s for s in samples if s.get("file_type", "").lower() in PE_FILE_TYPES]
        return pe_samples

    def download_sample(self, sha256: str) -> bytes | None:
        """
        Download a malware sample ZIP from MalwareBazaar.

        Returns the raw ZIP bytes, or None on failure.
        """
        headers = {"API-KEY": self.api_key} if self.api_key else {}
        params = {"query": "get_file", "sha256_hash": sha256}

        try:
            resp = requests.post(BAZAAR_API_URL, data=params, headers=headers, timeout=60)
            resp.raise_for_status()
            if resp.headers.get("Content-Type", "").startswith("application/json"):
                # API returned an error in JSON
                return None
            return resp.content
        except requests.RequestException as exc:
            print(f"  [ERROR] Download failed for {sha256[:16]}…: {exc}")
            return None

    @staticmethod
    def unzip_sample(zip_bytes: bytes, password: str = "infected") -> bytes | None:
        """
        Unzip an AES-encrypted MalwareBazaar sample ZIP.

        Returns the raw PE bytes, or None on failure.
        """
        return PEProcessor.unzip_encrypted(zip_bytes, password)

    @staticmethod
    def disassemble_pe_bytes(pe_bytes: bytes) -> list[str]:
        """
        Write PE bytes to a temp file, disassemble with Capstone, return opcodes.

        The temp file is deleted immediately after disassembly.
        """
        return PEProcessor.disassemble_pe_bytes(pe_bytes)

    @staticmethod
    def save_asm_file(opcodes: list[str], dest: Path) -> None:
        """Save an opcode sequence as a .asm text file (one opcode per line)."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w") as f:
            f.write("\n".join(opcodes))

    def download_family(
        self,
        family_name: str,
        signature: str,
        limit: int = 100,
    ) -> dict:
        """
        Download, disassemble, and save samples for one malware family.

        Returns a summary dict with counts.
        """
        family_dir = self.out_dir / family_name
        family_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'─' * 50}")
        print(f"  Family: {family_name} (sig: {signature})")
        print(f"{'─' * 50}")

        samples = self.query_family(signature, limit=limit)
        print(f"  Found {len(samples)} PE samples from API")

        stats = {
            "queried": len(samples),
            "downloaded": 0,
            "disassembled": 0,
            "skipped": 0,
            "failed": 0,
        }

        for i, sample in enumerate(samples, 1):
            sha = sample["sha256_hash"]
            dest = family_dir / f"{sha}.asm"

            if dest.exists():
                stats["skipped"] += 1
                continue

            print(f"  [{i}/{len(samples)}] {sha[:16]}… ", end="", flush=True)

            zip_bytes = self.download_sample(sha)
            if not zip_bytes:
                stats["failed"] += 1
                print("FAIL (download)")
                continue
            stats["downloaded"] += 1

            pe_bytes = self.unzip_sample(zip_bytes)
            if not pe_bytes:
                stats["failed"] += 1
                print("FAIL (unzip)")
                continue

            opcodes = self.disassemble_pe_bytes(pe_bytes)
            if not opcodes:
                stats["failed"] += 1
                print("FAIL (disasm)")
                continue

            self.save_asm_file(opcodes, dest)
            stats["disassembled"] += 1
            print(f"OK ({len(opcodes)} ops)")

            if self.delay > 0:
                time.sleep(self.delay)

        print(f"\n  Summary: {stats}")
        return stats


# ---------------------------------------------------------------------------
# MS Malware Classification Dataset helpers
# ---------------------------------------------------------------------------
class MSDatasetDownloader:
    """Helpers for the Microsoft Malware Classification dataset."""

    # 1-indexed class labels from the Kaggle dataset
    FAMILY_MAP = {
        1: "Ramnit",
        2: "Lollipop",
        3: "Kelihos_ver3",
        4: "Vundo",
        5: "Simda",
        6: "Tracur",
        7: "Kelihos_ver1",
        8: "Obfuscator.ACY",
        9: "Gatak",
    }

    @staticmethod
    def load_labels(labels_path: str | Path) -> dict[str, int]:
        """
        Load labels.csv → {sample_id: class_label (0-indexed)}.

        The CSV has columns: Id, Class
        Class is 1-based in the file; we shift to 0-based for training.
        """
        import csv

        labels: dict[str, int] = {}
        with open(labels_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels[row["Id"]] = int(row["Class"]) - 1  # 0-indexed
        return labels
