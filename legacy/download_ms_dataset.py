#!/usr/bin/env python3
"""
download_ms_dataset.py — Download Microsoft Malware Classification samples

Downloads the sample .asm/.bytes files and labels.csv from the GitHub repo.
For the full 10,868-sample dataset, see Kaggle instructions below.

Usage:
    python src/download_ms_dataset.py
    python src/download_ms_dataset.py --out-dir data/ms-malware
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# GitHub raw URLs
# ---------------------------------------------------------------------------
REPO_BASE = "https://raw.githubusercontent.com/czs108/Microsoft-Malware-Classification/main"
LABELS_URL = f"{REPO_BASE}/data/labels.csv"

# The 3 sample files available on GitHub
SAMPLE_IDS = [
    "0AnoOZDNbPXIr2MRBSCJ",
    "0giIqhw6e4mrHYzKFl8T",
    "0gkj92oIleU4SYiCWpaM",
]


def download_file(url: str, dest: Path) -> bool:
    """Download a file from a URL using curl (avoids Python SSL issues)."""
    try:
        print(f"  Downloading {dest.name} …")
        result = subprocess.run(
            ["curl", "-sL", "-o", str(dest), url],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
            return True
        print(f"  [WARN] curl failed for {url}")
        dest.unlink(missing_ok=True)
        return False
    except Exception as e:
        print(f"  [WARN] Failed to download {url}: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Microsoft Malware Classification dataset samples")
    parser.add_argument("--out-dir", type=str, default="data/ms-malware",
                        help="Output directory for downloaded files.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download labels.csv ---------------------------------------------------
    labels_dest = out_dir / "labels.csv"
    if labels_dest.exists():
        print(f"✓ labels.csv already exists at {labels_dest}")
    else:
        download_file(LABELS_URL, labels_dest)

    # 2. Download sample .asm and .bytes files ---------------------------------
    downloaded = 0
    for sample_id in SAMPLE_IDS:
        for ext in [".asm", ".bytes"]:
            filename = f"{sample_id}{ext}"
            dest = out_dir / filename
            if dest.exists():
                print(f"  ✓ {filename} already exists")
                downloaded += 1
                continue
            url = f"{REPO_BASE}/data/samples/{filename}"
            if download_file(url, dest):
                downloaded += 1

    print(f"\n✅  Downloaded {downloaded} files to {out_dir}/")

    # 3. Instructions for full dataset -----------------------------------------
    print(f"""
{'─' * 60}
📦  Full Dataset (10,868 samples)
{'─' * 60}

The GitHub repo only contains 3 sample files. For the full
dataset, download from the Kaggle competition:

  https://www.kaggle.com/c/malware-classification/data

Steps:
  1. Create a Kaggle account and accept the competition rules
  2. Download the 'train' partition
  3. Extract the .asm files into: {out_dir}/
  4. Ensure labels.csv is also in: {out_dir}/

Then run:
  python src/04_build_ms_dataset.py --samples-dir {out_dir}
{'─' * 60}
""")


if __name__ == "__main__":
    main()
