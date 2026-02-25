#!/usr/bin/env python3
"""
01_build_dataset.py — Wintermute Data Pipeline

Disassembles Windows PE files into opcode sequences and serialises them
as NumPy arrays ready for the MLX training loop.

Usage:
    python src/01_build_dataset.py                          # defaults
    python src/01_build_dataset.py --data-dir ./data        # custom paths
    python src/01_build_dataset.py --max-seq-length 1024    # shorter seqs
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import capstone
import numpy as np
import pefile


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_SEQ_LENGTH = 2048
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN]


# ---------------------------------------------------------------------------
# Phase A: Extraction
# ---------------------------------------------------------------------------
def extract_opcodes(filepath: str) -> list[str]:
    """
    Disassemble a PE file and return an ordered list of opcode mnemonics.

    Steps
    -----
    1. Parse the PE header with ``pefile``.
    2. Locate the ``.text`` section (primary executable code).
    3. Feed the raw bytes into the Capstone disassembler (x86‑64 first,
       falling back to x86‑32).
    4. Return **only** the mnemonics – operands, registers, and memory
       addresses are intentionally stripped because they change on every
       recompilation.
    """
    try:
        pe = pefile.PE(filepath)
    except pefile.PEFormatError as exc:
        print(f"  [SKIP] {filepath}: invalid PE — {exc}")
        return []

    # Choose disassembly mode based on the PE's "Optional Header" magic.
    if pe.OPTIONAL_HEADER.Magic == 0x20B:          # PE32+ (64-bit)
        mode = capstone.CS_MODE_64
    else:                                           # PE32  (32-bit)
        mode = capstone.CS_MODE_32

    # Find the .text section ---------------------------------------------------
    text_section = None
    for section in pe.sections:
        name = section.Name.rstrip(b"\x00").decode("ascii", errors="ignore")
        if name == ".text":
            text_section = section
            break

    if text_section is None:
        # Some binaries (e.g. Delphi, Go) use non-standard section names.
        # Fall back to the first section with the EXECUTE characteristic.
        for section in pe.sections:
            if section.Characteristics & 0x20000000:  # IMAGE_SCN_MEM_EXECUTE
                text_section = section
                break

    if text_section is None:
        print(f"  [SKIP] {filepath}: no executable section found")
        return []

    code_bytes = text_section.get_data()
    va = text_section.VirtualAddress

    # Disassemble ---------------------------------------------------------------
    md = capstone.Cs(capstone.CS_ARCH_X86, mode)
    md.detail = False                   # we only need mnemonics — faster
    opcodes = [insn.mnemonic for insn in md.disasm(code_bytes, va)]
    return opcodes


# ---------------------------------------------------------------------------
# Phase B: Tokenisation & Serialisation
# ---------------------------------------------------------------------------
def build_vocabulary(all_opcode_lists: list[list[str]]) -> dict[str, int]:
    """
    Create a ``stoi`` (string‑to‑int) vocabulary from all observed opcodes.

    Layout: <PAD>=0  <UNK>=1  then sorted unique mnemonics …
    """
    counter: Counter[str] = Counter()
    for ops in all_opcode_lists:
        counter.update(ops)

    sorted_ops = sorted(counter.keys())
    stoi: dict[str, int] = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
    for op in sorted_ops:
        if op not in stoi:
            stoi[op] = len(stoi)
    return stoi


def encode_sequence(opcodes: list[str], stoi: dict[str, int],
                    max_len: int) -> np.ndarray:
    """
    Map an opcode list to integer IDs, truncate / pad to ``max_len``.
    """
    unk_id = stoi[UNK_TOKEN]
    pad_id = stoi[PAD_TOKEN]
    ids = [stoi.get(op, unk_id) for op in opcodes[:max_len]]
    # Pad
    ids += [pad_id] * (max_len - len(ids))
    return np.array(ids, dtype=np.int32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def collect_files(data_dir: Path) -> tuple[list[str], list[int]]:
    """
    Walk ``data_dir/raw/safe`` and ``data_dir/raw/malicious``.
    Returns (filepaths, labels).  Safe=0, Malicious=1.
    """
    filepaths: list[str] = []
    labels: list[int] = []
    pe_extensions = {".exe", ".dll", ".sys", ".ocx", ".scr"}

    for label, subdir in [(0, "safe"), (1, "malicious")]:
        folder = data_dir / "raw" / subdir
        if not folder.exists():
            print(f"  [WARN] {folder} does not exist — skipping.")
            continue
        for fpath in sorted(folder.rglob("*")):
            if fpath.suffix.lower() in pe_extensions:
                filepaths.append(str(fpath))
                labels.append(label)

    return filepaths, labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wintermute — PE → opcode dataset builder")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Root data directory (contains raw/ and processed/).")
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH,
                        help="Max opcode sequence length per sample.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover files --------------------------------------------------------
    filepaths, labels = collect_files(data_dir)
    if not filepaths:
        print("[ERROR] No PE files found in data/raw/safe or data/raw/malicious.")
        print("        Place .exe / .dll files there first, or use")
        print("        generate_synthetic_data.py for testing.")
        sys.exit(1)

    print(f"Found {len(filepaths)} PE files "
          f"({labels.count(0)} safe, {labels.count(1)} malicious).")

    # 2. Extract opcodes -------------------------------------------------------
    all_opcodes: list[list[str]] = []
    for i, fp in enumerate(filepaths, 1):
        print(f"  [{i}/{len(filepaths)}] Extracting {os.path.basename(fp)} …")
        ops = extract_opcodes(fp)
        all_opcodes.append(ops)
        if ops:
            print(f"          → {len(ops)} instructions")
        # NOTE: empty sequences are kept (they'll become all-PAD rows).

    # 3. Build vocabulary -------------------------------------------------------
    stoi = build_vocabulary(all_opcodes)
    print(f"Vocabulary size: {len(stoi)} tokens")

    # 4. Encode & serialise ----------------------------------------------------
    x_data = np.stack(
        [encode_sequence(ops, stoi, args.max_seq_length) for ops in all_opcodes]
    )
    y_data = np.array(labels, dtype=np.int32)

    np.save(out_dir / "x_data.npy", x_data)
    np.save(out_dir / "y_data.npy", y_data)
    with open(out_dir / "vocab.json", "w") as f:
        json.dump(stoi, f, indent=2)

    print(f"\n✅  Dataset saved to {out_dir}/")
    print(f"    x_data.npy  shape={x_data.shape}  dtype={x_data.dtype}")
    print(f"    y_data.npy  shape={y_data.shape}  dtype={y_data.dtype}")
    print(f"    vocab.json   {len(stoi)} entries")


if __name__ == "__main__":
    main()
