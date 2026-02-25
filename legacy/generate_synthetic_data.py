#!/usr/bin/env python3
"""
generate_synthetic_data.py — Wintermute Synthetic Data Generator

Produces fake opcode-sequence datasets so you can test the model and
training loop without handling real PE binaries.

Safe samples are biased toward common "benign" opcodes (mov, push, call …).
Malicious samples are biased toward suspicious patterns (xor, shr, loop …).

Usage:
    python src/generate_synthetic_data.py                # 500 samples
    python src/generate_synthetic_data.py --n-samples 2000
"""

import argparse
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Fake opcode pools — intentionally distinct distributions
# ---------------------------------------------------------------------------
BENIGN_OPCODES = [
    "mov", "push", "pop", "call", "ret", "lea", "add", "sub",
    "cmp", "jmp", "je", "jne", "test", "nop", "xchg", "inc",
    "dec", "and", "or", "shl", "imul", "cdq", "movzx", "movsx",
]

MALICIOUS_OPCODES = [
    "xor", "shr", "shl", "ror", "rol", "loop", "stosb", "lodsb",
    "scasb", "repne", "rep", "int", "rdtsc", "cpuid", "vmcall",
    "syscall", "in", "out", "cli", "sti", "pushfd", "popfd",
    "bswap", "not",
]

# Shared opcodes that appear in both
SHARED_OPCODES = [
    "mov", "push", "pop", "call", "ret", "add", "sub", "cmp",
    "jmp", "je", "jne", "test", "nop", "xor", "and", "or",
]

MAX_SEQ_LENGTH = 2048
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def generate_sample(label: int, rng: np.random.Generator,
                    max_len: int) -> list[str]:
    """
    Generate a fake opcode sequence with label‑dependent bias.
    """
    # Variable-length sequences (50–max_len instructions)
    seq_len = rng.integers(50, max_len + 1)

    if label == 0:  # safe
        pool = BENIGN_OPCODES + SHARED_OPCODES
    else:           # malicious
        pool = MALICIOUS_OPCODES + SHARED_OPCODES

    return [pool[i] for i in rng.integers(0, len(pool), size=seq_len)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wintermute — synthetic opcode-dataset generator")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Total number of samples (split 50/50).")
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH,
                        help="Maximum sequence length per sample.")
    parser.add_argument("--out-dir", type=str, default="data/processed",
                        help="Output directory.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_safe = args.n_samples // 2
    n_mal = args.n_samples - n_safe

    # 1. Generate raw sequences -------------------------------------------------
    all_opcodes: list[list[str]] = []
    labels: list[int] = []

    for _ in range(n_safe):
        all_opcodes.append(generate_sample(0, rng, args.max_seq_length))
        labels.append(0)
    for _ in range(n_mal):
        all_opcodes.append(generate_sample(1, rng, args.max_seq_length))
        labels.append(1)

    # Shuffle
    order = rng.permutation(len(labels))
    all_opcodes = [all_opcodes[i] for i in order]
    labels = [labels[i] for i in order]

    # 2. Build vocabulary -------------------------------------------------------
    unique_ops = sorted({op for seq in all_opcodes for op in seq})
    stoi = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for op in unique_ops:
        stoi[op] = len(stoi)

    # 3. Encode & serialise -----------------------------------------------------
    pad_id = stoi[PAD_TOKEN]
    x_rows = []
    for seq in all_opcodes:
        ids = [stoi[op] for op in seq[:args.max_seq_length]]
        ids += [pad_id] * (args.max_seq_length - len(ids))
        x_rows.append(ids)

    x_data = np.array(x_rows, dtype=np.int32)
    y_data = np.array(labels, dtype=np.int32)

    np.save(out_dir / "x_data.npy", x_data)
    np.save(out_dir / "y_data.npy", y_data)
    with open(out_dir / "vocab.json", "w") as f:
        json.dump(stoi, f, indent=2)

    print(f"✅  Synthetic dataset saved to {out_dir}/")
    print(f"    x_data.npy  shape={x_data.shape}  dtype={x_data.dtype}")
    print(f"    y_data.npy  shape={y_data.shape}  dtype={y_data.dtype}")
    print(f"    vocab.json   {len(stoi)} entries")
    print(f"    {n_safe} safe + {n_mal} malicious = {args.n_samples} total")


if __name__ == "__main__":
    main()
