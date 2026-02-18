#!/usr/bin/env python3
"""
scan.py — Wintermute Inference CLI

Scan a Windows PE file and classify it as Safe or Malicious using
the trained MLX Transformer model.

Usage:
    python scan.py target_file.exe
    python scan.py target_file.dll --model malware_model.safetensors
    python scan.py target_file.exe --vocab data/processed/vocab.json
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Dynamic import for 01_build_dataset.py (filename starts with a digit)
_spec_ds = importlib.util.spec_from_file_location(
    "build_dataset", str(Path(__file__).resolve().parent / "src" / "01_build_dataset.py")
)
_mod_ds = importlib.util.module_from_spec(_spec_ds)
_spec_ds.loader.exec_module(_mod_ds)
extract_opcodes = _mod_ds.extract_opcodes

# Dynamic import for 02_model.py
_spec_model = importlib.util.spec_from_file_location(
    "model", str(Path(__file__).resolve().parent / "src" / "02_model.py")
)
_mod_model = importlib.util.module_from_spec(_spec_model)
_spec_model.loader.exec_module(_mod_model)
MalwareClassifier = _mod_model.MalwareClassifier

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_SEQ_LENGTH = 2048
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def tokenize(opcodes: list[str], stoi: dict[str, int],
             max_len: int) -> mx.array:
    """Convert opcode list → padded/truncated MLX integer tensor."""
    unk_id = stoi[UNK_TOKEN]
    pad_id = stoi[PAD_TOKEN]
    ids = [stoi.get(op, unk_id) for op in opcodes[:max_len]]
    ids += [pad_id] * (max_len - len(ids))
    return mx.array([ids])    # [1, T]


def scan_file(filepath: str, model: MalwareClassifier,
              stoi: dict[str, int], max_seq_length: int) -> None:
    """Disassemble, tokenise, classify, and print the verdict."""

    print(f"\n{'═' * 60}")
    print(f"  Scanning: {filepath}")
    print(f"{'═' * 60}")

    # 1. Extract opcodes -------------------------------------------------------
    opcodes = extract_opcodes(filepath)
    if not opcodes:
        print("  ⚠️  Could not extract opcodes from this file.")
        print("      It may not be a valid PE or has no executable section.")
        return

    print(f"  Disassembled {len(opcodes)} instructions")

    # 2. Tokenise ---------------------------------------------------------------
    x = tokenize(opcodes, stoi, max_seq_length)

    # 3. Inference ---------------------------------------------------------------
    logits = model(x)                     # [1, 2]
    probs = mx.softmax(logits, axis=1)    # [1, 2]
    mx.eval(probs)

    safe_prob = probs[0, 0].item() * 100
    mal_prob = probs[0, 1].item() * 100
    prediction = int(mx.argmax(probs, axis=1).item())

    # 4. Verdict ----------------------------------------------------------------
    print()
    if prediction == 0:
        print(f"  ✅  [SAFE]       Probability: {safe_prob:.1f}%")
    else:
        print(f"  🚨  [MALICIOUS]  Probability: {mal_prob:.1f}%")
    print()
    print(f"  Details:")
    print(f"    Safe probability:      {safe_prob:6.2f}%")
    print(f"    Malicious probability: {mal_prob:6.2f}%")
    print(f"{'═' * 60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wintermute — PE malware scanner (MLX inference)")
    parser.add_argument("target", type=str,
                        help="Path to the PE file to scan (.exe, .dll, etc.).")
    parser.add_argument("--model", type=str, default="malware_model.safetensors",
                        help="Path to trained model weights (.safetensors).")
    parser.add_argument("--vocab", type=str, default="data/processed/vocab.json",
                        help="Path to vocab.json produced by the data pipeline.")
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH,
                        help="Sequence length (must match training).")
    args = parser.parse_args()

    # Validate inputs -----------------------------------------------------------
    target = Path(args.target)
    if not target.exists():
        print(f"[ERROR] File not found: {args.target}")
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] Model weights not found: {args.model}")
        print("        Train a model first with: python src/03_train.py")
        sys.exit(1)

    vocab_path = Path(args.vocab)
    if not vocab_path.exists():
        print(f"[ERROR] Vocabulary not found: {args.vocab}")
        print("        Build the dataset first with: python src/01_build_dataset.py")
        sys.exit(1)

    # Load vocabulary -----------------------------------------------------------
    with open(vocab_path) as f:
        stoi = json.load(f)
    vocab_size = len(stoi)

    # Load model ----------------------------------------------------------------
    model = MalwareClassifier(
        vocab_size=vocab_size,
        max_seq_length=args.max_seq_length,
    )
    model.load_weights(str(model_path))
    MalwareClassifier.cast_to_bf16(model)
    model.eval()                  # disable dropout etc. (no-op here but good practice)

    # Scan! ---------------------------------------------------------------------
    scan_file(str(target), model, stoi, args.max_seq_length)


if __name__ == "__main__":
    main()
