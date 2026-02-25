#!/usr/bin/env python3
"""
scan_family.py — Wintermute Multi-Class Inference CLI

Scan a Windows PE file (or IDA .asm file) and classify it into one of
the 9 Microsoft Malware Classification families.

Usage:
    python scan_family.py target_file.asm
    python scan_family.py target_file.exe
    python scan_family.py target.asm --model malware_model.safetensors
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Dynamic imports for modules with numeric-prefix filenames
_src_dir = Path(__file__).resolve().parent / "src"

_spec_ds = importlib.util.spec_from_file_location(
    "build_dataset", str(_src_dir / "01_build_dataset.py")
)
_mod_ds = importlib.util.module_from_spec(_spec_ds)
_spec_ds.loader.exec_module(_mod_ds)
extract_opcodes_pe = _mod_ds.extract_opcodes

_spec_ms = importlib.util.spec_from_file_location(
    "build_ms_dataset", str(_src_dir / "04_build_ms_dataset.py")
)
_mod_ms = importlib.util.module_from_spec(_spec_ms)
_spec_ms.loader.exec_module(_mod_ms)
extract_opcodes_asm = _mod_ms.extract_opcodes_from_asm

_spec_model = importlib.util.spec_from_file_location(
    "model", str(_src_dir / "02_model.py")
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

DEFAULT_FAMILIES = {
    "0": "Ramnit",
    "1": "Lollipop",
    "2": "Kelihos_ver3",
    "3": "Vundo",
    "4": "Simda",
    "5": "Tracur",
    "6": "Kelihos_ver1",
    "7": "Obfuscator.ACY",
    "8": "Gatak",
}


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
              stoi: dict[str, int], families: dict[str, str],
              max_seq_length: int) -> None:
    """Disassemble / parse, tokenise, classify, and print the verdict."""

    print(f"\n{'═' * 60}")
    print(f"  Scanning: {filepath}")
    print(f"{'═' * 60}")

    # 1. Extract opcodes -------------------------------------------------------
    fpath = Path(filepath)
    if fpath.suffix.lower() == ".asm":
        opcodes = extract_opcodes_asm(filepath)
    else:
        opcodes = extract_opcodes_pe(filepath)

    if not opcodes:
        print("  ⚠️  Could not extract opcodes from this file.")
        return

    print(f"  Disassembled {len(opcodes)} instructions")

    # 2. Tokenise ---------------------------------------------------------------
    x = tokenize(opcodes, stoi, max_seq_length)

    # 3. Inference ---------------------------------------------------------------
    logits = model(x)                     # [1, C]
    probs = mx.softmax(logits, axis=1)    # [1, C]
    mx.eval(probs)

    num_classes = probs.shape[1]
    prediction = int(mx.argmax(probs, axis=1).item())
    top_prob = probs[0, prediction].item() * 100

    # 4. Verdict ----------------------------------------------------------------
    family_name = families.get(str(prediction), f"Class {prediction}")

    print(f"\n  🎯  Predicted Family: {family_name}")
    print(f"      Confidence:      {top_prob:.1f}%\n")

    # Show full probability distribution
    print(f"  {'Family':<20} {'Probability':>12}")
    print(f"  {'─' * 34}")

    # Sort by probability descending
    prob_list = [(i, probs[0, i].item()) for i in range(num_classes)]
    prob_list.sort(key=lambda x: x[1], reverse=True)

    for cls_idx, prob in prob_list:
        name = families.get(str(cls_idx), f"Class {cls_idx}")
        bar = "█" * int(prob * 30)
        marker = " ◄" if cls_idx == prediction else ""
        print(f"  {name:<20} {prob * 100:>6.2f}%  {bar}{marker}")

    print(f"\n{'═' * 60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wintermute — malware family classifier (multi-class)")
    parser.add_argument("target", type=str,
                        help="Path to the file to scan (.asm or .exe/.dll).")
    parser.add_argument("--model", type=str, default="malware_model.safetensors",
                        help="Path to trained model weights (.safetensors).")
    parser.add_argument("--vocab", type=str, default="data/processed/vocab.json",
                        help="Path to vocab.json.")
    parser.add_argument("--families", type=str, default="data/processed/families.json",
                        help="Path to families.json (class index → name mapping).")
    parser.add_argument("--num-classes", type=int, default=None,
                        help="Number of output classes (auto-detected from "
                             "families.json if not specified).")
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
        print("        Train first: python src/03_train.py --num-classes N")
        sys.exit(1)

    vocab_path = Path(args.vocab)
    if not vocab_path.exists():
        print(f"[ERROR] Vocabulary not found: {args.vocab}")
        sys.exit(1)

    # Load families mapping -----------------------------------------------------
    families_path = Path(args.families)
    if families_path.exists():
        with open(families_path) as f:
            families = json.load(f)
    else:
        families = DEFAULT_FAMILIES

    # Auto-detect num_classes from families.json --------------------------------
    num_classes = args.num_classes or len(families)

    # Load vocabulary -----------------------------------------------------------
    with open(vocab_path) as f:
        stoi = json.load(f)
    vocab_size = len(stoi)

    # Load model ----------------------------------------------------------------
    model = MalwareClassifier(
        vocab_size=vocab_size,
        max_seq_length=args.max_seq_length,
        num_classes=num_classes,
    )
    model.load_weights(str(model_path))
    MalwareClassifier.cast_to_bf16(model)
    model.eval()

    print(f"  Model loaded: {num_classes} classes, vocab size {vocab_size}")

    # Scan! ---------------------------------------------------------------------
    scan_file(str(target), model, stoi, families, args.max_seq_length)


if __name__ == "__main__":
    main()
