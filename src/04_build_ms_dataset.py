#!/usr/bin/env python3
"""
04_build_ms_dataset.py — Parse Microsoft Malware Classification .asm files

Extracts opcode sequences from IDA Pro disassembly (.asm) files and
produces NumPy arrays for training the MalwareClassifier in 9-class mode.

IDA .asm line format:
    .text:10001106 D9 C0        fld     st
    .text:1000110F 7A 0A        jp      short loc_1000111B

We extract the mnemonic (e.g. "fld", "jp") from lines that belong
to executable sections (.text, .code, CODE, etc.).

Usage:
    python src/04_build_ms_dataset.py
    python src/04_build_ms_dataset.py --samples-dir data/ms-malware --labels data/ms-malware/labels.csv
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_SEQ_LENGTH = 2048
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN]

# Malware family names indexed by class (1-based in the CSV)
FAMILY_NAMES = {
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

# ---------------------------------------------------------------------------
# .asm parser
# ---------------------------------------------------------------------------
# Regex to match IDA disassembly lines with instructions.
# Captures lines like:
#   .text:10001106 D9 C0        fld     st
#   .code:00401000 55           push    ebp
#   CODE:00401000 55            push    ebp
#
# Pattern breakdown:
#   ^(\.text|\.code|CODE)   — section name
#   :\w+                    — address
#   \s+                     — whitespace
#   (?:[0-9A-Fa-f]{2}\s+)+  — hex bytes (at least one pair)
#   (\w+)                   — mnemonic (our target)
ASM_LINE_RE = re.compile(
    r"^(?:\.text|\.code|CODE|\.itext)\s*:\s*\w+\s+"   # section:address
    r"(?:[0-9A-Fa-f]{2}\s+)+"                          # hex bytes
    r"(\w+)",                                           # mnemonic
    re.IGNORECASE
)

# Common IDA directives / pseudo-instructions to skip
SKIP_MNEMONICS = {
    "db", "dw", "dd", "dq", "dt", "align", "assume",
    "org", "end", "byte", "word", "dword", "qword",
    "proc", "endp", "public", "extrn", "include",
    "segment", "ends", "unicode", "dup",
}


def extract_opcodes_from_asm(filepath: str) -> list[str]:
    """
    Parse an IDA Pro .asm file and return an ordered list of opcodes.

    Only mnemonics from executable sections are extracted.
    Directives and data definitions are filtered out.
    """
    opcodes: list[str] = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = ASM_LINE_RE.match(line.strip())
                if m:
                    mnemonic = m.group(1).lower()
                    if mnemonic not in SKIP_MNEMONICS:
                        opcodes.append(mnemonic)
    except OSError as e:
        print(f"  [SKIP] {filepath}: {e}")
    return opcodes


# ---------------------------------------------------------------------------
# Labels loader
# ---------------------------------------------------------------------------
def load_labels(labels_path: str) -> dict[str, int]:
    """
    Load labels.csv → {sample_id: class_label (0-indexed)}.

    The CSV has columns: Id, Class
    Class is 1-based in the file; we shift to 0-based for training.
    """
    labels: dict[str, int] = {}
    with open(labels_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row["Id"].strip()
            cls = int(row["Class"].strip()) - 1   # 0-indexed
            labels[sample_id] = cls
    return labels


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------
def build_vocabulary(all_opcode_lists: list[list[str]]) -> dict[str, int]:
    """Build stoi vocabulary from observed opcodes."""
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
    """Map opcodes → integer IDs, truncate / pad to max_len."""
    unk_id = stoi[UNK_TOKEN]
    pad_id = stoi[PAD_TOKEN]
    ids = [stoi.get(op, unk_id) for op in opcodes[:max_len]]
    ids += [pad_id] * (max_len - len(ids))
    return np.array(ids, dtype=np.int32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wintermute — Microsoft Malware Classification dataset builder")
    parser.add_argument("--samples-dir", type=str, default="data/ms-malware",
                        help="Directory containing .asm files.")
    parser.add_argument("--labels", type=str, default="data/ms-malware/labels.csv",
                        help="Path to labels.csv.")
    parser.add_argument("--out-dir", type=str, default="data/processed",
                        help="Output directory for .npy and vocab.json.")
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH,
                        help="Max opcode sequence length per sample.")
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load labels -----------------------------------------------------------
    print("Loading labels …")
    labels_map = load_labels(args.labels)
    print(f"  {len(labels_map)} labelled samples found")

    # 2. Discover .asm files ---------------------------------------------------
    asm_files = sorted(samples_dir.glob("*.asm"))
    if not asm_files:
        print(f"[ERROR] No .asm files found in {samples_dir}")
        print("        Run download_ms_dataset.py first, or place .asm files there.")
        sys.exit(1)

    # Match files to labels
    matched_files: list[tuple[Path, int]] = []
    for asm in asm_files:
        sample_id = asm.stem
        if sample_id in labels_map:
            matched_files.append((asm, labels_map[sample_id]))
        else:
            print(f"  [SKIP] {asm.name}: no label found in labels.csv")

    if not matched_files:
        print("[ERROR] No .asm files matched labels.csv entries.")
        sys.exit(1)

    # Print class distribution
    class_counts = Counter(cls for _, cls in matched_files)
    print(f"\n  Matched {len(matched_files)} samples across {len(class_counts)} classes:")
    for cls in sorted(class_counts):
        name = FAMILY_NAMES.get(cls + 1, f"Class {cls}")
        print(f"    Class {cls} ({name}): {class_counts[cls]}")

    # 3. Extract opcodes -------------------------------------------------------
    print(f"\nExtracting opcodes from {len(matched_files)} files …")
    all_opcodes: list[list[str]] = []
    all_labels: list[int] = []

    for i, (asm_path, label) in enumerate(matched_files, 1):
        ops = extract_opcodes_from_asm(str(asm_path))
        all_opcodes.append(ops)
        all_labels.append(label)
        if i % 500 == 0 or i == len(matched_files):
            print(f"  [{i}/{len(matched_files)}] {asm_path.name} → {len(ops)} opcodes")

    # 4. Build vocabulary -------------------------------------------------------
    stoi = build_vocabulary(all_opcodes)
    print(f"\nVocabulary size: {len(stoi)} tokens")

    # 5. Encode & serialise ----------------------------------------------------
    x_data = np.stack(
        [encode_sequence(ops, stoi, args.max_seq_length) for ops in all_opcodes]
    )
    y_data = np.array(all_labels, dtype=np.int32)

    np.save(out_dir / "x_data.npy", x_data)
    np.save(out_dir / "y_data.npy", y_data)

    # Save vocab + metadata
    with open(out_dir / "vocab.json", "w") as f:
        json.dump(stoi, f, indent=2)

    # Save family mapping for inference
    family_map = {str(k - 1): v for k, v in FAMILY_NAMES.items()}
    with open(out_dir / "families.json", "w") as f:
        json.dump(family_map, f, indent=2)

    print(f"\n✅  Dataset saved to {out_dir}/")
    print(f"    x_data.npy    shape={x_data.shape}  dtype={x_data.dtype}")
    print(f"    y_data.npy    shape={y_data.shape}  dtype={y_data.dtype}")
    print(f"    vocab.json    {len(stoi)} entries")
    print(f"    families.json {len(FAMILY_NAMES)} families")
    print(f"\n  Next: python src/03_train.py --num-classes 9")


if __name__ == "__main__":
    main()
