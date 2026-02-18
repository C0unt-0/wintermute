#!/usr/bin/env python3
"""
06_build_extended_dataset.py — Unified Multi-Class Dataset Builder

Reads opcode .asm files from multiple sources (MalwareBazaar downloads,
Microsoft dataset, etc.) and produces a unified training dataset.

Each subdirectory under --data-dir is treated as a separate class (family).
Optionally merges Microsoft Malware Classification samples via --ms-dir.

Outputs:
    data/processed/x_data.npy     — tokenised opcode sequences
    data/processed/y_data.npy     — integer class labels (0-based)
    data/processed/vocab.json     — opcode-to-integer mapping
    data/processed/families.json  — class index-to-family name mapping

Usage:
    python src/06_build_extended_dataset.py --data-dir data/bazaar
    python src/06_build_extended_dataset.py --data-dir data/bazaar --ms-dir data/ms-malware
    python src/06_build_extended_dataset.py --data-dir data/bazaar --balance --max-seq-length 2048
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

# Regex for IDA-style .asm files (from 04_build_ms_dataset.py)
IDA_INSN_RE = re.compile(
    r"^\s*\.\w+:[0-9A-Fa-f]+\s+"   # section:address
    r"(?:[0-9A-Fa-f]{2}\s+)+"      # hex bytes
    r"(\w+)",                        # mnemonic (capture group 1)
)

# IDA directives to skip (not real instructions)
SKIP_MNEMONICS = {
    "db", "dw", "dd", "dq", "dt", "align", "assume",
    "org", "end", "byte", "word", "dword", "qword",
    "proc", "endp", "public", "extrn", "include",
    "segment", "ends", "unicode", "dup",
}


# ---------------------------------------------------------------------------
# .asm file readers
# ---------------------------------------------------------------------------
def read_bazaar_asm(filepath: str) -> list[str]:
    """
    Read a MalwareBazaar-style .asm file (one opcode per line).

    These are produced by 05_download_malwarebazaar.py.
    """
    opcodes = []
    with open(filepath, "r", errors="ignore") as f:
        for line in f:
            op = line.strip()
            if op and not op.startswith("#"):
                opcodes.append(op)
    return opcodes


def read_ida_asm(filepath: str) -> list[str]:
    """
    Read an IDA Pro-style .asm file and extract opcodes.

    These come from the Microsoft Malware Classification dataset.
    """
    opcodes = []
    with open(filepath, "r", errors="ignore") as f:
        for line in f:
            m = IDA_INSN_RE.match(line)
            if m:
                mnemonic = m.group(1).lower()
                if mnemonic not in SKIP_MNEMONICS:
                    opcodes.append(mnemonic)
    return opcodes


def detect_asm_format(filepath: str) -> str:
    """
    Detect whether an .asm file is Bazaar-style (simple) or IDA-style.

    Returns 'bazaar' or 'ida'.
    """
    with open(filepath, "r", errors="ignore") as f:
        for i, line in enumerate(f):
            if i > 20:
                break
            # IDA files have lines like ".text:10001106 D9 C0  fld  st"
            if re.match(r"^\s*\.\w+:[0-9A-Fa-f]+", line):
                return "ida"
    return "bazaar"


def read_asm_file(filepath: str) -> list[str]:
    """Auto-detect format and read opcodes from an .asm file."""
    fmt = detect_asm_format(filepath)
    if fmt == "ida":
        return read_ida_asm(filepath)
    return read_bazaar_asm(filepath)


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
# Data collectors
# ---------------------------------------------------------------------------
def collect_bazaar_samples(data_dir: Path) -> tuple[list[list[str]], list[int], dict[int, str]]:
    """
    Scan data_dir for family subdirectories containing .asm files.

    Returns (all_opcodes, labels, family_map).
    """
    all_opcodes: list[list[str]] = []
    labels: list[int] = []
    family_map: dict[int, str] = {}

    # Each subdirectory = one family
    subdirs = sorted([d for d in data_dir.iterdir()
                      if d.is_dir() and not d.name.startswith(".")])

    if not subdirs:
        return all_opcodes, labels, family_map

    for class_idx, subdir in enumerate(subdirs):
        family_name = subdir.name
        family_map[class_idx] = family_name

        asm_files = sorted(subdir.glob("*.asm"))
        print(f"  [{class_idx}] {family_name}: {len(asm_files)} samples")

        for fp in asm_files:
            opcodes = read_asm_file(str(fp))
            if len(opcodes) >= 10:  # skip near-empty files
                all_opcodes.append(opcodes)
                labels.append(class_idx)

    return all_opcodes, labels, family_map


def collect_ms_samples(
    ms_dir: Path,
    class_offset: int,
    labels_path: Path | None = None,
) -> tuple[list[list[str]], list[int], dict[int, str]]:
    """
    Collect Microsoft Malware Classification dataset samples.

    Returns (all_opcodes, labels, family_map) with class indices offset
    by class_offset to avoid collisions with Bazaar families.
    """
    MS_FAMILIES = {
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

    all_opcodes: list[list[str]] = []
    labels: list[int] = []
    family_map: dict[int, str] = {}

    # Load labels
    sample_labels: dict[str, int] = {}
    lpath = labels_path or ms_dir / "labels.csv"
    if lpath.exists():
        with open(lpath) as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample_id = row["Id"]
                cls = int(row["Class"]) - 1  # 1-based → 0-based
                sample_labels[sample_id] = cls
    else:
        print(f"  [WARN] No labels.csv found at {lpath}")
        return all_opcodes, labels, family_map

    # Build family map with offset
    for ms_cls, name in MS_FAMILIES.items():
        new_idx = (ms_cls - 1) + class_offset
        family_map[new_idx] = name

    # Read .asm files
    asm_files = sorted(ms_dir.glob("*.asm"))
    print(f"  MS dataset: {len(asm_files)} .asm files found")

    for fp in asm_files:
        sample_id = fp.stem
        if sample_id not in sample_labels:
            continue

        ms_cls = sample_labels[sample_id]
        new_label = ms_cls + class_offset

        opcodes = read_ida_asm(str(fp))
        if len(opcodes) >= 10:
            all_opcodes.append(opcodes)
            labels.append(new_label)

    return all_opcodes, labels, family_map


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wintermute — unified multi-class dataset builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/06_build_extended_dataset.py --data-dir data/bazaar
  python src/06_build_extended_dataset.py --data-dir data/bazaar --ms-dir data/ms-malware
  python src/06_build_extended_dataset.py --data-dir data/bazaar --balance
        """,
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Directory with family subdirectories containing .asm files.",
    )
    parser.add_argument(
        "--ms-dir", type=str, default=None,
        help="Optional: path to Microsoft Malware Classification .asm files.",
    )
    parser.add_argument(
        "--ms-labels", type=str, default=None,
        help="Path to labels.csv for the MS dataset (default: ms-dir/labels.csv).",
    )
    parser.add_argument(
        "--out-dir", type=str, default="data/processed",
        help="Output directory for .npy and .json files.",
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=MAX_SEQ_LENGTH,
        help="Maximum opcode sequence length per sample.",
    )
    parser.add_argument(
        "--balance", action="store_true",
        help="Balance classes by undersampling to the smallest class.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffling / balancing.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("═" * 60)
    print("  Wintermute — Extended Dataset Builder")
    print("═" * 60)

    # 1. Collect MalwareBazaar samples -----------------------------------------
    print(f"\nScanning {data_dir}/ for family directories …")
    all_opcodes, labels, family_map = collect_bazaar_samples(data_dir)
    print(f"  → {len(all_opcodes)} samples from {len(family_map)} families")

    # 2. Optionally merge Microsoft dataset ------------------------------------
    if args.ms_dir:
        ms_dir = Path(args.ms_dir)
        print(f"\nMerging Microsoft dataset from {ms_dir}/ …")
        class_offset = max(family_map.keys()) + 1 if family_map else 0
        ms_opcodes, ms_labels, ms_family_map = collect_ms_samples(
            ms_dir, class_offset,
            Path(args.ms_labels) if args.ms_labels else None,
        )
        all_opcodes.extend(ms_opcodes)
        labels.extend(ms_labels)
        family_map.update(ms_family_map)
        print(f"  → {len(ms_opcodes)} MS samples added "
              f"({len(ms_family_map)} families)")

    if not all_opcodes:
        print("\n[ERROR] No samples found. Check your data directories.")
        sys.exit(1)

    # 3. Remap labels to contiguous 0..N-1 -------------------------------------
    unique_labels = sorted(set(labels))
    label_remap = {old: new for new, old in enumerate(unique_labels)}
    labels = [label_remap[l] for l in labels]

    # Rebuild family map with contiguous keys
    new_family_map = {}
    for old_idx, new_idx in label_remap.items():
        if old_idx in family_map:
            new_family_map[new_idx] = family_map[old_idx]
        else:
            new_family_map[new_idx] = f"Class_{old_idx}"

    family_map = new_family_map
    num_classes = len(family_map)

    # 4. Print class distribution -----------------------------------------------
    class_counts = Counter(labels)
    print(f"\nClass distribution ({num_classes} classes):")
    for cls_idx in sorted(class_counts.keys()):
        name = family_map.get(cls_idx, f"Class_{cls_idx}")
        count = class_counts[cls_idx]
        bar = "█" * min(count // 5, 40)
        print(f"  [{cls_idx:2d}] {name:<20} {count:>5} {bar}")

    # 5. Balance (optional) -----------------------------------------------------
    rng = np.random.default_rng(args.seed)

    if args.balance and len(class_counts) > 1:
        min_count = min(class_counts.values())
        print(f"\nBalancing: undersampling all classes to {min_count} samples")

        indices_by_class: dict[int, list[int]] = {c: [] for c in class_counts}
        for i, label in enumerate(labels):
            indices_by_class[label].append(i)

        balanced_indices = []
        for cls_idx, idxs in indices_by_class.items():
            chosen = rng.choice(idxs, size=min_count, replace=False)
            balanced_indices.extend(chosen)

        rng.shuffle(balanced_indices)
        all_opcodes = [all_opcodes[i] for i in balanced_indices]
        labels = [labels[i] for i in balanced_indices]
        print(f"  → {len(labels)} total samples after balancing")

    # 6. Build vocabulary -------------------------------------------------------
    stoi = build_vocabulary(all_opcodes)
    print(f"\nVocabulary size: {len(stoi)} tokens")

    # 7. Encode & serialise -----------------------------------------------------
    print("Encoding sequences …")
    x_data = np.stack(
        [encode_sequence(ops, stoi, args.max_seq_length) for ops in all_opcodes]
    )
    y_data = np.array(labels, dtype=np.int32)

    # Shuffle
    order = rng.permutation(len(labels))
    x_data = x_data[order]
    y_data = y_data[order]

    # Save
    np.save(out_dir / "x_data.npy", x_data)
    np.save(out_dir / "y_data.npy", y_data)

    with open(out_dir / "vocab.json", "w") as f:
        json.dump(stoi, f, indent=2)

    families_json = {str(k): v for k, v in sorted(family_map.items())}
    with open(out_dir / "families.json", "w") as f:
        json.dump(families_json, f, indent=2)

    # Final report
    print(f"\n{'═' * 60}")
    print(f"  ✅  Dataset saved to {out_dir}/")
    print(f"{'═' * 60}")
    print(f"  x_data.npy     shape={x_data.shape}  dtype={x_data.dtype}")
    print(f"  y_data.npy     shape={y_data.shape}  dtype={y_data.dtype}")
    print(f"  vocab.json     {len(stoi)} entries")
    print(f"  families.json  {num_classes} classes")
    print(f"""
{'─' * 60}
  Next step — train the model:
  python src/03_train.py \\
      --data-dir {out_dir} \\
      --num-classes {num_classes} \\
      --epochs 20
{'─' * 60}
""")


if __name__ == "__main__":
    main()
