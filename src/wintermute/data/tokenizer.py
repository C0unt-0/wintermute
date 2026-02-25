"""
tokenizer.py — Wintermute Opcode Extraction & Tokenization

Consolidates all extraction and tokenization logic previously scattered
across 01_build_dataset.py, 04_build_ms_dataset.py, and 06_build_extended_dataset.py.

Responsibilities:
    - Disassemble PE files → opcode lists  (Capstone + pefile)
    - Parse IDA Pro .asm files → opcode lists
    - Parse MalwareBazaar .asm files → opcode lists
    - Build vocabulary (stoi mapping)
    - Encode opcode sequences → integer arrays
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import capstone
import numpy as np
import pefile
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Default config (overridable via load_data_config)
# ---------------------------------------------------------------------------
_DEFAULT_CFG = OmegaConf.create({
    "max_seq_length": 2048,
    "pad_token": "<PAD>",
    "unk_token": "<UNK>",
})

_CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"


def load_data_config(config_path: str | Path | None = None) -> OmegaConf:
    """Load data config, falling back to defaults."""
    cfg = OmegaConf.create(_DEFAULT_CFG)
    path = Path(config_path) if config_path else _CONFIGS_DIR / "data_config.yaml"
    if path.exists():
        file_cfg = OmegaConf.load(path)
        cfg = OmegaConf.merge(cfg, file_cfg)
    return cfg


# ---------------------------------------------------------------------------
# IDA .asm regex and skip-set (from 04_build_ms_dataset.py)
# ---------------------------------------------------------------------------
IDA_INSN_RE = re.compile(
    r"^\s*\.\w+:[0-9A-Fa-f]+\s+"   # section:address
    r"(?:[0-9A-Fa-f]{2}\s+)+"      # hex bytes
    r"(\w+)",                        # mnemonic (capture group 1)
)

SKIP_MNEMONICS = {
    "db", "dw", "dd", "dq", "dt", "align", "assume",
    "org", "end", "byte", "word", "dword", "qword",
    "proc", "endp", "public", "extrn", "include",
    "segment", "ends", "unicode", "dup",
}


# ---------------------------------------------------------------------------
# Phase A: Extraction — PE files (from 01_build_dataset.py)
# ---------------------------------------------------------------------------
def extract_opcodes_pe(filepath: str) -> list[str]:
    """
    Disassemble a PE file and return an ordered list of opcode mnemonics.

    Steps
    -----
    1. Parse the PE header with ``pefile``.
    2. Locate the ``.text`` section (primary executable code).
    3. Feed the raw bytes into the Capstone disassembler (x86-64 first,
       falling back to x86-32).
    4. Return **only** the mnemonics — operands, registers, and memory
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
# Phase A: Extraction — IDA Pro .asm files (from 04_build_ms_dataset.py)
# ---------------------------------------------------------------------------
def extract_opcodes_asm(filepath: str) -> list[str]:
    """
    Parse an IDA Pro .asm file and return an ordered list of opcodes.

    Only mnemonics from executable sections are extracted.
    Directives and data definitions are filtered out.
    """
    opcodes: list[str] = []
    try:
        with open(filepath, "r", errors="ignore") as f:
            for line in f:
                m = IDA_INSN_RE.match(line)
                if m:
                    mnemonic = m.group(1).lower()
                    if mnemonic not in SKIP_MNEMONICS:
                        opcodes.append(mnemonic)
    except OSError as exc:
        print(f"  [SKIP] {filepath}: {exc}")
    return opcodes


# ---------------------------------------------------------------------------
# Phase A: Extraction — MalwareBazaar .asm files (from 06_build_extended_dataset.py)
# ---------------------------------------------------------------------------
def read_bazaar_asm(filepath: str) -> list[str]:
    """
    Read a MalwareBazaar-style .asm file (one opcode per line).

    These are produced by the MalwareBazaar downloader pipeline.
    """
    opcodes: list[str] = []
    try:
        with open(filepath, "r", errors="ignore") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    opcodes.append(stripped)
    except OSError as exc:
        print(f"  [SKIP] {filepath}: {exc}")
    return opcodes


# ---------------------------------------------------------------------------
# Auto-detection (from 06_build_extended_dataset.py)
# ---------------------------------------------------------------------------
def detect_asm_format(filepath: str) -> str:
    """
    Detect whether an .asm file is Bazaar-style (simple) or IDA-style.

    Returns 'bazaar' or 'ida'.
    """
    try:
        with open(filepath, "r", errors="ignore") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                # IDA-style lines start with a section:address pattern
                if re.match(r"^\s*\.\w+:[0-9A-Fa-f]+", stripped):
                    return "ida"
                # Bazaar-style: single word per line (just an opcode)
                if re.match(r"^[a-z]{2,10}$", stripped):
                    return "bazaar"
    except OSError:
        pass
    return "ida"  # default to IDA if indeterminate


def read_asm_file(filepath: str) -> list[str]:
    """Auto-detect format and read opcodes from an .asm file."""
    fmt = detect_asm_format(filepath)
    if fmt == "bazaar":
        return read_bazaar_asm(filepath)
    return extract_opcodes_asm(filepath)


# ---------------------------------------------------------------------------
# Phase B: Tokenisation
# ---------------------------------------------------------------------------
def build_vocabulary(
    all_opcode_lists: list[list[str]],
    pad_token: str = "<PAD>",
    unk_token: str = "<UNK>",
) -> dict[str, int]:
    """
    Create a ``stoi`` (string-to-int) vocabulary from all observed opcodes.

    Layout: <PAD>=0  <UNK>=1  then sorted unique mnemonics …
    """
    counter: Counter[str] = Counter()
    for ops in all_opcode_lists:
        counter.update(ops)

    sorted_ops = sorted(counter.keys())
    stoi: dict[str, int] = {pad_token: 0, unk_token: 1}
    for op in sorted_ops:
        if op not in stoi:
            stoi[op] = len(stoi)
    return stoi


def encode_sequence(
    opcodes: list[str],
    stoi: dict[str, int],
    max_len: int,
    pad_token: str = "<PAD>",
    unk_token: str = "<UNK>",
) -> np.ndarray:
    """
    Map an opcode list to integer IDs, truncate / pad to ``max_len``.
    """
    unk_id = stoi[unk_token]
    pad_id = stoi[pad_token]
    ids = [stoi.get(op, unk_id) for op in opcodes[:max_len]]
    # Pad
    ids += [pad_id] * (max_len - len(ids))
    return np.array(ids, dtype=np.int32)


# ---------------------------------------------------------------------------
# File collection utilities
# ---------------------------------------------------------------------------
def collect_pe_files(data_dir: Path) -> tuple[list[str], list[int]]:
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


def collect_asm_files(asm_dir: Path) -> list[str]:
    """Collect all .asm files from a directory (recursively)."""
    if not asm_dir.exists():
        return []
    return sorted(str(f) for f in asm_dir.rglob("*.asm"))
