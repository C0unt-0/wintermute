"""asm_directory.py — Pre-disassembled .asm files from family subdirectories."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from wintermute.data.etl.base import DataSource, RawSample
from wintermute.data.etl.registry import register_source

logger = logging.getLogger("wintermute.data.etl")


@register_source("asm_directory")
class AsmDirectorySource(DataSource):
    """Read pre-disassembled .asm files from a directory tree.

    Each subdirectory represents a malware family. Auto-detects file format
    (MalwareBazaar one-opcode-per-line vs IDA Pro section:address format).
    """

    name = "asm_directory"

    def validate_config(self) -> list[str]:
        data_dir = Path(self.get("data_dir", "data/bazaar"))
        errors = []
        if not data_dir.exists():
            errors.append(f"data_dir '{data_dir}' does not exist")
        return errors

    def extract(self) -> Iterable[RawSample]:
        from wintermute.data.tokenizer import read_asm_file

        data_dir = Path(self.get("data_dir", "data/bazaar"))
        min_opcodes = self.get("min_opcodes", 10)
        max_samples_per_family = self.get("max_samples_per_family", None)

        # Each subdirectory is a family
        family_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
        if not family_dirs:
            logger.warning("No family subdirectories found in %s", data_dir)
            return

        for label, family_dir in enumerate(family_dirs):
            family_name = family_dir.name
            asm_files = sorted(family_dir.glob("*.asm"))
            logger.info(
                "Family '%s': %d .asm files (label=%d)",
                family_name, len(asm_files), label,
            )

            count = 0
            for fpath in asm_files:
                if max_samples_per_family is not None and count >= max_samples_per_family:
                    break

                opcodes = read_asm_file(str(fpath))
                if len(opcodes) < min_opcodes:
                    continue

                yield RawSample(
                    opcodes=opcodes,
                    label=label,
                    family=family_name,
                    source_id=fpath.stem,
                )
                count += 1
