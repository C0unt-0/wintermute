"""pe_files.py — Local PE binaries -> Capstone disassembly."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from wintermute.data.etl.base import DataSource, RawSample
from wintermute.data.etl.registry import register_source

logger = logging.getLogger("wintermute.data.etl")

PE_EXTENSIONS = {".exe", ".dll", ".sys", ".ocx", ".scr"}


@register_source("pe_files")
class PEFilesSource(DataSource):
    """Disassemble local PE binaries using Capstone.

    Expects ``data_dir/safe/`` and ``data_dir/malicious/`` subdirectories.
    Delegates to ``tokenizer.extract_opcodes_pe()``.
    """

    name = "pe_files"

    def validate_config(self) -> list[str]:
        data_dir = Path(self.get("data_dir", "data/raw"))
        errors = []
        if not data_dir.exists():
            errors.append(f"data_dir '{data_dir}' does not exist")
        return errors

    def extract(self) -> Iterable[RawSample]:
        from wintermute.data.tokenizer import extract_opcodes_pe

        data_dir = Path(self.get("data_dir", "data/raw"))
        min_opcodes = self.get("min_opcodes", 10)

        for label, subdir, family in [(0, "safe", "safe"), (1, "malicious", "malicious")]:
            folder = data_dir / subdir
            if not folder.exists():
                logger.warning("Directory %s does not exist, skipping.", folder)
                continue

            files = sorted(f for f in folder.rglob("*") if f.suffix.lower() in PE_EXTENSIONS)
            logger.info("Found %d PE files in %s", len(files), folder)

            for fpath in files:
                opcodes = extract_opcodes_pe(str(fpath))
                if len(opcodes) < min_opcodes:
                    continue
                yield RawSample(
                    opcodes=opcodes,
                    label=label,
                    family=family,
                    source_id=fpath.name,
                )
