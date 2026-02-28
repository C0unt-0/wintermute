"""ms_dataset.py — Microsoft Malware Classification Kaggle dataset."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from wintermute.data.etl.base import DataSource, RawSample
from wintermute.data.etl.registry import register_source

logger = logging.getLogger("wintermute.data.etl")

# 1-indexed family names from the Kaggle dataset
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


@register_source("ms_dataset")
class MSDatasetSource(DataSource):
    """Read IDA Pro .asm files from the Microsoft Malware Classification dataset.

    Maps the 9 malware families via ``labels.csv``. Labels are shifted
    from 1-indexed (CSV) to 0-indexed internally.
    """

    name = "ms_dataset"

    def validate_config(self) -> list[str]:
        data_dir = Path(self.get("data_dir", "data/ms-malware"))
        errors = []
        if not data_dir.exists():
            errors.append(f"data_dir '{data_dir}' does not exist")
        return errors

    def extract(self) -> Iterable[RawSample]:
        from wintermute.data.downloader import MSDatasetDownloader
        from wintermute.data.tokenizer import extract_opcodes_asm

        data_dir = Path(self.get("data_dir", "data/ms-malware"))
        labels_file = self.get("labels_file", "labels.csv")
        max_samples = self.get("max_samples", None)

        labels_path = data_dir / labels_file
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        # Load labels: {sample_id: 0-based label}
        labels = MSDatasetDownloader.load_labels(str(labels_path))
        logger.info("Loaded %d labels from %s", len(labels), labels_path)

        # Find .asm files
        asm_files = sorted(data_dir.glob("*.asm"))
        logger.info("Found %d .asm files in %s", len(asm_files), data_dir)

        count = 0
        for fpath in asm_files:
            if max_samples is not None and count >= max_samples:
                break

            sample_id = fpath.stem
            if sample_id not in labels:
                continue

            label = labels[sample_id]
            family = MS_FAMILIES.get(label + 1, f"class_{label}")

            opcodes = extract_opcodes_asm(str(fpath))
            if not opcodes:
                continue

            yield RawSample(
                opcodes=opcodes,
                label=label,
                family=family,
                source_id=sample_id,
            )
            count += 1
