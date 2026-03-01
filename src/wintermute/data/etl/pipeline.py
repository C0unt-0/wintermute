"""pipeline.py — ETL Pipeline orchestrator (Extract -> Transform -> Load)."""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from wintermute.data.etl.base import ExtractResult, PipelineResult, RawSample
from wintermute.data.etl.registry import SourceRegistry

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger("wintermute.data.etl")

# Default special tokens (MalBERT-compatible)
DEFAULT_SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<CLS>": 2,
    "<SEP>": 3,
    "<MASK>": 4,
}

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[4] / "configs" / "sources.yaml"


class Pipeline:
    """ETL pipeline: extract from sources, transform to encoded arrays, save."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        config_path: str | Path | None = None,
        db_session: Session | None = None,
    ) -> None:
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = self._load_config(Path(config_path))
        elif DEFAULT_CONFIG_PATH.exists():
            self.config = self._load_config(DEFAULT_CONFIG_PATH)
        else:
            self.config = {}

        pipe_cfg = self.config.get("pipeline", {})
        self.out_dir = Path(pipe_cfg.get("out_dir", "data/processed"))
        self.max_seq_length = pipe_cfg.get("max_seq_length", 2048)
        self.shuffle = pipe_cfg.get("shuffle", True)
        self.seed = pipe_cfg.get("seed", 42)

        self.special_tokens: dict[str, int] = dict(DEFAULT_SPECIAL_TOKENS)

        self._db_session = db_session
        self._etl_run = None

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        with open(path) as f:
            return yaml.safe_load(f) or {}

    # --- EXTRACT ---

    def _extract(
        self,
        source_filter: str | None = None,
    ) -> tuple[list[RawSample], list[ExtractResult]]:
        """Run extraction from all enabled sources."""
        sources_cfg = self.config.get("sources", {})
        all_samples: list[RawSample] = []
        all_results: list[ExtractResult] = []

        # Determine which sources to run
        if source_filter:
            source_names = [source_filter]
        else:
            source_names = list(sources_cfg.keys())

        for name in source_names:
            src_cfg = sources_cfg.get(name, {})

            # Check enabled flag
            if not src_cfg.get("enabled", True):
                logger.info("Source '%s' is disabled, skipping.", name)
                continue

            # Check if source is registered
            if SourceRegistry.get(name) is None:
                logger.warning(
                    "Source '%s' in config is not registered. "
                    "Available: %s",
                    name,
                    SourceRegistry.available(),
                )
                continue

            logger.info("Extracting from '%s' ...", name)
            source = SourceRegistry.create(name, config=src_cfg)
            samples, result = source.run()
            all_samples.extend(samples)
            all_results.append(result)

        return all_samples, all_results

    # --- TRANSFORM ---

    def _build_vocabulary(self, samples: list[RawSample]) -> dict[str, int]:
        """Build unified vocabulary from all samples."""
        counter: Counter[str] = Counter()
        for sample in samples:
            counter.update(sample.opcodes)

        # Start with special tokens
        stoi: dict[str, int] = dict(self.special_tokens)

        # Add sorted opcodes
        for op in sorted(counter.keys()):
            if op not in stoi:
                stoi[op] = len(stoi)

        return stoi

    def _remap_labels(
        self, samples: list[RawSample],
    ) -> tuple[list[RawSample], dict[str, str]]:
        """Remap labels to contiguous 0-based integers and build families map."""
        # Collect unique (label, family) pairs, preserving order of first appearance
        seen: dict[int, str] = {}
        for s in samples:
            if s.label not in seen:
                seen[s.label] = s.family

        # Build old_label -> new_label mapping (sorted by original label)
        sorted_labels = sorted(seen.keys())
        label_map = {old: new for new, old in enumerate(sorted_labels)}

        # Build families map
        families: dict[str, str] = {}
        for old_label in sorted_labels:
            new_label = label_map[old_label]
            families[str(new_label)] = seen[old_label]

        # Remap
        for s in samples:
            s.label = label_map[s.label]

        return samples, families

    def _encode(
        self,
        samples: list[RawSample],
        stoi: dict[str, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode opcode sequences to integer arrays."""
        unk_id = stoi.get("<UNK>", 1)
        pad_id = stoi.get("<PAD>", 0)

        x_rows = []
        y_labels = []
        for sample in samples:
            ids = [stoi.get(op, unk_id) for op in sample.opcodes[: self.max_seq_length]]
            ids += [pad_id] * (self.max_seq_length - len(ids))
            x_rows.append(ids)
            y_labels.append(sample.label)

        x_data = np.array(x_rows, dtype=np.int32)
        y_data = np.array(y_labels, dtype=np.int32)
        return x_data, y_data

    def _shuffle(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Deterministic shuffle."""
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(y))
        return x[order], y[order]

    # --- DB PERSISTENCE ---

    def _persist_to_db(
        self, samples: list[RawSample], extract_results: list[ExtractResult],
    ) -> None:
        """Write samples and ETL run to database. No-op if no session."""
        if self._db_session is None:
            return
        try:
            import hashlib

            from wintermute.db.models import EtlRun, EtlRunSource
            from wintermute.db.repos.samples import SampleRepo

            # Create ETL run record
            config_str = json.dumps(self.config, sort_keys=True)
            config_hash = hashlib.sha256(config_str.encode()).hexdigest()

            self._etl_run = EtlRun(
                config_hash=config_hash,
                config=self.config,
                output_dir=str(self.out_dir),
            )
            self._db_session.add(self._etl_run)
            self._db_session.flush()

            # Bulk insert samples
            repo = SampleRepo(self._db_session)
            sample_dicts = []
            for s in samples:
                sha256 = hashlib.sha256(
                    "|".join(s.opcodes).encode()
                ).hexdigest()
                sample_dicts.append({
                    "sha256": sha256,
                    "family": s.family or "",
                    "label": s.label,
                    "source": s.source_id or "unknown",
                    "opcode_count": len(s.opcodes),
                    "etl_run_id": self._etl_run.id,
                })
            if sample_dicts:
                repo.bulk_insert(sample_dicts)

            # Write extract results as EtlRunSource rows
            for er in extract_results:
                src = EtlRunSource(
                    etl_run_id=self._etl_run.id,
                    source_name=er.source_name,
                    samples_extracted=er.samples_extracted,
                    samples_skipped=er.samples_skipped,
                    samples_failed=er.samples_failed,
                    families_found=er.families_found,
                    errors=er.errors,
                    elapsed_seconds=er.elapsed_seconds,
                )
                self._db_session.add(src)

            self._db_session.flush()
            logger.info(
                "DB: recorded %d samples and ETL run %s",
                len(sample_dicts),
                self._etl_run.id,
            )

        except Exception:
            logger.warning("DB: failed to persist ETL data", exc_info=True)
            self._etl_run = None  # Clear so _complete_etl_run skips

    def _complete_etl_run(self, result: PipelineResult) -> None:
        """Update ETL run record with completion stats. No-op if no session/run."""
        if self._db_session is None or self._etl_run is None:
            return
        try:
            from datetime import datetime, timezone

            self._etl_run.total_samples = result.total_samples
            self._etl_run.vocab_size = result.vocab_size
            self._etl_run.num_classes = result.num_classes
            self._etl_run.completed_at = datetime.now(timezone.utc)
            self._etl_run.elapsed_seconds = result.elapsed_seconds
            self._db_session.flush()
            logger.info("DB: completed ETL run %s", self._etl_run.id)
        except Exception:
            logger.warning("DB: failed to update ETL run", exc_info=True)

    # --- LOAD ---

    def _save(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        stoi: dict[str, int],
        families: dict[str, str],
        extract_results: list[ExtractResult],
        elapsed: float,
    ) -> PipelineResult:
        """Save all artifacts to out_dir."""
        self.out_dir.mkdir(parents=True, exist_ok=True)

        np.save(self.out_dir / "x_data.npy", x_data)
        np.save(self.out_dir / "y_data.npy", y_data)

        with open(self.out_dir / "vocab.json", "w") as f:
            json.dump(stoi, f, indent=2)

        with open(self.out_dir / "families.json", "w") as f:
            json.dump(families, f, indent=2)

        # Class distribution
        unique, counts = np.unique(y_data, return_counts=True)
        class_dist = {int(k): int(v) for k, v in zip(unique, counts)}
        num_classes = len(unique)

        # Manifest
        manifest = {
            "pipeline_config": {
                "max_seq_length": self.max_seq_length,
                "seed": self.seed,
                "shuffle": self.shuffle,
            },
            "x_shape": list(x_data.shape),
            "y_shape": list(y_data.shape),
            "vocab_size": len(stoi),
            "num_classes": num_classes,
            "class_distribution": class_dist,
            "sources": {},
        }
        for er in extract_results:
            manifest["sources"][er.source_name] = {
                "samples": er.samples_extracted,
                "skipped": er.samples_skipped,
                "failed": er.samples_failed,
                "elapsed_s": round(er.elapsed_seconds, 2),
                "families": er.families_found,
                "errors": er.errors,
            }

        with open(self.out_dir / "etl_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        result = PipelineResult(
            extract_results=extract_results,
            total_samples=len(y_data),
            vocab_size=len(stoi),
            num_classes=num_classes,
            families=families,
            class_distribution=class_dist,
            output_dir=str(self.out_dir),
            elapsed_seconds=elapsed,
            x_shape=x_data.shape,
            y_shape=y_data.shape,
        )
        return result

    # --- PUBLIC API ---

    def run(
        self,
        source_filter: str | None = None,
        dry_run: bool = False,
    ) -> PipelineResult:
        """Execute the full ETL pipeline."""
        t0 = time.monotonic()

        # --- Extract ---
        logger.info("Starting ETL pipeline ...")
        samples, extract_results = self._extract(source_filter)

        # --- DB: record samples and ETL run ---
        self._persist_to_db(samples, extract_results)

        if not samples:
            logger.warning("No samples extracted from any source.")
            return PipelineResult(
                extract_results=extract_results,
                total_samples=0,
                vocab_size=len(self.special_tokens),
                num_classes=0,
                families={},
                class_distribution={},
                output_dir=str(self.out_dir),
                elapsed_seconds=time.monotonic() - t0,
                x_shape=(0, self.max_seq_length),
                y_shape=(0,),
            )

        if dry_run:
            logger.info("Dry run — skipping transform and load.")
            return PipelineResult(
                extract_results=extract_results,
                total_samples=len(samples),
                vocab_size=0,
                num_classes=len({s.label for s in samples}),
                families={},
                class_distribution={},
                output_dir=str(self.out_dir),
                elapsed_seconds=time.monotonic() - t0,
                x_shape=(len(samples), self.max_seq_length),
                y_shape=(len(samples),),
            )

        # --- Transform ---
        logger.info("Transforming %d samples ...", len(samples))

        # Remap labels to contiguous 0-based
        samples, families = self._remap_labels(samples)

        # Build vocabulary
        stoi = self._build_vocabulary(samples)
        logger.info("Vocabulary: %d tokens (%d special + %d opcodes)",
                     len(stoi), len(self.special_tokens),
                     len(stoi) - len(self.special_tokens))

        # Encode
        x_data, y_data = self._encode(samples, stoi)

        # Shuffle
        if self.shuffle:
            x_data, y_data = self._shuffle(x_data, y_data)

        # --- Load ---
        elapsed = time.monotonic() - t0
        result = self._save(x_data, y_data, stoi, families, extract_results, elapsed)

        # --- DB: update ETL run with completion stats ---
        self._complete_etl_run(result)

        logger.info(
            "ETL complete: %d samples, %d classes, vocab=%d, "
            "x=%s, y=%s -> %s (%.1fs)",
            result.total_samples,
            result.num_classes,
            result.vocab_size,
            result.x_shape,
            result.y_shape,
            result.output_dir,
            result.elapsed_seconds,
        )
        return result
