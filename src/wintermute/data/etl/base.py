"""base.py — DataSource ABC, RawSample, and result dataclasses."""

from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

logger = logging.getLogger("wintermute.data.etl")


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------
@dataclass
class RawSample:
    """Universal exchange format between Extract and Transform stages."""

    opcodes: list[str]
    label: int
    family: str = ""
    source_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractResult:
    """Summary returned by each source after extraction completes."""

    source_name: str
    samples_extracted: int = 0
    samples_skipped: int = 0
    samples_failed: int = 0
    elapsed_seconds: float = 0.0
    families_found: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"ExtractResult({self.source_name}: "
            f"{self.samples_extracted} extracted, "
            f"{self.samples_skipped} skipped, "
            f"{self.samples_failed} failed, "
            f"{self.elapsed_seconds:.1f}s)"
        )


@dataclass
class PipelineResult:
    """Full summary of a complete ETL run."""

    extract_results: list[ExtractResult]
    total_samples: int
    vocab_size: int
    num_classes: int
    families: dict[str, str]
    class_distribution: dict[int, int]
    output_dir: str
    elapsed_seconds: float
    x_shape: tuple[int, ...]
    y_shape: tuple[int, ...]


# ---------------------------------------------------------------------------
# Abstract Base Class
# ---------------------------------------------------------------------------
class DataSource(abc.ABC):
    """Base class for all ETL data sources.

    Subclasses must implement ``extract()`` to yield ``RawSample`` objects.
    Optional hooks: ``validate_config()``, ``setup()``, ``teardown()``.
    """

    name: str = "base"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    # --- Config helpers ---

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value with a default."""
        return self.config.get(key, default)

    def require(self, key: str) -> Any:
        """Get a config value or raise ValueError."""
        if key not in self.config:
            raise ValueError(f"Source '{self.name}' requires config key '{key}'")
        return self.config[key]

    # --- Lifecycle hooks ---

    @abc.abstractmethod
    def extract(self) -> Iterable[RawSample]:
        """Yield samples one at a time. MUST implement."""
        ...

    def validate_config(self) -> list[str]:
        """Return error messages. Empty list means valid. Optional."""
        return []

    def setup(self) -> None:
        """One-time init before extraction. Optional."""

    def teardown(self) -> None:
        """Cleanup after extraction. Optional."""

    # --- Pipeline interface ---

    def run(self) -> tuple[list[RawSample], ExtractResult]:
        """Called by pipeline. Wraps validate -> setup -> extract -> teardown."""
        result = ExtractResult(source_name=self.name)

        # Validate
        errors = self.validate_config()
        if errors:
            result.errors = errors
            logger.warning("Validation failed for '%s': %s", self.name, errors)
            return [], result

        t0 = time.monotonic()
        samples: list[RawSample] = []

        try:
            self.setup()
            for sample in self.extract():
                if not sample.opcodes:
                    result.samples_skipped += 1
                    continue
                samples.append(sample)
                result.samples_extracted += 1
                if sample.family:
                    result.families_found[sample.family] = (
                        result.families_found.get(sample.family, 0) + 1
                    )
        except Exception as exc:
            result.samples_failed += 1
            result.errors.append(str(exc))
            logger.error("Error during extraction from '%s': %s", self.name, exc)
        finally:
            try:
                self.teardown()
            except Exception as exc:
                logger.warning("Teardown error for '%s': %s", self.name, exc)

        result.elapsed_seconds = time.monotonic() - t0
        logger.info(
            "Source '%s': %d extracted, %d skipped, %d failed (%.1fs)",
            self.name,
            result.samples_extracted,
            result.samples_skipped,
            result.samples_failed,
            result.elapsed_seconds,
        )
        return samples, result
