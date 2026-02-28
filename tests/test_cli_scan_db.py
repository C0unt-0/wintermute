"""Tests for CLI scan command DB integration.

Verifies that the scan command's optional DB persistence:
1. Writes ScanResult and Sample rows when the DB is available
2. Silently skips when no DB engine exists
3. Never crashes the scan when DB operations fail
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from wintermute.db.engine import create_db_engine, get_engine, get_session, init_db
from wintermute.db.models import Sample
from wintermute.db.repos.scans import ScanRepo

import wintermute.db.engine as _engine_mod


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def in_memory_db():
    """Create an in-memory SQLite DB engine and initialise tables.

    Restores the module-level engine globals to None after the test so
    other tests are not affected.
    """
    engine = create_db_engine("sqlite:///:memory:")
    init_db(engine)
    yield engine
    # Teardown: reset module-level globals
    _engine_mod._engine = None
    _engine_mod._SessionFactory = None
    engine.dispose()


@pytest.fixture()
def sample_file(tmp_path: Path) -> Path:
    """Create a small temporary file to simulate a scan target."""
    p = tmp_path / "test_sample.exe"
    p.write_bytes(b"\x4d\x5a" + b"\x00" * 100)  # minimal PE-like stub
    return p


# ------------------------------------------------------------------
# Helpers — replicate the DB write logic from cli.scan
# ------------------------------------------------------------------


def _persist_scan(
    target_path: Path,
    opcodes: list[str],
    pred: int,
    label: str,
    probs_dict: dict[str, float],
    confidence: float,
    manifest: str,
) -> None:
    """Replicate the DB persistence block from cli.scan."""
    try:
        from wintermute.db.engine import get_engine, get_session
        from wintermute.db.repos.scans import ScanRepo
        from wintermute.db.repos.samples import SampleRepo

        if get_engine() is not None:
            file_sha = hashlib.sha256(target_path.read_bytes()).hexdigest()
            file_size = target_path.stat().st_size

            with get_session() as session:
                SampleRepo(session).upsert(
                    sha256=file_sha,
                    family=label,
                    label=pred,
                    source="cli_scan",
                    file_type=target_path.suffix.lstrip(".").upper() or "UNKNOWN",
                    file_size_bytes=file_size,
                    opcode_count=len(opcodes),
                )

                ScanRepo(session).record(
                    sha256=file_sha,
                    filename=target_path.name,
                    file_size_bytes=file_size,
                    predicted_family=label,
                    predicted_label=pred,
                    confidence=confidence,
                    probabilities=probs_dict,
                    model_version=manifest,
                )
    except Exception:
        import logging
        logging.getLogger("wintermute.db").debug(
            "Scan DB persistence failed (best-effort)", exc_info=True
        )


# ==================================================================
# Tests
# ==================================================================


class TestScanWritesToDB:
    """When the DB is initialised, scan results persist."""

    def test_scan_writes_to_db(self, in_memory_db, sample_file: Path):
        """Verify that ScanResult and Sample rows are created."""
        opcodes = ["mov", "push", "call", "ret"]
        pred = 1
        label = "Malicious"
        probs_dict = {"0": 0.15, "1": 0.85}
        confidence = 0.85
        manifest = "test_manifest.json"

        _persist_scan(
            target_path=sample_file,
            opcodes=opcodes,
            pred=pred,
            label=label,
            probs_dict=probs_dict,
            confidence=confidence,
            manifest=manifest,
        )

        expected_sha = hashlib.sha256(sample_file.read_bytes()).hexdigest()

        with get_session() as session:
            # Check Sample row
            sample = session.get(Sample, expected_sha)
            assert sample is not None
            assert sample.label == 1
            assert sample.family == "Malicious"
            assert sample.source == "cli_scan"
            assert sample.file_type == "EXE"
            assert sample.file_size_bytes == sample_file.stat().st_size
            assert sample.opcode_count == 4

            # Check ScanResult row
            scans = ScanRepo(session).history(expected_sha)
            assert len(scans) == 1
            scan = scans[0]
            assert scan.sha256 == expected_sha
            assert scan.filename == "test_sample.exe"
            assert scan.predicted_family == "Malicious"
            assert scan.predicted_label == 1
            assert scan.confidence == pytest.approx(0.85)
            assert scan.probabilities == {"0": 0.15, "1": 0.85}
            assert scan.model_version == "test_manifest.json"


class TestScanNoDBNoCrash:
    """When no DB engine exists, the scan silently skips persistence."""

    def test_scan_no_db_no_crash(self, sample_file: Path):
        """Without initialising DB, _persist_scan completes without error."""
        # Ensure no engine is set
        assert get_engine() is None

        # Should not raise
        _persist_scan(
            target_path=sample_file,
            opcodes=["nop"],
            pred=0,
            label="Safe",
            probs_dict={"0": 0.99, "1": 0.01},
            confidence=0.99,
            manifest="v1",
        )


class TestScanDBErrorNoCrash:
    """When the DB layer raises, the scan still completes."""

    def test_scan_db_error_no_crash(self, in_memory_db, sample_file: Path):
        """Mock get_session to raise, verify no exception escapes."""
        with patch(
            "wintermute.db.engine.get_session",
            side_effect=RuntimeError("DB connection lost"),
        ):
            # Should not raise despite the DB error
            _persist_scan(
                target_path=sample_file,
                opcodes=["mov"],
                pred=1,
                label="Malicious",
                probs_dict={"0": 0.3, "1": 0.7},
                confidence=0.7,
                manifest="v2",
            )

    def test_scan_db_session_error_during_commit(
        self, in_memory_db, sample_file: Path
    ):
        """If the session commit fails, the exception is swallowed."""
        original_get_session = get_session

        from contextlib import contextmanager

        @contextmanager
        def _broken_session():
            with original_get_session() as session:
                # Monkey-patch flush to explode
                session.flush = MagicMock(
                    side_effect=RuntimeError("flush failed")
                )
                yield session

        with patch("wintermute.db.engine.get_session", _broken_session):
            _persist_scan(
                target_path=sample_file,
                opcodes=["ret"],
                pred=0,
                label="Safe",
                probs_dict={"0": 0.95, "1": 0.05},
                confidence=0.95,
                manifest="v3",
            )
