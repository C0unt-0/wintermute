"""test_etl_sources.py — Tests for individual ETL data sources.

Tests for MalShareSource with mocked HTTP interactions.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import wintermute.data.etl.sources  # noqa: F401  # trigger source auto-registration
from wintermute.data.etl.registry import SourceRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clean_registry():
    """Save and restore the registry between tests."""
    original = dict(SourceRegistry._sources)
    yield
    SourceRegistry._sources = original


# ═══════════════════════════════════════════════════════════════════════════
# TestMalShareSource
# ═══════════════════════════════════════════════════════════════════════════
class TestMalShareSource:
    """Tests for the MalShare ETL data source."""

    FAKE_API_KEY = "test_api_key_12345"
    FAKE_SHA256 = "a" * 64
    FAKE_SHA256_B = "b" * 64
    FAKE_OPCODES = ["mov", "push", "xor", "call", "ret", "add", "sub", "cmp", "jmp", "nop", "test"]

    def test_malshare_registration(self):
        """Verify 'malshare' is registered in the source registry."""
        available = SourceRegistry.available()
        assert "malshare" in available

    def test_malshare_missing_api_key(self):
        """validate_config returns an error when no API key is provided."""
        src = SourceRegistry.create("malshare", config={})
        with patch.dict("os.environ", {}, clear=True):
            # Remove MALSHARE_API_KEY from environment if present
            import os

            env_backup = os.environ.pop("MALSHARE_API_KEY", None)
            try:
                errors = src.validate_config()
                assert len(errors) == 1
                assert "API key" in errors[0]
            finally:
                if env_backup is not None:
                    os.environ["MALSHARE_API_KEY"] = env_backup

    def test_malshare_api_key_from_config(self):
        """validate_config passes when API key is provided in config."""
        src = SourceRegistry.create("malshare", config={"api_key": self.FAKE_API_KEY})
        errors = src.validate_config()
        assert errors == []

    def test_malshare_api_key_from_env(self):
        """validate_config passes when API key is in environment."""
        src = SourceRegistry.create("malshare", config={})
        with patch.dict("os.environ", {"MALSHARE_API_KEY": self.FAKE_API_KEY}):
            errors = src.validate_config()
            assert errors == []

    @patch("wintermute.data.etl.sources.malshare.requests.get")
    @patch("wintermute.data.etl.pe_utils.PEProcessor.disassemble_pe_bytes")
    def test_malshare_extract_with_mocked_api(self, mock_disasm, mock_get, tmp_path):
        """Mock HTTP to return hashes and PE bytes, verify RawSample output."""
        # Mock the type query response (list of hashes)
        type_response = MagicMock()
        type_response.status_code = 200
        type_response.json.return_value = [
            {"sha256": self.FAKE_SHA256, "md5": "x" * 32},
            {"sha256": self.FAKE_SHA256_B, "md5": "y" * 32},
        ]
        type_response.raise_for_status = MagicMock()

        # Mock the getfile response (raw PE bytes)
        file_response = MagicMock()
        file_response.status_code = 200
        file_response.content = b"MZ" + b"\x00" * 100
        file_response.raise_for_status = MagicMock()

        # requests.get returns type_response first, then file_response for each hash
        mock_get.side_effect = [type_response, file_response, file_response]

        # Mock disassembly to return opcodes
        mock_disasm.return_value = self.FAKE_OPCODES

        src = SourceRegistry.create(
            "malshare",
            config={
                "api_key": self.FAKE_API_KEY,
                "cache_dir": str(tmp_path / "malshare_cache"),
                "delay": 0.0,
                "min_opcodes": 5,
                "max_samples": 10,
                "file_type": "PE32",
            },
        )

        samples, result = src.run()
        assert result.samples_extracted == 2
        for sample in samples:
            assert sample.label == 1
            assert sample.family == ""
            assert sample.opcodes == self.FAKE_OPCODES
            assert len(sample.source_id) == 64

    @patch("wintermute.data.etl.sources.malshare.requests.get")
    def test_malshare_cache_hit(self, mock_get, tmp_path):
        """Pre-populated cache prevents download HTTP calls."""
        cache_dir = tmp_path / "malshare_cache"
        cache_dir.mkdir(parents=True)
        # Write a cached .asm file
        asm_file = cache_dir / f"{self.FAKE_SHA256}.asm"
        asm_file.write_text("\n".join(self.FAKE_OPCODES))

        # Mock the type query response
        type_response = MagicMock()
        type_response.status_code = 200
        type_response.json.return_value = [{"sha256": self.FAKE_SHA256}]
        type_response.raise_for_status = MagicMock()

        # Only one call (the type query) — no getfile call expected
        mock_get.side_effect = [type_response]

        src = SourceRegistry.create(
            "malshare",
            config={
                "api_key": self.FAKE_API_KEY,
                "cache_dir": str(cache_dir),
                "delay": 0.0,
                "min_opcodes": 5,
                "max_samples": 10,
            },
        )

        samples, result = src.run()
        assert result.samples_extracted == 1
        assert samples[0].opcodes == self.FAKE_OPCODES
        assert samples[0].source_id == self.FAKE_SHA256
        # Only the type query was made — no download call
        assert mock_get.call_count == 1

    @patch("wintermute.data.etl.sources.malshare.requests.get")
    def test_malshare_api_error(self, mock_get, tmp_path):
        """API errors are handled gracefully (no crash, zero samples)."""
        import requests as req

        mock_get.side_effect = req.RequestException("API rate limit exceeded")

        src = SourceRegistry.create(
            "malshare",
            config={
                "api_key": self.FAKE_API_KEY,
                "cache_dir": str(tmp_path / "malshare_cache"),
                "delay": 0.0,
                "max_samples": 10,
            },
        )

        samples, result = src.run()
        assert result.samples_extracted == 0
        assert len(samples) == 0

    @patch("wintermute.data.etl.sources.malshare.requests.get")
    @patch("wintermute.data.etl.pe_utils.PEProcessor.disassemble_pe_bytes")
    def test_malshare_download_failure_skips_sample(self, mock_disasm, mock_get, tmp_path):
        """When a single download fails, that sample is skipped but others continue."""
        # Mock the type query response with two hashes
        type_response = MagicMock()
        type_response.status_code = 200
        type_response.json.return_value = [
            {"sha256": self.FAKE_SHA256},
            {"sha256": self.FAKE_SHA256_B},
        ]
        type_response.raise_for_status = MagicMock()

        # First download fails, second succeeds
        import requests as req

        fail_response = MagicMock()
        fail_response.raise_for_status.side_effect = req.HTTPError("404 Not Found")

        ok_response = MagicMock()
        ok_response.status_code = 200
        ok_response.content = b"MZ" + b"\x00" * 100
        ok_response.raise_for_status = MagicMock()

        mock_get.side_effect = [type_response, fail_response, ok_response]
        mock_disasm.return_value = self.FAKE_OPCODES

        src = SourceRegistry.create(
            "malshare",
            config={
                "api_key": self.FAKE_API_KEY,
                "cache_dir": str(tmp_path / "malshare_cache"),
                "delay": 0.0,
                "min_opcodes": 5,
                "max_samples": 10,
            },
        )

        samples, result = src.run()
        # Only the second sample should succeed
        assert result.samples_extracted == 1
        assert samples[0].source_id == self.FAKE_SHA256_B

    @patch("wintermute.data.etl.sources.malshare.requests.get")
    def test_malshare_empty_hash_list(self, mock_get, tmp_path):
        """Empty response from type query yields zero samples."""
        type_response = MagicMock()
        type_response.status_code = 200
        type_response.json.return_value = []
        type_response.raise_for_status = MagicMock()

        mock_get.side_effect = [type_response]

        src = SourceRegistry.create(
            "malshare",
            config={
                "api_key": self.FAKE_API_KEY,
                "cache_dir": str(tmp_path / "malshare_cache"),
                "delay": 0.0,
            },
        )

        samples, result = src.run()
        assert result.samples_extracted == 0
        assert len(samples) == 0

    @patch("wintermute.data.etl.sources.malshare.requests.get")
    def test_malshare_max_samples_limit(self, mock_get, tmp_path):
        """max_samples config limits the number of hashes processed."""
        # Return 10 hashes but limit to 2
        hashes_data = [{"sha256": f"{i:064x}"} for i in range(10)]

        type_response = MagicMock()
        type_response.status_code = 200
        type_response.json.return_value = hashes_data
        type_response.raise_for_status = MagicMock()

        mock_get.side_effect = [type_response]

        src = SourceRegistry.create(
            "malshare",
            config={
                "api_key": self.FAKE_API_KEY,
                "cache_dir": str(tmp_path / "malshare_cache"),
                "delay": 0.0,
                "max_samples": 2,
            },
        )

        # Call _get_sample_hashes directly to verify limiting
        hashes = src._get_sample_hashes(self.FAKE_API_KEY, "PE32", 2)
        assert len(hashes) == 2
