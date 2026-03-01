"""test_etl_sources.py — Tests for individual ETL data sources.

Tests for MalShareSource, URLhausSource, and VirusTotalSource with mocked HTTP interactions.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

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


# ═══════════════════════════════════════════════════════════════════════════
# TestURLhausSource
# ═══════════════════════════════════════════════════════════════════════════
class TestURLhausSource:
    """Tests for the URLhaus ETL data source."""

    FAKE_SHA256 = "a" * 64
    FAKE_SHA256_B = "b" * 64
    FAKE_OPCODES = ["mov", "push", "xor", "call", "ret", "add", "sub", "cmp", "jmp", "nop", "test"]

    def _make_payload(
        self,
        sha256: str,
        file_type: str = "exe",
        signature: str | None = "AgentTesla",
    ) -> dict:
        """Build a fake URLhaus payload entry."""
        return {
            "sha256_hash": sha256,
            "file_type": file_type,
            "signature": signature,
            "urlhaus_download": f"https://urlhaus-api.abuse.ch/v1/download/{sha256}/",
        }

    def test_urlhaus_registration(self):
        """Verify 'urlhaus' is registered in the source registry."""
        available = SourceRegistry.available()
        assert "urlhaus" in available

    def test_urlhaus_no_api_key_required(self):
        """validate_config returns empty list (no API key needed)."""
        src = SourceRegistry.create("urlhaus", config={})
        errors = src.validate_config()
        assert errors == []

    @patch("wintermute.data.etl.sources.urlhaus.requests.post")
    @patch("wintermute.data.etl.sources.urlhaus.requests.get")
    @patch("wintermute.data.etl.pe_utils.PEProcessor.unzip_encrypted")
    @patch("wintermute.data.etl.pe_utils.PEProcessor.disassemble_pe_bytes")
    def test_urlhaus_extract_with_mocked_api(
        self, mock_disasm, mock_unzip, mock_get, mock_post, tmp_path
    ):
        """Mock POST to payloads/recent/ and GET to download/, verify RawSample output."""
        # Mock recent payloads POST response
        recent_response = MagicMock()
        recent_response.status_code = 200
        recent_response.json.return_value = {
            "payloads": [
                self._make_payload(self.FAKE_SHA256, "exe", "AgentTesla"),
                self._make_payload(self.FAKE_SHA256_B, "dll", "Emotet"),
            ]
        }
        recent_response.raise_for_status = MagicMock()
        mock_post.return_value = recent_response

        # Mock download GET response (AES-encrypted ZIP bytes)
        dl_response = MagicMock()
        dl_response.status_code = 200
        dl_response.content = b"PK" + b"\x00" * 100  # fake zip
        dl_response.raise_for_status = MagicMock()
        mock_get.return_value = dl_response

        # Mock unzip -> PE bytes
        mock_unzip.return_value = b"MZ" + b"\x00" * 100

        # Mock disassembly -> opcodes
        mock_disasm.return_value = self.FAKE_OPCODES

        src = SourceRegistry.create(
            "urlhaus",
            config={
                "cache_dir": str(tmp_path / "urlhaus_cache"),
                "delay": 0.0,
                "min_opcodes": 5,
                "max_samples": 10,
                "recent_limit": 100,
                "file_types": ["exe", "dll"],
            },
        )

        samples, result = src.run()
        assert result.samples_extracted == 2
        # Verify family names come from the signature field
        families = {s.family for s in samples}
        assert "AgentTesla" in families
        assert "Emotet" in families
        for sample in samples:
            assert sample.label == 1
            assert sample.opcodes == self.FAKE_OPCODES
            assert len(sample.source_id) == 64

    @patch("wintermute.data.etl.sources.urlhaus.requests.post")
    @patch("wintermute.data.etl.sources.urlhaus.requests.get")
    @patch("wintermute.data.etl.pe_utils.PEProcessor.unzip_encrypted")
    @patch("wintermute.data.etl.pe_utils.PEProcessor.disassemble_pe_bytes")
    def test_urlhaus_pe_filtering(self, mock_disasm, mock_unzip, mock_get, mock_post, tmp_path):
        """Only PE file types (exe, dll) are processed; elf/doc are skipped."""
        recent_response = MagicMock()
        recent_response.status_code = 200
        recent_response.json.return_value = {
            "payloads": [
                self._make_payload(self.FAKE_SHA256, "exe", "AgentTesla"),
                self._make_payload("c" * 64, "elf", "Mirai"),
                self._make_payload("d" * 64, "doc", "Emotet"),
                self._make_payload(self.FAKE_SHA256_B, "dll", "TrickBot"),
            ]
        }
        recent_response.raise_for_status = MagicMock()
        mock_post.return_value = recent_response

        dl_response = MagicMock()
        dl_response.status_code = 200
        dl_response.content = b"PK" + b"\x00" * 100
        dl_response.raise_for_status = MagicMock()
        mock_get.return_value = dl_response

        mock_unzip.return_value = b"MZ" + b"\x00" * 100
        mock_disasm.return_value = self.FAKE_OPCODES

        src = SourceRegistry.create(
            "urlhaus",
            config={
                "cache_dir": str(tmp_path / "urlhaus_cache"),
                "delay": 0.0,
                "min_opcodes": 5,
                "max_samples": 10,
                "recent_limit": 100,
                "file_types": ["exe", "dll"],
            },
        )

        samples, result = src.run()
        # Only exe and dll should be processed (elf and doc filtered out)
        assert result.samples_extracted == 2
        source_ids = {s.source_id for s in samples}
        assert self.FAKE_SHA256 in source_ids
        assert self.FAKE_SHA256_B in source_ids
        # elf and doc hashes should not appear
        assert "c" * 64 not in source_ids
        assert "d" * 64 not in source_ids

    @patch("wintermute.data.etl.sources.urlhaus.requests.post")
    def test_urlhaus_cache_hit(self, mock_post, tmp_path):
        """Pre-populated cache prevents download HTTP calls."""
        cache_dir = tmp_path / "urlhaus_cache"
        cache_dir.mkdir(parents=True)
        # Write a cached .asm file
        asm_file = cache_dir / f"{self.FAKE_SHA256}.asm"
        asm_file.write_text("\n".join(self.FAKE_OPCODES))

        # Mock recent payloads response
        recent_response = MagicMock()
        recent_response.status_code = 200
        recent_response.json.return_value = {
            "payloads": [
                self._make_payload(self.FAKE_SHA256, "exe", "Formbook"),
            ]
        }
        recent_response.raise_for_status = MagicMock()
        mock_post.return_value = recent_response

        src = SourceRegistry.create(
            "urlhaus",
            config={
                "cache_dir": str(cache_dir),
                "delay": 0.0,
                "min_opcodes": 5,
                "max_samples": 10,
                "recent_limit": 100,
                "file_types": ["exe", "dll"],
            },
        )

        # Patch requests.get to track calls — should NOT be called
        with patch("wintermute.data.etl.sources.urlhaus.requests.get") as mock_get:
            samples, result = src.run()
            assert result.samples_extracted == 1
            assert samples[0].opcodes == self.FAKE_OPCODES
            assert samples[0].source_id == self.FAKE_SHA256
            assert samples[0].family == "Formbook"
            # No download calls should have been made
            mock_get.assert_not_called()

    @patch("wintermute.data.etl.sources.urlhaus.requests.post")
    def test_urlhaus_api_error(self, mock_post, tmp_path):
        """API errors are handled gracefully (no crash, zero samples)."""
        mock_post.side_effect = requests.RequestException("Connection timeout")

        src = SourceRegistry.create(
            "urlhaus",
            config={
                "cache_dir": str(tmp_path / "urlhaus_cache"),
                "delay": 0.0,
                "max_samples": 10,
                "recent_limit": 100,
            },
        )

        samples, result = src.run()
        assert result.samples_extracted == 0
        assert len(samples) == 0


# ═══════════════════════════════════════════════════════════════════════════
# TestVirusTotalSource
# ═══════════════════════════════════════════════════════════════════════════
class TestVirusTotalSource:
    """Tests for the VirusTotal ETL data source."""

    FAKE_API_KEY = "vt_test_api_key_12345"
    FAKE_SHA256 = "a" * 64
    FAKE_SHA256_B = "b" * 64
    FAKE_OPCODES = ["mov", "push", "xor", "call", "ret", "add", "sub", "cmp", "jmp", "nop", "test"]

    def _make_vt_report(
        self,
        sha256: str,
        malicious: int = 50,
        undetected: int = 10,
        popular_threat_name: str = "Trojan.GenericKD",
    ) -> dict:
        """Build a fake VirusTotal /files/{id} API response."""
        return {
            "data": {
                "id": sha256,
                "type": "file",
                "attributes": {
                    "sha256": sha256,
                    "last_analysis_stats": {
                        "malicious": malicious,
                        "undetected": undetected,
                        "suspicious": 0,
                        "harmless": 0,
                        "timeout": 0,
                        "failure": 0,
                        "type-unsupported": 0,
                    },
                    "popular_threat_name": popular_threat_name,
                },
            }
        }

    def test_vt_registration(self):
        """Verify 'virustotal' is registered in the source registry."""
        available = SourceRegistry.available()
        assert "virustotal" in available

    def test_vt_missing_api_key(self):
        """validate_config returns an error when no API key is provided."""
        src = SourceRegistry.create("virustotal", config={})
        import os

        env_backup = os.environ.pop("VT_API_KEY", None)
        try:
            errors = src.validate_config()
            assert len(errors) == 1
            assert "API key" in errors[0]
        finally:
            if env_backup is not None:
                os.environ["VT_API_KEY"] = env_backup

    @patch("wintermute.data.etl.sources.virustotal.requests.get")
    def test_vt_enrich_mode(self, mock_get, tmp_path):
        """Mock VT API response, pre-populate cache_dir with .asm file, verify enriched RawSample."""
        # Pre-populate a cache directory with an .asm file
        cache_dir = tmp_path / "bazaar_cache"
        cache_dir.mkdir(parents=True)
        asm_file = cache_dir / f"{self.FAKE_SHA256}.asm"
        asm_file.write_text("\n".join(self.FAKE_OPCODES))

        # Create a hash file
        hash_file = tmp_path / "hashes.txt"
        hash_file.write_text(f"{self.FAKE_SHA256}\n")

        # Mock VT API response
        vt_response = MagicMock()
        vt_response.status_code = 200
        vt_response.json.return_value = self._make_vt_report(
            self.FAKE_SHA256,
            malicious=50,
            undetected=10,
            popular_threat_name="Trojan.GenericKD",
        )
        vt_response.raise_for_status = MagicMock()
        mock_get.return_value = vt_response

        src = SourceRegistry.create(
            "virustotal",
            config={
                "api_key": self.FAKE_API_KEY,
                "cache_dir": str(tmp_path / "vt_cache"),
                "mode": "enrich",
                "delay": 0.0,
                "min_opcodes": 5,
                "max_samples": 10,
                "hash_file": str(hash_file),
                "cache_dirs": [str(cache_dir)],
                "min_detection_ratio": 0.5,
            },
        )

        samples, result = src.run()
        assert result.samples_extracted == 1
        sample = samples[0]
        assert sample.opcodes == self.FAKE_OPCODES
        assert sample.label == 1
        assert sample.family == "Trojan.GenericKD"
        assert sample.source_id == self.FAKE_SHA256
        assert "detection_ratio" in sample.metadata
        assert sample.metadata["detection_ratio"] == pytest.approx(50 / 60, rel=1e-2)

    @patch("wintermute.data.etl.sources.virustotal.requests.get")
    def test_vt_enrich_skips_uncached(self, mock_get, tmp_path):
        """Hash in hash_file but no .asm cached: skip without crashing."""
        hash_file = tmp_path / "hashes.txt"
        hash_file.write_text(f"{self.FAKE_SHA256}\n")

        # VT API returns a valid report
        vt_response = MagicMock()
        vt_response.status_code = 200
        vt_response.json.return_value = self._make_vt_report(self.FAKE_SHA256)
        vt_response.raise_for_status = MagicMock()
        mock_get.return_value = vt_response

        src = SourceRegistry.create(
            "virustotal",
            config={
                "api_key": self.FAKE_API_KEY,
                "cache_dir": str(tmp_path / "vt_cache"),
                "mode": "enrich",
                "delay": 0.0,
                "min_opcodes": 5,
                "max_samples": 10,
                "hash_file": str(hash_file),
                "cache_dirs": [],  # no cache dirs
                "min_detection_ratio": 0.5,
            },
        )

        samples, result = src.run()
        # Enrich mode with no cached .asm should produce zero samples
        assert result.samples_extracted == 0
        assert len(samples) == 0

    @patch("wintermute.data.etl.sources.virustotal.requests.get")
    def test_vt_detection_ratio_filter(self, mock_get, tmp_path):
        """VT response with low detection ratio is filtered out."""
        # Pre-populate cache
        cache_dir = tmp_path / "bazaar_cache"
        cache_dir.mkdir(parents=True)
        asm_file = cache_dir / f"{self.FAKE_SHA256}.asm"
        asm_file.write_text("\n".join(self.FAKE_OPCODES))

        hash_file = tmp_path / "hashes.txt"
        hash_file.write_text(f"{self.FAKE_SHA256}\n")

        # Low detection: 2 malicious out of 60 total = ~3.3%
        vt_response = MagicMock()
        vt_response.status_code = 200
        vt_response.json.return_value = self._make_vt_report(
            self.FAKE_SHA256,
            malicious=2,
            undetected=58,
            popular_threat_name="Benign.Test",
        )
        vt_response.raise_for_status = MagicMock()
        mock_get.return_value = vt_response

        src = SourceRegistry.create(
            "virustotal",
            config={
                "api_key": self.FAKE_API_KEY,
                "cache_dir": str(tmp_path / "vt_cache"),
                "mode": "enrich",
                "delay": 0.0,
                "min_opcodes": 5,
                "max_samples": 10,
                "hash_file": str(hash_file),
                "cache_dirs": [str(cache_dir)],
                "min_detection_ratio": 0.5,
            },
        )

        samples, result = src.run()
        # Low detection ratio should be filtered out
        assert result.samples_extracted == 0
        assert len(samples) == 0

    def test_vt_hash_file_loading(self, tmp_path):
        """Verify hashes are loaded correctly from a hash file."""
        from wintermute.data.etl.sources.virustotal import VirusTotalSource

        hash_file = tmp_path / "hashes.txt"
        hashes_content = "\n".join(
            [
                self.FAKE_SHA256,
                self.FAKE_SHA256_B,
                "not-a-valid-hash",
                "",
                "c" * 64,
            ]
        )
        hash_file.write_text(hashes_content)

        loaded = VirusTotalSource._get_hashes(str(hash_file), [], 100)
        # Should include 3 valid SHA-256 hashes, skip the invalid one and blank line
        assert len(loaded) == 3
        assert self.FAKE_SHA256 in loaded
        assert self.FAKE_SHA256_B in loaded
        assert "c" * 64 in loaded

    @patch("wintermute.data.etl.sources.virustotal.requests.get")
    def test_vt_api_error(self, mock_get, tmp_path):
        """API errors are handled gracefully (no crash, zero samples)."""
        hash_file = tmp_path / "hashes.txt"
        hash_file.write_text(f"{self.FAKE_SHA256}\n")

        mock_get.side_effect = requests.RequestException("API rate limit exceeded")

        src = SourceRegistry.create(
            "virustotal",
            config={
                "api_key": self.FAKE_API_KEY,
                "cache_dir": str(tmp_path / "vt_cache"),
                "mode": "enrich",
                "delay": 0.0,
                "min_opcodes": 5,
                "max_samples": 10,
                "hash_file": str(hash_file),
                "cache_dirs": [],
                "min_detection_ratio": 0.5,
            },
        )

        samples, result = src.run()
        assert result.samples_extracted == 0
        assert len(samples) == 0


# ═══════════════════════════════════════════════════════════════════════════
# TestThreatFoxSource
# ═══════════════════════════════════════════════════════════════════════════
class TestThreatFoxSource:
    """Tests for the ThreatFox ETL data source."""

    FAKE_SHA256 = "a" * 64
    FAKE_SHA256_B = "b" * 64
    FAKE_OPCODES = ["mov", "push", "xor", "call", "ret", "add", "sub", "cmp", "jmp", "nop", "test"]

    def _make_ioc(
        self,
        sha256: str,
        ioc_type: str = "sha256_hash",
        malware: str = "win.emotet",
        malware_printable: str = "Emotet",
        threat_type: str = "payload_delivery",
        tags: list[str] | None = None,
    ) -> dict:
        """Build a fake ThreatFox IOC entry."""
        return {
            "ioc_type": ioc_type,
            "ioc_value": sha256 if ioc_type == "sha256_hash" else f"https://evil.com/{sha256}",
            "threat_type": threat_type,
            "malware": malware,
            "malware_printable": malware_printable,
            "tags": tags or ["emotet"],
        }

    def test_threatfox_registration(self):
        """Verify 'threatfox' is registered in the source registry."""
        available = SourceRegistry.available()
        assert "threatfox" in available

    def test_threatfox_no_api_key_required(self):
        """validate_config returns empty list (no API key needed)."""
        src = SourceRegistry.create("threatfox", config={})
        errors = src.validate_config()
        assert errors == []

    @patch("wintermute.data.etl.sources.threatfox.requests.post")
    def test_threatfox_extract_with_cached_asm(self, mock_post, tmp_path):
        """Mock ThreatFox API returning IOCs, pre-populate cache dir, verify RawSample."""
        # Pre-populate a cache directory with a matching .asm file
        cache_dir = tmp_path / "bazaar_cache"
        cache_dir.mkdir(parents=True)
        asm_file = cache_dir / f"{self.FAKE_SHA256}.asm"
        asm_file.write_text("\n".join(self.FAKE_OPCODES))

        # Mock ThreatFox API response
        tf_response = MagicMock()
        tf_response.status_code = 200
        tf_response.json.return_value = {
            "query_status": "ok",
            "data": [
                self._make_ioc(
                    self.FAKE_SHA256,
                    malware="win.emotet",
                    malware_printable="Emotet",
                    threat_type="payload_delivery",
                    tags=["emotet", "epoch5"],
                ),
            ],
        }
        tf_response.raise_for_status = MagicMock()
        mock_post.return_value = tf_response

        src = SourceRegistry.create(
            "threatfox",
            config={
                "cache_dirs": [str(cache_dir)],
                "delay": 0.0,
                "min_opcodes": 5,
                "max_samples": 10,
                "days": 7,
                "download_missing": False,
            },
        )

        samples, result = src.run()
        assert result.samples_extracted == 1
        sample = samples[0]
        assert sample.opcodes == self.FAKE_OPCODES
        assert sample.label == 1
        assert sample.family == "Emotet"
        assert sample.source_id == self.FAKE_SHA256
        assert sample.metadata["threat_type"] == "payload_delivery"
        assert "emotet" in sample.metadata["tags"]

    @patch("wintermute.data.etl.sources.threatfox.requests.post")
    def test_threatfox_filters_non_hash_iocs(self, mock_post, tmp_path):
        """Only sha256_hash IOCs are processed; URLs and domains are skipped."""
        # Pre-populate cache for the hash IOC
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)
        asm_file = cache_dir / f"{self.FAKE_SHA256}.asm"
        asm_file.write_text("\n".join(self.FAKE_OPCODES))

        # Mix of IOC types — only sha256_hash should be processed
        tf_response = MagicMock()
        tf_response.status_code = 200
        tf_response.json.return_value = {
            "query_status": "ok",
            "data": [
                self._make_ioc(self.FAKE_SHA256, ioc_type="sha256_hash"),
                self._make_ioc("https://evil.com/malware.exe", ioc_type="url"),
                self._make_ioc("evil.com", ioc_type="domain"),
                self._make_ioc("192.168.1.1:443", ioc_type="ip:port"),
            ],
        }
        tf_response.raise_for_status = MagicMock()
        mock_post.return_value = tf_response

        src = SourceRegistry.create(
            "threatfox",
            config={
                "cache_dirs": [str(cache_dir)],
                "delay": 0.0,
                "min_opcodes": 5,
                "max_samples": 10,
                "days": 7,
                "download_missing": False,
            },
        )

        samples, result = src.run()
        # Only the sha256_hash IOC should produce a sample
        assert result.samples_extracted == 1
        assert samples[0].source_id == self.FAKE_SHA256

    @patch("wintermute.data.etl.sources.threatfox.requests.post")
    def test_threatfox_skips_uncached_when_download_disabled(self, mock_post, tmp_path):
        """No cached .asm and download_missing=false yields zero samples."""
        # No cache directory populated — no .asm files
        empty_cache = tmp_path / "empty_cache"
        empty_cache.mkdir(parents=True)

        tf_response = MagicMock()
        tf_response.status_code = 200
        tf_response.json.return_value = {
            "query_status": "ok",
            "data": [
                self._make_ioc(self.FAKE_SHA256),
                self._make_ioc(self.FAKE_SHA256_B),
            ],
        }
        tf_response.raise_for_status = MagicMock()
        mock_post.return_value = tf_response

        src = SourceRegistry.create(
            "threatfox",
            config={
                "cache_dirs": [str(empty_cache)],
                "delay": 0.0,
                "min_opcodes": 5,
                "max_samples": 10,
                "days": 7,
                "download_missing": False,
            },
        )

        samples, result = src.run()
        assert result.samples_extracted == 0
        assert len(samples) == 0

    @patch("wintermute.data.etl.sources.threatfox.requests.post")
    def test_threatfox_api_error(self, mock_post, tmp_path):
        """API errors are handled gracefully (no crash, zero samples)."""
        mock_post.side_effect = requests.RequestException("Connection timeout")

        src = SourceRegistry.create(
            "threatfox",
            config={
                "cache_dirs": [],
                "delay": 0.0,
                "max_samples": 10,
                "days": 7,
            },
        )

        samples, result = src.run()
        assert result.samples_extracted == 0
        assert len(samples) == 0

    @patch("wintermute.data.etl.sources.threatfox.requests.post")
    def test_threatfox_nested_cache_lookup(self, mock_post, tmp_path):
        """Verify nested cache layout (dir/<family>/<sha256>.asm) is found."""
        cache_dir = tmp_path / "cache"
        nested_dir = cache_dir / "Emotet"
        nested_dir.mkdir(parents=True)
        asm_file = nested_dir / f"{self.FAKE_SHA256}.asm"
        asm_file.write_text("\n".join(self.FAKE_OPCODES))

        tf_response = MagicMock()
        tf_response.status_code = 200
        tf_response.json.return_value = {
            "query_status": "ok",
            "data": [
                self._make_ioc(self.FAKE_SHA256, malware_printable="Emotet"),
            ],
        }
        tf_response.raise_for_status = MagicMock()
        mock_post.return_value = tf_response

        src = SourceRegistry.create(
            "threatfox",
            config={
                "cache_dirs": [str(cache_dir)],
                "delay": 0.0,
                "min_opcodes": 5,
                "max_samples": 10,
                "days": 7,
                "download_missing": False,
            },
        )

        samples, result = src.run()
        assert result.samples_extracted == 1
        assert samples[0].family == "Emotet"
