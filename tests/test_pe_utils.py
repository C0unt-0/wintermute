"""test_pe_utils.py — Tests for PEProcessor and RateLimiter utilities."""

from __future__ import annotations

import io
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pyzipper
import pytest
import requests

from wintermute.data.etl.pe_utils import PEProcessor, RateLimiter

# A valid 64-hex-char SHA-256 for use across tests.
VALID_SHA = "a" * 64
VALID_SHA_2 = "b" * 64


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def processor(tmp_path: Path) -> PEProcessor:
    """PEProcessor with a temporary cache directory."""
    return PEProcessor(cache_dir=tmp_path, min_opcodes=5)


# ═══════════════════════════════════════════════════════════════════════════
# TestRateLimiter
# ═══════════════════════════════════════════════════════════════════════════
class TestRateLimiter:
    def test_consecutive_calls_enforce_delay(self):
        """Two consecutive wait() calls respect the minimum delay."""
        limiter = RateLimiter(delay=0.1)

        limiter.wait()
        t0 = time.monotonic()
        limiter.wait()
        elapsed = time.monotonic() - t0

        # Should have waited ~0.1s (allow some slack for CI)
        assert elapsed >= 0.08

    def test_no_delay_when_enough_time_passed(self):
        """If enough time has passed, wait() returns immediately."""
        limiter = RateLimiter(delay=0.01)

        limiter.wait()
        time.sleep(0.05)  # much longer than the delay

        t0 = time.monotonic()
        limiter.wait()
        elapsed = time.monotonic() - t0

        # Should be nearly instant
        assert elapsed < 0.02

    def test_rpm_calculates_delay(self):
        """rpm=4 produces a 15-second delay (60/4)."""
        limiter = RateLimiter(rpm=4)
        assert abs(limiter.delay - 15.0) < 0.001

    def test_rpm_overrides_delay(self):
        """When rpm is provided, it overrides the delay parameter."""
        limiter = RateLimiter(delay=999.0, rpm=60)
        assert abs(limiter.delay - 1.0) < 0.001


# ═══════════════════════════════════════════════════════════════════════════
# TestPEProcessorCaching
# ═══════════════════════════════════════════════════════════════════════════
class TestPEProcessorCaching:
    def test_is_cached_miss(self, processor: PEProcessor):
        """Returns False when no .asm file exists."""
        assert processor.is_cached(VALID_SHA) is False

    def test_is_cached_hit(self, processor: PEProcessor, tmp_path: Path):
        """Returns True when the .asm file exists."""
        asm_file = tmp_path / f"{VALID_SHA}.asm"
        asm_file.write_text("mov\npush\n")
        assert processor.is_cached(VALID_SHA) is True

    def test_is_cached_with_family(self, processor: PEProcessor, tmp_path: Path):
        """Checks inside the family subdirectory."""
        fam_dir = tmp_path / "Emotet"
        fam_dir.mkdir()
        (fam_dir / f"{VALID_SHA}.asm").write_text("mov\n")
        assert processor.is_cached(VALID_SHA, family="Emotet") is True
        assert processor.is_cached(VALID_SHA, family="Other") is False

    def test_get_cached_path_no_family(self, processor: PEProcessor, tmp_path: Path):
        """Path is cache_dir/sha256.asm when no family."""
        path = processor.get_cached_path(VALID_SHA)
        assert path == tmp_path / f"{VALID_SHA}.asm"

    def test_get_cached_path_with_family(self, processor: PEProcessor, tmp_path: Path):
        """Path is cache_dir/family/sha256.asm when family is set."""
        path = processor.get_cached_path(VALID_SHA, family="Ramnit")
        assert path == tmp_path / "Ramnit" / f"{VALID_SHA}.asm"

    def test_save_and_read_roundtrip(self, processor: PEProcessor):
        """Save opcodes and read them back — exact match."""
        opcodes = ["mov", "push", "xor", "call", "ret"]
        processor.save_asm_cache(opcodes, VALID_SHA)
        result = processor.read_cached_asm(VALID_SHA)
        assert result == opcodes

    def test_save_and_read_with_family(self, processor: PEProcessor):
        """Roundtrip with family subdirectory."""
        opcodes = ["add", "sub", "cmp"]
        processor.save_asm_cache(opcodes, VALID_SHA, family="Gatak")
        result = processor.read_cached_asm(VALID_SHA, family="Gatak")
        assert result == opcodes

    def test_read_cached_missing_file(self, processor: PEProcessor):
        """Returns empty list for non-existent file."""
        result = processor.read_cached_asm(VALID_SHA_2)
        assert result == []

    def test_read_cached_skips_blank_lines(self, processor: PEProcessor, tmp_path: Path):
        """Blank lines in the .asm file are filtered out."""
        asm = tmp_path / f"{VALID_SHA}.asm"
        asm.write_text("mov\n\npush\n  \nxor\n")
        result = processor.read_cached_asm(VALID_SHA)
        assert result == ["mov", "push", "xor"]

    def test_save_creates_parent_dirs(self, processor: PEProcessor, tmp_path: Path):
        """save_asm_cache creates missing parent directories."""
        processor.save_asm_cache(["nop"], VALID_SHA, family="Deep/Nested")
        path = tmp_path / "Deep/Nested" / f"{VALID_SHA}.asm"
        assert path.exists()

    def test_invalid_sha256_raises(self, processor: PEProcessor):
        """Non-hex or wrong-length sha256 raises ValueError."""
        with pytest.raises(ValueError, match="Invalid SHA-256"):
            processor.is_cached("not_a_valid_hash")

        with pytest.raises(ValueError, match="Invalid SHA-256"):
            processor.is_cached("abc123")  # too short

        with pytest.raises(ValueError, match="Invalid SHA-256"):
            processor.is_cached("g" * 64)  # non-hex char

        with pytest.raises(ValueError, match="Invalid SHA-256"):
            processor.is_cached("../../etc/passwd" + "a" * 49)  # path traversal


# ═══════════════════════════════════════════════════════════════════════════
# TestPEProcessorDisassembly
# ═══════════════════════════════════════════════════════════════════════════
class TestPEProcessorDisassembly:
    @patch("wintermute.data.etl.pe_utils.extract_opcodes_pe")
    def test_disassemble_pe_bytes_success(self, mock_extract: MagicMock):
        """Writes temp file, calls extract_opcodes_pe, returns opcodes."""
        mock_extract.return_value = ["mov", "push", "ret"]

        result = PEProcessor.disassemble_pe_bytes(b"MZ\x90\x00fake_pe_data")

        assert result == ["mov", "push", "ret"]
        mock_extract.assert_called_once()
        # The temp file path passed to extract should have been an .exe
        call_arg = mock_extract.call_args[0][0]
        assert call_arg.endswith(".exe")

    @patch("wintermute.data.etl.pe_utils.extract_opcodes_pe")
    def test_disassemble_pe_bytes_cleans_temp(self, mock_extract: MagicMock):
        """Temp file is deleted even after successful disassembly."""
        import os

        captured_path = {}

        def capture_path(path):
            captured_path["path"] = path
            return ["mov"]

        mock_extract.side_effect = capture_path

        PEProcessor.disassemble_pe_bytes(b"MZ\x90data")

        assert not os.path.exists(captured_path["path"])

    @patch("wintermute.data.etl.pe_utils.extract_opcodes_pe")
    def test_disassemble_pe_bytes_cleans_temp_on_error(self, mock_extract: MagicMock):
        """Temp file is deleted even when disassembly raises an exception."""
        import os

        captured_path = {}

        def capture_and_fail(path):
            captured_path["path"] = path
            raise RuntimeError("disasm crash")

        mock_extract.side_effect = capture_and_fail

        result = PEProcessor.disassemble_pe_bytes(b"MZ\x90bad")

        assert result == []
        assert not os.path.exists(captured_path["path"])


# ═══════════════════════════════════════════════════════════════════════════
# TestPEProcessorUnzip
# ═══════════════════════════════════════════════════════════════════════════
class TestPEProcessorUnzip:
    def _make_aes_zip(self, content: bytes, password: str = "infected") -> bytes:
        """Create an AES-encrypted ZIP in memory."""
        buf = io.BytesIO()
        with pyzipper.AESZipFile(
            buf, "w", compression=pyzipper.ZIP_DEFLATED, encryption=pyzipper.WZ_AES
        ) as zf:
            zf.setpassword(password.encode())
            zf.writestr("sample.exe", content)
        return buf.getvalue()

    def test_unzip_success(self):
        """Correctly extracts file from AES-encrypted ZIP."""
        original = b"MZ\x90\x00PE_BINARY_CONTENT"
        zip_bytes = self._make_aes_zip(original)

        result = PEProcessor.unzip_encrypted(zip_bytes)
        assert result == original

    def test_unzip_custom_password(self):
        """Works with a custom password."""
        original = b"custom_content"
        zip_bytes = self._make_aes_zip(original, password="secret123")

        result = PEProcessor.unzip_encrypted(zip_bytes, password="secret123")
        assert result == original

    def test_unzip_wrong_password(self):
        """Returns None when the password is wrong."""
        zip_bytes = self._make_aes_zip(b"data", password="correct")
        result = PEProcessor.unzip_encrypted(zip_bytes, password="wrong")
        assert result is None

    def test_unzip_invalid_data(self):
        """Returns None for non-ZIP data."""
        result = PEProcessor.unzip_encrypted(b"not a zip file at all")
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# TestPEProcessorDownload
# ═══════════════════════════════════════════════════════════════════════════
class TestPEProcessorDownload:
    @patch("wintermute.data.etl.pe_utils.requests")
    def test_download_get_success(self, mock_req: MagicMock):
        """Successful GET download returns content bytes."""
        mock_req.RequestException = requests.RequestException
        mock_resp = MagicMock()
        mock_resp.content = b"file_content"
        mock_resp.raise_for_status = MagicMock()
        mock_req.get.return_value = mock_resp

        result = PEProcessor.download_file("https://example.com/file.bin")
        assert result == b"file_content"
        mock_req.get.assert_called_once_with(
            "https://example.com/file.bin", headers=None, timeout=60
        )

    @patch("wintermute.data.etl.pe_utils.requests")
    def test_download_post_success(self, mock_req: MagicMock):
        """Successful POST download returns content bytes."""
        mock_req.RequestException = requests.RequestException
        mock_resp = MagicMock()
        mock_resp.content = b"post_content"
        mock_resp.raise_for_status = MagicMock()
        mock_req.post.return_value = mock_resp

        result = PEProcessor.download_file(
            "https://api.example.com",
            method="POST",
            data={"query": "get_file"},
            headers={"API-KEY": "abc"},
        )
        assert result == b"post_content"
        mock_req.post.assert_called_once()

    @patch("wintermute.data.etl.pe_utils.requests")
    def test_download_failure_returns_none(self, mock_req: MagicMock):
        """Network error returns None instead of raising."""
        mock_req.RequestException = requests.RequestException
        mock_req.get.side_effect = requests.RequestException("timeout")

        result = PEProcessor.download_file("https://example.com/fail")
        assert result is None

    @patch("wintermute.data.etl.pe_utils.requests")
    def test_download_http_error_returns_none(self, mock_req: MagicMock):
        """HTTP 4xx/5xx returns None."""
        mock_req.RequestException = requests.RequestException
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.RequestException("404")
        mock_req.get.return_value = mock_resp

        result = PEProcessor.download_file("https://example.com/missing")
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# TestPEProcessorProcessBinary
# ═══════════════════════════════════════════════════════════════════════════
class TestPEProcessorProcessBinary:
    _cached = "c" * 64
    _new = "d" * 64
    _short = "e" * 64
    _fam = "f" * 64
    _fail = "1" * 64

    @patch.object(PEProcessor, "disassemble_pe_bytes")
    def test_cache_hit_skips_disassembly(self, mock_disasm: MagicMock, processor: PEProcessor):
        """When cached, returns cached opcodes without disassembling."""
        opcodes = ["mov", "push", "xor", "call", "ret", "add"]
        processor.save_asm_cache(opcodes, self._cached)

        result = processor.process_pe_binary(b"pe_data", self._cached)

        assert result == opcodes
        mock_disasm.assert_not_called()

    @patch.object(PEProcessor, "disassemble_pe_bytes")
    def test_cache_miss_disassembles_and_caches(
        self, mock_disasm: MagicMock, processor: PEProcessor
    ):
        """When not cached, disassembles PE and stores the result."""
        mock_disasm.return_value = ["mov", "push", "xor", "call", "ret", "nop"]

        result = processor.process_pe_binary(b"pe_data", self._new)

        assert result == ["mov", "push", "xor", "call", "ret", "nop"]
        mock_disasm.assert_called_once_with(b"pe_data")
        # Verify it was cached
        assert processor.is_cached(self._new)

    @patch.object(PEProcessor, "disassemble_pe_bytes")
    def test_min_opcodes_filter_returns_none(self, mock_disasm: MagicMock, processor: PEProcessor):
        """Samples with fewer than min_opcodes return None and are not cached."""
        mock_disasm.return_value = ["mov", "push"]  # only 2, min is 5

        result = processor.process_pe_binary(b"pe_data", self._short)

        assert result is None
        assert not processor.is_cached(self._short)

    @patch.object(PEProcessor, "disassemble_pe_bytes")
    def test_process_with_family(self, mock_disasm: MagicMock, processor: PEProcessor):
        """Caches into family subdirectory when family is provided."""
        mock_disasm.return_value = ["mov"] * 10

        result = processor.process_pe_binary(b"pe_data", self._fam, family="Emotet")

        assert result is not None
        assert processor.is_cached(self._fam, family="Emotet")
        assert not processor.is_cached(self._fam)  # not in root

    @patch.object(PEProcessor, "disassemble_pe_bytes")
    def test_disassembly_failure_returns_none(self, mock_disasm: MagicMock, processor: PEProcessor):
        """If disassembly returns an empty list, result is None (below min_opcodes)."""
        mock_disasm.return_value = []

        result = processor.process_pe_binary(b"bad_pe", self._fail)
        assert result is None
