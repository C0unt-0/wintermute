"""test_etl.py — Tests for the ETL pipeline.

25 tests across 8 classes covering registry, sources, pipeline, and plugin pattern.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import wintermute.data.etl.sources  # noqa: F401  # trigger source auto-registration
from wintermute.data.etl.base import DataSource, ExtractResult, RawSample
from wintermute.data.etl.pipeline import Pipeline
from wintermute.data.etl.registry import SourceRegistry, register_source


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
# TestRawSample
# ═══════════════════════════════════════════════════════════════════════════
class TestRawSample:
    def test_defaults(self):
        s = RawSample(opcodes=["mov", "push"], label=0)
        assert s.opcodes == ["mov", "push"]
        assert s.label == 0
        assert s.family == ""
        assert s.source_id == ""
        assert s.metadata == {}

    def test_full_fields(self):
        s = RawSample(
            opcodes=["xor", "call"],
            label=1,
            family="Emotet",
            source_id="abc123",
            metadata={"size": 4096},
        )
        assert s.family == "Emotet"
        assert s.source_id == "abc123"
        assert s.metadata["size"] == 4096


# ═══════════════════════════════════════════════════════════════════════════
# TestExtractResult
# ═══════════════════════════════════════════════════════════════════════════
class TestExtractResult:
    def test_defaults(self):
        r = ExtractResult(source_name="test")
        assert r.samples_extracted == 0
        assert r.samples_skipped == 0
        assert r.errors == []

    def test_repr(self):
        r = ExtractResult(source_name="syn", samples_extracted=100, elapsed_seconds=1.5)
        s = repr(r)
        assert "syn" in s
        assert "100" in s
        assert "1.5" in s


# ═══════════════════════════════════════════════════════════════════════════
# TestSourceRegistry
# ═══════════════════════════════════════════════════════════════════════════
class TestSourceRegistry:
    def test_register_and_create(self):
        @register_source("_test_reg")
        class _TestSource(DataSource):
            name = "_test_reg"
            def extract(self):
                yield RawSample(opcodes=["nop"], label=0)

        src = SourceRegistry.create("_test_reg")
        assert isinstance(src, _TestSource)

    def test_available_includes_builtins(self):
        available = SourceRegistry.available()
        assert "synthetic" in available
        assert "pe_files" in available
        assert "ms_dataset" in available
        assert "malware_bazaar" in available
        assert "asm_directory" in available

    def test_duplicate_raises(self):
        @register_source("_test_dup")
        class _Dup1(DataSource):
            name = "_test_dup"
            def extract(self):
                return []

        with pytest.raises(ValueError, match="already registered"):
            @register_source("_test_dup")
            class _Dup2(DataSource):
                name = "_test_dup"
                def extract(self):
                    return []

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown source"):
            SourceRegistry.create("nonexistent_source_xyz")

    def test_create_with_config(self):
        @register_source("_test_cfg")
        class _CfgSource(DataSource):
            name = "_test_cfg"
            def extract(self):
                return []

        src = SourceRegistry.create("_test_cfg", config={"key": "val"})
        assert src.config == {"key": "val"}

    def test_get_returns_none_for_unknown(self):
        assert SourceRegistry.get("does_not_exist_xyz") is None


# ═══════════════════════════════════════════════════════════════════════════
# TestDataSourceBase
# ═══════════════════════════════════════════════════════════════════════════
class TestDataSourceBase:
    def test_run_wraps_lifecycle(self):
        """run() calls validate -> setup -> extract -> teardown."""
        calls = []

        class _LifecycleSource(DataSource):
            name = "_lifecycle"
            def validate_config(self):
                calls.append("validate")
                return []
            def setup(self):
                calls.append("setup")
            def extract(self):
                calls.append("extract")
                yield RawSample(opcodes=["mov"], label=0)
            def teardown(self):
                calls.append("teardown")

        src = _LifecycleSource()
        samples, result = src.run()
        assert calls == ["validate", "setup", "extract", "teardown"]
        assert len(samples) == 1

    def test_empty_opcodes_skipped(self):
        """Samples with empty opcodes are counted as skipped."""
        class _EmptySource(DataSource):
            name = "_empty"
            def extract(self):
                yield RawSample(opcodes=[], label=0)
                yield RawSample(opcodes=["mov"], label=0)

        src = _EmptySource()
        samples, result = src.run()
        assert len(samples) == 1
        assert result.samples_skipped == 1
        assert result.samples_extracted == 1

    def test_validation_failure_prevents_extraction(self):
        """If validate_config returns errors, extraction is skipped."""
        class _BadConfig(DataSource):
            name = "_badcfg"
            def validate_config(self):
                return ["missing required key"]
            def extract(self):
                raise AssertionError("Should not be called")

        src = _BadConfig()
        samples, result = src.run()
        assert len(samples) == 0
        assert result.errors == ["missing required key"]

    def test_config_helpers(self):
        class _Helpers(DataSource):
            name = "_helpers"
            def extract(self):
                return []

        src = _Helpers(config={"api_key": "abc", "limit": 50})
        assert src.get("api_key") == "abc"
        assert src.get("missing", 99) == 99
        assert src.require("limit") == 50
        with pytest.raises(ValueError, match="requires config key"):
            src.require("nonexistent")

    def test_extraction_error_caught(self):
        """Exceptions during extraction are caught, not propagated."""
        class _Crasher(DataSource):
            name = "_crasher"
            def extract(self):
                yield RawSample(opcodes=["mov"], label=0)
                raise RuntimeError("boom")

        src = _Crasher()
        samples, result = src.run()
        # Samples yielded before the exception are preserved
        assert len(samples) == 1
        assert result.samples_failed == 1
        assert "boom" in result.errors[0]


# ═══════════════════════════════════════════════════════════════════════════
# TestSyntheticSource
# ═══════════════════════════════════════════════════════════════════════════
class TestSyntheticSource:
    def test_correct_count(self):
        src = SourceRegistry.create("synthetic", config={"n_samples": 20, "seed": 42})
        samples, result = src.run()
        assert result.samples_extracted == 20

    def test_binary_labels(self):
        src = SourceRegistry.create("synthetic", config={"n_samples": 10, "seed": 42})
        samples, _ = src.run()
        labels = {s.label for s in samples}
        assert labels == {0, 1}

    def test_deterministic(self):
        cfg = {"n_samples": 10, "seed": 123}
        src1 = SourceRegistry.create("synthetic", config=dict(cfg))
        src2 = SourceRegistry.create("synthetic", config=dict(cfg))
        s1, _ = src1.run()
        s2, _ = src2.run()
        for a, b in zip(s1, s2):
            assert a.opcodes == b.opcodes
            assert a.label == b.label

    def test_family_names(self):
        src = SourceRegistry.create("synthetic", config={"n_samples": 4, "seed": 42})
        samples, result = src.run()
        families = {s.family for s in samples}
        assert families == {"safe", "malicious"}


# ═══════════════════════════════════════════════════════════════════════════
# TestAsmDirectorySource
# ═══════════════════════════════════════════════════════════════════════════
class TestAsmDirectorySource:
    def test_bazaar_format(self, tmp_path):
        """Read Bazaar-format .asm files from family subdirectories."""
        fam_dir = tmp_path / "Emotet"
        fam_dir.mkdir()
        (fam_dir / "sample1.asm").write_text(
            "\n".join(["mov", "push", "xor", "call", "ret",
                       "add", "sub", "cmp", "jmp", "nop", "test"])
        )

        src = SourceRegistry.create("asm_directory", config={
            "data_dir": str(tmp_path),
            "min_opcodes": 5,
        })
        samples, result = src.run()
        assert len(samples) == 1
        assert samples[0].family == "Emotet"
        assert len(samples[0].opcodes) == 11

    def test_short_file_skipped(self, tmp_path):
        """Files with fewer opcodes than min_opcodes are skipped."""
        fam_dir = tmp_path / "Short"
        fam_dir.mkdir()
        (fam_dir / "tiny.asm").write_text("mov\npush\n")

        src = SourceRegistry.create("asm_directory", config={
            "data_dir": str(tmp_path),
            "min_opcodes": 10,
        })
        samples, result = src.run()
        assert len(samples) == 0

    def test_missing_dir_validation(self, tmp_path):
        src = SourceRegistry.create("asm_directory", config={
            "data_dir": str(tmp_path / "nonexistent"),
        })
        samples, result = src.run()
        assert len(samples) == 0
        assert len(result.errors) > 0


# ═══════════════════════════════════════════════════════════════════════════
# TestPipeline
# ═══════════════════════════════════════════════════════════════════════════
class TestPipeline:
    def _make_config(self, out_dir, sources_cfg):
        return {
            "pipeline": {
                "out_dir": str(out_dir),
                "max_seq_length": 64,
                "shuffle": True,
                "seed": 42,
            },
            "sources": sources_cfg,
        }

    def test_full_etl_flow(self, tmp_path):
        """End-to-end: synthetic -> transform -> save."""
        cfg = self._make_config(tmp_path, {
            "synthetic": {"enabled": True, "n_samples": 20, "seed": 42},
        })
        pipe = Pipeline(config=cfg)
        result = pipe.run()

        assert result.total_samples == 20
        assert result.num_classes == 2
        assert result.vocab_size > 5  # 5 specials + opcodes
        assert result.x_shape == (20, 64)
        assert result.y_shape == (20,)

        # Check files written
        assert (tmp_path / "x_data.npy").exists()
        assert (tmp_path / "y_data.npy").exists()
        assert (tmp_path / "vocab.json").exists()
        assert (tmp_path / "families.json").exists()
        assert (tmp_path / "etl_manifest.json").exists()

    def test_vocab_has_special_tokens(self, tmp_path):
        """Vocabulary starts with 5 special tokens."""
        cfg = self._make_config(tmp_path, {
            "synthetic": {"enabled": True, "n_samples": 10, "seed": 42},
        })
        pipe = Pipeline(config=cfg)
        pipe.run()

        with open(tmp_path / "vocab.json") as f:
            stoi = json.load(f)

        assert stoi["<PAD>"] == 0
        assert stoi["<UNK>"] == 1
        assert stoi["<CLS>"] == 2
        assert stoi["<SEP>"] == 3
        assert stoi["<MASK>"] == 4

    def test_manifest_provenance(self, tmp_path):
        """etl_manifest.json records per-source provenance."""
        cfg = self._make_config(tmp_path, {
            "synthetic": {"enabled": True, "n_samples": 10, "seed": 42},
        })
        pipe = Pipeline(config=cfg)
        pipe.run()

        with open(tmp_path / "etl_manifest.json") as f:
            manifest = json.load(f)

        assert "synthetic" in manifest["sources"]
        assert manifest["sources"]["synthetic"]["samples"] == 10
        assert manifest["x_shape"] == [10, 64]
        assert manifest["pipeline_config"]["max_seq_length"] == 64

    def test_disabled_source_skipped(self, tmp_path):
        """Disabled sources are not instantiated."""
        cfg = self._make_config(tmp_path, {
            "synthetic": {"enabled": False, "n_samples": 10},
        })
        pipe = Pipeline(config=cfg)
        result = pipe.run()
        assert result.total_samples == 0
        assert len(result.extract_results) == 0

    def test_unknown_source_warned(self, tmp_path):
        """Unknown source names in config produce warnings, not crashes."""
        cfg = self._make_config(tmp_path, {
            "nonexistent_source_xyz": {"enabled": True},
            "synthetic": {"enabled": True, "n_samples": 10, "seed": 42},
        })
        pipe = Pipeline(config=cfg)
        result = pipe.run()
        # Pipeline continues past the unknown source
        assert result.total_samples == 10

    def test_multiple_source_merge(self, tmp_path):
        """Multiple sources merge correctly."""
        # Register a second inline source
        @register_source("_test_extra")
        class _ExtraSource(DataSource):
            name = "_test_extra"
            def extract(self):
                for i in range(5):
                    yield RawSample(opcodes=["mov", "push", "ret"] * 5, label=2, family="extra")

        cfg = self._make_config(tmp_path, {
            "synthetic": {"enabled": True, "n_samples": 10, "seed": 42},
            "_test_extra": {"enabled": True},
        })
        pipe = Pipeline(config=cfg)
        result = pipe.run()
        assert result.total_samples == 15

    def test_config_file_loading(self, tmp_path):
        """Pipeline can load config from a YAML file."""
        config_file = tmp_path / "test_sources.yaml"
        config_file.write_text(
            "pipeline:\n"
            "  out_dir: '" + str(tmp_path / "out") + "'\n"
            "  max_seq_length: 32\n"
            "  seed: 99\n"
            "sources:\n"
            "  synthetic:\n"
            "    enabled: true\n"
            "    n_samples: 8\n"
            "    seed: 99\n"
        )
        pipe = Pipeline(config_path=config_file)
        result = pipe.run()
        assert result.total_samples == 8
        assert (tmp_path / "out" / "x_data.npy").exists()

    def test_dry_run(self, tmp_path):
        """Dry run validates without writing files."""
        cfg = self._make_config(tmp_path, {
            "synthetic": {"enabled": True, "n_samples": 10, "seed": 42},
        })
        pipe = Pipeline(config=cfg)
        result = pipe.run(dry_run=True)
        assert result.total_samples == 10
        assert not (tmp_path / "x_data.npy").exists()


# ═══════════════════════════════════════════════════════════════════════════
# TestPluginPattern
# ═══════════════════════════════════════════════════════════════════════════
class TestPluginPattern:
    def test_custom_source_end_to_end(self, tmp_path):
        """A custom source works end-to-end with zero framework changes."""
        @register_source("_test_custom_e2e")
        class _CustomE2E(DataSource):
            name = "_test_custom_e2e"
            def extract(self):
                for i in range(3):
                    yield RawSample(
                        opcodes=["mov", "xor", "call", "ret", "push"] * 3,
                        label=i % 2,
                        family="custom",
                        source_id=f"custom_{i}",
                    )

        cfg = {
            "pipeline": {
                "out_dir": str(tmp_path),
                "max_seq_length": 32,
                "seed": 42,
            },
            "sources": {
                "_test_custom_e2e": {"enabled": True},
            },
        }
        pipe = Pipeline(config=cfg)
        result = pipe.run()

        assert result.total_samples == 3
        assert result.num_classes == 2

        # Verify output files
        x = np.load(tmp_path / "x_data.npy")
        y = np.load(tmp_path / "y_data.npy")
        assert x.shape == (3, 32)
        assert y.shape == (3,)
        assert x.dtype == np.int32
        assert y.dtype == np.int32

        with open(tmp_path / "vocab.json") as f:
            stoi = json.load(f)
        assert stoi["<PAD>"] == 0
        assert "mov" in stoi
        assert "xor" in stoi
