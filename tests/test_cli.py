"""
test_cli.py — Tests for wintermute.cli

Covers:
  1. wintermute scan  — happy path (.asm file, mocked WintermuteMalwareDetector.load)
  2. wintermute scan  — file-not-found exits with code 1
  3. wintermute train — invokes JointTrainer.train() (JointTrainer mocked)
  4. wintermute --help — smoke test, exits 0
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn
import pytest
from typer.testing import CliRunner

from wintermute.cli import app
from wintermute.models.fusion import DetectorConfig, WintermuteMalwareDetector

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tiny_detector() -> WintermuteMalwareDetector:
    """Return a minimal WintermuteMalwareDetector with a 32-token vocab."""
    cfg = DetectorConfig(
        vocab_size=32,
        dims=16,
        num_heads=2,
        num_layers=1,
        mlp_dims=32,
        dropout=0.0,
        max_seq_length=64,
        gat_layers=1,
        gat_heads=2,
        num_fusion_heads=2,
        num_classes=2,
    )
    return WintermuteMalwareDetector(cfg)


def _write_vocab(path: Path) -> dict:
    """Write a minimal vocab.json to *path* and return the dict."""
    stoi = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2, "<SEP>": 3, "<MASK>": 4,
            "push": 5, "pop": 6, "mov": 7, "ret": 8, "nop": 9}
    path.write_text(json.dumps(stoi))
    return stoi


# ---------------------------------------------------------------------------
# Test 1: wintermute --help  (smoke test)
# ---------------------------------------------------------------------------

def test_help():
    """CLI --help should exit 0 and mention 'wintermute'."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "wintermute" in result.output.lower() or "Wintermute" in result.output


# ---------------------------------------------------------------------------
# Test 2: wintermute scan — file not found exits with code 1
# ---------------------------------------------------------------------------

def test_scan_file_not_found(tmp_path):
    """scan on a non-existent file must exit with code 1."""
    vocab_path = tmp_path / "vocab.json"
    _write_vocab(vocab_path)

    result = runner.invoke(app, [
        "scan", str(tmp_path / "nonexistent.asm"),
        "--vocab", str(vocab_path),
        "--model", "dummy.safetensors",
        "--manifest", "dummy_manifest.json",
    ])
    assert result.exit_code == 1
    assert "[ERROR]" in result.output or "[ERROR]" in (result.stderr or "")


# ---------------------------------------------------------------------------
# Test 3: wintermute scan — happy path with .asm file
# ---------------------------------------------------------------------------

def test_scan_asm_happy_path(tmp_path):
    """scan should print a verdict when given a valid .asm file."""
    # Create a minimal .asm file
    asm_path = tmp_path / "sample.asm"
    asm_path.write_text("push\npop\nmov\nret\n")

    vocab_path = tmp_path / "vocab.json"
    stoi = _write_vocab(vocab_path)

    tiny_model = _make_tiny_detector()

    # Patch WintermuteMalwareDetector.load so no real weights file is needed.
    # The scan command imports from wintermute.models.fusion, so we patch there.
    with patch("wintermute.models.fusion.WintermuteMalwareDetector.load",
               return_value=tiny_model) as mock_load:
        result = runner.invoke(app, [
            "scan", str(asm_path),
            "--vocab", str(vocab_path),
            "--model", "dummy.safetensors",
            "--manifest", "dummy_manifest.json",
        ])

    assert result.exit_code == 0, f"exit_code={result.exit_code}\n{result.output}"
    # The output must contain the separator line produced by the verdict block
    assert "=" * 10 in result.output
    # The model was loaded exactly once
    mock_load.assert_called_once()


# ---------------------------------------------------------------------------
# Test 4: wintermute train — JointTrainer.train() is called
# ---------------------------------------------------------------------------

def test_train_invokes_joint_trainer(tmp_path):
    """train should instantiate JointTrainer and call .train()."""
    # Prepare a minimal data directory with vocab.json, x_data.npy, y_data.npy
    import numpy as np

    vocab_path = tmp_path / "vocab.json"
    _write_vocab(vocab_path)

    x = np.zeros((4, 64), dtype=np.int32)
    y = np.array([0, 1, 0, 1], dtype=np.int32)
    np.save(tmp_path / "x_data.npy", x)
    np.save(tmp_path / "y_data.npy", y)

    mock_trainer_instance = MagicMock()
    mock_trainer_instance.train.return_value = 0.75

    # JointTrainer is imported inside the train() function body, so we must
    # patch it at the module where it is defined and used.
    with patch("wintermute.engine.joint_trainer.JointTrainer",
               return_value=mock_trainer_instance) as mock_cls:
        result = runner.invoke(app, [
            "train",
            "--data-dir", str(tmp_path),
            "--epochs-phase-a", "1",
            "--epochs-phase-b", "1",
            "--batch-size", "2",
            "--num-classes", "2",
        ])

    assert result.exit_code == 0, f"exit_code={result.exit_code}\n{result.output}"
    mock_cls.assert_called_once()
    mock_trainer_instance.train.assert_called_once()


# ---------------------------------------------------------------------------
# Smoke tests preserved from original test_cli.py
# ---------------------------------------------------------------------------

def test_data_help():
    """'wintermute data --help' should list sub-commands."""
    result = runner.invoke(app, ["data", "--help"])
    assert result.exit_code == 0
    assert "synthetic" in result.output


def test_train_help():
    """'wintermute train --help' should list options."""
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "epochs" in result.output.lower()


def test_scan_help():
    """'wintermute scan --help' should list options."""
    result = runner.invoke(app, ["scan", "--help"])
    assert result.exit_code == 0
    assert "target" in result.output.lower()


def test_data_synthetic(tmp_path):
    """'wintermute data synthetic' should produce output files."""
    result = runner.invoke(app, [
        "data", "synthetic",
        "--n-samples", "10",
        "--max-seq-length", "32",
        "--out-dir", str(tmp_path),
        "--seed", "42",
    ])
    assert result.exit_code == 0
    assert (tmp_path / "x_data.npy").exists()
    assert (tmp_path / "y_data.npy").exists()
    assert (tmp_path / "vocab.json").exists()
