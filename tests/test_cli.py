"""
test_cli.py — Tests for wintermute.cli
"""

from typer.testing import CliRunner

from wintermute.cli import app

runner = CliRunner()


def test_help():
    """CLI --help should run without error."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "wintermute" in result.output.lower() or "Wintermute" in result.output


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
