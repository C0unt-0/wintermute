"""
tracking.py — Wintermute MLflow Experiment Tracking

Provides an MLflowTracker that wraps the MLflow API for:
    - Run lifecycle management (start / end)
    - Hyperparameter logging
    - Per-epoch metric logging (loss, accuracy, F1)
    - Model artifact logging (safetensors + vocab)
    - Graceful no-op when MLflow is not installed
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MLflowTracker:
    """
    Thin wrapper around the MLflow Tracking API.

    Falls back to a no-op if ``mlflow`` is not installed, so the trainer
    works identically with or without the ``[mlops]`` extra.

    Usage
    -----
    >>> tracker = MLflowTracker(experiment="wintermute", enabled=True)
    >>> tracker.start_run(run_name="binary-v2")
    >>> tracker.log_params({"epochs": 20, "lr": 3e-4})
    >>> tracker.log_metrics({"loss": 0.42, "val_acc": 0.91}, step=5)
    >>> tracker.log_artifact("malware_model.safetensors")
    >>> tracker.end_run()
    """

    def __init__(
        self,
        experiment: str = "wintermute",
        tracking_uri: str = "mlruns",
        enabled: bool = True,
    ):
        self.enabled = enabled
        self._mlflow = None
        self._run = None

        if not enabled:
            return

        try:
            import mlflow

            self._mlflow = mlflow
            # Use local file-based tracking by default
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment)
        except ImportError:
            print("  [WARN] mlflow not installed — tracking disabled.")
            print("         Install with: pip install wintermute[mlops]")
            self.enabled = False

    @property
    def active(self) -> bool:
        """True if MLflow is available and a run is active."""
        return self.enabled and self._run is not None

    def start_run(self, run_name: str | None = None) -> None:
        """Start a new MLflow run."""
        if not self.enabled:
            return
        self._run = self._mlflow.start_run(run_name=run_name)
        print(f"  📊 MLflow run started: {self._run.info.run_id[:8]}…")

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current MLflow run."""
        if not self.active:
            return
        self._mlflow.end_run(status=status)
        print(f"  📊 MLflow run ended ({status})")
        self._run = None

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters to the current run."""
        if not self.active:
            return
        # Flatten nested dicts (e.g. model.dims → "model.dims")
        flat = self._flatten_dict(params)
        self._mlflow.log_params(flat)

    def log_metrics(
        self, metrics: dict[str, float], step: int | None = None
    ) -> None:
        """Log metrics at a given training step."""
        if not self.active:
            return
        self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, filepath: str | Path) -> None:
        """Log a file artifact (model weights, vocab, etc)."""
        if not self.active:
            return
        self._mlflow.log_artifact(str(filepath))

    def log_model_summary(
        self, n_params: int, precision: str, model_type: str = "MalwareClassifier"
    ) -> None:
        """Log model architecture summary as tags."""
        if not self.active:
            return
        self._mlflow.set_tags({
            "model_type": model_type,
            "n_params": str(n_params),
            "precision": precision,
        })

    def save_metrics_json(
        self, metrics: dict[str, Any], filepath: str | Path = "metrics.json"
    ) -> None:
        """
        Save metrics to a JSON file (for DVC metrics tracking).

        This is always written regardless of MLflow state, so DVC can pick it up.
        """
        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=2)

    # ------------------------------------------------------------------
    @staticmethod
    def _flatten_dict(
        d: dict, parent_key: str = "", sep: str = "."
    ) -> dict[str, str]:
        """Flatten a nested dict into dot-separated keys."""
        items: list[tuple[str, str]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(
                    MLflowTracker._flatten_dict(v, new_key, sep).items()
                )
            else:
                items.append((new_key, str(v)))
        return dict(items)
