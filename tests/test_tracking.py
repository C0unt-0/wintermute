"""
test_tracking.py — Tests for wintermute.engine.tracking
"""

from wintermute.engine.tracking import MLflowTracker


class TestMLflowTracker:
    def test_disabled_tracker(self):
        """Disabled tracker should be a no-op."""
        tracker = MLflowTracker(enabled=False)
        assert not tracker.enabled
        assert not tracker.active

        # These should all be no-ops (no exceptions)
        tracker.start_run(run_name="test")
        tracker.log_params({"lr": 0.001})
        tracker.log_metrics({"loss": 0.5}, step=1)
        tracker.log_artifact("/tmp/test.txt")
        tracker.end_run()

    def test_flatten_dict(self):
        """Test nested dict flattening."""
        nested = {
            "model": {
                "dims": 128,
                "layers": 4,
            },
            "training": {
                "lr": 0.001,
            },
        }
        flat = MLflowTracker._flatten_dict(nested)
        assert flat == {
            "model.dims": "128",
            "model.layers": "4",
            "training.lr": "0.001",
        }

    def test_flatten_dict_empty(self):
        flat = MLflowTracker._flatten_dict({})
        assert flat == {}

    def test_flatten_dict_flat_input(self):
        flat = MLflowTracker._flatten_dict({"a": 1, "b": "two"})
        assert flat == {"a": "1", "b": "two"}

    def test_save_metrics_json(self, tmp_path):
        """save_metrics_json should always work regardless of MLflow state."""
        tracker = MLflowTracker(enabled=False)
        out = tmp_path / "metrics.json"
        tracker.save_metrics_json(
            {"accuracy": 0.95, "loss": 0.12},
            filepath=out,
        )
        assert out.exists()

        import json
        with open(out) as f:
            data = json.load(f)
        assert data["accuracy"] == 0.95
        assert data["loss"] == 0.12
