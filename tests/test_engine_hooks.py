"""test_engine_hooks.py — Tests for transport-agnostic engine hooks."""

from wintermute.engine.hooks import AdversarialHook, PipelineHook, TrainingHook


class TestTrainingHookCallback:
    """TrainingHook emits correct event dicts via callback."""

    def test_on_epoch_emits_epoch_complete(self):
        received = []
        hook = TrainingHook(callback=received.append)
        hook.on_epoch(
            epoch=5, phase="A", loss=0.32, train_acc=0.88, val_acc=0.84, f1=0.86, elapsed=12.5
        )
        assert len(received) == 1
        d = received[0]
        assert d["type"] == "epoch_complete"
        assert d["epoch"] == 5
        assert d["phase"] == "A"
        assert d["loss"] == 0.32
        assert d["train_acc"] == 0.88
        assert d["val_acc"] == 0.84
        assert d["f1"] == 0.86
        assert d["elapsed"] == 12.5

    def test_on_log_emits_activity_log(self):
        received = []
        hook = TrainingHook(callback=received.append)
        hook.on_log("Training started", level="info")
        assert len(received) == 1
        d = received[0]
        assert d["type"] == "activity_log"
        assert d["text"] == "Training started"
        assert d["level"] == "info"

    def test_on_log_default_level(self):
        received = []
        hook = TrainingHook(callback=received.append)
        hook.on_log("Some message")
        assert received[0]["level"] == "info"

    def test_on_log_error_level(self):
        received = []
        hook = TrainingHook(callback=received.append)
        hook.on_log("Something broke", level="error")
        assert received[0]["level"] == "error"

    def test_multiple_events_accumulate(self):
        received = []
        hook = TrainingHook(callback=received.append)
        hook.on_epoch(1, "A", 0.5, 0.8, 0.75, 0.77, 3.2)
        hook.on_epoch(2, "B", 0.4, 0.85, 0.82, 0.83, 6.1)
        hook.on_log("done")
        assert len(received) == 3
        assert received[0]["type"] == "epoch_complete"
        assert received[1]["type"] == "epoch_complete"
        assert received[2]["type"] == "activity_log"


class TestTrainingHookNoOp:
    """TrainingHook with callback=None is a safe no-op."""

    def test_on_epoch_no_callback(self):
        hook = TrainingHook()
        # Should not raise
        hook.on_epoch(
            epoch=1, phase="A", loss=0.5, train_acc=0.8, val_acc=0.7, f1=0.75, elapsed=1.0
        )

    def test_on_log_no_callback(self):
        hook = TrainingHook()
        hook.on_log("no-op test")

    def test_explicit_none_callback(self):
        hook = TrainingHook(callback=None)
        hook.on_epoch(
            epoch=1, phase="A", loss=0.5, train_acc=0.8, val_acc=0.7, f1=0.75, elapsed=1.0
        )
        hook.on_log("also no-op")


class TestTrainingHookCancelReset:
    def test_initial_state_not_cancelled(self):
        hook = TrainingHook()
        assert hook.cancelled is False

    def test_cancel_sets_flag(self):
        hook = TrainingHook()
        hook.cancel()
        assert hook.cancelled is True

    def test_reset_clears_flag(self):
        hook = TrainingHook()
        hook.cancel()
        assert hook.cancelled is True
        hook.reset()
        assert hook.cancelled is False

    def test_cancel_reset_cycle(self):
        hook = TrainingHook()
        hook.cancel()
        hook.reset()
        hook.cancel()
        assert hook.cancelled is True


class TestAdversarialHookCallback:
    """AdversarialHook emits correct event dicts via callback."""

    def test_on_episode_step_emits_event(self):
        received = []
        hook = AdversarialHook(callback=received.append)
        hook.on_episode_step(step=1, action="nop_insert", pos=5, conf=0.8, ok=True)
        assert len(received) == 1
        d = received[0]
        assert d["type"] == "adversarial_episode_step"
        assert d["step"] == 1
        assert d["action"] == "nop_insert"
        assert d["position"] == 5
        assert d["confidence"] == 0.8
        assert d["valid"] is True

    def test_on_cycle_end_emits_event(self):
        received = []
        hook = AdversarialHook(callback=received.append)
        metrics = {"evasion_rate": 0.3, "loss": 0.5}
        hook.on_cycle_end(cycle=3, metrics=metrics)
        assert len(received) == 1
        d = received[0]
        assert d["type"] == "adversarial_cycle_end"
        assert d["cycle"] == 3
        assert d["metrics"] == {"evasion_rate": 0.3, "loss": 0.5}

    def test_on_vault_sample_emits_event(self):
        received = []
        hook = AdversarialHook(callback=received.append)
        sample = {"id": "v001", "family": "Ramnit", "confidence": 0.72}
        hook.on_vault_sample(sample=sample)
        assert len(received) == 1
        d = received[0]
        assert d["type"] == "vault_sample_added"
        assert d["sample"] == sample

    def test_on_log_emits_activity_log(self):
        received = []
        hook = AdversarialHook(callback=received.append)
        hook.on_log("Cycle complete", level="info")
        assert len(received) == 1
        d = received[0]
        assert d["type"] == "activity_log"
        assert d["text"] == "Cycle complete"
        assert d["level"] == "info"

    def test_on_log_default_level(self):
        received = []
        hook = AdversarialHook(callback=received.append)
        hook.on_log("message")
        assert received[0]["level"] == "info"

    def test_multiple_events(self):
        received = []
        hook = AdversarialHook(callback=received.append)
        hook.on_episode_step(1, "nop_insert", 0, 0.9, True)
        hook.on_cycle_end(1, {"loss": 0.1})
        hook.on_vault_sample({"id": "v002"})
        hook.on_log("done")
        assert len(received) == 4
        assert received[0]["type"] == "adversarial_episode_step"
        assert received[1]["type"] == "adversarial_cycle_end"
        assert received[2]["type"] == "vault_sample_added"
        assert received[3]["type"] == "activity_log"


class TestAdversarialHookNoOp:
    """AdversarialHook with callback=None is a safe no-op."""

    def test_on_episode_step_no_callback(self):
        hook = AdversarialHook()
        hook.on_episode_step(step=1, action="nop_insert", pos=5, conf=0.8, ok=True)

    def test_on_cycle_end_no_callback(self):
        hook = AdversarialHook()
        hook.on_cycle_end(cycle=1, metrics={})

    def test_on_vault_sample_no_callback(self):
        hook = AdversarialHook()
        hook.on_vault_sample(sample={"id": "v001"})

    def test_on_log_no_callback(self):
        hook = AdversarialHook()
        hook.on_log("no-op")

    def test_explicit_none_callback(self):
        hook = AdversarialHook(callback=None)
        hook.on_episode_step(1, "x", 0, 0.5, False)
        hook.on_cycle_end(1, {})
        hook.on_vault_sample({})
        hook.on_log("still no-op")


class TestAdversarialHookCancelReset:
    def test_initial_state_not_cancelled(self):
        hook = AdversarialHook()
        assert hook.cancelled is False

    def test_cancel_sets_flag(self):
        hook = AdversarialHook()
        hook.cancel()
        assert hook.cancelled is True

    def test_reset_clears_flag(self):
        hook = AdversarialHook()
        hook.cancel()
        hook.reset()
        assert hook.cancelled is False


class TestPipelineHookCallback:
    """PipelineHook emits correct event dicts via callback."""

    def test_on_progress_emits_event(self):
        received = []
        hook = PipelineHook(callback=received.append)
        hook.on_progress(operation="build", progress=0.5, message="Processing file 10/20")
        assert len(received) == 1
        d = received[0]
        assert d["type"] == "pipeline_progress"
        assert d["operation"] == "build"
        assert d["progress"] == 0.5
        assert d["message"] == "Processing file 10/20"

    def test_on_log_emits_activity_log(self):
        received = []
        hook = PipelineHook(callback=received.append)
        hook.on_log("Pipeline started", level="info")
        assert len(received) == 1
        d = received[0]
        assert d["type"] == "activity_log"
        assert d["text"] == "Pipeline started"
        assert d["level"] == "info"

    def test_on_log_default_level(self):
        received = []
        hook = PipelineHook(callback=received.append)
        hook.on_log("message")
        assert received[0]["level"] == "info"

    def test_on_progress_zero_and_one(self):
        received = []
        hook = PipelineHook(callback=received.append)
        hook.on_progress("synthetic", 0.0, "Starting")
        hook.on_progress("synthetic", 1.0, "Done")
        assert len(received) == 2
        assert received[0]["progress"] == 0.0
        assert received[1]["progress"] == 1.0

    def test_multiple_events(self):
        received = []
        hook = PipelineHook(callback=received.append)
        hook.on_progress("build", 0.25, "step 1")
        hook.on_log("logging something")
        hook.on_progress("build", 0.75, "step 2")
        assert len(received) == 3
        assert received[0]["type"] == "pipeline_progress"
        assert received[1]["type"] == "activity_log"
        assert received[2]["type"] == "pipeline_progress"


class TestPipelineHookNoOp:
    """PipelineHook with callback=None is a safe no-op."""

    def test_on_progress_no_callback(self):
        hook = PipelineHook()
        hook.on_progress(operation="build", progress=0.5, message="test")

    def test_on_log_no_callback(self):
        hook = PipelineHook()
        hook.on_log("no-op")

    def test_explicit_none_callback(self):
        hook = PipelineHook(callback=None)
        hook.on_progress("build", 0.5, "test")
        hook.on_log("still no-op")


class TestPipelineHookCancelReset:
    def test_initial_state_not_cancelled(self):
        hook = PipelineHook()
        assert hook.cancelled is False

    def test_cancel_sets_flag(self):
        hook = PipelineHook()
        hook.cancel()
        assert hook.cancelled is True

    def test_reset_clears_flag(self):
        hook = PipelineHook()
        hook.cancel()
        hook.reset()
        assert hook.cancelled is False


class TestHooksAreDataclasses:
    """Verify that hooks are dataclasses with the expected interface."""

    def test_training_hook_is_dataclass(self):
        from dataclasses import fields as dc_fields

        field_names = {f.name for f in dc_fields(TrainingHook)}
        assert "callback" in field_names
        assert "cancelled" in field_names

    def test_adversarial_hook_is_dataclass(self):
        from dataclasses import fields as dc_fields

        field_names = {f.name for f in dc_fields(AdversarialHook)}
        assert "callback" in field_names
        assert "cancelled" in field_names

    def test_pipeline_hook_is_dataclass(self):
        from dataclasses import fields as dc_fields

        field_names = {f.name for f in dc_fields(PipelineHook)}
        assert "callback" in field_names
        assert "cancelled" in field_names

    def test_cancelled_not_in_init(self):
        """cancelled should be a non-init field defaulting to False."""
        from dataclasses import fields as dc_fields

        for cls in (TrainingHook, AdversarialHook, PipelineHook):
            for f in dc_fields(cls):
                if f.name == "cancelled":
                    assert f.init is False, f"{cls.__name__}.cancelled should have init=False"
