"""test_engine_events.py — Tests for transport-agnostic engine events."""

import json

from wintermute.engine.events import (
    ActivityLogEntry,
    AdversarialCycleEnd,
    AdversarialEpisodeStep,
    EpochComplete,
    EvaluationComplete,
    PipelineProgress,
    ScanProgress,
    VaultSampleAdded,
)


class TestEpochComplete:
    def test_to_dict_type(self):
        e = EpochComplete(
            epoch=5, phase="A", loss=0.32, train_acc=0.88, val_acc=0.84, f1=0.86, elapsed=12.5
        )
        d = e.to_dict()
        assert d["type"] == "epoch_complete"

    def test_to_dict_all_fields(self):
        e = EpochComplete(
            epoch=5, phase="A", loss=0.32, train_acc=0.88, val_acc=0.84, f1=0.86, elapsed=12.5
        )
        d = e.to_dict()
        assert d["epoch"] == 5
        assert d["phase"] == "A"
        assert d["loss"] == 0.32
        assert d["train_acc"] == 0.88
        assert d["val_acc"] == 0.84
        assert d["f1"] == 0.86
        assert d["elapsed"] == 12.5

    def test_json_serializable(self):
        e = EpochComplete(
            epoch=1, phase="B", loss=0.1, train_acc=0.95, val_acc=0.93, f1=0.94, elapsed=3.0
        )
        serialized = json.dumps(e.to_dict())
        assert isinstance(serialized, str)


class TestScanProgress:
    def test_to_dict_type(self):
        s = ScanProgress(phase="disassemble", data={"n_ops": 100})
        d = s.to_dict()
        assert d["type"] == "scan_progress"

    def test_to_dict_all_fields(self):
        s = ScanProgress(phase="disassemble", data={"n_ops": 100})
        d = s.to_dict()
        assert d["phase"] == "disassemble"
        assert d["data"] == {"n_ops": 100}

    def test_default_data(self):
        s = ScanProgress(phase="tokenize")
        d = s.to_dict()
        assert d["data"] == {}

    def test_json_serializable(self):
        s = ScanProgress(phase="inference")
        serialized = json.dumps(s.to_dict())
        assert isinstance(serialized, str)


class TestAdversarialCycleEnd:
    def test_to_dict_type(self):
        a = AdversarialCycleEnd(cycle=3, metrics={"evasion_rate": 0.3})
        d = a.to_dict()
        assert d["type"] == "adversarial_cycle_end"

    def test_to_dict_all_fields(self):
        a = AdversarialCycleEnd(cycle=3, metrics={"evasion_rate": 0.3})
        d = a.to_dict()
        assert d["cycle"] == 3
        assert d["metrics"] == {"evasion_rate": 0.3}

    def test_default_metrics(self):
        a = AdversarialCycleEnd(cycle=1)
        d = a.to_dict()
        assert d["metrics"] == {}

    def test_json_serializable(self):
        a = AdversarialCycleEnd(cycle=1, metrics={"loss": 0.5})
        serialized = json.dumps(a.to_dict())
        assert isinstance(serialized, str)


class TestAdversarialEpisodeStep:
    def test_to_dict_type(self):
        e = AdversarialEpisodeStep(
            step=1, action="nop_insert", position=5, confidence=0.8, valid=True
        )
        d = e.to_dict()
        assert d["type"] == "adversarial_episode_step"

    def test_to_dict_all_fields(self):
        e = AdversarialEpisodeStep(
            step=2, action="reg_swap", position=10, confidence=0.65, valid=False
        )
        d = e.to_dict()
        assert d["step"] == 2
        assert d["action"] == "reg_swap"
        assert d["position"] == 10
        assert d["confidence"] == 0.65
        assert d["valid"] is False

    def test_json_serializable(self):
        e = AdversarialEpisodeStep(
            step=1, action="nop_insert", position=0, confidence=0.9, valid=True
        )
        serialized = json.dumps(e.to_dict())
        assert isinstance(serialized, str)


class TestActivityLogEntry:
    def test_to_dict_type(self):
        e = ActivityLogEntry(text="Training started")
        d = e.to_dict()
        assert d["type"] == "activity_log"

    def test_to_dict_all_fields(self):
        e = ActivityLogEntry(text="Error occurred", level="error")
        d = e.to_dict()
        assert d["text"] == "Error occurred"
        assert d["level"] == "error"

    def test_default_level(self):
        e = ActivityLogEntry(text="Some info")
        d = e.to_dict()
        assert d["level"] == "info"

    def test_json_serializable(self):
        e = ActivityLogEntry(text="test")
        serialized = json.dumps(e.to_dict())
        assert isinstance(serialized, str)


class TestPipelineProgress:
    def test_to_dict_type(self):
        e = PipelineProgress(operation="build", progress=0.5, message="Processing file 10/20")
        d = e.to_dict()
        assert d["type"] == "pipeline_progress"

    def test_to_dict_all_fields(self):
        e = PipelineProgress(operation="build", progress=0.5, message="Processing file 10/20")
        d = e.to_dict()
        assert d["operation"] == "build"
        assert d["progress"] == 0.5
        assert d["message"] == "Processing file 10/20"

    def test_json_serializable(self):
        e = PipelineProgress(operation="synthetic", progress=1.0, message="Done")
        serialized = json.dumps(e.to_dict())
        assert isinstance(serialized, str)


class TestEvaluationComplete:
    def test_to_dict_type(self):
        e = EvaluationComplete(f1=0.87, accuracy=0.91)
        d = e.to_dict()
        assert d["type"] == "evaluation_complete"

    def test_to_dict_all_fields(self):
        counts = {"Ramnit": 10, "Lollipop": 5}
        e = EvaluationComplete(f1=0.87, accuracy=0.91, family_counts=counts)
        d = e.to_dict()
        assert d["f1"] == 0.87
        assert d["accuracy"] == 0.91
        assert d["family_counts"] == {"Ramnit": 10, "Lollipop": 5}

    def test_default_family_counts(self):
        e = EvaluationComplete(f1=0.5, accuracy=0.6)
        d = e.to_dict()
        assert d["family_counts"] == {}

    def test_json_serializable(self):
        e = EvaluationComplete(f1=0.9, accuracy=0.92, family_counts={"X": 1})
        serialized = json.dumps(e.to_dict())
        assert isinstance(serialized, str)


class TestVaultSampleAdded:
    def test_to_dict_type(self):
        e = VaultSampleAdded(sample={"id": "v001"})
        d = e.to_dict()
        assert d["type"] == "vault_sample_added"

    def test_to_dict_all_fields(self):
        sample = {"id": "v001", "family": "Ramnit", "confidence": 0.72, "mutations": 3, "cycle": 1}
        e = VaultSampleAdded(sample=sample)
        d = e.to_dict()
        assert d["sample"] == sample

    def test_default_sample(self):
        e = VaultSampleAdded()
        d = e.to_dict()
        assert d["sample"] == {}

    def test_json_serializable(self):
        e = VaultSampleAdded(sample={"id": "v002"})
        serialized = json.dumps(e.to_dict())
        assert isinstance(serialized, str)


class TestFieldAccess:
    """Verify that fields are accessible as attributes, not just via to_dict()."""

    def test_epoch_complete_attrs(self):
        e = EpochComplete(
            epoch=1, phase="A", loss=0.5, train_acc=0.8, val_acc=0.75, f1=0.77, elapsed=3.2
        )
        assert e.epoch == 1
        assert e.phase == "A"
        assert e.loss == 0.5
        assert e.train_acc == 0.8
        assert e.val_acc == 0.75
        assert e.f1 == 0.77
        assert e.elapsed == 3.2

    def test_scan_progress_attrs(self):
        s = ScanProgress(phase="disassemble", data={"n_ops": 100})
        assert s.phase == "disassemble"
        assert s.data == {"n_ops": 100}

    def test_adversarial_cycle_end_attrs(self):
        a = AdversarialCycleEnd(cycle=3, metrics={"evasion_rate": 0.3})
        assert a.cycle == 3
        assert a.metrics == {"evasion_rate": 0.3}

    def test_adversarial_episode_step_attrs(self):
        e = AdversarialEpisodeStep(
            step=1, action="nop_insert", position=5, confidence=0.8, valid=True
        )
        assert e.step == 1
        assert e.action == "nop_insert"
        assert e.position == 5
        assert e.confidence == 0.8
        assert e.valid is True

    def test_activity_log_entry_attrs(self):
        e = ActivityLogEntry(text="hello", level="warn")
        assert e.text == "hello"
        assert e.level == "warn"

    def test_pipeline_progress_attrs(self):
        e = PipelineProgress(operation="build", progress=0.5, message="halfway")
        assert e.operation == "build"
        assert e.progress == 0.5
        assert e.message == "halfway"

    def test_evaluation_complete_attrs(self):
        e = EvaluationComplete(f1=0.87, accuracy=0.91, family_counts={"X": 1})
        assert e.f1 == 0.87
        assert e.accuracy == 0.91
        assert e.family_counts == {"X": 1}

    def test_vault_sample_added_attrs(self):
        e = VaultSampleAdded(sample={"id": "v001"})
        assert e.sample == {"id": "v001"}
