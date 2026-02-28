"""Tests for api/schemas.py — Pydantic request/response models."""

import pytest
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# Job lifecycle
# ---------------------------------------------------------------------------


class TestJobResponse:
    def test_valid(self):
        from api.schemas import JobResponse

        obj = JobResponse(job_id="abc-123", poll_url="/api/v1/status/abc-123")
        assert obj.job_id == "abc-123"
        assert obj.poll_url == "/api/v1/status/abc-123"

    def test_missing_job_id(self):
        from api.schemas import JobResponse

        with pytest.raises(ValidationError):
            JobResponse(poll_url="/api/v1/status/abc-123")

    def test_missing_poll_url(self):
        from api.schemas import JobResponse

        with pytest.raises(ValidationError):
            JobResponse(job_id="abc-123")

    def test_round_trip(self):
        from api.schemas import JobResponse

        obj = JobResponse(job_id="abc-123", poll_url="/api/v1/status/abc-123")
        raw = obj.model_dump_json()
        restored = JobResponse.model_validate_json(raw)
        assert restored == obj


class TestJobStatus:
    def test_defaults(self):
        from api.schemas import JobStatus

        obj = JobStatus(job_id="abc", status="pending")
        assert obj.error is None

    def test_with_error(self):
        from api.schemas import JobStatus

        obj = JobStatus(job_id="abc", status="failed", error="timeout")
        assert obj.error == "timeout"

    def test_missing_status(self):
        from api.schemas import JobStatus

        with pytest.raises(ValidationError):
            JobStatus(job_id="abc")

    def test_round_trip(self):
        from api.schemas import JobStatus

        obj = JobStatus(job_id="x", status="done", error="nope")
        assert JobStatus.model_validate_json(obj.model_dump_json()) == obj


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


class TestDashboardResponse:
    def test_all_defaults(self):
        from api.schemas import DashboardResponse

        obj = DashboardResponse()
        assert obj.model_version == "3.0.0"
        assert obj.f1 == 0.0
        assert obj.accuracy == 0.0
        assert obj.vault_size == 0
        assert obj.family_counts == {}

    def test_custom_values(self):
        from api.schemas import DashboardResponse

        obj = DashboardResponse(
            model_version="4.0.0",
            f1=0.95,
            accuracy=0.92,
            vault_size=42,
            family_counts={"Ramnit": 10, "Kelihos_ver3": 5},
        )
        assert obj.family_counts["Ramnit"] == 10

    def test_round_trip(self):
        from api.schemas import DashboardResponse

        obj = DashboardResponse(vault_size=7, family_counts={"a": 1})
        assert DashboardResponse.model_validate_json(obj.model_dump_json()) == obj


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------


class TestScanResponse:
    def test_defaults(self):
        from api.schemas import ScanResponse

        obj = ScanResponse(job_id="j1", status="pending")
        assert obj.result is None
        assert obj.error is None

    def test_with_result(self):
        from api.schemas import ScanResponse

        obj = ScanResponse(
            job_id="j1",
            status="complete",
            result={"label": "malicious", "confidence": 0.97},
        )
        assert obj.result["label"] == "malicious"

    def test_missing_required(self):
        from api.schemas import ScanResponse

        with pytest.raises(ValidationError):
            ScanResponse(status="pending")

    def test_round_trip(self):
        from api.schemas import ScanResponse

        obj = ScanResponse(job_id="j1", status="ok", result={"k": "v"})
        assert ScanResponse.model_validate_json(obj.model_dump_json()) == obj


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class TestTrainingRequest:
    def test_all_defaults(self):
        from api.schemas import TrainingRequest

        obj = TrainingRequest()
        assert obj.epochs_phase_a == 5
        assert obj.epochs_phase_b == 20
        assert obj.learning_rate == 3e-4
        assert obj.batch_size == 8
        assert obj.max_seq_length == 2048
        assert obj.num_classes == 2
        assert obj.mlflow is False
        assert obj.experiment_name == "default"

    def test_override(self):
        from api.schemas import TrainingRequest

        obj = TrainingRequest(epochs_phase_a=10, mlflow=True)
        assert obj.epochs_phase_a == 10
        assert obj.mlflow is True

    def test_round_trip(self):
        from api.schemas import TrainingRequest

        obj = TrainingRequest(epochs_phase_a=3, learning_rate=1e-3)
        assert TrainingRequest.model_validate_json(obj.model_dump_json()) == obj


class TestTrainingStatus:
    def test_defaults(self):
        from api.schemas import TrainingStatus

        obj = TrainingStatus(job_id="t1", status="running")
        assert obj.epoch == 0
        assert obj.phase == ""
        assert obj.loss == 0.0
        assert obj.train_acc == 0.0
        assert obj.val_acc == 0.0
        assert obj.f1 == 0.0

    def test_with_progress(self):
        from api.schemas import TrainingStatus

        obj = TrainingStatus(
            job_id="t1",
            status="running",
            epoch=5,
            phase="B",
            loss=0.12,
            train_acc=0.93,
            val_acc=0.88,
            f1=0.90,
        )
        assert obj.epoch == 5
        assert obj.phase == "B"

    def test_round_trip(self):
        from api.schemas import TrainingStatus

        obj = TrainingStatus(job_id="t1", status="done", epoch=10, f1=0.95)
        assert TrainingStatus.model_validate_json(obj.model_dump_json()) == obj


# ---------------------------------------------------------------------------
# Adversarial
# ---------------------------------------------------------------------------


class TestAdversarialRequest:
    def test_all_defaults(self):
        from api.schemas import AdversarialRequest

        obj = AdversarialRequest()
        assert obj.cycles == 10
        assert obj.episodes_per_cycle == 500
        assert obj.trades_beta == 1.0
        assert obj.ewc_lambda == 0.4
        assert obj.ppo_lr == 3e-4
        assert obj.ppo_epochs == 4

    def test_override(self):
        from api.schemas import AdversarialRequest

        obj = AdversarialRequest(cycles=20, ppo_epochs=8)
        assert obj.cycles == 20
        assert obj.ppo_epochs == 8

    def test_round_trip(self):
        from api.schemas import AdversarialRequest

        obj = AdversarialRequest(cycles=5)
        assert AdversarialRequest.model_validate_json(obj.model_dump_json()) == obj


class TestAdversarialStatus:
    def test_defaults(self):
        from api.schemas import AdversarialStatus

        obj = AdversarialStatus(job_id="a1", status="running")
        assert obj.cycle == 0
        assert obj.evasion_rate == 0.0
        assert obj.adv_tpr == 0.0
        assert obj.vault_size == 0

    def test_round_trip(self):
        from api.schemas import AdversarialStatus

        obj = AdversarialStatus(job_id="a1", status="done", cycle=10, evasion_rate=0.15)
        assert AdversarialStatus.model_validate_json(obj.model_dump_json()) == obj


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class TestPipelineRequest:
    def test_all_defaults(self):
        from api.schemas import PipelineRequest

        obj = PipelineRequest()
        assert obj.data_dir == "data"
        assert obj.max_seq_length == 2048
        assert obj.vocab_size is None
        assert obj.n_samples == 500
        assert obj.output_dir == "data/processed"
        assert obj.seed == 42
        assert obj.epochs == 50
        assert obj.learning_rate == 3e-4
        assert obj.batch_size == 8
        assert obj.mask_prob == 0.15

    def test_override(self):
        from api.schemas import PipelineRequest

        obj = PipelineRequest(vocab_size=10000, seed=7)
        assert obj.vocab_size == 10000
        assert obj.seed == 7

    def test_round_trip(self):
        from api.schemas import PipelineRequest

        obj = PipelineRequest(epochs=10)
        assert PipelineRequest.model_validate_json(obj.model_dump_json()) == obj


class TestPipelineStatus:
    def test_defaults(self):
        from api.schemas import PipelineStatus

        obj = PipelineStatus(job_id="p1", status="running")
        assert obj.operation == ""
        assert obj.progress == 0.0
        assert obj.message == ""

    def test_round_trip(self):
        from api.schemas import PipelineStatus

        obj = PipelineStatus(job_id="p1", status="running", operation="build", progress=0.5)
        assert PipelineStatus.model_validate_json(obj.model_dump_json()) == obj


# ---------------------------------------------------------------------------
# Vault
# ---------------------------------------------------------------------------


class TestVaultSample:
    def test_required_fields(self):
        from api.schemas import VaultSample

        with pytest.raises(ValidationError):
            VaultSample()

    def test_valid(self):
        from api.schemas import VaultSample

        obj = VaultSample(id="v1", family="Ramnit", confidence=0.95, mutations=3, cycle=2)
        assert obj.id == "v1"
        assert obj.family == "Ramnit"
        assert obj.confidence == 0.95
        assert obj.mutations == 3
        assert obj.cycle == 2

    def test_round_trip(self):
        from api.schemas import VaultSample

        obj = VaultSample(id="v1", family="Ramnit", confidence=0.95, mutations=3, cycle=2)
        assert VaultSample.model_validate_json(obj.model_dump_json()) == obj


class TestVaultSampleDetail:
    def test_inherits_from_vault_sample(self):
        from api.schemas import VaultSample, VaultSampleDetail

        assert issubclass(VaultSampleDetail, VaultSample)

    def test_defaults(self):
        from api.schemas import VaultSampleDetail

        obj = VaultSampleDetail(id="v1", family="Ramnit", confidence=0.95, mutations=3, cycle=2)
        assert obj.original_bytes == ""
        assert obj.mutated_bytes == ""
        assert obj.diff == ""

    def test_with_detail_fields(self):
        from api.schemas import VaultSampleDetail

        obj = VaultSampleDetail(
            id="v1",
            family="Ramnit",
            confidence=0.95,
            mutations=3,
            cycle=2,
            original_bytes="deadbeef",
            mutated_bytes="cafebabe",
            diff="--- a\n+++ b\n@@ -1 +1 @@\n-dead\n+cafe",
        )
        assert obj.original_bytes == "deadbeef"
        assert obj.mutated_bytes == "cafebabe"
        assert "cafe" in obj.diff

    def test_round_trip(self):
        from api.schemas import VaultSampleDetail

        obj = VaultSampleDetail(
            id="v1",
            family="X",
            confidence=0.5,
            mutations=1,
            cycle=0,
            original_bytes="aa",
            mutated_bytes="bb",
            diff="d",
        )
        assert VaultSampleDetail.model_validate_json(obj.model_dump_json()) == obj

    def test_has_all_parent_fields(self):
        """VaultSampleDetail should serialize parent + child fields."""
        from api.schemas import VaultSampleDetail

        obj = VaultSampleDetail(
            id="v1",
            family="X",
            confidence=0.5,
            mutations=1,
            cycle=0,
        )
        data = obj.model_dump()
        assert "id" in data
        assert "family" in data
        assert "confidence" in data
        assert "mutations" in data
        assert "cycle" in data
        assert "original_bytes" in data
        assert "mutated_bytes" in data
        assert "diff" in data
