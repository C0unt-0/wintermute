"""test_tui.py — Tests for wintermute.tui"""

import pytest
textual = pytest.importorskip("textual")


class TestTheme:
    def test_color_tokens_exist(self):
        from wintermute.tui import theme
        assert theme.CYAN == "#00e5ff"
        assert theme.RED == "#ff3366"
        assert theme.GREEN == "#00ff9f"
        assert theme.BG == "#0a0e14"

    def test_stylesheet_nonempty(self):
        from wintermute.tui.theme import STYLESHEET
        assert len(STYLESHEET) > 200
        assert "Screen" in STYLESHEET

    def test_stylesheet_has_drawer_styles(self):
        from wintermute.tui import theme
        assert "ConfigDrawer" in theme.STYLESHEET
        assert "StatusBar" in theme.STYLESHEET


class TestEvents:
    def test_epoch_complete(self):
        from wintermute.tui.events import EpochComplete
        e = EpochComplete(epoch=1, phase="A", loss=0.5,
                          train_acc=0.8, val_acc=0.75, f1=0.77, elapsed=3.2)
        assert e.epoch == 1 and e.phase == "A"

    def test_scan_progress(self):
        from wintermute.tui.events import ScanProgress
        s = ScanProgress("disassemble", {"n_ops": 100})
        assert s.phase == "disassemble"

    def test_adversarial_cycle_end(self):
        from wintermute.tui.events import AdversarialCycleEnd
        a = AdversarialCycleEnd(cycle=3, metrics={"evasion_rate": 0.3})
        assert a.cycle == 3

    def test_pipeline_progress(self):
        from wintermute.tui.events import PipelineProgress
        e = PipelineProgress(operation="build", progress=0.5, message="Processing file 10/20")
        assert e.operation == "build"
        assert e.progress == 0.5
        assert e.message == "Processing file 10/20"

    def test_evaluation_complete(self):
        from wintermute.tui.events import EvaluationComplete
        counts = {"Ramnit": 10, "Lollipop": 5}
        e = EvaluationComplete(f1=0.87, accuracy=0.91, family_counts=counts)
        assert e.f1 == 0.87
        assert e.accuracy == 0.91
        assert e.family_counts["Ramnit"] == 10

    def test_vault_sample_added(self):
        from wintermute.tui.events import VaultSampleAdded
        sample = {"id": "v001", "family": "Ramnit", "confidence": 0.72, "mutations": 3, "cycle": 1}
        e = VaultSampleAdded(sample=sample)
        assert e.sample["id"] == "v001"
        assert e.sample["confidence"] == 0.72


class TestWidgets:
    def test_stat_card_render(self):
        from wintermute.tui.widgets.stat_card import StatCard
        from wintermute.tui import theme
        card = StatCard("METRIC", "42%", "subtitle", accent=theme.GREEN)
        assert "METRIC" in card.render() and "42%" in card.render()

    def test_stat_card_update(self):
        from wintermute.tui.widgets.stat_card import StatCard
        card = StatCard("X", "0")
        card.update_value("99%", "new sub")
        assert card._value == "99%"

    def test_confidence_bar_render(self):
        from wintermute.tui.widgets.confidence_bar import ConfidenceBar
        bar = ConfidenceBar("Malicious", 0.95, "#ff3366")
        assert "95.0%" in bar.render()

    def test_confidence_bar_clamps(self):
        from wintermute.tui.widgets.confidence_bar import ConfidenceBar
        bar = ConfidenceBar("Test", 0.5)
        bar.update_value(1.5)
        assert bar._value == 1.0
        bar.update_value(-0.5)
        assert bar._value == 0.0

    def test_diff_view_render(self):
        from wintermute.tui.widgets.diff_view import DiffView
        diff = DiffView(lines=[("same", "push ebp"), ("del", "xor eax, eax"),
                                ("add", "sub eax, eax")])
        assert "push ebp" in diff.render()

    def test_diff_view_empty(self):
        from wintermute.tui.widgets.diff_view import DiffView
        assert "Select" in DiffView().render()


class TestApp:
    def test_app_creates(self):
        from wintermute.tui.app import WintermuteApp
        assert WintermuteApp().title == "WINTERMUTE v3.0"

    def test_app_css_loaded(self):
        from wintermute.tui.app import WintermuteApp
        assert len(WintermuteApp().CSS) > 200


class TestHooks:
    def test_training_hook_no_app(self):
        from wintermute.tui.hooks import TrainingHook
        TrainingHook().on_epoch(1, "B", 0.1, 0.9, 0.85, 0.87, 2.0)

    def test_adversarial_hook_no_app(self):
        from wintermute.tui.hooks import AdversarialHook
        hook = AdversarialHook()
        hook.on_episode_step(1, "nop_insert", 5, 0.8, True)
        hook.on_cycle_end(1, {"evasion_rate": 0.3})

    def test_training_hook_cancelled(self):
        from wintermute.tui.hooks import TrainingHook
        hook = TrainingHook()
        assert hook.cancelled is False
        hook.cancel()
        assert hook.cancelled is True
        hook.reset()
        assert hook.cancelled is False

    def test_adversarial_hook_cancelled(self):
        from wintermute.tui.hooks import AdversarialHook
        hook = AdversarialHook()
        assert hook.cancelled is False
        hook.cancel()
        assert hook.cancelled is True

    def test_pipeline_hook_no_app(self):
        from wintermute.tui.hooks import PipelineHook
        hook = PipelineHook()
        hook.on_progress("build", 0.5, "halfway")
        hook.on_log("test message")
        assert hook.cancelled is False
        hook.cancel()
        assert hook.cancelled is True

    def test_adversarial_hook_vault_sample(self):
        from wintermute.tui.hooks import AdversarialHook
        hook = AdversarialHook()
        sample = {"id": "v001", "family": "Ramnit"}
        hook.on_vault_sample(sample)  # Should not raise without app


class TestConfigDrawer:
    def test_create_drawer(self):
        from wintermute.tui.widgets.config_drawer import ConfigDrawer
        drawer = ConfigDrawer()
        assert drawer is not None

    def test_field_definitions(self):
        from wintermute.tui.widgets.config_drawer import FieldDef
        f = FieldDef(name="epochs", label="Epochs", default="50", field_type="int")
        assert f.name == "epochs"
        assert f.default == "50"
        assert f.field_type == "int"

    def test_select_field(self):
        from wintermute.tui.widgets.config_drawer import FieldDef
        f = FieldDef(
            name="num_classes",
            label="Num Classes",
            default="2",
            field_type="select",
            options=["2", "9"],
        )
        assert f.options == ["2", "9"]

    def test_switch_field(self):
        from wintermute.tui.widgets.config_drawer import FieldDef
        f = FieldDef(
            name="mlflow",
            label="MLflow Tracking",
            default="off",
            field_type="switch",
        )
        assert f.field_type == "switch"

    def test_get_values_defaults(self):
        from wintermute.tui.widgets.config_drawer import ConfigDrawer, FieldDef
        fields = [
            FieldDef(name="epochs", label="Epochs", default="50", field_type="int"),
            FieldDef(name="lr", label="Learning Rate", default="3e-4", field_type="float"),
        ]
        drawer = ConfigDrawer(fields=fields)
        values = drawer.get_values()
        assert values["epochs"] == "50"
        assert values["lr"] == "3e-4"


class TestStatusBar:
    def test_create(self):
        from wintermute.tui.widgets.status_bar import StatusBar
        bar = StatusBar()
        assert bar is not None

    def test_initial_render(self):
        from wintermute.tui.widgets.status_bar import StatusBar
        bar = StatusBar()
        text = bar.render()
        assert "Ready" in str(text)

    def test_set_task(self):
        from wintermute.tui.widgets.status_bar import StatusBar
        bar = StatusBar()
        bar.set_task("training", "epoch 5/50", 0.1)
        assert bar._tasks["training"]["label"] == "epoch 5/50"

    def test_clear_task(self):
        from wintermute.tui.widgets.status_bar import StatusBar
        bar = StatusBar()
        bar.set_task("training", "epoch 5/50", 0.1)
        bar.clear_task("training")
        assert "training" not in bar._tasks

    def test_render_with_task(self):
        from wintermute.tui.widgets.status_bar import StatusBar
        bar = StatusBar()
        bar.set_task("training", "epoch 5/50", 0.5)
        text = str(bar.render())
        assert "Training" in text
        assert "epoch 5/50" in text

    def test_render_indeterminate(self):
        from wintermute.tui.widgets.status_bar import StatusBar
        bar = StatusBar()
        bar.set_task("pipeline", "building...", -1)
        text = str(bar.render())
        assert "Pipeline" in text
        assert "building..." in text


class TestEngineHookIntegration:
    def test_joint_trainer_accepts_hook(self):
        import inspect
        from wintermute.engine.joint_trainer import JointTrainer
        sig = inspect.signature(JointTrainer.__init__)
        assert "tui_hook" in sig.parameters

    def test_pretrain_accepts_hook(self):
        import inspect
        from wintermute.engine.pretrain import MLMPretrainer
        sig = inspect.signature(MLMPretrainer.__init__)
        assert "tui_hook" in sig.parameters

    def test_orchestrator_accepts_hook(self):
        import inspect
        from wintermute.adversarial.orchestrator import AdversarialOrchestrator
        sig = inspect.signature(AdversarialOrchestrator.__init__)
        assert "tui_hook" in sig.parameters


class TestTrainingScreen:
    def test_has_training_fields(self):
        from wintermute.tui.screens.training import TRAINING_FIELDS
        names = [f.name for f in TRAINING_FIELDS]
        assert "epochs_phase_a" in names
        assert "epochs_phase_b" in names
        assert "learning_rate" in names
        assert "batch_size" in names

    def test_has_cancel_operation(self):
        from wintermute.tui.screens.training import TrainingScreen
        screen = TrainingScreen()
        assert hasattr(screen, "cancel_operation")


class TestAdversarialScreen:
    def test_has_adversarial_fields(self):
        from wintermute.tui.screens.adversarial import ADVERSARIAL_FIELDS
        names = [f.name for f in ADVERSARIAL_FIELDS]
        assert "cycles" in names
        assert "trades_beta" in names
        assert "ewc_lambda" in names

    def test_has_cancel_operation(self):
        from wintermute.tui.screens.adversarial import AdversarialScreen
        screen = AdversarialScreen()
        assert hasattr(screen, "cancel_operation")


class TestScanScreen:
    def test_has_scan_fields(self):
        from wintermute.tui.screens.scan import SCAN_FIELDS
        names = [f.name for f in SCAN_FIELDS]
        assert "file_path" in names
        assert "family" in names

    def test_has_cancel_operation(self):
        from wintermute.tui.screens.scan import ScanScreen
        screen = ScanScreen()
        assert hasattr(screen, "cancel_operation")


class TestPipelineScreen:
    def test_create(self):
        from wintermute.tui.screens.pipeline import PipelineScreen
        screen = PipelineScreen()
        assert screen is not None

    def test_operation_fields(self):
        from wintermute.tui.screens.pipeline import BUILD_FIELDS, SYNTHETIC_FIELDS, PRETRAIN_FIELDS
        assert any(f.name == "data_dir" for f in BUILD_FIELDS)
        assert any(f.name == "n_samples" for f in SYNTHETIC_FIELDS)
        assert any(f.name == "epochs" for f in PRETRAIN_FIELDS)

    def test_has_cancel_operation(self):
        from wintermute.tui.screens.pipeline import PipelineScreen
        screen = PipelineScreen()
        assert hasattr(screen, "cancel_operation")


class TestDashboardWiring:
    def test_has_update_family_chart(self):
        from wintermute.tui.screens.dashboard import DashboardScreen
        screen = DashboardScreen()
        assert hasattr(screen, "update_family_chart")

    def test_family_chart_accepts_data(self):
        from wintermute.tui.screens.dashboard import FamilyChart
        chart = FamilyChart()
        assert hasattr(chart, "update_counts")


class TestVaultWiring:
    def test_has_add_sample(self):
        from wintermute.tui.screens.vault import VaultScreen
        screen = VaultScreen()
        assert hasattr(screen, "add_sample")


class TestCLI:
    def test_tui_in_help(self):
        from typer.testing import CliRunner
        from wintermute.cli import app
        result = CliRunner().invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "tui" in result.output.lower()
