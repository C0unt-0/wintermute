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


class TestCLI:
    def test_tui_in_help(self):
        from typer.testing import CliRunner
        from wintermute.cli import app
        result = CliRunner().invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "tui" in result.output.lower()
