"""app.py — Main Wintermute TUI. Launch: wintermute tui. Keys: 1-6 switch tabs, c configure, q quit."""

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, TabbedContent, TabPane

from wintermute.tui import theme
from wintermute.tui.screens.dashboard import DashboardScreen
from wintermute.tui.screens.scan import ScanScreen
from wintermute.tui.screens.training import TrainingScreen
from wintermute.tui.screens.adversarial import AdversarialScreen
from wintermute.tui.screens.pipeline import PipelineScreen
from wintermute.tui.screens.vault import VaultScreen
from wintermute.tui.widgets.status_bar import StatusBar
from wintermute.tui.widgets.config_drawer import ConfigDrawer
from wintermute.tui.events import (
    EpochComplete,
    AdversarialCycleEnd,
    AdversarialEpisodeStep,
    ActivityLogEntry,
    PipelineProgress,
    EvaluationComplete,
    VaultSampleAdded,
)


class WintermuteApp(App):
    TITLE = "WINTERMUTE v3.0"
    SUB_TITLE = "MLX-native static malware classifier"
    CSS = theme.STYLESHEET

    BINDINGS = [
        Binding("1", "switch_tab('dashboard')", "Dashboard", show=True),
        Binding("2", "switch_tab('scan')", "Scan", show=True),
        Binding("3", "switch_tab('training')", "Training", show=True),
        Binding("4", "switch_tab('adversarial')", "Adversarial", show=True),
        Binding("5", "switch_tab('pipeline')", "Pipeline", show=True),
        Binding("6", "switch_tab('vault')", "Vault", show=True),
        Binding("c", "toggle_drawer", "Configure", show=True),
        Binding("ctrl+x", "cancel_task", "Cancel", show=False),
        Binding("q", "quit", "Quit", show=True),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(initial="dashboard"):
            with TabPane("◉ DASHBOARD", id="dashboard"):
                yield DashboardScreen()
            with TabPane("⊕ SCAN", id="scan"):
                yield ScanScreen()
            with TabPane("◈ TRAINING", id="training"):
                yield TrainingScreen()
            with TabPane("⚔ ADVERSARIAL", id="adversarial"):
                yield AdversarialScreen()
            with TabPane("⚙ PIPELINE", id="pipeline"):
                yield PipelineScreen()
            with TabPane("▣ VAULT", id="vault"):
                yield VaultScreen()
        yield StatusBar(id="status-bar")
        yield Footer()

    def action_switch_tab(self, tab_id: str) -> None:
        self.query_one(TabbedContent).active = tab_id

    def action_toggle_drawer(self) -> None:
        """Toggle the config drawer on the active screen."""
        try:
            active = self.query_one("TabPane.-active")
            drawer = active.query_one(ConfigDrawer)
            drawer.toggle()
        except Exception:
            pass

    def action_cancel_task(self) -> None:
        """Cancel the running operation on the active screen."""
        try:
            active = self.query_one("TabPane.-active")
            screen = active.children[0] if active.children else None
            if screen and hasattr(screen, "cancel_operation"):
                screen.cancel_operation()
        except Exception:
            pass

    def on_mount(self) -> None:
        self._log("TUI initialized", "ok")

    # ── Message routing ──

    def on_epoch_complete(self, event: EpochComplete) -> None:
        try:
            self.query_one(TrainingScreen).handle_epoch(event)
        except Exception:
            pass
        self._log(
            f"Epoch {event.epoch} ({event.phase}) — "
            f"loss={event.loss:.4f} val_acc={event.val_acc:.1%}",
            "info",
        )

    def on_adversarial_cycle_end(self, event: AdversarialCycleEnd) -> None:
        try:
            self.query_one(AdversarialScreen).update_cycle(event.cycle, event.metrics)
        except Exception:
            pass
        self._log(
            f"Adversarial cycle {event.cycle} — evasion={event.metrics.get('evasion_rate', 0):.1%}",
            "info",
        )

    def on_adversarial_episode_step(self, event: AdversarialEpisodeStep) -> None:
        try:
            from wintermute.tui.widgets.action_log import ActionLog

            log = self.query_one("#adv-episode", ActionLog)
            log.add_action(event.step, event.action, event.position, event.confidence, event.valid)
        except Exception:
            pass

    def on_pipeline_progress(self, event: PipelineProgress) -> None:
        try:
            bar = self.query_one("#status-bar", StatusBar)
            bar.set_task("pipeline", f"{event.operation}: {event.message}", event.progress)
        except Exception:
            pass

    def on_evaluation_complete(self, event: EvaluationComplete) -> None:
        try:
            dash = self.query_one(DashboardScreen)
            dash.update_stat("stat-f1", f"{event.f1:.2f}", f"accuracy {event.accuracy:.2f}")
            if event.family_counts:
                dash.update_family_chart(event.family_counts)
        except Exception:
            pass

    def on_vault_sample_added(self, event: VaultSampleAdded) -> None:
        try:
            vault = self.query_one(VaultScreen)
            vault.add_sample(event.sample)
        except Exception:
            pass

    def on_activity_log_entry(self, event: ActivityLogEntry) -> None:
        self._log(event.text, event.level)

    def _log(self, text: str, level: str = "info") -> None:
        try:
            dash_log = self.query_one("#dash-log")
            if hasattr(dash_log, "add_entry"):
                dash_log.add_entry(text, level)
        except Exception:
            pass


def run() -> None:
    WintermuteApp().run()


if __name__ == "__main__":
    run()
