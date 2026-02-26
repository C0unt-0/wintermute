"""app.py — Main Wintermute TUI. Launch: wintermute tui. Keys: 1-5 switch tabs, q quit."""

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, TabbedContent, TabPane

from wintermute.tui import theme
from wintermute.tui.screens.dashboard import DashboardScreen
from wintermute.tui.screens.scan import ScanScreen
from wintermute.tui.screens.training import TrainingScreen
from wintermute.tui.screens.adversarial import AdversarialScreen
from wintermute.tui.screens.vault import VaultScreen
from wintermute.tui.events import EpochComplete, AdversarialCycleEnd, AdversarialEpisodeStep


class WintermuteApp(App):

    TITLE = "WINTERMUTE v3.0"
    SUB_TITLE = "MLX-native static malware classifier"
    CSS = theme.STYLESHEET

    BINDINGS = [
        Binding("1", "switch_tab('dashboard')", "Dashboard", show=True),
        Binding("2", "switch_tab('scan')", "Scan", show=True),
        Binding("3", "switch_tab('training')", "Training", show=True),
        Binding("4", "switch_tab('adversarial')", "Adversarial", show=True),
        Binding("5", "switch_tab('vault')", "Vault", show=True),
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
            with TabPane("▣ VAULT", id="vault"):
                yield VaultScreen()
        yield Footer()

    def action_switch_tab(self, tab_id: str) -> None:
        self.query_one(TabbedContent).active = tab_id

    def on_mount(self) -> None:
        self._log("TUI initialized", "ok")

    # ── Message routing ──

    def on_epoch_complete(self, event: EpochComplete) -> None:
        try:
            self.query_one(TrainingScreen).handle_epoch(event)
        except Exception:
            pass
        self._log(f"Epoch {event.epoch} ({event.phase}) — "
                  f"loss={event.loss:.4f} val_acc={event.val_acc:.1%}", "info")

    def on_adversarial_cycle_end(self, event: AdversarialCycleEnd) -> None:
        try:
            self.query_one(AdversarialScreen).update_cycle(
                event.cycle, event.metrics)
        except Exception:
            pass
        self._log(f"Adversarial cycle {event.cycle} — "
                  f"evasion={event.metrics.get('evasion_rate', 0):.1%}", "info")

    def on_adversarial_episode_step(self, event: AdversarialEpisodeStep) -> None:
        try:
            from wintermute.tui.widgets.action_log import ActionLog
            log = self.query_one("#adv-episode", ActionLog)
            log.add_action(event.step, event.action, event.position,
                           event.confidence, event.valid)
        except Exception:
            pass

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
