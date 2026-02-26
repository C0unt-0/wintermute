"""training.py — Live training visualization. Matches JointTrainer phases A/B."""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, Sparkline, ProgressBar, DataTable
from wintermute.tui import theme
from wintermute.tui.events import EpochComplete


class TrainingScreen(Vertical):

    DEFAULT_CSS = """
    TrainingScreen {
        height: 100%;
        padding: 1;
    }
    #train-status { height: 3; margin-bottom: 1; }
    #train-progress { height: 1; margin-bottom: 1; }
    #train-charts { layout: horizontal; height: 1fr; margin-bottom: 1; }
    .train-chart { width: 1fr; margin: 0 1; }
    #train-epochs { height: 12; }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._loss_data: list[float] = []
        self._acc_data: list[float] = []

    def compose(self) -> ComposeResult:
        yield TrainStatusBar(id="train-status")
        yield ProgressBar(total=100, show_eta=True, id="train-progress")
        with Horizontal(id="train-charts"):
            yield SparkPanel("LOSS", theme.AMBER, id="loss-panel",
                             classes="train-chart")
            yield SparkPanel("ACCURACY", theme.GREEN, id="acc-panel",
                             classes="train-chart")
        yield EpochTable(id="train-epochs")

    def handle_epoch(self, event: EpochComplete) -> None:
        """Called by app.on_epoch_complete()."""
        status = self.query_one("#train-status", TrainStatusBar)
        status.set_state(event.phase, event.epoch, event.loss)

        self._loss_data.append(event.loss)
        self._acc_data.append(event.val_acc)

        loss_panel = self.query_one("#loss-panel", SparkPanel)
        loss_panel.set_data(self._loss_data, f"{event.loss:.4f}")
        acc_panel = self.query_one("#acc-panel", SparkPanel)
        acc_panel.set_data(self._acc_data, f"{event.val_acc:.1%}")

        table = self.query_one("#train-epochs", EpochTable)
        table.add_epoch(event.epoch, event.phase, event.loss,
                        event.train_acc, event.val_acc, event.f1,
                        event.elapsed)


class TrainStatusBar(Static):

    DEFAULT_CSS = f"""
    TrainStatusBar {{
        background: {theme.BG_CARD};
        border: solid {theme.BORDER};
        padding: 0 2;
        height: 3;
    }}
    """

    def render(self) -> str:
        return (f"  [{theme.TEXT_MUTED}]PHASE[/] [{theme.TEXT_MUTED}]—[/]"
                f"  [{theme.TEXT_MUTED}]Start training to see live metrics[/]")

    def set_state(self, phase: str, epoch: int, loss: float) -> None:
        phase_label = "A — ENCODER FROZEN" if phase == "A" else "B — FULL FINETUNE"
        phase_color = theme.AMBER if phase == "A" else theme.CYAN
        self.update(
            f"  [{theme.TEXT_MUTED}]PHASE[/] "
            f"[bold {phase_color} on {theme.BG_HOVER}] {phase_label} [/]"
            f"  [{theme.TEXT_MUTED}]EPOCH[/] [{theme.TEXT_BRIGHT}]{epoch}[/]"
            f"  [{theme.TEXT_MUTED}]LOSS[/] [{theme.AMBER}]{loss:.4f}[/]")


class SparkPanel(Vertical):

    DEFAULT_CSS = f"""
    SparkPanel {{
        background: {theme.BG_CARD};
        border: solid {theme.BORDER};
        padding: 1;
    }}
    """

    def __init__(self, label: str, color: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._label = label
        self._color = color

    def compose(self) -> ComposeResult:
        yield Static(
            f"[{theme.TEXT_MUTED}]{self._label}[/]  [{self._color}]—[/]",
            id=f"{self.id}-lbl")
        yield Sparkline([], id=f"{self.id}-spark")

    def set_data(self, data: list[float], current: str) -> None:
        try:
            self.query_one(f"#{self.id}-lbl", Static).update(
                f"[{theme.TEXT_MUTED}]{self._label}[/]  [{self._color}]{current}[/]")
            self.query_one(f"#{self.id}-spark", Sparkline).data = data
        except Exception:
            pass


class EpochTable(DataTable):
    """Columns match JointTrainer output."""

    DEFAULT_CSS = f"""
    EpochTable {{
        background: {theme.BG_CARD};
        border: solid {theme.BORDER};
    }}
    """

    def on_mount(self) -> None:
        self.add_columns("Epoch", "Phase", "Loss", "Train Acc",
                         "Val Acc", "F1", "Time")

    def add_epoch(self, epoch: int, phase: str, loss: float,
                  train_acc: float, val_acc: float, f1: float,
                  elapsed: float) -> None:
        self.add_row(str(epoch), phase, f"{loss:.4f}", f"{train_acc:.1%}",
                     f"{val_acc:.1%}", f"{f1:.3f}", f"{elapsed:.1f}s")
        self.scroll_end()
