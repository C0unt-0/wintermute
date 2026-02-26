"""events.py — Custom Textual messages for real-time TUI updates."""

from textual.message import Message


class EpochComplete(Message):
    """Fired after each training epoch. phase is 'A' or 'B' matching JointTrainer."""

    def __init__(self, epoch: int, phase: str, loss: float,
                 train_acc: float, val_acc: float, f1: float,
                 elapsed: float) -> None:
        super().__init__()
        self.epoch = epoch
        self.phase = phase
        self.loss = loss
        self.train_acc = train_acc
        self.val_acc = val_acc
        self.f1 = f1
        self.elapsed = elapsed


class ScanProgress(Message):
    """Fired during scan phases."""

    def __init__(self, phase: str, data: dict | None = None) -> None:
        super().__init__()
        self.phase = phase
        self.data = data or {}


class AdversarialCycleEnd(Message):
    """Fired at end of each adversarial training cycle."""

    def __init__(self, cycle: int, metrics: dict) -> None:
        super().__init__()
        self.cycle = cycle
        self.metrics = metrics


class AdversarialEpisodeStep(Message):
    """Fired for each action in a live adversarial episode."""

    def __init__(self, step: int, action: str, position: int,
                 confidence: float, valid: bool) -> None:
        super().__init__()
        self.step = step
        self.action = action
        self.position = position
        self.confidence = confidence
        self.valid = valid


class ActivityLogEntry(Message):
    """Generic log entry for the dashboard activity log."""

    def __init__(self, text: str, level: str = "info") -> None:
        super().__init__()
        self.text = text
        self.level = level
