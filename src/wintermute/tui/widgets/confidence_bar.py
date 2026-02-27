"""confidence_bar.py — Horizontal confidence bar with label and percentage."""

from textual.widgets import Static
from wintermute.tui import theme

BAR_WIDTH = 30


class ConfidenceBar(Static):
    """Horizontal bar showing a 0.0–1.0 value."""

    DEFAULT_CSS = """
    ConfidenceBar {
        height: 2;
        padding: 0 1;
    }
    """

    def __init__(self, label: str, value: float = 0.0, color: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self._label = label
        self._value = max(0.0, min(1.0, value))
        self._color = color or theme.CYAN

    def render(self) -> str:
        pct = f"{self._value * 100:.1f}%"
        filled = int(self._value * BAR_WIDTH)
        empty = BAR_WIDTH - filled
        bar = f"[{self._color}]{'█' * filled}[/][{theme.BORDER}]{'░' * empty}[/]"
        return f"[{theme.TEXT_MUTED}]{self._label:<14}[/] [{self._color}]{pct:>6}[/]  {bar}"

    def update_value(self, value: float) -> None:
        self._value = max(0.0, min(1.0, value))
        self.refresh()
