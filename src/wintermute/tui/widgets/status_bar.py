"""Global status bar showing background task state across all screens."""

from __future__ import annotations

from textual.widgets import Static
from rich.text import Text

from wintermute.tui import theme


class StatusBar(Static):
    """Persistent bar above footer showing running background tasks."""

    DEFAULT_CSS = f"""
    StatusBar {{
        height: 1;
        background: {theme.BG_PANEL};
        color: {theme.TEXT_MUTED};
        padding: 0 2;
    }}
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._tasks: dict[str, dict] = {}

    def set_task(self, key: str, label: str, progress: float = -1) -> None:
        """Set or update a running task. progress in [0, 1] or -1 for indeterminate."""
        self._tasks[key] = {"label": label, "progress": progress}
        self.refresh()

    def clear_task(self, key: str) -> None:
        """Remove a completed/cancelled task."""
        self._tasks.pop(key, None)
        self.refresh()

    def render(self) -> Text:
        if not self._tasks:
            return Text.from_markup(
                f"[{theme.TEXT_MUTED}]Ready — press [{theme.CYAN} bold]c[/] to configure"
            )
        parts = []
        for key, info in self._tasks.items():
            label = info["label"]
            progress = info["progress"]
            if progress >= 0:
                filled = int(progress * 10)
                bar = "▪" * filled + "░" * (10 - filled)
                pct = f"{progress * 100:.0f}%"
                parts.append(
                    f"[{theme.CYAN}]{key.title()}:[/] {label} {bar} {pct}"
                )
            else:
                parts.append(f"[{theme.CYAN}]{key.title()}:[/] {label}")
        return Text.from_markup("  ".join(parts))
