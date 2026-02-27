"""action_log.py — Streaming action log for adversarial episodes."""

from rich.text import Text
from textual.widgets import RichLog
from wintermute.tui import theme


class ActionLog(RichLog):
    """Scrolling log for adversarial episode actions and system events."""

    DEFAULT_CSS = f"""
    ActionLog {{
        height: 100%;
        background: {theme.BG_CARD};
        border: solid {theme.BORDER};
    }}
    """

    _LEVEL_COLORS = {
        "info": theme.CYAN,
        "ok": theme.GREEN,
        "warn": theme.AMBER,
        "error": theme.RED,
    }

    def add_action(self, step: int, action: str, pos: int, conf: float, ok: bool) -> None:
        conf_color = theme.GREEN if conf < 0.5 else theme.AMBER if conf < 0.7 else theme.TEXT_MUTED
        line = Text()
        line.append(f"{step:3d} ", style=theme.TEXT_MUTED)
        line.append("✓ " if ok else "✗ ", style=theme.GREEN if ok else theme.RED)
        line.append(f"{action:<20s}", style=theme.TEXT_BRIGHT if ok else theme.RED)
        line.append(f" @{pos:<4d}", style=theme.TEXT_MUTED)
        line.append(f" {conf:.3f}", style=conf_color)
        self.write(line)

    def add_entry(self, text: str, level: str = "info") -> None:
        from datetime import datetime

        color = self._LEVEL_COLORS.get(level, theme.TEXT)
        ts = datetime.now().strftime("%H:%M:%S")
        line = Text()
        line.append(f"  {ts} ", style=theme.TEXT_MUTED)
        line.append("● ", style=color)
        line.append(text, style=theme.TEXT)
        self.write(line)
