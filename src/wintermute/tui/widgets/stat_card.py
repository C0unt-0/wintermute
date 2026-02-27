"""stat_card.py — Key metric display widget."""

from textual.widgets import Static
from wintermute.tui import theme


class StatCard(Static):
    """Displays a labeled metric value with optional subtitle."""

    DEFAULT_CSS = f"""
    StatCard {{
        height: auto;
        padding: 1 2;
        background: {theme.BG_CARD};
        border: solid {theme.BORDER};
    }}
    """

    def __init__(
        self, label: str, value: str, subtitle: str = "", accent: str = "", **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._label = label
        self._value = value
        self._subtitle = subtitle
        self._accent = accent or theme.CYAN

    def render(self) -> str:
        parts = [
            f"[{theme.TEXT_MUTED}]{self._label}[/]",
            f"[bold {self._accent}]{self._value}[/]",
        ]
        if self._subtitle:
            parts.append(f"[{theme.TEXT_MUTED}]{self._subtitle}[/]")
        return "\n".join(parts)

    def update_value(self, value: str, subtitle: str | None = None) -> None:
        self._value = value
        if subtitle is not None:
            self._subtitle = subtitle
        self.refresh()
