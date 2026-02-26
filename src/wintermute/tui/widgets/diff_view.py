"""diff_view.py — Mutation diff viewer for adversarial vault samples."""

from textual.widgets import Static
from wintermute.tui import theme


class DiffView(Static):
    """Red/green diff of original vs mutated assembly."""

    DEFAULT_CSS = """
    DiffView {
        height: auto;
        padding: 0 1;
    }
    """

    def __init__(self, lines: list[tuple[str, str]] | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._lines = lines or []

    def render(self) -> str:
        if not self._lines:
            return f"[{theme.TEXT_MUTED}]Select a sample to view diff[/]"
        out = []
        for kind, text in self._lines:
            if kind == "add":
                out.append(f"[{theme.GREEN}]+ {text}[/]")
            elif kind == "del":
                out.append(f"[{theme.RED}]- {text}[/]")
            else:
                out.append(f"[{theme.TEXT_MUTED}]  {text}[/]")
        return "\n".join(out)

    def update_lines(self, lines: list[tuple[str, str]]) -> None:
        self._lines = lines
        self.refresh()
