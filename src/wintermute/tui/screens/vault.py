"""vault.py — Adversarial vault browser with diff viewer."""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, DataTable
from wintermute.tui import theme
from wintermute.tui.widgets.diff_view import DiffView


class VaultScreen(Horizontal):

    DEFAULT_CSS = """
    VaultScreen {
        height: 100%;
        padding: 1;
    }
    #vault-table { width: 1fr; margin-right: 1; }
    #vault-detail-col { width: 38; }
    """

    def compose(self) -> ComposeResult:
        yield VaultTable(id="vault-table")
        with Vertical(id="vault-detail-col"):
            yield Static(
                f"[bold {theme.TEXT_BRIGHT}]SAMPLE DETAIL[/]\n\n"
                f"[{theme.TEXT_MUTED}]Select a vault entry to inspect[/]",
                id="vault-info")
            yield DiffView(id="vault-diff")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self.query_one("#vault-info", Static).update(
            f"[bold {theme.TEXT_BRIGHT}]SAMPLE DETAIL[/]\n\n"
            f"  [{theme.TEXT_MUTED}]Row key:[/] [{theme.PURPLE}]{event.row_key}[/]\n"
            f"  [{theme.TEXT_MUTED}]Full detail requires vault data[/]")

    def load_entries(self, entries: list[dict]) -> None:
        table = self.query_one("#vault-table", VaultTable)
        for e in entries:
            table.add_row(
                e.get("id", "—"), e.get("family", "—"),
                f"{e.get('confidence', 0):.3f}",
                str(e.get("mutations", 0)), str(e.get("cycle", 0)),
                key=e.get("id"))


class VaultTable(DataTable):

    DEFAULT_CSS = f"""
    VaultTable {{
        background: {theme.BG_CARD};
        border: solid {theme.BORDER};
    }}
    """

    def on_mount(self) -> None:
        self.add_columns("ID", "Family", "Confidence", "Mutations", "Cycle")
        self.cursor_type = "row"
