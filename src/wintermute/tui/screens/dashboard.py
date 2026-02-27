"""dashboard.py — System overview screen."""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, RichLog
from rich.text import Text
from wintermute.tui import theme
from wintermute.tui.widgets.stat_card import StatCard


class DashboardScreen(Vertical):
    DEFAULT_CSS = """
    DashboardScreen {
        height: 1fr;
        padding: 1;
    }
    #dash-stats-row {
        layout: horizontal;
        height: auto;
        margin-bottom: 1;
    }
    #dash-stats-row StatCard {
        width: 1fr;
        margin: 0 1;
    }
    #dash-mid-row {
        layout: horizontal;
        height: 1fr;
        margin-bottom: 1;
    }
    #dash-arch {
        width: 45;
        margin-left: 1;
    }
    #dash-families {
        width: 1fr;
    }
    #dash-log {
        height: 10;
    }
    """

    def compose(self) -> ComposeResult:
        with Horizontal(id="dash-stats-row"):
            yield StatCard(
                "MODEL", "v3.0.0", "MalBERT + GAT + Fusion", accent=theme.CYAN, id="stat-model"
            )
            yield StatCard("CLEAN TPR", "—", "not yet evaluated", accent=theme.GREEN, id="stat-tpr")
            yield StatCard(
                "ADV. TPR", "—", "Phase 5 not trained", accent=theme.CYAN, id="stat-adv-tpr"
            )
            yield StatCard("MACRO F1", "—", "target ≥ 0.90", accent=theme.AMBER, id="stat-f1")
            yield StatCard(
                "VAULT", "0", "no adversarial cycles", accent=theme.PURPLE, id="stat-vault"
            )

        with Horizontal(id="dash-mid-row"):
            yield FamilyChart(id="dash-families")
            yield ArchitecturePanel(id="dash-arch")

        yield DashboardLog(id="dash-log")

    def update_stat(self, stat_id: str, value: str, subtitle: str | None = None) -> None:
        try:
            card = self.query_one(f"#{stat_id}", StatCard)
            card.update_value(value, subtitle)
        except Exception:
            pass

    def update_family_chart(self, counts: dict[str, int]) -> None:
        try:
            chart = self.query_one("#dash-families", FamilyChart)
            chart.update_counts(counts)
        except Exception:
            pass


class ArchitecturePanel(Static):
    """Values from DetectorConfig defaults in models/fusion.py."""

    DEFAULT_CSS = f"""
    ArchitecturePanel {{
        background: {theme.BG_CARD};
        border: solid {theme.BORDER};
        padding: 1 2;
    }}
    """

    _ROWS = [
        ("Embedding", "256-D shared tokens", theme.TEXT_BRIGHT),
        ("MalBERT", "6 layers · 8 heads · 1024 FFN", theme.CYAN),
        ("RoPE", "rotary positional encoding", theme.CYAN),
        ("GAT", "3 layers · 4 heads", theme.GREEN),
        ("Fusion", "cross-attn · 4 heads", theme.PURPLE),
        ("Classifier", "Linear(256, num_classes)", theme.AMBER),
        ("Seq length", "2048 tokens", theme.TEXT_MUTED),
        ("Dropout", "0.1", theme.TEXT_MUTED),
    ]

    def render(self) -> str:
        header = f"[bold {theme.TEXT_BRIGHT}]ARCHITECTURE[/]\n"
        lines = []
        for name, desc, color in self._ROWS:
            lines.append(f"  [{color}]▸ {name:<12}[/] [{theme.TEXT_MUTED}]{desc}[/]")
        return header + "\n".join(lines)


class FamilyChart(Static):
    """Family names from DEFAULT_FAMILIES in cli.py. Counts are placeholder."""

    DEFAULT_CSS = f"""
    FamilyChart {{
        background: {theme.BG_CARD};
        border: solid {theme.BORDER};
        padding: 1 2;
    }}
    """

    _FAMILIES = [
        ("Ramnit", 0, theme.RED),
        ("Lollipop", 0, theme.AMBER),
        ("Kelihos_ver3", 0, theme.CYAN),
        ("Vundo", 0, theme.PURPLE),
        ("Simda", 0, theme.GREEN),
        ("Tracur", 0, theme.AMBER),
        ("Kelihos_ver1", 0, theme.CYAN),
        ("Obfuscator.ACY", 0, theme.PURPLE),
        ("Gatak", 0, theme.AMBER),
    ]

    def update_counts(self, counts: dict[str, int]) -> None:
        for i, (name, _, color) in enumerate(self._FAMILIES):
            if name in counts:
                self._FAMILIES[i] = (name, counts[name], color)
        self.refresh()

    def render(self) -> str:
        header = f"[bold {theme.TEXT_BRIGHT}]FAMILY DISTRIBUTION[/]\n"
        max_count = max((f[1] for f in self._FAMILIES), default=1) or 1
        lines = []
        for name, count, color in self._FAMILIES:
            bar_len = int((count / max_count) * 25) if count > 0 else 0
            bar = "█" * bar_len if bar_len > 0 else "░"
            lines.append(
                f"  [{theme.TEXT_MUTED}]{name:<16}[/] [{color}]{bar}[/] "
                f"[{theme.TEXT_MUTED}]{count}[/]"
            )
        return header + "\n".join(lines)


class DashboardLog(RichLog):
    DEFAULT_CSS = f"""
    DashboardLog {{
        background: {theme.BG_CARD};
        border: solid {theme.BORDER};
        padding: 0 1;
        height: 10;
    }}
    """

    def on_mount(self) -> None:
        self.write(Text("ACTIVITY LOG", style=f"bold {theme.TEXT_BRIGHT}"))
        self.write(Text())

    def add_entry(self, text: str, level: str = "info") -> None:
        from datetime import datetime

        colors = {"info": theme.CYAN, "ok": theme.GREEN, "warn": theme.AMBER, "error": theme.RED}
        color = colors.get(level, theme.TEXT)
        ts = datetime.now().strftime("%H:%M:%S")
        line = Text()
        line.append(f"  {ts} ", style=theme.TEXT_MUTED)
        line.append("● ", style=color)
        line.append(text, style=theme.TEXT)
        self.write(line)
