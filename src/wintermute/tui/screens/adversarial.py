"""adversarial.py — Red Team vs Blue Team. Graceful if Phase 5 absent."""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static, Sparkline, DataTable
from wintermute.tui import theme
from wintermute.tui.widgets.action_log import ActionLog
from wintermute.tui.widgets.config_drawer import ConfigDrawer, FieldDef
from wintermute.tui.hooks import AdversarialHook


ADVERSARIAL_FIELDS = [
    FieldDef("cycles", "Cycles", "10", "int"),
    FieldDef("episodes_per_cycle", "Episodes / Cycle", "500", "int"),
    FieldDef("trades_beta", "TRADES Beta", "1.0", "float"),
    FieldDef("ewc_lambda", "EWC Lambda", "0.4", "float"),
    FieldDef("ppo_lr", "PPO Learning Rate", "3e-4", "float"),
    FieldDef("ppo_epochs", "PPO Epochs", "4", "int"),
]


def _phase5_available() -> bool:
    try:
        import gymnasium  # noqa: F401
        return True
    except ImportError:
        return False


class AdversarialScreen(Vertical):

    DEFAULT_CSS = """
    AdversarialScreen {
        height: 1fr;
        padding: 1;
    }
    #adv-placeholder { height: 100%; content-align: center middle; }
    #adv-body { height: 1fr; }
    #adv-main { width: 1fr; }
    #adv-team-row { layout: horizontal; height: 5; margin-bottom: 1; }
    #adv-charts-row { layout: horizontal; height: 1fr; margin-bottom: 1; }
    #adv-sparks { width: 1fr; margin-right: 1; }
    #adv-episode { width: 35; margin-right: 1; }
    #adv-sidebar { width: 28; }
    #adv-cycles { height: 10; }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._evasion_data: list[float] = []
        self._confidence_data: list[float] = []
        self._hook = None

    def compose(self) -> ComposeResult:
        if not _phase5_available():
            yield Static(
                f"\n\n  [{theme.TEXT_MUTED}]Phase 5 adversarial module not installed.[/]\n\n"
                f"  [{theme.TEXT_MUTED}]Install with:[/]\n"
                f"  [{theme.CYAN}]pip install -e '.[adversarial]'[/]\n\n"
                f"  [{theme.TEXT_MUTED}]Then implement the adversarial training loop[/]\n"
                f"  [{theme.TEXT_MUTED}]per wintermute-phase5-mlx-spec.md[/]",
                id="adv-placeholder")
            return

        with Horizontal(id="adv-body"):
            with Vertical(id="adv-main"):
                with Horizontal(id="adv-team-row"):
                    yield TeamCard(
                        team="RED TEAM — ATTACKER", icon="⚔", color=theme.RED,
                        detail="PPO Agent · lr=1e-4 · γ=0.5 · 256-step rollouts",
                        metric="—", metric_label="evasion rate", id="red-card")
                    yield TeamCard(
                        team="BLUE TEAM — DEFENDER", icon="🛡", color=theme.CYAN,
                        detail="WintermuteMalwareDetector · TRADES β=1.0 · EWC λ=0.4",
                        metric="—", metric_label="adversarial TPR", id="blue-card")

                with Horizontal(id="adv-charts-row"):
                    with Vertical(id="adv-sparks"):
                        yield SparkChart("EVASION RATE", theme.RED, id="evasion-spark")
                        yield SparkChart("DEFENDER CONFIDENCE", theme.CYAN, id="conf-spark")
                    yield ActionLog(id="adv-episode")
                    yield SidebarStats(id="adv-sidebar")

                yield CycleTable(id="adv-cycles")
            yield ConfigDrawer(
                fields=ADVERSARIAL_FIELDS,
                title="ADVERSARIAL CONFIG",
                start_label="START ADVERSARIAL",
                id="adv-drawer",
            )

    def update_cycle(self, cycle: int, metrics: dict) -> None:
        evasion = metrics.get("evasion_rate", 0)
        self._evasion_data.append(evasion)
        self._confidence_data.append(metrics.get("mean_confidence", 0.5))
        try:
            self.query_one("#red-card", TeamCard).update_metric(f"{evasion:.1%}")
            self.query_one("#blue-card", TeamCard).update_metric(
                f"{metrics.get('adv_tpr', 0):.1%}")
            self.query_one("#evasion-spark", SparkChart).set_data(
                self._evasion_data, f"↗ {evasion:.1%}")
            self.query_one("#conf-spark", SparkChart).set_data(
                self._confidence_data,
                f"↘ {metrics.get('mean_confidence', 0):.3f}")
            self.query_one("#adv-cycles", CycleTable).add_cycle(cycle, metrics)
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "drawer-start":
            drawer = self.query_one("#adv-drawer", ConfigDrawer)
            values = drawer.get_values()
            drawer.lock()
            self._hook = AdversarialHook(app=self.app)
            self.run_worker(self._do_adversarial(values), exclusive=True)

    async def _do_adversarial(self, values: dict) -> None:
        """Run adversarial training cycles in background."""
        self.app.call_from_thread(
            self.app._log, "Adversarial training started", "ok"
        )
        # Note: actual orchestrator integration requires model + data loading
        # which is handled by the CLI. For now, the worker structure is in place.
        # Full integration will wire to AdversarialOrchestrator.run_cycle()
        n_cycles = int(values["cycles"])
        self.app.call_from_thread(
            self.app._log, f"Adversarial: {n_cycles} cycles configured", "info"
        )

    def on_worker_state_changed(self, event) -> None:
        if str(event.state) in ("CANCELLED", "ERROR", "SUCCESS"):
            try:
                self.query_one("#adv-drawer", ConfigDrawer).unlock()
            except Exception:
                pass

    def cancel_operation(self) -> None:
        if self._hook:
            self._hook.cancel()


class TeamCard(Static):

    DEFAULT_CSS = f"""
    TeamCard {{
        width: 1fr;
        margin: 0 1;
        padding: 1 2;
        background: {theme.BG_CARD};
        border: solid {theme.BORDER};
    }}
    """

    def __init__(self, team: str, icon: str, color: str,
                 detail: str, metric: str, metric_label: str,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._team = team
        self._icon = icon
        self._color = color
        self._detail = detail
        self._metric = metric
        self._metric_label = metric_label

    def render(self) -> str:
        return (f"  {self._icon}  [bold {self._color}]{self._team}[/]\n"
                f"     [{theme.TEXT_MUTED}]{self._detail}[/]\n"
                f"     [bold {self._color}]{self._metric}[/] "
                f"[{theme.TEXT_MUTED}]{self._metric_label}[/]")

    def update_metric(self, value: str) -> None:
        self._metric = value
        self.refresh()


class SparkChart(Vertical):

    DEFAULT_CSS = f"""
    SparkChart {{
        height: 1fr;
        background: {theme.BG_CARD};
        border: solid {theme.BORDER};
        padding: 1;
        margin-bottom: 1;
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
        yield Sparkline([], id=f"{self.id}-data")

    def set_data(self, data: list[float], current: str) -> None:
        try:
            self.query_one(f"#{self.id}-lbl", Static).update(
                f"[{theme.TEXT_MUTED}]{self._label}[/]  [{self._color}]{current}[/]")
            self.query_one(f"#{self.id}-data", Sparkline).data = data
        except Exception:
            pass


class SidebarStats(Static):

    DEFAULT_CSS = f"""
    SidebarStats {{
        background: {theme.BG_CARD};
        border: solid {theme.BORDER};
        padding: 1 2;
    }}
    """

    def render(self) -> str:
        return (
            f"[bold {theme.TEXT_BRIGHT}]STATUS[/]\n\n"
            f"  [{theme.TEXT_MUTED}]Cycles[/]       [{theme.TEXT_BRIGHT}]0[/]\n"
            f"  [{theme.TEXT_MUTED}]Vault[/]        [{theme.PURPLE}]0[/]\n"
            f"  [{theme.TEXT_MUTED}]NFR[/]          [{theme.GREEN}]—[/]\n"
            f"  [{theme.TEXT_MUTED}]PPO Loss[/]     [{theme.AMBER}]—[/]\n"
            f"  [{theme.TEXT_MUTED}]Clean TPR[/]    [{theme.GREEN}]target ≥0.92[/]\n"
            f"  [{theme.TEXT_MUTED}]Adv. TPR[/]     [{theme.CYAN}]target ≥0.80[/]\n"
            f"  [{theme.TEXT_MUTED}]Evasion[/]      [{theme.RED}]target 0.20-0.50[/]\n")


class CycleTable(DataTable):

    DEFAULT_CSS = f"""
    CycleTable {{
        background: {theme.BG_CARD};
        border: solid {theme.BORDER};
    }}
    """

    def on_mount(self) -> None:
        self.add_columns("Cycle", "Evasion", "PPO Loss",
                         "Adv TPR", "NFR", "Vault", "Time")

    def add_cycle(self, cycle: int, m: dict) -> None:
        self.add_row(
            str(cycle), f"{m.get('evasion_rate', 0):.1%}",
            f"{m.get('ppo_loss', 0):.3f}", f"{m.get('adv_tpr', 0):.1%}",
            f"{m.get('nfr', 0):.3f}", str(m.get("vault_size", 0)),
            m.get("time", "—"))
        self.scroll_end()
