# Wintermute TUI Build Plan

Execute tasks 0–13 in order. Each task: create/edit file → run verify command.

---

### 0. Edit `pyproject.toml`

Add to `[project.optional-dependencies]` after `dev`:

```toml
tui = [
    "textual>=1.0.0",
]
adversarial = [
    "gymnasium>=1.0.0",
    "lief>=0.15.0",
]
```

Change `all` to:

```toml
all = [
    "wintermute[api,mlops,dev,tui,adversarial]",
    "angr>=9.2.0",
]
```

```bash
pip install -e ".[tui]"
python -c "from textual.app import App; print('ok')"
```

---

### 1a. Create `src/wintermute/tui/__init__.py`

```python
"""Wintermute Terminal User Interface."""
```

### 1b. Create `src/wintermute/tui/screens/__init__.py`

```python
"""TUI screen modules."""
```

### 1c. Create `src/wintermute/tui/widgets/__init__.py`

```python
"""TUI widget modules."""
```

### 1d. Create `src/wintermute/tui/theme.py`

```python
"""
theme.py — Wintermute TUI design system.
Color tokens and master Textual CSS stylesheet.
"""

# ── Color tokens ──
BG = "#0a0e14"
BG_PANEL = "#0d1117"
BG_CARD = "#131820"
BG_HOVER = "#1a2030"
BORDER = "#1e2a3a"
BORDER_ACTIVE = "#00e5ff"

TEXT = "#c5cdd8"
TEXT_MUTED = "#5a6577"
TEXT_BRIGHT = "#e8ecf1"

CYAN = "#00e5ff"
GREEN = "#00ff9f"
RED = "#ff3366"
AMBER = "#ffb300"
PURPLE = "#b388ff"

SAFE = GREEN
MALICIOUS = RED
ATTACKER = RED
DEFENDER = CYAN

# ── Master TCSS stylesheet ──
STYLESHEET = f"""
Screen {{
    background: {BG};
}}

.panel {{
    background: {BG_CARD};
    border: solid {BORDER};
    padding: 1 2;
}}

TabbedContent {{
    background: {BG};
}}

TabPane {{
    padding: 1;
}}

Tabs {{
    background: {BG_PANEL};
    border-bottom: solid {BORDER};
}}

Tab {{
    color: {TEXT_MUTED};
    background: {BG_PANEL};
    padding: 1 2;
}}

Tab.-active {{
    color: {TEXT_BRIGHT};
    border-bottom: tall {CYAN};
}}

Tab:hover {{
    color: {TEXT_BRIGHT};
}}

Footer {{
    background: {BG_PANEL};
    color: {TEXT_MUTED};
    border-top: solid {BORDER};
}}

Header {{
    background: {BG_PANEL};
    color: {TEXT_BRIGHT};
}}

DataTable {{
    background: {BG_CARD};
    border: solid {BORDER};
}}

DataTable > .datatable--header {{
    color: {TEXT_MUTED};
    text-style: bold;
    background: {BG_CARD};
}}

DataTable > .datatable--cursor {{
    background: {BG_HOVER};
    color: {TEXT_BRIGHT};
}}

RichLog {{
    background: {BG_CARD};
    border: solid {BORDER};
    padding: 0 1;
}}

Sparkline {{
    margin: 0 1;
}}

ProgressBar {{
    padding: 0 1;
}}

Bar > .bar--bar {{
    color: {CYAN};
    background: {BORDER};
}}

Bar > .bar--complete {{
    color: {GREEN};
}}

Input {{
    background: {BG_CARD};
    border: solid {BORDER};
    color: {TEXT_BRIGHT};
}}

Input:focus {{
    border: solid {BORDER_ACTIVE};
}}

Button {{
    background: {CYAN};
    color: {BG};
    border: none;
    text-style: bold;
}}

Button:hover {{
    background: {GREEN};
}}

.row {{
    layout: horizontal;
    height: auto;
}}

.flex-1 {{
    width: 1fr;
}}
"""
```

```bash
python -c "from wintermute.tui.theme import STYLESHEET, CYAN; print(len(STYLESHEET), 'chars')"
```

---

### 2. Create `src/wintermute/tui/events.py`

```python
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
```

```bash
python -c "from wintermute.tui.events import EpochComplete, ScanProgress; print('ok')"
```

---

### 3a. Create `src/wintermute/tui/widgets/stat_card.py`

```python
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

    def __init__(self, label: str, value: str, subtitle: str = "",
                 accent: str = "", **kwargs) -> None:
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
```

### 3b. Create `src/wintermute/tui/widgets/confidence_bar.py`

```python
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

    def __init__(self, label: str, value: float = 0.0,
                 color: str = "", **kwargs) -> None:
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
```

### 3c. Create `src/wintermute/tui/widgets/action_log.py`

```python
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
        "info": theme.CYAN, "ok": theme.GREEN,
        "warn": theme.AMBER, "error": theme.RED,
    }

    def add_action(self, step: int, action: str, pos: int,
                   conf: float, ok: bool) -> None:
        conf_color = (theme.GREEN if conf < 0.5
                      else theme.AMBER if conf < 0.7
                      else theme.TEXT_MUTED)
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
```

### 3d. Create `src/wintermute/tui/widgets/diff_view.py`

```python
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
```

```bash
python -c "
from wintermute.tui.widgets.stat_card import StatCard
from wintermute.tui.widgets.confidence_bar import ConfidenceBar
from wintermute.tui.widgets.action_log import ActionLog
from wintermute.tui.widgets.diff_view import DiffView
print('all widgets ok')
"
```

---

### 4. Create `src/wintermute/tui/screens/dashboard.py`

Architecture values come from `DetectorConfig` defaults in `src/wintermute/models/fusion.py`.
Family names come from `DEFAULT_FAMILIES` in `src/wintermute/cli.py`.

```python
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
        height: 100%;
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
            yield StatCard("MODEL", "v3.0.0", "MalBERT + GAT + Fusion",
                           accent=theme.CYAN, id="stat-model")
            yield StatCard("CLEAN TPR", "—", "not yet evaluated",
                           accent=theme.GREEN, id="stat-tpr")
            yield StatCard("ADV. TPR", "—", "Phase 5 not trained",
                           accent=theme.CYAN, id="stat-adv-tpr")
            yield StatCard("MACRO F1", "—", "target ≥ 0.90",
                           accent=theme.AMBER, id="stat-f1")
            yield StatCard("VAULT", "0", "no adversarial cycles",
                           accent=theme.PURPLE, id="stat-vault")

        with Horizontal(id="dash-mid-row"):
            yield FamilyChart(id="dash-families")
            yield ArchitecturePanel(id="dash-arch")

        yield DashboardLog(id="dash-log")

    def update_stat(self, stat_id: str, value: str,
                    subtitle: str | None = None) -> None:
        try:
            card = self.query_one(f"#{stat_id}", StatCard)
            card.update_value(value, subtitle)
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
        ("Embedding",  "256-D shared tokens",           theme.TEXT_BRIGHT),
        ("MalBERT",    "6 layers · 8 heads · 1024 FFN", theme.CYAN),
        ("RoPE",       "rotary positional encoding",    theme.CYAN),
        ("GAT",        "3 layers · 4 heads",            theme.GREEN),
        ("Fusion",     "cross-attn · 4 heads",          theme.PURPLE),
        ("Classifier", "Linear(256, num_classes)",      theme.AMBER),
        ("Seq length",  "2048 tokens",                  theme.TEXT_MUTED),
        ("Dropout",    "0.1",                           theme.TEXT_MUTED),
    ]

    def render(self) -> str:
        header = f"[bold {theme.TEXT_BRIGHT}]ARCHITECTURE[/]\n"
        lines = []
        for name, desc, color in self._ROWS:
            lines.append(
                f"  [{color}]▸ {name:<12}[/] [{theme.TEXT_MUTED}]{desc}[/]"
            )
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
        ("Ramnit",        0, theme.RED),
        ("Lollipop",      0, theme.AMBER),
        ("Kelihos_ver3",  0, theme.CYAN),
        ("Vundo",         0, theme.PURPLE),
        ("Simda",         0, theme.GREEN),
        ("Tracur",        0, theme.AMBER),
        ("Kelihos_ver1",  0, theme.CYAN),
        ("Obfuscator.ACY",0, theme.PURPLE),
        ("Gatak",         0, theme.AMBER),
    ]

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
        colors = {"info": theme.CYAN, "ok": theme.GREEN,
                  "warn": theme.AMBER, "error": theme.RED}
        color = colors.get(level, theme.TEXT)
        ts = datetime.now().strftime("%H:%M:%S")
        line = Text()
        line.append(f"  {ts} ", style=theme.TEXT_MUTED)
        line.append("● ", style=color)
        line.append(text, style=theme.TEXT)
        self.write(line)
```

```bash
python -c "from wintermute.tui.screens.dashboard import DashboardScreen; print('ok')"
```

---

### 5. Create `src/wintermute/tui/screens/scan.py`

Uses real scan pipeline: `WintermuteMalwareDetector.load()`, `HeadlessDisassembler`, `read_asm_file()`. Manifest + vocab SHA256 verification. Worker thread keeps TUI responsive.

```python
"""
scan.py — File scanning screen.
Left: disassembly log. Right: verdict + confidence bars.
Uses real inference pipeline from wintermute.models.fusion and wintermute.data.*.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, RichLog, Input, Button
from rich.text import Text

from wintermute.tui import theme
from wintermute.tui.widgets.confidence_bar import ConfidenceBar


class ScanScreen(Vertical):

    DEFAULT_CSS = """
    ScanScreen {
        height: 100%;
        padding: 1;
    }
    #scan-input-row {
        height: 3;
        layout: horizontal;
        margin-bottom: 1;
    }
    #scan-path {
        width: 1fr;
    }
    #scan-btn {
        width: 16;
        margin-left: 1;
    }
    #scan-body {
        layout: horizontal;
        height: 1fr;
    }
    #scan-disasm {
        width: 1fr;
        margin-right: 1;
    }
    #scan-verdict-col {
        width: 38;
    }
    """

    def compose(self) -> ComposeResult:
        with Horizontal(id="scan-input-row"):
            yield Input(placeholder="Path to .exe / .asm file", id="scan-path")
            yield Button("⊕ SCAN", id="scan-btn", variant="primary")
        with Horizontal(id="scan-body"):
            yield DisassemblyLog(id="scan-disasm")
            yield VerdictPanel(id="scan-verdict-col")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "scan-btn":
            path = self.query_one("#scan-path", Input).value.strip()
            if path:
                self._start_scan(path)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "scan-path" and event.value.strip():
            self._start_scan(event.value.strip())

    def _start_scan(self, path: str) -> None:
        disasm = self.query_one("#scan-disasm", DisassemblyLog)
        verdict = self.query_one("#scan-verdict-col", VerdictPanel)
        disasm.clear()
        verdict.reset()
        disasm.add_header(f"Scanning: {path}")
        self.run_worker(self._do_scan(path), exclusive=True)

    async def _do_scan(self, path: str) -> None:
        """Real scan pipeline in background thread. Same flow as cli.py scan."""
        import mlx.core as mx
        from wintermute.models.fusion import WintermuteMalwareDetector
        from wintermute.data.tokenizer import read_asm_file

        disasm = self.query_one("#scan-disasm", DisassemblyLog)
        verdict = self.query_one("#scan-verdict-col", VerdictPanel)

        target = Path(path)
        if not target.exists():
            self.app.call_from_thread(
                disasm.add_line, "", f"[ERROR] File not found: {path}", "error")
            return

        # 1. Extract opcodes
        self.app.call_from_thread(disasm.add_line, "", "Extracting opcodes...", "info")

        if target.suffix.lower() == ".asm":
            opcodes = read_asm_file(str(target))
        else:
            try:
                from wintermute.data.disassembler import HeadlessDisassembler
                result = HeadlessDisassembler(str(target)).extract()
                opcodes = result.sequence
            except ImportError:
                self.app.call_from_thread(
                    disasm.add_line, "", "r2pipe not available", "warn")
                return

        if not opcodes:
            self.app.call_from_thread(
                disasm.add_line, "", "No opcodes extracted", "warn")
            return

        # Show disassembly (cap display at 200 lines)
        for i, op in enumerate(opcodes[:200]):
            addr = f"0x{0x401000 + i * 4:08x}"
            style = ("call" if op.startswith("call") or op == "ret"
                     else "jump" if op.startswith("j")
                     else "nop" if op == "nop" else "normal")
            self.app.call_from_thread(disasm.add_line, addr, op, style)

        self.app.call_from_thread(
            disasm.add_line, "", f"{len(opcodes)} instructions total", "info")

        # 2. Load model + inference (same as cli.py scan command)
        import time
        self.app.call_from_thread(disasm.add_line, "", "Loading model...", "info")

        try:
            vocab_path = Path("data/processed/vocab.json")
            if not vocab_path.exists():
                self.app.call_from_thread(
                    disasm.add_line, "", "vocab.json not found", "error")
                return

            with open(vocab_path) as f:
                stoi = json.load(f)
            vocab_sha = hashlib.sha256(
                json.dumps(stoi, sort_keys=True).encode()).hexdigest()

            detector = WintermuteMalwareDetector.load(
                "malware_detector.safetensors",
                "malware_detector_manifest.json",
                vocab_sha256=vocab_sha,
            )
            WintermuteMalwareDetector.cast_to_bf16(detector)
            detector.eval()

            max_seq = detector.config.max_seq_length
            unk = stoi.get("<UNK>", 1)
            pad = stoi.get("<PAD>", 0)
            ids = [stoi.get(op, unk) for op in opcodes[:max_seq]]
            ids += [pad] * (max_seq - len(ids))

            t0 = time.perf_counter()
            logits = detector(mx.array([ids]))
            probs = mx.softmax(logits, axis=1)
            mx.eval(probs)
            inference_ms = (time.perf_counter() - t0) * 1000

            pred = int(mx.argmax(probs, axis=1).item())
            n_classes = probs.shape[1]

            from wintermute.cli import DEFAULT_FAMILIES
            family_probs = {}
            for i in range(n_classes):
                name = DEFAULT_FAMILIES.get(str(i), f"Class {i}")
                family_probs[name] = probs[0, i].item()

            is_malicious = pred != 0
            confidence = probs[0, pred].item()
            family = DEFAULT_FAMILIES.get(str(pred), f"Class {pred}")

            self.app.call_from_thread(
                verdict.show_result,
                is_malicious, confidence, family, family_probs,
                len(opcodes), inference_ms,
            )

        except FileNotFoundError as e:
            self.app.call_from_thread(
                disasm.add_line, "", f"Model not found: {e}", "error")
        except Exception as e:
            self.app.call_from_thread(
                disasm.add_line, "", f"Inference error: {e}", "error")


class DisassemblyLog(RichLog):

    DEFAULT_CSS = f"""
    DisassemblyLog {{
        background: {theme.BG_CARD};
        border: solid {theme.BORDER};
        padding: 0;
    }}
    """

    _STYLES = {
        "normal": theme.TEXT_BRIGHT, "call": theme.PURPLE,
        "jump": theme.GREEN, "nop": theme.TEXT_MUTED,
        "info": theme.CYAN, "warn": theme.AMBER, "error": theme.RED,
    }

    def add_header(self, text: str) -> None:
        line = Text()
        line.append(f"  {text}", style=f"bold {theme.CYAN}")
        self.write(line)
        self.write(Text(f"  {'─' * 50}", style=theme.BORDER))

    def add_line(self, addr: str, instruction: str,
                 style: str = "normal") -> None:
        color = self._STYLES.get(style, theme.TEXT)
        line = Text()
        if addr:
            line.append(f"  {addr}  ", style=theme.TEXT_MUTED)
        else:
            line.append("           ", style=theme.TEXT_MUTED)
        line.append(instruction, style=color)
        self.write(line)


class VerdictPanel(Vertical):

    DEFAULT_CSS = f"""
    VerdictPanel {{
        background: {theme.BG_CARD};
        border: solid {theme.BORDER};
        padding: 1 2;
    }}
    """

    def compose(self) -> ComposeResult:
        yield Static(f"[{theme.TEXT_MUTED}]Awaiting scan...[/]", id="verdict-label")
        yield ConfidenceBar("Malicious", 0.0, theme.RED, id="conf-mal")
        yield ConfidenceBar("Safe", 0.0, theme.GREEN, id="conf-safe")
        yield Static("", id="verdict-detail")

    def reset(self) -> None:
        self.query_one("#verdict-label", Static).update(
            f"[{theme.TEXT_MUTED}]Analyzing...[/]")
        self.query_one("#conf-mal", ConfidenceBar).update_value(0.0)
        self.query_one("#conf-safe", ConfidenceBar).update_value(0.0)
        self.query_one("#verdict-detail", Static).update("")

    def show_result(self, is_malicious: bool, confidence: float,
                    family: str, family_probs: dict,
                    n_instructions: int, inference_ms: float) -> None:
        icon = "🚨" if is_malicious else "✅"
        label = "MALICIOUS" if is_malicious else "SAFE"
        color = theme.RED if is_malicious else theme.GREEN

        self.query_one("#verdict-label", Static).update(
            f"\n  {icon}  [bold {color}]{label}[/]\n"
            f"  [{theme.TEXT_MUTED}]family: {family}[/]\n")

        mal_conf = confidence if is_malicious else 1.0 - confidence
        self.query_one("#conf-mal", ConfidenceBar).update_value(mal_conf)
        self.query_one("#conf-safe", ConfidenceBar).update_value(1.0 - mal_conf)

        lines = [
            f"\n[{theme.TEXT_MUTED}]{'─' * 30}[/]",
            f"  [{theme.TEXT_MUTED}]Instructions[/]  [{theme.TEXT_BRIGHT}]{n_instructions}[/]",
            f"  [{theme.TEXT_MUTED}]Inference[/]     [{theme.TEXT_BRIGHT}]{inference_ms:.1f}ms[/]",
            f"  [{theme.TEXT_MUTED}]Model[/]         [{theme.CYAN}]v3.0.0[/]",
        ]
        if family_probs:
            lines.append(f"\n[{theme.TEXT_MUTED}]FAMILY SCORES[/]")
            for name, prob in sorted(family_probs.items(), key=lambda x: -x[1])[:5]:
                bar_len = int(prob * 20)
                bar = "█" * bar_len
                lines.append(
                    f"  [{theme.TEXT_MUTED}]{name:<16}[/] "
                    f"[{theme.PURPLE}]{bar}[/] "
                    f"[{theme.TEXT_MUTED}]{prob * 100:.1f}%[/]")
        self.query_one("#verdict-detail", Static).update("\n".join(lines))
```

```bash
python -c "from wintermute.tui.screens.scan import ScanScreen; print('ok')"
```

---

### 6. Create `src/wintermute/tui/screens/training.py`

Phase A = encoder frozen, Phase B = full finetune. Matches `JointTrainer` two-phase structure.

```python
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
```

```bash
python -c "from wintermute.tui.screens.training import TrainingScreen; print('ok')"
```

---

### 7. Create `src/wintermute/tui/screens/adversarial.py`

Shows "Phase 5 not installed" if gymnasium is missing. No crash.

```python
"""adversarial.py — Red Team vs Blue Team. Graceful if Phase 5 absent."""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, Sparkline, DataTable
from wintermute.tui import theme
from wintermute.tui.widgets.stat_card import StatCard
from wintermute.tui.widgets.action_log import ActionLog


def _phase5_available() -> bool:
    try:
        import gymnasium  # noqa: F401
        return True
    except ImportError:
        return False


class AdversarialScreen(Vertical):

    DEFAULT_CSS = """
    AdversarialScreen {
        height: 100%;
        padding: 1;
    }
    #adv-placeholder { height: 100%; content-align: center middle; }
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
```

```bash
python -c "from wintermute.tui.screens.adversarial import AdversarialScreen; print('ok')"
```

---

### 8. Create `src/wintermute/tui/screens/vault.py`

```python
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
```

```bash
python -c "from wintermute.tui.screens.vault import VaultScreen; print('ok')"
```

---

### 9. Create `src/wintermute/tui/app.py`

```python
"""app.py — Main Wintermute TUI. Launch: wintermute tui. Keys: 1-5 switch tabs, q quit."""

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, TabbedContent, TabPane

from wintermute.tui import theme
from wintermute.tui.screens.dashboard import DashboardScreen
from wintermute.tui.screens.scan import ScanScreen
from wintermute.tui.screens.training import TrainingScreen
from wintermute.tui.screens.adversarial import AdversarialScreen
from wintermute.tui.screens.vault import VaultScreen
from wintermute.tui.events import EpochComplete, AdversarialCycleEnd, AdversarialEpisodeStep


class WintermuteApp(App):

    TITLE = "WINTERMUTE v3.0"
    SUB_TITLE = "MLX-native static malware classifier"
    CSS = theme.STYLESHEET

    BINDINGS = [
        Binding("1", "switch_tab('dashboard')", "Dashboard", show=True),
        Binding("2", "switch_tab('scan')", "Scan", show=True),
        Binding("3", "switch_tab('training')", "Training", show=True),
        Binding("4", "switch_tab('adversarial')", "Adversarial", show=True),
        Binding("5", "switch_tab('vault')", "Vault", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(initial="dashboard"):
            with TabPane("◉ DASHBOARD", id="dashboard"):
                yield DashboardScreen()
            with TabPane("⊕ SCAN", id="scan"):
                yield ScanScreen()
            with TabPane("◈ TRAINING", id="training"):
                yield TrainingScreen()
            with TabPane("⚔ ADVERSARIAL", id="adversarial"):
                yield AdversarialScreen()
            with TabPane("▣ VAULT", id="vault"):
                yield VaultScreen()
        yield Footer()

    def action_switch_tab(self, tab_id: str) -> None:
        self.query_one(TabbedContent).active = tab_id

    def on_mount(self) -> None:
        self._log("TUI initialized", "ok")

    # ── Message routing ──

    def on_epoch_complete(self, event: EpochComplete) -> None:
        try:
            self.query_one(TrainingScreen).handle_epoch(event)
        except Exception:
            pass
        self._log(f"Epoch {event.epoch} ({event.phase}) — "
                  f"loss={event.loss:.4f} val_acc={event.val_acc:.1%}", "info")

    def on_adversarial_cycle_end(self, event: AdversarialCycleEnd) -> None:
        try:
            self.query_one(AdversarialScreen).update_cycle(
                event.cycle, event.metrics)
        except Exception:
            pass
        self._log(f"Adversarial cycle {event.cycle} — "
                  f"evasion={event.metrics.get('evasion_rate', 0):.1%}", "info")

    def on_adversarial_episode_step(self, event: AdversarialEpisodeStep) -> None:
        try:
            from wintermute.tui.widgets.action_log import ActionLog
            log = self.query_one("#adv-episode", ActionLog)
            log.add_action(event.step, event.action, event.position,
                           event.confidence, event.valid)
        except Exception:
            pass

    def _log(self, text: str, level: str = "info") -> None:
        try:
            dash_log = self.query_one("#dash-log")
            if hasattr(dash_log, "add_entry"):
                dash_log.add_entry(text, level)
        except Exception:
            pass


def run() -> None:
    WintermuteApp().run()


if __name__ == "__main__":
    run()
```

```bash
python -c "from wintermute.tui.app import WintermuteApp; print(WintermuteApp.TITLE)"
```

---

### 10. Create `src/wintermute/tui/hooks.py`

```python
"""
hooks.py — Callback bridge: training loops → TUI messages.

Usage in JointTrainer:
    if self._tui_hook:
        self._tui_hook.on_epoch(epoch, phase, loss, train_acc, val_acc, f1, elapsed)

Usage in AdversarialOrchestrator:
    if self._tui_hook:
        self._tui_hook.on_episode_step(step, action, pos, conf, ok)
        self._tui_hook.on_cycle_end(cycle, metrics_dict)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wintermute.tui.app import WintermuteApp


@dataclass
class TrainingHook:
    """Pass to JointTrainer. It calls on_epoch() after each epoch."""

    app: WintermuteApp | None = None

    def on_epoch(self, epoch: int, phase: str, loss: float,
                 train_acc: float, val_acc: float, f1: float,
                 elapsed: float) -> None:
        if self.app is None:
            return
        from wintermute.tui.events import EpochComplete
        self.app.call_from_thread(
            self.app.post_message,
            EpochComplete(epoch=epoch, phase=phase, loss=loss,
                          train_acc=train_acc, val_acc=val_acc,
                          f1=f1, elapsed=elapsed))

    def on_log(self, text: str, level: str = "info") -> None:
        if self.app is None:
            return
        self.app.call_from_thread(self.app._log, text, level)


@dataclass
class AdversarialHook:
    """Pass to AdversarialOrchestrator. Calls on_episode_step() and on_cycle_end()."""

    app: WintermuteApp | None = None

    def on_episode_step(self, step: int, action: str, pos: int,
                        conf: float, ok: bool) -> None:
        if self.app is None:
            return
        from wintermute.tui.events import AdversarialEpisodeStep
        self.app.call_from_thread(
            self.app.post_message,
            AdversarialEpisodeStep(step=step, action=action,
                                    position=pos, confidence=conf,
                                    valid=ok))

    def on_cycle_end(self, cycle: int, metrics: dict) -> None:
        if self.app is None:
            return
        from wintermute.tui.events import AdversarialCycleEnd
        self.app.call_from_thread(
            self.app.post_message,
            AdversarialCycleEnd(cycle=cycle, metrics=metrics))

    def on_log(self, text: str, level: str = "info") -> None:
        if self.app is None:
            return
        self.app.call_from_thread(self.app._log, text, level)
```

```bash
python -c "from wintermute.tui.hooks import TrainingHook, AdversarialHook; print('ok')"
```

---

### 11. Edit `src/wintermute/cli.py`

Add this command **before** the `if __name__ == "__main__":` block:

```python
# ═══════════════════════════════════════════════════════════════════════════
# wintermute tui
# ═══════════════════════════════════════════════════════════════════════════
@app.command()
def tui() -> None:
    """Launch the Wintermute Terminal User Interface."""
    try:
        from wintermute.tui.app import run
        run()
    except ImportError:
        typer.echo("TUI requires the [tui] extra:")
        typer.echo("  pip install -e '.[tui]'")
        raise typer.Exit(1)
```

```bash
wintermute --help | grep -i tui
```

---

### 12. Create `tests/test_tui.py`

Flat file in `tests/` matching existing convention.

```python
"""test_tui.py — Tests for wintermute.tui"""

import pytest
textual = pytest.importorskip("textual")


class TestTheme:
    def test_color_tokens_exist(self):
        from wintermute.tui import theme
        assert theme.CYAN == "#00e5ff"
        assert theme.RED == "#ff3366"
        assert theme.GREEN == "#00ff9f"
        assert theme.BG == "#0a0e14"

    def test_stylesheet_nonempty(self):
        from wintermute.tui.theme import STYLESHEET
        assert len(STYLESHEET) > 200
        assert "Screen" in STYLESHEET


class TestEvents:
    def test_epoch_complete(self):
        from wintermute.tui.events import EpochComplete
        e = EpochComplete(epoch=1, phase="A", loss=0.5,
                          train_acc=0.8, val_acc=0.75, f1=0.77, elapsed=3.2)
        assert e.epoch == 1 and e.phase == "A"

    def test_scan_progress(self):
        from wintermute.tui.events import ScanProgress
        s = ScanProgress("disassemble", {"n_ops": 100})
        assert s.phase == "disassemble"

    def test_adversarial_cycle_end(self):
        from wintermute.tui.events import AdversarialCycleEnd
        a = AdversarialCycleEnd(cycle=3, metrics={"evasion_rate": 0.3})
        assert a.cycle == 3


class TestWidgets:
    def test_stat_card_render(self):
        from wintermute.tui.widgets.stat_card import StatCard
        from wintermute.tui import theme
        card = StatCard("METRIC", "42%", "subtitle", accent=theme.GREEN)
        assert "METRIC" in card.render() and "42%" in card.render()

    def test_stat_card_update(self):
        from wintermute.tui.widgets.stat_card import StatCard
        card = StatCard("X", "0")
        card.update_value("99%", "new sub")
        assert card._value == "99%"

    def test_confidence_bar_render(self):
        from wintermute.tui.widgets.confidence_bar import ConfidenceBar
        bar = ConfidenceBar("Malicious", 0.95, "#ff3366")
        assert "95.0%" in bar.render()

    def test_confidence_bar_clamps(self):
        from wintermute.tui.widgets.confidence_bar import ConfidenceBar
        bar = ConfidenceBar("Test", 0.5)
        bar.update_value(1.5)
        assert bar._value == 1.0
        bar.update_value(-0.5)
        assert bar._value == 0.0

    def test_diff_view_render(self):
        from wintermute.tui.widgets.diff_view import DiffView
        diff = DiffView(lines=[("same", "push ebp"), ("del", "xor eax, eax"),
                                ("add", "sub eax, eax")])
        assert "push ebp" in diff.render()

    def test_diff_view_empty(self):
        from wintermute.tui.widgets.diff_view import DiffView
        assert "Select" in DiffView().render()


class TestApp:
    def test_app_creates(self):
        from wintermute.tui.app import WintermuteApp
        assert WintermuteApp().title == "WINTERMUTE v3.0"

    def test_app_css_loaded(self):
        from wintermute.tui.app import WintermuteApp
        assert len(WintermuteApp().CSS) > 200


class TestHooks:
    def test_training_hook_no_app(self):
        from wintermute.tui.hooks import TrainingHook
        TrainingHook().on_epoch(1, "B", 0.1, 0.9, 0.85, 0.87, 2.0)

    def test_adversarial_hook_no_app(self):
        from wintermute.tui.hooks import AdversarialHook
        hook = AdversarialHook()
        hook.on_episode_step(1, "nop_insert", 5, 0.8, True)
        hook.on_cycle_end(1, {"evasion_rate": 0.3})


class TestCLI:
    def test_tui_in_help(self):
        from typer.testing import CliRunner
        from wintermute.cli import app
        result = CliRunner().invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "tui" in result.output.lower()
```

```bash
pytest tests/test_tui.py -v
```

---

### 13. Manual verification

```bash
pip install -e ".[tui]"
wintermute tui
```

Check:

- Header: "WINTERMUTE v3.0" with clock
- 5 tabs work with keys 1-5
- q quits
- Dashboard: stat cards, architecture panel, family chart, activity log
- Scan: input + button, "Awaiting scan..."
- Training: status bar, empty sparklines, empty epoch table
- Adversarial: shows battle view OR "Phase 5 not installed"
- Vault: empty table + "Select a vault entry"

---

## File creation summary

```
NEW FILES (17):
  src/wintermute/tui/__init__.py
  src/wintermute/tui/theme.py
  src/wintermute/tui/events.py
  src/wintermute/tui/app.py
  src/wintermute/tui/hooks.py
  src/wintermute/tui/screens/__init__.py
  src/wintermute/tui/screens/dashboard.py
  src/wintermute/tui/screens/scan.py
  src/wintermute/tui/screens/training.py
  src/wintermute/tui/screens/adversarial.py
  src/wintermute/tui/screens/vault.py
  src/wintermute/tui/widgets/__init__.py
  src/wintermute/tui/widgets/stat_card.py
  src/wintermute/tui/widgets/confidence_bar.py
  src/wintermute/tui/widgets/action_log.py
  src/wintermute/tui/widgets/diff_view.py
  tests/test_tui.py

EDITED FILES (2):
  pyproject.toml              (add tui + adversarial extras)
  src/wintermute/cli.py       (add tui command)
```

## Codebase facts used

```
pyproject.toml version: "2.0.0"
WintermuteMalwareDetector.VERSION: "3.0.0"
Entry point: wintermute = "wintermute.cli:app" (Typer)
Existing extras: api, mlops, dev, all

DetectorConfig defaults:
  vocab_size=512, dims=256, num_heads=8, num_layers=6,
  mlp_dims=1024, dropout=0.1, max_seq_length=2048,
  gat_layers=3, gat_heads=4, num_fusion_heads=4, num_classes=2

Scan flow: WintermuteMalwareDetector.load(model, manifest, vocab_sha256=sha)
Training: JointTrainer(cfg, data_dir, overrides, pretrained).train()
  Phase A = encoder frozen, Phase B = full finetune
Families: Ramnit, Lollipop, Kelihos_ver3, Vundo, Simda,
          Tracur, Kelihos_ver1, Obfuscator.ACY, Gatak
Tests: flat in tests/ (test_cli.py, test_fusion.py, etc.)
Phase 5 adversarial: spec only, not yet implemented
```
