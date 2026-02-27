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
from textual.widgets import Static, RichLog, Button
from rich.text import Text

from wintermute.tui import theme
from wintermute.tui.widgets.confidence_bar import ConfidenceBar
from wintermute.tui.widgets.config_drawer import ConfigDrawer, FieldDef


SCAN_FIELDS = [
    FieldDef("file_path", "File Path", "", "str"),
    FieldDef("family", "Family Detection", "off", "switch"),
    FieldDef("model_path", "Model Path", "malware_detector.safetensors", "str"),
]


class ScanScreen(Vertical):
    DEFAULT_CSS = """
    ScanScreen {
        height: 1fr;
        padding: 1;
    }
    #scan-body {
        layout: horizontal;
        height: 1fr;
    }
    #scan-main {
        width: 1fr;
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
        with Horizontal(id="scan-body"):
            with Vertical(id="scan-main"):
                yield DisassemblyLog(id="scan-disasm")
                yield VerdictPanel(id="scan-verdict-col")
            yield ConfigDrawer(
                fields=SCAN_FIELDS,
                title="SCAN CONFIG",
                start_label="SCAN",
                id="scan-drawer",
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "drawer-start":
            drawer = self.query_one("#scan-drawer", ConfigDrawer)
            values = drawer.get_values()
            path = values.get("file_path", "").strip()
            if path:
                self._start_scan(path)

    def cancel_operation(self) -> None:
        pass  # Scans are fast enough that cancellation isn't needed

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
                disasm.add_line, "", f"[ERROR] File not found: {path}", "error"
            )
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
                self.app.call_from_thread(disasm.add_line, "", "r2pipe not available", "warn")
                return

        if not opcodes:
            self.app.call_from_thread(disasm.add_line, "", "No opcodes extracted", "warn")
            return

        # Show disassembly (cap display at 200 lines)
        for i, op in enumerate(opcodes[:200]):
            addr = f"0x{0x401000 + i * 4:08x}"
            style = (
                "call"
                if op.startswith("call") or op == "ret"
                else "jump"
                if op.startswith("j")
                else "nop"
                if op == "nop"
                else "normal"
            )
            self.app.call_from_thread(disasm.add_line, addr, op, style)

        self.app.call_from_thread(disasm.add_line, "", f"{len(opcodes)} instructions total", "info")

        # 2. Load model + inference (same as cli.py scan command)
        import time

        self.app.call_from_thread(disasm.add_line, "", "Loading model...", "info")

        try:
            vocab_path = Path("data/processed/vocab.json")
            if not vocab_path.exists():
                self.app.call_from_thread(disasm.add_line, "", "vocab.json not found", "error")
                return

            with open(vocab_path) as f:
                stoi = json.load(f)
            vocab_sha = hashlib.sha256(json.dumps(stoi, sort_keys=True).encode()).hexdigest()

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
                is_malicious,
                confidence,
                family,
                family_probs,
                len(opcodes),
                inference_ms,
            )

        except FileNotFoundError as e:
            self.app.call_from_thread(disasm.add_line, "", f"Model not found: {e}", "error")
        except Exception as e:
            self.app.call_from_thread(disasm.add_line, "", f"Inference error: {e}", "error")


class DisassemblyLog(RichLog):
    DEFAULT_CSS = f"""
    DisassemblyLog {{
        background: {theme.BG_CARD};
        border: solid {theme.BORDER};
        padding: 0;
    }}
    """

    _STYLES = {
        "normal": theme.TEXT_BRIGHT,
        "call": theme.PURPLE,
        "jump": theme.GREEN,
        "nop": theme.TEXT_MUTED,
        "info": theme.CYAN,
        "warn": theme.AMBER,
        "error": theme.RED,
    }

    def add_header(self, text: str) -> None:
        line = Text()
        line.append(f"  {text}", style=f"bold {theme.CYAN}")
        self.write(line)
        self.write(Text(f"  {'─' * 50}", style=theme.BORDER))

    def add_line(self, addr: str, instruction: str, style: str = "normal") -> None:
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
        self.query_one("#verdict-label", Static).update(f"[{theme.TEXT_MUTED}]Analyzing...[/]")
        self.query_one("#conf-mal", ConfidenceBar).update_value(0.0)
        self.query_one("#conf-safe", ConfidenceBar).update_value(0.0)
        self.query_one("#verdict-detail", Static).update("")

    def show_result(
        self,
        is_malicious: bool,
        confidence: float,
        family: str,
        family_probs: dict,
        n_instructions: int,
        inference_ms: float,
    ) -> None:
        icon = "🚨" if is_malicious else "✅"
        label = "MALICIOUS" if is_malicious else "SAFE"
        color = theme.RED if is_malicious else theme.GREEN

        self.query_one("#verdict-label", Static).update(
            f"\n  {icon}  [bold {color}]{label}[/]\n  [{theme.TEXT_MUTED}]family: {family}[/]\n"
        )

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
                    f"[{theme.TEXT_MUTED}]{prob * 100:.1f}%[/]"
                )
        self.query_one("#verdict-detail", Static).update("\n".join(lines))
