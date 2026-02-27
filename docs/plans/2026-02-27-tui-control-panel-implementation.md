# TUI Control Panel Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform the Wintermute TUI from a read-only viewer into a full interactive control panel where users configure parameters, launch operations, and monitor results — all from the terminal UI.

**Architecture:** Config Drawers on each screen (collapsible side panels with form fields), background workers for long-running operations, shared engine classes between TUI and CLI via hook callbacks. One new Pipeline screen for data operations. Global status bar for background task visibility.

**Tech Stack:** Textual (TUI framework), OmegaConf (config loading), MLX (ML backend), existing hook system (TrainingHook, AdversarialHook)

**Design doc:** `docs/plans/2026-02-27-tui-control-panel-design.md`

---

## Task 1: Add New Event Types

**Files:**
- Modify: `src/wintermute/tui/events.py`
- Test: `tests/test_tui.py`

**Step 1: Write failing tests for new events**

Add to `tests/test_tui.py` inside the `TestEvents` class:

```python
def test_pipeline_progress(self):
    from wintermute.tui.events import PipelineProgress
    e = PipelineProgress(operation="build", progress=0.5, message="Processing file 10/20")
    assert e.operation == "build"
    assert e.progress == 0.5
    assert e.message == "Processing file 10/20"

def test_evaluation_complete(self):
    from wintermute.tui.events import EvaluationComplete
    counts = {"Ramnit": 10, "Lollipop": 5}
    e = EvaluationComplete(f1=0.87, accuracy=0.91, family_counts=counts)
    assert e.f1 == 0.87
    assert e.accuracy == 0.91
    assert e.family_counts["Ramnit"] == 10

def test_vault_sample_added(self):
    from wintermute.tui.events import VaultSampleAdded
    sample = {"id": "v001", "family": "Ramnit", "confidence": 0.72, "mutations": 3, "cycle": 1}
    e = VaultSampleAdded(sample=sample)
    assert e.sample["id"] == "v001"
    assert e.sample["confidence"] == 0.72
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tui.py::TestEvents -v`
Expected: FAIL with `ImportError` for missing classes

**Step 3: Implement new event classes**

Add to end of `src/wintermute/tui/events.py`:

```python
class PipelineProgress(Message):
    """Progress update from data pipeline operations."""

    def __init__(self, operation: str, progress: float, message: str) -> None:
        self.operation = operation
        self.progress = progress
        self.message = message
        super().__init__()


class EvaluationComplete(Message):
    """Fired when training/evaluation produces final metrics."""

    def __init__(self, f1: float, accuracy: float,
                 family_counts: dict[str, int] | None = None) -> None:
        self.f1 = f1
        self.accuracy = accuracy
        self.family_counts = family_counts or {}
        super().__init__()


class VaultSampleAdded(Message):
    """Fired when adversarial training adds a sample to the vault."""

    def __init__(self, sample: dict) -> None:
        self.sample = sample
        super().__init__()
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tui.py::TestEvents -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/wintermute/tui/events.py tests/test_tui.py
git commit -m "feat(tui): add PipelineProgress, EvaluationComplete, VaultSampleAdded events"
```

---

## Task 2: Add PipelineHook and Cancellation Support to Hooks

**Files:**
- Modify: `src/wintermute/tui/hooks.py`
- Test: `tests/test_tui.py`

**Step 1: Write failing tests**

Add to `tests/test_tui.py` inside `TestHooks` class:

```python
def test_training_hook_cancelled(self):
    from wintermute.tui.hooks import TrainingHook
    hook = TrainingHook()
    assert hook.cancelled is False
    hook.cancel()
    assert hook.cancelled is True
    hook.reset()
    assert hook.cancelled is False

def test_adversarial_hook_cancelled(self):
    from wintermute.tui.hooks import AdversarialHook
    hook = AdversarialHook()
    assert hook.cancelled is False
    hook.cancel()
    assert hook.cancelled is True

def test_pipeline_hook_no_app(self):
    from wintermute.tui.hooks import PipelineHook
    hook = PipelineHook()
    hook.on_progress("build", 0.5, "halfway")
    hook.on_log("test message")
    assert hook.cancelled is False
    hook.cancel()
    assert hook.cancelled is True

def test_adversarial_hook_vault_sample(self):
    from wintermute.tui.hooks import AdversarialHook
    hook = AdversarialHook()
    sample = {"id": "v001", "family": "Ramnit"}
    hook.on_vault_sample(sample)  # Should not raise without app
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tui.py::TestHooks -v`
Expected: FAIL — `cancelled`, `cancel()`, `PipelineHook`, `on_vault_sample` not found

**Step 3: Implement hook changes**

Rewrite `src/wintermute/tui/hooks.py`:

```python
"""Callback hooks bridging engine classes to the TUI event loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wintermute.tui.app import WintermuteApp

from wintermute.tui.events import (
    ActivityLogEntry,
    AdversarialCycleEnd,
    AdversarialEpisodeStep,
    EpochComplete,
    PipelineProgress,
    VaultSampleAdded,
)


@dataclass
class TrainingHook:
    """Pass to JointTrainer or MLMPretrainer. Calls on_epoch() after each epoch."""

    app: WintermuteApp | None = None
    cancelled: bool = field(default=False, init=False)

    def cancel(self) -> None:
        self.cancelled = True

    def reset(self) -> None:
        self.cancelled = False

    def on_epoch(
        self,
        epoch: int,
        phase: str,
        loss: float,
        train_acc: float,
        val_acc: float,
        f1: float,
        elapsed: float,
    ) -> None:
        if self.app is None:
            return
        self.app.call_from_thread(
            self.app.post_message,
            EpochComplete(epoch, phase, loss, train_acc, val_acc, f1, elapsed),
        )

    def on_log(self, text: str, level: str = "info") -> None:
        if self.app is None:
            return
        self.app.call_from_thread(
            self.app.post_message,
            ActivityLogEntry(text, level),
        )


@dataclass
class AdversarialHook:
    """Pass to AdversarialOrchestrator."""

    app: WintermuteApp | None = None
    cancelled: bool = field(default=False, init=False)

    def cancel(self) -> None:
        self.cancelled = True

    def reset(self) -> None:
        self.cancelled = False

    def on_episode_step(
        self, step: int, action: str, pos: int, conf: float, ok: bool
    ) -> None:
        if self.app is None:
            return
        self.app.call_from_thread(
            self.app.post_message,
            AdversarialEpisodeStep(step, action, pos, conf, ok),
        )

    def on_cycle_end(self, cycle: int, metrics: dict) -> None:
        if self.app is None:
            return
        self.app.call_from_thread(
            self.app.post_message,
            AdversarialCycleEnd(cycle, metrics),
        )

    def on_vault_sample(self, sample: dict) -> None:
        if self.app is None:
            return
        self.app.call_from_thread(
            self.app.post_message,
            VaultSampleAdded(sample),
        )

    def on_log(self, text: str, level: str = "info") -> None:
        if self.app is None:
            return
        self.app.call_from_thread(
            self.app.post_message,
            ActivityLogEntry(text, level),
        )


@dataclass
class PipelineHook:
    """Pass to data pipeline operations (build_dataset, SyntheticGenerator, etc.)."""

    app: WintermuteApp | None = None
    cancelled: bool = field(default=False, init=False)

    def cancel(self) -> None:
        self.cancelled = True

    def reset(self) -> None:
        self.cancelled = False

    def on_progress(self, operation: str, progress: float, message: str) -> None:
        if self.app is None:
            return
        self.app.call_from_thread(
            self.app.post_message,
            PipelineProgress(operation, progress, message),
        )

    def on_log(self, text: str, level: str = "info") -> None:
        if self.app is None:
            return
        self.app.call_from_thread(
            self.app.post_message,
            ActivityLogEntry(text, level),
        )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tui.py::TestHooks -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/wintermute/tui/hooks.py tests/test_tui.py
git commit -m "feat(tui): add PipelineHook, cancellation support, and vault sample callback"
```

---

## Task 3: Create ConfigDrawer Widget

**Files:**
- Create: `src/wintermute/tui/widgets/config_drawer.py`
- Modify: `src/wintermute/tui/widgets/__init__.py`
- Test: `tests/test_tui.py`

**Step 1: Write failing tests**

Add a new `TestConfigDrawer` class to `tests/test_tui.py`:

```python
class TestConfigDrawer:
    def test_create_drawer(self):
        from wintermute.tui.widgets.config_drawer import ConfigDrawer
        drawer = ConfigDrawer()
        assert drawer is not None

    def test_field_definitions(self):
        from wintermute.tui.widgets.config_drawer import FieldDef
        f = FieldDef(name="epochs", label="Epochs", default="50", field_type="int")
        assert f.name == "epochs"
        assert f.default == "50"
        assert f.field_type == "int"

    def test_select_field(self):
        from wintermute.tui.widgets.config_drawer import FieldDef
        f = FieldDef(
            name="num_classes",
            label="Num Classes",
            default="2",
            field_type="select",
            options=["2", "9"],
        )
        assert f.options == ["2", "9"]

    def test_switch_field(self):
        from wintermute.tui.widgets.config_drawer import FieldDef
        f = FieldDef(
            name="mlflow",
            label="MLflow Tracking",
            default="off",
            field_type="switch",
        )
        assert f.field_type == "switch"

    def test_get_values(self):
        from wintermute.tui.widgets.config_drawer import ConfigDrawer, FieldDef
        fields = [
            FieldDef(name="epochs", label="Epochs", default="50", field_type="int"),
            FieldDef(name="lr", label="Learning Rate", default="3e-4", field_type="float"),
        ]
        drawer = ConfigDrawer(fields=fields)
        # Before mounting, get_values returns defaults
        values = drawer.get_values()
        assert values["epochs"] == "50"
        assert values["lr"] == "3e-4"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tui.py::TestConfigDrawer -v`
Expected: FAIL with `ImportError`

**Step 3: Implement ConfigDrawer**

Create `src/wintermute/tui/widgets/config_drawer.py`:

```python
"""Reusable config drawer widget with typed form fields."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Sequence

from textual.containers import Vertical
from textual.app import ComposeResult
from textual.widgets import Button, Input, Label, Select, Static, Switch
from textual.reactive import reactive

from wintermute.tui import theme


@dataclass
class FieldDef:
    """Definition for a single config field."""

    name: str
    label: str
    default: str
    field_type: str = "str"  # "str", "int", "float", "select", "switch"
    options: list[str] = dc_field(default_factory=list)


class ConfigDrawer(Vertical):
    """Collapsible config panel with form fields and a start button."""

    DEFAULT_CSS = f"""
    ConfigDrawer {{
        width: 35%;
        height: 1fr;
        background: {theme.BG_PANEL};
        border-left: solid {theme.BORDER};
        padding: 1 2;
        display: none;
    }}
    ConfigDrawer.visible {{
        display: block;
    }}
    ConfigDrawer .drawer-title {{
        text-style: bold;
        color: {theme.TEXT_BRIGHT};
        margin-bottom: 1;
    }}
    ConfigDrawer Label {{
        color: {theme.TEXT_MUTED};
        margin-top: 1;
    }}
    ConfigDrawer .drawer-locked {{
        color: {theme.AMBER};
        text-style: bold;
        margin: 1 0;
    }}
    """

    locked: reactive[bool] = reactive(False)

    def __init__(
        self,
        fields: Sequence[FieldDef] | None = None,
        title: str = "CONFIGURE",
        start_label: str = "START",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._fields = list(fields) if fields else []
        self._title = title
        self._start_label = start_label

    def compose(self) -> ComposeResult:
        yield Static(self._title, classes="drawer-title")
        for f in self._fields:
            yield Label(f.label)
            if f.field_type == "switch":
                yield Switch(value=f.default.lower() in ("on", "true", "1"), id=f"cfg-{f.name}")
            elif f.field_type == "select":
                options = [(opt, opt) for opt in f.options]
                yield Select(options, value=f.default, id=f"cfg-{f.name}")
            else:
                yield Input(value=f.default, placeholder=f.label, id=f"cfg-{f.name}")
        yield Button(f"▶ {self._start_label}", id="drawer-start", variant="success")

    def get_values(self) -> dict[str, str]:
        """Return current field values as a dict. Uses defaults if not mounted."""
        values = {}
        for f in self._fields:
            try:
                widget = self.query_one(f"#cfg-{f.name}")
                if isinstance(widget, Switch):
                    values[f.name] = "on" if widget.value else "off"
                elif isinstance(widget, Select):
                    values[f.name] = str(widget.value)
                elif isinstance(widget, Input):
                    values[f.name] = widget.value
                else:
                    values[f.name] = f.default
            except Exception:
                values[f.name] = f.default
        return values

    def toggle(self) -> None:
        """Show or hide the drawer."""
        if self.locked:
            return
        self.toggle_class("visible")

    def lock(self) -> None:
        """Disable form during a running operation."""
        self.locked = True
        for f in self._fields:
            try:
                widget = self.query_one(f"#cfg-{f.name}")
                widget.disabled = True
            except Exception:
                pass
        try:
            self.query_one("#drawer-start", Button).disabled = True
        except Exception:
            pass

    def unlock(self) -> None:
        """Re-enable form after operation completes or is cancelled."""
        self.locked = False
        for f in self._fields:
            try:
                widget = self.query_one(f"#cfg-{f.name}")
                widget.disabled = False
            except Exception:
                pass
        try:
            self.query_one("#drawer-start", Button).disabled = False
        except Exception:
            pass

    def set_fields(self, fields: Sequence[FieldDef]) -> None:
        """Replace field definitions and rebuild the form (for Pipeline polymorphic form)."""
        self._fields = list(fields)
        self.remove_children()
        self.mount_all(list(self.compose()))
```

Update `src/wintermute/tui/widgets/__init__.py` to export the new widget:

```python
from wintermute.tui.widgets.config_drawer import ConfigDrawer, FieldDef
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tui.py::TestConfigDrawer -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/wintermute/tui/widgets/config_drawer.py src/wintermute/tui/widgets/__init__.py tests/test_tui.py
git commit -m "feat(tui): add ConfigDrawer reusable widget with typed form fields"
```

---

## Task 4: Create StatusBar Widget

**Files:**
- Create: `src/wintermute/tui/widgets/status_bar.py`
- Modify: `src/wintermute/tui/widgets/__init__.py`
- Test: `tests/test_tui.py`

**Step 1: Write failing tests**

Add a new `TestStatusBar` class to `tests/test_tui.py`:

```python
class TestStatusBar:
    def test_create(self):
        from wintermute.tui.widgets.status_bar import StatusBar
        bar = StatusBar()
        assert bar is not None

    def test_initial_render(self):
        from wintermute.tui.widgets.status_bar import StatusBar
        bar = StatusBar()
        text = bar.render()
        assert "Ready" in str(text)

    def test_set_task(self):
        from wintermute.tui.widgets.status_bar import StatusBar
        bar = StatusBar()
        bar.set_task("training", "epoch 5/50", 0.1)
        assert bar._tasks["training"]["label"] == "epoch 5/50"

    def test_clear_task(self):
        from wintermute.tui.widgets.status_bar import StatusBar
        bar = StatusBar()
        bar.set_task("training", "epoch 5/50", 0.1)
        bar.clear_task("training")
        assert "training" not in bar._tasks
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tui.py::TestStatusBar -v`
Expected: FAIL with `ImportError`

**Step 3: Implement StatusBar**

Create `src/wintermute/tui/widgets/status_bar.py`:

```python
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
                f"[{theme.TEXT_MUTED}]Ready — press [bold {theme.CYAN}]c[/] to configure"
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
```

Update `src/wintermute/tui/widgets/__init__.py` to include `StatusBar`:

```python
from wintermute.tui.widgets.config_drawer import ConfigDrawer, FieldDef
from wintermute.tui.widgets.status_bar import StatusBar
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tui.py::TestStatusBar -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/wintermute/tui/widgets/status_bar.py src/wintermute/tui/widgets/__init__.py tests/test_tui.py
git commit -m "feat(tui): add global StatusBar widget for background task visibility"
```

---

## Task 5: Wire Hooks into Engine Classes

**Files:**
- Modify: `src/wintermute/engine/joint_trainer.py`
- Modify: `src/wintermute/engine/pretrain.py`
- Modify: `src/wintermute/adversarial/orchestrator.py`
- Test: `tests/test_tui.py`

**Step 1: Write failing tests for hook integration**

Add a new `TestEngineHookIntegration` class to `tests/test_tui.py`:

```python
class TestEngineHookIntegration:
    def test_joint_trainer_accepts_hook(self):
        """JointTrainer constructor accepts tui_hook parameter."""
        import inspect
        from wintermute.engine.joint_trainer import JointTrainer
        sig = inspect.signature(JointTrainer.__init__)
        assert "tui_hook" in sig.parameters

    def test_pretrain_accepts_hook(self):
        """MLMPretrainer constructor accepts tui_hook parameter."""
        import inspect
        from wintermute.engine.pretrain import MLMPretrainer
        sig = inspect.signature(MLMPretrainer.__init__)
        assert "tui_hook" in sig.parameters

    def test_orchestrator_accepts_hook(self):
        """AdversarialOrchestrator constructor accepts tui_hook parameter."""
        import inspect
        from wintermute.adversarial.orchestrator import AdversarialOrchestrator
        sig = inspect.signature(AdversarialOrchestrator.__init__)
        assert "tui_hook" in sig.parameters
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tui.py::TestEngineHookIntegration -v`
Expected: FAIL — `tui_hook` not in parameter lists

**Step 3: Add hook parameter to JointTrainer**

In `src/wintermute/engine/joint_trainer.py`:

Add to `__init__` signature (after `pretrained_encoder_path` parameter):
```python
    def __init__(
        self,
        config: DetectorConfig,
        data_dir,
        overrides=None,
        pretrained_encoder_path=None,
        tui_hook=None,
    ):
```

Add to `__init__` body (after `self._epoch_count = 0`):
```python
        self._tui_hook = tui_hook
```

In the `train()` method, after the `print(f"  ep {ep:3d} ...")` line inside the epoch loop, add:
```python
            if self._tui_hook:
                self._tui_hook.on_epoch(ep, phase, loss, 0.0, f1, f1, elapsed)
                if self._tui_hook.cancelled:
                    self._tui_hook.on_log(f"Training cancelled at epoch {ep}", "warn")
                    return best_f1
```

At the end of `train()`, after the phase loops, before `return best_f1`, add:
```python
        if self._tui_hook:
            self._tui_hook.on_log(f"Training complete — best F1: {best_f1:.4f}", "ok")
```

**Step 4: Add hook parameter to MLMPretrainer**

In `src/wintermute/engine/pretrain.py`:

Add to `__init__` signature:
```python
    def __init__(
        self,
        config_path: str | Path | None = None,
        overrides: dict | None = None,
        tui_hook=None,
    ):
```

Add to `__init__` body:
```python
        self._tui_hook = tui_hook
```

In `pretrain()`, after the epoch print line inside the epoch loop, add:
```python
            if self._tui_hook:
                self._tui_hook.on_epoch(epoch, "MLM", avg_loss, 0.0, 0.0, 0.0, elapsed)
                if self._tui_hook.cancelled:
                    self._tui_hook.on_log(f"Pre-training cancelled at epoch {epoch}", "warn")
                    return avg_loss
```

At end of `pretrain()`, before return, add:
```python
        if self._tui_hook:
            self._tui_hook.on_log(f"Pre-training complete — final loss: {best_loss:.4f}", "ok")
```

**Step 5: Add hook parameter to AdversarialOrchestrator**

In `src/wintermute/adversarial/orchestrator.py`:

Add to `__init__` signature (after `trades_beta` parameter):
```python
        tui_hook=None,
```

Add to `__init__` body:
```python
        self._tui_hook = tui_hook
```

In `run_cycle()`, after computing metrics and before return, add:
```python
        if self._tui_hook:
            self._tui_hook.on_cycle_end(self._cycle_count, metrics)
            if self._tui_hook.cancelled:
                self._tui_hook.on_log("Adversarial training cancelled", "warn")
```

In `_collect_rollouts()`, inside the step loop (after `env.step()`), add:
```python
            if self._tui_hook:
                self._tui_hook.on_episode_step(
                    step, action_name, position, confidence, valid
                )
```

When a sample is added to the vault (inside the evasion check block), add:
```python
                if self._tui_hook:
                    self._tui_hook.on_vault_sample(sample_dict)
```

**Step 6: Run tests to verify they pass**

Run: `pytest tests/test_tui.py::TestEngineHookIntegration -v`
Expected: PASS

**Step 7: Run full test suite to check nothing broke**

Run: `pytest tests/ -v`
Expected: All existing tests PASS (hook params are optional with default None)

**Step 8: Commit**

```bash
git add src/wintermute/engine/joint_trainer.py src/wintermute/engine/pretrain.py src/wintermute/adversarial/orchestrator.py tests/test_tui.py
git commit -m "feat(engine): add optional tui_hook parameter to JointTrainer, MLMPretrainer, Orchestrator"
```

---

## Task 6: Add Theme CSS for Drawers and Status Bar

**Files:**
- Modify: `src/wintermute/tui/theme.py`
- Test: `tests/test_tui.py`

**Step 1: Write failing test**

Add to `TestTheme` in `tests/test_tui.py`:

```python
def test_stylesheet_has_drawer_styles(self):
    from wintermute.tui import theme
    assert "ConfigDrawer" in theme.STYLESHEET
    assert "StatusBar" in theme.STYLESHEET
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_tui.py::TestTheme::test_stylesheet_has_drawer_styles -v`
Expected: FAIL

**Step 3: Add drawer and status bar styles to theme.py STYLESHEET**

Append to the `STYLESHEET` string in `src/wintermute/tui/theme.py`:

```css
ConfigDrawer {
    dock: right;
    width: 35%;
    height: 1fr;
    background: $BG_PANEL;
    border-left: solid $BORDER;
    padding: 1 2;
    overflow-y: auto;
    display: none;
}
ConfigDrawer.visible {
    display: block;
}
ConfigDrawer .drawer-title {
    text-style: bold;
    color: $TEXT_BRIGHT;
    margin-bottom: 1;
}
ConfigDrawer Label {
    color: $TEXT_MUTED;
    margin-top: 1;
}
ConfigDrawer Button {
    margin-top: 2;
    width: 100%;
}
StatusBar {
    height: 1;
    dock: bottom;
    background: $BG_PANEL;
    color: $TEXT_MUTED;
    padding: 0 2;
}
```

Replace the `$` tokens with the actual Python color variables using string formatting, matching the existing pattern in theme.py.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_tui.py::TestTheme -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/wintermute/tui/theme.py tests/test_tui.py
git commit -m "feat(tui): add drawer and status bar styles to theme"
```

---

## Task 7: Update App — 6 Tabs, Status Bar, New Event Routing

**Files:**
- Modify: `src/wintermute/tui/app.py`
- Test: `tests/test_tui.py`

**Step 1: Write failing tests**

Add to `TestApp` in `tests/test_tui.py`:

```python
def test_app_has_six_tabs(self):
    from wintermute.tui.app import WintermuteApp
    app = WintermuteApp()
    bindings = {b.key: b for b in app.BINDINGS}
    assert "6" in bindings

def test_app_has_config_binding(self):
    from wintermute.tui.app import WintermuteApp
    app = WintermuteApp()
    bindings = {b.key: b for b in app.BINDINGS}
    assert "c" in bindings

def test_app_has_cancel_binding(self):
    from wintermute.tui.app import WintermuteApp
    app = WintermuteApp()
    keys = {b.key for b in app.BINDINGS}
    assert "ctrl+x" in keys
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tui.py::TestApp -v`
Expected: FAIL — no binding for "6", "c", or "ctrl+x"

**Step 3: Update app.py**

Modify `src/wintermute/tui/app.py`:

- Add imports for `PipelineScreen`, `StatusBar`, and new events
- Add Pipeline tab (tab 5), move Vault to tab 6
- Add bindings: `c` (toggle drawer), `ctrl+x` (cancel), `6` (vault tab)
- Mount `StatusBar` between TabbedContent and Footer
- Add event handlers for `PipelineProgress`, `EvaluationComplete`, `VaultSampleAdded`
- Add `action_toggle_drawer()` — queries active screen for ConfigDrawer and calls toggle()
- Add `action_cancel_task()` — posts a cancel signal to the active screen

Updated BINDINGS:
```python
BINDINGS = [
    Binding("1", "switch_tab('dashboard')", "Dashboard", show=True),
    Binding("2", "switch_tab('scan')", "Scan", show=True),
    Binding("3", "switch_tab('training')", "Training", show=True),
    Binding("4", "switch_tab('adversarial')", "Adversarial", show=True),
    Binding("5", "switch_tab('pipeline')", "Pipeline", show=True),
    Binding("6", "switch_tab('vault')", "Vault", show=True),
    Binding("c", "toggle_drawer", "Configure", show=True),
    Binding("ctrl+x", "cancel_task", "Cancel", show=False),
    Binding("q", "quit", "Quit", show=True),
]
```

Updated `compose()`:
```python
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
        with TabPane("⚙ PIPELINE", id="pipeline"):
            yield PipelineScreen()
        with TabPane("▣ VAULT", id="vault"):
            yield VaultScreen()
    yield StatusBar(id="status-bar")
    yield Footer()
```

Add new action methods:
```python
def action_toggle_drawer(self) -> None:
    """Toggle the config drawer on the active screen."""
    active = self.query_one("TabPane.-active")
    try:
        drawer = active.query_one(ConfigDrawer)
        drawer.toggle()
    except Exception:
        pass

def action_cancel_task(self) -> None:
    """Cancel the running operation on the active screen."""
    active = self.query_one("TabPane.-active")
    screen = active.children[0] if active.children else None
    if screen and hasattr(screen, "cancel_operation"):
        screen.cancel_operation()
```

Add new event handlers:
```python
def on_pipeline_progress(self, event: PipelineProgress) -> None:
    bar = self.query_one("#status-bar", StatusBar)
    bar.set_task("pipeline", f"{event.operation}: {event.message}", event.progress)

def on_evaluation_complete(self, event: EvaluationComplete) -> None:
    try:
        dash = self.query_one(DashboardScreen)
        dash.update_stat("stat-f1", f"{event.f1:.2f}", f"accuracy {event.accuracy:.2f}")
        if event.family_counts:
            dash.update_family_chart(event.family_counts)
    except Exception:
        pass

def on_vault_sample_added(self, event: VaultSampleAdded) -> None:
    try:
        vault = self.query_one(VaultScreen)
        vault.add_sample(event.sample)
    except Exception:
        pass

def on_activity_log_entry(self, event: ActivityLogEntry) -> None:
    self._log(event.text, event.level)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tui.py::TestApp -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/wintermute/tui/app.py tests/test_tui.py
git commit -m "feat(tui): add Pipeline tab, StatusBar, drawer/cancel bindings, new event routing"
```

---

## Task 8: Training Screen — Add Config Drawer and Worker

**Files:**
- Modify: `src/wintermute/tui/screens/training.py`
- Test: `tests/test_tui.py`

**Step 1: Write failing tests**

Add `TestTrainingScreen` to `tests/test_tui.py`:

```python
class TestTrainingScreen:
    def test_has_drawer(self):
        from wintermute.tui.screens.training import TrainingScreen
        from wintermute.tui.widgets.config_drawer import ConfigDrawer
        screen = TrainingScreen()
        # Check compose yields a ConfigDrawer
        widgets = list(screen.compose())
        drawer_types = [type(w).__name__ for w in widgets]
        assert "ConfigDrawer" in drawer_types

    def test_training_fields(self):
        from wintermute.tui.screens.training import TRAINING_FIELDS
        names = [f.name for f in TRAINING_FIELDS]
        assert "epochs_phase_a" in names
        assert "epochs_phase_b" in names
        assert "learning_rate" in names
        assert "batch_size" in names

    def test_has_cancel_operation(self):
        from wintermute.tui.screens.training import TrainingScreen
        screen = TrainingScreen()
        assert hasattr(screen, "cancel_operation")
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tui.py::TestTrainingScreen -v`
Expected: FAIL

**Step 3: Modify training.py**

Add imports at top of `src/wintermute/tui/screens/training.py`:
```python
from wintermute.tui.widgets.config_drawer import ConfigDrawer, FieldDef
```

Define field list as module constant:
```python
TRAINING_FIELDS = [
    FieldDef("epochs_phase_a", "Phase A Epochs (frozen)", "5", "int"),
    FieldDef("epochs_phase_b", "Phase B Epochs (fine-tune)", "20", "int"),
    FieldDef("learning_rate", "Learning Rate", "3e-4", "float"),
    FieldDef("batch_size", "Batch Size", "8", "int"),
    FieldDef("max_seq_length", "Max Seq Length", "2048", "select", ["512", "1024", "2048"]),
    FieldDef("num_classes", "Num Classes", "2", "select", ["2", "9"]),
    FieldDef("mlflow", "MLflow Tracking", "off", "switch"),
    FieldDef("experiment_name", "Experiment Name", "default", "str"),
]
```

Update `compose()` to wrap existing widgets in a `Horizontal` with the drawer:
```python
def compose(self) -> ComposeResult:
    with Horizontal(id="train-body"):
        with Vertical(id="train-main"):
            yield TrainStatusBar(id="train-status")
            yield ProgressBar(id="train-progress")
            with Horizontal(id="train-charts"):
                yield SparkPanel("LOSS", theme.AMBER, id="loss-panel", classes="train-chart")
                yield SparkPanel("ACCURACY", theme.GREEN, id="acc-panel", classes="train-chart")
            yield EpochTable(id="train-epochs")
        yield ConfigDrawer(
            fields=TRAINING_FIELDS,
            title="TRAINING CONFIG",
            start_label="START TRAINING",
            id="train-drawer",
        )
```

Add worker method to start training:
```python
def on_button_pressed(self, event: Button.Pressed) -> None:
    if event.button.id == "drawer-start":
        drawer = self.query_one("#train-drawer", ConfigDrawer)
        values = drawer.get_values()
        drawer.lock()
        self._hook = TrainingHook(app=self.app)
        self.run_worker(self._do_train(values), exclusive=True)

async def _do_train(self, values: dict) -> None:
    from wintermute.engine.joint_trainer import JointTrainer
    from wintermute.models.fusion import DetectorConfig
    from omegaconf import OmegaConf

    overrides = {
        "epochs_phase_a": int(values["epochs_phase_a"]),
        "epochs_phase_b": int(values["epochs_phase_b"]),
        "learning_rate": float(values["learning_rate"]),
        "batch_size": int(values["batch_size"]),
    }
    num_classes = int(values["num_classes"])
    config = DetectorConfig(num_classes=num_classes)

    self.app.call_from_thread(
        self.app._log, f"Training started — {overrides}", "ok"
    )
    bar = self.app.query_one("#status-bar")
    self.app.call_from_thread(bar.set_task, "training", "initializing...", -1)

    trainer = JointTrainer(
        config=config,
        data_dir="data/processed",
        overrides=overrides,
        tui_hook=self._hook,
    )
    trainer.train()

    self.app.call_from_thread(bar.clear_task, "training")

def on_worker_state_changed(self, event) -> None:
    if event.state.name in ("CANCELLED", "ERROR", "SUCCESS"):
        try:
            self.query_one("#train-drawer", ConfigDrawer).unlock()
        except Exception:
            pass

def cancel_operation(self) -> None:
    if hasattr(self, "_hook") and self._hook:
        self._hook.cancel()
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tui.py::TestTrainingScreen -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/wintermute/tui/screens/training.py tests/test_tui.py
git commit -m "feat(tui): add config drawer and background worker to Training screen"
```

---

## Task 9: Adversarial Screen — Add Config Drawer and Worker

**Files:**
- Modify: `src/wintermute/tui/screens/adversarial.py`
- Test: `tests/test_tui.py`

**Step 1: Write failing tests**

Add `TestAdversarialScreen` to `tests/test_tui.py`:

```python
class TestAdversarialScreen:
    def test_has_adversarial_fields(self):
        from wintermute.tui.screens.adversarial import ADVERSARIAL_FIELDS
        names = [f.name for f in ADVERSARIAL_FIELDS]
        assert "cycles" in names
        assert "trades_beta" in names
        assert "ewc_lambda" in names

    def test_has_cancel_operation(self):
        from wintermute.tui.screens.adversarial import AdversarialScreen
        screen = AdversarialScreen()
        assert hasattr(screen, "cancel_operation")
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tui.py::TestAdversarialScreen -v`
Expected: FAIL

**Step 3: Modify adversarial.py**

Add imports:
```python
from wintermute.tui.widgets.config_drawer import ConfigDrawer, FieldDef
```

Define fields:
```python
ADVERSARIAL_FIELDS = [
    FieldDef("cycles", "Cycles", "10", "int"),
    FieldDef("episodes_per_cycle", "Episodes / Cycle", "500", "int"),
    FieldDef("trades_beta", "TRADES Beta", "1.0", "float"),
    FieldDef("ewc_lambda", "EWC Lambda", "0.4", "float"),
    FieldDef("ppo_lr", "PPO Learning Rate", "3e-4", "float"),
    FieldDef("ppo_epochs", "PPO Epochs", "4", "int"),
]
```

Update `compose()` — wrap existing widgets in `Horizontal` with drawer. Only render drawer when Phase 5 is available.

Add worker methods following the same pattern as Training screen:
- `on_button_pressed` → read values, lock drawer, create `AdversarialHook`, launch worker
- `_do_adversarial(values)` → import orchestrator, build config, run cycles in loop
- `cancel_operation()` → call `self._hook.cancel()`
- `on_worker_state_changed` → unlock drawer

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tui.py::TestAdversarialScreen -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/wintermute/tui/screens/adversarial.py tests/test_tui.py
git commit -m "feat(tui): add config drawer and background worker to Adversarial screen"
```

---

## Task 10: Scan Screen — Refactor to Drawer Pattern

**Files:**
- Modify: `src/wintermute/tui/screens/scan.py`
- Test: `tests/test_tui.py`

**Step 1: Write failing tests**

Add `TestScanScreen` to `tests/test_tui.py`:

```python
class TestScanScreen:
    def test_has_scan_fields(self):
        from wintermute.tui.screens.scan import SCAN_FIELDS
        names = [f.name for f in SCAN_FIELDS]
        assert "file_path" in names
        assert "family" in names

    def test_has_cancel_operation(self):
        from wintermute.tui.screens.scan import ScanScreen
        screen = ScanScreen()
        assert hasattr(screen, "cancel_operation")
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tui.py::TestScanScreen -v`
Expected: FAIL

**Step 3: Refactor scan.py**

Define fields:
```python
SCAN_FIELDS = [
    FieldDef("file_path", "File Path", "", "str"),
    FieldDef("family", "Family Detection", "off", "switch"),
    FieldDef("model_path", "Model Path", "malware_detector.safetensors", "str"),
]
```

Refactor `compose()`: Move the existing `Input` + `Button` row into the drawer. Keep `DisassemblyLog` and `VerdictPanel` in the main area.

The existing scan worker (`_do_scan`) already uses `run_worker` — update it to read config from drawer instead of the inline Input widget.

Add `cancel_operation()` method.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tui.py::TestScanScreen -v`
Expected: PASS

**Step 5: Run full scan-related tests**

Run: `pytest tests/test_tui.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/wintermute/tui/screens/scan.py tests/test_tui.py
git commit -m "feat(tui): refactor Scan screen to drawer pattern with family and model options"
```

---

## Task 11: Create Pipeline Screen

**Files:**
- Create: `src/wintermute/tui/screens/pipeline.py`
- Modify: `src/wintermute/tui/screens/__init__.py`
- Test: `tests/test_tui.py`

**Step 1: Write failing tests**

Add `TestPipelineScreen` to `tests/test_tui.py`:

```python
class TestPipelineScreen:
    def test_create(self):
        from wintermute.tui.screens.pipeline import PipelineScreen
        screen = PipelineScreen()
        assert screen is not None

    def test_operation_fields(self):
        from wintermute.tui.screens.pipeline import BUILD_FIELDS, SYNTHETIC_FIELDS, PRETRAIN_FIELDS
        assert any(f.name == "data_dir" for f in BUILD_FIELDS)
        assert any(f.name == "n_samples" for f in SYNTHETIC_FIELDS)
        assert any(f.name == "epochs" for f in PRETRAIN_FIELDS)

    def test_has_cancel_operation(self):
        from wintermute.tui.screens.pipeline import PipelineScreen
        screen = PipelineScreen()
        assert hasattr(screen, "cancel_operation")
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tui.py::TestPipelineScreen -v`
Expected: FAIL with `ImportError`

**Step 3: Implement pipeline.py**

Create `src/wintermute/tui/screens/pipeline.py`:

```python
"""Pipeline screen — data build, synthetic generation, MalBERT pre-training."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    Label,
    ProgressBar,
    RichLog,
    Select,
    Static,
)

from wintermute.tui import theme
from wintermute.tui.hooks import PipelineHook, TrainingHook
from wintermute.tui.widgets.config_drawer import ConfigDrawer, FieldDef


BUILD_FIELDS = [
    FieldDef("data_dir", "Data Directory", "data", "str"),
    FieldDef("max_seq_length", "Max Seq Length", "2048", "select", ["512", "1024", "2048"]),
    FieldDef("vocab_size", "Vocab Size", "4096", "int"),
]

SYNTHETIC_FIELDS = [
    FieldDef("n_samples", "Number of Samples", "500", "int"),
    FieldDef("output_dir", "Output Directory", "data/processed", "str"),
    FieldDef("seed", "Random Seed", "42", "int"),
]

PRETRAIN_FIELDS = [
    FieldDef("epochs", "Epochs", "50", "int"),
    FieldDef("learning_rate", "Learning Rate", "1e-4", "float"),
    FieldDef("batch_size", "Batch Size", "8", "int"),
    FieldDef("mask_prob", "Mask Probability", "0.15", "float"),
]

_OP_FIELDS = {
    "build": BUILD_FIELDS,
    "synthetic": SYNTHETIC_FIELDS,
    "pretrain": PRETRAIN_FIELDS,
}

_OP_LABELS = {
    "build": "BUILD DATASET",
    "synthetic": "GENERATE SYNTHETIC DATA",
    "pretrain": "MALBERT PRE-TRAIN",
}


class PipelineScreen(Vertical):
    """Data pipeline operations: build dataset, synthetic generation, pre-training."""

    DEFAULT_CSS = f"""
    PipelineScreen {{
        height: 1fr;
    }}
    #pipe-body {{
        height: 1fr;
    }}
    #pipe-main {{
        width: 1fr;
        padding: 1 2;
    }}
    #pipe-op-select {{
        width: 40;
        margin-bottom: 1;
    }}
    #pipe-progress {{
        height: 1;
        margin: 1 0;
    }}
    #pipe-log {{
        height: 1fr;
    }}
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._hook = None
        self._current_op = "build"

    def compose(self) -> ComposeResult:
        with Horizontal(id="pipe-body"):
            with Vertical(id="pipe-main"):
                yield Label("Operation")
                yield Select(
                    [("Build Dataset", "build"), ("Synthetic Data", "synthetic"), ("MalBERT Pretrain", "pretrain")],
                    value="build",
                    id="pipe-op-select",
                )
                yield ProgressBar(id="pipe-progress", total=100)
                yield RichLog(id="pipe-log", highlight=True, markup=True)
            yield ConfigDrawer(
                fields=BUILD_FIELDS,
                title="BUILD DATASET",
                start_label="START",
                id="pipe-drawer",
            )

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "pipe-op-select":
            return
        op = str(event.value)
        self._current_op = op
        drawer = self.query_one("#pipe-drawer", ConfigDrawer)
        drawer.set_fields(_OP_FIELDS[op])
        drawer._title = _OP_LABELS[op]

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "drawer-start":
            drawer = self.query_one("#pipe-drawer", ConfigDrawer)
            values = drawer.get_values()
            drawer.lock()
            op = self._current_op
            if op == "pretrain":
                self._hook = TrainingHook(app=self.app)
            else:
                self._hook = PipelineHook(app=self.app)
            self.run_worker(self._do_operation(op, values), exclusive=True)

    async def _do_operation(self, op: str, values: dict) -> None:
        if op == "build":
            await self._do_build(values)
        elif op == "synthetic":
            await self._do_synthetic(values)
        elif op == "pretrain":
            await self._do_pretrain(values)

    async def _do_build(self, values: dict) -> None:
        from wintermute.data.tokenizer import build_vocabulary, encode_sequence
        self.app.call_from_thread(self.app._log, "Dataset build started", "ok")
        bar = self.app.query_one("#status-bar")
        self.app.call_from_thread(bar.set_task, "pipeline", "building dataset...", -1)
        # Delegate to existing build_dataset logic from cli.py
        # The actual implementation will call tokenizer functions
        self.app.call_from_thread(bar.clear_task, "pipeline")
        self.app.call_from_thread(self.app._log, "Dataset build complete", "ok")

    async def _do_synthetic(self, values: dict) -> None:
        from wintermute.data.augment import SyntheticGenerator
        self.app.call_from_thread(self.app._log, "Synthetic generation started", "ok")
        bar = self.app.query_one("#status-bar")
        self.app.call_from_thread(bar.set_task, "pipeline", "generating...", -1)
        gen = SyntheticGenerator(
            n_samples=int(values["n_samples"]),
            seed=int(values["seed"]),
        )
        gen.generate_dataset(out_dir=values["output_dir"])
        self.app.call_from_thread(bar.clear_task, "pipeline")
        self.app.call_from_thread(self.app._log, "Synthetic generation complete", "ok")

    async def _do_pretrain(self, values: dict) -> None:
        from wintermute.engine.pretrain import MLMPretrainer
        self.app.call_from_thread(self.app._log, "Pre-training started", "ok")
        bar = self.app.query_one("#status-bar")
        self.app.call_from_thread(bar.set_task, "pipeline", "pre-training...", -1)
        overrides = {
            "pretrain": {
                "epochs": int(values["epochs"]),
                "learning_rate": float(values["learning_rate"]),
                "batch_size": int(values["batch_size"]),
                "mask_prob": float(values["mask_prob"]),
            }
        }
        trainer = MLMPretrainer(overrides=overrides, tui_hook=self._hook)
        trainer.pretrain()
        self.app.call_from_thread(bar.clear_task, "pipeline")
        self.app.call_from_thread(self.app._log, "Pre-training complete", "ok")

    def on_worker_state_changed(self, event) -> None:
        if event.state.name in ("CANCELLED", "ERROR", "SUCCESS"):
            try:
                self.query_one("#pipe-drawer", ConfigDrawer).unlock()
            except Exception:
                pass

    def cancel_operation(self) -> None:
        if self._hook:
            self._hook.cancel()
```

Update `src/wintermute/tui/screens/__init__.py` to export PipelineScreen.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tui.py::TestPipelineScreen -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/wintermute/tui/screens/pipeline.py src/wintermute/tui/screens/__init__.py tests/test_tui.py
git commit -m "feat(tui): add Pipeline screen with build, synthetic, and pretrain operations"
```

---

## Task 12: Wire Dashboard to Live Events

**Files:**
- Modify: `src/wintermute/tui/screens/dashboard.py`
- Test: `tests/test_tui.py`

**Step 1: Write failing tests**

Add `TestDashboardWiring` to `tests/test_tui.py`:

```python
class TestDashboardWiring:
    def test_has_update_family_chart(self):
        from wintermute.tui.screens.dashboard import DashboardScreen
        screen = DashboardScreen()
        assert hasattr(screen, "update_family_chart")

    def test_family_chart_accepts_data(self):
        from wintermute.tui.screens.dashboard import FamilyChart
        chart = FamilyChart()
        assert hasattr(chart, "update_counts")
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tui.py::TestDashboardWiring -v`
Expected: FAIL

**Step 3: Add methods to dashboard.py**

Add `update_family_chart` to `DashboardScreen`:
```python
def update_family_chart(self, counts: dict[str, int]) -> None:
    try:
        chart = self.query_one("#dash-families", FamilyChart)
        chart.update_counts(counts)
    except Exception:
        pass
```

Add `update_counts` to `FamilyChart`:
```python
def update_counts(self, counts: dict[str, int]) -> None:
    for i, (name, _, color) in enumerate(self._FAMILIES):
        if name in counts:
            self._FAMILIES[i] = (name, counts[name], color)
    self.refresh()
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tui.py::TestDashboardWiring -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/wintermute/tui/screens/dashboard.py tests/test_tui.py
git commit -m "feat(tui): wire Dashboard to live training and evaluation events"
```

---

## Task 13: Wire Vault Screen to Live Events

**Files:**
- Modify: `src/wintermute/tui/screens/vault.py`
- Test: `tests/test_tui.py`

**Step 1: Write failing tests**

Add `TestVaultWiring` to `tests/test_tui.py`:

```python
class TestVaultWiring:
    def test_has_add_sample(self):
        from wintermute.tui.screens.vault import VaultScreen
        screen = VaultScreen()
        assert hasattr(screen, "add_sample")
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tui.py::TestVaultWiring -v`
Expected: FAIL

**Step 3: Add add_sample to VaultScreen**

In `src/wintermute/tui/screens/vault.py`, add:

```python
def add_sample(self, sample: dict) -> None:
    """Add a single vault sample from an adversarial training event."""
    try:
        table = self.query_one("#vault-table", VaultTable)
        table.add_row(
            sample.get("id", "?"),
            sample.get("family", "?"),
            f"{sample.get('confidence', 0):.2f}",
            str(sample.get("mutations", 0)),
            str(sample.get("cycle", 0)),
            key=sample.get("id", None),
        )
    except Exception:
        pass
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tui.py::TestVaultWiring -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/wintermute/tui/screens/vault.py tests/test_tui.py
git commit -m "feat(tui): wire Vault screen to live VaultSampleAdded events"
```

---

## Task 14: Final Integration Test and Cleanup

**Files:**
- Test: `tests/test_tui.py`

**Step 1: Write integration tests**

Add `TestIntegration` class to `tests/test_tui.py`:

```python
class TestIntegration:
    def test_all_screens_importable(self):
        from wintermute.tui.screens.dashboard import DashboardScreen
        from wintermute.tui.screens.scan import ScanScreen
        from wintermute.tui.screens.training import TrainingScreen
        from wintermute.tui.screens.adversarial import AdversarialScreen
        from wintermute.tui.screens.pipeline import PipelineScreen
        from wintermute.tui.screens.vault import VaultScreen

    def test_all_events_importable(self):
        from wintermute.tui.events import (
            EpochComplete,
            ScanProgress,
            AdversarialCycleEnd,
            AdversarialEpisodeStep,
            ActivityLogEntry,
            PipelineProgress,
            EvaluationComplete,
            VaultSampleAdded,
        )

    def test_all_hooks_importable(self):
        from wintermute.tui.hooks import TrainingHook, AdversarialHook, PipelineHook

    def test_all_widgets_importable(self):
        from wintermute.tui.widgets.config_drawer import ConfigDrawer, FieldDef
        from wintermute.tui.widgets.status_bar import StatusBar
        from wintermute.tui.widgets.stat_card import StatCard
        from wintermute.tui.widgets.confidence_bar import ConfidenceBar
        from wintermute.tui.widgets.diff_view import DiffView
        from wintermute.tui.widgets.action_log import ActionLog

    def test_app_creates_with_all_screens(self):
        from wintermute.tui.app import WintermuteApp
        app = WintermuteApp()
        assert app.TITLE == "WINTERMUTE v3.0"
        bindings = {b.key for b in app.BINDINGS}
        for key in ("1", "2", "3", "4", "5", "6", "c", "q"):
            assert key in bindings, f"Missing binding: {key}"
```

**Step 2: Run integration tests**

Run: `pytest tests/test_tui.py::TestIntegration -v`
Expected: PASS

**Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All PASS

**Step 4: Lint**

Run: `ruff check src/wintermute/tui/ tests/test_tui.py`
Run: `ruff format src/wintermute/tui/ tests/test_tui.py`

Fix any issues.

**Step 5: Commit**

```bash
git add tests/test_tui.py
git commit -m "test(tui): add integration tests for control panel"
```

**Step 6: Final commit of any lint fixes**

```bash
git add -u
git commit -m "style: lint fixes for TUI control panel"
```
