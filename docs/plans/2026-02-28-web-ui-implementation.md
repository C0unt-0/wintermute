# Web UI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the Textual TUI and replace it with a React + Vite web UI (Terminal Noir aesthetic) that serves as an analyst workstation, extending the existing FastAPI backend.

**Architecture:** Monorepo with `web/` for the React SPA and extended `api/` for the backend. Transport-agnostic hooks in `engine/` replace TUI-specific hooks. WebSocket broadcasts live events (epoch progress, adversarial steps, pipeline progress) to connected browsers. FastAPI serves the built frontend in production.

**Tech Stack:** Python (FastAPI, Celery, Redis), React 18, Vite, TypeScript, Tailwind CSS, Recharts, WebSocket

**Design Doc:** `docs/plans/2026-02-28-web-ui-design.md`

---

## Phase 1: TUI Removal + Hook Generalization

### Task 1: Create transport-agnostic event dataclasses

**Files:**
- Create: `src/wintermute/engine/events.py`
- Test: `tests/test_engine_events.py`

**Context:** The TUI events (`tui/events.py`) are subclasses of `textual.message.Message`. We need plain dataclasses with the same fields that can serialize to JSON for WebSocket transport.

**Step 1: Write the test**

```python
# tests/test_engine_events.py
from wintermute.engine.events import (
    EpochComplete, ScanProgress, AdversarialCycleEnd,
    AdversarialEpisodeStep, ActivityLogEntry, PipelineProgress,
    EvaluationComplete, VaultSampleAdded,
)


def test_epoch_complete_to_dict():
    e = EpochComplete(epoch=1, phase="A", loss=0.42, train_acc=0.85, val_acc=0.83, f1=0.81, elapsed=12.5)
    d = e.to_dict()
    assert d["type"] == "epoch_complete"
    assert d["epoch"] == 1
    assert d["phase"] == "A"
    assert d["loss"] == 0.42


def test_adversarial_cycle_end_to_dict():
    e = AdversarialCycleEnd(cycle=3, metrics={"evasion_rate": 0.12})
    d = e.to_dict()
    assert d["type"] == "adversarial_cycle_end"
    assert d["cycle"] == 3
    assert d["metrics"]["evasion_rate"] == 0.12


def test_pipeline_progress_to_dict():
    e = PipelineProgress(operation="synthetic", progress=0.65, message="Generated 325/500")
    d = e.to_dict()
    assert d["type"] == "pipeline_progress"
    assert d["progress"] == 0.65


def test_vault_sample_added_to_dict():
    sample = {"id": "abc", "family": "Ramnit", "confidence": 0.23, "mutations": 5, "cycle": 2}
    e = VaultSampleAdded(sample=sample)
    d = e.to_dict()
    assert d["type"] == "vault_sample_added"
    assert d["sample"]["family"] == "Ramnit"


def test_all_events_have_type_field():
    events = [
        EpochComplete(epoch=0, phase="A", loss=0.0, train_acc=0.0, val_acc=0.0, f1=0.0, elapsed=0.0),
        ScanProgress(phase="disassembly", data={}),
        AdversarialCycleEnd(cycle=0, metrics={}),
        AdversarialEpisodeStep(step=0, action="nop", position=0, confidence=0.0, valid=True),
        ActivityLogEntry(text="test", level="info"),
        PipelineProgress(operation="build", progress=0.0, message=""),
        EvaluationComplete(f1=0.0, accuracy=0.0, family_counts={}),
        VaultSampleAdded(sample={}),
    ]
    for e in events:
        d = e.to_dict()
        assert "type" in d
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_engine_events.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'wintermute.engine.events'`

**Step 3: Write the implementation**

```python
# src/wintermute/engine/events.py
"""engine/events.py — Transport-agnostic event dataclasses.

These replace tui/events.py (Textual Messages) with plain dataclasses
that serialize to dicts for WebSocket/JSON transport.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict


@dataclass
class EpochComplete:
    epoch: int
    phase: str
    loss: float
    train_acc: float
    val_acc: float
    f1: float
    elapsed: float

    def to_dict(self) -> dict:
        return {"type": "epoch_complete", **asdict(self)}


@dataclass
class ScanProgress:
    phase: str
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"type": "scan_progress", **asdict(self)}


@dataclass
class AdversarialCycleEnd:
    cycle: int
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"type": "adversarial_cycle_end", **asdict(self)}


@dataclass
class AdversarialEpisodeStep:
    step: int
    action: str
    position: int
    confidence: float
    valid: bool

    def to_dict(self) -> dict:
        return {"type": "adversarial_episode_step", **asdict(self)}


@dataclass
class ActivityLogEntry:
    text: str
    level: str = "info"

    def to_dict(self) -> dict:
        return {"type": "activity_log", **asdict(self)}


@dataclass
class PipelineProgress:
    operation: str
    progress: float
    message: str

    def to_dict(self) -> dict:
        return {"type": "pipeline_progress", **asdict(self)}


@dataclass
class EvaluationComplete:
    f1: float
    accuracy: float
    family_counts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"type": "evaluation_complete", **asdict(self)}


@dataclass
class VaultSampleAdded:
    sample: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"type": "vault_sample_added", **asdict(self)}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_engine_events.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/wintermute/engine/events.py tests/test_engine_events.py
git commit -m "feat: add transport-agnostic event dataclasses in engine/events.py"
```

---

### Task 2: Create transport-agnostic hook base classes

**Files:**
- Create: `src/wintermute/engine/hooks.py`
- Test: `tests/test_engine_hooks.py`

**Context:** The TUI hooks (`tui/hooks.py`) are dataclasses that hold a reference to the Textual App and call `app.call_from_thread()`. The new hooks use a callback-based interface — each hook accepts a `callback` function that receives event dicts. This makes them usable by WebSocket, CLI, or any other transport.

**Step 1: Write the test**

```python
# tests/test_engine_hooks.py
from wintermute.engine.hooks import TrainingHook, AdversarialHook, PipelineHook


def test_training_hook_emits_epoch():
    events = []
    hook = TrainingHook(callback=events.append)
    hook.on_epoch(epoch=1, phase="A", loss=0.5, train_acc=0.8, val_acc=0.75, f1=0.77, elapsed=5.0)
    assert len(events) == 1
    assert events[0]["type"] == "epoch_complete"
    assert events[0]["epoch"] == 1


def test_training_hook_no_callback():
    hook = TrainingHook()
    hook.on_epoch(epoch=1, phase="A", loss=0.5, train_acc=0.8, val_acc=0.75, f1=0.77, elapsed=5.0)
    # Should not raise


def test_training_hook_cancel():
    hook = TrainingHook()
    assert not hook.cancelled
    hook.cancel()
    assert hook.cancelled
    hook.reset()
    assert not hook.cancelled


def test_training_hook_on_log():
    events = []
    hook = TrainingHook(callback=events.append)
    hook.on_log("Training started", "info")
    assert events[0]["type"] == "activity_log"
    assert events[0]["text"] == "Training started"


def test_adversarial_hook_emits_episode_step():
    events = []
    hook = AdversarialHook(callback=events.append)
    hook.on_episode_step(step=1, action="insert_nop", pos=42, conf=0.3, ok=True)
    assert events[0]["type"] == "adversarial_episode_step"
    assert events[0]["action"] == "insert_nop"


def test_adversarial_hook_emits_cycle_end():
    events = []
    hook = AdversarialHook(callback=events.append)
    hook.on_cycle_end(cycle=2, metrics={"evasion_rate": 0.15})
    assert events[0]["type"] == "adversarial_cycle_end"


def test_adversarial_hook_emits_vault_sample():
    events = []
    hook = AdversarialHook(callback=events.append)
    hook.on_vault_sample({"id": "x", "family": "Ramnit"})
    assert events[0]["type"] == "vault_sample_added"


def test_pipeline_hook_emits_progress():
    events = []
    hook = PipelineHook(callback=events.append)
    hook.on_progress("synthetic", 0.5, "Generating...")
    assert events[0]["type"] == "pipeline_progress"
    assert events[0]["progress"] == 0.5
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_engine_hooks.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'wintermute.engine.hooks'`

**Step 3: Write the implementation**

```python
# src/wintermute/engine/hooks.py
"""engine/hooks.py — Transport-agnostic callback hooks.

Replaces tui/hooks.py. Each hook accepts an optional callback function
that receives event dicts. The callback could be a WebSocket broadcast,
a CLI printer, or anything else.

Usage in engines (unchanged API):
    if self._hook:
        self._hook.on_epoch(epoch, phase, loss, train_acc, val_acc, f1, elapsed)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from wintermute.engine.events import (
    EpochComplete, AdversarialCycleEnd, AdversarialEpisodeStep,
    ActivityLogEntry, PipelineProgress, VaultSampleAdded,
)


@dataclass
class TrainingHook:
    callback: Callable[[dict], None] | None = None
    cancelled: bool = field(default=False, init=False)

    def cancel(self) -> None:
        self.cancelled = True

    def reset(self) -> None:
        self.cancelled = False

    def on_epoch(self, epoch: int, phase: str, loss: float,
                 train_acc: float, val_acc: float, f1: float, elapsed: float) -> None:
        if self.callback:
            self.callback(EpochComplete(
                epoch=epoch, phase=phase, loss=loss,
                train_acc=train_acc, val_acc=val_acc, f1=f1, elapsed=elapsed,
            ).to_dict())

    def on_log(self, text: str, level: str = "info") -> None:
        if self.callback:
            self.callback(ActivityLogEntry(text=text, level=level).to_dict())


@dataclass
class AdversarialHook:
    callback: Callable[[dict], None] | None = None
    cancelled: bool = field(default=False, init=False)

    def cancel(self) -> None:
        self.cancelled = True

    def reset(self) -> None:
        self.cancelled = False

    def on_episode_step(self, step: int, action: str, pos: int,
                        conf: float, ok: bool) -> None:
        if self.callback:
            self.callback(AdversarialEpisodeStep(
                step=step, action=action, position=pos,
                confidence=conf, valid=ok,
            ).to_dict())

    def on_cycle_end(self, cycle: int, metrics: dict) -> None:
        if self.callback:
            self.callback(AdversarialCycleEnd(cycle=cycle, metrics=metrics).to_dict())

    def on_vault_sample(self, sample: dict) -> None:
        if self.callback:
            self.callback(VaultSampleAdded(sample=sample).to_dict())

    def on_log(self, text: str, level: str = "info") -> None:
        if self.callback:
            self.callback(ActivityLogEntry(text=text, level=level).to_dict())


@dataclass
class PipelineHook:
    callback: Callable[[dict], None] | None = None
    cancelled: bool = field(default=False, init=False)

    def cancel(self) -> None:
        self.cancelled = True

    def reset(self) -> None:
        self.cancelled = False

    def on_progress(self, operation: str, progress: float, message: str) -> None:
        if self.callback:
            self.callback(PipelineProgress(
                operation=operation, progress=progress, message=message,
            ).to_dict())

    def on_log(self, text: str, level: str = "info") -> None:
        if self.callback:
            self.callback(ActivityLogEntry(text=text, level=level).to_dict())
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_engine_hooks.py -v`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add src/wintermute/engine/hooks.py tests/test_engine_hooks.py
git commit -m "feat: add transport-agnostic hook classes in engine/hooks.py"
```

---

### Task 3: Update engine code to use new hooks

**Files:**
- Modify: `src/wintermute/engine/joint_trainer.py:78,93,492-503` — change `tui_hook` → `hook`
- Modify: `src/wintermute/engine/pretrain.py:136,145,259-274` — change `tui_hook` → `hook`
- Modify: `src/wintermute/adversarial/orchestrator.py:46,86,118-119` — change `tui_hook` → `hook`

**Context:** The engines currently accept `tui_hook` parameter. Rename to `hook` and ensure the same duck-typed interface works. The new `engine/hooks.py` classes have the same method signatures (`on_epoch`, `on_log`, `cancelled`, etc.), so only the parameter name changes.

**Step 1: Update `joint_trainer.py`**

In `src/wintermute/engine/joint_trainer.py`:
- Line 78: `tui_hook=None,` → `hook=None,`
- Line 93: `self._tui_hook = tui_hook` → `self._hook = hook`
- Lines 492-503: Replace all `self._tui_hook` with `self._hook`

**Step 2: Update `pretrain.py`**

In `src/wintermute/engine/pretrain.py`:
- Line 136: `tui_hook=None,` → `hook=None,`
- Line 145: `self._tui_hook = tui_hook` → `self._hook = hook`
- Lines 259-274: Replace all `self._tui_hook` with `self._hook`

**Step 3: Update `adversarial/orchestrator.py`**

In `src/wintermute/adversarial/orchestrator.py`:
- Line 46: `tui_hook=None,` → `hook=None,`
- Line 86: `self._tui_hook = tui_hook` → `self._hook = hook`
- Lines 118-119: Replace all `self._tui_hook` with `self._hook`

**Step 4: Run existing tests**

Run: `pytest tests/ -v --ignore=tests/test_tui.py`
Expected: All existing tests PASS (engines use duck-typed hook interface, so renaming doesn't break anything)

**Step 5: Commit**

```bash
git add src/wintermute/engine/joint_trainer.py src/wintermute/engine/pretrain.py src/wintermute/adversarial/orchestrator.py
git commit -m "refactor: rename tui_hook → hook in engine/trainer/orchestrator"
```

---

### Task 4: Remove TUI code

**Files:**
- Delete: `src/wintermute/tui/` (entire directory)
- Delete: `tests/test_tui.py`
- Modify: `src/wintermute/cli.py:442-453` — remove `tui` command
- Modify: `pyproject.toml:38-40,46` — remove `tui` optional dep

**Step 1: Delete TUI directory and test file**

```bash
rm -rf src/wintermute/tui/
rm tests/test_tui.py
```

**Step 2: Remove `tui` command from `cli.py`**

Remove lines 442-453 from `src/wintermute/cli.py`:

```python
# DELETE THIS BLOCK:
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

**Step 3: Remove `tui` from `pyproject.toml`**

Remove line 38-40:
```toml
tui = [
    "textual>=1.0.0",
]
```

Update line 46 — remove `tui` from `all` extras:
```toml
# Before:
all = ["wintermute[api,mlops,dev,tui,adversarial]", ...]
# After:
all = ["wintermute[api,mlops,dev,adversarial]", ...]
```

**Step 4: Run all tests (excluding deleted test)**

Run: `pytest tests/ -v`
Expected: All remaining tests PASS. No imports of `wintermute.tui` anywhere.

**Step 5: Verify no dangling imports**

Run: `grep -r "from wintermute.tui" src/ tests/`
Expected: No matches

**Step 6: Commit**

```bash
git add -A
git commit -m "feat!: remove Textual TUI — replaced by web UI

BREAKING: The 'wintermute tui' command and textual dependency are removed.
The TUI hook system is replaced by transport-agnostic hooks in engine/hooks.py."
```

---

## Phase 2: Backend API Extension

### Task 5: Create Pydantic schemas

**Files:**
- Modify: `api/schemas.py` (replace stub)
- Test: `tests/test_api_schemas.py`

**Step 1: Write the test**

```python
# tests/test_api_schemas.py
from api.schemas import (
    TrainingRequest, TrainingStatus, JobResponse,
    AdversarialRequest, PipelineRequest,
    VaultSample, VaultSampleDetail,
    DashboardResponse,
)


def test_training_request_defaults():
    r = TrainingRequest()
    assert r.epochs_phase_a == 5
    assert r.epochs_phase_b == 20
    assert r.learning_rate == 3e-4
    assert r.batch_size == 8
    assert r.num_classes == 2
    assert r.mlflow is False


def test_adversarial_request_defaults():
    r = AdversarialRequest()
    assert r.cycles == 10
    assert r.episodes_per_cycle == 500
    assert r.trades_beta == 1.0


def test_job_response_model():
    r = JobResponse(job_id="abc-123", poll_url="/api/v1/training/abc-123/status")
    assert r.job_id == "abc-123"


def test_vault_sample_model():
    s = VaultSample(id="x", family="Ramnit", confidence=0.23, mutations=5, cycle=2)
    assert s.family == "Ramnit"


def test_dashboard_response():
    d = DashboardResponse(model_version="3.0.0", f1=0.92, accuracy=0.95, vault_size=42)
    assert d.model_version == "3.0.0"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_api_schemas.py -v`
Expected: FAIL — `ImportError`

**Step 3: Write the implementation**

```python
# api/schemas.py
"""api/schemas.py — Pydantic request/response models for all API endpoints."""
from __future__ import annotations
from pydantic import BaseModel, Field


# ── Job lifecycle ────────────────────────────────────────────────────────────

class JobResponse(BaseModel):
    job_id: str
    poll_url: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    error: str | None = None


# ── Dashboard ────────────────────────────────────────────────────────────────

class DashboardResponse(BaseModel):
    model_version: str = "3.0.0"
    f1: float = 0.0
    accuracy: float = 0.0
    vault_size: int = 0
    family_counts: dict[str, int] = Field(default_factory=dict)


# ── Scan ─────────────────────────────────────────────────────────────────────

class ScanResponse(BaseModel):
    job_id: str
    status: str
    result: dict | None = None
    error: str | None = None


# ── Training ─────────────────────────────────────────────────────────────────

class TrainingRequest(BaseModel):
    epochs_phase_a: int = 5
    epochs_phase_b: int = 20
    learning_rate: float = 3e-4
    batch_size: int = 8
    max_seq_length: int = 2048
    num_classes: int = 2
    mlflow: bool = False
    experiment_name: str = "default"


class TrainingStatus(BaseModel):
    job_id: str
    status: str
    epoch: int = 0
    phase: str = ""
    loss: float = 0.0
    train_acc: float = 0.0
    val_acc: float = 0.0
    f1: float = 0.0


# ── Adversarial ──────────────────────────────────────────────────────────────

class AdversarialRequest(BaseModel):
    cycles: int = 10
    episodes_per_cycle: int = 500
    trades_beta: float = 1.0
    ewc_lambda: float = 0.4
    ppo_lr: float = 3e-4
    ppo_epochs: int = 4


class AdversarialStatus(BaseModel):
    job_id: str
    status: str
    cycle: int = 0
    evasion_rate: float = 0.0
    adv_tpr: float = 0.0
    vault_size: int = 0


# ── Pipeline ─────────────────────────────────────────────────────────────────

class PipelineRequest(BaseModel):
    # Build
    data_dir: str = "data"
    max_seq_length: int = 2048
    vocab_size: int | None = None
    # Synthetic
    n_samples: int = 500
    output_dir: str = "data/processed"
    seed: int = 42
    # Pretrain
    epochs: int = 50
    learning_rate: float = 3e-4
    batch_size: int = 8
    mask_prob: float = 0.15


class PipelineStatus(BaseModel):
    job_id: str
    status: str
    operation: str = ""
    progress: float = 0.0
    message: str = ""


# ── Vault ────────────────────────────────────────────────────────────────────

class VaultSample(BaseModel):
    id: str
    family: str
    confidence: float
    mutations: int
    cycle: int


class VaultSampleDetail(VaultSample):
    original_bytes: str = ""
    mutated_bytes: str = ""
    diff: str = ""
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_api_schemas.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add api/schemas.py tests/test_api_schemas.py
git commit -m "feat: add Pydantic schemas for all API endpoints"
```

---

### Task 6: Create WebSocket manager

**Files:**
- Create: `api/ws.py`
- Test: `tests/test_ws_manager.py`

**Step 1: Write the test**

```python
# tests/test_ws_manager.py
import asyncio
import pytest
from api.ws import ConnectionManager


@pytest.fixture
def manager():
    return ConnectionManager()


class FakeWebSocket:
    def __init__(self):
        self.sent = []
        self.accepted = False

    async def accept(self):
        self.accepted = True

    async def send_json(self, data):
        self.sent.append(data)

    async def receive_text(self):
        await asyncio.sleep(100)  # Block forever


@pytest.mark.asyncio
async def test_connect_and_disconnect(manager):
    ws = FakeWebSocket()
    await manager.connect(ws)
    assert len(manager.active_connections) == 1
    manager.disconnect(ws)
    assert len(manager.active_connections) == 0


@pytest.mark.asyncio
async def test_broadcast(manager):
    ws1, ws2 = FakeWebSocket(), FakeWebSocket()
    await manager.connect(ws1)
    await manager.connect(ws2)
    await manager.broadcast({"type": "test", "value": 42})
    assert ws1.sent == [{"type": "test", "value": 42}]
    assert ws2.sent == [{"type": "test", "value": 42}]


@pytest.mark.asyncio
async def test_broadcast_removes_dead_connections(manager):
    class DeadSocket(FakeWebSocket):
        async def send_json(self, data):
            raise RuntimeError("connection closed")

    dead = DeadSocket()
    alive = FakeWebSocket()
    await manager.connect(dead)
    await manager.connect(alive)
    await manager.broadcast({"type": "test"})
    assert dead not in manager.active_connections
    assert alive in manager.active_connections
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ws_manager.py -v`
Expected: FAIL — `ModuleNotFoundError`

Note: You may need `pip install pytest-asyncio` for async tests.

**Step 3: Write the implementation**

```python
# api/ws.py
"""api/ws.py — WebSocket connection manager for live event broadcasting."""
from __future__ import annotations

from fastapi import WebSocket


class ConnectionManager:
    """Manages WebSocket connections and broadcasts events to all clients."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, data: dict) -> None:
        dead: list[WebSocket] = []
        for conn in self.active_connections:
            try:
                await conn.send_json(data)
            except Exception:
                dead.append(conn)
        for conn in dead:
            self.disconnect(conn)


ws_manager = ConnectionManager()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ws_manager.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add api/ws.py tests/test_ws_manager.py
git commit -m "feat: add WebSocket connection manager for live events"
```

---

### Task 7: Create API routers — dashboard

**Files:**
- Create: `api/routers/__init__.py`
- Create: `api/routers/dashboard.py`

**Step 1: Create the router**

```python
# api/routers/__init__.py
# (empty file)
```

```python
# api/routers/dashboard.py
"""Dashboard endpoint — system metrics overview."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter

from api.schemas import DashboardResponse

router = APIRouter(prefix="/api/v1", tags=["dashboard"])


@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard():
    """Return current model metrics and system status."""
    manifest_path = Path("malware_detector_manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        return DashboardResponse(
            model_version=manifest.get("version", "unknown"),
            f1=manifest.get("best_val_f1", 0.0),
            accuracy=manifest.get("best_val_acc", 0.0),
        )
    return DashboardResponse()
```

**Step 2: Commit**

```bash
git add api/routers/
git commit -m "feat: add dashboard API router"
```

---

### Task 8: Create API routers — training

**Files:**
- Create: `api/routers/training.py`

**Context:** Training jobs run in background threads. The router stores job state in a dict keyed by `job_id`. The WebSocket manager broadcasts epoch events as they arrive via the `TrainingHook` callback.

**Step 1: Write the router**

```python
# api/routers/training.py
"""Training endpoints — start, poll, cancel training jobs."""
from __future__ import annotations

import threading
import uuid

from fastapi import APIRouter

from api.schemas import TrainingRequest, TrainingStatus, JobResponse
from api.ws import ws_manager
from wintermute.engine.hooks import TrainingHook

router = APIRouter(prefix="/api/v1/training", tags=["training"])

_jobs: dict[str, dict] = {}


def _run_training(job_id: str, config: TrainingRequest, hook: TrainingHook) -> None:
    """Background thread: runs JointTrainer with the given config."""
    import json as _json
    from pathlib import Path
    from wintermute.engine.joint_trainer import JointTrainer
    from wintermute.models.fusion import DetectorConfig

    _jobs[job_id]["status"] = "RUNNING"
    try:
        dp = Path(config.data_dir if hasattr(config, "data_dir") else "data/processed")
        vocab = _json.loads((dp / "vocab.json").read_text()) if (dp / "vocab.json").exists() else {}
        overrides = {
            "epochs_phase_a": config.epochs_phase_a,
            "epochs_phase_b": config.epochs_phase_b,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
        }
        cfg = DetectorConfig(vocab_size=len(vocab) or 49, num_classes=config.num_classes)
        JointTrainer(cfg, dp, overrides=overrides, hook=hook).train()
        _jobs[job_id]["status"] = "COMPLETED"
    except Exception as e:
        _jobs[job_id]["status"] = "FAILED"
        _jobs[job_id]["error"] = str(e)


@router.post("/start", response_model=JobResponse, status_code=202)
async def start_training(config: TrainingRequest):
    job_id = str(uuid.uuid4())

    import asyncio
    loop = asyncio.get_event_loop()

    def callback(event: dict):
        asyncio.run_coroutine_threadsafe(ws_manager.broadcast(event), loop)
        if event.get("type") == "epoch_complete":
            _jobs[job_id].update({
                "epoch": event["epoch"], "phase": event["phase"],
                "loss": event["loss"], "train_acc": event["train_acc"],
                "val_acc": event["val_acc"], "f1": event["f1"],
            })

    hook = TrainingHook(callback=callback)
    _jobs[job_id] = {"status": "PENDING", "hook": hook, "epoch": 0, "phase": "",
                     "loss": 0.0, "train_acc": 0.0, "val_acc": 0.0, "f1": 0.0}

    thread = threading.Thread(target=_run_training, args=(job_id, config, hook), daemon=True)
    thread.start()

    return JobResponse(job_id=job_id, poll_url=f"/api/v1/training/{job_id}/status")


@router.get("/{job_id}/status", response_model=TrainingStatus)
async def get_training_status(job_id: str):
    if job_id not in _jobs:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Job not found")
    job = _jobs[job_id]
    return TrainingStatus(
        job_id=job_id, status=job["status"],
        epoch=job.get("epoch", 0), phase=job.get("phase", ""),
        loss=job.get("loss", 0.0), train_acc=job.get("train_acc", 0.0),
        val_acc=job.get("val_acc", 0.0), f1=job.get("f1", 0.0),
    )


@router.post("/{job_id}/cancel")
async def cancel_training(job_id: str):
    if job_id not in _jobs:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Job not found")
    hook = _jobs[job_id].get("hook")
    if hook:
        hook.cancel()
    _jobs[job_id]["status"] = "CANCELLED"
    return {"job_id": job_id, "status": "CANCELLED"}
```

**Step 2: Commit**

```bash
git add api/routers/training.py
git commit -m "feat: add training API router with background jobs + WebSocket events"
```

---

### Task 9: Create API routers — adversarial, pipeline, vault

**Files:**
- Create: `api/routers/adversarial.py`
- Create: `api/routers/pipeline.py`
- Create: `api/routers/vault.py`

**Context:** These follow the same pattern as the training router. The adversarial and pipeline routers use `AdversarialHook` and `PipelineHook` respectively. The vault router reads from an in-memory store populated by adversarial events.

**Step 1: Write `api/routers/adversarial.py`**

Same pattern as training router — `POST /start`, `GET /{job_id}/status`, `POST /{job_id}/cancel`. Uses `AdversarialHook` and runs `AdversarialOrchestrator.run_cycle()` in a background thread.

**Step 2: Write `api/routers/pipeline.py`**

Three start endpoints: `POST /pipeline/{operation}` where `operation` is `build`, `synthetic`, or `pretrain`. Each spawns the appropriate engine function in a background thread.

**Step 3: Write `api/routers/vault.py`**

Simple CRUD — `GET /vault/samples` (list all), `GET /vault/samples/{id}` (detail). Reads from an in-memory `_vault_samples` list that gets populated via `VaultSampleAdded` events.

**Step 4: Commit**

```bash
git add api/routers/adversarial.py api/routers/pipeline.py api/routers/vault.py
git commit -m "feat: add adversarial, pipeline, and vault API routers"
```

---

### Task 10: Wire routers into main.py + add WebSocket endpoint + static files

**Files:**
- Modify: `api/main.py` — add router includes, WebSocket endpoint, static file mount

**Context:** The main app needs to:
1. Include all routers
2. Add `WS /api/v1/ws` endpoint
3. Mount `web/dist/` as static files (for production)
4. Add a catch-all route for SPA client-side routing

**Step 1: Rewrite `api/main.py`**

Keep existing scan endpoints but extract them. Add all new routers. Add WebSocket endpoint. Mount static files if `web/dist/` exists.

```python
# Key additions to api/main.py:

from api.routers import dashboard, training, adversarial, pipeline, vault
from api.ws import ws_manager

# Include routers
app.include_router(dashboard.router)
app.include_router(training.router)
app.include_router(adversarial.router)
app.include_router(pipeline.router)
app.include_router(vault.router)

# WebSocket
@app.websocket("/api/v1/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)

# Static files (production)
dist = Path(__file__).parent.parent / "web" / "dist"
if dist.is_dir():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(dist), html=True), name="spa")
```

**Step 2: Run a quick smoke test**

Run: `cd api && python -c "from main import app; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add api/main.py
git commit -m "feat: wire all routers, WebSocket, and static file serving into FastAPI"
```

---

## Phase 3: Frontend Setup

### Task 11: Scaffold React + Vite + TypeScript project

**Step 1: Create Vite project**

```bash
cd /path/to/wintermute
npm create vite@latest web -- --template react-ts
cd web
npm install
npm install react-router-dom recharts tailwindcss @tailwindcss/vite
```

**Step 2: Configure `vite.config.ts`**

```typescript
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
```

**Step 3: Add `.gitignore` entries for `web/node_modules/`, `web/dist/`**

**Step 4: Verify dev server starts**

Run: `cd web && npm run dev`
Expected: Vite dev server on `http://localhost:5173`

**Step 5: Commit**

```bash
git add web/
git commit -m "feat: scaffold React + Vite + TypeScript + Tailwind project"
```

---

### Task 12: Create Terminal Noir theme

**Files:**
- Create: `web/src/styles/theme.css`
- Modify: `web/src/index.css`

**Context:** Apply the Terminal Noir color palette, import fonts (JetBrains Mono, Space Mono, Outfit), set up CSS custom properties.

**Step 1: Write theme CSS**

```css
/* web/src/styles/theme.css */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Outfit:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

:root {
  --bg-primary: #0a0e14;
  --bg-surface: #111820;
  --bg-elevated: #1a2332;
  --border: #1e2d3d;
  --text-primary: #e0e6ed;
  --text-muted: #6b7d8e;
  --safe: #00e88f;
  --threat: #ff3b5c;
  --data: #00d4ff;
  --warn: #ffb224;
  --purple: #b48ead;

  --font-code: 'JetBrains Mono', monospace;
  --font-heading: 'Space Mono', monospace;
  --font-body: 'Outfit', sans-serif;
}

body {
  margin: 0;
  background-color: var(--bg-primary);
  color: var(--text-primary);
  font-family: var(--font-body);
}
```

Use the `@frontend-design` skill guidelines for creative execution: subtle dot-grid background pattern, glow effects on accent colors, refined spacing.

**Step 2: Commit**

```bash
git add web/src/styles/
git commit -m "feat: add Terminal Noir theme with CSS custom properties and fonts"
```

---

### Task 13: Create API client and WebSocket hook

**Files:**
- Create: `web/src/api/client.ts`
- Create: `web/src/api/ws.ts`
- Create: `web/src/hooks/useWebSocket.ts`
- Create: `web/src/hooks/useJob.ts`

**Step 1: Write API client**

```typescript
// web/src/api/client.ts
const BASE = "/api/v1";

export async function fetchJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
  return res.json();
}

export const api = {
  dashboard: () => fetchJSON<DashboardData>("/dashboard"),
  startTraining: (config: TrainingConfig) =>
    fetchJSON<JobResponse>("/training/start", { method: "POST", body: JSON.stringify(config) }),
  trainingStatus: (id: string) => fetchJSON<TrainingStatus>(`/training/${id}/status`),
  cancelTraining: (id: string) =>
    fetchJSON<{ status: string }>(`/training/${id}/cancel`, { method: "POST" }),
  // ... similar for adversarial, pipeline, vault, scan
};
```

**Step 2: Write WebSocket client and React hook**

```typescript
// web/src/api/ws.ts
export function createWS(onMessage: (data: any) => void): WebSocket {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${protocol}//${location.host}/api/v1/ws`);
  ws.onmessage = (e) => onMessage(JSON.parse(e.data));
  ws.onclose = () => setTimeout(() => createWS(onMessage), 2000); // Auto-reconnect
  return ws;
}

// web/src/hooks/useWebSocket.ts
import { useEffect, useRef, useCallback, useState } from "react";
import { createWS } from "../api/ws";

export function useWebSocket() {
  const [events, setEvents] = useState<any[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    wsRef.current = createWS((data) => setEvents((prev) => [...prev.slice(-200), data]));
    return () => wsRef.current?.close();
  }, []);

  const subscribe = useCallback((type: string, handler: (data: any) => void) => {
    // Returns unsubscribe function
  }, []);

  return { events, subscribe };
}
```

**Step 3: Write `useJob` hook**

```typescript
// web/src/hooks/useJob.ts — generic hook for start → poll → done lifecycle
```

**Step 4: Commit**

```bash
git add web/src/api/ web/src/hooks/
git commit -m "feat: add API client, WebSocket client, and React hooks"
```

---

### Task 14: Create App shell with tab navigation

**Files:**
- Modify: `web/src/App.tsx`
- Create: `web/src/pages/Dashboard.tsx` (placeholder)
- Create: `web/src/pages/Scan.tsx` (placeholder)
- Create: `web/src/pages/Training.tsx` (placeholder)
- Create: `web/src/pages/Adversarial.tsx` (placeholder)
- Create: `web/src/pages/Pipeline.tsx` (placeholder)
- Create: `web/src/pages/Vault.tsx` (placeholder)

**Context:** Six tabs matching the TUI: Dashboard, Scan, Training, Adversarial, Pipeline, Vault. Use `react-router-dom` for routing. The shell includes a top nav bar and the WebSocket provider context.

**Step 1: Write App shell with routing**

Tab bar across the top with Terminal Noir styling. Active tab highlighted with `--data` (cyan) underline. Each tab renders its page component.

**Step 2: Create placeholder pages**

Each page exports a component with just a heading and the config panel placeholder. These will be fleshed out in Phase 4.

**Step 3: Verify navigation works**

Run: `cd web && npm run dev`
Expected: All 6 tabs clickable, pages render with headings

**Step 4: Commit**

```bash
git add web/src/
git commit -m "feat: add App shell with 6-tab navigation and placeholder pages"
```

---

### Task 15: Build shared components

**Files:**
- Create: `web/src/components/StatCard.tsx`
- Create: `web/src/components/ConfidenceBar.tsx`
- Create: `web/src/components/ConfigPanel.tsx`
- Create: `web/src/components/ActivityLog.tsx`
- Create: `web/src/components/DiffView.tsx`
- Create: `web/src/components/SparklineChart.tsx`

**Context:** These map directly to the TUI widgets. Use the `@frontend-design` skill for creative execution — these should feel distinctly Terminal Noir, not generic.

**Component specs:**

| Component | TUI Widget | Description |
|-----------|-----------|-------------|
| `StatCard` | `stat_card.py` | Metric display with label, value, subtitle. Glowing border on accent colors. |
| `ConfidenceBar` | `confidence_bar.py` | Horizontal 0.0–1.0 bar with gradient fill (green→red). Animated on change. |
| `ConfigPanel` | `config_drawer.py` | Right-side collapsible panel with form fields. Slides in/out with animation. |
| `ActivityLog` | `action_log.py` | Timestamped event stream. Color-coded by level (info/ok/warn/error). Auto-scrolls. |
| `DiffView` | `diff_view.py` | Red/green assembly code diff. Monospace, line-by-line. |
| `SparklineChart` | Sparkline widget | Mini trend chart using Recharts `<LineChart>`. No axes, just the line. |

**Step 1: Implement all 6 components**

Follow `@frontend-design` guidelines:
- JetBrains Mono for data/code components
- Subtle glow effects on StatCard borders
- Smooth CSS transitions on ConfigPanel open/close
- Auto-scroll with `useRef` + `scrollIntoView` on ActivityLog

**Step 2: Commit**

```bash
git add web/src/components/
git commit -m "feat: add shared UI components — StatCard, ConfidenceBar, ConfigPanel, etc."
```

---

## Phase 4: Frontend Pages

### Task 16: Dashboard page

**Files:**
- Modify: `web/src/pages/Dashboard.tsx`

**Context:** Mirrors TUI Dashboard: 5 stat cards (MODEL, CLEAN TPR, ADV. TPR, MACRO F1, VAULT), family distribution bar chart (Recharts), activity log. Fetches data from `GET /api/v1/dashboard` on mount. Subscribes to WebSocket events for live updates.

**Step 1: Implement Dashboard with stat cards + family chart + activity log**
**Step 2: Wire up `useDashboard` hook for initial fetch**
**Step 3: Subscribe to WebSocket events for live metric updates**
**Step 4: Commit**

```bash
git add web/src/pages/Dashboard.tsx web/src/hooks/useDashboard.ts
git commit -m "feat: implement Dashboard page with stat cards, family chart, activity log"
```

---

### Task 17: Scan page

**Files:**
- Modify: `web/src/pages/Scan.tsx`

**Context:** File upload (drag-and-drop), disassembly view (syntax-highlighted opcode list), verdict panel with confidence bars. Uses `POST /api/v1/scan` → polls `GET /api/v1/status/{id}`.

**Key differences from TUI:**
- Drag-and-drop file upload instead of path input
- Syntax highlighting via CSS classes (call=purple, jump=green, nop=muted)
- Animated verdict reveal

**Step 1: Implement file upload with drag-and-drop zone**
**Step 2: Add disassembly view with opcode syntax highlighting**
**Step 3: Add verdict panel with ConfidenceBar components**
**Step 4: Wire up scan → poll lifecycle with `useJob` hook**
**Step 5: Commit**

```bash
git add web/src/pages/Scan.tsx
git commit -m "feat: implement Scan page with drag-and-drop upload and verdict panel"
```

---

### Task 18: Training page

**Files:**
- Modify: `web/src/pages/Training.tsx`

**Context:** ConfigPanel on the right with training form fields (same defaults as TUI). Epoch table + loss/accuracy sparklines on the left. Real-time updates via WebSocket `epoch_complete` events.

**Step 1: Add ConfigPanel with training fields**
**Step 2: Add epoch DataTable (Epoch | Phase | Loss | Train Acc | Val Acc | F1 | Time)**
**Step 3: Add SparklineCharts for loss and accuracy trends**
**Step 4: Wire start/cancel to API, subscribe to WebSocket events**
**Step 5: Commit**

```bash
git add web/src/pages/Training.tsx
git commit -m "feat: implement Training page with config panel, epoch table, live charts"
```

---

### Task 19: Adversarial page

**Files:**
- Modify: `web/src/pages/Adversarial.tsx`

**Context:** Red/blue team cards, episode action log (streaming via WebSocket), cycle table, evasion/confidence sparklines. Most complex page — mirrors TUI adversarial screen.

**Step 1: Add red/blue team stat cards**
**Step 2: Add episode action log (streaming via WebSocket)**
**Step 3: Add cycle DataTable and sparklines**
**Step 4: Add ConfigPanel with adversarial settings**
**Step 5: Wire start/cancel to API**
**Step 6: Commit**

```bash
git add web/src/pages/Adversarial.tsx
git commit -m "feat: implement Adversarial page with red/blue teams and live episode log"
```

---

### Task 20: Pipeline page

**Files:**
- Modify: `web/src/pages/Pipeline.tsx`

**Context:** Operation selector dropdown (Build Dataset | Synthetic Data | MalBERT Pretrain), config form that changes per operation, progress bar + log. Updates via WebSocket `pipeline_progress` events.

**Step 1: Add operation selector with polymorphic config form**
**Step 2: Add progress bar and log output**
**Step 3: Wire to API with `useJob` hook**
**Step 4: Commit**

```bash
git add web/src/pages/Pipeline.tsx
git commit -m "feat: implement Pipeline page with operation selector and progress tracking"
```

---

### Task 21: Vault page

**Files:**
- Modify: `web/src/pages/Vault.tsx`

**Context:** Sample table on the left, detail panel with DiffView on the right. Click a row to view the mutation diff. Fetches from `GET /api/v1/vault/samples`.

**Step 1: Add sample DataTable (ID | Family | Confidence | Mutations | Cycle)**
**Step 2: Add detail panel with DiffView**
**Step 3: Wire to vault API endpoints**
**Step 4: Commit**

```bash
git add web/src/pages/Vault.tsx
git commit -m "feat: implement Vault page with sample table and mutation diff viewer"
```

---

## Phase 5: Docker + Documentation

### Task 22: Create Dockerfile and docker-compose.yml

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`

**Step 1: Write multi-stage Dockerfile**

```dockerfile
# Stage 1: Build frontend
FROM node:20-alpine AS frontend
WORKDIR /app/web
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# Stage 2: Python + static files
FROM python:3.11-slim
WORKDIR /app
COPY --from=frontend /app/web/dist ./web/dist
COPY pyproject.toml ./
COPY src/ ./src/
COPY api/ ./api/
COPY configs/ ./configs/
RUN pip install --no-cache-dir -e ".[api]"
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Step 2: Write docker-compose.yml**

```yaml
services:
  api:
    build: .
    ports: ["8000:8000"]
    depends_on: [redis]
  worker:
    build: .
    command: celery -A src.wintermute.engine.worker worker -l info
    depends_on: [redis]
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
```

**Step 3: Verify build**

Run: `docker compose build`
Expected: Multi-stage build succeeds

**Step 4: Commit**

```bash
git add Dockerfile docker-compose.yml
git commit -m "feat: add multi-stage Dockerfile and docker-compose.yml for web UI"
```

---

### Task 23: Update CLAUDE.md and project docs

**Files:**
- Modify: `CLAUDE.md` — update architecture table, remove TUI references, add web UI commands
- Modify: `pyproject.toml` — bump version to 4.0.0

**Step 1: Update CLAUDE.md**

- Remove TUI from "Two Runtimes" table, replace with Web UI
- Add `web/` to Code Layout
- Add `npm run dev` and `npm run build` to Common Commands
- Update Docker Stack section
- Remove TUI safety note

**Step 2: Bump version**

In `pyproject.toml`: `version = "4.0.0"`

**Step 3: Commit**

```bash
git add CLAUDE.md pyproject.toml
git commit -m "docs: update CLAUDE.md for web UI, remove TUI references, bump to v4.0.0"
```

---

## Task Summary

| Phase | Tasks | Description |
|-------|-------|-------------|
| 1 | 1–4 | TUI removal + transport-agnostic hooks |
| 2 | 5–10 | Backend API extension (schemas, WebSocket, routers, main.py) |
| 3 | 11–15 | Frontend setup (Vite, theme, API client, shell, components) |
| 4 | 16–21 | Frontend pages (Dashboard, Scan, Training, Adversarial, Pipeline, Vault) |
| 5 | 22–23 | Docker + documentation |

**Total: 23 tasks, ~23 commits**
