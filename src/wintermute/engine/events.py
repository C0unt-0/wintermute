"""events.py — Transport-agnostic event dataclasses for engine → UI communication.

These replace the TUI-specific textual.Message subclasses with plain
dataclasses that serialize to JSON via `to_dict()`, making them suitable
for WebSocket broadcast, logging, or any other transport.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields


def _snake_case(name: str) -> str:
    """Convert CamelCase class name to snake_case for the 'type' field."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


@dataclass
class EpochComplete:
    """Fired after each training epoch. phase is 'A' or 'B' matching JointTrainer."""

    epoch: int
    phase: str
    loss: float
    train_acc: float
    val_acc: float
    f1: float
    elapsed: float

    def to_dict(self) -> dict:
        return {"type": _snake_case(type(self).__name__), **{f.name: getattr(self, f.name) for f in fields(self)}}


@dataclass
class ScanProgress:
    """Fired during scan phases."""

    phase: str
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"type": _snake_case(type(self).__name__), **{f.name: getattr(self, f.name) for f in fields(self)}}


@dataclass
class AdversarialCycleEnd:
    """Fired at end of each adversarial training cycle."""

    cycle: int
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"type": _snake_case(type(self).__name__), **{f.name: getattr(self, f.name) for f in fields(self)}}


@dataclass
class AdversarialEpisodeStep:
    """Fired for each action in a live adversarial episode."""

    step: int
    action: str
    position: int
    confidence: float
    valid: bool

    def to_dict(self) -> dict:
        return {"type": _snake_case(type(self).__name__), **{f.name: getattr(self, f.name) for f in fields(self)}}


@dataclass
class ActivityLogEntry:
    """Generic log entry for the dashboard activity log."""

    text: str
    level: str = "info"

    def to_dict(self) -> dict:
        return {"type": "activity_log", **{f.name: getattr(self, f.name) for f in fields(self)}}


@dataclass
class PipelineProgress:
    """Progress update from data pipeline operations."""

    operation: str
    progress: float
    message: str

    def to_dict(self) -> dict:
        return {"type": _snake_case(type(self).__name__), **{f.name: getattr(self, f.name) for f in fields(self)}}


@dataclass
class EvaluationComplete:
    """Fired when training/evaluation produces final metrics."""

    f1: float
    accuracy: float
    family_counts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"type": _snake_case(type(self).__name__), **{f.name: getattr(self, f.name) for f in fields(self)}}


@dataclass
class VaultSampleAdded:
    """Fired when adversarial training adds a sample to the vault."""

    sample: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"type": _snake_case(type(self).__name__), **{f.name: getattr(self, f.name) for f in fields(self)}}
