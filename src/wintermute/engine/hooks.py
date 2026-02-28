"""hooks.py — Transport-agnostic callback hooks for engine → UI communication.

These replace the TUI-specific hooks in `tui/hooks.py` with generic
callback-based hooks.  Each hook accepts an optional `callback` function
that receives event dicts (from `engine/events.py`).

Usage in JointTrainer:
    if self._hook:
        self._hook.on_epoch(epoch, phase, loss, train_acc, val_acc, f1, elapsed)

Usage in AdversarialOrchestrator:
    if self._hook:
        self._hook.on_episode_step(step, action, pos, conf, ok)
        self._hook.on_cycle_end(cycle, metrics_dict)
        self._hook.on_vault_sample(sample)

Usage in Pipeline operations:
    if self._hook:
        self._hook.on_progress(operation, progress, message)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from wintermute.engine.events import (
    ActivityLogEntry,
    AdversarialCycleEnd,
    AdversarialEpisodeStep,
    EpochComplete,
    PipelineProgress,
    VaultSampleAdded,
)


@dataclass
class TrainingHook:
    """Pass to JointTrainer / pretrain loops.  Calls on_epoch() after each epoch."""

    callback: Callable[[dict], None] | None = None
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
        if self.callback is None:
            return
        event = EpochComplete(
            epoch=epoch,
            phase=phase,
            loss=loss,
            train_acc=train_acc,
            val_acc=val_acc,
            f1=f1,
            elapsed=elapsed,
        )
        self.callback(event.to_dict())

    def on_log(self, text: str, level: str = "info") -> None:
        if self.callback is None:
            return
        event = ActivityLogEntry(text=text, level=level)
        self.callback(event.to_dict())


@dataclass
class AdversarialHook:
    """Pass to AdversarialOrchestrator.  Calls on_episode_step() and on_cycle_end()."""

    callback: Callable[[dict], None] | None = None
    cancelled: bool = field(default=False, init=False)

    def cancel(self) -> None:
        self.cancelled = True

    def reset(self) -> None:
        self.cancelled = False

    def on_episode_step(self, step: int, action: str, pos: int, conf: float, ok: bool) -> None:
        if self.callback is None:
            return
        event = AdversarialEpisodeStep(
            step=step, action=action, position=pos, confidence=conf, valid=ok
        )
        self.callback(event.to_dict())

    def on_cycle_end(self, cycle: int, metrics: dict) -> None:
        if self.callback is None:
            return
        event = AdversarialCycleEnd(cycle=cycle, metrics=metrics)
        self.callback(event.to_dict())

    def on_vault_sample(self, sample: dict) -> None:
        if self.callback is None:
            return
        event = VaultSampleAdded(sample=sample)
        self.callback(event.to_dict())

    def on_log(self, text: str, level: str = "info") -> None:
        if self.callback is None:
            return
        event = ActivityLogEntry(text=text, level=level)
        self.callback(event.to_dict())


@dataclass
class PipelineHook:
    """Pass to data pipeline operations (build_dataset, SyntheticGenerator, etc.)."""

    callback: Callable[[dict], None] | None = None
    cancelled: bool = field(default=False, init=False)

    def cancel(self) -> None:
        self.cancelled = True

    def reset(self) -> None:
        self.cancelled = False

    def on_progress(self, operation: str, progress: float, message: str) -> None:
        if self.callback is None:
            return
        event = PipelineProgress(operation=operation, progress=progress, message=message)
        self.callback(event.to_dict())

    def on_log(self, text: str, level: str = "info") -> None:
        if self.callback is None:
            return
        event = ActivityLogEntry(text=text, level=level)
        self.callback(event.to_dict())
