"""
hooks.py — Callback bridge: training loops → TUI messages.

Usage in JointTrainer:
    if self._hook:
        self._hook.on_epoch(epoch, phase, loss, train_acc, val_acc, f1, elapsed)

Usage in AdversarialOrchestrator:
    if self._hook:
        self._hook.on_episode_step(step, action, pos, conf, ok)
        self._hook.on_cycle_end(cycle, metrics_dict)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wintermute.tui.app import WintermuteApp


@dataclass
class TrainingHook:
    """Pass to JointTrainer. It calls on_epoch() after each epoch."""

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
        from wintermute.tui.events import EpochComplete

        self.app.call_from_thread(
            self.app.post_message,
            EpochComplete(
                epoch=epoch,
                phase=phase,
                loss=loss,
                train_acc=train_acc,
                val_acc=val_acc,
                f1=f1,
                elapsed=elapsed,
            ),
        )

    def on_log(self, text: str, level: str = "info") -> None:
        if self.app is None:
            return
        self.app.call_from_thread(self.app._log, text, level)


@dataclass
class AdversarialHook:
    """Pass to AdversarialOrchestrator. Calls on_episode_step() and on_cycle_end()."""

    app: WintermuteApp | None = None
    cancelled: bool = field(default=False, init=False)

    def cancel(self) -> None:
        self.cancelled = True

    def reset(self) -> None:
        self.cancelled = False

    def on_episode_step(self, step: int, action: str, pos: int, conf: float, ok: bool) -> None:
        if self.app is None:
            return
        from wintermute.tui.events import AdversarialEpisodeStep

        self.app.call_from_thread(
            self.app.post_message,
            AdversarialEpisodeStep(
                step=step, action=action, position=pos, confidence=conf, valid=ok
            ),
        )

    def on_cycle_end(self, cycle: int, metrics: dict) -> None:
        if self.app is None:
            return
        from wintermute.tui.events import AdversarialCycleEnd

        self.app.call_from_thread(
            self.app.post_message, AdversarialCycleEnd(cycle=cycle, metrics=metrics)
        )

    def on_vault_sample(self, sample: dict) -> None:
        if self.app is None:
            return
        from wintermute.tui.events import VaultSampleAdded

        self.app.call_from_thread(self.app.post_message, VaultSampleAdded(sample=sample))

    def on_log(self, text: str, level: str = "info") -> None:
        if self.app is None:
            return
        self.app.call_from_thread(self.app._log, text, level)


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
        from wintermute.tui.events import PipelineProgress

        self.app.call_from_thread(
            self.app.post_message,
            PipelineProgress(operation=operation, progress=progress, message=message),
        )

    def on_log(self, text: str, level: str = "info") -> None:
        if self.app is None:
            return
        self.app.call_from_thread(self.app._log, text, level)
