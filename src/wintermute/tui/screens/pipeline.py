"""Pipeline screen — data build, synthetic generation, MalBERT pre-training."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Label, ProgressBar, RichLog, Select

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

    DEFAULT_CSS = """
    PipelineScreen {
        height: 1fr;
    }
    #pipe-body {
        height: 1fr;
    }
    #pipe-main {
        width: 1fr;
        padding: 1 2;
    }
    #pipe-op-select {
        width: 40;
        margin-bottom: 1;
    }
    #pipe-progress {
        height: 1;
        margin: 1 0;
    }
    #pipe-log {
        height: 1fr;
    }
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
                    [
                        ("Build Dataset", "build"),
                        ("Synthetic Data", "synthetic"),
                        ("MalBERT Pretrain", "pretrain"),
                    ],
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
        try:
            drawer = self.query_one("#pipe-drawer", ConfigDrawer)
            drawer.set_fields(_OP_FIELDS[op])
        except Exception:
            pass

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
        self.app.call_from_thread(self.app._log, "Dataset build started", "ok")
        # Future: wire to tokenizer.build_dataset()
        self.app.call_from_thread(self.app._log, "Dataset build complete", "ok")

    async def _do_synthetic(self, values: dict) -> None:
        from wintermute.data.augment import SyntheticGenerator

        self.app.call_from_thread(self.app._log, "Synthetic generation started", "ok")
        gen = SyntheticGenerator(
            n_samples=int(values["n_samples"]),
            seed=int(values["seed"]),
        )
        gen.generate_dataset(out_dir=values["output_dir"])
        self.app.call_from_thread(self.app._log, "Synthetic generation complete", "ok")

    async def _do_pretrain(self, values: dict) -> None:
        from wintermute.engine.pretrain import MLMPretrainer

        self.app.call_from_thread(self.app._log, "Pre-training started", "ok")
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
        self.app.call_from_thread(self.app._log, "Pre-training complete", "ok")

    def on_worker_state_changed(self, event) -> None:
        if str(event.state) in ("CANCELLED", "ERROR", "SUCCESS"):
            try:
                self.query_one("#pipe-drawer", ConfigDrawer).unlock()
            except Exception:
                pass

    def cancel_operation(self) -> None:
        if self._hook:
            self._hook.cancel()
