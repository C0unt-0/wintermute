# TUI Control Panel Design

**Date:** 2026-02-27
**Status:** Approved
**Scope:** Transform the Textual TUI from a read-only viewer into the full interactive control panel for Wintermute

---

## Context

The TUI currently has five screens (Dashboard, Scan, Training, Adversarial, Vault). Only Scan runs real inference. Training and Adversarial screens have hook-ready event handlers but no way to configure or launch operations. Users must drop to the CLI for everything except scanning.

This design makes the TUI the primary interactive interface — users configure parameters, launch operations, and monitor results all from the terminal UI. The CLI remains fully functional for scripting, CI/CD, and DVC pipelines.

---

## Decisions

- **Approach:** Config Drawers — each screen gets a collapsible side panel with form fields
- **Engine integration:** Shared engine, two frontends (TUI and CLI call the same Python API)
- **Config editing:** Form-based with defaults loaded from YAML configs
- **Operations scope:** Full pipeline — train, scan, evaluate, adversarial, data build, synthetic data, pretrain
- **Background tasks:** Operations run in worker threads; users can switch tabs while they run
- **Session persistence:** Ephemeral — results live in the current session only; MLflow/DVC handle history
- **No mid-run parameter changes:** Drawer locks during runs; cancel and restart to change config

---

## Section 1: Screen Structure & Navigation

### Tab Layout (6 tabs)

| # | Tab | Role | Config Drawer |
|---|-----|------|---------------|
| 1 | Dashboard | Session overview, live stats, activity log | No (read-only) |
| 2 | Scan | Binary classification | Yes — file path, model path, family toggle |
| 3 | Training | Joint training (fine-tune) | Yes — epochs, LR, batch size, phases, etc. |
| 4 | Adversarial | Red/blue team training | Yes — PPO, TRADES, EWC params, cycle count |
| 5 | Pipeline | Data ops: build, synthetic, pretrain | Yes — one form per operation, switchable |
| 6 | Vault | Adversarial sample browser | No (read-only, populated by adversarial runs) |

### Changes from Current State

- Scan screen: existing input formalized as a drawer with additional options
- Training screen: gains config drawer (currently display-only)
- Adversarial screen: gains config drawer (currently display-only)
- Pipeline screen: new tab, replaces CLI-only data operations
- Vault: moves from tab 5 to tab 6
- Dashboard: stays read-only, wired to live stats from running operations

### Keybindings

| Key | Action |
|-----|--------|
| `1`–`6` | Switch tabs |
| `c` | Toggle config drawer on current screen |
| `Enter` / Start button | Launch operation from drawer |
| `Ctrl+x` | Cancel running background operation |
| `q` | Quit |

### Global Status Bar

Persistent bar above the footer showing background task state:

```
[Training: epoch 23/50 ▪▪▪▪▪▪▪▪░░ 46%]  [Scan: idle]  [Pipeline: idle]
```

When idle: `Ready — press c to configure`

---

## Section 2: Config Drawer Architecture

### Behavior

Each drawer is a `Vertical` container occupying ~35% width on the right. The remaining ~65% shows live visualization.

**States:**
- **Collapsed (default):** Full-width visualization. Subtle `[c] Configure` hint.
- **Expanded:** Form fields on the right, visualization shrinks on the left.
- **Locked:** During a running operation, form is disabled. Shows "Running..." with cancel button.

### Form Fields

**Training Drawer:**

| Field | Widget | Default Source |
|-------|--------|---------------|
| Epochs | `Input` (int) | `model_config.yaml` |
| Learning Rate | `Input` (float) | `model_config.yaml` |
| Batch Size | `Input` (int) | `model_config.yaml` |
| Max Seq Length | `Select` (512/1024/2048) | `data_config.yaml` |
| Phase A Epochs | `Input` (int) | `model_config.yaml` |
| Num Classes | `Select` (2/9) | `model_config.yaml` |
| MLflow Tracking | `Switch` (on/off) | off |
| Experiment Name | `Input` (str) | `"default"` |
| [START TRAINING] | `Button` | |

**Adversarial Drawer:**

| Field | Widget | Default Source |
|-------|--------|---------------|
| Cycles | `Input` (int) | `adversarial_config.yaml` |
| TRADES beta | `Input` (float) | 1.0 |
| EWC lambda | `Input` (float) | 0.4 |
| PPO LR | `Input` (float) | 3e-4 |
| PPO Epochs | `Input` (int) | 4 |
| Evasion Target | `Input` (float) | 0.20-0.50 |
| [START ADVERSARIAL] | `Button` | |

**Scan Drawer:**

| Field | Widget | Default Source |
|-------|--------|---------------|
| File Path | `Input` (str) | — |
| Family Detection | `Switch` (on/off) | off |
| Model Path | `Input` (str) | auto-detect |
| [SCAN] | `Button` | |

**Pipeline Screen** (polymorphic form — `Select` dropdown switches operation):

| Operation | Key Fields |
|-----------|-----------|
| Build Dataset | `--data-dir`, `--max-seq-length`, `--vocab-size` |
| Synthetic Data | `--n-samples`, `--output-dir` |
| MalBERT Pretrain | `--epochs`, `--lr`, `--batch-size`, `--mask-prob` |

### Config Loading

On drawer open, defaults load from YAML configs via `OmegaConf`:
- `configs/model_config.yaml` — training params
- `configs/data_config.yaml` — seq length, vocab size, data paths
- `configs/malbert_config.yaml` — pretrain params

Form values are assembled into a config dict passed to the engine class.

---

## Section 3: Engine Integration & Background Tasks

### Worker Pattern

Operations run in background threads via Textual's `run_worker()`:

1. Drawer form produces a config dict
2. `run_worker()` spawns a thread
3. Thread instantiates engine class with TUI hook
4. Hook callbacks use `app.call_from_thread()` to post Textual Messages
5. `on_worker_done()` unlocks the drawer

### Engine-to-Hook Mapping

| Operation | Engine Class | Hook | Messages |
|-----------|-------------|------|----------|
| Train | `JointTrainer` | `TrainingHook` | `EpochComplete`, `ActivityLogEntry` |
| Adversarial | `AdversarialOrchestrator` | `AdversarialHook` | `AdversarialCycleEnd`, `AdversarialEpisodeStep`, `ActivityLogEntry` |
| Scan | inline pipeline | — | `ScanProgress` |
| Pretrain | `MLMPretrainer` | `TrainingHook` (reused) | `EpochComplete`, `ActivityLogEntry` |
| Data Build | `build_dataset()` | `PipelineHook` (new) | `PipelineProgress` (new), `ActivityLogEntry` |
| Synthetic | `SyntheticGenerator` | `PipelineHook` (new) | `PipelineProgress`, `ActivityLogEntry` |

### Cancellation

Each screen tracks its worker reference. `Ctrl+x` calls `worker.cancel()`, which sets a flag. Engine classes check `hook.cancelled` between epochs/steps and exit gracefully. Drawer unlocks on `on_worker_done()`.

---

## Section 4: Dashboard Wiring

### StatCard Updates

| Card | Source | Updated When |
|------|--------|-------------|
| Model Version | `.safetensors` metadata | App startup + training complete |
| Clean TPR | `EpochComplete.val_acc` | Training finishes |
| Adv TPR | `AdversarialCycleEnd.metrics["adv_tpr"]` | Each adversarial cycle |
| Macro F1 | `EpochComplete.f1` | Training finishes |
| Vault Size | `AdversarialCycleEnd.metrics["vault_size"]` | Each adversarial cycle |

### Family Chart

After training/evaluation with `num_classes=9`, updates with predicted class distribution. Trainer posts `EvaluationComplete` event with per-family counts.

### Activity Log

Unified event stream — all `ActivityLogEntry` messages from every hook land here, color-coded by level (info/ok/warn/error).

---

## Section 5: Pipeline Screen & Vault Wiring

### Pipeline Screen (New — Tab 5)

Single screen with a `Select` dropdown for three operations. Shared progress bar and output log, reset on each run. The config drawer form fields swap based on the selected operation.

### Vault Wiring (Tab 6)

- `AdversarialHook` gains `on_vault_sample(sample_dict)` callback
- Posts `VaultSampleAdded` event
- `VaultScreen` appends to `VaultTable`
- Row selection populates `DiffView` with mutation diff
- Ephemeral — clears on app restart

---

## Section 6: New & Modified Components

### New Files

| File | Purpose |
|------|---------|
| `tui/screens/pipeline.py` | Pipeline screen — data build, synthetic, pretrain |
| `tui/widgets/config_drawer.py` | Reusable drawer widget with form rendering |
| `tui/widgets/status_bar.py` | Global background task status bar |

### Modified Files

| File | Change |
|------|--------|
| `tui/app.py` | Add Pipeline tab (6 total), mount status bar, route new events |
| `tui/events.py` | Add `PipelineProgress`, `EvaluationComplete`, `VaultSampleAdded` |
| `tui/hooks.py` | Add `PipelineHook`, `on_vault_sample()`, cancellation flag |
| `tui/theme.py` | Add drawer/form CSS tokens |
| `tui/screens/training.py` | Add config drawer, wire `run_worker()` to `JointTrainer` |
| `tui/screens/adversarial.py` | Add config drawer, wire `run_worker()` to `AdversarialOrchestrator` |
| `tui/screens/scan.py` | Refactor input into drawer pattern |
| `tui/screens/dashboard.py` | Wire StatCards to completion events, activate activity log |
| `tui/screens/vault.py` | Wire `VaultTable` to `VaultSampleAdded` events |
| `engine/joint_trainer.py` | Accept optional `hook`, call `hook.on_epoch()` |
| `engine/pretrain.py` | Accept optional `hook`, call `hook.on_epoch()` |
| `adversarial/orchestrator.py` | Accept optional `hook`, call vault sample callback |
| `data/tokenizer.py` | Accept optional `hook` for progress reporting |
| `data/augment.py` | Accept optional `hook` for progress reporting |

### Explicitly Out of Scope

- No persistent run history (MLflow/DVC handle that)
- No mid-run parameter tuning
- No multi-run queuing
- No TUI-specific config format (reuse YAML)
- No new CLI commands (TUI launched via existing `wintermute tui`)
