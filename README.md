# 🧠 Project Wintermute

**MLX-native malware classifier for Apple Silicon — treats disassembled binaries as an NLP problem.**

Wintermute reverse-engineers executables with Radare2, extracts both a linear opcode sequence and a control-flow graph, then classifies the sample through a unified deep learning model that fuses a Transformer encoder (MalBERT) with a Graph Attention Network (GAT) via cross-attention. An adversarial RL pipeline continuously probes the model for blind spots and hardens it against evasion.

Everything runs natively on Apple Silicon through [MLX](https://github.com/ml-explore/mlx) — no CUDA, no PyTorch.

---

## Architecture

```
                           ┌──────────────────────────────────────────────────────────────────┐
                           │                 WintermuteMalwareDetector v3.0                    │
                           │                                                                  │
  Raw .exe / .asm          │   ┌──────────────────────────┐                                   │
        │                  │   │  Shared Token Embedding   │                                   │
        ▼                  │   │    nn.Embedding(V, 256)   │                                   │
  ┌───────────┐            │   └────────┬─────────┬────────┘                                   │
  │  Radare2  │            │            │         │                                            │
  │ Headless  │            │            ▼         ▼                                            │
  │Disassembly│            │   ┌────────────┐  ┌────────────┐                                  │
  └─────┬─────┘            │   │   MalBERT   │  │    GAT     │                                  │
        │                  │   │  Encoder    │  │  Encoder   │                                  │
        ├──── opcodes ────▶│   │  6L · 8H    │  │  3L · 4H   │                                  │
        │                  │   │  1024 FFN   │  │  sparse    │                                  │
        └──── CFG ────────▶│   │  RoPE PE    │  │  COO edges │                                  │
                           │   └─────┬──────┘  └──────┬─────┘                                  │
                           │         │ [CLS]          │ mean-pool                              │
                           │         ▼                ▼                                        │
                           │   ┌─────────────────────────────┐                                 │
                           │   │  Cross-Attention Fusion      │                                 │
                           │   │  [CLS] queries ← GAT nodes  │                                 │
                           │   │  4 heads · LayerNorm         │                                 │
                           │   └──────────────┬──────────────┘                                 │
                           │                  ▼                                                │
                           │   concat(seq_cls, fused_graph) → Linear(512, 256) → GELU          │
                           │                  ▼                                                │
                           │          Linear(256, C) → logits                                  │
                           │          C = 2 (binary) or 9 (family)                             │
                           └──────────────────────────────────────────────────────────────────┘
```

When no CFG is available (single `.asm` file, or Radare2 extraction fails), a learned `<NO_GRAPH>` embedding is used in place of the GAT output — the model degrades gracefully to sequence-only classification.

### DetectorConfig Defaults

| Parameter          | Value          | Notes                               |
| :----------------- | :------------- | :---------------------------------- |
| `dims`             | 256            | Shared embedding + hidden dimension |
| `num_layers`       | 6              | MalBERT Transformer blocks          |
| `num_heads`        | 8              | MalBERT attention heads             |
| `mlp_dims`         | 1024           | FFN inner dimension                 |
| `gat_layers`       | 3              | GAT encoder depth                   |
| `gat_heads`        | 4              | GAT attention heads                 |
| `num_fusion_heads` | 4              | Cross-attention fusion heads        |
| `max_seq_length`   | 2048           | Tokens before `[CLS]`/`[SEP]`       |
| `dropout`          | 0.1            | All sub-layers                      |
| Precision          | `bfloat16`     | Cast via `cast_to_bf16()`           |
| Checkpoint         | `.safetensors` | + `manifest.json` with vocab SHA256 |

### Training Pipeline

The `JointTrainer` runs two phases:

**Phase A — Encoder frozen.** MalBERT encoder gradients are zeroed. Only the GAT, fusion layer, and classifier receive updates. This stabilises the pre-trained encoder weights.

**Phase B — Full finetune.** All parameters update. The encoder receives a differential learning rate (0.1× base LR). Cosine decay with linear warmup. Gradient clipping at 1.0.

Both phases support embedding-space Mixup augmentation (`augment.py`) and graph collation with no-graph fallback.

### Adversarial Training (Phase 5)

An RL agent learns to mutate malware samples until they evade the defender, then the defender retrains on the evasive samples.

```
  ┌─────────────────────┐             ┌────────────────────────┐
  │  RED TEAM (Attacker) │             │  BLUE TEAM (Defender)   │
  │                     │             │                        │
  │  PPO Agent (MLX)    │  mutated    │  WintermuteMalwareDet. │
  │  lr=1e-4 · γ=0.5   │──sample───▶│  + TRADES loss β=1.0  │
  │  256-step rollouts  │             │  + EWC λ=0.4          │
  │                     │◀─conf/det──│                        │
  └──────────┬──────────┘             └───────────┬────────────┘
             │                                    │
     shaped reward                         vault replay
     (confidence Δ)                    (stratified sampling)
             │                                    │
             ▼                                    ▼
  ┌─────────────────────┐             ┌────────────────────────┐
  │  Gymnasium Env       │             │  Adversarial Vault     │
  │  AsmMutationEnv     │             │  max 50k samples       │
  │  4 action types:    │             │  evasive variants for  │
  │  NOP · dead code ·  │             │  defender retraining   │
  │  substitution ·     │             │                        │
  │  register swap      │             │                        │
  └─────────────────────┘             └────────────────────────┘
```

Mutations are validated by a `TieredOracle` (token-level structural checks; CFG diff and sandbox tiers planned). The `AdversarialOrchestrator` coordinates the full attack → update → store → retrain loop.

---

## Quick Start

```bash
# Clone and set up
git clone https://github.com/C0unt-0/wintermute.git && cd wintermute
python3 -m venv venv && source venv/bin/activate

# Install base
pip install -e .

# Generate synthetic test data
wintermute data synthetic --n-samples 500

# Train the unified model (Phase A → Phase B)
wintermute train --epochs 20

# Scan a file
wintermute scan target.exe

# Launch the terminal UI
pip install -e ".[tui]"
wintermute tui
```

**Requirements:** Apple Silicon Mac (M1–M4), macOS 14+, Python 3.10+

---

## Installation

```bash
pip install -e .                 # Base: MLX, tokenizer, CLI
pip install -e ".[dev]"          # + pytest, ruff
pip install -e ".[mlops]"        # + MLflow tracking
pip install -e ".[tui]"          # + Textual terminal UI
pip install -e ".[adversarial]"  # + Gymnasium, LIEF
pip install -e ".[api]"          # + FastAPI, Celery, r2pipe
pip install -e ".[all]"          # Everything
```

---

## Terminal UI

A 5-tab Textual interface for real-time monitoring. Launch with `wintermute tui`.

| Key | Tab             | What it shows                                                                  |
| :-- | :-------------- | :----------------------------------------------------------------------------- |
| `1` | **Dashboard**   | Model stats, architecture panel, 9-family distribution chart, activity log     |
| `2` | **Scan**        | File path input → live disassembly log + verdict with confidence bars          |
| `3` | **Training**    | Phase A/B indicator, loss/accuracy sparklines, epoch table                     |
| `4` | **Adversarial** | Red vs Blue team cards, evasion rate charts, episode action log, cycle metrics |
| `5` | **Vault**       | Adversarial sample browser with mutation diff viewer                           |

The TUI connects to training and adversarial loops via callback hooks (`wintermute.tui.hooks`). `TrainingHook` emits `EpochComplete` messages; `AdversarialHook` emits `AdversarialEpisodeStep` and `AdversarialCycleEnd` messages. Press `q` to quit.

---

## CLI Reference

```bash
# ── Scan ──
wintermute scan target.exe                        # Binary: safe / malicious
wintermute scan target.asm --family               # 9-class family detection

# ── Train ──
wintermute train                                  # Joint training (Phase A → B)
wintermute train --pretrained malbert_pretrained.safetensors
wintermute train --num-classes 9 --epochs 50
wintermute train --track --experiment "baseline"  # MLflow tracking

# ── Pre-train ──
wintermute pretrain --epochs 50 --data-dir data/processed

# ── Adversarial ──
wintermute adv run --cycles 10 --episodes 500 --trades-beta 1.0

# ── Evaluate ──
wintermute evaluate --data-dir data/processed --model malware_detector.safetensors

# ── Data ──
wintermute data build --data-dir data             # Tokenize raw PEs
wintermute data synthetic --n-samples 500         # Synthetic test data
wintermute data download --families "AgentTesla,Emotet" --limit 50

# ── TUI ──
wintermute tui
```

---

## DVC Pipeline

The full reproducible pipeline is defined in `dvc.yaml`:

```
disassemble → pretrain → train → evaluate
```

```bash
dvc repro               # Run the full DAG
dvc metrics show        # Show metrics.json + eval_metrics.json
dvc params diff         # Compare hyperparameter changes
```

| Stage         | Command                 | Inputs                                     | Outputs                                                         |
| :------------ | :---------------------- | :----------------------------------------- | :-------------------------------------------------------------- |
| `disassemble` | `wintermute data build` | `tokenizer.py`, raw data                   | `x_data.npy`, `y_data.npy`, `vocab.json`                        |
| `pretrain`    | `wintermute pretrain`   | processed data, `transformer.py`           | `malbert_pretrained.safetensors`                                |
| `train`       | `wintermute train`      | processed data, graphs, pretrained weights | `malware_detector.safetensors`, `manifest.json`, `metrics.json` |
| `evaluate`    | `wintermute evaluate`   | model weights, test data                   | `eval_metrics.json`                                             |

---

## Microsoft Malware Families (9-class)

| ID  | Family         | Type              | Samples |
| :-- | :------------- | :---------------- | :------ |
| 0   | Ramnit         | Worm              | 1,541   |
| 1   | Lollipop       | Adware            | 2,478   |
| 2   | Kelihos_ver3   | Backdoor          | 2,942   |
| 3   | Vundo          | Trojan            | 475     |
| 4   | Simda          | Backdoor          | 42      |
| 5   | Tracur         | Trojan Downloader | 751     |
| 6   | Kelihos_ver1   | Backdoor          | 398     |
| 7   | Obfuscator.ACY | Obfuscated        | 1,228   |
| 8   | Gatak          | Backdoor          | 1,013   |

Full dataset: [Kaggle — Microsoft Malware Classification Challenge](https://www.kaggle.com/c/malware-classification/data)

---

## Project Structure

```
wintermute/
├── configs/
│   ├── data_config.yaml                # Vocab sizes, sequence lengths, paths
│   ├── model_config.yaml               # Training hyperparameters
│   └── malbert_config.yaml             # MalBERT pre-training config
├── data/
│   ├── raw/                            # PE files (safe/ & malicious/)
│   ├── processed/                      # x_data.npy, y_data.npy, vocab.json
│   │   └── graphs/                     # DisassemblyResult pickles for GAT
│   ├── bazaar/                         # MalwareBazaar downloads
│   └── ms-malware/                     # Microsoft dataset .asm files
├── src/wintermute/
│   ├── cli.py                          # Typer CLI — all commands
│   ├── data/
│   │   ├── tokenizer.py                # PE/ASM opcode extraction + vocabulary
│   │   ├── disassembler.py             # r2pipe → DisassemblyResult (seq + CFG)
│   │   ├── downloader.py               # MalwareBazaar downloader
│   │   └── augment.py                  # Embedding Mixup + synthetic generation
│   ├── models/
│   │   ├── transformer.py              # MalBERT encoder (6L, 8H, RoPE, Pre-LN)
│   │   ├── gat.py                      # GAT encoder (sparse COO, scatter softmax)
│   │   └── fusion.py                   # Cross-attention fusion + WintermuteMalwareDetector
│   ├── engine/
│   │   ├── joint_trainer.py            # Two-phase training (frozen → finetune)
│   │   ├── pretrain.py                 # MalBERT MLM pre-training (15% masking)
│   │   ├── trainer.py                  # Sequence-only training (legacy compat)
│   │   ├── metrics.py                  # Accuracy, macro F1, confusion matrix
│   │   ├── tracking.py                 # MLflow integration
│   │   └── worker.py                   # Celery async worker (Phase 4)
│   ├── adversarial/
│   │   ├── orchestrator.py             # Red vs Blue training loop coordinator
│   │   ├── ppo.py                      # PPO agent (MLX-native actor-critic)
│   │   ├── environment.py              # Gymnasium env — AsmMutationEnv
│   │   ├── trades_loss.py              # TRADES loss with β warmup + EWC
│   │   ├── reward.py                   # Shaped reward (confidence Δ)
│   │   ├── oracle.py                   # Tiered mutation validator
│   │   ├── vault.py                    # Adversarial sample vault (50k cap)
│   │   ├── bridge.py                   # MLX ↔ numpy defender bridge
│   │   └── actions/
│   │       ├── code_actions.py         # NOP, dead code, substitution, reg swap
│   │       └── substitution_table.py   # Equivalent instruction mappings
│   └── tui/
│       ├── app.py                      # WintermuteApp — 5-tab Textual TUI
│       ├── theme.py                    # Color tokens + TCSS stylesheet
│       ├── events.py                   # Custom Textual messages
│       ├── hooks.py                    # Training/adversarial → TUI bridge
│       ├── screens/
│       │   ├── dashboard.py            # System overview + architecture panel
│       │   ├── scan.py                 # Live scan with real inference pipeline
│       │   ├── training.py             # Phase A/B visualization
│       │   ├── adversarial.py          # Red vs Blue (graceful if Phase 5 absent)
│       │   └── vault.py                # Adversarial sample browser
│       └── widgets/
│           ├── stat_card.py            # Key metric display
│           ├── confidence_bar.py       # Horizontal confidence bar
│           ├── action_log.py           # Scrollable action/event log
│           └── diff_view.py            # Before/after mutation diff
├── api/
│   ├── main.py                         # FastAPI server (Phase 4, async)
│   └── schemas.py                      # Pydantic request/response models
├── tests/
│   ├── test_cli.py                     # CLI command tests
│   ├── test_fusion.py                  # WintermuteMalwareDetector tests
│   ├── test_malbert.py                 # MalBERT encoder tests
│   ├── test_gat.py                     # GAT encoder tests
│   ├── test_tokenizer.py              # Tokenizer tests
│   ├── test_disassembler.py            # Headless disassembler tests
│   ├── test_metrics.py                 # Metrics tests
│   ├── test_pipeline.py                # End-to-end pipeline tests
│   ├── test_learning.py                # Learning convergence tests
│   ├── test_robustness.py              # Model robustness tests
│   ├── test_tracking.py                # MLflow tracking tests
│   ├── test_tui.py                     # TUI widget + screen tests
│   └── adversarial/
│       ├── test_actions.py             # Code mutation action tests
│       ├── test_environment.py         # Gymnasium env tests
│       ├── test_oracle.py              # Oracle validation tests
│       ├── test_ppo.py                 # PPO agent tests
│       ├── test_reward.py              # Reward shaping tests
│       └── test_trades.py              # TRADES loss tests
├── docs/plans/                         # Architecture design documents
├── legacy/                             # Original prototype scripts (01_–06_)
├── dvc.yaml                            # Reproducible pipeline DAG
└── pyproject.toml                      # Dependencies + extras
```

---

## Implementation Status

| Phase   | Scope                                                | Status                                 |
| :------ | :--------------------------------------------------- | :------------------------------------- |
| **1**   | Package restructure, CLI, config management          | ✅ Complete                            |
| **2**   | MLflow tracking, DVC pipelines                       | ✅ Complete                            |
| **3**   | MalBERT + GAT + Fusion unified model, joint training | ✅ Complete                            |
| **4**   | FastAPI server, Celery worker, Docker                | 🚧 API scaffolded, worker needs update |
| **5**   | Adversarial RL pipeline (PPO + TRADES + vault)       | ✅ Complete                            |
| **TUI** | Textual terminal interface (5 screens)               | ✅ Complete                            |

---

## Memory & Troubleshooting

| Symptom               | Fix                                                             |
| :-------------------- | :-------------------------------------------------------------- |
| macOS starts swapping | Reduce `--batch-size` to 4 or 2                                 |
| Out-of-memory crash   | Reduce `--max-seq-length` to 1024                               |
| Large graphs OOM      | Limit nodes via `max_nodes` in `disassembler.py` (default 5000) |
| `r2pipe` not found    | Install Radare2: `brew install radare2`                         |

---

## Safety

> **Handle malware samples only in an isolated VM or air-gapped environment.**
> Never run unknown binaries on your host machine.
> Wintermute performs **static analysis only** — no samples are executed.
