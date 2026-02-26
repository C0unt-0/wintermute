# 🧠 Project Wintermute

**MLX-powered static malware analysis framework for Apple Silicon.**

Wintermute treats disassembled executables as an NLP problem — mapping opcode sequences and control flow graphs through a unified deep learning model to classify malware families without relying on easily-bypassed file hashes.

---

## ✨ Features

| Capability | Status |
|:---|:---|
| Unified multi-modal classifier (MalBERT + GAT + Fusion) | ✅ |
| Sequential opcode analysis (MalBERT, 6-layer Transformer with RoPE) | ✅ |
| Graph-based CFG analysis (GAT with sparse edge index) | ✅ |
| Cross-attention fusion (sequence ↔ graph) | ✅ |
| 9-class Microsoft Malware family classification | ✅ |
| Binary safe/malicious classification | ✅ |
| MLM pre-training (MalBERT) | ✅ |
| Joint two-phase training (frozen encoder → full finetune) | ✅ |
| Adversarial training (PPO attacker vs TRADES+EWC defender) | ✅ |
| Terminal User Interface (Textual) | ✅ |
| Embedding Mixup + adversarial opcode augmentation | ✅ |
| MLflow experiment tracking | ✅ |
| DVC data pipeline orchestration | ✅ |
| MalwareBazaar dataset downloader | ✅ |

---

## 🚀 Quick Start

```bash
# 1. Clone and create virtual environment
git clone <repo-url> && cd wintermute
python3 -m venv venv && source venv/bin/activate

# 2. Install (base)
pip install -e .

# 3. Generate synthetic test data
wintermute data synthetic --n-samples 500

# 4. Train the unified model
wintermute train --epochs 20

# 5. Scan a file
wintermute scan target.exe

# 6. Launch the TUI
pip install -e ".[tui]"
wintermute tui
```

---

## 📦 Installation Options

```bash
pip install -e .                 # Base (MLX, tokenizer, CLI)
pip install -e ".[dev]"          # + pytest, ruff
pip install -e ".[mlops]"        # + MLflow tracking
pip install -e ".[tui]"          # + Textual terminal UI
pip install -e ".[adversarial]"  # + Gymnasium, LIEF (Phase 5)
pip install -e ".[api]"          # + FastAPI, r2pipe
pip install -e ".[all]"          # Everything
```

**Requirements:** Apple Silicon Mac (M1–M4), macOS 14+, Python 3.10+

---

## 🖥️ Terminal User Interface

Wintermute includes a rich terminal UI built with [Textual](https://textual.textualize.io/). Launch it with:

```bash
wintermute tui
```

**5 tabs** (switch with keys `1`–`5`, quit with `q`):

| Tab | Description |
|:---|:---|
| **Dashboard** | System overview — model stats, architecture panel, family distribution, activity log |
| **Scan** | File scanning — input path, live disassembly log, verdict + confidence bars |
| **Training** | Live training visualization — sparklines for loss/accuracy, epoch table, phase indicators (A/B) |
| **Adversarial** | Red vs Blue team — evasion rate charts, episode action log, cycle metrics table |
| **Vault** | Adversarial sample browser — vault table with mutation diff viewer |

The TUI connects to training loops and adversarial orchestration via callback hooks (`wintermute.tui.hooks`), receiving real-time `EpochComplete` and `AdversarialCycleEnd` messages.

---

## 📂 Project Structure

```text
wintermute/
├── configs/                        # Centralized YAML configuration
│   ├── data_config.yaml            # Vocab sizes, sequence lengths, paths
│   ├── model_config.yaml           # Model & training hyperparameters
│   └── malbert_config.yaml         # MalBERT pre-training config
├── data/
│   ├── raw/                        # PE files (safe/ & malicious/)
│   ├── processed/                  # x_data.npy, y_data.npy, vocab.json
│   │   └── graphs/                 # DisassemblyResult graphs for GAT
│   ├── bazaar/                     # MalwareBazaar downloads
│   └── ms-malware/                 # Microsoft dataset .asm files
├── src/wintermute/                 # Main Python package
│   ├── data/
│   │   ├── tokenizer.py            # PE/ASM opcode extraction + vocabulary
│   │   ├── disassembler.py         # Unified r2pipe extractor → DisassemblyResult
│   │   ├── downloader.py           # MalwareBazaar downloader
│   │   └── augment.py              # Embedding Mixup + heuristic augmentation
│   ├── models/
│   │   ├── transformer.py          # MalBERT encoder (6L, 8H, RoPE)
│   │   ├── gat.py                  # GAT encoder (3L, sparse edge index)
│   │   └── fusion.py               # Cross-attention fusion + WintermuteMalwareDetector
│   ├── engine/
│   │   ├── joint_trainer.py        # Two-phase joint training loop
│   │   ├── pretrain.py             # MalBERT MLM pre-training
│   │   ├── trainer.py              # Sequence-only training (legacy compat)
│   │   ├── metrics.py              # F1, AUC-ROC, FPR@FNR, confusion matrix
│   │   └── tracking.py             # MLflow integration
│   ├── adversarial/                # Phase 5 — adversarial training
│   │   ├── orchestrator.py         # Red vs Blue training loop
│   │   ├── ppo.py                  # PPO agent (MLX-native)
│   │   ├── environment.py          # Gymnasium RL environment
│   │   ├── trades_loss.py          # TRADES + EWC defender loss
│   │   ├── reward.py               # Attacker reward shaping
│   │   ├── vault.py                # Adversarial sample vault
│   │   └── actions/                # Code mutation actions
│   ├── tui/                        # Terminal User Interface
│   │   ├── app.py                  # Main WintermuteApp (Textual)
│   │   ├── theme.py                # Color tokens + TCSS stylesheet
│   │   ├── events.py               # Custom Textual messages
│   │   ├── hooks.py                # Training/adversarial → TUI bridge
│   │   ├── screens/                # Dashboard, Scan, Training, Adversarial, Vault
│   │   └── widgets/                # StatCard, ConfidenceBar, ActionLog, DiffView
│   └── cli.py                      # Unified Typer CLI
├── tests/                          # pytest test suite
├── dvc.yaml                        # DVC pipeline DAG
└── pyproject.toml                  # Modern dependency management
```

---

## 🏗️ Model Architecture

### WintermuteMalwareDetector — Unified Multi-Modal (v3.0)

```
Raw Binary → r2pipe → opcode sequence + CFG
                          │                │
                   Shared Token Embedding (256D)
                          │                │
                    MalBERT Encoder    GAT Encoder
                    (6L, 8H, RoPE)    (3L, 4H, sparse)
                          │                │
                    Cross-Attention Fusion
                          │
                    Linear → logits (2 or 9 classes)
```

| Component | Spec |
|:---|:---|
| Shared embedding | `nn.Embedding(vocab_size, 256)` — same for sequence + graph |
| MalBERT encoder | 6 layers, 8 heads, 1024 FFN, Pre-LayerNorm, RoPE |
| GAT encoder | 3 layers, 4 heads, sparse COO edge index |
| Fusion | Cross-attention — `[CLS]` queries over GAT node vectors |
| Graceful degradation | `<NO_GRAPH>` learned embedding if CFG extraction fails |
| Precision | `bfloat16` throughout |
| Checkpoint | `.safetensors` + `manifest.json` (vocab SHA256 validation) |

### Adversarial Training (Phase 5)

| Component | Spec |
|:---|:---|
| Attacker | PPO agent (MLX-native), lr=1e-4, γ=0.5, 256-step rollouts |
| Defender | WintermuteMalwareDetector with TRADES β=1.0 + EWC λ=0.4 |
| Actions | NOP insertion, dead code injection, instruction substitution, register swap |
| Environment | Gymnasium with opcode sequence state space |

---

## 🔧 CLI Reference

### Scan a file

```bash
# Binary classification (safe / malicious)
wintermute scan target.exe

# Multi-class family detection (9 MS malware families)
wintermute scan target.asm --family
```

### Train

```bash
# Train unified model with defaults
wintermute train

# With pre-trained MalBERT encoder
wintermute train --pretrained malbert_pretrained.safetensors

# With MLflow tracking
wintermute train --track --experiment "baseline"

# 9-class malware families
wintermute train --num-classes 9 --epochs 50
```

### MalBERT Pre-training

```bash
wintermute pretrain --epochs 50 --data-dir data/processed
```

### Adversarial Training

```bash
wintermute adv train --cycles 10 --episodes 50
```

### Evaluate

```bash
wintermute evaluate --data-dir data/processed --model malware_detector.safetensors
```

### Terminal UI

```bash
wintermute tui
```

### Data Pipeline

```bash
wintermute data build --data-dir data        # Build dataset from raw PE files
wintermute data synthetic --n-samples 500    # Generate synthetic test data
wintermute data download --families "AgentTesla,Emotet" --limit 50
```

### DVC Pipeline

```bash
dvc repro                   # Full pipeline: disassemble → pretrain → train → evaluate
dvc metrics show            # Show metrics.json + eval_metrics.json
dvc params diff             # Compare hyperparameter changes
```

---

## 🦠 Microsoft Malware Dataset (9-class)

| Class | Family | Type | Samples |
|:---|:---|:---|:---|
| 1 | Ramnit | Worm | 1,541 |
| 2 | Lollipop | Adware | 2,478 |
| 3 | Kelihos_ver3 | Backdoor | 2,942 |
| 4 | Vundo | Trojan | 475 |
| 5 | Simda | Backdoor | 42 |
| 6 | Tracur | Trojan Downloader | 751 |
| 7 | Kelihos_ver1 | Backdoor | 398 |
| 8 | Obfuscator.ACY | Obfuscated | 1,228 |
| 9 | Gatak | Backdoor | 1,013 |

Full dataset: [Kaggle — Microsoft Malware Classification Challenge](https://www.kaggle.com/c/malware-classification/data)

---

## ⚙️ Memory Troubleshooting

| Symptom | Fix |
|:---|:---|
| macOS starts swapping | Reduce `--batch-size` to 4 or 2 |
| Out-of-memory crash | Reduce `--max-seq-length` to 1024 |
| Large graphs OOM | Limit nodes via `max_nodes` config |

---

## ⚠️ Safety Notice

> **Handle malware samples only in an isolated VM or air-gapped environment.**
> Never run unknown binaries on your host machine.
> Wintermute performs **static analysis only** — no samples are executed.
