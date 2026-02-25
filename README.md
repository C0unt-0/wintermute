# 🧠 Project Wintermute

**MLX-powered static malware analysis framework for Apple Silicon.**

Wintermute treats disassembled executables as an NLP problem — mapping opcode sequences and control flow graphs through deep learning models to classify malware families without relying on easily-bypassed file hashes.

---

## ✨ Features

| Capability | Status |
|:---|:---|
| Sequential opcode analysis (Transformer/MalBERT) | ✅ |
| Graph-based CFG analysis (GNN via angr) | ✅ |
| 9-class Microsoft Malware family classification | ✅ |
| Binary safe/malicious classification | ✅ |
| MLM pre-training (MalBERT) | ✅ |
| MLflow experiment tracking | ✅ |
| DVC data pipeline orchestration | ✅ |
| MalwareBazaar dataset downloader | ✅ |
| FastAPI REST inference server | 🚧 Phase 4 |
| Docker containerisation | 🚧 Phase 4 |

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

# 4. Train the sequence model
wintermute train --epochs 20

# 5. Scan a file
wintermute scan target.exe
```

---

## 📦 Installation Options

```bash
pip install -e .              # Base (MLX, tokenizer, CLI)
pip install -e ".[dev]"       # + pytest, ruff
pip install -e ".[mlops]"     # + MLflow tracking
pip install -e ".[api]"       # + FastAPI server
pip install -e ".[all]"       # Everything + angr (GNN)
```

**Requirements:** Apple Silicon Mac (M1–M4), macOS 14+, Python 3.10+

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
│   │   └── graphs/                 # CFG graph pickles (.pkl) for GNN
│   ├── bazaar/                     # MalwareBazaar downloads
│   └── ms-malware/                 # Microsoft dataset .asm files
├── src/wintermute/                 # Main Python package
│   ├── data/
│   │   ├── tokenizer.py            # PE/ASM opcode extraction + vocabulary
│   │   ├── cfg.py                  # angr CFG extractor (GNN pipeline)
│   │   ├── downloader.py           # MalwareBazaar downloader
│   │   └── augment.py              # SMOTE + heuristic augmentation
│   ├── models/
│   │   ├── sequence.py             # MalwareClassifier (Transformer v1)
│   │   ├── transformer.py          # MalBERT (Transformer v2)
│   │   └── gnn.py                  # MalwareGNN (GCN layers)
│   ├── engine/
│   │   ├── trainer.py              # Sequence model training loop
│   │   ├── gnn_trainer.py          # GNN training loop
│   │   ├── pretrain.py             # MalBERT MLM pre-training
│   │   ├── metrics.py              # F1-score, confusion matrix
│   │   └── tracking.py             # MLflow integration
│   └── cli.py                      # Unified Typer CLI
├── api/                            # FastAPI server (Phase 4)
├── legacy/                         # Original numbered scripts (reference)
├── dvc.yaml                        # DVC pipeline DAG
├── pyproject.toml                  # Modern dependency management
└── spec2.md                        # Full v2.0 technical specification
```

---

## 🏗️ Models

### MalwareClassifier — Sequence Transformer (v1)

| Component | Spec |
|:---|:---|
| Embeddings | 128-D token + positional |
| Encoder | 4 × Pre-Norm blocks, 4 attention heads |
| FFN | 128 → 512 → 128 (GELU) |
| Pooling | Global mean over sequence |
| Head | Linear(128, C) — 2 or 9 classes |
| Params | ~1M |

### MalBERT — Bidirectional Transformer (v2)

| Component | Spec |
|:---|:---|
| Embeddings | 256-D token + positional |
| Encoder | 6 × Pre-Norm blocks, 8 attention heads |
| Special tokens | `<PAD>`, `<UNK>`, `<CLS>`, `<SEP>`, `<MASK>` |
| Pre-training | Optional MLM (masked language modelling) |
| Head | Linear(256, C) for classification or Linear(256, V) for MLM |
| Params | ~4M |

### MalwareGNN — Graph Convolutional Network (v3)

| Component | Spec |
|:---|:---|
| Input | Control Flow Graph (nodes = basic blocks, edges = jumps/calls) |
| Node features | Mean-pooled opcode embeddings per basic block |
| GCN layers | 3 × Graph Convolutional layers (symmetric normalised adjacency) |
| Pooling | Global mean over all nodes |
| Head | Linear(128, C) |
| Extractor | `angr` CFGFast |

All models run at `bfloat16` precision on Apple Silicon.

---

## 🔧 CLI Reference

### Scan a file

```bash
# Binary classification (safe / malicious)
wintermute scan target.exe

# Multi-class family detection (9 MS malware families)
wintermute scan target.asm --family

# Custom model and vocab
wintermute scan target.exe --model path/to/model.safetensors --vocab path/to/vocab.json
```

### Train — Sequence Model

```bash
# Train with defaults
wintermute train

# With MLflow tracking
wintermute train --track --experiment "baseline" --run-name "v1-lr3e4"

# 9-class Microsoft malware families
wintermute train --num-classes 9 --epochs 50

# Override any parameter
wintermute train --epochs 50 --lr 1e-4 --batch-size 4
```

### Train — GNN (Graph Neural Network)

```bash
# Step 1: Extract CFGs from PE binaries
wintermute data cfg --data-dir data

# Step 2: Train the GNN
wintermute train-gnn --epochs 50 --num-classes 2

# Step 3: Custom save path
wintermute train-gnn --save-path gnn_model.safetensors
```

### MalBERT Pre-training

```bash
# Unsupervised MLM pre-training
wintermute pretrain --epochs 50 --data-dir data/processed

# Custom config
wintermute pretrain --config configs/malbert_config.yaml
```

### Evaluate

```bash
# Compute accuracy, F1, confusion matrix → eval_metrics.json
wintermute evaluate --data-dir data/processed --model malware_model.safetensors
```

### Data Pipeline

```bash
# Build dataset from raw PE files
wintermute data build --data-dir data

# Generate synthetic test data (no real PE files needed)
wintermute data synthetic --n-samples 500

# Download from MalwareBazaar
wintermute data download --families "AgentTesla,Emotet" --limit 50

# Extract Control Flow Graphs (for GNN)
wintermute data cfg --data-dir data --out-dir data/processed/graphs
```

### DVC Pipeline

```bash
pip install dvc
dvc repro                   # Run full pipeline: prepare → train → evaluate
dvc metrics show            # Show metrics.json + eval_metrics.json
dvc params diff             # Compare hyperparameter changes
```

---

## 🔬 Pipeline Details

### 1. Sequential Pipeline (NLP)

- Parses PE headers with `pefile`, disassembles `.text` via `capstone`
- Parses IDA Pro `.asm` files (Microsoft Malware dataset)
- Extracts **only mnemonics** — operands stripped (change per compilation)
- Tokenises to `MAX_SEQ_LENGTH = 2048`, pads/truncates
- Outputs `.npy` arrays + `vocab.json`

### 2. Graph Pipeline (GNN)

- Loads PE binary via `angr`, generates `CFGFast`
- Extracts basic blocks as nodes, jumps/calls as edges
- Node features = mean-pooled opcode embeddings per block
- Symmetric normalised adjacency: D⁻¹/² Ã D⁻¹/²
- Saves graph as `.pkl` in `data/processed/graphs/`

### 3. Data Augmentation

- **SMOTE** — oversamples minority classes via k-NN interpolation
- **Heuristic augmentation** — NOP insertion, dead code injection, instruction reordering

### 4. Training

- **Optimizer:** AdamW (weight decay 0.01)
- **LR Schedule:** Cosine decay from `3e-4`
- **Precision:** `bfloat16` — 50% memory reduction
- **Checkpointing:** Best model saved as `.safetensors`
- **Tracking:** Optional MLflow integration (`--track`)

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

```bash
# Download sample files
python src/download_ms_dataset.py

# Build dataset from .asm files
python src/04_build_ms_dataset.py --samples-dir data/ms-malware

# Train 9-class model
wintermute train --num-classes 9 --epochs 20
```

Full dataset: [Kaggle — Microsoft Malware Classification Challenge](https://www.kaggle.com/c/malware-classification/data)

---

## ⚙️ Memory Troubleshooting

| Symptom | Fix |
|:---|:---|
| macOS starts swapping | Reduce `--batch-size` to 4 or 2 |
| Out-of-memory crash | Reduce `--max-seq-length` to 1024 |
| angr CFG extraction slow | Use `CFGFast` (default) not `CFGAccurate` |
| Large graphs OOM | Limit nodes via `max_nodes` config |

---

## ⚠️ Safety Notice

> **Handle malware samples only in an isolated VM or air-gapped environment.**
> Never run unknown binaries on your host machine.
> Wintermute performs **static analysis only** — no samples are executed.
