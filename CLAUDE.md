# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Wintermute is an **MLX-powered static malware analysis framework** for Apple Silicon. It treats disassembled executables as an NLP problem — opcode sequences and control flow graphs feed into deep learning models to classify malware without executing binaries.

**Requirements:** Apple Silicon Mac (M1–M4), macOS 14+, Python 3.10+

---

## Setup & Installation

```bash
python3 -m venv venv && source venv/bin/activate

pip install -e .          # Base (MLX, tokenizer, CLI)
pip install -e ".[dev]"   # + pytest, ruff
pip install -e ".[api]"   # + FastAPI, Celery, Redis, r2pipe, networkx
pip install -e ".[all]"   # Everything + angr (GNN CFG extraction)
```

---

## Common Commands

### CLI (local, Apple Silicon)

```bash
wintermute scan target.exe                   # Binary safe/malicious classification
wintermute scan target.asm --family          # 9-class Microsoft family detection
wintermute train                             # Train sequence model with defaults
wintermute train --track --experiment "v1"   # Train with MLflow tracking
wintermute train --num-classes 9 --epochs 50 # 9-class malware family model
wintermute pretrain --epochs 50              # MalBERT MLM pre-training
wintermute evaluate --data-dir data/processed --model malware_model.safetensors
wintermute data synthetic --n-samples 500    # Generate synthetic test data
wintermute data build --data-dir data        # Build dataset from raw PE files
wintermute data cfg --data-dir data          # Extract CFGs for GNN
wintermute train-gnn --epochs 50 --num-classes 2
```

### DVC Pipeline

```bash
dvc repro           # Full pipeline: prepare → train → evaluate
dvc metrics show    # Show metrics.json + eval_metrics.json
dvc params diff     # Compare hyperparameter changes across runs
```

### Testing & Linting

```bash
pytest                                        # Run all tests
pytest tests/test_malbert.py                  # Run a single test file
pytest tests/test_tokenizer.py::test_name     # Run a single test
ruff check src/ tests/ api/                   # Lint
ruff format src/ tests/ api/                  # Auto-format
```

### Docker Stack (API + Worker)

```bash
docker compose up --build       # Start API, Celery worker, Redis, Flower
docker compose up api worker    # Start only the inference stack (no monitoring)
# API: http://localhost:8000
# Flower dashboard: http://localhost:5555
```

---

## Architecture

### Two Runtimes

| Mode | Entry Point | ML Backend | Disassembler |
|------|-------------|------------|--------------|
| Local CLI | `wintermute.cli` (Typer) | MLX (bfloat16, Apple Silicon) | capstone / pefile / angr |
| Docker API | `api/main.py` (FastAPI) | PyTorch (CUDA) | Radare2 (r2pipe) |

### Three Model Architectures

| Model | File | Input | Classes |
|-------|------|-------|---------|
| `MalwareClassifier` | `models/sequence.py` | Tokenized opcode sequence | 2 or 9 |
| `MalBERT` | `models/transformer.py` | Tokenized sequence + `<CLS>/<MASK>` tokens | 2, 9, or vocab (MLM) |
| `MalwareGNN` | `models/gnn.py` | CFG nodes + edges | 2 or 9 |

All models save/load via `.safetensors`. bfloat16 precision is used throughout for memory efficiency.

### Async API Pipeline (Docker)

```
POST /api/v1/scan → FastAPI (api/main.py)
    → saves binary to /tmp/wintermute_uploads/
    → enqueues analyze_binary_task via Celery → Redis broker
    → returns job_id (HTTP 202)

GET /api/v1/status/{job_id} → polls Celery result backend

Celery Worker (engine/worker.py):
    HeadlessDisassembler (r2pipe/Radare2)
        → sequence (str of opcodes) + CFG (networkx DiGraph)
    → ML inference (ensemble: 60% sequence, 40% GNN)
    → secure delete of raw binary
```

### Data Pipeline

```
Raw PE/.asm files
    → tokenizer.py (pefile + capstone) → x_data.npy, y_data.npy, vocab.json
    → cfg.py (angr CFGFast) → data/processed/graphs/*.pkl
    → augment.py (SMOTE + heuristic) → balanced training set
```

### Key Configuration Files

- `configs/data_config.yaml` — vocab sizes, `MAX_SEQ_LENGTH` (default 2048), data paths
- `configs/model_config.yaml` — model hyperparameters, training schedule (AdamW + cosine LR)
- `configs/malbert_config.yaml` — MalBERT-specific pre-training settings

---

## Code Layout

```
src/wintermute/
├── cli.py              # Unified Typer CLI — all `wintermute` subcommands
├── data/
│   ├── tokenizer.py    # PE/ASM opcode extraction + vocabulary building
│   ├── extractor.py    # HeadlessDisassembler (Radare2/r2pipe) — API pipeline
│   ├── cfg.py          # angr CFGFast extractor — CLI/GNN pipeline
│   ├── downloader.py   # MalwareBazaar API downloader
│   └── augment.py      # SMOTE + heuristic data augmentation
├── models/
│   ├── sequence.py     # MalwareClassifier (~1M params)
│   ├── transformer.py  # MalBERT (~4M params, supports MLM pre-training)
│   └── gnn.py          # MalwareGNN (3× GCN layers)
└── engine/
    ├── trainer.py      # Sequence model training loop
    ├── gnn_trainer.py  # GNN training loop
    ├── pretrain.py     # MalBERT MLM pre-training loop
    ├── metrics.py      # F1-score, confusion matrix, eval_metrics.json
    ├── tracking.py     # MLflow integration
    └── worker.py       # Celery task — async binary analysis

api/
├── main.py             # FastAPI server — /api/v1/scan, /api/v1/status/{id}
└── schemas.py          # Pydantic request/response models

tests/                  # pytest — test_malbert, test_model, test_tokenizer, test_cli, test_tracking
legacy/                 # Original numbered scripts (reference only, not imported)
```

---

## Important Notes

- **Safety:** Wintermute performs static analysis only — never execute unknown binaries on the host machine. Use an isolated VM for real malware samples.
- **Memory:** If macOS starts swapping, reduce `--batch-size` (try 4 or 2) or `--max-seq-length` (try 1024).
- The `worker.py` ML inference is currently stubbed with mock scores. Real model loading is commented out pending model path wiring.
- `legacy/` scripts are reference implementations — use the `src/wintermute` package instead.
