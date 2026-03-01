# 🧠 Project Wintermute

![Python](https://img.shields.io/badge/python-≥3.10-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple)
![MLX](https://img.shields.io/badge/backend-MLX-orange)

**MLX-native malware classifier for Apple Silicon — treats disassembled binaries as an NLP problem.**

Wintermute reverse-engineers executables with Radare2, extracts both a linear opcode sequence and a control-flow graph, then classifies the sample through a unified deep learning model that fuses a Transformer encoder (MalBERT) with a Graph Attention Network (GAT) via cross-attention. An adversarial RL pipeline continuously probes the model for blind spots and hardens it against evasion.

Everything runs natively on Apple Silicon through [MLX](https://github.com/ml-explore/mlx) — no CUDA, no PyTorch.

---

## Table of Contents

- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Web UI](#-web-ui)
- [CLI Reference](#-cli-reference)
- [DVC Pipeline](#-dvc-pipeline)
- [Microsoft Malware Families](#-microsoft-malware-families-9-class)
- [Project Structure](#-project-structure)
- [Database Layer](#-database-layer)
- [Implementation Status](#-implementation-status)
- [Memory & Troubleshooting](#-memory--troubleshooting)
- [Safety](#-safety)

---

## 🏗️ Architecture

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

> [!NOTE]
> When no CFG is available (single `.asm` file, or Radare2 extraction fails), a learned `<NO_GRAPH>` embedding is used in place of the GAT output — the model degrades gracefully to sequence-only classification.

<details>
<summary><strong>DetectorConfig Defaults</strong></summary>

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

</details>

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

### Database Layer (Phase 6)

All scan results, models, training runs, adversarial cycles, and ETL provenance are persisted in a relational database.

```
                  ┌───────────────────────────────────────────┐
                  │           Wintermute Database              │
                  │                                           │
  CLI / API ────▶ │  ┌─────────────┐     ┌───────────────┐   │
                  │  │  SampleRepo  │     │  EmbeddingRepo │   │
                  │  │  ScanRepo    │     │  (k-NN vector  │   │
                  │  │  ModelRepo   │     │   similarity)  │   │
                  │  │  Adversarial │     └───────────────┘   │
                  │  │  Repo        │                         │
                  │  └──────┬──────┘                         │
                  │         │                                 │
                  │         ▼                                 │
                  │  ┌─────────────────────────────────────┐  │
                  │  │  SQLAlchemy ORM (8 tables)           │  │
                  │  │  + Alembic migrations                │  │
                  │  └──────┬──────────────┬───────────────┘  │
                  │         │              │                   │
                  │    ┌────▼────┐    ┌────▼──────┐           │
                  │    │ SQLite  │    │ PostgreSQL │           │
                  │    │ + vec   │    │ + pgvector │           │
                  │    │ (local) │    │  (Docker)  │           │
                  │    └─────────┘    └───────────┘           │
                  └───────────────────────────────────────────┘
```

| Backend | Use case | Vector search |
| :------ | :------- | :------------ |
| SQLite + [sqlite-vec](https://github.com/asg017/sqlite-vec) | Local CLI development | `vec_distance_L2()` |
| PostgreSQL 17 + [pgvector](https://github.com/pgvector/pgvector) | Docker production stack | `<->` L2 operator |

**8 tables:** `samples`, `scan_results`, `models`, `training_runs`, `adversarial_cycles`, `adversarial_variants`, `etl_runs`, `etl_run_sources`

---

## 🚀 Quick Start

> [!TIP]
> Generate synthetic data to try Wintermute without real malware samples.

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

# Launch the web UI
pip install -e ".[api]"
cd web && npm install && npm run dev    # Frontend on http://localhost:5173
uvicorn api.main:app --reload           # Backend on http://localhost:8000
```

> [!IMPORTANT]
> **Requirements:** Apple Silicon Mac (M1–M4), macOS 14+, Python 3.10+

---

## 📦 Installation

```bash
pip install -e .                 # Base: MLX, tokenizer, CLI
pip install -e ".[dev]"          # + pytest, ruff
pip install -e ".[mlops]"        # + MLflow tracking
pip install -e ".[adversarial]"  # + Gymnasium, LIEF
pip install -e ".[api]"          # + FastAPI, Celery, r2pipe (Web UI backend)
pip install -e ".[db]"           # + SQLAlchemy, Alembic, sqlite-vec, psycopg
pip install -e ".[all]"          # Everything (api + mlops + dev + adversarial + db + angr)
```

---

## 🖥️ Web UI

A 6-tab React dashboard for real-time monitoring and control, styled with a **Terminal Noir** dark theme.

```bash
# Development (two terminals)
cd web && npm install && npm run dev    # Vite dev server → http://localhost:5173
uvicorn api.main:app --reload           # FastAPI backend → http://localhost:8000

# Production (Docker — includes PostgreSQL + pgvector)
docker compose up --build               # Web UI + API → http://localhost:8000
```

| Tab             | What it shows                                                                  |
| :-------------- | :----------------------------------------------------------------------------- |
| **Dashboard**   | Model stats, 9-family distribution chart (Recharts), live activity log         |
| **Scan**        | Drag-and-drop file upload → async analysis → verdict with confidence bars     |
| **Training**    | Config panel, Phase A/B epoch table, loss/accuracy sparklines, live WebSocket  |
| **Adversarial** | Red vs Blue team cards, episode action log, cycle metrics, evasion sparklines  |
| **Pipeline**    | Operation selector (build/synthetic/pretrain), progress bar, log output        |
| **Vault**       | Adversarial sample table with mutation diff viewer                             |

The web UI receives real-time updates via WebSocket (`/api/v1/ws`). The engine emits transport-agnostic events (`engine/events.py`) through callback hooks (`engine/hooks.py`), which the API routers bridge to WebSocket broadcasts.

---

## 🔧 CLI Reference

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

# ── Database ──
wintermute db init                               # Create all tables
wintermute db migrate                            # Run Alembic migrations
wintermute db stats                              # Show database statistics
wintermute db samples --family Ramnit            # Query samples by family
wintermute db scans --recent 10                  # Show recent scan results
wintermute db scans --uncertain 0.7              # Scans below confidence threshold
wintermute db models                             # List registered models
wintermute db models --promote <model-id>        # Promote model to active
wintermute db similar <sha256> --k 5             # k-NN embedding search
wintermute db vault                              # Adversarial vault stats
wintermute db embed                              # Embedding coverage stats

# ── Web UI ──
cd web && npm run dev                            # Frontend dev server
uvicorn api.main:app --reload                    # Backend API server
```

---

## 🔁 DVC Pipeline

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

## 🦠 Microsoft Malware Families (9-class)

<details>
<summary><strong>Family reference table</strong></summary>

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

</details>

Full dataset: [Kaggle — Microsoft Malware Classification Challenge](https://www.kaggle.com/c/malware-classification/data)

---

## 📂 Project Structure

<details>
<summary><strong>Expand full directory tree</strong></summary>

```
wintermute/
├── configs/
│   ├── data_config.yaml                # Vocab sizes, sequence lengths, paths
│   ├── model_config.yaml               # Training hyperparameters
│   ├── malbert_config.yaml             # MalBERT pre-training config
│   ├── database.yaml                   # Database URL + engine options
│   └── sources.yaml                    # ETL pipeline source configuration
├── data/
│   ├── raw/                            # PE files (safe/ & malicious/)
│   ├── processed/                      # x_data.npy, y_data.npy, vocab.json
│   │   └── graphs/                     # DisassemblyResult graphs for GAT
│   ├── bazaar/                         # MalwareBazaar downloads
│   └── ms-malware/                     # Microsoft dataset .asm files
├── src/wintermute/
│   ├── cli.py                          # Typer CLI — all commands
│   ├── data/
│   │   ├── tokenizer.py                # PE/ASM opcode extraction + vocabulary
│   │   ├── disassembler.py             # r2pipe → DisassemblyResult (seq + CFG)
│   │   ├── downloader.py               # MalwareBazaar downloader
│   │   ├── augment.py                  # Embedding Mixup + synthetic generation
│   │   └── etl/                        # ETL pipeline
│   │       ├── pipeline.py             # Orchestrator: extract → transform → load
│   │       ├── registry.py             # Source plugin registry
│   │       ├── base.py                 # ExtractResult, RawSample dataclasses
│   │       └── sources/                # Pluggable data sources
│   │           ├── synthetic.py        # Synthetic sample generator
│   │           ├── pe_files.py         # PE binary extractor
│   │           ├── ms_dataset.py       # Microsoft Malware Classification dataset
│   │           ├── malware_bazaar.py   # MalwareBazaar API source
│   │           └── asm_directory.py    # Pre-disassembled .asm directories
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
│   │   ├── worker.py                   # Celery async worker
│   │   ├── events.py                   # Transport-agnostic event dataclasses
│   │   └── hooks.py                    # Callback-based hooks (training, adversarial, pipeline)
│   ├── db/                             # Database persistence layer
│   │   ├── engine.py                   # SQLAlchemy engine factory + session management
│   │   ├── models.py                   # 8 ORM models (Sample, ScanResult, Model, ...)
│   │   ├── cli_db.py                   # `wintermute db` CLI subcommands
│   │   ├── repos/                      # Repository pattern — data access layer
│   │   │   ├── samples.py              # Sample CRUD + family/source queries
│   │   │   ├── scans.py                # Scan result persistence + history
│   │   │   ├── models_repo.py          # Model registry + promote/retire lifecycle
│   │   │   ├── adversarial.py          # Adversarial cycle + variant + vault CRUD
│   │   │   └── embeddings.py           # Vector similarity search (sqlite-vec / pgvector)
│   │   └── migrations/                 # Alembic schema migrations
│   │       └── versions/
│   │           └── 001_initial_schema.py
│   └── adversarial/
│       ├── orchestrator.py             # Red vs Blue training loop coordinator
│       ├── ppo.py                      # PPO agent (MLX-native actor-critic)
│       ├── environment.py              # Gymnasium env — AsmMutationEnv
│       ├── trades_loss.py              # TRADES loss with β warmup + EWC
│       ├── reward.py                   # Shaped reward (confidence Δ)
│       ├── oracle.py                   # Tiered mutation validator
│       ├── vault.py                    # Adversarial sample vault (50k cap)
│       ├── bridge.py                   # MLX ↔ numpy defender bridge
│       └── actions/
│           ├── code_actions.py         # NOP, dead code, substitution, reg swap
│           └── substitution_table.py   # Equivalent instruction mappings
├── api/
│   ├── main.py                         # FastAPI server + WebSocket + static file serving
│   ├── schemas.py                      # Pydantic request/response models
│   ├── dependencies.py                 # FastAPI dependency injection (DB sessions)
│   ├── ws.py                           # WebSocket connection manager
│   └── routers/
│       ├── dashboard.py                # Dashboard stats + activity log
│       ├── training.py                 # Training control endpoints
│       ├── adversarial.py              # Adversarial pipeline endpoints
│       ├── pipeline.py                 # Data pipeline operations
│       ├── vault.py                    # Adversarial vault viewer
│       └── db_endpoints.py            # Database-backed read API (samples, scans, models, k-NN)
├── web/                                # React + Vite + TypeScript + Tailwind frontend
│   ├── src/
│   │   ├── api/                        # API client + WebSocket client
│   │   ├── hooks/                      # React hooks (useWebSocket, useJob, useDashboard)
│   │   ├── components/                 # Shared UI (StatCard, ConfidenceBar, ConfigPanel, etc.)
│   │   ├── pages/                      # Dashboard, Scan, Training, Adversarial, Pipeline, Vault
│   │   └── styles/                     # Terminal Noir theme CSS
│   └── vite.config.ts                  # Vite config with Tailwind + API proxy
├── tests/
│   ├── test_cli.py                     # CLI command tests
│   ├── test_fusion.py                  # WintermuteMalwareDetector tests
│   ├── test_malbert.py                 # MalBERT encoder tests
│   ├── test_gat.py                     # GAT encoder tests
│   ├── test_tokenizer.py              # Tokenizer tests
│   ├── test_disassembler.py            # Headless disassembler tests
│   ├── test_metrics.py                 # Metrics tests
│   ├── test_pipeline.py                # End-to-end pipeline tests
│   ├── test_etl.py                     # ETL pipeline tests
│   ├── test_learning.py                # Learning convergence tests
│   ├── test_robustness.py              # Model robustness tests
│   ├── test_tracking.py                # MLflow tracking tests
│   ├── test_engine_events.py           # Engine event dataclass tests
│   ├── test_engine_hooks.py            # Engine hook tests
│   ├── test_api_schemas.py             # API Pydantic schema tests
│   ├── test_ws_manager.py              # WebSocket manager tests
│   ├── test_db_engine.py               # Database engine + session tests
│   ├── test_db_models.py               # ORM model tests
│   ├── test_db_repos.py                # Repository unit tests
│   ├── test_db_cli.py                  # `wintermute db` CLI tests
│   ├── test_db_etl_integration.py      # ETL → database integration tests
│   ├── test_cli_scan_db.py             # CLI scan → database persistence tests
│   ├── test_trainer_db_integration.py  # Training → database integration tests
│   ├── test_orchestrator_db_integration.py  # Adversarial → database integration tests
│   ├── test_api_db_endpoints.py        # Database API endpoint tests
│   ├── test_alembic.py                 # Alembic migration tests
│   └── adversarial/
│       ├── test_actions.py             # Code mutation action tests
│       ├── test_environment.py         # Gymnasium env tests
│       ├── test_oracle.py              # Oracle validation tests
│       ├── test_ppo.py                 # PPO agent tests
│       ├── test_reward.py              # Reward shaping tests
│       └── test_trades.py              # TRADES loss tests
├── alembic.ini                         # Alembic migration configuration
├── docs/plans/                         # Architecture design documents
├── legacy/                             # Original prototype scripts (01_–06_)
├── dvc.yaml                            # Reproducible pipeline DAG
└── pyproject.toml                      # Dependencies + extras
```

</details>

---

## 💾 Database Layer

Wintermute persists all operational data — samples, scan results, trained models, adversarial cycles, and ETL provenance — in a relational database with vector similarity search.

### Setup

```bash
# Install database dependencies
pip install -e ".[db]"

# Local development (SQLite — zero config)
wintermute db init                    # Creates data/wintermute.db

# Docker (PostgreSQL + pgvector — auto-provisioned)
docker compose up --build             # Includes pgvector/pgvector:pg17
```

### Configuration

Database URL is resolved in order: `WINTERMUTE_DATABASE_URL` env var → `configs/database.yaml` → SQLite default.

```yaml
# configs/database.yaml
database:
  url: "sqlite:///data/wintermute.db"
  echo: false
```

For Docker, the compose file sets `WINTERMUTE_DATABASE_URL=postgresql+psycopg://wintermute:...@db:5432/wintermute`.

### What Gets Persisted

| Operation | Stored Data |
| :-------- | :---------- |
| `wintermute scan` | Scan result, SHA256, predicted family, confidence, probabilities, model version |
| `wintermute train` | Training run metadata, epoch metrics, model registration (staged → active → retired) |
| `wintermute adv run` | Adversarial cycles, evasive variants, confidence deltas, vault membership |
| `wintermute data etl` | ETL run provenance, per-source extraction stats, sample catalog with embeddings |

### API Endpoints

The database exposes read-only REST endpoints:

| Endpoint | Description |
| :------- | :---------- |
| `GET /api/v1/stats` | Aggregate counts (samples, scans, models, families) |
| `GET /api/v1/samples/{sha256}` | Sample metadata by hash |
| `GET /api/v1/scans/recent` | Recent scan history |
| `GET /api/v1/models` | Model registry listing |
| `GET /api/v1/similar/{sha256}` | k-NN vector similarity search |

---

## 📊 Implementation Status

| Phase   | Scope                                                | Status                                 |
| :------ | :--------------------------------------------------- | :------------------------------------- |
| **1**   | Package restructure, CLI, config management          | ✅ Complete                            |
| **2**   | MLflow tracking, DVC pipelines                       | ✅ Complete                            |
| **3**   | MalBERT + GAT + Fusion unified model, joint training | ✅ Complete                            |
| **4**   | FastAPI server, Celery worker, Docker, Web UI        | ✅ Complete                            |
| **5**   | Adversarial RL pipeline (PPO + TRADES + vault)       | ✅ Complete                            |
| **6**   | Database persistence (SQLite/PostgreSQL, vector search, ETL) | ✅ Complete                     |

---

## 🛠️ Memory & Troubleshooting

| Symptom               | Fix                                                             |
| :-------------------- | :-------------------------------------------------------------- |
| macOS starts swapping | Reduce `--batch-size` to 4 or 2                                 |
| Out-of-memory crash   | Reduce `--max-seq-length` to 1024                               |
| Large graphs OOM      | Limit nodes via `max_nodes` in `disassembler.py` (default 5000) |
| `r2pipe` not found    | Install Radare2: `brew install radare2`                         |
| DB not initialized    | Run `wintermute db init` or `wintermute db migrate`              |
| `sqlite-vec` missing  | `pip install sqlite-vec` (included in `.[db]`)                   |
| Docker DB won't start | Check `DB_PASSWORD` env or use default: `wintermute_dev`         |

---

## ⚠️ Safety

> [!CAUTION]
> **Handle malware samples only in an isolated VM or air-gapped environment.**
> Never run unknown binaries on your host machine.
> Wintermute performs **static analysis only** — no samples are executed.
