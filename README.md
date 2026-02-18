# 🧠 Wintermute

**MLX-powered malware classifier** that reads the "grammar" of compiled code to detect cyberattacks.

Instead of scanning raw bytes, Wintermute disassembles Windows PE binaries into Assembly language, extracts the opcode sequence, and feeds it through a **Bidirectional Transformer Encoder** — treating malware detection as an NLP problem.

Built for **Apple Silicon** with [MLX](https://github.com/ml-explore/mlx), running entirely in unified memory at `bfloat16` precision.

---

## ⚡ Quick Start

```bash
# 1. Create & activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate synthetic test data (no real PE files needed)
python src/generate_synthetic_data.py --n-samples 500

# 4. Train the model
python src/03_train.py --epochs 20

# 5. Scan a file
python scan.py target_file.exe
```

---

## 🏗️ Architecture

```
PE Binary → Capstone Disassembler → Opcode Sequence → Transformer Encoder → Safe / Malicious
```

| Component | Spec |
|:---|:---|
| Embeddings | 128-D token + positional |
| Encoder | 4 × Pre-Norm blocks, 4 attention heads |
| FFN | 128 → 512 → 128 (GELU) |
| Pooling | Global mean over sequence |
| Head | Linear(128, 2) |
| Precision | `bfloat16` throughout |
| Params | ~1M |

---

## 📂 Project Structure

```text
wintermute/
├── data/
│   ├── raw/                        # PE files (safe/ & malicious/)
│   └── processed/                  # x_data.npy, y_data.npy, vocab.json
├── src/
│   ├── 01_build_dataset.py         # PE → opcode → .npy pipeline
│   ├── 02_model.py                 # MLX Transformer Encoder
│   ├── 03_train.py                 # Training loop (AdamW + cosine decay)
│   └── generate_synthetic_data.py  # Fake data for testing
├── scan.py                         # CLI scanner
├── requirements.txt
├── spec.md                         # Full specification
└── README.md
```

---

## 🔬 Pipeline Details

### 1. Data Ingestion (`src/01_build_dataset.py`)

- Parses PE headers with `pefile`
- Disassembles `.text` sections via `capstone` (x86-32 / x86-64)
- Extracts **only mnemonics** — operands are stripped (they change per compilation)
- Tokenises sequences to `MAX_SEQ_LENGTH = 2048`, pads/truncates
- Outputs `.npy` arrays + `vocab.json`

```bash
python src/01_build_dataset.py --data-dir data
```

### 2. Training (`src/03_train.py`)

- **Optimizer:** AdamW (weight decay 0.01)
- **LR Schedule:** Cosine decay from `3e-4`
- **Batch Size:** 8 (fits in 24 GB unified memory)
- **Precision:** `bfloat16` — 50% memory reduction
- **Checkpointing:** Best model saved as `.safetensors`

```bash
python src/03_train.py --epochs 20 --lr 3e-4 --batch-size 8
```

### 3. Inference (`scan.py`)

```bash
python scan.py suspicious_file.exe
```

Output:
```
✅  [SAFE]       Probability: 99.2%
```
or
```
🚨  [MALICIOUS]  Probability: 87.5%
```

---

## 📖 How to Train the Model

A step-by-step guide from raw PE files to a trained classifier.

### Step 1: Prepare Your Dataset

Collect Windows PE binaries and sort them into two folders:

```text
data/raw/
├── safe/           # Known-clean executables (.exe, .dll, .sys)
└── malicious/      # Confirmed malware samples
```

**Where to get samples:**
- **Safe files:** Copy system binaries from a Windows machine (`C:\Windows\System32\*.exe`), or download trusted open-source tools
- **Malware:** [MalwareBazaar](https://bazaar.abuse.ch/), [VX-Underground](https://vx-underground.org/) (⚠️ handle in an isolated VM only)

> 💡 **Tip:** Aim for at least 200–500 files per class. More data = better accuracy.

### Step 2: Build the Dataset

```bash
python src/01_build_dataset.py --data-dir data
```

This will:
1. Disassemble each PE file into opcodes via Capstone
2. Build the vocabulary (`vocab.json`)
3. Tokenise & pad all sequences to 2048 tokens
4. Save `x_data.npy` and `y_data.npy` to `data/processed/`

**No PE files yet?** Use synthetic data to test the pipeline:
```bash
python src/generate_synthetic_data.py --n-samples 500
```

### Step 3: Train

```bash
python src/03_train.py --epochs 20
```

**All available flags:**

| Flag | Default | Description |
|:---|:---|:---|
| `--epochs` | `20` | Number of training epochs |
| `--batch-size` | `8` | Samples per mini-batch |
| `--lr` | `3e-4` | Peak learning rate |
| `--weight-decay` | `0.01` | AdamW regularisation strength |
| `--max-seq-length` | `2048` | Must match the dataset |
| `--data-dir` | `data/processed` | Path to .npy + vocab.json |
| `--save-path` | `malware_model.safetensors` | Output weights file |

**Example — longer training with lower LR:**
```bash
python src/03_train.py --epochs 50 --lr 1e-4 --batch-size 4
```

### Step 4: Monitor Training

The training loop prints a live table:

```
Epoch        Loss   Train Acc     Val Acc      Time
────────────────────────────────────────────────────
    1      0.6931      52.3%      51.0%      5.2s
    2      0.5842      64.1%      60.5%      4.8s
    3      0.4127      78.6%      73.2%      4.9s
       ↑ new best — saved to malware_model.safetensors
   ...
```

**What to look for:**
- ✅ **Loss decreasing** each epoch — model is learning
- ✅ **Val Acc rising** — model generalises to unseen data
- ⚠️ **Train Acc ≫ Val Acc** — overfitting; try more data or fewer epochs
- ❌ **Loss stuck / NaN** — reduce `--lr` or check your dataset

### Step 5: Use the Trained Model

Once training completes, `malware_model.safetensors` is saved in the project root. Scan any PE file:

```bash
python scan.py suspicious_file.exe
```

To use a different model checkpoint or vocab:
```bash
python scan.py target.exe --model path/to/model.safetensors --vocab path/to/vocab.json
```

### 💾 Memory Troubleshooting

The model runs in `bfloat16` to fit in Apple Silicon's unified memory.

| Symptom | Fix |
|:---|:---|
| macOS starts swapping (slow) | Reduce `--batch-size` to 4 or 2 |
| Out-of-memory crash | Reduce `--max-seq-length` to 1024 |
| Training too slow | Increase `--batch-size` (if memory allows) |

> 💡 Open **Activity Monitor → Memory** during training. "Memory Used" should stay under ~16 GB on a 24 GB machine.

---

## ⚙️ Requirements

- **Hardware:** Apple Silicon Mac (M1/M2/M3/M4) with ≥16 GB unified memory
- **OS:** macOS 14+ (Sonoma)
- **Python:** 3.10+
- **Dependencies:** `mlx`, `capstone`, `pefile`, `numpy`

---

## ⚠️ Safety Notice

> **Never run real malware samples directly on your host machine.**
> Use an isolated VM or sandboxed environment for data ingestion.
> The `data/raw/` directory is git-ignored by default.

---

## 📄 License

MIT
