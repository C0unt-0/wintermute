# 📄 Project Specification: MLX-ASM Malware Classifier

## 1. Executive Summary

* **Project Name:** Wintermute
* **Objective:** Build a high-performance, machine learning pipeline that classifies Windows Portable Executable (PE) files as either **Safe (0)** or **Malicious (1)**.
* **Methodology:** The system shifts away from analyzing raw binary bytes. Instead, it acts as a Natural Language Processing (NLP) model for computer code. It disassembles the binary into Assembly language, extracts the sequence of operational verbs (opcodes), and uses a **Bidirectional Transformer Encoder** to detect the structural "grammar" of a cyberattack.

---

## 2. Target Environment & Constraints

This architecture is specifically optimized to prevent $O(N^2)$ memory exhaustion on Apple Silicon, ensuring the GPU runs at 100% utilization without crashing the macOS system.

* **Hardware:** Apple Mac (M3 Chip) with 24 GB Unified Memory.
* **OS:** macOS 14+ (Sonoma or later).
* **Precision Paradigm:** The entire neural network and dataset will be cast to `bfloat16` (Brain Float 16-bit). This cuts RAM usage by exactly 50% while utilizing the M3's dedicated 16-bit hardware accelerators.
* **Core Dependencies:**
  * `mlx` (Apple's neural network framework)
  * `capstone` (x86/x64 Machine code disassembler)
  * `pefile` (Windows PE header parser)
  * `numpy` (High-speed matrix serialization)

---

## 3. Data Pipeline (ETL)

To prevent the Mac's CPU from bottlenecking the GPU during training, data extraction is completely decoupled from the training loop.

### Phase A: Extraction (`capstone` + `pefile`)

1. Read the raw `.exe` or `.dll` file.
2. Use `pefile` to locate the executable `.text` segment.
3. Feed the raw hex bytes into the `capstone` engine (x86_32 / x86_64 mode).
4. **Crucial Filtering:** Strip all operands, registers, and memory addresses (which change upon every compilation). Extract **only the mnemonics** (opcodes like `push`, `mov`, `xor`, `call`).

### Phase B: Tokenization & Serialization

1. **Vocabulary:** Build a dynamic dictionary mapping the ~1,500 standard x86 opcodes to integer IDs (e.g., `{"<PAD>": 0, "<UNK>": 1, "mov": 2, "xor": 3}`).
2. **Sequence Limits (`MAX_SEQ_LENGTH = 2048`):**
   * Truncate all files to a maximum of `2048` instructions. (Malware usually reveals unpacking stubs or malicious API imports early).
   * Pad files shorter than 2048 with the `<PAD>` token.
3. **Storage:** Export the final matrix as `x_data.npy` (features) and `y_data.npy` (labels) to the hard drive.

---

## 4. Model Architecture (The AI Engine)

A Bidirectional Transformer Encoder adapted from standard LLM architectures, but designed to output a single binary classification rather than generate text.

| Component | Specification | Description |
| :--- | :--- | :--- |
| **Input Shape** | `[Batch, 2048]` | Array of tokenized assembly instructions. |
| **Embeddings** | `128 Dimensions` | Size of the latent vector for each token + Positional Encoding. |
| **Encoder Blocks** | `4 Layers` | Stacked Pre-Norm Transformer blocks. |
| **Attention Heads** | `4 Heads` | 32 dimensions per head. Causal Masking is **disabled** (Bidirectional context). |
| **Feed-Forward** | `512 Dimensions` | Expands 128D -> 512D -> 128D using `GELU` activation. |
| **Pooling** | `Global Mean Pooling` | Averages the 2048 output vectors into one single 128D Master Summary vector. |
| **Classifier Head** | `Linear(128, 2)` | Projects the 128D summary into `[Prob Safe, Prob Malicious]`. |

---

## 5. Training Strategy

Optimized specifically for the 24GB Unified Memory architecture.

* **Batch Size:** `8` (Allows the 2048 x 2048 Attention matrices to fit comfortably in RAM alongside the OS).
* **Optimizer:** `AdamW` (Adaptive Moment Estimation with Weight Decay to prevent overfitting).
* **Learning Rate:** `3e-4` with a Cosine Decay schedule.
* **Loss Function:** `CrossEntropyLoss` (Compares the 2-class logits against the 1D target label).
* **Memory Management:**
  * Load the `.npy` arrays directly into MLX Unified Memory at runtime.
  * Apply `mx.bfloat16` casting to model weights immediately after initialization.

---

## 6. Target Folder Structure

```text
wintermute/
│
├── data/
│   ├── raw/                 # Ignored in git. Contains raw .exe files.
│   │   ├── safe/
│   │   └── malicious/
│   └── processed/           # x_data.npy, y_data.npy, vocab.json
│
├── src/
│   ├── 01_build_dataset.py  # Uses capstone & pefile to generate .npy files
│   ├── 02_model.py          # Contains the MLX Transformer classes
│   └── 03_train.py          # Loads .npy, sets bfloat16, runs MLX training
│
├── scan.py                  # CLI tool: `python scan.py file.exe`
├── requirements.txt         # capstone, pefile, mlx, numpy
└── spec.md                  # This document

🟩 Milestone 1: Environment & Sandboxing
[ ] Set up Python Virtual Environment (python -m venv venv).

[ ] Install MLX, Capstone, Pefile, and Numpy.

[ ] CRITICAL: If handling live malware (e.g., from VX-Underground), ensure data ingestion is done on an isolated VM or secure cloud bucket, never directly on the host Mac's main filesystem.

🟩 Milestone 2: The Data Engine (01_build_dataset.py)
[ ] Write the PE parser and Capstone disassembly loop.

[ ] Test extraction on a single safe file (e.g., calc.exe pulled from a Windows machine).

[ ] Build the Tokenizer (create the stoi dictionary), enforce the 2048 limit, and save to .npy arrays.

🟩 Milestone 3: The MLX Model (02_model.py)
[ ] Define the EncoderBlock (Attention + Pre-Norm + FFN). Ensure mask=None.

[ ] Define the MalwareClassifier (Embeddings + EncoderBlocks + Mean Pooling + Linear Head).

[ ] Implement the mx.bfloat16 type casting logic.

🟩 Milestone 4: Training & Profiling (03_train.py)
[ ] Load .npy files into unified memory.

[ ] Write the MLX value_and_grad loop.

[ ] Run a single batch. Open macOS Activity Monitor. Verify "Memory Used" stays below ~16GB to prevent swapping to the SSD.

[ ] Save model weights (mx.save_safetensors("malware_model.safetensors", model.state())) upon highest accuracy.

🟩 Milestone 5: Inference CLI (scan.py)
[ ] Write a command-line script: python scan.py target_file.exe.

[ ] The script should extract the opcodes on the fly, apply the vocab.json, run it through the MLX model, apply mx.softmax to the output logits, and print a terminal alert:


[OK] Safe Probability: 99.2%

[WARNING] Malicious Probability: 0.8% (OR)