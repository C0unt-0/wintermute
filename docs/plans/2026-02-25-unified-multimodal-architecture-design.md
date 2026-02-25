# Wintermute v3.0 — Unified Multi-Modal Architecture Design

**Date:** 2026-02-25
**Status:** Approved
**Scope:** Model architecture + training data pipeline (no API layer changes)

---

## Context

The current codebase has three critical blockers preventing end-to-end operation:

1. `MalwareClassifier` and `MalBERT` are separate models with no promotion path — the CLI uses only `MalwareClassifier`, and the pre-train → fine-tune pipeline is not wired
2. The GNN (`gnn.py`) has broken gradient tracking (`weight`/`bias` stored as raw `mx.array` instead of `nn.Linear` parameters), an O(N²) dense adjacency matrix, and `batch_size=1`
3. Training CFG extraction uses `angr`; inference uses `r2pipe` — incompatible formats

The redesign removes all three blockers and replaces the parallel-model ensemble with a single jointly-trained multi-modal architecture.

---

## Decisions

- **Remove:** `angr`, `MalwareClassifier` (`sequence.py`), `MalwareGNN` / `GCNLayer` (`gnn.py`), `cfg.py`, `gnn_trainer.py`, `docker-compose.yml`, `Dockerfile`
- **Single disassembler:** `r2pipe/Radare2` for both training data extraction and inference
- **Single model:** `WintermuteMalwareDetector` — jointly-trained sequence + graph architecture
- **Deployment:** Native macOS processes on Mac Mini (MLX cannot run inside Linux Docker containers)

---

## Section 1: Model Architecture

### Overview

```
                    Raw Binary
                        │
               HeadlessDisassembler (r2pipe)
               ┌─────────────────────────┐
               │                         │
         opcode sequence           CFG (edge_index +
         [push, mov, xor, ...]      node opcode lists)
               │                         │
               ▼                         ▼
    ┌─── Shared Token Embedding ─────────┤
    │        (vocab_size → 256D)         │
    │                                    │
    ▼                                    ▼
MalBERT Encoder                   GAT Encoder
(6 layers, 8 heads, RoPE)     (3 layers, attention
    [CLS] → [B, 256D]           over neighbors)
                                  nodes → [N, 256D]
                                  mean pool → [B, 256D]
               │                         │
               └──────── Fusion ─────────┘
                    Cross-Attention
                 (seq [CLS] queries over
                    GAT node vectors)
                         │
                         ▼
                  [B, 256D] fused repr
                         │
                  Linear(256, num_classes)
                         │
                       logits
```

### Shared Token Embedding

Both the sequence encoder and the graph encoder share a single `nn.Embedding(vocab_size, 256)` layer. This forces consistent opcode representations across both modalities — the "meaning" of `xor` is the same whether it appears in the linear instruction stream or as part of a basic block in the CFG.

### MalBERT Encoder (upgraded)

- Architecture: 6 layers, 8 heads, 256D, Pre-LayerNorm, GELU FFN — unchanged
- **Upgrade:** Replace learned absolute positional embeddings (`nn.Embedding(max_seq_length, 256)`) with Rotary Positional Embeddings (RoPE)
  - RoPE has no learnable parameters
  - Encodes position as a rotation applied to Q/K vectors inside attention
  - Generalises to sequence lengths not seen during training
  - Standard in modern LLMs (LLaMA, Mistral)
- Input: opcode token sequence `[B, T]`
- Output: `[CLS]` hidden state `[B, 256D]`

### GAT Encoder (replaces GCN)

- Graph Attention Network: 3 layers
- Input: node features `[N, 256D]` (from shared embedding, mean-pooled per basic block) + `edge_index [2, E]` sparse COO format
- Each layer computes attention weights over neighbours, then weighted-aggregates
- More expressive than GCN: learns which neighbouring basic blocks are relevant rather than uniform averaging
- Output: per-node hidden states `[N, 256D]`, globally mean-pooled to `[B, 256D]` per graph

### Cross-Attention Fusion

- `[CLS]` token from MalBERT acts as query `[B, 1, 256D]`
- GAT node representations act as keys and values `[B, N, 256D]`
- Output: `[B, 256D]` fused representation
- Attention weights are interpretable: shows which basic blocks drove the prediction

### Graceful Degradation

If Radare2 CFG extraction fails (timeout, node limit exceeded, packed binary), the GAT output is replaced with a learned `<NO_GRAPH>` embedding `[B, 256D]`. The cross-attention fusion runs as normal. The model always produces a prediction — no silent failures.

### New Files

| File | Purpose |
|---|---|
| `src/wintermute/models/gat.py` | GAT encoder with sparse edge-index, proper `nn.Linear` parameters |
| `src/wintermute/models/fusion.py` | Cross-attention fusion + `WintermuteMalwareDetector` top-level model |
| `src/wintermute/data/disassembler.py` | Unified r2pipe extractor → `DisassemblyResult` dataclass |
| `src/wintermute/engine/joint_trainer.py` | Joint training loop for the unified model |

### Removed Files

| File | Reason |
|---|---|
| `src/wintermute/models/sequence.py` | `MalwareClassifier` replaced by `MalBERT` |
| `src/wintermute/models/gnn.py` | Broken GCN replaced by GAT |
| `src/wintermute/data/cfg.py` | `angr` removed, replaced by unified disassembler |
| `src/wintermute/engine/gnn_trainer.py` | Replaced by `joint_trainer.py` |
| `docker-compose.yml` | Redis/Celery removed |
| `Dockerfile` | Linux/CUDA image incompatible with MLX |

---

## Section 2: Data Pipeline

### DisassemblyResult Dataclass

```python
@dataclass
class DisassemblyResult:
    sequence: list[str]                          # flat ordered opcode mnemonics
    edge_index: tuple[list[int], list[int]]      # COO sparse: (src_nodes, dst_nodes)
    node_opcodes: list[list[str]]                # opcodes per basic block
    n_nodes: int
    n_edges: int
    extraction_failed: bool                      # True → model uses NO_GRAPH fallback
```

Single `r2pipe` pass per binary: `aaa` → `aflj` → `agj` per function. Two safety guards:
- **Timeout:** 30-second limit on r2pipe call. If exceeded, `extraction_failed=True`
- **Node limit:** If `n_nodes > 5000`, `extraction_failed=True`

### Dataset Storage Layout

```
data/processed/
├── vocab.json                  # token vocabulary (unchanged format)
├── manifest.json               # NEW: vocab_size, num_classes, n_samples, vocab_sha256
├── x_data.npy                  # encoded sequences [N, 2048] (unchanged)
├── y_data.npy                  # labels [N] (unchanged)
└── graphs/
    ├── graph_index.json        # {array_row_idx: "graphs/sample_id.pkl"}
    └── *.pkl                   # DisassemblyResult per sample
```

`graph_index.json` maps each row of `x_data.npy` to a graph file. Missing files or `extraction_failed=True` entries trigger the NO_GRAPH fallback — training continues without that graph.

### Graph Batching — Disjoint Packing

Variable-size graphs are packed into a single disjoint batch by offsetting node indices:

```
Graph 0: N=3, edges (0→1, 1→2)
Graph 1: N=2, edges (3→4)          ← indices offset by 3
Graph 2: N=4, edges (5→6, 6→7)    ← indices offset by 5

Combined edge_index: [[0,1,3,5,6], [1,2,4,6,7]]
batch_idx: [0,0,0,1,1,2,2,2,2]    ← for per-graph mean pool
```

Replaces `batch_size=1` with proper mini-batch training.

### Augmentation

**Layer 1 — Embedding Mixup (replaces SMOTE on token IDs)**

Applied after the shared token embedding layer during training (30% probability per batch):

```
λ ~ Beta(0.4, 0.4)
mixed_emb  = λ · embed(x_a) + (1-λ) · embed(x_b)
mixed_label = λ · onehot(y_a) + (1-λ) · onehot(y_b)
loss = soft_cross_entropy(logits, mixed_label)
```

**Layer 2 — Adversarial Opcode Augmentation (applied to token sequences, 40% prob)**

| Technique | Simulates |
|---|---|
| NOP insertion | NOP sled padding for signature evasion |
| Dead code injection | Cancel-out pairs (`push`/`pop`, `inc`/`dec`) |
| Instruction substitution | `xor eax,eax` ↔ `mov eax,0` |
| Sequence jitter | Random front/back truncation ±10% |

### Updated DVC Pipeline

```yaml
stages:
  disassemble:
    cmd: wintermute data build --data-dir data
    # Produces: x_data.npy, y_data.npy, vocab.json, manifest.json, graphs/

  pretrain:
    cmd: wintermute pretrain --data-dir data/processed

  train:
    cmd: wintermute train --data-dir data/processed --pretrained malbert_pretrained.safetensors

  evaluate:
    cmd: wintermute evaluate --data-dir data/processed
```

---

## Section 3: Training Pipeline

### Phase 1 — MalBERT MLM Pre-training

- Existing `MLMPretrainer` kept, with `--pretrained` wiring fixed
- Trains on `x_data.npy` sequences only (no graphs needed)
- Output: `malbert_pretrained.safetensors`
- ~50 epochs

### Phase 2 — Joint Fine-tuning

**Stage A — Frozen encoder (first 5 epochs)**
- MalBERT encoder weights frozen
- Only GAT + fusion + classifier train
- Lets new components warm up without corrupting pre-trained encoder

**Stage B — Full fine-tuning (remaining epochs)**
- All weights unfrozen
- Differential learning rates:
  - Encoder: `base_lr × 0.1` (3e-5)
  - GAT + fusion + classifier: `base_lr` (3e-4)

### Optimizer

- **AdamW**, weight decay 0.01 (unchanged)
- **Cosine decay with linear warmup** over first 5% of total steps (warmup added)
- **Gradient clipping** `max_grad_norm=1.0` (added — prevents spikes when encoder unfreezes)

### Training Loop (per batch)

1. Load sequences `[B, T]` and labels `[B]`
2. Load graphs for each sample → disjoint pack; missing → NO_GRAPH slot
3. Apply Layer 2 augmentation to token sequences (40% prob, training only)
4. Forward pass through `WintermuteMalwareDetector`
5. Apply Layer 1 Embedding Mixup (30% prob, training only) — modifies embedded sequences and labels
6. Compute loss: `cross_entropy` (hard labels) or `soft_cross_entropy` (Mixup samples)
7. Backward + gradient clip + AdamW step

### Checkpointing

- Save best model on **validation macro F1** (not accuracy — handles class imbalance)
- Save two files per checkpoint: `malware_detector.safetensors` + `malware_detector_manifest.json`

```json
{
  "arch": "WintermuteMalwareDetector",
  "version": "3.0.0",
  "vocab_size": 287,
  "num_classes": 2,
  "dims": 256,
  "num_heads": 8,
  "num_layers": 6,
  "gat_layers": 3,
  "vocab_sha256": "a3f9...",
  "best_val_macro_f1": 0.934,
  "trained_with_pretrained_encoder": true
}
```

`scan` command validates `vocab_sha256` against loaded `vocab.json` before inference.

### Evaluation Metrics

Added to `eval_metrics.json`:
- `auc_roc` — performance across all thresholds
- `fpr_at_1pct_fnr` — false positive rate when tuned to miss only 1% of malware (operational metric)

---

## Section 4: Testing Approach

### Unit Tests

| File | Key tests |
|---|---|
| `test_disassembler.py` | Mock r2pipe; timeout → `extraction_failed`; node limit → `extraction_failed` |
| `test_gat.py` | Shape correctness; disjoint batch packing; gradient flow to embedding layer |
| `test_fusion.py` | Cross-attention shape; NO_GRAPH fallback; mixed batch (some with graphs, some without) |
| `test_detector.py` | Full forward pass shapes; bfloat16 cast; manifest `vocab_sha256` validation |

### Integration Tests

| File | Key tests |
|---|---|
| `test_pipeline.py` | Synthetic dataset → `JointTrainer.load_dataset()` → one training step → loss decreases → checkpoint save/reload |
| `test_augmentation.py` | Mixup soft labels sum to 1.0; opcode augmentation lengths; augmentation disabled in eval mode |

### Behavioural Tests

| File | Key tests |
|---|---|
| `test_learning.py` | 200 synthetic samples, 5 epochs → val accuracy > 70%; macro F1 improves monotonically; pre-trained encoder outperforms random init at 3 epochs |
| `test_robustness.py` | 50 correctly-classified samples + each augmentation technique → prediction unchanged for ≥80% |

### Existing Tests

| File | Action |
|---|---|
| `test_malbert.py` | Keep + extend with RoPE tests |
| `test_tokenizer.py` | Keep unchanged |
| `test_tracking.py` | Keep unchanged |
| `test_model.py` | **Delete** — `MalwareClassifier` removed |
| `test_cli.py` | **Rewrite** — update for `WintermuteMalwareDetector` |
