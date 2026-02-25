"""
augment.py — Wintermute Synthetic Data & Augmentation

Consolidated from generate_synthetic_data.py.
Placeholder hooks for SMOTE integration (Phase 3).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Fake opcode pools — intentionally distinct distributions
# ---------------------------------------------------------------------------
BENIGN_OPCODES = [
    "mov", "push", "pop", "call", "ret", "lea", "add", "sub",
    "cmp", "jmp", "je", "jne", "test", "nop", "xchg", "inc",
    "dec", "and", "or", "shl", "imul", "cdq", "movzx", "movsx",
]

MALICIOUS_OPCODES = [
    "xor", "shr", "shl", "ror", "rol", "loop", "stosb", "lodsb",
    "scasb", "repne", "rep", "int", "rdtsc", "cpuid", "vmcall",
    "syscall", "in", "out", "cli", "sti", "pushfd", "popfd",
    "bswap", "not",
]

SHARED_OPCODES = [
    "mov", "push", "pop", "call", "ret", "add", "sub", "cmp",
    "jmp", "je", "jne", "test", "nop", "xor", "and", "or",
]


class SyntheticGenerator:
    """
    Generate synthetic opcode-sequence datasets for pipeline testing.

    Safe samples are biased toward common "benign" opcodes.
    Malicious samples are biased toward suspicious patterns.
    """

    def __init__(
        self,
        n_samples: int = 500,
        max_seq_length: int = 2048,
        seed: int = 42,
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
    ):
        self.n_samples = n_samples
        self.max_seq_length = max_seq_length
        self.seed = seed
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.rng = np.random.default_rng(seed)

    def generate_sample(self, label: int) -> list[str]:
        """Generate a fake opcode sequence with label-dependent bias."""
        min_len = min(50, self.max_seq_length)
        seq_len = self.rng.integers(min_len, self.max_seq_length + 1)
        if label == 0:
            pool = BENIGN_OPCODES + SHARED_OPCODES
        else:
            pool = MALICIOUS_OPCODES + SHARED_OPCODES
        return [pool[i] for i in self.rng.integers(0, len(pool), size=seq_len)]

    def generate_dataset(
        self, out_dir: str | Path = "data/processed"
    ) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        """
        Generate a full synthetic dataset and save to disk.

        Returns (x_data, y_data, stoi).
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        n_safe = self.n_samples // 2
        n_mal = self.n_samples - n_safe

        # 1. Generate raw sequences
        all_opcodes: list[list[str]] = []
        labels: list[int] = []

        for _ in range(n_safe):
            all_opcodes.append(self.generate_sample(0))
            labels.append(0)
        for _ in range(n_mal):
            all_opcodes.append(self.generate_sample(1))
            labels.append(1)

        # Shuffle
        order = self.rng.permutation(len(labels))
        all_opcodes = [all_opcodes[i] for i in order]
        labels = [labels[i] for i in order]

        # 2. Build vocabulary
        unique_ops = sorted({op for seq in all_opcodes for op in seq})
        stoi = {self.pad_token: 0, self.unk_token: 1}
        for op in unique_ops:
            stoi[op] = len(stoi)

        # 3. Encode & serialise
        pad_id = stoi[self.pad_token]
        x_rows = []
        for seq in all_opcodes:
            ids = [stoi[op] for op in seq[: self.max_seq_length]]
            ids += [pad_id] * (self.max_seq_length - len(ids))
            x_rows.append(ids)

        x_data = np.array(x_rows, dtype=np.int32)
        y_data = np.array(labels, dtype=np.int32)

        np.save(out_dir / "x_data.npy", x_data)
        np.save(out_dir / "y_data.npy", y_data)
        with open(out_dir / "vocab.json", "w") as f:
            json.dump(stoi, f, indent=2)

        print(f"✅  Synthetic dataset saved to {out_dir}/")
        print(f"    x_data.npy  shape={x_data.shape}  dtype={x_data.dtype}")
        print(f"    y_data.npy  shape={y_data.shape}  dtype={y_data.dtype}")
        print(f"    vocab.json   {len(stoi)} entries")
        print(f"    {n_safe} safe + {n_mal} malicious = {self.n_samples} total")

        return x_data, y_data, stoi


# ---------------------------------------------------------------------------
# SMOTE Augmentation (Phase 3)
# ---------------------------------------------------------------------------
class SMOTEAugmenter:
    """
    Synthetic Minority Over-sampling Technique in embedding space.

    Generates new samples for under-represented classes by interpolating
    between existing samples and their nearest neighbours.

    Usage
    -----
    >>> augmenter = SMOTEAugmenter(k_neighbors=5, seed=42)
    >>> x_aug, y_aug = augmenter.augment(x_data, y_data, target_ratio=1.0)
    """

    def __init__(self, k_neighbors: int = 5, seed: int = 42):
        self.k_neighbors = k_neighbors
        self.rng = np.random.default_rng(seed)

    def augment(
        self,
        x: np.ndarray,
        y: np.ndarray,
        target_ratio: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Oversample minority classes to reach ``target_ratio`` relative to majority.

        Args:
            x: [N, T] encoded sequences (int32)
            y: [N] labels
            target_ratio: desired ratio of minority to majority samples

        Returns:
            x_augmented, y_augmented: concatenated original + synthetic samples
        """
        classes, counts = np.unique(y, return_counts=True)
        max_count = counts.max()
        target_count = int(max_count * target_ratio)

        x_parts = [x]
        y_parts = [y]

        for cls, count in zip(classes, counts):
            if count >= target_count:
                continue

            n_synthetic = target_count - count
            cls_mask = y == cls
            x_cls = x[cls_mask]  # All samples of this class

            if len(x_cls) < 2:
                # Can't interpolate with < 2 samples, just duplicate
                indices = self.rng.integers(0, len(x_cls), size=n_synthetic)
                x_parts.append(x_cls[indices])
                y_parts.append(np.full(n_synthetic, cls, dtype=y.dtype))
                continue

            # Generate synthetic samples via interpolation
            synthetic = self._generate_synthetic(x_cls, n_synthetic)
            x_parts.append(synthetic)
            y_parts.append(np.full(n_synthetic, cls, dtype=y.dtype))

        x_aug = np.concatenate(x_parts, axis=0)
        y_aug = np.concatenate(y_parts, axis=0)

        # Shuffle
        order = self.rng.permutation(len(y_aug))
        return x_aug[order], y_aug[order]

    def _generate_synthetic(
        self, x_cls: np.ndarray, n_synthetic: int
    ) -> np.ndarray:
        """Generate synthetic samples by interpolating between neighbors."""
        n_samples, seq_len = x_cls.shape
        k = min(self.k_neighbors, n_samples - 1)
        synthetic = np.zeros((n_synthetic, seq_len), dtype=x_cls.dtype)

        # Cast to float for distance computation
        x_float = x_cls.astype(np.float32)

        for i in range(n_synthetic):
            # Pick a random sample
            idx = self.rng.integers(0, n_samples)
            anchor = x_float[idx]

            # Find k nearest neighbors (L2 distance)
            dists = np.sum((x_float - anchor) ** 2, axis=1)
            dists[idx] = np.inf  # exclude self
            neighbor_indices = np.argsort(dists)[:k]

            # Pick a random neighbor
            nn_idx = neighbor_indices[self.rng.integers(0, k)]
            neighbor = x_float[nn_idx]

            # Interpolate: new = anchor + lambda * (neighbor - anchor)
            lam = self.rng.random()
            interp = anchor + lam * (neighbor - anchor)

            # Round back to integer token IDs
            synthetic[i] = np.round(interp).astype(x_cls.dtype)

        return synthetic


# ---------------------------------------------------------------------------
# Heuristic Augmentation (Phase 3)
# ---------------------------------------------------------------------------

# Register groups for swapping
_REGISTER_GROUPS = {
    "gp32": ["eax", "ebx", "ecx", "edx", "esi", "edi"],
    "gp64": ["rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10",
             "r11", "r12", "r13", "r14", "r15"],
}

# Dead code instruction patterns
_DEAD_CODE_PATTERNS = [
    ["nop"],
    ["push", "pop"],
    ["xchg", "xchg"],        # swap twice = no-op
    ["inc", "dec"],           # +1 then -1 = no-op
    ["add", "sub"],
]

# Opcode substitution map — semantically equivalent replacements
_SUBSTITUTION_MAP = {
    "sub": "add",
    "inc": "add",
    "dec": "sub",
}


class HeuristicAugmenter:
    """
    Instruction-level heuristic augmentation for opcode sequences.

    Techniques:
        1. **NOP insertion** — inject semantically neutral instructions
        2. **Dead code injection** — insert instruction pairs that cancel out
        3. **Instruction reordering** — swap independent adjacent instructions

    These are common obfuscation techniques in real malware, so augmenting
    training data this way improves robustness.

    Usage
    -----
    >>> aug = HeuristicAugmenter(seed=42)
    >>> augmented = aug.augment_sequence(opcodes, techniques=["nop", "reorder"])
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def augment_sequence(
        self,
        opcodes: list[str],
        techniques: list[str] | None = None,
        max_insertions: int = 10,
    ) -> list[str]:
        """
        Apply augmentation techniques to an opcode sequence.

        Args:
            opcodes: list of opcode mnemonics
            techniques: subset of ["nop", "dead_code", "reorder"]
            max_insertions: max number of insertions for nop/dead_code

        Returns:
            Augmented opcode sequence (may be longer than original)
        """
        if techniques is None:
            techniques = ["nop", "dead_code", "reorder"]

        result = list(opcodes)

        if "nop" in techniques:
            result = self._insert_nops(result, max_insertions)

        if "dead_code" in techniques:
            result = self._insert_dead_code(result, max_insertions // 2)

        if "reorder" in techniques:
            result = self._reorder_independent(result)

        if "substitute" in techniques:
            result = self._substitute(result)

        return result

    def augment_dataset(
        self,
        x: np.ndarray,
        y: np.ndarray,
        stoi: dict[str, int],
        techniques: list[str] | None = None,
        augment_ratio: float = 0.5,
        max_seq_length: int = 2048,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Augment a fraction of the dataset with heuristic transforms.

        Returns concatenated (original + augmented) arrays.
        """
        itos = {v: k for k, v in stoi.items()}
        pad_id = stoi.get("<PAD>", 0)
        n_augment = int(len(y) * augment_ratio)

        indices = self.rng.choice(len(y), size=n_augment, replace=False)
        aug_rows = []

        for idx in indices:
            # Decode token IDs back to opcodes (skip PAD)
            ids = x[idx]
            opcodes = [itos.get(int(tid), "<UNK>") for tid in ids if tid != pad_id]

            # Augment
            augmented = self.augment_sequence(opcodes, techniques)

            # Re-encode
            new_ids = [stoi.get(op, stoi.get("<UNK>", 1))
                       for op in augmented[:max_seq_length]]
            new_ids += [pad_id] * (max_seq_length - len(new_ids))
            aug_rows.append(new_ids)

        if not aug_rows:
            return x, y

        x_aug = np.array(aug_rows, dtype=x.dtype)
        y_aug = y[indices]

        return np.concatenate([x, x_aug]), np.concatenate([y, y_aug])

    # --- Internal techniques ---

    def _insert_nops(self, ops: list[str], max_n: int) -> list[str]:
        """Insert NOP instructions at random positions."""
        n_insert = self.rng.integers(1, max_n + 1)
        result = list(ops)
        for _ in range(n_insert):
            pos = self.rng.integers(0, len(result) + 1)
            result.insert(pos, "nop")
        return result

    def _insert_dead_code(self, ops: list[str], max_n: int) -> list[str]:
        """Insert dead code patterns (pairs that cancel out)."""
        n_insert = self.rng.integers(1, max_n + 1)
        result = list(ops)
        for _ in range(n_insert):
            pattern = _DEAD_CODE_PATTERNS[
                self.rng.integers(0, len(_DEAD_CODE_PATTERNS))
            ]
            pos = self.rng.integers(0, len(result) + 1)
            for j, instr in enumerate(pattern):
                result.insert(pos + j, instr)
        return result

    def _reorder_independent(self, ops: list[str]) -> list[str]:
        """Swap adjacent independent instructions."""
        result = list(ops)
        # Instructions that don't depend on each other can be freely swapped
        independent = {"nop", "push", "mov", "lea", "add", "sub", "xor",
                       "and", "or", "inc", "dec", "shl", "shr"}
        for i in range(len(result) - 1):
            if result[i] in independent and result[i + 1] in independent:
                if self.rng.random() < 0.3:  # 30% swap probability
                    result[i], result[i + 1] = result[i + 1], result[i]
        return result

    def _substitute(self, ops: list[str]) -> list[str]:
        """Replace opcodes with semantically equivalent alternatives."""
        result = list(ops)
        for i, op in enumerate(result):
            if op in _SUBSTITUTION_MAP and self.rng.random() < 0.3:
                result[i] = _SUBSTITUTION_MAP[op]
        return result


# ---------------------------------------------------------------------------
# Embedding-space Mixup
# ---------------------------------------------------------------------------

def apply_embedding_mixup(
    emb_a: "mx.array",
    emb_b: "mx.array",
    labels_a: "mx.array",
    labels_b: "mx.array",
    num_classes: int,
    lam: float,
) -> "tuple[mx.array, mx.array]":
    """
    Mixup in embedding space. Returns (mixed_embeddings, soft_labels).
    mixed_embeddings shape: [B, T, D] — same as inputs.
    soft_labels shape: [B, num_classes] — sum to 1.0 per sample.
    lam=1.0 returns emb_a/labels_a unchanged; lam=0.0 returns emb_b/labels_b.
    """
    import mlx.core as _mx

    mixed_emb = lam * emb_a + (1.0 - lam) * emb_b

    def onehot(labels):
        n = labels.shape[0]
        oh = _mx.zeros((n, num_classes))
        return oh.at[_mx.arange(n), labels].add(1.0)

    soft_labels = lam * onehot(labels_a) + (1.0 - lam) * onehot(labels_b)
    return mixed_emb, soft_labels
