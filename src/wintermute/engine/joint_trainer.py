# src/wintermute/engine/joint_trainer.py
"""
joint_trainer.py — Two-Phase Joint Training for WintermuteMalwareDetector

Phase A: encoder frozen (gradients zeroed), other components trained.
Phase B: full fine-tuning of all parameters.

Features:
  - Soft cross-entropy loss with optional Mixup augmentation
  - Embedding-space Mixup via apply_embedding_mixup (augment.py)
  - Layer 2 token-sequence augmentation via HeuristicAugmenter (augment.py)
  - Phase B differential learning rates: encoder at 0.1× base LR
  - Cosine LR decay with linear warmup
  - Gradient clipping
  - Macro F1 checkpoint saving
  - Graph collation with no-graph fallback (empty graph_index)
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils
import numpy as np
from omegaconf import OmegaConf

from wintermute.engine.metrics import compute_macro_f1
from wintermute.models.fusion import DetectorConfig, WintermuteMalwareDetector


class JointTrainer:
    """Two-phase trainer for WintermuteMalwareDetector.

    Phase A: encoder parameters receive zero gradients (frozen approximation).
    Phase B: all parameters are updated (full fine-tuning).

    Parameters
    ----------
    config : DetectorConfig
        Model architecture configuration.
    data_dir : str | Path
        Directory containing x_data.npy, y_data.npy, vocab.json,
        graph_index.json, and optionally graphs/*.pkl.
    overrides : dict | None
        Override default training hyperparameters.
    pretrained_encoder_path : str | None
        Optional path to a pretrained MalBERT encoder weights file.
    """

    DEFAULTS = {
        "epochs_phase_a": 5,
        "epochs_phase_b": 20,
        "batch_size": 8,
        "learning_rate": 3e-4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "max_grad_norm": 1.0,
        "val_ratio": 0.2,
        "seed": 42,
        "mixup_prob": 0.3,
        "augment_prob": 0.4,
        "save_path": "malware_detector.safetensors",
        "manifest_path": "malware_detector_manifest.json",
    }

    def __init__(
        self,
        config: DetectorConfig,
        data_dir,
        overrides=None,
        pretrained_encoder_path=None,
        hook=None,
        db_session=None,
    ):
        cfg = OmegaConf.create(self.DEFAULTS)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
        self.cfg = cfg
        self.model_config = config
        self.data_dir = Path(data_dir)
        self.pretrained_path = pretrained_encoder_path
        self.model = None
        self.optimizer = None
        # Lazy-initialized HeuristicAugmenter (created once with fixed seed)
        self._augmenter = None
        # Epoch counter used to seed per-epoch RNG deterministically
        self._epoch_count = 0
        self._hook = hook
        self._db_session = db_session
        self._training_run_id = None
        self._load_data()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _load_data(self):
        x_np = np.load(self.data_dir / "x_data.npy")
        y_np = np.load(self.data_dir / "y_data.npy")

        with open(self.data_dir / "vocab.json") as f:
            self.vocab = json.load(f)

        self.vocab_sha = hashlib.sha256(json.dumps(self.vocab, sort_keys=True).encode()).hexdigest()

        # Build reverse vocab once for Layer 2 augmentation decode/re-encode
        self.id_to_op = {v: k for k, v in self.vocab.items()}

        rng = np.random.default_rng(self.cfg.seed)
        idx = rng.permutation(len(y_np))
        split = int(len(y_np) * (1 - self.cfg.val_ratio))
        ti, vi = idx[:split], idx[split:]

        self.x_train = mx.array(x_np[ti])
        self.y_train = mx.array(y_np[ti])
        self.x_val = mx.array(x_np[vi])
        self.y_val = mx.array(y_np[vi])
        # Keep original indices for graph lookups
        self.train_orig_idx = ti

        gi_path = self.data_dir / "graph_index.json"
        self.graph_index = {}
        if gi_path.exists():
            raw = json.loads(gi_path.read_text())
            # Keys stored as strings in JSON; convert to int
            self.graph_index = {int(k): v for k, v in raw.items()}

    # ------------------------------------------------------------------
    # Layer 2 token-sequence augmentation
    # ------------------------------------------------------------------
    def _augment_sequences(self, xb: mx.array) -> mx.array:
        """Apply heuristic augmentation to a batch of token-ID sequences.

        Decodes each sequence back to opcodes, applies HeuristicAugmenter
        (NOP insertion, dead-code injection, reordering), then re-encodes
        to token IDs.  Output is padded / truncated to the original length T.
        """
        from wintermute.data.augment import HeuristicAugmenter

        # Lazy-init augmenter with the configured seed so it is created once
        if self._augmenter is None:
            self._augmenter = HeuristicAugmenter(seed=int(self.cfg.seed))

        unk_id = self.vocab.get("<UNK>", 1)
        T = xb.shape[1]
        result = []
        for i in range(xb.shape[0]):
            ids = xb[i].tolist()
            # Strip PAD tokens before decoding so the augmenter works on real
            # opcodes only.  Without this, PAD tokens end up in the middle of
            # the augmented sequence and push real opcodes past the truncation
            # boundary.
            pad_id_val = self.vocab.get("<PAD>", 0)
            ops = [self.id_to_op.get(tok_id, "<UNK>") for tok_id in ids if tok_id != pad_id_val]
            # Augment
            ops_aug = self._augmenter.augment_sequence(ops)
            # Re-encode
            ids_aug = [self.vocab.get(op, unk_id) for op in ops_aug]
            # Pad / truncate to original length T
            ids_aug = ids_aug[:T] + [0] * max(0, T - len(ids_aug))
            result.append(ids_aug)
        return mx.array(result, dtype=xb.dtype)

    # ------------------------------------------------------------------
    # Graph collation
    # ------------------------------------------------------------------
    def _collate_graphs(self, orig_indices):
        """Collate graphs for a mini-batch.

        Returns (node_embs, edge_src, edge_dst, batch_idx, n_graphs)
        or (None, None, None, None, 0) when no graphs are available
        (e.g. graph_index is empty in tests).
        """
        D = self.model_config.dims
        all_nodes = []
        all_src, all_dst = [], []
        batch_ids = []
        node_offset = 0

        for graph_id, oi in enumerate(orig_indices):
            pkl_path = self.graph_index.get(oi)
            if not pkl_path:
                continue
            p = self.data_dir / pkl_path
            if not p.exists():
                continue
            try:
                r = pickle.load(open(p, "rb"))
            except Exception:
                continue
            if r.extraction_failed or not r.node_opcodes:
                continue

            unk = self.vocab.get("<UNK>", 1)
            for ops in r.node_opcodes:
                if not ops:
                    all_nodes.append(mx.zeros((D,)))
                else:
                    ids = mx.array([self.vocab.get(o, unk) for o in ops])
                    emb = mx.mean(self.model.token_embedding(ids), axis=0)
                    all_nodes.append(emb)
                batch_ids.append(graph_id)

            src_local, dst_local = r.edge_index
            for s, d in zip(src_local, dst_local):
                all_src.append(s + node_offset)
                all_dst.append(d + node_offset)
            node_offset += len(r.node_opcodes)

        if not all_nodes:
            return None, None, None, None, 0

        node_embs = mx.stack(all_nodes)  # [N_total, D]
        edge_src = mx.array(all_src, dtype=mx.int32)
        edge_dst = mx.array(all_dst, dtype=mx.int32)
        batch_idx = mx.array(batch_ids, dtype=mx.int32)
        n_graphs = len(orig_indices)
        return node_embs, edge_src, edge_dst, batch_idx, n_graphs

    # ------------------------------------------------------------------
    # Single epoch
    # ------------------------------------------------------------------
    def train_one_epoch(self, phase: str = "B") -> float:
        """Run one training epoch and return the average loss.

        Parameters
        ----------
        phase : str
            "A" -- encoder gradients are zeroed (frozen approximation).
            "B" -- all parameters are updated with differential LRs
                   (encoder at 0.1× base LR).
        """
        from wintermute.data.augment import apply_embedding_mixup

        # Auto-initialize model and optimizer if called standalone (e.g. in tests)
        if self.model is None:
            self.model = WintermuteMalwareDetector(self.model_config)
            WintermuteMalwareDetector.cast_to_bf16(self.model)
        if self.optimizer is None:
            steps_per_epoch = (
                self.x_train.shape[0] + self.cfg.batch_size - 1
            ) // self.cfg.batch_size
            total_steps = (self.cfg.epochs_phase_a + self.cfg.epochs_phase_b) * steps_per_epoch
            self.optimizer = self._make_optimizer(total_steps)

        x, y = self.x_train, self.y_train
        n = x.shape[0]
        rng = np.random.default_rng(self.cfg.seed + self._epoch_count)
        self._epoch_count += 1
        indices = rng.permutation(n)
        B = self.cfg.batch_size
        C = self.model_config.num_classes
        total_loss, n_steps = 0.0, 0

        def soft_xent_normal(
            model, xb, yb_soft, node_embs, edge_src, edge_dst, batch_idx, n_graphs
        ):
            """Standard soft cross-entropy (no Mixup)."""
            logits = model(
                xb,
                node_embs=node_embs,
                edge_src=edge_src,
                edge_dst=edge_dst,
                batch_idx=batch_idx,
                n_graphs=n_graphs,
            )
            log_p = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            return -mx.mean(mx.sum(yb_soft * log_p, axis=-1))

        def soft_xent_mixup(
            model,
            xb_a,
            xb_b,  # token IDs for the two batches
            yb_a,
            yb_b,  # integer labels for the two batches
            lam,  # float
            node_embs,
            edge_src,
            edge_dst,
            batch_idx,
            n_graphs,
        ):
            """Embedding-space Mixup soft cross-entropy."""
            # Prepend [CLS] / append [SEP] at token-ID level first so that
            # the embedding dimension (T+2) matches what the encoder expects.
            xb_a_special = model._prepend_cls_append_sep(xb_a)  # [B, T+2]
            xb_b_special = model._prepend_cls_append_sep(xb_b)  # [B, T+2]

            # Compute token embeddings for both batches [B, T+2, D]
            emb_a = model.token_embedding(xb_a_special)
            emb_b = model.token_embedding(xb_b_special)

            # Mixup in embedding space; get mixed embeddings + soft labels
            mixed_emb, yb_soft = apply_embedding_mixup(
                emb_a, emb_b, yb_a, yb_b, num_classes=C, lam=lam
            )

            # Forward: pass sequence=None and pre-computed token_embeddings.
            # Use xb_a as the token-ID reference for positional + pad-mask
            # computation inside the encoder (values will be ignored since
            # token_embs is provided).
            logits = model(
                sequence=None,
                token_embeddings=mixed_emb,
                node_embs=node_embs,
                edge_src=edge_src,
                edge_dst=edge_dst,
                batch_idx=batch_idx,
                n_graphs=n_graphs,
            )
            log_p = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            return -mx.mean(mx.sum(yb_soft * log_p, axis=-1))

        loss_and_grad_normal = nn.value_and_grad(self.model, soft_xent_normal)
        loss_and_grad_mixup = nn.value_and_grad(self.model, soft_xent_mixup)

        for start in range(0, n, B):
            end = min(start + B, n)
            bi = indices[start:end]
            xb = x[mx.array(bi)]
            yb = y[mx.array(bi)]
            batch_size_actual = xb.shape[0]

            orig_idx = [int(self.train_orig_idx[i]) for i in bi]
            ne, es, ed, bidx, ng = self._collate_graphs(orig_idx)

            # Layer 2: token-sequence augmentation (40% prob, training only)
            if rng.random() < self.cfg.augment_prob:
                xb = self._augment_sequences(xb)

            # Embedding-space Mixup augmentation
            use_mixup = rng.random() < self.cfg.mixup_prob and batch_size_actual >= 2
            if use_mixup:
                lam = float(rng.beta(0.4, 0.4))
                perm = rng.permutation(batch_size_actual)
                xb_b = xb[mx.array(perm)]
                yb_b = yb[mx.array(perm)]

                loss, grads = loss_and_grad_mixup(
                    self.model,
                    xb,
                    xb_b,
                    yb,
                    yb_b,
                    lam,
                    ne,
                    es,
                    ed,
                    bidx,
                    ng,
                )
            else:
                # Build one-hot soft labels for normal path
                yb_soft = (
                    mx.zeros((batch_size_actual, C)).at[mx.arange(batch_size_actual), yb].add(1.0)
                )

                loss, grads = loss_and_grad_normal(self.model, xb, yb_soft, ne, es, ed, bidx, ng)

            # Phase A: zero encoder gradients to approximate frozen encoder
            if phase == "A" and "malbert_encoder" in grads:
                zero_enc = mlx.utils.tree_map(
                    lambda g: mx.zeros_like(g) if isinstance(g, mx.array) else g,
                    grads["malbert_encoder"],
                )
                grads = dict(grads)
                grads["malbert_encoder"] = zero_enc

            # Phase B: encoder gets 0.1× base LR (differential learning rates)
            if phase == "B" and "malbert_encoder" in grads:
                scaled_enc = mlx.utils.tree_map(
                    lambda g: g * 0.1 if isinstance(g, mx.array) else g,
                    grads["malbert_encoder"],
                )
                grads = dict(grads)
                grads["malbert_encoder"] = scaled_enc

            # Gradient clipping via global norm (applied AFTER differential LR scaling)
            flat_leaves = [v for _, v in mlx.utils.tree_flatten(grads) if isinstance(v, mx.array)]
            if flat_leaves:
                norm = mx.sqrt(sum(mx.sum(g * g) for g in flat_leaves))
                # mx.eval materializes lazy MLX computation (not Python eval)
                mx.eval(norm)
                norm_val = float(norm.item())
                if norm_val > self.cfg.max_grad_norm:
                    scale = self.cfg.max_grad_norm / (norm_val + 1e-6)
                    grads = mlx.utils.tree_map(
                        lambda g: g * scale if isinstance(g, mx.array) else g,
                        grads,
                    )

            self.optimizer.update(self.model, grads)
            # mx.eval materializes the lazy MLX computation graph
            mx.eval(self.model.parameters(), self.optimizer.state)

            total_loss += float(loss.item())
            n_steps += 1

        return total_loss / max(n_steps, 1)

    # ------------------------------------------------------------------
    # Optimizer with warmup + cosine decay
    # ------------------------------------------------------------------
    def _make_optimizer(self, total_steps: int) -> optim.AdamW:
        ws = max(int(total_steps * self.cfg.warmup_ratio), 1)
        base = float(self.cfg.learning_rate)
        decay_steps = max(total_steps - ws, 1)

        # Use MLX built-in schedule primitives so the learning_rate is an
        # MLX array rather than a plain Python float (required by AdamW).
        warmup = optim.linear_schedule(init=0.0, end=base, steps=ws)
        cosine = optim.cosine_decay(init=base, decay_steps=decay_steps, end=0.0)
        lr_schedule = optim.join_schedules([warmup, cosine], boundaries=[ws])

        return optim.AdamW(
            learning_rate=lr_schedule,
            weight_decay=float(self.cfg.weight_decay),
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate(self) -> float:
        return compute_macro_f1(
            self.model,
            self.x_val,
            self.y_val,
            self.cfg.batch_size,
            self.model_config.num_classes,
        )

    # ------------------------------------------------------------------
    # Checkpoint saving
    # ------------------------------------------------------------------
    def _save_checkpoint(self, f1: float) -> None:
        self.model.save_weights(self.cfg.save_path)
        self.model.save_manifest(
            self.cfg.manifest_path,
            vocab_sha256=self.vocab_sha,
            best_val_macro_f1=f1,
            trained_with_pretrained_encoder=bool(self.pretrained_path),
        )
        print(f"       checkpoint saved (f1={f1:.4f})")

    # ------------------------------------------------------------------
    # Main training entry point
    # ------------------------------------------------------------------
    def train(self) -> float:
        """Run the full two-phase training loop.

        Returns the best macro F1 score achieved on the validation set.
        """
        print(
            f"Vocab: {len(self.vocab)}  Train: {self.x_train.shape[0]}  Val: {self.x_val.shape[0]}"
        )

        self.model = WintermuteMalwareDetector(self.model_config)
        WintermuteMalwareDetector.cast_to_bf16(self.model)

        # Load pretrained encoder weights if provided
        if self.pretrained_path and Path(self.pretrained_path).exists():
            print(f"Loading pretrained encoder from {self.pretrained_path}")
            self.model.malbert_encoder.load_weights(self.pretrained_path, strict=False)

        # --- DB: create TrainingRun row ---
        self._training_run_id = None
        if self._db_session is not None:
            try:
                from wintermute.db.models import TrainingRun

                run = TrainingRun(
                    config=dict(self.cfg),
                    pretrained_weights=self.pretrained_path,
                    total_samples=int(self.x_train.shape[0] + self.x_val.shape[0]),
                    num_classes=self.model_config.num_classes,
                    train_split_size=int(self.x_train.shape[0]),
                    val_split_size=int(self.x_val.shape[0]),
                )
                self._db_session.add(run)
                self._db_session.flush()
                self._training_run_id = run.id
            except Exception:
                import logging

                logging.getLogger("wintermute.db").debug(
                    "Failed to create TrainingRun row", exc_info=True
                )

        steps_per_epoch = (self.x_train.shape[0] + self.cfg.batch_size - 1) // self.cfg.batch_size
        total_steps = (self.cfg.epochs_phase_a + self.cfg.epochs_phase_b) * steps_per_epoch

        self.optimizer = self._make_optimizer(total_steps)
        best_f1 = 0.0

        for label, phase, n_epochs in [
            ("Phase A --- encoder frozen", "A", self.cfg.epochs_phase_a),
            ("Phase B --- full fine-tune", "B", self.cfg.epochs_phase_b),
        ]:
            print(f"\n{label} ({n_epochs} epochs)")
            for ep in range(1, n_epochs + 1):
                t0 = time.perf_counter()
                loss = self.train_one_epoch(phase)
                f1 = self._validate()
                elapsed = time.perf_counter() - t0
                print(f"  ep {ep:3d}  loss={loss:.4f}  val_f1={f1:.4f}  ({elapsed:.1f}s)")
                if self._hook:
                    self._hook.on_epoch(ep, phase, loss, 0.0, f1, f1, elapsed)
                    if self._hook.cancelled:
                        self._hook.on_log(f"Training cancelled at epoch {ep}", "warn")
                        return best_f1
                if f1 > best_f1:
                    best_f1 = f1
                    self._save_checkpoint(f1)

                    # --- DB: update best metrics on TrainingRun ---
                    if self._db_session is not None and self._training_run_id is not None:
                        try:
                            from wintermute.db.models import TrainingRun

                            run = self._db_session.get(TrainingRun, self._training_run_id)
                            if run:
                                run.best_epoch = ep
                                run.best_val_loss = loss
                                run.best_val_macro_f1 = f1
                                run.best_val_accuracy = f1  # approximate
                                run.epochs_completed = ep
                                self._db_session.flush()
                        except Exception:
                            import logging

                            logging.getLogger("wintermute.db").debug(
                                "Failed to update TrainingRun row", exc_info=True
                            )

        # --- DB: create Model row and finalize TrainingRun ---
        if self._db_session is not None and self._training_run_id is not None:
            try:
                from datetime import datetime, timezone

                from wintermute.db.models import Model, TrainingRun

                run = self._db_session.get(TrainingRun, self._training_run_id)
                if run:
                    run.completed_at = datetime.now(timezone.utc)
                    run.epochs_completed = self.cfg.epochs_phase_a + self.cfg.epochs_phase_b

                model_row = Model(
                    version=f"v-{self._training_run_id[:8]}",
                    architecture="WintermuteMalwareDetector",
                    weights_path=str(self.cfg.save_path),
                    manifest_path=str(self.cfg.manifest_path),
                    vocab_size=len(self.vocab),
                    num_classes=self.model_config.num_classes,
                    dims=self.model_config.dims,
                    max_seq_length=self.model_config.max_seq_length,
                    vocab_sha256=self.vocab_sha,
                    training_run_id=self._training_run_id,
                    pretrained_from=self.pretrained_path,
                    best_val_macro_f1=best_f1,
                    status="staged",
                )
                self._db_session.add(model_row)
                self._db_session.flush()

                # Link the training run back to the model
                if run:
                    run.model_id = model_row.id
                    self._db_session.flush()
            except Exception:
                import logging

                logging.getLogger("wintermute.db").debug(
                    "Failed to create Model row", exc_info=True
                )

        print(f"\nDone. Best macro F1: {best_f1:.4f}")
        if self._hook:
            self._hook.on_log(f"Training complete — best F1: {best_f1:.4f}", "ok")
        return best_f1
