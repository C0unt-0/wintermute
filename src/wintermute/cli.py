"""
cli.py — Wintermute Unified CLI

Replaces scan.py, scan_family.py, and the manual script invocations.

Usage:
    wintermute scan target.exe                 # binary safe/malicious
    wintermute scan target.asm --family        # multi-class family detection
    wintermute train                           # train from default config
    wintermute train --track                   # train with MLflow tracking
    wintermute evaluate                        # evaluate saved model
    wintermute data build                      # build dataset from raw files
    wintermute data synthetic                  # generate synthetic test data
    wintermute pretrain                        # MalBERT MLM pre-training
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(
    name="wintermute",
    help="🧠 Wintermute — MLX-powered static malware classifier",
    no_args_is_help=True,
)

data_app = typer.Typer(
    name="data",
    help="Data pipeline commands (build, download, synthetic, etl)",
    no_args_is_help=True,
)
app.add_typer(data_app, name="data")

# Register ETL pipeline commands
from wintermute.data.etl.cli_etl import register_etl_commands
register_etl_commands(data_app)


# ---------------------------------------------------------------------------
# Default family map (MS Malware Classification dataset)
# ---------------------------------------------------------------------------
DEFAULT_FAMILIES = {
    "0": "Ramnit",
    "1": "Lollipop",
    "2": "Kelihos_ver3",
    "3": "Vundo",
    "4": "Simda",
    "5": "Tracur",
    "6": "Kelihos_ver1",
    "7": "Obfuscator.ACY",
    "8": "Gatak",
}


# ═══════════════════════════════════════════════════════════════════════════
# wintermute scan
# ═══════════════════════════════════════════════════════════════════════════
@app.command()
def scan(
    target: str = typer.Argument(...),
    model: str = typer.Option("malware_detector.safetensors", "--model", "-m"),
    manifest: str = typer.Option("malware_detector_manifest.json", "--manifest"),
    vocab: str = typer.Option("data/processed/vocab.json", "--vocab", "-v"),
    family: bool = typer.Option(False, "--family"),
) -> None:
    """Scan a binary using WintermuteMalwareDetector."""
    import hashlib
    import json as _json
    import mlx.core as mx
    from wintermute.data.disassembler import HeadlessDisassembler
    from wintermute.data.tokenizer import read_asm_file
    from wintermute.models.fusion import WintermuteMalwareDetector

    target_path = Path(target)
    if not target_path.exists():
        typer.echo(f"[ERROR] Not found: {target}", err=True)
        raise typer.Exit(1)

    with open(vocab) as f:
        stoi = _json.load(f)
    vocab_sha = hashlib.sha256(_json.dumps(stoi, sort_keys=True).encode()).hexdigest()

    typer.echo(f"Loading {model} ...")
    detector = WintermuteMalwareDetector.load(model, manifest, vocab_sha256=vocab_sha)
    WintermuteMalwareDetector.cast_to_bf16(detector)
    detector.eval()

    if target_path.suffix.lower() == ".asm":
        opcodes = read_asm_file(str(target_path))
    else:
        typer.echo("Disassembling ...")
        result = HeadlessDisassembler(str(target_path)).extract()
        opcodes = result.sequence

    if not opcodes:
        typer.echo("  No opcodes extracted.")
        return

    max_seq = detector.config.max_seq_length
    unk, pad = stoi.get("<UNK>", 1), stoi.get("<PAD>", 0)
    ids = [stoi.get(op, unk) for op in opcodes[:max_seq]] + [pad] * (max_seq - min(len(opcodes), max_seq))
    x = mx.array([ids])

    logits = detector(x)
    probs = mx.softmax(logits, axis=1)
    mx.eval(probs)

    pred = int(mx.argmax(probs, axis=1).item())
    conf = probs[0, pred].item() * 100
    label = ("Safe" if pred == 0 else "Malicious") if not family else f"Class {pred}"

    typer.echo(f"\n{'='*50}")
    icon = "✅" if pred == 0 else "🚨"
    typer.echo(f"  {icon}  {label.upper():<12}  Confidence: {conf:.1f}%")
    typer.echo(f"{'='*50}\n")

    # --- Optional DB persistence ---
    try:
        from wintermute.db.engine import get_engine, get_session
        from wintermute.db.repos.scans import ScanRepo
        from wintermute.db.repos.samples import SampleRepo

        if get_engine() is not None:
            file_sha = hashlib.sha256(target_path.read_bytes()).hexdigest()
            file_size = target_path.stat().st_size
            probs_dict = {
                str(i): float(probs[0, i].item()) for i in range(probs.shape[1])
            }

            with get_session() as session:
                # Upsert the sample
                SampleRepo(session).upsert(
                    sha256=file_sha,
                    family=label,
                    label=pred,
                    source="cli_scan",
                    file_type=target_path.suffix.lstrip(".").upper() or "UNKNOWN",
                    file_size_bytes=file_size,
                    opcode_count=len(opcodes),
                )

                # Record the scan result
                ScanRepo(session).record(
                    sha256=file_sha,
                    filename=target_path.name,
                    file_size_bytes=file_size,
                    predicted_family=label,
                    predicted_label=pred,
                    confidence=float(probs[0, pred].item()),
                    probabilities=probs_dict,
                    model_version=manifest,
                )
    except Exception:
        import logging
        logging.getLogger("wintermute.db").debug(
            "Scan DB persistence failed (best-effort)", exc_info=True
        )


# ═══════════════════════════════════════════════════════════════════════════
# wintermute train
# ═══════════════════════════════════════════════════════════════════════════
@app.command()
def train(
    data_dir: str = typer.Option("data/processed", "--data-dir", "-d"),
    pretrained: str = typer.Option(None, "--pretrained",
        help="Path to malbert_pretrained.safetensors to initialise encoder."),
    epochs_phase_a: int = typer.Option(None, "--epochs-phase-a"),
    epochs_phase_b: int = typer.Option(None, "--epochs-phase-b"),
    batch_size: int = typer.Option(None, "--batch-size", "-b"),
    lr: float = typer.Option(None, "--lr"),
    num_classes: int = typer.Option(None, "--num-classes", "-c"),
    save_path: str = typer.Option(None, "--save-path", "-o"),
) -> None:
    """Train WintermuteMalwareDetector (MalBERT + GAT unified model)."""
    import json as _json
    from wintermute.engine.joint_trainer import JointTrainer
    from wintermute.models.fusion import DetectorConfig

    dp = Path(data_dir)
    vocab = _json.loads((dp / "vocab.json").read_text())
    overrides = {k: v for k, v in {
        "epochs_phase_a": epochs_phase_a, "epochs_phase_b": epochs_phase_b,
        "batch_size": batch_size, "learning_rate": lr, "save_path": save_path,
    }.items() if v is not None}

    cfg = DetectorConfig(vocab_size=len(vocab), num_classes=num_classes or 2)
    JointTrainer(cfg, dp, overrides=overrides or None,
                 pretrained_encoder_path=pretrained).train()


# ═══════════════════════════════════════════════════════════════════════════
# wintermute evaluate
# ═══════════════════════════════════════════════════════════════════════════
@app.command()
def evaluate(
    data_dir: str = typer.Option("data/processed", "--data-dir", "-d"),
    model: str = typer.Option("malware_detector.safetensors", "--model", "-m"),
    manifest: str = typer.Option("malware_detector_manifest.json", "--manifest"),
    vocab: str = typer.Option("data/processed/vocab.json", "--vocab", "-v"),
    output: str = typer.Option("eval_metrics.json", "--output", "-o"),
) -> None:
    """Evaluate WintermuteMalwareDetector and produce metrics JSON."""
    import hashlib
    import json as _json
    import mlx.core as mx
    import numpy as np
    from wintermute.engine.metrics import compute_macro_f1, compute_auc_roc, fpr_at_fnr_threshold
    from wintermute.models.fusion import WintermuteMalwareDetector

    dp = Path(data_dir)
    with open(vocab) as f:
        stoi = _json.load(f)
    vocab_sha = hashlib.sha256(_json.dumps(stoi, sort_keys=True).encode()).hexdigest()

    typer.echo("Loading model ...")
    detector = WintermuteMalwareDetector.load(str(model), str(manifest), vocab_sha256=vocab_sha)
    WintermuteMalwareDetector.cast_to_bf16(detector)
    num_classes = detector.config.num_classes

    typer.echo("Loading dataset ...")
    x = mx.array(np.load(dp / "x_data.npy"))
    y = mx.array(np.load(dp / "y_data.npy"))

    typer.echo("Computing metrics ...")
    batch_size = 8
    macro_f1 = compute_macro_f1(detector, x, y, batch_size, num_classes)

    metrics: dict = {
        "macro_f1": macro_f1,
        "num_samples": int(x.shape[0]),
        "num_classes": num_classes,
    }

    # AUC-ROC and FPR@1%FNR are binary-only metrics
    if num_classes == 2:
        detector.eval()
        all_scores, all_labels = [], []
        n = x.shape[0]
        for start in range(0, n, batch_size):
            xb = x[start:start + batch_size]
            yb = y[start:start + batch_size]
            logits = detector(xb)
            probs = mx.softmax(logits, axis=1)
            mx.eval(probs)
            all_scores.extend(probs[i, 1].item() for i in range(xb.shape[0]))
            all_labels.extend(yb.tolist())
        detector.train()

        scores_np = np.array(all_scores)
        labels_np = np.array(all_labels)
        metrics["auc_roc"] = compute_auc_roc(scores_np, labels_np)
        metrics["fpr_at_1pct_fnr"] = fpr_at_fnr_threshold(scores_np, labels_np, target_fnr=0.01)

    with open(output, "w") as f:
        _json.dump(metrics, f, indent=2)

    typer.echo(f"\n{'='*50}")
    typer.echo(f"  Macro F1:        {macro_f1:.4f}")
    if num_classes == 2:
        typer.echo(f"  AUC-ROC:         {metrics['auc_roc']:.4f}")
        typer.echo(f"  FPR@1%FNR:       {metrics['fpr_at_1pct_fnr']:.4f}")
    typer.echo(f"  Samples:         {x.shape[0]}")
    typer.echo(f"\n  Metrics saved to {output}")
    typer.echo(f"{'='*50}\n")


# ═══════════════════════════════════════════════════════════════════════════
# wintermute data build
# ═══════════════════════════════════════════════════════════════════════════
@data_app.command("build")
def data_build(
    data_dir: str = typer.Option(
        "data", "--data-dir", "-d",
        help="Root data directory (contains raw/).",
    ),
    max_seq_length: int = typer.Option(
        2048, "--max-seq-length",
        help="Max opcode sequence length per sample.",
    ),
) -> None:
    """Build dataset from raw PE files in data/raw/safe and data/raw/malicious."""
    import sys

    import numpy as np

    from wintermute.data.tokenizer import (
        build_vocabulary,
        collect_pe_files,
        encode_sequence,
        extract_opcodes_pe,
    )

    data_path = Path(data_dir)
    out_dir = data_path / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover files
    filepaths, labels = collect_pe_files(data_path)
    if not filepaths:
        typer.echo("[ERROR] No PE files found in data/raw/safe or data/raw/malicious.")
        typer.echo("        Place .exe/.dll files there, or use: wintermute data synthetic")
        raise typer.Exit(code=1)

    typer.echo(f"Found {len(filepaths)} PE files "
               f"({labels.count(0)} safe, {labels.count(1)} malicious).")

    # 2. Extract opcodes
    all_opcodes: list[list[str]] = []
    for i, fp in enumerate(filepaths, 1):
        typer.echo(f"  [{i}/{len(filepaths)}] Extracting {Path(fp).name} …")
        ops = extract_opcodes_pe(fp)
        all_opcodes.append(ops)
        if ops:
            typer.echo(f"          → {len(ops)} instructions")

    # 3. Build vocabulary
    stoi = build_vocabulary(all_opcodes)
    typer.echo(f"Vocabulary size: {len(stoi)} tokens")

    # 4. Encode & serialise
    x_data = np.stack(
        [encode_sequence(ops, stoi, max_seq_length) for ops in all_opcodes]
    )
    y_data = np.array(labels, dtype=np.int32)

    np.save(out_dir / "x_data.npy", x_data)
    np.save(out_dir / "y_data.npy", y_data)
    with open(out_dir / "vocab.json", "w") as f:
        json.dump(stoi, f, indent=2)

    typer.echo(f"\n✅  Dataset saved to {out_dir}/")
    typer.echo(f"    x_data.npy  shape={x_data.shape}  dtype={x_data.dtype}")
    typer.echo(f"    y_data.npy  shape={y_data.shape}  dtype={y_data.dtype}")
    typer.echo(f"    vocab.json   {len(stoi)} entries")


# ═══════════════════════════════════════════════════════════════════════════
# wintermute data synthetic
# ═══════════════════════════════════════════════════════════════════════════
@data_app.command("synthetic")
def data_synthetic(
    n_samples: int = typer.Option(500, "--n-samples", "-n", help="Total samples."),
    max_seq_length: int = typer.Option(2048, "--max-seq-length", help="Sequence length."),
    out_dir: str = typer.Option("data/processed", "--out-dir", "-o", help="Output directory."),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed."),
) -> None:
    """Generate synthetic opcode dataset for testing."""
    from wintermute.data.augment import SyntheticGenerator

    gen = SyntheticGenerator(
        n_samples=n_samples,
        max_seq_length=max_seq_length,
        seed=seed,
    )
    gen.generate_dataset(out_dir=out_dir)


# ═══════════════════════════════════════════════════════════════════════════
# wintermute data download
# ═══════════════════════════════════════════════════════════════════════════
@data_app.command("download")
def data_download(
    families: str = typer.Option(
        "AgentTesla,Emotet,TrickBot",
        "--families", "-f",
        help="Comma-separated family signatures to download.",
    ),
    limit: int = typer.Option(50, "--limit", "-l", help="Max samples per family."),
    out_dir: str = typer.Option("data/bazaar", "--out-dir", "-o", help="Output directory."),
    api_key: str = typer.Option("", "--api-key", help="MalwareBazaar API key (optional)."),
    delay: float = typer.Option(1.0, "--delay", help="Delay between API calls (seconds)."),
) -> None:
    """Download malware samples from MalwareBazaar."""
    from wintermute.data.downloader import MalwareBazaarDownloader

    downloader = MalwareBazaarDownloader(
        api_key=api_key, out_dir=out_dir, delay=delay
    )

    family_list = [f.strip() for f in families.split(",")]
    for fam in family_list:
        downloader.download_family(
            family_name=fam, signature=fam, limit=limit
        )


# ═══════════════════════════════════════════════════════════════════════════
# wintermute pretrain
# ═══════════════════════════════════════════════════════════════════════════
@app.command()
def pretrain(
    data_dir: str = typer.Option(
        "data/processed", "--data-dir", "-d",
        help="Directory containing x_data.npy and vocab.json.",
    ),
    config: str = typer.Option(
        None, "--config",
        help="Path to malbert_config.yaml (optional).",
    ),
    epochs: int = typer.Option(None, "--epochs", "-e", help="Override epochs."),
    batch_size: int = typer.Option(None, "--batch-size", "-b", help="Override batch size."),
    lr: float = typer.Option(None, "--lr", help="Override learning rate."),
    save_path: str = typer.Option(None, "--save-path", "-o", help="Override save path."),
) -> None:
    """Run MalBERT MLM pre-training on unlabelled opcode data."""
    from wintermute.engine.pretrain import MLMPretrainer

    overrides: dict = {}
    if epochs is not None:
        overrides.setdefault("pretrain", {})["epochs"] = epochs
    if batch_size is not None:
        overrides.setdefault("pretrain", {})["batch_size"] = batch_size
    if lr is not None:
        overrides.setdefault("pretrain", {})["learning_rate"] = lr
    if save_path is not None:
        overrides.setdefault("pretrain", {})["save_path"] = save_path

    config_path = config or "configs/malbert_config.yaml"
    pretrainer = MLMPretrainer(config_path=config_path, overrides=overrides or None)
    pretrainer.pretrain(data_dir=data_dir)



# ═══════════════════════════════════════════════════════════════════════════
# wintermute adv — Adversarial training (Phase 5)
# ═══════════════════════════════════════════════════════════════════════════
adv_app = typer.Typer(
    name="adv",
    help="Adversarial training pipeline (Phase 5)",
    no_args_is_help=True,
)
app.add_typer(adv_app, name="adv")


@adv_app.command()
def run(
    model: str = typer.Option("malware_detector.safetensors", "--model", "-m"),
    manifest: str = typer.Option("malware_detector_manifest.json", "--manifest"),
    vocab: str = typer.Option("data/processed/vocab.json", "--vocab", "-v"),
    data_dir: str = typer.Option("data/processed", "--data-dir"),
    episodes: int = typer.Option(500, "--episodes", "-n"),
    cycles: int = typer.Option(5, "--cycles"),
    trades_beta: float = typer.Option(1.0, "--trades-beta"),
) -> None:
    """Run adversarial training cycles: attack → update → store."""
    import json as _json
    import numpy as np
    from wintermute.models.fusion import DetectorConfig, WintermuteMalwareDetector
    from wintermute.adversarial.orchestrator import AdversarialOrchestrator

    # Load model
    dp = Path(data_dir)
    with open(dp / "vocab.json") as f:
        voc = _json.load(f)

    manifest_data = _json.loads(Path(manifest).read_text())
    cfg = DetectorConfig(
        vocab_size=manifest_data.get("vocab_size", len(voc)),
        num_classes=manifest_data.get("num_classes", 2),
    )
    detector = WintermuteMalwareDetector(cfg)
    detector.load_weights(model)
    typer.echo(f"Loaded model from {model}")

    # Build sample pool from training data (malicious samples only)
    x_data = np.load(dp / "x_data.npy")
    y_data = np.load(dp / "y_data.npy")
    pool = [
        (x_data[i], int(y_data[i]), "unknown")
        for i in range(len(y_data)) if y_data[i] == 1  # malicious only
    ]
    typer.echo(f"Sample pool: {len(pool)} malicious samples")

    orch = AdversarialOrchestrator(
        model=detector, vocab=voc, sample_pool=pool, trades_beta=trades_beta,
    )

    for c in range(cycles):
        metrics = orch.run_cycle(n_episodes=episodes)
        typer.echo(_json.dumps(metrics, indent=2))


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app()
