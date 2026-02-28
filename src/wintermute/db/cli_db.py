"""CLI commands for Wintermute database management."""

from __future__ import annotations

import typer

db_app = typer.Typer(
    name="db",
    help="Database management commands",
    no_args_is_help=True,
)


def register_db_commands(parent_app: typer.Typer) -> None:
    """Register the db sub-Typer on the parent app."""
    parent_app.add_typer(db_app, name="db")


@db_app.command()
def init() -> None:
    """Create all database tables."""
    from wintermute.db.engine import create_db_engine, init_db

    engine = create_db_engine()
    init_db(engine)
    typer.echo("Database tables created successfully.")


@db_app.command()
def stats() -> None:
    """Show database statistics."""
    from wintermute.db.engine import create_db_engine, get_session
    from wintermute.db.repos.samples import SampleRepo
    from wintermute.db.repos.scans import ScanRepo

    create_db_engine()

    with get_session() as session:
        sample_repo = SampleRepo(session)
        scan_repo = ScanRepo(session)

        # Family counts
        families = sample_repo.count_by_family()
        total_samples = sum(families.values())

        # Source counts
        sources = sample_repo.count_by_source()

        # Scan stats
        scan_stats = scan_repo.stats()

        typer.echo(f"\n{'='*50}")
        typer.echo("  DATABASE STATISTICS")
        typer.echo(f"{'='*50}")
        typer.echo(f"\n  Samples: {total_samples}")
        for fam, count in sorted(families.items()):
            typer.echo(f"    {fam}: {count}")
        typer.echo("\n  Sources:")
        for src, count in sorted(sources.items()):
            typer.echo(f"    {src}: {count}")
        typer.echo(f"\n  Scans: {scan_stats.get('total_scans', 0)}")
        typer.echo(
            f"  Mean confidence: {scan_stats.get('avg_confidence', 0):.2f}"
        )
        typer.echo(f"{'='*50}\n")


@db_app.command()
def samples(
    family: str = typer.Option(None, "--family", "-f", help="Filter by family"),
    source: str = typer.Option(None, "--source", "-s", help="Filter by source"),
    min_opcodes: int = typer.Option(
        None, "--min-opcodes", help="Minimum opcode count"
    ),
    limit: int = typer.Option(50, "--limit", "-n", help="Max results"),
) -> None:
    """Query samples with filters."""
    from wintermute.db.engine import create_db_engine, get_session
    from wintermute.db.repos.samples import SampleRepo

    create_db_engine()

    with get_session() as session:
        results = SampleRepo(session).find(
            family=family,
            source=source,
            min_opcodes=min_opcodes,
            limit=limit,
        )
        if not results:
            typer.echo("No samples found.")
            return
        typer.echo(f"\n  Found {len(results)} sample(s):\n")
        for s in results:
            typer.echo(
                f"  {s.sha256[:16]}..  {s.family:<15} "
                f"{s.source:<20} opcodes={s.opcode_count or '?'}"
            )


@db_app.command()
def scans(
    recent: int = typer.Option(
        None, "--recent", "-r", help="Show N most recent scans"
    ),
    sha256: str = typer.Option(None, "--sha256", help="Filter by SHA256"),
    uncertain: float = typer.Option(
        None, "--uncertain", help="Show scans below confidence threshold"
    ),
    limit: int = typer.Option(50, "--limit", "-n", help="Max results"),
) -> None:
    """Query scan results with filters."""
    from wintermute.db.engine import create_db_engine, get_session
    from wintermute.db.repos.scans import ScanRepo

    create_db_engine()

    with get_session() as session:
        repo = ScanRepo(session)
        if sha256:
            results = repo.history(sha256)
        elif uncertain is not None:
            results = repo.uncertain(threshold=uncertain, limit=limit)
        elif recent:
            results = repo.recent(limit=recent)
        else:
            results = repo.recent(limit=limit)

        if not results:
            typer.echo("No scan results found.")
            return
        typer.echo(f"\n  Found {len(results)} scan(s):\n")
        for s in results:
            label_str = "SAFE" if s.predicted_label == 0 else "MAL"
            typer.echo(
                f"  [{label_str}] {s.sha256[:16]}..  "
                f"{s.predicted_family:<15} "
                f"conf={s.confidence:.2f}  "
                f"{s.scanned_at.strftime('%Y-%m-%d %H:%M')}"
            )


@db_app.command()
def models(
    promote: str = typer.Option(
        None, "--promote", help="Promote model ID to active"
    ),
) -> None:
    """List models or promote a version."""
    from wintermute.db.engine import create_db_engine, get_session
    from wintermute.db.repos.models_repo import ModelRepo

    create_db_engine()

    with get_session() as session:
        repo = ModelRepo(session)
        if promote:
            repo.promote(promote)
            typer.echo(f"Model {promote} promoted to active.")
            return

        model_list = repo.history(limit=20)
        if not model_list:
            typer.echo("No models registered.")
            return
        typer.echo("\n  Registered models:\n")
        for m in model_list:
            status_tag = {
                "active": "[ACTIVE]",
                "staged": "[STAGED]",
                "retired": "[RETIRED]",
            }.get(m.status, f"[{m.status}]")
            f1 = (
                f"f1={m.best_val_macro_f1:.4f}"
                if m.best_val_macro_f1
                else "f1=n/a"
            )
            typer.echo(
                f"  {status_tag:<10} {m.version:<20} {f1}  {m.architecture}"
            )


@db_app.command()
def similar(
    sha256: str = typer.Argument(
        ..., help="SHA256 of sample to find neighbors for"
    ),
    k: int = typer.Option(5, "--k", "-k", help="Number of neighbors"),
) -> None:
    """Find similar samples by embedding (k-NN search)."""
    import struct

    from wintermute.db.engine import create_db_engine, get_session
    from wintermute.db.repos.embeddings import EmbeddingRepo
    from wintermute.db.repos.samples import SampleRepo

    create_db_engine()

    with get_session() as session:
        sample = SampleRepo(session).get(sha256)
        if not sample:
            typer.echo(f"Sample not found: {sha256}", err=True)
            raise typer.Exit(1)
        if not sample.embedding:
            typer.echo(f"Sample has no embedding: {sha256}", err=True)
            raise typer.Exit(1)

        # Unpack bytes to list[float]
        dim = len(sample.embedding) // 4
        query_vec = list(struct.unpack(f"{dim}f", sample.embedding))

        neighbors = EmbeddingRepo(session).find_nearest(query_vec, k=k)
        if not neighbors:
            typer.echo("No similar samples found.")
            return
        typer.echo(f"\n  Top {len(neighbors)} similar samples:\n")
        for n in neighbors:
            typer.echo(
                f"  {n['sha256'][:16]}..  {n['family']:<15} "
                f"distance={n['distance']:.4f}"
            )


@db_app.command()
def vault(
    unused: bool = typer.Option(
        True, "--unused/--all", help="Show unused variants only"
    ),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results"),
) -> None:
    """Show adversarial vault stats and variants."""
    from wintermute.db.engine import create_db_engine, get_session
    from wintermute.db.repos.adversarial import AdversarialRepo

    create_db_engine()

    with get_session() as session:
        repo = AdversarialRepo(session)
        variants = repo.get_vault(limit=limit, unused_only=unused)
        report = repo.vulnerability_report()

        typer.echo(f"\n{'='*50}")
        typer.echo("  ADVERSARIAL VAULT")
        typer.echo(f"{'='*50}")
        typer.echo(
            f"\n  Variants: {len(variants)} "
            f"{'(unused only)' if unused else '(all)'}"
        )

        if report:
            typer.echo("\n  Vulnerability Report:")
            for r in report:
                typer.echo(
                    f"    {r['family']:<15} attacks={r['total_attacks']}  "
                    f"evasions={r['evasions']}  "
                    f"rate={r['evasion_rate']:.1%}"
                )
        typer.echo(f"{'='*50}\n")


@db_app.command()
def embed() -> None:
    """Show embedding coverage statistics."""
    from wintermute.db.engine import create_db_engine, get_session
    from wintermute.db.repos.embeddings import EmbeddingRepo

    create_db_engine()

    with get_session() as session:
        coverage = EmbeddingRepo(session).coverage_stats()
        typer.echo("\n  Embedding Coverage:")
        typer.echo(f"    Total samples: {coverage.get('total_samples', 0)}")
        typer.echo(f"    With embeddings: {coverage.get('with_embedding', 0)}")
        pct = coverage.get("pct_covered", 0)
        typer.echo(f"    Coverage: {pct:.1f}%\n")
