"""cli_etl.py — Typer CLI commands for the ETL pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

logger = logging.getLogger("wintermute.data.etl")


def register_etl_commands(data_app: typer.Typer) -> None:
    """Register ETL commands on the ``data`` sub-app."""

    @data_app.command("etl")
    def data_etl(
        source: str = typer.Option(
            None, "--source", "-s",
            help="Run only this source (by name).",
        ),
        out_dir: str = typer.Option(
            None, "--out-dir", "-o",
            help="Override output directory.",
        ),
        config: str = typer.Option(
            None, "--config", "-c",
            help="Path to sources.yaml config file.",
        ),
        dry_run: bool = typer.Option(
            False, "--dry-run",
            help="Validate config without extracting.",
        ),
        list_sources: bool = typer.Option(
            False, "--list-sources",
            help="List all registered source plugins and exit.",
        ),
        verbose: bool = typer.Option(
            False, "--verbose", "-v",
            help="Enable verbose (DEBUG) logging.",
        ),
    ) -> None:
        """Run the ETL pipeline to build a training dataset."""
        # Configure logging
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(levelname)-8s %(name)s: %(message)s",
        )

        # Import here to trigger source auto-registration
        import wintermute.data.etl.sources  # noqa: F401
        from wintermute.data.etl.pipeline import Pipeline
        from wintermute.data.etl.registry import SourceRegistry

        if list_sources:
            names = SourceRegistry.available()
            typer.echo(f"Registered sources ({len(names)}):")
            for name in names:
                cls = SourceRegistry.get(name)
                doc = (cls.__doc__ or "").strip().split("\n")[0] if cls else ""
                typer.echo(f"  {name:<20} {doc}")
            return

        # Build pipeline config
        pipeline = Pipeline(config_path=config)

        # Apply CLI overrides
        if out_dir:
            pipeline.out_dir = Path(out_dir)

        # Run
        result = pipeline.run(source_filter=source, dry_run=dry_run)

        # Print summary
        typer.echo(f"\n{'=' * 60}")
        if dry_run:
            typer.echo("  DRY RUN — no files written")
        typer.echo(f"  Samples:    {result.total_samples}")
        typer.echo(f"  Classes:    {result.num_classes}")
        typer.echo(f"  Vocab:      {result.vocab_size}")
        typer.echo(f"  x_shape:    {result.x_shape}")
        typer.echo(f"  y_shape:    {result.y_shape}")
        typer.echo(f"  Output:     {result.output_dir}")
        typer.echo(f"  Time:       {result.elapsed_seconds:.1f}s")

        for er in result.extract_results:
            typer.echo(f"\n  Source '{er.source_name}':")
            typer.echo(f"    extracted={er.samples_extracted}  "
                       f"skipped={er.samples_skipped}  "
                       f"failed={er.samples_failed}  "
                       f"time={er.elapsed_seconds:.1f}s")
            if er.families_found:
                for fam, cnt in sorted(er.families_found.items()):
                    typer.echo(f"      {fam}: {cnt}")
            if er.errors:
                for err in er.errors:
                    typer.echo(f"    [ERROR] {err}")

        typer.echo(f"{'=' * 60}\n")

    @data_app.command("etl-sources")
    def data_etl_sources() -> None:
        """List available ETL source plugins with descriptions."""
        import wintermute.data.etl.sources  # noqa: F401
        from wintermute.data.etl.registry import SourceRegistry

        names = SourceRegistry.available()
        typer.echo(f"Available ETL sources ({len(names)}):\n")
        for name in names:
            cls = SourceRegistry.get(name)
            doc = (cls.__doc__ or "No description.").strip() if cls else "No description."
            typer.echo(f"  {name}")
            for line in doc.split("\n"):
                typer.echo(f"    {line.strip()}")
            typer.echo()
