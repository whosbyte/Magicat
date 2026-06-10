# magicat/cli.py
"""CLI entry point: `magicat run <url-or-file> [--workdir PATH]`."""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

import typer

from magicat.core.pipeline import run_job

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Magicat: deconstruct short-form videos into editable layers."""


@app.command()
def run(
    input_arg: str = typer.Argument(..., metavar="URL_OR_FILE"),
    workdir: Path | None = typer.Option(None, "--workdir"),
) -> None:
    """Deconstruct a short-form video into a layered project."""
    logging.basicConfig(level=logging.INFO)
    if workdir is None:
        workdir = Path("jobs") / uuid.uuid4().hex[:12]   # fresh dir per job
    try:
        manifest = run_job(input_arg, workdir)
    except Exception as exc:  # ingest failure is fatal (spec section 5)
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"job: {manifest.job_id}")
    typer.echo(f"source: {manifest.source.file} "
               f"({manifest.source.resolution}, {manifest.source.duration}s)")
    typer.echo(f"shots: {len(manifest.shots)}")
    for layer, state in manifest.layers_status.items():
        typer.echo(f"  layer {layer}: {state.value}")
    for export in manifest.exports:
        typer.echo(f"  export {export.format}: {export.artifact}")
    typer.echo(f"manifest: {Path(workdir).resolve() / 'manifest.json'}")


if __name__ == "__main__":
    app()
