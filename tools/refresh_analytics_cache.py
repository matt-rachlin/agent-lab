"""Refresh the local DuckDB analytics cache (~/.cache/lab/analytics.duckdb).

Usage:
    uv run python -m tools.refresh_analytics_cache              # refresh if stale
    uv run python -m tools.refresh_analytics_cache --force      # always refresh
    uv run python -m tools.refresh_analytics_cache --db PATH    # alternate location
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from lab.analyze.duckdb_cache import (
    CACHE_DEFAULT,
    MIRRORED_TABLES,
    is_stale,
    refresh,
)

app = typer.Typer(add_completion=False, no_args_is_help=False)
console = Console()


@app.command()
def main(
    force: bool = typer.Option(False, "--force", "-f", help="Refresh even if the cache is fresh."),
    db: Path = typer.Option(CACHE_DEFAULT, "--db", help="Path to the DuckDB cache file."),
    max_age_min: int = typer.Option(30, "--max-age-min", help="Cache is stale beyond this age."),
) -> None:
    """Refresh the cache. Skips if fresh (unless --force)."""
    db = db.expanduser()
    if not force and not is_stale(max_age_min=max_age_min, db_path=db):
        console.print(f"[green]cache is fresh:[/green] {db}")
        raise typer.Exit(0)

    console.print(f"refreshing {db} (mirroring {len(MIRRORED_TABLES)} tables)...")
    report = refresh(db_path=db)
    console.print(f"[green]done in {report.wall_time_sec:.2f}s[/green]")
    for t in MIRRORED_TABLES:
        n = report.table_counts.get(t, 0)
        console.print(f"  {t:18s} {n:>7d} rows")


if __name__ == "__main__":
    app()
