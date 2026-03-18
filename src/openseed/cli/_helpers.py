"""Shared CLI utilities."""

from __future__ import annotations

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from openseed.config import OpenSeedConfig
from openseed.models.paper import Paper
from openseed.storage.library import PaperLibrary

console = Console()


def get_library(ctx: click.Context) -> PaperLibrary:
    return PaperLibrary(ctx.obj["config"].library_dir)


def get_config(ctx: click.Context) -> OpenSeedConfig:
    return ctx.obj["config"]


def require_paper(lib: PaperLibrary, paper_id: str) -> Paper:
    """Return paper or exit with error."""
    p = lib.get_paper(paper_id)
    if not p:
        console.print(f"[red]Paper {paper_id} not found.[/red]")
        raise SystemExit(1)
    return p


def _build_metrics_table(metrics: list[dict]) -> Table:
    table = Table(title="Key Metrics", box=None, padding=(0, 2))
    table.add_column("Metric", style="bold", width=18)
    table.add_column("Proposed", justify="right", width=10)
    table.add_column("Baseline", justify="right", width=10)
    table.add_column("Δ", justify="right", width=9)
    for m in metrics:
        p, b = float(m.get("proposed", 0)), float(m.get("baseline", 0))
        delta = f"[green]+{p - b:.2f}[/green]" if p >= b else f"[red]{p - b:.2f}[/red]"
        table.add_row(m["name"], f"{p:.2f}", f"{b:.2f}", delta)
    return table


def library_status_for_arxiv(lib: PaperLibrary, arxiv_id: str) -> str:
    """Return a Rich-formatted status badge if the paper is in the local library."""
    paper = lib.get_paper_by_arxiv(arxiv_id)
    if not paper:
        return ""
    if paper.summary:
        return "[green]✓ analyzed[/green]"
    return "[dim]in lib[/dim]"


def render_paper_visuals(data: dict, out: Console) -> None:
    """Render method pipeline panel and metrics comparison table."""
    if pipeline := data.get("pipeline"):
        steps = "  →  ".join(f"[cyan bold]{s}[/cyan bold]" for s in pipeline)
        out.print(Panel(steps, title="Method Pipeline", border_style="blue"))
    if metrics := data.get("metrics"):
        out.print(_build_metrics_table(metrics))
