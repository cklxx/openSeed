"""Paper management commands."""

from __future__ import annotations

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from openseed.models.paper import Paper, Tag
from openseed.services.arxiv import search_papers
from openseed.storage.library import PaperLibrary

console = Console()


def _get_library(ctx: click.Context) -> PaperLibrary:
    config = ctx.obj["config"]
    return PaperLibrary(config.library_dir)


@click.group()
@click.pass_context
def paper(ctx: click.Context) -> None:
    """Manage research papers."""
    ctx.ensure_object(dict)


@paper.command()
@click.argument("url")
@click.pass_context
def add(ctx: click.Context, url: str) -> None:
    """Add a paper by URL (ArXiv or direct link)."""
    lib = _get_library(ctx)
    p = Paper(title=url, url=url)

    # Try to extract ArXiv ID
    if "arxiv.org" in url:
        parts = url.rstrip("/").split("/")
        arxiv_id = parts[-1]
        p.arxiv_id = arxiv_id
        p.title = f"ArXiv paper {arxiv_id}"

    lib.add_paper(p)
    console.print(f"[green]✓[/green] Added paper [bold]{p.title}[/bold] (id: {p.id})")


@paper.command("list")
@click.option("--status", type=click.Choice(["unread", "reading", "read", "archived"]))
@click.option("--tag", help="Filter by tag name")
@click.pass_context
def list_papers(ctx: click.Context, status: str | None, tag: str | None) -> None:
    """List papers in the library."""
    lib = _get_library(ctx)
    papers = lib.list_papers()

    if status:
        papers = [p for p in papers if p.status == status]
    if tag:
        papers = [p for p in papers if any(t.name == tag for t in p.tags)]

    if not papers:
        console.print("[dim]No papers found.[/dim]")
        return

    table = Table(title="Papers")
    table.add_column("ID", style="cyan", width=12)
    table.add_column("Title", style="bold")
    table.add_column("Status", width=10)
    table.add_column("Tags")

    for p in papers:
        tags = ", ".join(t.name for t in p.tags) if p.tags else ""
        table.add_row(p.id, p.title, p.status, tags)

    console.print(table)


@paper.command()
@click.argument("paper_id")
@click.pass_context
def show(ctx: click.Context, paper_id: str) -> None:
    """Show paper details."""
    lib = _get_library(ctx)
    p = lib.get_paper(paper_id)
    if not p:
        console.print(f"[red]Paper {paper_id} not found.[/red]")
        raise SystemExit(1)

    authors = ", ".join(a.name for a in p.authors) if p.authors else "Unknown"
    content = (
        f"[bold]Title:[/bold] {p.title}\n"
        f"[bold]Authors:[/bold] {authors}\n"
        f"[bold]Status:[/bold] {p.status}\n"
        f"[bold]ArXiv:[/bold] {p.arxiv_id or 'N/A'}\n"
        f"[bold]Added:[/bold] {p.added_at:%Y-%m-%d}\n"
    )
    if p.abstract:
        content += f"\n[bold]Abstract:[/bold]\n{p.abstract}"
    if p.summary:
        content += f"\n\n[bold]Summary:[/bold]\n{p.summary}"

    console.print(Panel(content, title=f"Paper {p.id}", border_style="blue"))


@paper.command()
@click.argument("paper_id")
@click.pass_context
def remove(ctx: click.Context, paper_id: str) -> None:
    """Remove a paper from the library."""
    lib = _get_library(ctx)
    if lib.remove_paper(paper_id):
        console.print(f"[green]✓[/green] Removed paper {paper_id}")
    else:
        console.print(f"[red]Paper {paper_id} not found.[/red]")
        raise SystemExit(1)


@paper.command()
@click.argument("paper_id")
@click.argument("name")
@click.option("--color", default="blue", help="Tag color")
@click.pass_context
def tag(ctx: click.Context, paper_id: str, name: str, color: str) -> None:
    """Add a tag to a paper."""
    lib = _get_library(ctx)
    p = lib.get_paper(paper_id)
    if not p:
        console.print(f"[red]Paper {paper_id} not found.[/red]")
        raise SystemExit(1)

    p.tags.append(Tag(name=name, color=color))
    lib.update_paper(p)
    console.print(f"[green]✓[/green] Tagged [bold]{p.title}[/bold] with '{name}'")


@paper.command("search")
@click.argument("query")
@click.option("--max-results", default=10, show_default=True, help="Maximum number of results.")
@click.option("--add", is_flag=True, help="Auto-add the first result to the library.")
@click.pass_context
def search(ctx: click.Context, query: str, max_results: int, add: bool) -> None:
    """Search ArXiv for papers by keyword."""
    with console.status(f"[cyan]Searching ArXiv for '{query}'…[/cyan]"):
        results = search_papers(query, max_results=max_results)

    if not results:
        console.print("[dim]No results found.[/dim]")
        return

    table = Table(title=f"ArXiv Search: {query}", show_lines=False)
    table.add_column("#", style="dim", width=4)
    table.add_column("ArXiv ID", style="cyan", width=15)
    table.add_column("Title", style="bold", max_width=50)
    table.add_column("Authors", max_width=30)
    table.add_column("Year", width=6)

    for i, p in enumerate(results, 1):
        authors_str = ", ".join(a.name for a in p.authors[:3])
        if len(p.authors) > 3:
            authors_str += f" +{len(p.authors) - 3}"
        year = f"20{p.arxiv_id[:2]}" if p.arxiv_id and len(p.arxiv_id) >= 2 else "—"
        table.add_row(str(i), p.arxiv_id or "—", p.title, authors_str, year)

    console.print(table)

    if add and results:
        lib = _get_library(ctx)
        first = results[0]
        lib.add_paper(first)
        console.print(f"[green]✓[/green] Added [bold]{first.title}[/bold] (id: {first.id})")


@paper.command()
@click.argument("paper_id")
@click.argument("new_status", type=click.Choice(["unread", "reading", "read", "archived"]))
@click.pass_context
def status(ctx: click.Context, paper_id: str, new_status: str) -> None:
    """Update paper reading status."""
    lib = _get_library(ctx)
    p = lib.get_paper(paper_id)
    if not p:
        console.print(f"[red]Paper {paper_id} not found.[/red]")
        raise SystemExit(1)

    p.status = new_status  # type: ignore[assignment]
    lib.update_paper(p)
    console.print(f"[green]✓[/green] Status updated to '{new_status}'")
