"""Paper management commands."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from openseed.agent.reader import discover_papers, enrich_citations
from openseed.models.paper import Paper, Tag
from openseed.services.arxiv import (
    download_pdf,
    fetch_paper_metadata,
    parse_arxiv_id,
)
from openseed.services.pdf import extract_text
from openseed.storage.library import PaperLibrary

console = Console()


def _get_library(ctx: click.Context) -> PaperLibrary:
    config = ctx.obj["config"]
    return PaperLibrary(config.library_dir)


def _pdf_dir(ctx: click.Context) -> Path:
    config = ctx.obj["config"]
    pdf_path = Path(config.library_dir) / "pdfs"
    pdf_path.mkdir(parents=True, exist_ok=True)
    return pdf_path


@click.group()
@click.pass_context
def paper(ctx: click.Context) -> None:
    """Manage research papers."""
    ctx.ensure_object(dict)


@paper.command()
@click.argument("url")
@click.option("--fetch-pdf/--no-fetch-pdf", default=False, help="Download and extract PDF text")
@click.pass_context
def add(ctx: click.Context, url: str, fetch_pdf: bool) -> None:
    """Add a paper by URL (ArXiv or direct link)."""
    lib = _get_library(ctx)

    arxiv_id = parse_arxiv_id(url)
    if arxiv_id:
        try:
            p = asyncio.run(fetch_paper_metadata(arxiv_id))
            console.print(f"[dim]Fetched metadata for {arxiv_id}[/dim]")
        except Exception as exc:
            console.print(f"[yellow]Warning:[/yellow] Could not fetch metadata: {exc}")
            p = Paper(
                title=f"ArXiv paper {arxiv_id}",
                arxiv_id=arxiv_id,
                url=f"https://arxiv.org/abs/{arxiv_id}",
            )
        if fetch_pdf:
            _download_and_extract(ctx, p, arxiv_id)
    else:
        p = Paper(title=url, url=url)

    lib.add_paper(p)
    console.print(f"[green]✓[/green] Added paper [bold]{p.title}[/bold] (id: {p.id})")


def _download_and_extract(ctx: click.Context, paper: Paper, arxiv_id: str) -> None:
    """Download PDF and extract text for a paper."""
    dest = _pdf_dir(ctx) / f"{arxiv_id}.pdf"
    try:
        console.print(f"[dim]Downloading PDF for {arxiv_id}...[/dim]")
        asyncio.run(download_pdf(arxiv_id, str(dest)))
        paper.pdf_path = str(dest)
        console.print("[dim]Extracting text...[/dim]")
        text = extract_text(str(dest))
        paper.summary = text[:5000] if len(text) > 5000 else text
        console.print(f"[green]✓[/green] Extracted {len(text)} chars of text")
    except Exception as exc:
        console.print(f"[yellow]Warning:[/yellow] PDF processing failed: {exc}")


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
    table.add_column("ArXiv", width=14)
    table.add_column("Status", width=10)
    table.add_column("Tags")

    for p in papers:
        tags = ", ".join(t.name for t in p.tags) if p.tags else ""
        table.add_row(p.id, p.title, p.arxiv_id or "", p.status, tags)

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
        f"[bold]PDF:[/bold] {p.pdf_path or 'N/A'}\n"
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
def fetch(ctx: click.Context, paper_id: str) -> None:
    """Download PDF and extract text for an existing paper."""
    lib = _get_library(ctx)
    p = lib.get_paper(paper_id)
    if not p:
        console.print(f"[red]Paper {paper_id} not found.[/red]")
        raise SystemExit(1)

    if not p.arxiv_id:
        console.print("[red]Paper has no ArXiv ID — PDF download not supported.[/red]")
        raise SystemExit(1)

    _download_and_extract(ctx, p, p.arxiv_id)
    lib.update_paper(p)
    console.print(f"[green]✓[/green] Updated paper [bold]{p.title}[/bold] with extracted content")


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


def _fmt_citations(n: int) -> str:
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


@paper.command("search")
@click.argument("query")
@click.option("--count", default=10, show_default=True, help="Number of results to find.")
@click.option("--add", is_flag=True, help="Auto-add the top result to the library.")
@click.pass_context
def search(ctx: click.Context, query: str, count: int, add: bool) -> None:
    """Search for papers using AI, sorted by citation count."""
    config = ctx.obj["config"]
    with console.status(f"[cyan]Searching '{query}'…[/cyan]") as status:
        papers = discover_papers(query, model=config.default_model, count=count)
        status.update(f"[cyan]Found {len(papers)} papers — verifying citations…[/cyan]")
        results = enrich_citations(papers)

    if not results:
        console.print("[dim]No results found.[/dim]")
        return

    table = Table(title=f"Search: {query} (by freshness-weighted score)", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("ArXiv ID", style="cyan", width=13)
    table.add_column("Title", style="bold", max_width=38)
    table.add_column("Authors", max_width=20)
    table.add_column("Year", justify="right", width=6)
    table.add_column("Cite", justify="right", width=7)
    table.add_column("Score", justify="right", width=7)
    table.add_column("Relevance", max_width=28)

    for i, r in enumerate(results, 1):
        table.add_row(
            str(i),
            r["arxiv_id"],
            r["title"],
            r["authors"],
            str(r.get("year", "")),
            _fmt_citations(r["citations"]),
            f"{r.get('score', 0):.1f}",
            r["relevance"],
        )

    console.print(table)

    if add and results:
        lib = _get_library(ctx)
        top = results[0]
        try:
            paper_obj = asyncio.run(fetch_paper_metadata(top["arxiv_id"]))
        except Exception:
            paper_obj = Paper(title=top["title"], arxiv_id=top["arxiv_id"])
        lib.add_paper(paper_obj)
        console.print(f"[green]✓[/green] Added [bold]{paper_obj.title}[/bold] (id: {paper_obj.id})")


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
