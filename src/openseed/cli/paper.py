"""Paper management commands."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from openseed.agent.reader import discover_papers, enrich_citations
from openseed.models.paper import Paper, Tag, paper_to_bibtex
from openseed.models.watch import ArxivWatch
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


def _search_download(ctx: click.Context, results: list[dict]) -> None:
    selected = results[:10]
    console.print(f"\n[bold]Downloading top {len(selected)} papers…[/bold]\n")
    lib = _get_library(ctx)
    for r in selected:
        try:
            p = asyncio.run(fetch_paper_metadata(r["arxiv_id"]))
        except Exception as exc:
            console.print(f"[red]Failed to fetch {r['arxiv_id']}: {exc}[/red]")
            continue
        _download_and_extract(ctx, p, r["arxiv_id"])
        added = lib.add_paper(p)
        status = "[green]✓ Added[/green]" if added else "[yellow]Already exists[/yellow]"
        console.print(f"{status} [bold]{p.title}[/bold] (id: {p.id})")


def _fmt_citations(n: int) -> str:
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


@paper.command("search")
@click.argument("query")
@click.option("--count", default=10, show_default=True, help="Number of results to find.")
@click.option("--since", default=None, type=int, metavar="YEAR", help="Filter by publication year.")
@click.option("--add", is_flag=True, help="Auto-add the top result to the library.")
@click.option("--download", is_flag=True, help="Select papers to add and download PDFs.")
@click.pass_context
def search(
    ctx: click.Context, query: str, count: int, since: int | None, add: bool, download: bool
) -> None:
    """Search for papers using AI, ranked by freshness-weighted score."""
    config = ctx.obj["config"]
    with console.status("[cyan]Searching…[/cyan]") as status:

        def _on_step(label: str) -> None:
            status.update(f"[cyan]{label}[/cyan]")

        papers = discover_papers(
            query, model=config.default_model, count=count, since_year=since, on_step=_on_step
        )
        status.update(f"[cyan]Found {len(papers)} papers — verifying citations…[/cyan]")
        results = enrich_citations(papers)

    if since:
        results = [r for r in results if r.get("year", 0) >= since]

    if not results:
        console.print("[dim]No results found.[/dim]")
        return

    title = f"Search: {query}" + (f" (since {since})" if since else "")
    table = Table(title=title, show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Title", style="bold", max_width=42)
    table.add_column("Relevance", max_width=30)
    table.add_column("Score", justify="right", width=7)
    table.add_column("Year", justify="right", width=6)
    table.add_column("Cite", justify="right", width=7)
    table.add_column("Authors", style="dim", max_width=20)
    table.add_column("ArXiv ID", style="cyan", width=13)

    for i, r in enumerate(results, 1):
        table.add_row(
            str(i),
            r["title"],
            r["relevance"],
            f"{r.get('score', 0):.1f}",
            str(r.get("year", "")),
            _fmt_citations(r["citations"]),
            r["authors"],
            r["arxiv_id"],
        )

    console.print(table)

    if download:
        _search_download(ctx, results)
        return

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


@paper.command("next")
@click.pass_context
def next_paper(ctx: click.Context) -> None:
    """Show the oldest unread paper and mark it as reading."""
    lib = _get_library(ctx)
    unread = [p for p in lib.list_papers() if p.status == "unread"]
    if not unread:
        console.print("[dim]No unread papers.[/dim]")
        return
    p = min(unread, key=lambda x: x.added_at)
    p.status = "reading"
    lib.update_paper(p)
    authors = ", ".join(a.name for a in p.authors) if p.authors else "Unknown"
    excerpt = p.abstract[:300] if p.abstract else ""
    console.print(
        Panel(
            f"[bold]{p.title}[/bold]\n{authors}\n\n{excerpt}",
            title=f"Now reading · {p.id}",
            border_style="cyan",
        )
    )


@paper.command("done")
@click.argument("paper_id")
@click.option("--note", default="", help="Reading note to save.")
@click.pass_context
def done(ctx: click.Context, paper_id: str, note: str) -> None:
    """Mark a paper as read and optionally save a note."""
    lib = _get_library(ctx)
    p = lib.get_paper(paper_id)
    if not p:
        console.print(f"[red]Paper {paper_id} not found.[/red]")
        raise SystemExit(1)
    p.status = "read"
    if note:
        p.note = note
    lib.update_paper(p)
    console.print(f"[green]✓[/green] Marked [bold]{p.title}[/bold] as read")


@paper.command("export")
@click.argument("paper_ids", nargs=-1, required=True)
@click.option("--format", "fmt", type=click.Choice(["bibtex"]), default="bibtex", show_default=True)
@click.option("--output", "out_path", default=None, help="Write to file instead of stdout.")
@click.pass_context
def export(ctx: click.Context, paper_ids: tuple[str, ...], fmt: str, out_path: str | None) -> None:
    """Export papers to BibTeX (or other formats)."""
    lib = _get_library(ctx)
    entries = []
    for pid in paper_ids:
        p = lib.get_paper(pid)
        if not p:
            console.print(f"[red]Paper {pid} not found.[/red]")
            raise SystemExit(1)
        entries.append(paper_to_bibtex(p))
    content = "\n\n".join(entries)
    if out_path:
        Path(out_path).write_text(content)
        n = len(entries)
        console.print(f"[green]✓[/green] Wrote {n} entr{'y' if n == 1 else 'ies'} to {out_path}")
    else:
        console.print(content)


def _arxiv_year(arxiv_id: str | None) -> int | None:
    if not arxiv_id:
        return None
    m = re.match(r"^(\d{2})\d{2}\.", arxiv_id)
    return (2000 + int(m.group(1))) if m else None


@paper.group("watch")
def watch_group() -> None:
    """Manage arXiv paper watches."""


@watch_group.command("add")
@click.argument("query")
@click.option("--since", "since_year", default=None, type=int, metavar="YEAR", help="Min year.")
@click.pass_context
def watch_add(ctx: click.Context, query: str, since_year: int | None) -> None:
    """Add a new arXiv watch query."""
    lib = _get_library(ctx)
    w = ArxivWatch(query=query, since_year=since_year)
    lib.add_watch(w)
    since_str = f" (since {since_year})" if since_year else ""
    console.print(f"[green]✓[/green] Watch '{query}'{since_str} added (id: {w.id})")


@watch_group.command("list")
@click.pass_context
def watch_list(ctx: click.Context) -> None:
    """List all watches."""
    lib = _get_library(ctx)
    watches = lib.list_watches()
    if not watches:
        console.print("[dim]No watches configured.[/dim]")
        return
    table = Table(title="Watches", show_lines=True)
    table.add_column("ID", style="cyan", width=12)
    table.add_column("Query", style="bold")
    table.add_column("Since", width=6)
    table.add_column("Last run", width=20)
    for w in watches:
        last = w.last_run.strftime("%Y-%m-%d %H:%M") if w.last_run else "never"
        table.add_row(w.id, w.query, str(w.since_year or ""), last)
    console.print(table)


@watch_group.command("remove")
@click.argument("watch_id")
@click.pass_context
def watch_remove(ctx: click.Context, watch_id: str) -> None:
    """Remove a watch by ID."""
    lib = _get_library(ctx)
    if lib.remove_watch(watch_id):
        console.print(f"[green]✓[/green] Removed watch {watch_id}")
    else:
        console.print(f"[red]Watch {watch_id} not found.[/red]")
        raise SystemExit(1)


def _run_watch(lib, w: ArxivWatch) -> None:
    from datetime import UTC, datetime

    from openseed.services.arxiv import search_papers

    console.print(f"\n[bold cyan]Watch:[/bold cyan] '{w.query}'")
    papers = search_papers(w.query, max_results=20)
    results = [
        p for p in papers if w.since_year is None or (_arxiv_year(p.arxiv_id) or 0) >= w.since_year
    ]
    w.last_run = datetime.now(UTC)
    lib.update_watch(w)
    if not results:
        console.print("  [dim]No results.[/dim]")
        return
    for p in results:
        year = _arxiv_year(p.arxiv_id) or ""
        console.print(f"  [{year}] [bold]{p.title}[/bold]  [dim]{p.arxiv_id}[/dim]")


@watch_group.command("run")
@click.pass_context
def watch_run(ctx: click.Context) -> None:
    """Run all watches and show new papers."""
    lib = _get_library(ctx)
    watches = lib.list_watches()
    if not watches:
        console.print("[dim]No watches configured.[/dim]")
        return
    for w in watches:
        _run_watch(lib, w)
