"""Paper management commands."""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from openseed.agent.reader import (
    PaperReader,
    discover_papers,
    enrich_citations,
    extract_paper_visuals,
)
from openseed.cli._helpers import get_config, get_library, render_paper_visuals, require_paper
from openseed.models.paper import Paper, Tag, paper_to_bibtex
from openseed.models.watch import ArxivWatch
from openseed.services.arxiv import (
    download_pdf,
    fetch_paper_metadata,
    parse_arxiv_id,
    search_papers,
)
from openseed.services.pdf import extract_text

console = Console()


def _pdf_dir(ctx: click.Context) -> Path:
    path = Path(get_config(ctx).library_dir) / "pdfs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download_and_extract(ctx: click.Context, paper: Paper, arxiv_id: str) -> None:
    """Download PDF and extract text for a paper."""
    dest = _pdf_dir(ctx) / f"{arxiv_id}.pdf"
    try:
        console.print(f"[dim]Downloading PDF for {arxiv_id}...[/dim]")
        asyncio.run(download_pdf(arxiv_id, str(dest)))
        paper.pdf_path = str(dest)
        text = extract_text(str(dest))
        paper.summary = text[:5000] if len(text) > 5000 else text
        console.print(f"[green]✓[/green] Extracted {len(text)} chars")
    except Exception as exc:
        console.print(f"[yellow]Warning:[/yellow] PDF processing failed: {exc}")


def _fmt_citations(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def _arxiv_year(arxiv_id: str | None) -> int | None:
    m = re.match(r"^(\d{2})\d{2}\.", arxiv_id or "")
    return (2000 + int(m.group(1))) if m else None


def _score_bar(score: float, max_score: float, width: int = 10) -> str:
    return f"[yellow]{score:5.1f}[/yellow]"


def _build_search_table(results: list[dict], since: int | None) -> Table:
    title = "Search results" + (f" (since {since})" if since else "")
    max_score = max((r.get("score", 0) for r in results), default=1) or 1
    table = Table(title=title, show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Title", style="bold", max_width=38)
    table.add_column("Relevance", max_width=28)
    table.add_column("Rank", width=7, justify="right")
    table.add_column("Year", justify="right", width=6)
    table.add_column("Cite", justify="right", width=7)
    table.add_column("Authors", style="dim", max_width=18)
    table.add_column("ArXiv ID", style="cyan", width=13)
    for i, r in enumerate(results, 1):
        table.add_row(
            str(i),
            r["title"],
            r["relevance"],
            _score_bar(r.get("score", 0), max_score),
            str(r.get("year", "")),
            _fmt_citations(r["citations"]),
            r["authors"],
            r["arxiv_id"],
        )
    return table


def _run_discover(query: str, config, count: int, since: int | None) -> list[dict]:
    with console.status("[cyan]Searching…[/cyan]") as status:

        def _on_step(label: str) -> None:
            status.update(f"[cyan]{label}[/cyan]")

        papers = discover_papers(
            query, model=config.default_model, count=count, since_year=since, on_step=_on_step
        )
        status.update(f"[cyan]Found {len(papers)} — verifying citations…[/cyan]")
        return enrich_citations(papers)


def _summarize_cn(p: Paper, config, lib) -> None:
    with console.status(f"[cyan]Summarizing '{p.title[:40]}…'[/cyan]"):
        p.summary = PaperReader(model=config.default_model).summarize_paper(
            p.abstract or p.title, cn=True
        )
    md_path = lib.save_summary(p)
    console.print(Panel(Markdown(p.summary), title=p.title[:60], border_style="green"))
    console.print(f"[dim]Saved → {md_path}[/dim]")
    with console.status("[cyan]Extracting visuals…[/cyan]"):
        visuals = extract_paper_visuals(p.abstract or p.title, config.default_model)
    render_paper_visuals(visuals, console)


def _fetch_and_add(ctx: click.Context, r: dict, lib, config, cn: bool) -> None:
    try:
        p = asyncio.run(fetch_paper_metadata(r["arxiv_id"]))
    except Exception as exc:
        console.print(f"[red]Failed to fetch {r['arxiv_id']}: {exc}[/red]")
        return
    _download_and_extract(ctx, p, r["arxiv_id"])
    if cn:
        _summarize_cn(p, config, lib)
    added = lib.add_paper(p)
    status = "[green]✓ Added[/green]" if added else "[yellow]Already exists[/yellow]"
    console.print(f"{status} [bold]{p.title}[/bold] (id: {p.id})")


def _run_watch(lib, w: ArxivWatch) -> None:
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
    lib = get_library(ctx)
    arxiv_id = parse_arxiv_id(url)
    if arxiv_id:
        try:
            p = asyncio.run(fetch_paper_metadata(arxiv_id))
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
    console.print(f"[green]✓[/green] Added [bold]{p.title}[/bold] (id: {p.id})")


@paper.command("list")
@click.option("--status", type=click.Choice(["unread", "reading", "read", "archived"]))
@click.option("--tag", help="Filter by tag name")
@click.pass_context
def list_papers(ctx: click.Context, status: str | None, tag: str | None) -> None:
    """List papers in the library."""
    lib = get_library(ctx)
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
        table.add_row(p.id, p.title, p.arxiv_id or "", p.status, ", ".join(t.name for t in p.tags))
    console.print(table)


@paper.command()
@click.argument("paper_id")
@click.pass_context
def show(ctx: click.Context, paper_id: str) -> None:
    """Show paper details."""
    lib = get_library(ctx)
    p = require_paper(lib, paper_id)
    authors = ", ".join(a.name for a in p.authors) if p.authors else "Unknown"
    lines = [
        f"[bold]Title:[/bold] {p.title}",
        f"[bold]Authors:[/bold] {authors}",
        f"[bold]Status:[/bold] {p.status}",
        f"[bold]ArXiv:[/bold] {p.arxiv_id or 'N/A'}",
        f"[bold]PDF:[/bold] {p.pdf_path or 'N/A'}",
        f"[bold]Added:[/bold] {p.added_at:%Y-%m-%d}",
    ]
    if p.abstract:
        lines += ["", f"[bold]Abstract:[/bold]\n{p.abstract}"]
    if p.summary:
        lines += ["", f"[bold]Summary:[/bold]\n{p.summary}"]
    if p.note:
        lines += ["", f"[bold]Note:[/bold]\n{p.note}"]
    console.print(Panel("\n".join(lines), title=f"Paper {p.id}", border_style="blue"))


@paper.command()
@click.argument("paper_id")
@click.pass_context
def fetch(ctx: click.Context, paper_id: str) -> None:
    """Download PDF and extract text for an existing paper."""
    lib = get_library(ctx)
    p = require_paper(lib, paper_id)
    if not p.arxiv_id:
        console.print("[red]Paper has no ArXiv ID — PDF download not supported.[/red]")
        raise SystemExit(1)
    _download_and_extract(ctx, p, p.arxiv_id)
    lib.update_paper(p)
    console.print(f"[green]✓[/green] Updated [bold]{p.title}[/bold] with extracted content")


@paper.command()
@click.argument("paper_id")
@click.pass_context
def remove(ctx: click.Context, paper_id: str) -> None:
    """Remove a paper from the library."""
    lib = get_library(ctx)
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
    lib = get_library(ctx)
    p = require_paper(lib, paper_id)
    p.tags.append(Tag(name=name, color=color))
    lib.update_paper(p)
    console.print(f"[green]✓[/green] Tagged [bold]{p.title}[/bold] with '{name}'")


@paper.command("search")
@click.argument("query")
@click.option("--count", default=10, show_default=True, help="Number of results to find.")
@click.option("--since", default=None, type=int, metavar="YEAR", help="Filter by publication year.")
@click.option("--add", is_flag=True, help="Auto-add the top result to the library.")
@click.option("--download", is_flag=True, help="Download top 10 papers and add to library.")
@click.option("--cn", is_flag=True, help="Generate Chinese summary after download.")
@click.pass_context
def search(
    ctx: click.Context,
    query: str,
    count: int,
    since: int | None,
    add: bool,
    download: bool,
    cn: bool,
) -> None:
    """Search for papers using AI, ranked by freshness-weighted score."""
    config = get_config(ctx)
    results = _run_discover(query, config, count, since)
    if since:
        results = [r for r in results if r.get("year", 0) >= since]
    if not results:
        console.print("[dim]No results found.[/dim]")
        return
    console.print(_build_search_table(results, since))
    if download:
        lib = get_library(ctx)
        console.print(f"\n[bold]Downloading top {min(10, len(results))} papers…[/bold]\n")
        for r in results[:10]:
            _fetch_and_add(ctx, r, lib, config, cn)
        return
    if add:
        lib = get_library(ctx)
        top = results[0]
        try:
            p = asyncio.run(fetch_paper_metadata(top["arxiv_id"]))
        except Exception:
            p = Paper(title=top["title"], arxiv_id=top["arxiv_id"])
        lib.add_paper(p)
        console.print(f"[green]✓[/green] Added [bold]{p.title}[/bold] (id: {p.id})")


@paper.command()
@click.argument("paper_id")
@click.argument("new_status", type=click.Choice(["unread", "reading", "read", "archived"]))
@click.pass_context
def status(ctx: click.Context, paper_id: str, new_status: str) -> None:
    """Update paper reading status."""
    lib = get_library(ctx)
    p = require_paper(lib, paper_id)
    p.status = new_status  # type: ignore[assignment]
    lib.update_paper(p)
    console.print(f"[green]✓[/green] Status updated to '{new_status}'")


@paper.command("next")
@click.pass_context
def next_paper(ctx: click.Context) -> None:
    """Show the oldest unread paper and mark it as reading."""
    lib = get_library(ctx)
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
    lib = get_library(ctx)
    p = require_paper(lib, paper_id)
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
    lib = get_library(ctx)
    entries = [paper_to_bibtex(require_paper(lib, pid)) for pid in paper_ids]
    content = "\n\n".join(entries)
    if out_path:
        Path(out_path).write_text(content)
        n = len(entries)
        console.print(f"[green]✓[/green] Wrote {n} entr{'y' if n == 1 else 'ies'} to {out_path}")
    else:
        console.print(content)


@paper.group("watch")
def watch_group() -> None:
    """Manage arXiv paper watches."""


@watch_group.command("add")
@click.argument("query")
@click.option("--since", "since_year", default=None, type=int, metavar="YEAR", help="Min year.")
@click.pass_context
def watch_add(ctx: click.Context, query: str, since_year: int | None) -> None:
    """Add a new arXiv watch query."""
    lib = get_library(ctx)
    w = ArxivWatch(query=query, since_year=since_year)
    lib.add_watch(w)
    since_str = f" (since {since_year})" if since_year else ""
    console.print(f"[green]✓[/green] Watch '{query}'{since_str} added (id: {w.id})")


@watch_group.command("list")
@click.pass_context
def watch_list(ctx: click.Context) -> None:
    """List all watches."""
    lib = get_library(ctx)
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
    lib = get_library(ctx)
    if lib.remove_watch(watch_id):
        console.print(f"[green]✓[/green] Removed watch {watch_id}")
    else:
        console.print(f"[red]Watch {watch_id} not found.[/red]")
        raise SystemExit(1)


@watch_group.command("run")
@click.pass_context
def watch_run(ctx: click.Context) -> None:
    """Run all watches and show new papers."""
    lib = get_library(ctx)
    watches = lib.list_watches()
    if not watches:
        console.print("[dim]No watches configured.[/dim]")
        return
    for w in watches:
        _run_watch(lib, w)
