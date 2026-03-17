"""AI agent commands."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from openseed.agent.assistant import ResearchAssistant
from openseed.agent.reader import (
    PaperReader,
    auto_tag_paper,
    extract_paper_visuals,
    generate_experiment_code,
    search_papers_agent,
    synthesize_papers,
)
from openseed.auth import has_anthropic_auth
from openseed.cli._helpers import (
    get_config,
    get_library,
    render_paper_visuals,
    require_paper,
)
from openseed.models.paper import Paper, Tag
from openseed.services.arxiv import fetch_paper_metadata
from openseed.storage.library import PaperLibrary

console = Console()


def _require_auth() -> None:
    ok, _ = has_anthropic_auth()
    if not ok:
        console.print("[red]No auth configured.[/red] Try one of:")
        console.print("  • [bold]export ANTHROPIC_API_KEY=sk-...[/bold]")
        console.print("  • Get an API key at [bold]console.anthropic.com[/bold]")
        console.print("  • [bold]openseed setup[/bold]  (runs claude setup-token)")
        raise SystemExit(1)


@click.group()
@click.pass_context
def agent(ctx: click.Context) -> None:
    """AI-powered research assistant."""
    ctx.ensure_object(dict)


@agent.command()
@click.argument("question")
@click.pass_context
def ask(ctx: click.Context, question: str) -> None:
    """Ask a research question."""
    _require_auth()
    config = get_config(ctx)
    answer = ResearchAssistant(model=config.default_model).ask(question)
    console.print(Panel(Markdown(answer), title="Answer", border_style="blue"))


@agent.command()
@click.argument("paper_id")
@click.option("--cn", is_flag=True, help="Output summary in Chinese.")
@click.pass_context
def summarize(ctx: click.Context, paper_id: str, cn: bool) -> None:
    """Summarize a paper using AI."""
    _require_auth()
    lib = get_library(ctx)
    p = require_paper(lib, paper_id)
    config = get_config(ctx)
    reader = PaperReader(model=config.default_model)
    p.summary = reader.summarize_paper(p.abstract or p.title, cn=cn)
    lib.update_paper(p)
    md_path = lib.save_summary(p)
    console.print(Panel(Markdown(p.summary), title=f"Summary: {p.title}", border_style="green"))
    console.print(f"[dim]Saved → {md_path}[/dim]")
    with console.status("[cyan]Extracting visuals…[/cyan]"):
        visuals = extract_paper_visuals(p.abstract or p.title, config.default_model)
    render_paper_visuals(visuals, console)


@agent.command()
@click.argument("paper_id")
@click.pass_context
def review(ctx: click.Context, paper_id: str) -> None:
    """Generate an AI review of a paper."""
    _require_auth()
    lib = get_library(ctx)
    p = require_paper(lib, paper_id)
    config = get_config(ctx)
    review_text = ResearchAssistant(model=config.default_model).review_paper(p)
    console.print(Panel(Markdown(review_text), title=f"Review: {p.title}", border_style="yellow"))


def _extract_arxiv_ids(text: str) -> list[str]:
    found = re.findall(r"\b(\d{4}\.\d{4,5})\b", text)
    return list(dict.fromkeys(found))


def _parse_md_table(text: str) -> dict[str, dict]:
    """Parse markdown table → {arxiv_id: {title, authors, year, citations}}."""
    info: dict[str, dict] = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        m = re.search(r"\b(\d{4}\.\d{4,5})\b", cells[0])
        if not m:
            continue
        aid = m.group(1)
        info[aid] = {
            "title": cells[1] if len(cells) > 1 else "",
            "authors": cells[2] if len(cells) > 2 else "",
            "year": cells[3] if len(cells) > 3 else "",
        }
    return info


def _display_id_table(arxiv_ids: list[str], info: dict[str, dict]) -> None:
    table = Table(title="Papers found", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("ArXiv ID", style="cyan", width=13)
    table.add_column("Title", style="bold", max_width=48)
    table.add_column("Authors", style="dim", max_width=22)
    table.add_column("Year", justify="right", width=6)
    for i, aid in enumerate(arxiv_ids, 1):
        meta = info.get(aid, {})
        table.add_row(
            str(i), aid, meta.get("title", ""), meta.get("authors", ""), meta.get("year", "")
        )
    console.print(table)


def _parse_selection(raw: str, count: int) -> list[int]:
    if raw.strip().lower() == "all":
        return list(range(count))
    indices = []
    for part in raw.split(","):
        part = part.strip()
        if "-" in part:
            lo, _, hi = part.partition("-")
            try:
                indices.extend(i for i in range(int(lo) - 1, int(hi)) if 0 <= i < count)
            except ValueError:
                pass
        else:
            try:
                idx = int(part) - 1
                if 0 <= idx < count:
                    indices.append(idx)
            except ValueError:
                pass
    return indices


def _make_step(progress, task_id) -> Callable:
    def _step(msg: str) -> None:
        if progress and task_id is not None:
            progress.update(task_id, description=f"[cyan]{msg}[/cyan]")

    return _step


def _do_summarize(paper: Paper, model: str, step: Callable, cn: bool) -> None:
    paper.summary = PaperReader(model=model).summarize_paper(
        paper.abstract or paper.title, cn=cn, on_step=step
    )


def _do_tag(paper: Paper, model: str, step: Callable) -> None:
    tags = auto_tag_paper(paper.abstract or paper.title, model, on_step=step)
    paper.tags = [Tag(name=t) for t in tags]


def _analyze_and_save(
    paper: Paper, model: str, lib: PaperLibrary, progress=None, task_id=None, cn: bool = False
) -> None:
    """Summarize + tag + save a paper."""
    step = _make_step(progress, task_id)
    step(f"Summarizing '{paper.title[:35]}…'")
    _do_summarize(paper, model, step, cn)
    step(f"Tagging '{paper.title[:35]}…'")
    _do_tag(paper, model, step)
    if not lib.add_paper(paper):
        console.print(f"[yellow]Skipped (already exists)[/yellow] {paper.title}")
        return
    md_path = lib.save_summary(paper)
    tags_str = ", ".join(t.name for t in paper.tags)
    console.print(f"[green]✓[/green] [bold]{paper.title}[/bold]")
    console.print(f"   Tags: [yellow]{tags_str}[/yellow]  •  id: {paper.id}")
    console.print(Panel(Markdown(paper.summary or ""), border_style="green"))
    console.print(f"[dim]Saved → {md_path}[/dim]")
    step("Extracting visuals…")
    render_paper_visuals(extract_paper_visuals(paper.abstract or paper.title, model), console)


def _search_with_status(query: str, model: str, count: int) -> str:
    with console.status("[cyan]Searching…[/cyan]") as status:

        def _on_step(label: str) -> None:
            status.update(f"[cyan]{label}[/cyan]")

        return search_papers_agent(query, model=model, count=count, on_step=_on_step)


async def _fetch_papers(arxiv_ids: list[str]) -> list[tuple[str, Paper | Exception]]:
    results = await asyncio.gather(
        *[fetch_paper_metadata(aid) for aid in arxiv_ids], return_exceptions=True
    )
    return list(zip(arxiv_ids, results))


def _pipeline_loop(
    selected_ids: list[str], model: str, lib: PaperLibrary, cn: bool = False
) -> None:
    fetched = asyncio.run(_fetch_papers(selected_ids))
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[dim]{task.completed}/{task.total}[/dim]"),
        console=console,
        transient=False,
    ) as progress:
        overall = progress.add_task("[bold]Pipeline[/bold]", total=len(selected_ids))
        paper_task = progress.add_task("", total=2)
        for arxiv_id, paper in fetched:
            if isinstance(paper, Exception):
                console.print(f"[red]Failed to fetch {arxiv_id}: {paper}[/red]")
                progress.advance(overall)
                continue
            _analyze_and_save(paper, model, lib, progress=progress, task_id=paper_task, cn=cn)
            progress.advance(paper_task)
            progress.advance(overall)


@agent.command()
@click.argument("query")
@click.option("--count", default=20, show_default=True, help="Number of papers to search for.")
@click.option("--cn", is_flag=True, help="Summarize in Chinese.")
@click.pass_context
def pipeline(ctx: click.Context, query: str, count: int, cn: bool) -> None:
    """Search → select → auto-analyze and save (with citation counts)."""
    _require_auth()
    config = get_config(ctx)
    lib = get_library(ctx)
    md_result = _search_with_status(query, config.default_model, count)
    console.print(Panel(Markdown(md_result), title=f"Search: {query}", border_style="cyan"))
    arxiv_ids = _extract_arxiv_ids(md_result)
    if not arxiv_ids:
        console.print("[yellow]No ArXiv IDs found. Try a more specific query.[/yellow]")
        return
    _display_id_table(arxiv_ids, _parse_md_table(md_result))
    raw = click.prompt("\nSelect papers to analyze (e.g. 1,3 or 1-10 or all, q to quit)")
    if raw.strip().lower() == "q":
        return
    selected_ids = [arxiv_ids[i] for i in _parse_selection(raw, len(arxiv_ids))]
    if not selected_ids:
        console.print("[yellow]Invalid selection.[/yellow]")
        return
    console.print()
    _pipeline_loop(selected_ids, config.default_model, lib, cn=cn)


def _paper_year(p: Paper) -> int | None:
    m = re.match(r"^(\d{2})\d{2}\.", p.arxiv_id or "")
    return (2000 + int(m.group(1))) if m else None


def _synthesis_chart(papers: list[Paper]) -> Panel:
    """Render a year-timeline + tags comparison for a set of papers."""
    year_pairs = sorted((((_paper_year(p) or 0), p) for p in papers), key=lambda x: x[0])
    years = [y for y, _ in year_pairs if y]
    min_y, max_y = (min(years), max(years)) if years else (2020, 2024)
    span = max(max_y - min_y, 1)

    table = Table(show_header=True, box=None, padding=(0, 1))
    table.add_column("Year", style="cyan", width=6)
    table.add_column("Timeline", width=26, no_wrap=True)
    table.add_column("Title", style="bold", max_width=36)
    table.add_column("Tags", style="yellow", max_width=28)

    for year, p in year_pairs:
        if year:
            pos = round((year - min_y) / span * 23)
            line = "─" * pos + "[bold cyan]◆[/bold cyan]" + "─" * (23 - pos)
        else:
            line = "─" * 24
        tags = " ".join(t.name for t in p.tags[:4]) or "—"
        table.add_row(str(year or "?"), line, p.title[:34], tags)

    return Panel(table, title="Paper Timeline", border_style="blue")


@agent.command()
@click.argument("paper_ids", nargs=-1, required=True)
@click.pass_context
def synthesize(ctx: click.Context, paper_ids: tuple[str, ...]) -> None:
    """Compare and synthesize findings across multiple papers."""
    _require_auth()
    lib = get_library(ctx)
    papers = [require_paper(lib, pid) for pid in paper_ids]
    texts = [f"Title: {p.title}\n\n{p.summary or p.abstract or p.title}" for p in papers]
    config = get_config(ctx)
    with console.status("[cyan]Synthesizing…[/cyan]"):
        result = synthesize_papers(texts, config.default_model)
    console.print(Panel(Markdown(result), title="Synthesis", border_style="cyan"))
    console.print(_synthesis_chart(papers))
    md_path = lib.save_synthesis(list(paper_ids), result)
    console.print(f"[dim]Saved → {md_path}[/dim]")


def _resolve_codegen_path(config, paper_id: str, out_path: str | None) -> Path:
    if out_path:
        return Path(out_path)
    experiments_dir = Path(config.config_dir) / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    return experiments_dir / f"{paper_id}.py"


def _save_codegen(lib, p, dest: Path, code: str) -> None:
    dest.write_text(code)
    p.experiment_path = str(dest)
    lib.update_paper(p)
    console.print(
        Panel(
            Markdown(f"```python\n{code}\n```"),
            title=f"Experiment: {p.title}",
            border_style="magenta",
        )
    )
    console.print(f"[green]✓[/green] Saved to [bold]{dest}[/bold]")


@agent.command()
@click.argument("paper_id")
@click.option("--output", "out_path", default=None, help="Output .py path.")
@click.pass_context
def codegen(ctx: click.Context, paper_id: str, out_path: str | None) -> None:
    """Generate experiment code for a paper and save to file."""
    _require_auth()
    lib = get_library(ctx)
    p = require_paper(lib, paper_id)
    config = get_config(ctx)
    dest = _resolve_codegen_path(config, p.id, out_path)
    text = f"Title: {p.title}\n\n{p.abstract or ''}\n\n{p.summary or ''}"
    with console.status("[cyan]Generating experiment code…[/cyan]"):
        code = generate_experiment_code(text, config.default_model)
    _save_codegen(lib, p, dest, code)
