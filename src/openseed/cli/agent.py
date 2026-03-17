"""AI agent commands."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from openseed.agent.assistant import ResearchAssistant
from openseed.agent.reader import (
    PaperReader,
    auto_tag_paper,
    generate_experiment_code,
    search_papers_agent,
    synthesize_papers,
)
from openseed.auth import has_anthropic_auth
from openseed.models.paper import Tag
from openseed.services.arxiv import fetch_paper_metadata
from openseed.storage.library import PaperLibrary

console = Console()


def _get_library(ctx: click.Context) -> PaperLibrary:
    config = ctx.obj["config"]
    return PaperLibrary(config.library_dir)


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
    config = ctx.obj["config"]
    assistant = ResearchAssistant(model=config.default_model)
    answer = assistant.ask(question)
    console.print(Panel(Markdown(answer), title="Answer", border_style="blue"))


@agent.command()
@click.argument("paper_id")
@click.option("--cn", is_flag=True, help="Output summary in Chinese.")
@click.pass_context
def summarize(ctx: click.Context, paper_id: str, cn: bool) -> None:
    """Summarize a paper using AI."""
    _require_auth()
    lib = _get_library(ctx)
    p = lib.get_paper(paper_id)
    if not p:
        console.print(f"[red]Paper {paper_id} not found.[/red]")
        raise SystemExit(1)

    config = ctx.obj["config"]
    reader = PaperReader(model=config.default_model)
    summary = reader.summarize_paper(p.abstract or p.title, cn=cn)

    p.summary = summary
    lib.update_paper(p)
    md_path = lib.save_summary(p)

    console.print(Panel(Markdown(summary), title=f"Summary: {p.title}", border_style="green"))
    console.print(f"[dim]Saved → {md_path}[/dim]")


@agent.command()
@click.argument("query")
@click.pass_context
def search(ctx: click.Context, query: str) -> None:
    """Intelligently search for papers using AI."""
    _require_auth()
    config = ctx.obj["config"]
    with console.status(f"[cyan]Searching for '{query}'…[/cyan]"):
        result = search_papers_agent(query, model=config.default_model)
    console.print(Panel(Markdown(result), title=f"Search: {query}", border_style="cyan"))


@agent.command()
@click.argument("paper_id")
@click.pass_context
def review(ctx: click.Context, paper_id: str) -> None:
    """Generate an AI review of a paper."""
    _require_auth()
    lib = _get_library(ctx)
    p = lib.get_paper(paper_id)
    if not p:
        console.print(f"[red]Paper {paper_id} not found.[/red]")
        raise SystemExit(1)

    config = ctx.obj["config"]
    assistant = ResearchAssistant(model=config.default_model)
    review_text = assistant.review_paper(p)

    console.print(Panel(Markdown(review_text), title=f"Review: {p.title}", border_style="yellow"))


def _extract_arxiv_ids(text: str) -> list[str]:
    found = re.findall(r"\b(\d{4}\.\d{4,5})\b", text)
    return list(dict.fromkeys(found))  # deduplicate, preserve order


def _parse_md_table(text: str) -> dict[str, dict]:
    """Parse a markdown table and return {arxiv_id: {title, authors, year, citations}}."""
    info: dict[str, dict] = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        arxiv_id = re.search(r"\b(\d{4}\.\d{4,5})\b", cells[0])
        if not arxiv_id:
            continue
        aid = arxiv_id.group(1)
        info[aid] = {
            "title": cells[1] if len(cells) > 1 else "",
            "authors": cells[2] if len(cells) > 2 else "",
            "year": cells[3] if len(cells) > 3 else "",
            "citations": cells[4] if len(cells) > 4 else "",
        }
    return info


def _display_id_table(arxiv_ids: list[str], info: dict[str, dict]) -> None:
    from rich.table import Table

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
                for i in range(int(lo) - 1, int(hi)):
                    if 0 <= i < count:
                        indices.append(i)
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


def _analyze_and_save(
    paper, model: str, lib: PaperLibrary, progress=None, task_id=None, cn: bool = False
) -> None:
    """Run summary + auto-tag pipeline on a paper and save it."""
    text = paper.abstract or paper.title

    def _step(msg: str) -> None:
        if progress and task_id is not None:
            progress.update(task_id, description=f"[cyan]{msg}[/cyan]")

    _step(f"Summarizing '{paper.title[:35]}…'")
    paper.summary = PaperReader(model=model).summarize_paper(text, cn=cn, on_step=_step)

    _step(f"Tagging '{paper.title[:35]}…'")
    paper.tags = [Tag(name=t) for t in auto_tag_paper(text, model, on_step=_step)]

    added = lib.add_paper(paper)
    if not added:
        console.print(f"[yellow]Skipped (already exists)[/yellow] {paper.title}")
        return
    md_path = lib.save_summary(paper)
    tags_str = ", ".join(t.name for t in paper.tags)
    console.print(f"[green]✓[/green] [bold]{paper.title}[/bold]")
    console.print(f"   Tags: [yellow]{tags_str}[/yellow]  •  id: {paper.id}")
    console.print(Panel(Markdown(paper.summary), border_style="green"))
    console.print(f"[dim]Saved → {md_path}[/dim]")


@agent.command()
@click.argument("query")
@click.option("--count", default=20, show_default=True, help="Number of papers to search for.")
@click.pass_context
def pipeline(ctx: click.Context, query: str, count: int) -> None:
    """Search → select → auto-analyze and save (with citation counts)."""
    _require_auth()
    config = ctx.obj["config"]
    lib = _get_library(ctx)

    with console.status("[cyan]Searching…[/cyan]") as status:

        def _on_search_step(label: str) -> None:
            status.update(f"[cyan]{label}[/cyan]")

        md_result = search_papers_agent(
            query, model=config.default_model, count=count, on_step=_on_search_step
        )
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

    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

    console.print()
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

        for arxiv_id in selected_ids:
            progress.update(
                paper_task, description=f"[cyan]Fetching {arxiv_id}…[/cyan]", completed=0
            )
            try:
                paper = asyncio.run(fetch_paper_metadata(arxiv_id))
            except Exception as exc:
                console.print(f"[red]Failed to fetch {arxiv_id}: {exc}[/red]")
                progress.advance(overall)
                continue
            progress.advance(paper_task)
            _analyze_and_save(
                paper, config.default_model, lib, progress=progress, task_id=paper_task
            )
            progress.advance(paper_task)
            progress.advance(overall)


@agent.command()
@click.argument("paper_ids", nargs=-1, required=True)
@click.pass_context
def synthesize(ctx: click.Context, paper_ids: tuple[str, ...]) -> None:
    """Compare and synthesize findings across multiple papers."""
    _require_auth()
    lib = _get_library(ctx)
    texts = []
    for pid in paper_ids:
        p = lib.get_paper(pid)
        if not p:
            console.print(f"[red]Paper {pid} not found.[/red]")
            raise SystemExit(1)
        texts.append(f"Title: {p.title}\n\n{p.summary or p.abstract or p.title}")
    config = ctx.obj["config"]
    with console.status("[cyan]Synthesizing…[/cyan]"):
        result = synthesize_papers(texts, config.default_model)
    console.print(Panel(Markdown(result), title="Synthesis", border_style="cyan"))


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
    lib = _get_library(ctx)
    p = lib.get_paper(paper_id)
    if not p:
        console.print(f"[red]Paper {paper_id} not found.[/red]")
        raise SystemExit(1)
    config = ctx.obj["config"]
    dest = _resolve_codegen_path(config, p.id, out_path)
    text = f"Title: {p.title}\n\n{p.abstract or ''}\n\n{p.summary or ''}"
    with console.status("[cyan]Generating experiment code…[/cyan]"):
        code = generate_experiment_code(text, config.default_model)
    _save_codegen(lib, p, dest, code)
