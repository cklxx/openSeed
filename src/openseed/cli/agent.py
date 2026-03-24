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
from openseed.agent.discovery import search_papers_agent
from openseed.agent.reader import (
    PaperReader,
    auto_tag_paper,
    extract_paper_visuals,
    generate_experiment_code,
    synthesize_papers,
)
from openseed.agent.strategy import ResearchStrategy
from openseed.auth import has_anthropic_auth
from openseed.cli._helpers import (
    get_config,
    get_library,
    library_status_for_arxiv,
    render_paper_visuals,
    require_paper,
)
from openseed.models.paper import Paper, Tag
from openseed.services.arxiv import fetch_paper_metadata
from openseed.storage.library import PaperLibrary

console = Console()


def _show_similar_papers(paper: Paper, lib: PaperLibrary) -> None:
    """Show recommended similar papers via Semantic Scholar."""
    if not paper.arxiv_id:
        return
    from openseed.services.scholar import get_recommendations

    recs = get_recommendations(paper.arxiv_id, limit=5)
    recs = [r for r in recs if not lib.get_paper_by_arxiv(r["arxiv_id"])]
    if not recs:
        return
    table = Table(title="Similar papers (not in library)", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Title", style="bold", max_width=50)
    table.add_column("ArXiv ID", style="cyan", width=13)
    table.add_column("Year", width=6)
    for i, r in enumerate(recs[:5], 1):
        table.add_row(str(i), r["title"][:48], r["arxiv_id"], str(r.get("year", "")))
    console.print(table)


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
    lib = get_library(ctx)
    assistant = ResearchAssistant(library=lib, model=config.default_model)
    answer = assistant.ask(question)
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
    _show_similar_papers(p, lib)


@agent.command()
@click.argument("paper_id")
@click.pass_context
def review(ctx: click.Context, paper_id: str) -> None:
    """Generate an AI review of a paper."""
    _require_auth()
    lib = get_library(ctx)
    p = require_paper(lib, paper_id)
    config = get_config(ctx)
    review_text = ResearchAssistant(library=lib, model=config.default_model).review_paper(p)
    console.print(Panel(Markdown(review_text), title=f"Review: {p.title}", border_style="yellow"))


def _chat_repl(assistant: ResearchAssistant, debug: bool) -> None:
    """Interactive REPL loop for multi-turn research chat."""
    console.print("[dim]Type /quit to exit, /clear to reset history, /debug to toggle debug.[/dim]")
    while True:
        try:
            question = console.input("[bold cyan]> [/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            break
        if not question.strip():
            continue
        if question.strip() == "/quit":
            break
        if question.strip() == "/clear":
            assistant.clear_history()
            console.print("[dim]History cleared.[/dim]")
            continue
        if question.strip() == "/debug":
            debug = not debug
            console.print(f"[dim]Debug {'on' if debug else 'off'}.[/dim]")
            continue
        if debug:
            info = assistant.get_debug_info()
            if info:
                console.print(f"[dim]Context: {info}[/dim]")
        for chunk in assistant.stream(question):
            console.print(chunk, end="")
        console.print()


@agent.command()
@click.option("--debug", is_flag=True, help="Show context retrieval details.")
@click.pass_context
def chat(ctx: click.Context, debug: bool) -> None:
    """Interactive multi-turn research conversation."""
    _require_auth()
    config = get_config(ctx)
    lib = get_library(ctx)
    assistant = ResearchAssistant(library=lib, model=config.default_model)
    _chat_repl(assistant, debug)


@agent.command()
@click.pass_context
def gaps(ctx: click.Context) -> None:
    """Analyze research gaps in your library."""
    _require_auth()
    config = get_config(ctx)
    lib = get_library(ctx)
    with console.status("[cyan]Analyzing gaps…[/cyan]"):
        results = ResearchStrategy(lib, model=config.default_model).analyze_gaps()
    if not results:
        console.print("[yellow]No papers in library to analyze.[/yellow]")
        return
    table = Table(title="Research Gaps", show_lines=True)
    table.add_column("Cluster", style="bold", max_width=20)
    table.add_column("Papers", justify="right", width=8)
    table.add_column("Gap", max_width=40)
    table.add_column("Suggested Queries", max_width=30)
    table.add_column("Confidence", justify="right", width=10)
    for gap in results:
        queries = ", ".join(gap.suggested_queries[:3])
        table.add_row(
            gap.cluster_name,
            str(gap.paper_count),
            gap.gap_description,
            queries,
            f"{gap.confidence:.0%}",
        )
    console.print(table)


@agent.command("reading-order")
@click.argument("topic")
@click.pass_context
def reading_order(ctx: click.Context, topic: str) -> None:
    """Suggest reading order for papers on a topic."""
    _require_auth()
    config = get_config(ctx)
    lib = get_library(ctx)
    with console.status("[cyan]Analyzing reading order…[/cyan]"):
        recs = ResearchStrategy(lib, model=config.default_model).suggest_reading_order(topic)
    if not recs:
        console.print(f"[yellow]No papers found for topic '{topic}'.[/yellow]")
        return
    for rec in recs:
        console.print(f"  [bold cyan]{rec.priority}.[/bold cyan] [bold]{rec.paper.title}[/bold]")
        console.print(f"     [dim]{rec.reason}[/dim]")


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


def _display_id_table(arxiv_ids: list[str], info: dict[str, dict], lib: PaperLibrary) -> None:
    table = Table(title="Papers found", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("ArXiv ID", style="cyan", width=13)
    table.add_column("Title", style="bold", max_width=46)
    table.add_column("Authors", style="dim", max_width=20)
    table.add_column("Year", justify="right", width=6)
    table.add_column("Library", width=12)
    for i, aid in enumerate(arxiv_ids, 1):
        meta = info.get(aid, {})
        table.add_row(
            str(i),
            aid,
            meta.get("title", ""),
            meta.get("authors", ""),
            meta.get("year", ""),
            library_status_for_arxiv(lib, aid),
        )
    console.print(table)


def _parse_range(token: str, count: int) -> list[int]:
    """Parse '3-7' → [2,3,4,5,6] (0-indexed, clamped to count)."""
    lo, _, hi = token.partition("-")
    try:
        return [i for i in range(int(lo) - 1, int(hi)) if 0 <= i < count]
    except ValueError:
        return []


def _parse_single(token: str, count: int) -> list[int]:
    """Parse '3' → [2] (0-indexed, clamped to count)."""
    try:
        idx = int(token) - 1
        return [idx] if 0 <= idx < count else []
    except ValueError:
        return []


def _parse_selection(raw: str, count: int) -> list[int]:
    if raw.strip().lower() == "all":
        return list(range(count))
    indices: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        parser = _parse_range if "-" in token else _parse_single
        indices.extend(parser(token, count))
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


def _make_pipeline_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[dim]{task.completed}/{task.total}[/dim]"),
        console=console,
        transient=False,
    )


def _process_fetched_paper(
    arxiv_id: str,
    paper: Paper | Exception,
    model: str,
    lib: PaperLibrary,
    progress,
    overall,
    paper_task,
    cn: bool,
) -> None:
    """Process a single fetched paper in the pipeline."""
    if isinstance(paper, Exception):
        console.print(f"[red]Failed to fetch {arxiv_id}: {paper}[/red]")
        progress.advance(overall)
        return
    existing = lib.get_paper_by_arxiv(arxiv_id)
    if existing and existing.summary:
        console.print(
            f"[dim]Skipping {arxiv_id} — already analyzed:[/dim] [bold]{existing.title}[/bold]"
        )
        progress.advance(overall)
        return
    _analyze_and_save(paper, model, lib, progress=progress, task_id=paper_task, cn=cn)
    progress.advance(paper_task)
    progress.advance(overall)


def _pipeline_loop(
    selected_ids: list[str], model: str, lib: PaperLibrary, cn: bool = False
) -> None:
    fetched = asyncio.run(_fetch_papers(selected_ids))
    with _make_pipeline_progress() as progress:
        overall = progress.add_task("[bold]Pipeline[/bold]", total=len(selected_ids))
        paper_task = progress.add_task("", total=2)
        for arxiv_id, paper in fetched:
            _process_fetched_paper(arxiv_id, paper, model, lib, progress, overall, paper_task, cn)


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
    _display_id_table(arxiv_ids, _parse_md_table(md_result), lib)
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


_TIMELINE_WIDTH = 23


def _timeline_bar(year: int, min_y: int, span: int) -> str:
    if not year:
        return "─" * (_TIMELINE_WIDTH + 1)
    pos = round((year - min_y) / span * _TIMELINE_WIDTH)
    return "─" * pos + "[bold cyan]◆[/bold cyan]" + "─" * (_TIMELINE_WIDTH - pos)


def _synthesis_chart(papers: list[Paper]) -> Panel:
    """Render a year-timeline + tags comparison for a set of papers."""
    year_pairs = sorted((((_paper_year(p) or 0), p) for p in papers), key=lambda x: x[0])
    years = [y for y, _ in year_pairs if y]
    min_y = min(years) if years else 2020
    span = max((max(years) if years else 2024) - min_y, 1)
    table = Table(show_header=True, box=None, padding=(0, 1))
    table.add_column("Year", style="cyan", width=6)
    table.add_column("Timeline", width=26, no_wrap=True)
    table.add_column("Title", style="bold", max_width=36)
    table.add_column("Tags", style="yellow", max_width=28)
    for year, p in year_pairs:
        tags = " ".join(t.name for t in p.tags[:4]) or "—"
        table.add_row(str(year or "?"), _timeline_bar(year, min_y, span), p.title[:34], tags)
    return Panel(table, title="Paper Timeline", border_style="blue")


@agent.command()
@click.argument("paper_ids", nargs=-1, required=True)
@click.pass_context
def synthesize(ctx: click.Context, paper_ids: tuple[str, ...]) -> None:
    """Compare and synthesize findings across multiple papers."""
    _require_auth()
    lib = get_library(ctx)
    papers = [require_paper(lib, pid) for pid in paper_ids]
    texts = [_paper_text(p) for p in papers]
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


def _paper_text(p: Paper) -> str:
    return f"Title: {p.title}\n\n{p.abstract or ''}\n\n{p.summary or p.title}"


@agent.command()
@click.argument("paper_id_a")
@click.argument("paper_id_b")
@click.pass_context
def compare(ctx: click.Context, paper_id_a: str, paper_id_b: str) -> None:
    """Compare two papers side-by-side."""
    _require_auth()
    if paper_id_a == paper_id_b:
        console.print("[red]Cannot compare a paper to itself.[/red]")
        raise SystemExit(1)
    lib = get_library(ctx)
    pa = require_paper(lib, paper_id_a)
    pb = require_paper(lib, paper_id_b)
    if not (pa.abstract or pa.summary):
        console.print(f"[red]Paper {pa.id} has no text.[/red] Run: agent summarize {pa.id}")
        raise SystemExit(1)
    if not (pb.abstract or pb.summary):
        console.print(f"[red]Paper {pb.id} has no text.[/red] Run: agent summarize {pb.id}")
        raise SystemExit(1)
    config = get_config(ctx)
    from openseed.agent.compare import compare_papers

    with console.status("[cyan]Comparing papers…[/cyan]"):
        result = compare_papers(
            _paper_text(pa), _paper_text(pb), pa.title, pb.title, config.default_model
        )
    console.print(
        Panel(Markdown(result), title=f"{pa.title[:30]} vs {pb.title[:30]}", border_style="yellow")
    )


def _load_synthesis(config, paper_ids: tuple[str, ...]) -> str:
    """Load synthesis markdown, raising SystemExit if not found."""
    slug = "_".join(sorted(paper_ids)[:4])
    synth_path = Path(config.config_dir) / "summaries" / f"synthesis_{slug}.md"
    if not synth_path.exists():
        console.print("[red]No synthesis found for these papers.[/red]")
        console.print(f"[dim]Run: openseed agent synthesize {' '.join(paper_ids)}[/dim]")
        raise SystemExit(1)
    return synth_path.read_text(encoding="utf-8")


def _write_latex_files(dest: Path, latex: str, bibtex: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "related_work.tex").write_text(latex, encoding="utf-8")
    (dest / "references.bib").write_text(bibtex, encoding="utf-8")
    console.print(f"[green]✓[/green] LaTeX → [bold]{dest / 'related_work.tex'}[/bold]")
    console.print(f"[green]✓[/green] BibTeX → [bold]{dest / 'references.bib'}[/bold]")


@agent.command("export-latex")
@click.argument("paper_ids", nargs=-1, required=True)
@click.option("--output", "out_dir", default=None, help="Output directory for .tex and .bib files.")
@click.pass_context
def export_latex(ctx: click.Context, paper_ids: tuple[str, ...], out_dir: str | None) -> None:
    """Export synthesis as LaTeX related-work section with BibTeX."""
    lib = get_library(ctx)
    config = get_config(ctx)
    papers = [require_paper(lib, pid) for pid in paper_ids]
    synthesis = _load_synthesis(config, paper_ids)
    from openseed.agent.latex import export_related_work

    latex, bibtex = export_related_work(synthesis, papers)
    _write_latex_files(Path(out_dir) if out_dir else Path("."), latex, bibtex)
