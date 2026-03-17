"""AI agent commands."""

from __future__ import annotations

import asyncio
import re

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

    console.print(Panel(Markdown(summary), title=f"Summary: {p.title}", border_style="green"))


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


def _display_id_table(arxiv_ids: list[str]) -> None:
    from rich.table import Table

    table = Table(title="找到的论文", show_lines=False)
    table.add_column("#", style="dim", width=4)
    table.add_column("ArXiv ID", style="cyan")
    for i, aid in enumerate(arxiv_ids, 1):
        table.add_row(str(i), aid)
    console.print(table)


def _parse_selection(raw: str, count: int) -> list[int]:
    if raw.strip().lower() in ("all", "全部"):
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


def _analyze_and_save(paper, model: str, lib: PaperLibrary, cn: bool = False) -> None:
    """Run summary + auto-tag pipeline on a paper and save it."""
    text = paper.abstract or paper.title
    with console.status(f"[cyan]Summarizing '{paper.title[:40]}…'[/cyan]"):
        paper.summary = PaperReader(model=model).summarize_paper(text, cn=cn)
    with console.status("[cyan]Generating tags…[/cyan]"):
        paper.tags = [Tag(name=t) for t in auto_tag_paper(text, model)]
    added = lib.add_paper(paper)
    if not added:
        console.print(f"[yellow]跳过（已存在）[/yellow] {paper.title}")
        return
    tags_str = ", ".join(t.name for t in paper.tags)
    console.print(f"[green]✓[/green] [bold]{paper.title}[/bold]")
    console.print(f"   Tags: [yellow]{tags_str}[/yellow]  •  id: {paper.id}")
    console.print(Panel(Markdown(paper.summary), border_style="green"))


@agent.command()
@click.argument("query")
@click.option("--count", default=20, show_default=True, help="Number of papers to search for.")
@click.pass_context
def pipeline(ctx: click.Context, query: str, count: int) -> None:
    """AI搜索 → 选择 → 自动分析入库（含引用数、中文摘要）。"""
    _require_auth()
    config = ctx.obj["config"]
    lib = _get_library(ctx)

    with console.status(f"[cyan]AI 搜索 '{query}'（目标 {count} 篇）…[/cyan]"):
        md_result = search_papers_agent(query, model=config.default_model, count=count)
    console.print(Panel(Markdown(md_result), title=f"搜索：{query}", border_style="cyan"))

    arxiv_ids = _extract_arxiv_ids(md_result)
    if not arxiv_ids:
        console.print("[yellow]未找到 ArXiv ID，请尝试更具体的查询。[/yellow]")
        return

    _display_id_table(arxiv_ids)
    raw = click.prompt("\n选择要分析的论文（如 1,3 或 1-10 或 all，q 退出）")
    if raw.strip().lower() == "q":
        return

    selected_ids = [arxiv_ids[i] for i in _parse_selection(raw, len(arxiv_ids))]
    if not selected_ids:
        console.print("[yellow]无效选择。[/yellow]")
        return

    console.print(f"\n[bold]正在抓取并分析 {len(selected_ids)} 篇论文…[/bold]\n")
    for arxiv_id in selected_ids:
        with console.status(f"[cyan]抓取 {arxiv_id}…[/cyan]"):
            try:
                paper = asyncio.run(fetch_paper_metadata(arxiv_id))
            except Exception as exc:
                console.print(f"[red]无法获取 {arxiv_id}: {exc}[/red]")
                continue
        _analyze_and_save(paper, config.default_model, lib)


@agent.command()
@click.argument("paper_id")
@click.pass_context
def codegen(ctx: click.Context, paper_id: str) -> None:
    """Generate experiment code for a paper."""
    _require_auth()
    lib = _get_library(ctx)
    p = lib.get_paper(paper_id)
    if not p:
        console.print(f"[red]Paper {paper_id} not found.[/red]")
        raise SystemExit(1)

    config = ctx.obj["config"]
    text = f"Title: {p.title}\n\n{p.abstract or ''}\n\n{p.summary or ''}"
    with console.status("[cyan]Generating experiment code…[/cyan]"):
        code = generate_experiment_code(text, config.default_model)
    console.print(
        Panel(
            Markdown(f"```python\n{code}\n```"),
            title=f"Experiment: {p.title}",
            border_style="magenta",
        )
    )
