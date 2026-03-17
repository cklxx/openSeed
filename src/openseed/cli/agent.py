"""AI agent commands."""

from __future__ import annotations

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from openseed.agent.assistant import ResearchAssistant
from openseed.agent.reader import PaperReader
from openseed.storage.library import PaperLibrary

console = Console()


def _get_library(ctx: click.Context) -> PaperLibrary:
    config = ctx.obj["config"]
    return PaperLibrary(config.library_dir)


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
    config = ctx.obj["config"]
    assistant = ResearchAssistant(model=config.default_model)
    answer = assistant.ask(question)
    console.print(Panel(Markdown(answer), title="Answer", border_style="blue"))


@agent.command()
@click.argument("paper_id")
@click.pass_context
def summarize(ctx: click.Context, paper_id: str) -> None:
    """Summarize a paper using AI."""
    lib = _get_library(ctx)
    p = lib.get_paper(paper_id)
    if not p:
        console.print(f"[red]Paper {paper_id} not found.[/red]")
        raise SystemExit(1)

    config = ctx.obj["config"]
    reader = PaperReader(model=config.default_model)
    text = p.abstract or p.title
    summary = reader.summarize_paper(text)

    p.summary = summary
    lib.update_paper(p)

    console.print(Panel(Markdown(summary), title=f"Summary: {p.title}", border_style="green"))


@agent.command()
@click.argument("paper_id")
@click.pass_context
def review(ctx: click.Context, paper_id: str) -> None:
    """Generate an AI review of a paper."""
    lib = _get_library(ctx)
    p = lib.get_paper(paper_id)
    if not p:
        console.print(f"[red]Paper {paper_id} not found.[/red]")
        raise SystemExit(1)

    config = ctx.obj["config"]
    assistant = ResearchAssistant(model=config.default_model)
    review_text = assistant.review_paper(p)

    console.print(Panel(Markdown(review_text), title=f"Review: {p.title}", border_style="yellow"))
