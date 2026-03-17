"""Research session commands."""

from __future__ import annotations

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from openseed.agent.autoresearch import AutoResearcher
from openseed.auth import has_anthropic_auth
from openseed.cli._helpers import get_config, get_library
from openseed.monitor import get_usage_summary, record_research_lesson

console = Console()


def _require_auth() -> None:
    ok, _ = has_anthropic_auth()
    if not ok:
        console.print("[red]No auth configured.[/red] Run [bold]openseed setup[/bold].")
        raise SystemExit(1)


@click.group()
@click.pass_context
def research(ctx: click.Context) -> None:
    """Autonomous multi-paper research sessions."""
    ctx.ensure_object(dict)


@research.command("run")
@click.argument("topic")
@click.option("--count", default=15, show_default=True, help="Papers to discover.")
@click.option("--depth", default=2, show_default=True, help="Query expansion rounds.")
@click.option(
    "--since", "since_year", default=None, type=int, help="Only papers from this year or later."
)
@click.pass_context
def run_research(
    ctx: click.Context, topic: str, count: int, depth: int, since_year: int | None
) -> None:
    """Run autonomous research: discover → analyze → synthesize → report."""
    _require_auth()
    lib = get_library(ctx)
    config = get_config(ctx)
    researcher = AutoResearcher(model=config.default_model, lib=lib)

    with console.status(f"[cyan]Researching '{topic}'…[/cyan]") as status:

        def _on_step(msg: str) -> None:
            status.update(f"[cyan]{msg}[/cyan]")

        session = researcher.run(
            topic, count=count, depth=depth, since_year=since_year, on_step=_on_step
        )

    lib.add_research_session(session)
    _render_session(session, lib)
    record_research_lesson(topic, f"Analyzed {len(session.paper_ids)} papers on: {topic}")
    _print_usage()


@research.command("list")
@click.pass_context
def list_sessions(ctx: click.Context) -> None:
    """List past research sessions."""
    lib = get_library(ctx)
    sessions = lib.list_research_sessions()
    if not sessions:
        console.print("[yellow]No research sessions yet.[/yellow]")
        return
    _render_sessions_table(sessions)


@research.command("show")
@click.argument("session_id")
@click.pass_context
def show_session(ctx: click.Context, session_id: str) -> None:
    """Show report and details of a research session."""
    lib = get_library(ctx)
    session = lib.get_research_session(session_id)
    if not session:
        console.print(f"[red]Session not found: {session_id}[/red]")
        raise SystemExit(1)
    _render_session(session, lib)


def _render_session(session, lib) -> None:
    papers = [lib.get_paper(pid) for pid in session.paper_ids if lib.get_paper(pid)]
    console.print(
        Panel(
            f"[bold]{session.topic}[/bold]\n"
            f"[dim]ID: {session.id}  •  Papers: {len(papers)}  •  "
            f"{session.created_at.strftime('%Y-%m-%d %H:%M')}[/dim]",
            title="Research Session",
            border_style="cyan",
        )
    )
    if session.report:
        console.print(Panel(Markdown(session.report), title="Report", border_style="green"))
    if session.synthesis:
        console.print(Panel(Markdown(session.synthesis), title="Synthesis", border_style="blue"))


def _render_sessions_table(sessions) -> None:
    table = Table(title="Research Sessions", show_lines=True)
    table.add_column("ID", style="cyan", width=13)
    table.add_column("Topic", style="bold", max_width=50)
    table.add_column("Papers", justify="right", width=7)
    table.add_column("Created", width=16)
    for s in sorted(sessions, key=lambda x: x.created_at, reverse=True):
        table.add_row(
            s.id,
            s.topic,
            str(len(s.paper_ids)),
            s.created_at.strftime("%Y-%m-%d %H:%M"),
        )
    console.print(table)


def _print_usage() -> None:
    usage = get_usage_summary()
    if usage:
        console.print(f"[dim]openMax usage: {usage}[/dim]")
