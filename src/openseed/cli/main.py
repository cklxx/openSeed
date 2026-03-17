"""OpenSeed CLI entry point."""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

from openseed import __version__
from openseed.auth import (
    append_export_to_rc,
    detect_rc_file,
    has_anthropic_auth,
)
from openseed.cli.agent import agent, ask
from openseed.cli.experiment import experiment
from openseed.cli.paper import paper
from openseed.cli.research import research
from openseed.config import ensure_dirs, load_config, save_config
from openseed.doctor import render_results, run_checks

console = Console()

_MODELS = [
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]


@click.group()
@click.version_option(version=__version__, prog_name="openseed")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """OpenSeed — AI-powered Research Workflow Management."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config()


@cli.command()
def init() -> None:
    """Initialize the OpenSeed library."""
    config = load_config()
    ensure_dirs(config)
    console.print(f"[green]✓[/green] Library initialized at {config.library_dir}")


@cli.command()
@click.option("--status", is_flag=True, default=False, help="Show current auth status")
@click.option(
    "--model",
    type=click.Choice(_MODELS),
    default=None,
    help="Set default Claude model",
)
def setup(status: bool, model: str | None) -> None:
    """Configure auth and model for OpenSeed."""
    if status:
        ok, detail = has_anthropic_auth()
        if ok:
            console.print(f"[green]Auth OK:[/green] {detail}")
        else:
            console.print("[yellow]No auth configured.[/yellow]")
            console.print(
                "Set [bold]ANTHROPIC_API_KEY[/bold] or run [bold]claude setup-token[/bold]."
            )
        return

    # ── Auth ─────────────────────────────────────────────────────────
    ok, detail = has_anthropic_auth()
    if ok:
        console.print(f"[green]✓[/green] Already authenticated: {detail}")
    else:
        console.print("[yellow]⚠[/yellow]  ANTHROPIC_API_KEY not set.\n")
        api_key = click.prompt(
            "Paste your Anthropic API key (sk-...)",
            hide_input=True,
            default="",
            show_default=False,
        )
        if api_key.startswith("sk-"):
            rc_file = detect_rc_file()
            if click.confirm(f"Save to {rc_file}?", default=True):
                append_export_to_rc("ANTHROPIC_API_KEY", api_key, rc_file)
                console.print(f"[green]✓[/green] Saved. Run: [bold]source {rc_file}[/bold]")
            else:
                console.print(f"[dim]Run manually: export ANTHROPIC_API_KEY={api_key}[/dim]")
            detail, ok = "ANTHROPIC_API_KEY", True
        else:
            console.print(
                "[yellow]Skipped.[/yellow] Agent features won't work without ANTHROPIC_API_KEY."
            )
            console.print("[dim]Note: CLAUDE_CODE_OAUTH_TOKEN only works for the claude CLI.[/dim]")

    # ── Model ────────────────────────────────────────────────────────
    config = load_config()
    if model:
        config.default_model = model
    else:
        console.print("\nAvailable models:")
        for i, m in enumerate(_MODELS, 1):
            marker = " [cyan](current)[/cyan]" if m == config.default_model else ""
            console.print(f"  {i}. {m}{marker}")
        choice = click.prompt(
            "Default model (number or name, Enter to keep)",
            default=config.default_model,
            show_default=True,
        )
        if choice.isdigit() and 1 <= int(choice) <= len(_MODELS):
            config.default_model = _MODELS[int(choice) - 1]
        elif choice in _MODELS:
            config.default_model = choice

    # ── Save ─────────────────────────────────────────────────────────
    ensure_dirs(config)
    save_config(config)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("key", style="dim")
    table.add_column("value", style="bold")
    table.add_row("model", config.default_model)
    table.add_row("library", str(config.library_dir))
    table.add_row("auth", detail if ok else "[red]not set[/red]")
    console.print(table)
    console.print(
        '\n[green]Setup complete.[/green] Try: [bold]openseed paper search "attention"[/bold]'
    )


@cli.command()
def doctor() -> None:
    """Check the environment — Python, dependencies, and auth."""
    results = run_checks()
    lines, issues = render_results(results)
    for line in lines:
        console.print(line)
    if issues:
        raise SystemExit(1)


cli.add_command(paper)
cli.add_command(experiment)
cli.add_command(agent)
cli.add_command(research)
cli.add_command(ask, name="ask")
