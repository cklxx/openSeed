"""OpenSeed CLI entry point."""

from __future__ import annotations

import click
from rich.console import Console

from openseed import __version__
from openseed.cli.agent import agent
from openseed.cli.experiment import experiment
from openseed.cli.paper import paper
from openseed.config import ensure_dirs, load_config

console = Console()


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


cli.add_command(paper)
cli.add_command(experiment)
cli.add_command(agent)
