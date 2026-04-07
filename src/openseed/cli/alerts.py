"""CLI commands for claim-based research alerts."""

from __future__ import annotations

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from openseed.auth import has_anthropic_auth
from openseed.config import load_config
from openseed.storage.library import PaperLibrary

console = Console()


def _get_library() -> PaperLibrary:
    config = load_config()
    return PaperLibrary(config.library_dir)


@click.group()
def alerts() -> None:
    """Manage research insight alerts."""


@alerts.command("list")
@click.option("--all", "show_all", is_flag=True, help="Show all alerts, not just unread.")
def list_alerts(show_all: bool) -> None:
    """List research alerts sorted by confidence."""
    lib = _get_library()
    items = lib.list_alerts(unread_only=not show_all)
    if not items:
        console.print("[dim]No alerts. Add papers to generate insights.[/dim]")
        return
    table = Table(title="Research Alerts", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Type", width=12)
    table.add_column("Summary")
    table.add_column("Conf", width=5)
    table.add_column("Status", width=8)
    for a in items:
        status = "read" if a["is_read"] else "[bold]new[/bold]"
        if a["is_dismissed"]:
            status = "[dim]dismissed[/dim]"
        style = "red" if a["alert_type"] == "contradicts" else "yellow"
        table.add_row(
            str(a["id"]),
            f"[{style}]{a['alert_type']}[/{style}]",
            a["summary"],
            f"{a['confidence']:.2f}",
            status,
        )
    console.print(table)


@alerts.command("read")
@click.argument("alert_id", type=int)
def read_alert(alert_id: int) -> None:
    """Mark an alert as read."""
    lib = _get_library()
    if lib.update_alert(alert_id, is_read=1):
        console.print(f"[green]Alert {alert_id} marked as read.[/green]")
    else:
        console.print(f"[red]Alert {alert_id} not found.[/red]")


@alerts.command("dismiss")
@click.argument("alert_id", type=int)
def dismiss_alert(alert_id: int) -> None:
    """Dismiss an alert."""
    lib = _get_library()
    if lib.update_alert(alert_id, is_dismissed=1):
        console.print(f"[dim]Alert {alert_id} dismissed.[/dim]")
    else:
        console.print(f"[red]Alert {alert_id} not found.[/red]")


@alerts.command("useful")
@click.argument("alert_id", type=int)
def useful_alert(alert_id: int) -> None:
    """Mark an alert as useful (quality feedback)."""
    lib = _get_library()
    if lib.update_alert(alert_id, is_useful=1):
        console.print(f"[green]Alert {alert_id} marked as useful.[/green]")
    else:
        console.print(f"[red]Alert {alert_id} not found.[/red]")


@alerts.command("backfill")
def backfill() -> None:
    """Extract claims from papers that haven't been analyzed yet."""
    ok, _ = has_anthropic_auth()
    if not ok:
        console.print("[red]No auth configured.[/red] Run: openseed setup")
        return

    from openseed.agent.claims import extract_claims, get_paper_text
    from openseed.agent.matcher import match_claims

    config = load_config()
    lib = _get_library()
    papers = lib.papers_needing_claims()
    if not papers:
        console.print("[dim]All papers have been analyzed.[/dim]")
        return

    console.print(f"Backfilling claims for {len(papers)} papers...")
    model = config.default_model
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task("Processing...", total=len(papers))
        for p in papers:
            label = p["title"][:50] or p["id"]
            progress.update(task, description=f"[bold]{label}[/bold]")
            text = get_paper_text(p["id"], lib)
            if not text:
                lib.set_claims_status(p["id"], "complete")
                progress.advance(task)
                continue
            try:
                extract_claims(p["id"], text, model, lib)
                match_claims(p["id"], model, lib)
            except Exception:
                pass  # claims.py already sets status to 'failed'
            progress.advance(task)

    console.print("[green]Backfill complete.[/green]")
    items = lib.list_alerts(unread_only=True)
    if items:
        console.print(f"[bold]{len(items)} new alerts[/bold] — run `openseed alerts list`")


def run_claim_analysis(paper_id: str, lib: PaperLibrary) -> None:
    """Run claim extraction + matching for a single paper. Called from paper add."""
    ok, _ = has_anthropic_auth()
    if not ok:
        return

    from openseed.agent.claims import extract_claims, get_paper_text
    from openseed.agent.matcher import match_claims

    config = load_config()
    text = get_paper_text(paper_id, lib)
    if not text:
        return

    try:
        with console.status("[bold]Analyzing claims...[/bold]"):
            extract_claims(paper_id, text, config.default_model, lib)
            edges, alert_count = match_claims(paper_id, config.default_model, lib)
        if alert_count > 0:
            console.print(f"[bold]{alert_count} new insight(s)[/bold] — run `openseed alerts list`")
        elif edges > 0:
            console.print(f"[dim]{edges} connections found, no new alerts.[/dim]")
    except Exception:
        console.print("[dim]Claim analysis failed — retry with `openseed alerts backfill`[/dim]")
