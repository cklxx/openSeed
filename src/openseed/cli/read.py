"""CLI command: open a markdown file in a beautiful browser viewer."""

from __future__ import annotations

import os
import socket
import threading
import webbrowser

import click
from rich.console import Console

console = Console()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@click.command()
@click.argument("filepath", type=click.Path(exists=True, readable=True))
@click.option("--port", default=0, help="Port to serve on (0 = auto-select).")
def read(filepath: str, port: int) -> None:
    """Open a markdown file or folder in the browser with beautiful rendering."""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]uvicorn not installed.[/red] Run: pip install openseed[web]")
        raise SystemExit(1)

    from pathlib import Path

    path = Path(filepath).resolve()
    if port == 0:
        port = _find_free_port()

    is_dir = path.is_dir()
    os.environ["_OPENSEED_READ_FILE"] = str(path)
    os.environ["_OPENSEED_READ_MODE"] = "dir" if is_dir else "file"

    config = uvicorn.Config(
        "openseed.viewer.app:app", host="127.0.0.1", port=port, log_level="error"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    label = f"[bold]{path.name}/[/bold]" if is_dir else f"[bold]{path.name}[/bold]"
    url = f"http://127.0.0.1:{port}"
    console.print(f"[green]✓[/green] Viewing {label} at {url}")
    webbrowser.open(url)
    console.print("[dim]Press Ctrl+C to stop[/dim]")

    try:
        thread.join()
    except KeyboardInterrupt:
        server.should_exit = True
