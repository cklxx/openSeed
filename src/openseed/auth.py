"""Authentication helpers for OpenSeed."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys


def has_anthropic_auth() -> tuple[bool, str]:
    """Check available auth. Returns (ok, detail).

    claude-agent-sdk uses the claude CLI, so claude auth is valid.
    ANTHROPIC_API_KEY is also accepted by the CLI.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True, "ANTHROPIC_API_KEY"

    claude_bin = shutil.which("claude")
    if claude_bin:
        try:
            r = subprocess.run(
                [claude_bin, "auth", "status"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if r.returncode == 0 and "true" in r.stdout.lower():
                return True, "claude auth"
        except Exception:
            pass

    return False, "not configured"


def run_claude_setup_token() -> tuple[bool, str | None]:
    """Run `claude setup-token`, tee output to terminal, capture OAuth token.

    Returns (success, token_value_or_None).
    """
    claude_bin = shutil.which("claude")
    if not claude_bin:
        return False, None

    token: str | None = None
    proc = subprocess.Popen(
        [claude_bin, "setup-token"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=sys.stdin,
    )
    for raw in proc.stdout:
        line = raw.decode(errors="replace")
        sys.stdout.write(line)
        sys.stdout.flush()
        if "CLAUDE_CODE_OAUTH_TOKEN=" in line:
            token = line.split("CLAUDE_CODE_OAUTH_TOKEN=", 1)[1].strip().strip('"').strip("'")
    proc.wait()
    return proc.returncode == 0, token


def detect_rc_file() -> str:
    """Return the most appropriate shell rc file path."""
    shell = os.environ.get("SHELL", "")
    home = os.path.expanduser("~")
    if "zsh" in shell:
        return f"{home}/.zshrc"
    rc = f"{home}/.bashrc"
    return rc if os.path.exists(rc) else f"{home}/.bash_profile"


def append_export_to_rc(var: str, value: str, rc_file: str) -> None:
    """Append an export line to the given rc file."""
    with open(rc_file, "a") as f:
        f.write(f"\n# Added by openseed setup\nexport {var}={value}\n")
