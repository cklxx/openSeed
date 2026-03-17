"""Environment health checks for OpenSeed."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class CheckResult:
    name: str
    ok: bool
    version: str | None = None
    detail: str | None = None
    fix_hint: str | None = None


def _get_version(cmd: str, flag: str = "--version") -> str | None:
    try:
        result = subprocess.run([cmd, flag], capture_output=True, text=True, timeout=5)
        out = (result.stdout + result.stderr).strip()
        first = next((ln.strip() for ln in out.splitlines() if ln.strip()), "")
        return first[:40] or None
    except Exception:
        return None


def _check_python() -> CheckResult:
    v = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    ok = sys.version_info >= (3, 11)
    return CheckResult(
        name="Python",
        ok=ok,
        version=v,
        fix_hint=None if ok else "OpenSeed requires Python 3.11+",
    )


def _check_cli(name: str, cmd: str, fix: str, optional: bool = False) -> CheckResult:
    found = shutil.which(cmd)
    if not found:
        return CheckResult(
            name=name,
            ok=optional,
            detail="not installed" if optional else None,
            fix_hint=fix,
        )
    v = _get_version(cmd)
    return CheckResult(name=name, ok=True, version=v)


def _check_auth() -> CheckResult:
    from openseed.auth import has_anthropic_auth

    ok, detail = has_anthropic_auth()
    if ok:
        return CheckResult(name="Anthropic auth", ok=True, detail=detail)
    return CheckResult(
        name="Anthropic auth",
        ok=False,
        fix_hint="run: openseed setup",
    )


def run_checks() -> list[CheckResult]:
    return [
        _check_python(),
        _check_cli("openseed", "openseed", "pip install openseed"),
        _check_cli(
            "claude CLI",
            "claude",
            "See https://docs.anthropic.com/en/docs/claude-code",
            optional=True,
        ),
        _check_auth(),
    ]


def render_results(results: list[CheckResult]) -> tuple[list[str], int]:
    lines: list[str] = ["OpenSeed environment check", "─" * 42]
    issues = 0
    for r in results:
        icon = "✅" if r.ok else "❌"
        ver = f"  {r.version}" if r.version else ""
        detail = f"  ({r.detail})" if r.detail else ""
        lines.append(f"  {icon}  {r.name:<18}{ver}{detail}")
        if not r.ok:
            issues += 1
            if r.fix_hint:
                lines.append(f"       Fix: {r.fix_hint}")
    lines.append("─" * 42)
    if issues == 0:
        lines.append("All checks passed ✅")
    else:
        noun = "issue" if issues == 1 else "issues"
        lines.append(f"{issues} {noun} found.")
    return lines, issues
