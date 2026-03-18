"""JSON → SQLite auto-migration.

Migration flow:
┌─────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│ Detect  │────▶│ Backup   │────▶│ BEGIN    │────▶│ INSERT   │────▶│ VERIFY   │
│ JSON    │     │ JSON→bak │     │ TRANS    │     │ ALL rows │     │ counts   │
└─────────┘     └──────────┘     └──────────┘     └──────────┘     └────┬─────┘
                                                                        │
                                                             ┌──────────▼──────┐
                                                             │ COMMIT or       │
                                                             │ ROLLBACK on err │
                                                             └─────────────────┘
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
from pathlib import Path

_log = logging.getLogger(__name__)

_JSON_FILES = {
    "papers": "papers.json",
    "experiments": "experiments.json",
    "watches": "watches.json",
    "research_sessions": "research_sessions.json",
}


def _has_json_data(library_dir: Path) -> bool:
    return any((library_dir / f).exists() for f in _JSON_FILES.values())


def _is_migrated(library_dir: Path) -> bool:
    marker = library_dir / ".migrated"
    return marker.exists()


def _backup_json(library_dir: Path) -> Path:
    backup_dir = library_dir / "json_backup"
    backup_dir.mkdir(exist_ok=True)
    for name in _JSON_FILES.values():
        src = library_dir / name
        if src.exists():
            shutil.copy2(src, backup_dir / name)
    return backup_dir


def _load_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _paper_row(d: dict) -> tuple:
    return (
        d.get("id", ""),
        d.get("arxiv_id"),
        d.get("title", ""),
        d.get("status", "unread"),
        d.get("added_at", ""),
        json.dumps(d, default=str),
    )


def _experiment_row(d: dict) -> tuple:
    return (d.get("id", ""), d.get("name", ""), d.get("paper_id"), json.dumps(d, default=str))


def _watch_row(d: dict) -> tuple:
    return (d.get("id", ""), d.get("query", ""), json.dumps(d, default=str))


def _session_row(d: dict) -> tuple:
    return (
        d.get("id", ""),
        d.get("topic", ""),
        d.get("created_at", ""),
        json.dumps(d, default=str),
    )


def migrate_json_to_sqlite(library_dir: Path, conn: sqlite3.Connection) -> bool:
    """Migrate legacy JSON files into the SQLite database.

    Returns True if migration was performed, False if skipped.
    """
    if not _has_json_data(library_dir) or _is_migrated(library_dir):
        return False

    already_has_data = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] > 0
    if already_has_data:
        (library_dir / ".migrated").touch()
        return False

    _log.info("Migrating JSON data to SQLite…")
    backup_dir = _backup_json(library_dir)
    _log.info("JSON backup saved to %s", backup_dir)

    try:
        conn.execute("BEGIN")
        _migrate_papers(library_dir, conn)
        _migrate_experiments(library_dir, conn)
        _migrate_watches(library_dir, conn)
        _migrate_sessions(library_dir, conn)
        _verify_counts(library_dir, conn)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        _log.error("Migration failed — JSON backup preserved at %s", backup_dir)
        raise

    (library_dir / ".migrated").touch()
    _log.info("Migration complete")
    return True


def _migrate_papers(library_dir: Path, conn: sqlite3.Connection) -> None:
    items = _load_json(library_dir / "papers.json")
    for d in items:
        conn.execute(
            "INSERT OR IGNORE INTO papers (id, arxiv_id, title, status, added_at, data) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            _paper_row(d),
        )


def _migrate_experiments(library_dir: Path, conn: sqlite3.Connection) -> None:
    items = _load_json(library_dir / "experiments.json")
    for d in items:
        conn.execute(
            "INSERT OR IGNORE INTO experiments (id, name, paper_id, data) VALUES (?, ?, ?, ?)",
            _experiment_row(d),
        )


def _migrate_watches(library_dir: Path, conn: sqlite3.Connection) -> None:
    items = _load_json(library_dir / "watches.json")
    for d in items:
        conn.execute(
            "INSERT OR IGNORE INTO watches (id, query, data) VALUES (?, ?, ?)",
            _watch_row(d),
        )


def _migrate_sessions(library_dir: Path, conn: sqlite3.Connection) -> None:
    items = _load_json(library_dir / "research_sessions.json")
    for d in items:
        conn.execute(
            "INSERT OR IGNORE INTO research_sessions (id, topic, created_at, data) "
            "VALUES (?, ?, ?, ?)",
            _session_row(d),
        )


def _verify_counts(library_dir: Path, conn: sqlite3.Connection) -> None:
    for table, filename in _JSON_FILES.items():
        expected = len(_load_json(library_dir / filename))
        actual = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608
        if actual < expected:
            _log.warning(
                "Migration count mismatch for %s: expected %d, got %d",
                table,
                expected,
                actual,
            )
