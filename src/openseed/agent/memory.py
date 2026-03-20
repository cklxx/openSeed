"""Research memory persistence — save and search conversation history."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from openseed.storage.library import PaperLibrary


@dataclass
class MemoryEntry:
    id: int
    session_id: str
    role: str
    content: str
    topics: list[str] | None
    created_at: str


class MemoryStore:
    """FTS5-backed conversation memory using PaperLibrary's SQLite connection."""

    def __init__(self, library: PaperLibrary) -> None:
        self._conn = library._conn

    def save_memory(
        self,
        session_id: str,
        role: str,
        content: str,
        topics: list[str] | None = None,
    ) -> int:
        topics_json = json.dumps(topics) if topics is not None else None
        cur = self._conn.execute(
            "INSERT INTO research_memory (session_id, role, content, topics) VALUES (?, ?, ?, ?)",
            (session_id, role, content, topics_json),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def search_memories(self, query: str, top_k: int = 10) -> list[MemoryEntry]:
        if not query.strip():
            return []
        try:
            rows = self._conn.execute(
                "SELECT m.id, m.session_id, m.role, m.content, m.topics, m.created_at "
                "FROM research_memory_fts f "
                "JOIN research_memory m ON f.rowid = m.id "
                "WHERE research_memory_fts MATCH ? "
                "ORDER BY f.rank "
                "LIMIT ?",
                (query, top_k),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = self._conn.execute(
                "SELECT id, session_id, role, content, topics, created_at "
                "FROM research_memory "
                "WHERE content LIKE ? OR topics LIKE ? "
                "LIMIT ?",
                (f"%{query}%", f"%{query}%", top_k),
            ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get_session_history(self, session_id: str) -> list[MemoryEntry]:
        rows = self._conn.execute(
            "SELECT id, session_id, role, content, topics, created_at "
            "FROM research_memory "
            "WHERE session_id = ? "
            "ORDER BY created_at ASC",
            (session_id,),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def clear_session(self, session_id: str) -> int:
        cur = self._conn.execute(
            "DELETE FROM research_memory WHERE session_id = ?",
            (session_id,),
        )
        self._conn.commit()
        return cur.rowcount

    @staticmethod
    def _row_to_entry(row: tuple) -> MemoryEntry:
        id_, session_id, role, content, topics_json, created_at = row
        topics = json.loads(topics_json) if topics_json is not None else None
        return MemoryEntry(
            id=id_,
            session_id=session_id,
            role=role,
            content=content,
            topics=topics,
            created_at=created_at,
        )
