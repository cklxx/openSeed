"""SQLite-backed paper and experiment library.

Schema (hybrid: indexed columns + JSON blob):
┌─────────────────────────────────────────────────────────────┐
│ papers: id PK, arxiv_id UNIQUE, title, status, added_at,   │
│         data (full Pydantic JSON)                           │
│ experiments: id PK, name, paper_id, data                    │
│ watches: id PK, query, data                                 │
│ research_sessions: id PK, topic, created_at, data           │
│ paper_edges: (source_id, target_id, edge_type) PK,          │
│              weight, metadata, created_at                    │
│ papers_fts: FTS5 virtual table (title, abstract, summary,   │
│             authors, tags) — synced via triggers             │
│ research_memory: id, session_id, role, content, topics,      │
│                  created_at — conversation memory             │
│ research_memory_fts: FTS5 on content+topics                  │
│ schema_version: version INTEGER                              │
└─────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from openseed.models.experiment import Experiment
from openseed.models.paper import Paper
from openseed.models.research import ResearchSession
from openseed.models.watch import ArxivWatch

_SCHEMA_VERSION = 1

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS papers (
    id       TEXT PRIMARY KEY,
    arxiv_id TEXT UNIQUE,
    title    TEXT NOT NULL,
    status   TEXT NOT NULL DEFAULT 'unread',
    added_at TEXT NOT NULL,
    data     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_papers_status ON papers(status);

CREATE TABLE IF NOT EXISTS experiments (
    id       TEXT PRIMARY KEY,
    name     TEXT NOT NULL,
    paper_id TEXT,
    data     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS watches (
    id    TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    data  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS research_sessions (
    id         TEXT PRIMARY KEY,
    topic      TEXT NOT NULL,
    created_at TEXT NOT NULL,
    data       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_edges (
    source_id  TEXT NOT NULL,
    target_id  TEXT NOT NULL,
    edge_type  TEXT NOT NULL DEFAULT 'cites',
    weight     REAL DEFAULT 1.0,
    metadata   TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (source_id, target_id, edge_type)
);
CREATE INDEX IF NOT EXISTS idx_edges_source ON paper_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON paper_edges(target_id);

CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);

CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
    title, abstract, summary, authors, tags
);

CREATE TABLE IF NOT EXISTS research_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    topics TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS research_memory_fts USING fts5(
    content,
    topics,
    content='research_memory',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS research_memory_ai AFTER INSERT ON research_memory BEGIN
    INSERT INTO research_memory_fts(rowid, content, topics)
    VALUES (new.id, new.content, new.topics);
END;

CREATE TRIGGER IF NOT EXISTS research_memory_ad AFTER DELETE ON research_memory BEGIN
    INSERT INTO research_memory_fts(research_memory_fts, rowid, content, topics)
    VALUES('delete', old.id, old.content, old.topics);
END;
"""


def _fts_text(p: Paper) -> tuple[str, str, str, str, str]:
    """Extract FTS-indexable text fields from a Paper."""
    authors = " ".join(a.name for a in p.authors)
    tags = " ".join(t.name for t in p.tags)
    return (p.title, p.abstract, p.summary or "", authors, tags)


def _paper_to_row(p: Paper) -> tuple:
    return (p.id, p.arxiv_id, p.title, p.status, p.added_at.isoformat(), _dump(p))


def _row_to_paper(data_json: str) -> Paper:
    return Paper.model_validate(json.loads(data_json))


def _row_to_experiment(data_json: str) -> Experiment:
    return Experiment.model_validate(json.loads(data_json))


def _row_to_session(data_json: str) -> ResearchSession:
    return ResearchSession(**json.loads(data_json))


def _dump(model) -> str:
    return json.dumps(model.model_dump(mode="json"), default=str)


def _searchable_text(p: Paper) -> str:
    """All searchable text from a paper, lowercased."""
    return " ".join([*_fts_text(p), p.note]).lower()


def _title_score(p: Paper, tokens: list[str]) -> int:
    """Score: 2 points per token in title, 1 otherwise."""
    title_l = p.title.lower()
    return sum(2 if t in title_l else 1 for t in tokens)


def _save_markdown(base_dir: Path, filename: str, content: str) -> Path:
    """Write markdown file, skipping if unchanged."""
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / filename
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return path
    path.write_text(content, encoding="utf-8")
    return path


def _bfs_components(adj: dict[str, set[str]]) -> list[list[str]]:
    """Find connected components via BFS."""
    visited: set[str] = set()
    clusters: list[list[str]] = []
    for node in adj:
        if node in visited:
            continue
        cluster: list[str] = []
        queue = [node]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            cluster.append(current)
            queue.extend(adj.get(current, set()) - visited)
        if cluster:
            clusters.append(sorted(cluster))
    return sorted(clusters, key=len, reverse=True)


class PaperLibrary:
    """CRUD operations for papers and experiments backed by SQLite."""

    def __init__(self, library_dir: Path) -> None:
        self._dir = Path(library_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._dir / "library.db"
        self._conn = self._connect()
        self._ensure_schema()
        self._auto_migrate()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _ensure_schema(self) -> None:
        self._conn.executescript(_CREATE_SQL)
        row = self._conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        if row is None:
            self._conn.execute("INSERT INTO schema_version VALUES (?)", (_SCHEMA_VERSION,))
            self._conn.commit()

    def _auto_migrate(self) -> None:
        """Detect legacy JSON files and migrate them into SQLite."""
        from openseed.storage.migrate import migrate_json_to_sqlite

        migrate_json_to_sqlite(self._dir, self._conn)

    # ── Papers ────────────────────────────────────────────────

    def _index_fts(self, rowid: int, paper: Paper) -> None:
        """Insert a paper's text fields into the FTS index."""
        self._conn.execute(
            "INSERT INTO papers_fts(rowid, title, abstract, summary, authors, tags) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (rowid, *_fts_text(paper)),
        )

    def _paper_exists(self, paper: Paper) -> bool:
        """Check if paper already exists by arxiv_id or url."""
        if paper.arxiv_id:
            if self._conn.execute(
                "SELECT 1 FROM papers WHERE arxiv_id = ?", (paper.arxiv_id,)
            ).fetchone():
                return True
        if paper.url:
            if self._conn.execute(
                "SELECT 1 FROM papers WHERE json_extract(data, '$.url') = ?", (paper.url,)
            ).fetchone():
                return True
        return False

    def add_paper(self, paper: Paper) -> bool:
        """Add paper; skip if same arxiv_id or url already exists."""
        if self._paper_exists(paper):
            return False
        cur = self._conn.execute(
            "INSERT INTO papers (id, arxiv_id, title, status, added_at, data) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            _paper_to_row(paper),
        )
        self._index_fts(cur.lastrowid, paper)
        self._conn.commit()
        return True

    def get_paper(self, paper_id: str) -> Paper | None:
        row = self._conn.execute("SELECT data FROM papers WHERE id = ?", (paper_id,)).fetchone()
        return _row_to_paper(row[0]) if row else None

    def get_paper_by_arxiv(self, arxiv_id: str) -> Paper | None:
        row = self._conn.execute(
            "SELECT data FROM papers WHERE arxiv_id = ?", (arxiv_id,)
        ).fetchone()
        return _row_to_paper(row[0]) if row else None

    def list_papers(self) -> list[Paper]:
        rows = self._conn.execute("SELECT data FROM papers").fetchall()
        return [_row_to_paper(r[0]) for r in rows]

    def remove_paper(self, paper_id: str) -> bool:
        row = self._conn.execute("SELECT rowid FROM papers WHERE id = ?", (paper_id,)).fetchone()
        if row is None:
            return False
        self._conn.execute("DELETE FROM papers WHERE id = ?", (paper_id,))
        self._conn.execute("DELETE FROM papers_fts WHERE rowid = ?", (row[0],))
        self._conn.commit()
        return True

    def _get_rowid(self, paper_id: str) -> int:
        """Get the rowid for a paper, raising KeyError if not found."""
        row = self._conn.execute("SELECT rowid FROM papers WHERE id = ?", (paper_id,)).fetchone()
        if row is None:
            raise KeyError(f"Paper {paper_id} not found")
        return row[0]

    def update_paper(self, paper: Paper) -> None:
        rowid = self._get_rowid(paper.id)
        self._conn.execute(
            "UPDATE papers SET arxiv_id=?, title=?, status=?, added_at=?, data=? WHERE id = ?",
            (
                paper.arxiv_id,
                paper.title,
                paper.status,
                paper.added_at.isoformat(),
                _dump(paper),
                paper.id,
            ),
        )
        self._conn.execute("DELETE FROM papers_fts WHERE rowid = ?", (rowid,))
        self._index_fts(rowid, paper)
        self._conn.commit()

    def search_papers(self, query: str) -> list[Paper]:
        tokens = query.strip().split()
        if not tokens:
            return []
        fts_query = " AND ".join(f'"{t}"' for t in tokens)
        try:
            rows = self._conn.execute(
                "SELECT p.data, rank FROM papers_fts f "
                "JOIN papers p ON p.rowid = f.rowid "
                "WHERE papers_fts MATCH ? ORDER BY rank",
                (fts_query,),
            ).fetchall()
            if rows:
                return [_row_to_paper(r[0]) for r in rows]
        except sqlite3.OperationalError:
            pass
        return self._fallback_search(tokens)

    def _fallback_search(self, tokens: list[str]) -> list[Paper]:
        """Token-based search when FTS fails (e.g. special characters)."""
        low = [t.lower() for t in tokens]
        papers = self.list_papers()
        matches = [p for p in papers if all(t in _searchable_text(p) for t in low)]
        return sorted(matches, key=lambda p: _title_score(p, tokens), reverse=True)

    def rebuild_fts(self) -> int:
        """Re-index all papers into FTS — call when FTS is out of sync."""
        self._conn.execute("DELETE FROM papers_fts")
        rows = self._conn.execute("SELECT rowid, data FROM papers").fetchall()
        for rowid, data in rows:
            self._index_fts(rowid, _row_to_paper(data))
        self._conn.commit()
        return len(rows)

    # ── Experiments ───────────────────────────────────────────

    def add_experiment(self, experiment: Experiment) -> None:
        self._conn.execute(
            "INSERT INTO experiments (id, name, paper_id, data) VALUES (?, ?, ?, ?)",
            (experiment.id, experiment.name, experiment.paper_id, _dump(experiment)),
        )
        self._conn.commit()

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        row = self._conn.execute(
            "SELECT data FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
        return _row_to_experiment(row[0]) if row else None

    def get_experiment_by_name(self, name: str) -> Experiment | None:
        row = self._conn.execute("SELECT data FROM experiments WHERE name = ?", (name,)).fetchone()
        return _row_to_experiment(row[0]) if row else None

    def list_experiments(self) -> list[Experiment]:
        rows = self._conn.execute("SELECT data FROM experiments").fetchall()
        return [_row_to_experiment(r[0]) for r in rows]

    def remove_experiment(self, experiment_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
        self._conn.commit()
        return cur.rowcount > 0

    # ── Watches ───────────────────────────────────────────────

    def add_watch(self, watch: ArxivWatch) -> None:
        self._conn.execute(
            "INSERT INTO watches (id, query, data) VALUES (?, ?, ?)",
            (watch.id, watch.query, _dump(watch)),
        )
        self._conn.commit()

    def list_watches(self) -> list[ArxivWatch]:
        rows = self._conn.execute("SELECT data FROM watches").fetchall()
        return [ArxivWatch.model_validate(json.loads(r[0])) for r in rows]

    def update_watch(self, watch: ArxivWatch) -> None:
        cur = self._conn.execute(
            "UPDATE watches SET query=?, data=? WHERE id = ?",
            (watch.query, _dump(watch), watch.id),
        )
        self._conn.commit()
        if cur.rowcount == 0:
            raise KeyError(f"Watch {watch.id} not found")

    def remove_watch(self, watch_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM watches WHERE id = ?", (watch_id,))
        self._conn.commit()
        return cur.rowcount > 0

    # ── Research Sessions ─────────────────────────────────────

    def list_research_sessions(self) -> list[ResearchSession]:
        rows = self._conn.execute("SELECT data FROM research_sessions").fetchall()
        try:
            return [_row_to_session(r[0]) for r in rows]
        except (json.JSONDecodeError, ValueError):
            return []

    def add_research_session(self, session: ResearchSession) -> None:
        self._conn.execute(
            "INSERT INTO research_sessions (id, topic, created_at, data) VALUES (?, ?, ?, ?)",
            (session.id, session.topic, session.created_at.isoformat(), _dump(session)),
        )
        self._conn.commit()

    def get_research_session(self, session_id: str) -> ResearchSession | None:
        row = self._conn.execute(
            "SELECT data FROM research_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        return _row_to_session(row[0])

    # ── Knowledge Graph ───────────────────────────────────────

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str = "cites",
        weight: float = 1.0,
        metadata: dict | None = None,
    ) -> bool:
        """Add a directed edge between two papers. Returns True if new."""
        try:
            self._conn.execute(
                "INSERT OR IGNORE INTO paper_edges "
                "(source_id, target_id, edge_type, weight, metadata, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    source_id,
                    target_id,
                    edge_type,
                    weight,
                    json.dumps(metadata) if metadata else None,
                    datetime.now(UTC).isoformat(),
                ),
            )
            self._conn.commit()
            return self._conn.execute("SELECT changes()").fetchone()[0] > 0
        except sqlite3.IntegrityError:
            return False

    def get_neighbors(self, paper_id: str) -> list[dict]:
        """Get all papers connected to a given paper (both directions)."""
        rows = self._conn.execute(
            "SELECT target_id AS neighbor, edge_type, weight FROM paper_edges "
            "WHERE source_id = ? "
            "UNION ALL "
            "SELECT source_id AS neighbor, edge_type, weight FROM paper_edges "
            "WHERE target_id = ?",
            (paper_id, paper_id),
        ).fetchall()
        return [{"paper_id": r[0], "edge_type": r[1], "weight": r[2]} for r in rows]

    def get_edges_from(self, paper_id: str) -> list[dict]:
        """Get outgoing edges (papers this paper cites)."""
        rows = self._conn.execute(
            "SELECT target_id, edge_type, weight FROM paper_edges WHERE source_id = ?",
            (paper_id,),
        ).fetchall()
        return [{"paper_id": r[0], "edge_type": r[1], "weight": r[2]} for r in rows]

    def get_edges_to(self, paper_id: str) -> list[dict]:
        """Get incoming edges (papers that cite this paper)."""
        rows = self._conn.execute(
            "SELECT source_id, edge_type, weight FROM paper_edges WHERE target_id = ?",
            (paper_id,),
        ).fetchall()
        return [{"paper_id": r[0], "edge_type": r[1], "weight": r[2]} for r in rows]

    def list_all_edges(self) -> list[dict]:
        """Return all edges in the graph."""
        rows = self._conn.execute(
            "SELECT source_id, target_id, edge_type, weight FROM paper_edges"
        ).fetchall()
        return [{"source": r[0], "target": r[1], "edge_type": r[2], "weight": r[3]} for r in rows]

    def get_clusters(self) -> list[list[str]]:
        """Find connected components in the paper graph via BFS."""
        edges = self.list_all_edges()
        adj: dict[str, set[str]] = {}
        for e in edges:
            adj.setdefault(e["source"], set()).add(e["target"])
            adj.setdefault(e["target"], set()).add(e["source"])
        return _bfs_components(adj)

    def get_neighbor_counts(self) -> dict[str, int]:
        """Return {paper_id: neighbor_count} for all papers with at least one edge."""
        rows = self._conn.execute(
            "SELECT paper_id, COUNT(*) FROM ("
            "  SELECT source_id AS paper_id FROM paper_edges"
            "  UNION ALL"
            "  SELECT target_id AS paper_id FROM paper_edges"
            ") GROUP BY paper_id"
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def edge_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM paper_edges").fetchone()
        return row[0] if row else 0

    # ── Summaries ─────────────────────────────────────────────

    @property
    def _summaries_dir(self) -> Path:
        return self._dir.parent / "summaries"

    def save_summary(self, paper: Paper) -> Path:
        """Write paper.summary to ~/.openseed/summaries/{arxiv_id|id}.md."""
        slug = (paper.arxiv_id or paper.id).replace("/", "_")
        content = f"# {paper.title}\n\n{paper.summary}\n"
        return _save_markdown(self._summaries_dir, f"{slug}.md", content)

    def save_synthesis(self, paper_ids: list[str], content: str) -> Path:
        """Write synthesis markdown."""
        slug = "_".join(sorted(paper_ids)[:4])
        body = f"# Synthesis\n\n{content}\n"
        return _save_markdown(self._summaries_dir, f"synthesis_{slug}.md", body)

    def save_report(self, session_id: str, topic: str, content: str) -> Path:
        """Write research report."""
        slug = topic.lower().replace(" ", "_")[:40]
        return _save_markdown(
            self._summaries_dir,
            f"report_{slug}_{session_id}.md",
            f"# Research Report: {topic}\n\n{content}\n",
        )
