"""JSON-based paper and experiment library."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from openseed.models.experiment import Experiment
from openseed.models.paper import Paper
from openseed.models.research import ResearchSession
from openseed.models.watch import ArxivWatch


class PaperLibrary:
    """CRUD operations for papers and experiments backed by JSON files."""

    def __init__(self, library_dir: Path) -> None:
        self._dir = Path(library_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._papers_path = self._dir / "papers.json"
        self._experiments_path = self._dir / "experiments.json"
        self._watches_path = self._dir / "watches.json"
        self._papers_cache: list[Paper] | None = None
        self._experiments_cache: list[Experiment] | None = None
        self._watches_cache: list[ArxivWatch] | None = None
        self._papers_by_id: dict[str, Paper] | None = None
        self._papers_by_arxiv: dict[str, Paper] | None = None
        self._sessions_cache: list[ResearchSession] | None = None

    @property
    def _research_path(self) -> Path:
        return self._dir / "research_sessions.json"

    def _invalidate_cache(self) -> None:
        self._papers_cache = None
        self._experiments_cache = None
        self._watches_cache = None

    # ── Papers ────────────────────────────────────────────────

    def _load_papers(self) -> list[Paper]:
        if self._papers_cache is not None:
            return self._papers_cache
        if not self._papers_path.exists():
            return []
        data = json.loads(self._papers_path.read_text())
        self._papers_cache = [Paper.model_validate(d) for d in data]
        self._papers_by_id = {p.id: p for p in self._papers_cache}
        self._papers_by_arxiv = {p.arxiv_id: p for p in self._papers_cache if p.arxiv_id}
        return self._papers_cache

    def _save_papers(self, papers: list[Paper]) -> None:
        self._atomic_write(
            self._papers_path,
            json.dumps([p.model_dump(mode="json") for p in papers], indent=2, default=str),
        )
        self._papers_cache = papers
        self._papers_by_id = {p.id: p for p in papers}
        self._papers_by_arxiv = {p.arxiv_id: p for p in papers if p.arxiv_id}

    def add_paper(self, paper: Paper) -> bool:
        """Add paper; skip if same arxiv_id or url already exists. Returns True if added."""
        self._load_papers()
        if paper.arxiv_id and (self._papers_by_arxiv or {}).get(paper.arxiv_id):
            return False
        if paper.url and any(p.url == paper.url for p in self._papers_cache or []):
            return False
        papers = list(self._papers_cache or [])
        papers.append(paper)
        self._save_papers(papers)
        return True

    def get_paper(self, paper_id: str) -> Paper | None:
        self._load_papers()
        return (self._papers_by_id or {}).get(paper_id)

    def list_papers(self) -> list[Paper]:
        return self._load_papers()

    def remove_paper(self, paper_id: str) -> bool:
        papers = self._load_papers()
        filtered = [p for p in papers if p.id != paper_id]
        if len(filtered) == len(papers):
            return False
        self._save_papers(filtered)
        return True

    def update_paper(self, paper: Paper) -> None:
        papers = self._load_papers()
        for i, p in enumerate(papers):
            if p.id == paper.id:
                papers[i] = paper
                self._save_papers(papers)
                return
        raise KeyError(f"Paper {paper.id} not found")

    def search_papers(self, query: str) -> list[Paper]:
        tokens = query.lower().split()
        if not tokens:
            return []

        def _text(p: Paper) -> str:
            authors = " ".join(a.name for a in p.authors)
            tags = " ".join(t.name for t in p.tags)
            return " ".join([p.title, p.abstract, p.note, p.summary or "", authors, tags]).lower()

        def _score(p: Paper) -> int:
            title_l = p.title.lower()
            return sum(2 if t in title_l else 1 for t in tokens)

        matches = [p for p in self._load_papers() if all(t in _text(p) for t in tokens)]
        return sorted(matches, key=_score, reverse=True)

    # ── Experiments ───────────────────────────────────────────

    def _load_experiments(self) -> list[Experiment]:
        if self._experiments_cache is not None:
            return self._experiments_cache
        if not self._experiments_path.exists():
            return []
        data = json.loads(self._experiments_path.read_text())
        self._experiments_cache = [Experiment.model_validate(d) for d in data]
        return self._experiments_cache

    def _save_experiments(self, experiments: list[Experiment]) -> None:
        self._atomic_write(
            self._experiments_path,
            json.dumps([e.model_dump(mode="json") for e in experiments], indent=2, default=str),
        )
        self._experiments_cache = experiments

    def add_experiment(self, experiment: Experiment) -> None:
        experiments = self._load_experiments()
        experiments.append(experiment)
        self._save_experiments(experiments)

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        for e in self._load_experiments():
            if e.id == experiment_id:
                return e
        return None

    def get_experiment_by_name(self, name: str) -> Experiment | None:
        for e in self._load_experiments():
            if e.name == name:
                return e
        return None

    def list_experiments(self) -> list[Experiment]:
        return self._load_experiments()

    def remove_experiment(self, experiment_id: str) -> bool:
        experiments = self._load_experiments()
        filtered = [e for e in experiments if e.id != experiment_id]
        if len(filtered) == len(experiments):
            return False
        self._save_experiments(filtered)
        return True

    # ── Watches ───────────────────────────────────────────────

    def _load_watches(self) -> list[ArxivWatch]:
        if self._watches_cache is not None:
            return self._watches_cache
        if not self._watches_path.exists():
            return []
        data = json.loads(self._watches_path.read_text())
        self._watches_cache = [ArxivWatch.model_validate(d) for d in data]
        return self._watches_cache

    def _save_watches(self, watches: list[ArxivWatch]) -> None:
        self._atomic_write(
            self._watches_path,
            json.dumps([w.model_dump(mode="json") for w in watches], indent=2, default=str),
        )
        self._watches_cache = watches

    def add_watch(self, watch: ArxivWatch) -> None:
        watches = self._load_watches()
        watches.append(watch)
        self._save_watches(watches)

    def list_watches(self) -> list[ArxivWatch]:
        return self._load_watches()

    def update_watch(self, watch: ArxivWatch) -> None:
        watches = self._load_watches()
        for i, w in enumerate(watches):
            if w.id == watch.id:
                watches[i] = watch
                self._save_watches(watches)
                return
        raise KeyError(f"Watch {watch.id} not found")

    def remove_watch(self, watch_id: str) -> bool:
        watches = self._load_watches()
        filtered = [w for w in watches if w.id != watch_id]
        if len(filtered) == len(watches):
            return False
        self._save_watches(filtered)
        return True

    # ── Research Sessions ─────────────────────────────────────

    def list_research_sessions(self) -> list[ResearchSession]:
        if self._sessions_cache is not None:
            return self._sessions_cache
        if not self._research_path.exists():
            return []
        try:
            self._sessions_cache = [
                ResearchSession(**d) for d in json.loads(self._research_path.read_text())
            ]
            return self._sessions_cache
        except Exception:
            return []

    def add_research_session(self, session: ResearchSession) -> None:
        sessions = self.list_research_sessions()
        sessions.append(session)
        self._sessions_cache = sessions
        self._atomic_write(
            self._research_path,
            json.dumps([s.model_dump(mode="json") for s in sessions], indent=2, default=str),
        )

    def get_research_session(self, session_id: str) -> ResearchSession | None:
        return next((s for s in self.list_research_sessions() if s.id == session_id), None)

    # ── Summaries ─────────────────────────────────────────────

    def save_summary(self, paper: Paper) -> Path:
        """Write paper.summary to ~/.openseed/summaries/{arxiv_id|id}.md; skip if unchanged."""
        summaries_dir = self._dir.parent / "summaries"
        summaries_dir.mkdir(parents=True, exist_ok=True)
        slug = (paper.arxiv_id or paper.id).replace("/", "_")
        path = summaries_dir / f"{slug}.md"
        content = f"# {paper.title}\n\n{paper.summary}\n"
        if path.exists() and path.read_text(encoding="utf-8") == content:
            return path
        path.write_text(content, encoding="utf-8")
        return path

    def save_synthesis(self, paper_ids: list[str], content: str) -> Path:
        """Write synthesis markdown to summaries/synthesis_{ids}.md; skip if unchanged."""
        summaries_dir = self._dir.parent / "summaries"
        summaries_dir.mkdir(parents=True, exist_ok=True)
        slug = "_".join(sorted(paper_ids)[:4])
        path = summaries_dir / f"synthesis_{slug}.md"
        new_content = f"# Synthesis\n\n{content}\n"
        if path.exists() and path.read_text(encoding="utf-8") == new_content:
            return path
        path.write_text(new_content, encoding="utf-8")
        return path

    def save_report(self, session_id: str, topic: str, content: str) -> Path:
        """Write research report to summaries/report_{slug}_{session_id}.md."""
        summaries_dir = self._dir.parent / "summaries"
        summaries_dir.mkdir(parents=True, exist_ok=True)
        slug = topic.lower().replace(" ", "_")[:40]
        path = summaries_dir / f"report_{slug}_{session_id}.md"
        new_content = f"# Research Report: {topic}\n\n{content}\n"
        if path.exists() and path.read_text(encoding="utf-8") == new_content:
            return path
        path.write_text(new_content, encoding="utf-8")
        return path

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        tmp = tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", delete=False)
        try:
            tmp.write(content)
            tmp.flush()
            Path(tmp.name).replace(path)
        finally:
            tmp.close()
