"""JSON-based paper and experiment library."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from openseed.models.experiment import Experiment
from openseed.models.paper import Paper
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
        return self._papers_cache

    def _save_papers(self, papers: list[Paper]) -> None:
        self._atomic_write(
            self._papers_path,
            json.dumps([p.model_dump(mode="json") for p in papers], indent=2, default=str),
        )
        self._papers_cache = papers

    def add_paper(self, paper: Paper) -> bool:
        """Add paper; skip if same arxiv_id already exists. Returns True if added."""
        papers = self._load_papers()
        if paper.arxiv_id and any(p.arxiv_id == paper.arxiv_id for p in papers):
            return False
        papers.append(paper)
        self._save_papers(papers)
        return True

    def get_paper(self, paper_id: str) -> Paper | None:
        for p in self._load_papers():
            if p.id == paper_id:
                return p
        return None

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
        q = query.lower()
        return [p for p in self._load_papers() if q in p.title.lower() or q in p.abstract.lower()]

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

    # ── Summaries ─────────────────────────────────────────────

    def save_summary(self, paper: Paper) -> Path:
        """Write paper.summary to ~/.openseed/summaries/{arxiv_id|id}.md and return the path."""
        summaries_dir = self._dir.parent / "summaries"
        summaries_dir.mkdir(parents=True, exist_ok=True)
        slug = (paper.arxiv_id or paper.id).replace("/", "_")
        path = summaries_dir / f"{slug}.md"
        path.write_text(f"# {paper.title}\n\n{paper.summary}\n", encoding="utf-8")
        return path

    def save_synthesis(self, paper_ids: list[str], content: str) -> Path:
        """Write synthesis markdown to ~/.openseed/summaries/synthesis_{ids}.md."""
        summaries_dir = self._dir.parent / "summaries"
        summaries_dir.mkdir(parents=True, exist_ok=True)
        slug = "_".join(paper_ids[:4])
        path = summaries_dir / f"synthesis_{slug}.md"
        path.write_text(f"# Synthesis\n\n{content}\n", encoding="utf-8")
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
