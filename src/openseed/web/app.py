"""FastAPI web dashboard for OpenSeed."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from openseed.config import load_config
from openseed.storage.library import PaperLibrary

_TEMPLATES_DIR = Path(__file__).parent / "templates"

app = FastAPI(title="OpenSeed", docs_url=None, redoc_url=None)
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


def _lib() -> PaperLibrary:
    config = load_config()
    return PaperLibrary(config.library_dir)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    lib = _lib()
    papers = lib.list_papers()
    stats = {
        "total": len(papers),
        "unread": sum(1 for p in papers if p.status == "unread"),
        "reading": sum(1 for p in papers if p.status == "reading"),
        "read": sum(1 for p in papers if p.status == "read"),
        "edges": lib.edge_count(),
        "clusters": len(lib.get_clusters()),
    }
    return templates.TemplateResponse(request, "index.html", {"stats": stats})


@app.get("/papers", response_class=HTMLResponse)
async def papers_list(request: Request, status: str | None = None, q: str | None = None):
    lib = _lib()
    if q:
        papers = lib.search_papers(q)
    else:
        papers = lib.list_papers()
    if status:
        papers = [p for p in papers if p.status == status]
    return templates.TemplateResponse(
        request, "papers.html", {"papers": papers, "status": status, "q": q or ""}
    )


@app.get("/papers/{paper_id}", response_class=HTMLResponse)
async def paper_detail(request: Request, paper_id: str):
    lib = _lib()
    paper = lib.get_paper(paper_id)
    if not paper:
        return HTMLResponse("<h1>Paper not found</h1>", status_code=404)
    neighbors = lib.get_neighbors(paper_id)
    neighbor_papers = []
    for n in neighbors:
        p = lib.get_paper(n["paper_id"])
        if p:
            neighbor_papers.append({"paper": p, "edge_type": n["edge_type"]})
    return templates.TemplateResponse(
        request, "paper_detail.html", {"paper": paper, "neighbors": neighbor_papers}
    )


@app.get("/graph", response_class=HTMLResponse)
async def graph_view(request: Request):
    lib = _lib()
    papers = lib.list_papers()
    edges = lib.list_all_edges()
    clusters = lib.get_clusters()
    paper_map = {p.id: p for p in papers}
    neighbor_counts = lib.get_neighbor_counts()
    nodes = [
        {"id": p.id, "title": p.title[:40], "connections": neighbor_counts[p.id]}
        for p in papers
        if p.id in neighbor_counts
    ]
    return templates.TemplateResponse(
        request,
        "graph.html",
        {"nodes": nodes, "edges": edges, "clusters": clusters, "paper_map": paper_map},
    )


@app.get("/digests", response_class=HTMLResponse)
async def digests_list(request: Request):
    config = load_config()
    digest_dir = Path(config.config_dir) / "digests"
    digests = []
    if digest_dir.exists():
        for f in sorted(digest_dir.glob("digest_*.md"), reverse=True):
            digests.append({"name": f.stem, "content": f.read_text(encoding="utf-8")})
    return templates.TemplateResponse(request, "digests.html", {"digests": digests})


@app.get("/sessions", response_class=HTMLResponse)
async def sessions_list(request: Request):
    lib = _lib()
    sessions = lib.list_research_sessions()
    return templates.TemplateResponse(request, "sessions.html", {"sessions": sessions})
