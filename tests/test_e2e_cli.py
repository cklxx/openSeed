"""End-to-end CLI tests: isolated filesystem, ArXiv fetch mocked."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from openseed.cli.main import cli
from openseed.storage.library import PaperLibrary

_URL = "https://arxiv.org/abs/1706.03762"
_ID = "1706.03762"


@pytest.fixture(autouse=True)
def mock_fetch(sample_paper):
    target = "openseed.cli.paper.fetch_paper_metadata"
    with patch(target, new_callable=AsyncMock, return_value=sample_paper):
        yield


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def env(tmp_path):
    """Isolated config dir — no writes to ~/.openseed."""
    return {"OPENSEED_CONFIG_DIR": str(tmp_path)}


@pytest.fixture
def lib(tmp_path):
    return PaperLibrary(tmp_path / "library")


def _invoke(runner, args, env, **kw):
    return runner.invoke(cli, args, env=env, catch_exceptions=False, **kw)


# ── Basic ────────────────────────────────────────────────────────────────────


def test_version(runner, env):
    r = _invoke(runner, ["--version"], env)
    assert r.exit_code == 0
    assert "openseed" in r.output and "version" in r.output


def test_doctor_exits_cleanly(runner, env):
    r = runner.invoke(cli, ["doctor"], env=env)
    assert r.exit_code in (0, 1)
    assert "Python" in r.output


# ── Paper lifecycle ──────────────────────────────────────────────────────────


def test_paper_add(runner, env):
    r = _invoke(runner, ["paper", "add", _URL], env)
    assert r.exit_code == 0
    assert "Attention" in r.output


def test_paper_add_dedup(runner, env):
    _invoke(runner, ["paper", "add", _URL], env)
    r = _invoke(runner, ["paper", "add", _URL], env)
    assert r.exit_code == 0
    assert "already" in r.output.lower()


def test_paper_list(runner, env):
    _invoke(runner, ["paper", "add", _URL], env)
    r = _invoke(runner, ["paper", "list"], env)
    assert r.exit_code == 0
    assert "Attention" in r.output


def test_paper_search_single_token(runner, env):
    _invoke(runner, ["paper", "add", _URL], env)
    r = _invoke(runner, ["paper", "search", "attention"], env)
    assert r.exit_code == 0
    # Rich table truncates titles; check the stable arxiv ID column instead
    assert _ID in r.output


def test_paper_search_multi_token(runner, env):
    _invoke(runner, ["paper", "add", _URL], env)
    r = _invoke(runner, ["paper", "search", "attention transformer"], env)
    assert r.exit_code == 0
    assert _ID in r.output


def test_paper_search_no_match(runner, env):
    _invoke(runner, ["paper", "add", _URL], env)
    r = _invoke(runner, ["paper", "search", "quantum_xyz_noop_token"], env)
    assert r.exit_code == 0
    assert "Attention" not in r.output


def test_paper_show(runner, env, lib):
    _invoke(runner, ["paper", "add", _URL], env)
    pid = lib.list_papers()[0].id
    r = _invoke(runner, ["paper", "show", pid], env)
    assert r.exit_code == 0
    assert "Attention" in r.output


def test_paper_tag(runner, env, lib):
    _invoke(runner, ["paper", "add", _URL], env)
    pid = lib.list_papers()[0].id
    r = _invoke(runner, ["paper", "tag", pid, "nlp"], env)
    assert r.exit_code == 0
    paper = PaperLibrary(lib._dir).get_paper(pid)  # fresh instance reads disk
    assert any(t.name == "nlp" for t in paper.tags)


def test_paper_status_change(runner, env, lib):
    _invoke(runner, ["paper", "add", _URL], env)
    pid = lib.list_papers()[0].id
    r = _invoke(runner, ["paper", "status", pid, "reading"], env)
    assert r.exit_code == 0
    paper = PaperLibrary(lib._dir).get_paper(pid)
    assert paper.status == "reading"


def test_paper_done(runner, env, lib):
    _invoke(runner, ["paper", "add", _URL], env)
    pid = lib.list_papers()[0].id
    _invoke(runner, ["paper", "status", pid, "reading"], env)
    r = _invoke(runner, ["paper", "done", pid, "--note", "great paper"], env)
    assert r.exit_code == 0
    paper = PaperLibrary(lib._dir).get_paper(pid)
    assert paper.status == "read"


def test_paper_export_bibtex(runner, env, lib, tmp_path):
    _invoke(runner, ["paper", "add", _URL], env)
    pid = lib.list_papers()[0].id
    out = str(tmp_path / "refs.bib")
    r = _invoke(runner, ["paper", "export", pid, "--format", "bibtex", "--output", out], env)
    assert r.exit_code == 0
    content = (tmp_path / "refs.bib").read_text()
    assert "@" in content
    assert "1706.03762" in content or "Vaswani" in content


def test_paper_remove(runner, env, lib):
    _invoke(runner, ["paper", "add", _URL], env)
    pid = lib.list_papers()[0].id
    r = _invoke(runner, ["paper", "remove", pid], env)
    assert r.exit_code == 0
    assert PaperLibrary(lib._dir).list_papers() == []


# ── Persistence across instances ─────────────────────────────────────────────


def test_library_persistence(tmp_path, lib):
    """Data written by one PaperLibrary instance is visible to a fresh one."""
    from openseed.models.paper import Paper

    p = Paper(title="Persist Test", abstract="test abstract", arxiv_id="2000.00001")
    lib.add_paper(p)

    lib2 = PaperLibrary(lib._dir)
    papers = lib2.list_papers()
    assert len(papers) == 1
    assert papers[0].arxiv_id == "2000.00001"


# ── Watch lifecycle ───────────────────────────────────────────────────────────


def test_watch_add_and_list(runner, env):
    r = _invoke(runner, ["paper", "watch", "add", "diffusion models"], env)
    assert r.exit_code == 0
    r2 = _invoke(runner, ["paper", "watch", "list"], env)
    assert r2.exit_code == 0
    assert "diffusion" in r2.output


def test_watch_remove(runner, env, lib):
    _invoke(runner, ["paper", "watch", "add", "diffusion models"], env)
    wid = lib.list_watches()[0].id
    r = _invoke(runner, ["paper", "watch", "remove", wid], env)
    assert r.exit_code == 0
    assert PaperLibrary(lib._dir).list_watches() == []  # fresh instance reads disk


# ── Research session lifecycle ────────────────────────────────────────────────


def test_research_list_empty(runner, env):
    r = _invoke(runner, ["research", "list"], env)
    assert r.exit_code == 0
    assert "No research sessions" in r.output


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_init(runner, env):
    r = _invoke(runner, ["init"], env)
    assert r.exit_code == 0
    assert "initialized" in r.output.lower()


def test_paper_list_empty(runner, env):
    r = _invoke(runner, ["paper", "list"], env)
    assert r.exit_code == 0
    assert "No papers found" in r.output


def test_paper_show_not_found(runner, env):
    r = _invoke(runner, ["paper", "show", "nonexistent"], env)
    assert r.exit_code != 0


def test_experiment_list_empty(runner, env):
    r = _invoke(runner, ["experiment", "list"], env)
    assert r.exit_code == 0
    assert "No experiments found" in r.output
