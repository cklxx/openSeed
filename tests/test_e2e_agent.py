"""E2E agent command tests: real CLI + real ArXiv metadata, mocked Claude _ask."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from openseed.cli.main import cli
from openseed.models.paper import Paper
from openseed.storage.library import PaperLibrary

pytestmark = pytest.mark.integration

_URL = "https://arxiv.org/abs/1706.03762"
_SUMMARY = "## Key Contributions\n- Transformer\n\n**Relevance Score:** 9/10"
_TAGS = "transformers, attention, nlp, deep-learning, neural-networks"
_REVIEW = "## Review\nHighly influential. Self-attention is a key innovation."
_SYNTHESIS = "## Shared Themes\n- Attention\n\n## Synthesis\nBoth influential."
_VISUALS = '{"pipeline": ["Encode", "Attend", "Decode"], "metrics": []}'


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def env(tmp_path):
    return {
        "OPENSEED_CONFIG_DIR": str(tmp_path),
        "ANTHROPIC_API_KEY": "sk-test-fake-key",
    }


@pytest.fixture
def lib(tmp_path):
    return PaperLibrary(tmp_path / "library")


def _invoke(runner, args, env, **kw):
    return runner.invoke(cli, args, env=env, catch_exceptions=False, **kw)


def _add_real_paper(runner, env):
    """Fetches real metadata from ArXiv, no Claude calls."""
    return _invoke(runner, ["paper", "add", _URL], env)


def _paper_id(lib):
    papers = lib.list_papers()
    assert papers, "No papers in library"
    return papers[0].id


# ── summarize ────────────────────────────────────────────────────────────────


@patch("openseed.agent.reader._ask")
def test_summarize_saves_to_library(mock_ask, runner, env, lib):
    mock_ask.return_value = _SUMMARY
    _add_real_paper(runner, env)
    pid = _paper_id(lib)

    r = _invoke(runner, ["agent", "summarize", pid], env)

    assert r.exit_code == 0
    assert "Key Contributions" in r.output
    paper = PaperLibrary(lib._dir).get_paper(pid)
    assert paper.summary == _SUMMARY


@patch("openseed.agent.reader._ask")
def test_summarize_writes_markdown_file(mock_ask, runner, env, lib, tmp_path):
    mock_ask.return_value = _SUMMARY
    _add_real_paper(runner, env)
    pid = _paper_id(lib)
    _invoke(runner, ["agent", "summarize", pid], env)

    summaries = list((tmp_path / "summaries").glob("*.md"))
    assert len(summaries) == 1
    assert "Key Contributions" in summaries[0].read_text()


@patch("openseed.agent.reader._ask")
def test_summarize_cn(mock_ask, runner, env, lib):
    mock_ask.return_value = "## 核心贡献\n- Transformer\n\n**相关性评分:** 9/10"
    _add_real_paper(runner, env)
    pid = _paper_id(lib)

    r = _invoke(runner, ["agent", "summarize", pid, "--cn"], env)

    assert r.exit_code == 0
    assert mock_ask.called


# ── review ───────────────────────────────────────────────────────────────────


@patch("openseed.agent.assistant._ask")
def test_review_output(mock_ask, runner, env, lib):
    mock_ask.return_value = _REVIEW
    _add_real_paper(runner, env)
    pid = _paper_id(lib)

    r = _invoke(runner, ["agent", "review", pid], env)

    assert r.exit_code == 0
    assert "Review" in r.output
    assert mock_ask.called


# ── synthesize ───────────────────────────────────────────────────────────────


@patch("openseed.agent.reader._ask")
def test_synthesize_two_papers(mock_ask, runner, env, lib, tmp_path):
    mock_ask.return_value = _SYNTHESIS
    p1 = Paper(title="Paper A", abstract="attention is useful", arxiv_id="1000.00001")
    p2 = Paper(title="Paper B", abstract="transformer model", arxiv_id="1000.00002")
    lib.add_paper(p1)
    lib.add_paper(p2)

    r = _invoke(runner, ["agent", "synthesize", p1.id, p2.id], env)

    assert r.exit_code == 0
    assert mock_ask.called
    syntheses = list((tmp_path / "summaries").glob("synthesis_*.md"))
    assert len(syntheses) == 1


# ── ask alias ────────────────────────────────────────────────────────────────


@patch("openseed.agent.assistant._ask")
def test_ask_toplevel_alias(mock_ask, runner, env):
    mock_ask.return_value = "Transformers use self-attention."
    r = _invoke(runner, ["ask", "What is a transformer?"], env)
    assert r.exit_code == 0
    assert mock_ask.called


# ── codegen ──────────────────────────────────────────────────────────────────


@patch("openseed.agent.reader._ask")
def test_codegen_writes_file(mock_ask, runner, env, lib, tmp_path):
    mock_ask.return_value = "import torch\n# Transformer implementation\n"
    _add_real_paper(runner, env)
    pid = _paper_id(lib)
    out = str(tmp_path / "exp.py")

    r = _invoke(runner, ["agent", "codegen", pid, "--output", out], env)

    assert r.exit_code == 0
    assert (tmp_path / "exp.py").exists()
    content = (tmp_path / "exp.py").read_text()
    assert "torch" in content
    paper = PaperLibrary(lib._dir).get_paper(pid)
    assert paper.experiment_path == out


# ── pipeline ─────────────────────────────────────────────────────────────────


@patch("openseed.agent.reader._ask")
def test_pipeline_adds_and_analyzes(mock_ask, runner, env, lib):
    """pipeline: real ArXiv search result selected, Claude calls mocked."""
    # _ask is called for: search markdown, summarize, tag, visuals — return plausible strings
    _arxiv_id = _URL.split("/")[-1]
    _search_row = (
        f"| {_arxiv_id} | Attention Is All You Need | Vaswani et al."
        " | 2017 | 120000 | Transformer |"
    )
    mock_ask.side_effect = [
        _search_row,  # search result markdown with one known ArXiv ID
        _SUMMARY,  # summarize
        _TAGS,  # auto_tag
        _VISUALS,  # extract_paper_visuals
    ]
    r = runner.invoke(
        cli,
        ["agent", "pipeline", "transformer attention", "--count", "5"],
        env=env,
        input="1\n",  # select first paper
        catch_exceptions=False,
    )
    assert r.exit_code == 0
