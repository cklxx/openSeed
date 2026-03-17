"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from openseed.config import OpenSeedConfig
from openseed.models.experiment import Experiment
from openseed.models.paper import Author, Paper, Tag
from openseed.storage.library import PaperLibrary


@pytest.fixture
def tmp_library(tmp_path: Path) -> PaperLibrary:
    """Create a temporary paper library."""
    return PaperLibrary(tmp_path / "library")


@pytest.fixture
def sample_paper() -> Paper:
    """Create a sample paper for testing."""
    return Paper(
        id="test123",
        title="Attention Is All You Need",
        authors=[
            Author(name="Ashish Vaswani", affiliation="Google Brain"),
            Author(name="Noam Shazeer"),
        ],
        abstract="We propose a new simple network architecture, the Transformer.",
        arxiv_id="1706.03762",
        url="https://arxiv.org/abs/1706.03762",
        tags=[Tag(name="transformers"), Tag(name="NLP")],
    )


@pytest.fixture
def sample_experiment() -> Experiment:
    """Create a sample experiment for testing."""
    return Experiment(
        id="exp123",
        name="Reproduce attention results",
        paper_id="test123",
        repo_url="https://github.com/example/attention",
        description="Reproducing key results from the paper.",
    )


@pytest.fixture
def config(tmp_path: Path) -> OpenSeedConfig:
    """Create a test config with temporary directories."""
    return OpenSeedConfig(
        library_dir=tmp_path / "library",
        config_dir=tmp_path / "config",
    )
