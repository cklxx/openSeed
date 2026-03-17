"""Tests for CLI commands."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from openseed.cli.main import cli
from openseed.config import OpenSeedConfig


class TestCLI:
    def _invoke(self, args: list[str], tmp_path: Path) -> object:
        runner = CliRunner()
        config = OpenSeedConfig(
            library_dir=tmp_path / "library",
            config_dir=tmp_path / "config",
        )
        with patch("openseed.cli.main.load_config", return_value=config):
            return runner.invoke(cli, args)

    def test_version(self, tmp_path: Path) -> None:
        result = self._invoke(["--version"], tmp_path)
        assert result.exit_code == 0
        assert version("openseed") in result.output

    def test_init(self, tmp_path: Path) -> None:
        result = self._invoke(["init"], tmp_path)
        assert result.exit_code == 0
        assert "initialized" in result.output.lower()

    def test_paper_list_empty(self, tmp_path: Path) -> None:
        result = self._invoke(["paper", "list"], tmp_path)
        assert result.exit_code == 0
        assert "No papers found" in result.output

    def test_paper_add_and_list(self, tmp_path: Path) -> None:
        result = self._invoke(["paper", "add", "https://arxiv.org/abs/2301.00001"], tmp_path)
        assert result.exit_code == 0
        assert "Added" in result.output

        result = self._invoke(["paper", "list"], tmp_path)
        assert result.exit_code == 0
        assert "2301.00001" in result.output

    def test_paper_show_not_found(self, tmp_path: Path) -> None:
        result = self._invoke(["paper", "show", "nonexistent"], tmp_path)
        assert result.exit_code != 0

    def test_experiment_list_empty(self, tmp_path: Path) -> None:
        result = self._invoke(["experiment", "list"], tmp_path)
        assert result.exit_code == 0
        assert "No experiments found" in result.output
