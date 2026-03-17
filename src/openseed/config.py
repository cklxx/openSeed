"""OpenSeed configuration."""

from __future__ import annotations

import tomllib
from pathlib import Path

import toml
from pydantic import BaseModel, Field

_DEFAULT_DIR = Path.home() / ".openseed"


class OpenSeedConfig(BaseModel):
    """Global configuration."""

    library_dir: Path = Field(default_factory=lambda: _DEFAULT_DIR / "library")
    config_dir: Path = Field(default_factory=lambda: _DEFAULT_DIR)
    default_model: str = "claude-sonnet-4-6"


def _config_path() -> Path:
    return _DEFAULT_DIR / "config.toml"


def load_config() -> OpenSeedConfig:
    """Load config from disk, returning defaults if missing."""
    path = _config_path()
    if path.exists():
        data = tomllib.loads(path.read_text())
        return OpenSeedConfig(**data)
    return OpenSeedConfig()


def save_config(config: OpenSeedConfig) -> None:
    """Persist config to disk."""
    path = _config_path()
    ensure_dirs(config)
    path.write_text(toml.dumps(config.model_dump(mode="json")))


def ensure_dirs(config: OpenSeedConfig | None = None) -> None:
    """Create required directories."""
    cfg = config or OpenSeedConfig()
    cfg.config_dir.mkdir(parents=True, exist_ok=True)
    cfg.library_dir.mkdir(parents=True, exist_ok=True)
