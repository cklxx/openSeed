"""Experiment tracking models."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from openseed.models.paper import Tag


class ExperimentRun(BaseModel):
    """Single execution of an experiment."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    status: Literal["running", "completed", "failed"] = "running"
    metrics: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""


class Experiment(BaseModel):
    """Experiment linked to a paper."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str
    paper_id: str
    repo_url: str | None = None
    local_path: str | None = None
    description: str = ""
    runs: list[ExperimentRun] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tags: list[Tag] = Field(default_factory=list)
