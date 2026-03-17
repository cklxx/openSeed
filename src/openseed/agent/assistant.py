"""Research assistant powered by Claude."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anthropic

if TYPE_CHECKING:
    from openseed.models.paper import Paper


class ResearchAssistant:
    """General-purpose research assistant using Claude."""

    def __init__(self, model: str = "claude-sonnet-4-20250514") -> None:
        self._client = anthropic.Anthropic()
        self._model = model

    def _ask(self, system: str, prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def ask(self, question: str, context: str = "") -> str:
        """Ask a research question."""
        system = "You are a knowledgeable research assistant."
        prompt = question
        if context:
            prompt = f"Context:\n{context}\n\nQuestion: {question}"
        return self._ask(system, prompt)

    def review_paper(self, paper: Paper) -> str:
        """Generate a review of a paper."""
        text = f"Title: {paper.title}\n"
        if paper.authors:
            text += f"Authors: {', '.join(a.name for a in paper.authors)}\n"
        if paper.abstract:
            text += f"\nAbstract:\n{paper.abstract}\n"
        if paper.summary:
            text += f"\nSummary:\n{paper.summary}\n"

        return self._ask(
            "You are a peer reviewer. Provide a constructive review.",
            f"Review the following paper:\n\n{text}",
        )
