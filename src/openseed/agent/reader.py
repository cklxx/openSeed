"""Claude-powered paper reader."""

from __future__ import annotations

import anthropic


class PaperReader:
    """Reads and analyzes papers using Claude."""

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

    def summarize_paper(self, text: str) -> str:
        """Generate a concise summary of a paper."""
        return self._ask(
            "You are a research paper summarizer. Provide a clear, concise summary.",
            f"Summarize the following paper:\n\n{text}",
        )

    def extract_key_findings(self, text: str) -> list[str]:
        """Extract key findings from a paper."""
        result = self._ask(
            "You are a research analyst. Extract key findings as a numbered list.",
            f"Extract the key findings from this paper:\n\n{text}",
        )
        return [line.strip() for line in result.strip().split("\n") if line.strip()]

    def generate_questions(self, text: str) -> list[str]:
        """Generate research questions based on a paper."""
        result = self._ask(
            "You are a research advisor. Generate insightful questions about this paper.",
            f"Generate research questions about:\n\n{text}",
        )
        return [line.strip() for line in result.strip().split("\n") if line.strip()]
