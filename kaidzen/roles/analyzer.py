"""Analyzer: раскладывает сырую идею на компоненты и реестр допущений."""
from __future__ import annotations

from pydantic import BaseModel, Field

from kaidzen.candidate import Candidate
from kaidzen.state import Assumption

EFFORT = "medium"


class AnalyzerOutput(BaseModel):
    problem: str
    audience: str
    mechanism: str
    assumptions: list[Assumption]
    unknowns: list[str] = Field(default_factory=list)


def run_analyzer(backend, candidate: Candidate, *, idea_text: str) -> AnalyzerOutput:
    user = (f"Идея для декомпозиции:\n\n{idea_text}\n\n"
            f"Домен: {candidate.config.domain}\n"
            f"Подсказки: {candidate.config.analyzer_hints}")
    return backend.structured(model=candidate.config.roles["analyzer"].model,
                              system=candidate.prompts["analyzer"], user=user,
                              schema=AnalyzerOutput, effort=EFFORT)
