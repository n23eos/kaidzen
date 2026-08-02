"""Researcher: проверяет допущения веб-поиском и возвращает факты с источниками."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from kaidzen.candidate import Candidate
from kaidzen.state import Assumption, Fact

TEMPERATURE = 0.2
MAX_SEARCHES_PER_CALL = 8


class ResearchFinding(BaseModel):
    assumption_id: str
    verdict: Literal["confirmed", "refuted", "partial", "untestable"]
    facts: list[Fact] = Field(default_factory=list)
    notes: str = ""


class ResearcherOutput(BaseModel):
    findings: list[ResearchFinding]


def run_researcher(llm, candidate: Candidate, *, idea_text: str,
                   assumptions: list[Assumption]) -> ResearcherOutput:
    listing = "\n".join(f"- {a.id} [{a.criticality}]: {a.text}" for a in assumptions)
    user = (f"Текущая версия идеи:\n\n{idea_text}\n\n"
            f"Проверь эти допущения веб-поиском:\n{listing}\n\n"
            f"Фокус поиска: {candidate.config.researcher_focus}")
    return llm.structured(model=candidate.config.models["researcher"],
                          system=candidate.prompts["researcher"], user=user,
                          schema=ResearcherOutput, temperature=TEMPERATURE,
                          web_search=True, max_searches=MAX_SEARCHES_PER_CALL)
