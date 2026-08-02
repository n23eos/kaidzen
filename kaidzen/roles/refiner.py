"""Refiner: переписывает идею по фактам Researcher и критике Judge."""
from __future__ import annotations

from pydantic import BaseModel, Field

from kaidzen.candidate import Candidate
from kaidzen.state import ChangelogEntry

EFFORT = "high"
# Refiner пишет самый длинный ответ цикла: полный текст идеи плюс changelog.
# Бюджет общий с размышлением, поэтому запас больше, чем у остальных ролей.
MAX_TOKENS = 32000
NO_CRITIQUE = "(первая итерация, критики нет)"


class RefinerOutput(BaseModel):
    idea_text: str
    changelog: list[ChangelogEntry] = Field(default_factory=list)


def run_refiner(llm, candidate: Candidate, *, idea_text: str,
                findings_json: str, critique: list[str]) -> RefinerOutput:
    crit = "\n".join(f"- {c}" for c in critique) or NO_CRITIQUE
    user = (f"Текущая версия идеи:\n\n{idea_text}\n\n"
            f"Свежие находки Researcher (JSON):\n{findings_json}\n\n"
            f"Критика Judge с прошлой итерации:\n{crit}")
    return llm.structured(model=candidate.config.models["refiner"],
                          system=candidate.prompts["refiner"], user=user,
                          schema=RefinerOutput, effort=EFFORT,
                          max_tokens=MAX_TOKENS)
