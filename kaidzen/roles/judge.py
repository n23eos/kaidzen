"""Judge: оценивает новую версию по рубрике и сравнивает с предыдущей."""
from __future__ import annotations

from kaidzen.candidate import Candidate
from kaidzen.state import Assumption, JudgeResult

TEMPERATURE = 0.1


def run_judge(llm, candidate: Candidate, *, new_idea: str, previous_idea: str,
              assumptions: list[Assumption]) -> JudgeResult:
    rubric = "\n".join(f"- {axis}: {desc}"
                       for axis, desc in candidate.config.rubric.items())
    registry = "\n".join(f"- {a.id} [{a.status}]: {a.text}" for a in assumptions)
    # ВАЖНО: список изменений Refiner'а сюда НЕ передаётся — Judge должен
    # оценивать результат, а не рассказ об изменениях.
    user = (f"Рубрика (каждая ось 0–10):\n{rubric}\n\n"
            f"Реестр допущений:\n{registry}\n\n"
            f"ПРЕДЫДУЩАЯ версия идеи:\n{previous_idea}\n\n"
            f"НОВАЯ версия идеи:\n{new_idea}")
    return llm.structured(model=candidate.config.models["judge"],
                          system=candidate.prompts["judge"], user=user,
                          schema=JudgeResult, temperature=TEMPERATURE)
