"""Judge: оценивает новую версию по рубрике и сравнивает с предыдущей."""
from __future__ import annotations

from kaidzen.candidate import Candidate
from kaidzen.state import Assumption, JudgeResult

# оценка по фиксированной рубрике должна быть воспроизводимой, а не изобретательной
EFFORT = "low"


def run_judge(llm, candidate: Candidate, *, new_idea: str, previous_idea: str,
              assumptions: list[Assumption]) -> JudgeResult:
    rubric = "\n".join(f"- {axis}: {desc}"
                       for axis, desc in candidate.config.rubric.items())
    registry = "\n".join(f"- {a.id} [{a.status}]: {a.text}" for a in assumptions)
    axes = ", ".join(candidate.config.rubric)
    # ВАЖНО: список изменений Refiner'а сюда НЕ передаётся — Judge должен
    # оценивать результат, а не рассказ об изменениях.
    user = (f"Рубрика (каждая ось 0–10):\n{rubric}\n\n"
            f"В поле scores верни ровно эти ключи: {axes} — "
            f"не переводи их и не добавляй своих.\n\n"
            f"Реестр допущений:\n{registry}\n\n"
            f"ПРЕДЫДУЩАЯ версия идеи:\n{previous_idea}\n\n"
            f"НОВАЯ версия идеи:\n{new_idea}")
    result = llm.structured(model=candidate.config.models["judge"],
                            system=candidate.prompts["judge"], user=user,
                            schema=JudgeResult, effort=EFFORT)
    _check_axes(result, candidate)
    return result


def _check_axes(result: JudgeResult, candidate: Candidate) -> None:
    """Ключи scores обязаны совпадать с рубрикой кандидата.

    Схема проверяет только диапазон и сумму, поэтому три оси вместо пяти или
    русские названия проходят валидацию. Так же молча исчезает обязательная
    ось groundedness — главная защита системы от полировки идеи в вакууме.
    """
    expected = set(candidate.config.rubric)
    actual = set(result.scores)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    raise ValueError(f"Judge вернул оси, не совпадающие с рубрикой: "
                     f"пропущены {missing}, лишние {unexpected}")
