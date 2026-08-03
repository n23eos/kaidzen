"""Слепой судья: сравнивает два отчёта по существу и ничего больше не знает.

Сигнатура `run_meta_judge` намеренно бедна: кроме двух текстов, ей нечего
передать. Любой дополнительный аргумент — id кандидата, диагноз, описание
правок, пометка «этот был раньше» — превратил бы оценку результата в оценку
намерения, а весь смысл мета-лупа в том, чтобы ловить правки, которые звучат
убедительно и измеряются хуже.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from kaidzen.candidate import Candidate
from kaidzen.roles.meta import load_meta_prompt, meta_model

# сравнение по фиксированным критериям должно быть воспроизводимым
EFFORT = "low"
PROMPT_NAME = "meta_judge"


class Comparison(BaseModel):
    winner: Literal["A", "B", "tie"]
    reason: str


def run_meta_judge(backend, candidate: Candidate, *, report_a: str,
                   report_b: str) -> Comparison:
    user = (f"ОТЧЁТ A:\n\n{report_a}\n\n"
            f"---\n\n"
            f"ОТЧЁТ B:\n\n{report_b}\n\n"
            f"---\n\n"
            f"Какой отчёт лучше по существу? Допустим ответ «ничья» (tie).")
    return backend.structured(model=meta_model(candidate),
                              system=load_meta_prompt(PROMPT_NAME), user=user,
                              schema=Comparison, effort=EFFORT)
