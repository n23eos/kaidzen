"""Слепой судья: сравнивает два отчёта по существу и ничего больше не знает.

Сигнатура `run_meta_judge` намеренно бедна: кроме двух текстов, ей нечего
передать. Любой дополнительный аргумент — id кандидата, диагноз, описание
правок, пометка «этот был раньше» — превратил бы оценку результата в оценку
намерения, а весь смысл мета-лупа в том, чтобы ловить правки, которые звучат
убедительно и измеряются хуже.

Второй параметр — `MetaConfig`, то есть транспорт и имя модели самого судьи.
О сравниваемых кандидатах он не знает ничего, поэтому слепоту не нарушает.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from kaidzen.roles.meta import MetaConfig, load_meta_prompt

# сравнение по фиксированным критериям должно быть воспроизводимым
EFFORT = "low"
PROMPT_NAME = "meta_judge"


class Comparison(BaseModel):
    winner: Literal["A", "B", "tie"]
    reason: str


def run_meta_judge(backend, meta: MetaConfig, *, report_a: str,
                   report_b: str) -> Comparison:
    user = (f"ОТЧЁТ A:\n\n{report_a}\n\n"
            f"---\n\n"
            f"ОТЧЁТ B:\n\n{report_b}\n\n"
            f"---\n\n"
            f"Какой отчёт лучше по существу? Допустим ответ «ничья» (tie).")
    return backend.structured(model=meta.judge_model,
                              system=load_meta_prompt(PROMPT_NAME), user=user,
                              schema=Comparison, effort=EFFORT)
