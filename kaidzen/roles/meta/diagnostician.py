"""Диагност: по метрикам и отчётам называет слабые места набора инструкций."""
from __future__ import annotations

import json

from pydantic import BaseModel

from kaidzen.candidate import Candidate
from kaidzen.metrics import RunMetrics
from kaidzen.roles.meta import load_meta_prompt, meta_model

# поиск причины по числам и текстам — задача на рассуждение, не на пересказ
EFFORT = "high"
PROMPT_NAME = "diagnostician"
JSON_INDENT = 2


class Diagnosis(BaseModel):
    weaknesses: list[str]
    hypotheses: list[str]


def run_diagnostician(backend, candidate: Candidate, *, metrics: RunMetrics,
                      reports: list[str]) -> Diagnosis:
    """Метрики идут числами, отчёты — целиком: гипотеза обязана опираться на оба."""
    numbers = json.dumps(metrics.model_dump(), ensure_ascii=False,
                         indent=JSON_INDENT)
    texts = "\n\n".join(f"### Отчёт {i}\n\n{text}"
                        for i, text in enumerate(reports, start=1))
    user = (f"Агрегированные метрики прогонов:\n\n{numbers}\n\n"
            f"Отчёты этих прогонов:\n\n{texts}\n\n"
            f"Назови слабые места и 2–3 гипотезы улучшения. "
            f"Каждая гипотеза — одна роль, одно конкретное изменение, "
            f"одна метрика в обоснование.")
    return backend.structured(model=meta_model(candidate),
                              system=load_meta_prompt(PROMPT_NAME), user=user,
                              schema=Diagnosis, effort=EFFORT)
