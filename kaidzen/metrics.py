"""Объективные метрики прогона: считаются кодом, не моделью.

Мета-луп сравнивает отчёты моделью, и модель можно уговорить красивым текстом.
Эти числа уговорить нельзя — они и решают, прошёл ли челленджер Gate.
"""
from __future__ import annotations

from pydantic import BaseModel

from kaidzen.orchestrator import CLOSED_STATUSES
from kaidzen.state import RunState

HTTP_PREFIXES = ("http://", "https://")


class RunMetrics(BaseModel):
    runs: int = 1
    high_total: int = 0
    high_closed: int = 0
    assumptions_closed_rate: float = 0.0
    partial_rate: float = 0.0
    facts_total: int = 0
    facts_with_sources: int = 0
    grounded_changelog_rate: float = 0.0
    iterations: int = 0
    stop_reason: str | None = None
    hit_iteration_limit: bool = False
    rollbacks: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    web_searches: int = 0


def _rate(part: int, whole: int) -> float:
    return part / whole if whole else 0.0


def output_tokens_per_run(m: RunMetrics) -> float:
    """Расход на один прогон, а не на пачку.

    Кандидаты сравниваются по разному числу успешных прогонов (упавший прогон
    в агрегат не попадает), и сумма токенов сама по себе сравнивала бы не
    жадность кандидата, а размер его выборки.
    """
    return m.output_tokens / m.runs if m.runs else float(m.output_tokens)


def run_metrics(state: RunState) -> RunMetrics:
    """Метрики одного прогона. Состояние только читается, не меняется."""
    high = [a for a in state.assumptions if a.criticality == "high"]
    closed = [a for a in high if a.status in CLOSED_STATUSES]
    # partial сознательно считается по всем допущениям, а не только high:
    # хеджирование Researcher — это поведение роли, а не свойство критичности
    partial = [a for a in state.assumptions if a.status == "partial"]
    facts = [f for a in state.assumptions for f in a.facts]
    entries = [e for v in state.versions for e in v.changelog]
    grounded = [e for e in entries if e.grounded_in]
    return RunMetrics(
        high_total=len(high),
        high_closed=len(closed),
        assumptions_closed_rate=_rate(len(closed), len(high)),
        partial_rate=_rate(len(partial), len(state.assumptions)),
        facts_total=len(facts),
        facts_with_sources=sum(
            1 for f in facts if f.source_url.startswith(HTTP_PREFIXES)),
        grounded_changelog_rate=_rate(len(grounded), len(entries)),
        iterations=state.iteration,
        stop_reason=state.stop_reason,
        hit_iteration_limit=state.stop_reason == "max_iterations",
        rollbacks=state.rollbacks,
        input_tokens=state.api_usage.input_tokens,
        output_tokens=state.api_usage.output_tokens,
        web_searches=state.api_usage.web_searches,
    )


def aggregate(items: list[RunMetrics]) -> RunMetrics:
    """Среднее по прогонам одного кандидата; счётчики — суммой.

    Доли усредняются, а не считаются по сумме: кандидат оценивается по
    среднему поведению на идеях бенчмарка, иначе один многословный прогон
    перевесил бы все остальные.
    """
    if not items:
        return RunMetrics(runs=0)
    n = len(items)

    def mean(attr: str) -> float:
        return sum(getattr(m, attr) for m in items) / n

    def total(attr: str) -> int:
        return sum(getattr(m, attr) for m in items)

    return RunMetrics(
        runs=n,
        high_total=total("high_total"), high_closed=total("high_closed"),
        assumptions_closed_rate=mean("assumptions_closed_rate"),
        partial_rate=mean("partial_rate"),
        facts_total=total("facts_total"),
        facts_with_sources=total("facts_with_sources"),
        grounded_changelog_rate=mean("grounded_changelog_rate"),
        iterations=total("iterations"),
        # общая причина остановки у пачки прогонов бессмысленна
        stop_reason=None,
        hit_iteration_limit=any(m.hit_iteration_limit for m in items),
        rollbacks=total("rollbacks"),
        input_tokens=total("input_tokens"),
        output_tokens=total("output_tokens"),
        web_searches=total("web_searches"),
    )
