"""Мост между поколением и журналом эволюции.

Отдельный модуль, а не часть `evolve.py`: оркестратор и без того самый крупный
файл проекта, а здесь одна ясная задача — превратить итоги поколения в записи
журнала и обратно достать из журнала выжимку для мета-ролей.

Типы поколения (`EvolveState`, `GenerationRecord`, `CandidateRecord`) берутся
только для аннотаций: импорт `evolve` живёт под TYPE_CHECKING, поэтому кольца
импортов не возникает — `evolve` зависит от этого модуля, но не наоборот.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from kaidzen.evolution_log import (OUTCOME_PROMOTED, OUTCOME_REJECTED,
                                   OUTCOME_UNSTABLE, EvolutionRecord,
                                   append_record, load_records)
from kaidzen.metrics import RunMetrics, output_tokens_per_run

if TYPE_CHECKING:      # только для аннотаций: в рантайме импорта нет
    from kaidzen.evolve import CandidateRecord, EvolveState, GenerationRecord

# статусы кандидата повторены здесь, а не взяты из `evolve`: импорт оттуда
# замкнул бы кольцо. Совпадение строк со значениями `outcome` — совпадение, и
# перевод одного в другое сделан явно (`outcome_of`), чтобы расхождение
# статусов и исходов не проехало молча
STATUS_PROMOTED = "promoted"
STATUS_UNSTABLE = "unstable"

# на сколько знаков округляем доли в дельтах: журнал читает человек и модель,
# и хвост из пятнадцати знаков не значит ничего ни для того, ни для другого
DELTA_PRECISION = 4
TOKENS_PRECISION = 1


def load_memory(root: Path, state: EvolveState) -> list[EvolutionRecord]:
    """Журнал домена. Лежит рядом с кандидатами, а не в каталоге прогона:
    знание переживает прогон, артефакты прогона — нет."""
    return load_records(root, state.domain)


def remember(root: Path, state: EvolveState, record: EvolutionRecord) -> None:
    append_record(root, state.domain, record)


def log_generation(root: Path, state: EvolveState,
                   gen: GenerationRecord) -> None:
    """Итоги поколения в журнал — после того, как исходы стали окончательными.

    Именно после смены чемпиона: до неё челленджер, прошедший Gate, но
    проигравший тай-брейк, ещё числится promoted, и журнал запомнил бы как
    принятую правку, которая никуда не поехала.
    """
    for record in gen.challengers:
        remember(root, state, build_record(
            state, gen, candidate_id=record.candidate_id,
            roles=record.roles_touched, hypothesis=record.rationale,
            outcome=outcome_of(record), gate_reason=record.gate_reason,
            win_rate=record.win_rate,
            metrics_delta=metrics_delta(gen.champion.metrics, record.metrics),
            comparable_ideas=comparable_ideas(gen, record)))


def build_record(state: EvolveState, gen: GenerationRecord, *,
                 candidate_id: str, roles: list[str], hypothesis: str,
                 outcome: str, gate_reason: str = "",
                 win_rate: float | None = None,
                 metrics_delta: dict[str, float] | None = None,
                 comparable_ideas: int = 0) -> EvolutionRecord:
    return EvolutionRecord(
        evolve_id=state.evolve_id, generation=gen.number,
        candidate_id=candidate_id, parent_id=gen.champion_id,
        hypothesis=hypothesis, roles_touched=roles, outcome=outcome,
        gate_reason=gate_reason, win_rate=win_rate,
        metrics_delta=metrics_delta or {}, comparable_ideas=comparable_ideas)


def outcome_of(record: CandidateRecord) -> str:
    if record.status == STATUS_PROMOTED:
        return OUTCOME_PROMOTED
    if record.status == STATUS_UNSTABLE:
        return OUTCOME_UNSTABLE
    return OUTCOME_REJECTED


def metrics_delta(champion: RunMetrics | None,
                  challenger: RunMetrics | None) -> dict[str, float]:
    """Насколько челленджер разошёлся с чемпионом того же поколения.

    Токены сравниваются на прогон, а не на пачку: у кандидатов разное число
    успешных прогонов, и сумма сравнивала бы размер выборки, а не жадность.
    """
    if champion is None or challenger is None:
        return {}
    return {
        "assumptions_closed_rate": round(challenger.assumptions_closed_rate
                                         - champion.assumptions_closed_rate,
                                         DELTA_PRECISION),
        "partial_rate": round(challenger.partial_rate - champion.partial_rate,
                              DELTA_PRECISION),
        "high_closed": float(challenger.high_closed - champion.high_closed),
        "output_tokens": round(output_tokens_per_run(challenger)
                               - output_tokens_per_run(champion),
                               TOKENS_PRECISION),
    }


def comparable_ideas(gen: GenerationRecord, record: CandidateRecord) -> int:
    """На скольких идеях сравнение вообще состоялось: упавший прогон любой из
    сторон выбрасывает идею из попарок, и win_rate без этого числа не читается."""
    champion_ideas = {run.idea for run in gen.champion.runs if run.ok}
    return sum(1 for run in record.runs if run.ok and run.idea in champion_ideas)
