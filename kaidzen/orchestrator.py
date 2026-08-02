"""Оркестратор цикла: критерии остановки, отбор допущений, вердикт Judge."""
from __future__ import annotations

from kaidzen.candidate import LoopConfig
from kaidzen.state import Assumption, JudgeResult, RunState

# два подряд отката = идея деградирует, дальше крутить бессмысленно
ROLLBACK_LIMIT = 2
# два подряд слабых прироста = плато
PLATEAU_STREAK = 2

# статусы, после которых допущение больше не проверяется;
# partial сюда НЕ входит — его ещё можно доуточнить
CLOSED_STATUSES = frozenset({"confirmed", "refuted", "untestable"})
OPEN_STATUS = "unverified"

# порядок отбора: сначала самые критичные допущения
CRITICALITY_ORDER = {"high": 0, "medium": 1, "low": 2}

STOP_MAX_ITERATIONS = "max_iterations"
STOP_DEGRADING = "degrading"
STOP_PLATEAU = "plateau"
STOP_ASSUMPTIONS_EXHAUSTED = "assumptions_exhausted"


def check_stop(state: RunState, loop: LoopConfig) -> str | None:
    """Причина остановки цикла или None. Порядок приоритетов важен:
    жёсткий лимит итераций перекрывает всё остальное, деградация — плато.
    """
    if state.iteration >= loop.max_iterations:
        return STOP_MAX_ITERATIONS
    if state.consecutive_rollbacks >= ROLLBACK_LIMIT:
        return STOP_DEGRADING
    if state.low_delta_streak >= PLATEAU_STREAK:
        return STOP_PLATEAU
    if _high_assumptions_closed(state.assumptions):
        return STOP_ASSUMPTIONS_EXHAUSTED
    return None


def _high_assumptions_closed(assumptions: list[Assumption]) -> bool:
    """Все критичные допущения закрыты. Пустой реестр критичных не считается."""
    high = [a for a in assumptions if a.criticality == "high"]
    if not high:
        return False
    return all(a.status in CLOSED_STATUSES for a in high)


def select_assumptions(assumptions: list[Assumption],
                       limit: int) -> list[Assumption]:
    """До limit непроверенных допущений: сначала high, затем medium, затем low.
    Внутри одной критичности порядок реестра сохраняется (sorted стабилен).
    """
    open_ones = [a for a in assumptions if a.status == OPEN_STATUS]
    ordered = sorted(open_ones, key=lambda a: CRITICALITY_ORDER[a.criticality])
    return ordered[:limit]


def apply_judge_verdict(state: RunState, judge: JudgeResult,
                        loop: LoopConfig) -> None:
    """Применяет вердикт к последней версии и обновляет счётчики остановки.

    При откате результат Judge к версии НЕ прикрепляется: версия выброшена,
    и её оценка не должна попадать в отчёт как оценка идеи.
    """
    version = state.versions[-1]
    if judge.verdict == "rollback":
        version.rolled_back = True
        state.rollbacks += 1
        state.consecutive_rollbacks += 1
        return

    version.judge = judge
    state.consecutive_rollbacks = 0
    if judge.delta_vs_previous < loop.plateau_threshold:
        state.low_delta_streak += 1
    else:
        state.low_delta_streak = 0
