"""Пускать ли челленджера в чемпионы.

Два независимых условия: выиграл слепые попарки и не просадил объективные
метрики. Первое проверяет модель, второе — арифметика. Второе главнее:
отчёт, который нравится судье, но закрывает меньше допущений, — регресс.
"""
from __future__ import annotations

from pydantic import BaseModel

from kaidzen.metrics import RunMetrics

MIN_WIN_RATE = 0.55          # доля побед в попарках
MAX_REGRESSION = 0.10        # относительное проседание метрики, которое терпим


class GateDecision(BaseModel):
    promote: bool
    reason: str


def _regressed(before: float, after: float) -> bool:
    """Метрика, которую плохо ронять (доля закрытых, доля со ссылками)."""
    if before <= 0:
        return False
    return (before - after) / before > MAX_REGRESSION


def _grew(before: float, after: float) -> bool:
    """Метрика, которую плохо растить (доля partial).

    Несимметрична `_regressed` намеренно: у нуля нет относительного роста,
    поэтому от нулевой базы сравниваем абсолютную величину.
    """
    if before <= 0:
        return after > MAX_REGRESSION
    return (after - before) / before > MAX_REGRESSION


def _source_rate(m: RunMetrics) -> float:
    return m.facts_with_sources / m.facts_total if m.facts_total else 0.0


def _broken_metrics(champion: RunMetrics, challenger: RunMetrics) -> list[str]:
    checks = [
        ("assumptions_closed_rate", _regressed(champion.assumptions_closed_rate,
                                               challenger.assumptions_closed_rate)),
        ("grounded_changelog_rate", _regressed(champion.grounded_changelog_rate,
                                               challenger.grounded_changelog_rate)),
        ("source_rate", _regressed(_source_rate(champion),
                                   _source_rate(challenger))),
        # partial растёт — значит цикл стал чаще хеджировать вместо вердикта
        ("partial_rate", _grew(champion.partial_rate, challenger.partial_rate)),
    ]
    return [name for name, bad in checks if bad]


def decide(*, champion: RunMetrics, challenger: RunMetrics,
           win_rate: float) -> GateDecision:
    if win_rate < MIN_WIN_RATE:
        return GateDecision(
            promote=False,
            reason=f"проиграл попарки: {win_rate:.0%} < {MIN_WIN_RATE:.0%}")
    broken = _broken_metrics(champion, challenger)
    if broken:
        return GateDecision(
            promote=False,
            reason=f"выиграл попарки ({win_rate:.0%}), "
                   f"но просадил: {', '.join(broken)}")
    return GateDecision(
        promote=True,
        reason=f"выиграл попарки ({win_rate:.0%}), метрики не просели")
