"""Пускать ли челленджера в чемпионы.

Два независимых условия: выиграл слепые попарки и не просадил объективные
метрики. Первое проверяет модель, второе — арифметика. Второе главнее:
отчёт, который нравится судье, но закрывает меньше допущений, — регресс.
"""
from __future__ import annotations

from pydantic import BaseModel

from kaidzen.metrics import RunMetrics, output_tokens_per_run

MIN_WIN_RATE = 0.55          # доля побед в попарках
MAX_REGRESSION = 0.10        # относительное проседание метрики, которое терпим
MAX_REGISTRY_SHRINK = 0.20   # насколько можно ужать реестр допущений «бесплатно»
MAX_COST_GROWTH = 1.5        # во сколько раз можно подорожать по выходным токенам


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
    """Доля фактов с http(s)-ссылкой.

    Сейчас это всегда 1.0: валидатор Fact.source_url отбраковывает ссылку без
    схемы ещё при разборе ответа модели, поэтому факт без ссылки в RunState не
    доезжает. Проверка оставлена как страховка на случай, если валидатор
    когда-нибудь ослабят, но считать её защитой от выдуманных источников
    нельзя: правдоподобный, но несуществующий URL проходит и её, и валидатор.

    Настоящая защита от выдуманного источника — SearchNotPerformed в слое
    бэкендов: вызов с поиском, не сделавший ни одного запроса, не засчитывается.
    """
    return m.facts_with_sources / m.facts_total if m.facts_total else 0.0


def _gamed_the_denominator(champion: RunMetrics, challenger: RunMetrics) -> bool:
    """Доля выросла за счёт ужатого реестра, а не за счёт закрытых вопросов.

    Поймано на первом живом поколении: чемпион закрывал 4 допущения из 6,
    челленджер — 4 из 4. Доля скакнула с 0.67 до 1.00, хотя закрыто ровно
    столько же. Знаменатель задаёт Analyzer, то есть система сама, поэтому
    доля закрытых без абсолютного числа — метрика, которую можно нарисовать.

    Ужимать реестр не запрещено: короткий список по делу лучше длинного из
    домыслов. Но тогда закрытых должно стать больше в штуках, иначе
    улучшения не произошло.
    """
    if champion.high_total <= 0:
        return False
    shrink = (champion.high_total - challenger.high_total) / champion.high_total
    if shrink <= MAX_REGISTRY_SHRINK:
        return False
    return challenger.high_closed <= champion.high_closed


def _too_expensive(champion: RunMetrics, challenger: RunMetrics) -> bool:
    """Кандидат купил улучшение непропорциональной ценой.

    Просадка качества и удорожание блокируют промоцию одинаково: кандидат,
    ставший лучше втрое большей ценой, — не улучшение, а обмен. Такой обмен
    должен утверждать человек на чекпоинте, а не Gate молча.
    """
    before = output_tokens_per_run(champion)
    if before <= 0:
        return False
    return output_tokens_per_run(challenger) > before * MAX_COST_GROWTH


def _broken_metrics(champion: RunMetrics, challenger: RunMetrics) -> list[str]:
    checks = [
        ("assumptions_closed_rate", _regressed(champion.assumptions_closed_rate,
                                               challenger.assumptions_closed_rate)),
        ("high_closed", challenger.high_closed < champion.high_closed),
        ("реестр ужат без роста числа закрытых",
         _gamed_the_denominator(champion, challenger)),
        ("grounded_changelog_rate", _regressed(champion.grounded_changelog_rate,
                                               challenger.grounded_changelog_rate)),
        ("source_rate", _regressed(_source_rate(champion),
                                   _source_rate(challenger))),
        # partial растёт — значит цикл стал чаще хеджировать вместо вердикта
        ("partial_rate", _grew(champion.partial_rate, challenger.partial_rate)),
        (f"output_tokens вырос больше чем в {MAX_COST_GROWTH:g} раза",
         _too_expensive(champion, challenger)),
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
