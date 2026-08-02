"""Сборка report.md из финального RunState.

Модуль намеренно почти без LLM: единственная генерируемая часть (executive
summary) приходит извне параметром summary_text, а сам билдер — чистая
функция без файлового I/O и без мутации переданного состояния.
"""
from __future__ import annotations

from kaidzen.state import Assumption, Fact, RunState, Version

# сколько первых записей changelog показывать в эволюции версий —
# дальше идут технические детали, не нужные для обзора истории
MAX_CHANGELOG_ENTRIES_IN_EVOLUTION = 2

DASH = "-"


def _escape_cell(value: str) -> str:
    """Экранирует '|' в значении ячейки, чтобы не сломать разметку таблицы."""
    return value.replace("|", "\\|")


def _render_fact(fact: Fact) -> str:
    link_text = fact.source_title or fact.source_url
    return f"[{link_text}]({fact.source_url}): {fact.claim}"


def _render_facts_cell(facts: list[Fact]) -> str:
    if not facts:
        return DASH
    return "; ".join(_render_fact(f) for f in facts)


def _scored_versions(state: RunState) -> list[Version]:
    """Не откаченные версии с оценкой судьи, в порядке появления."""
    return [v for v in state.versions if not v.rolled_back and v.judge is not None]


def _render_header(state: RunState, summary_text: str) -> str:
    usage = state.api_usage
    lines = [
        f"# Отчёт по прогону {state.run_id}",
        "",
        f"- Кандидат: {state.candidate_id}",
        f"- Итераций: {state.iteration}",
        f"- Причина остановки: {state.stop_reason or DASH}",
        f"- Использование API: {usage.input_tokens} входных токенов, "
        f"{usage.output_tokens} выходных токенов, {usage.web_searches} веб-поисков",
        "",
        "## Executive summary",
        "",
        summary_text,
    ]
    return "\n".join(lines)


def _render_final_idea(state: RunState) -> str:
    return "\n".join([
        "## Финальная версия идеи",
        "",
        state.current_idea_text(),
    ])


def _render_rubric(state: RunState) -> str:
    scored = _scored_versions(state)
    lines = ["## Оценки по рубрике", ""]
    if not scored:
        lines.append("Ни одна версия не была оценена судьёй.")
        return "\n".join(lines)

    first, final = scored[0], scored[-1]
    axes = sorted(first.judge.scores.keys() | final.judge.scores.keys())

    lines.append(f"| Ось | v{first.n} | v{final.n} |")
    lines.append("| --- | --- | --- |")
    for axis in axes:
        first_score = first.judge.scores.get(axis)
        final_score = final.judge.scores.get(axis)
        first_cell = f"{first_score:.1f}" if first_score is not None else DASH
        final_cell = f"{final_score:.1f}" if final_score is not None else DASH
        lines.append(f"| {_escape_cell(axis)} | {first_cell} | {final_cell} |")
    lines.append(f"| total | {first.judge.total:.1f} | {final.judge.total:.1f} |")
    return "\n".join(lines)


def _unverified(state: RunState) -> list[Assumption]:
    return [a for a in state.assumptions if a.status == "unverified"]


def _assumptions_summary(state: RunState) -> str:
    """Одна строка над таблицей: сколько допущений реально закрыто.

    Прогон, не закрывший ни одного допущения, должен быть виден сразу, а не
    вычисляться пользователем глазами по колонке «статус».
    """
    total = len(state.assumptions)
    if not total:
        return "Реестр допущений пуст."
    open_count = len(_unverified(state))
    if open_count == total:
        return ("**Ни одно допущение не проверено: прогон не закрыл фактами "
                f"ни одного из {total}.**")
    return (f"Закрыто {total - open_count} из {total} допущений; "
            f"осталось непроверенных: {open_count}.")


def _render_assumptions(state: RunState) -> str:
    lines = [
        "## Допущения",
        "",
        _assumptions_summary(state),
        "",
        "| id | текст | критичность | статус | факты |",
        "| --- | --- | --- | --- | --- |",
    ]
    for a in state.assumptions:
        lines.append(
            f"| {_escape_cell(a.id)} | {_escape_cell(a.text)} | "
            f"{a.criticality} | {a.status} | "
            f"{_escape_cell(_render_facts_cell(a.facts))} |"
        )
    return "\n".join(lines)


def _render_changelog_entry(entry) -> str:
    return f"{entry.change} ({entry.reason})"


def _render_evolution(state: RunState) -> str:
    lines = ["## Эволюция версий", ""]
    for v in state.versions:
        score = f"{v.judge.total:.1f}" if v.judge is not None else DASH
        marker = " [ОТКАЧЕНА]" if v.rolled_back else ""
        entries = v.changelog[:MAX_CHANGELOG_ENTRIES_IN_EVOLUTION]
        changes = "; ".join(_render_changelog_entry(e) for e in entries) or DASH
        lines.append(f"- v{v.n}{marker}: оценка {score} — {changes}")
    if not state.versions:
        lines.append("Версий пока нет.")
    return "\n".join(lines)


def _render_next_steps(state: RunState) -> str:
    """Открытые вопросы прогона: сначала непроверенное, потом непроверяемое.

    Непроверенные допущения идут первыми: «untestable» — честный результат
    работы, а «unverified» — это работа, которая НЕ сделана, и говорить в
    таком прогоне «всё проверено» значит врать пользователю.
    """
    lines = ["## Next steps", ""]
    if not state.assumptions:
        lines.append("Реестр допущений пуст — проверять было нечего.")
        return "\n".join(lines)

    unverified = _unverified(state)
    untestable = [a for a in state.assumptions if a.status == "untestable"]
    if not unverified and not untestable:
        lines.append(
            "Все допущения проверены по источникам — экспериментов в реальном мире не требуется."
        )
        return "\n".join(lines)

    if unverified:
        lines.append("Эти допущения остались непроверенными — прогон не закрыл "
                     "их фактами:")
        lines.append("")
        lines.extend(f"- [{a.id}] {a.text}" for a in unverified)
        if untestable:
            lines.append("")
    if untestable:
        lines.append("Эти допущения можно проверить только реальным экспериментом:")
        lines.append("")
        lines.extend(f"- [{a.id}] {a.text}" for a in untestable)
    return "\n".join(lines)


def build_report(state: RunState, *, summary_text: str = "") -> str:
    """Собирает markdown-отчёт из финального состояния прогона.

    Чистая функция: не читает и не пишет файлы, не меняет state.
    """
    sections = [
        _render_header(state, summary_text),
        _render_final_idea(state),
        _render_rubric(state),
        _render_assumptions(state),
        _render_evolution(state),
        _render_next_steps(state),
    ]
    return "\n\n".join(sections) + "\n"
