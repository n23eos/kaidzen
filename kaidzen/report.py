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

# закрытым считается допущение, доведённое до вердикта; `partial` — не закрыто,
# ровно как в orchestrator.CLOSED_STATUSES: расхождение здесь означало бы, что
# отчёт рапортует о проверке, которой цикл не признал
CLOSED_STATUSES = frozenset({"confirmed", "refuted", "untestable"})

# заголовок раздела с полными фактами; на него ссылается ячейка таблицы
FACTS_SECTION = "Факты и источники"

DASH = "-"


def _escape_cell(value: str) -> str:
    """Экранирует '|' в значении ячейки, чтобы не сломать разметку таблицы."""
    return value.replace("|", "\\|")


def _render_fact(fact: Fact) -> str:
    link_text = fact.source_title or fact.source_url
    return f"[{link_text}]({fact.source_url}): {fact.claim}"


def _render_facts_cell(facts: list[Fact], anchor: str) -> str:
    """В таблице — только счётчик и отсылка вниз.

    Разворачивать все факты прямо в ячейке нельзя: на реальном прогоне одно
    допущение набирает шесть фактов, и строка таблицы раздувается до нескольких
    тысяч символов, после чего таблицу невозможно читать.
    """
    if not facts:
        return DASH
    return f"{len(facts)} — см. «{anchor}»"


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

    def cell(version: Version, axis: str) -> str:
        score = version.judge.scores.get(axis)
        return f"{score:.1f}" if score is not None else DASH

    # одна оценённая версия — одна колонка: «v1 | v1» повторяет само себя
    # и создаёт видимость динамики там, где сравнивать ещё не с чем
    if first is final:
        lines.append(f"| Ось | v{first.n} |")
        lines.append("| --- | --- |")
        for axis in axes:
            lines.append(f"| {_escape_cell(axis)} | {cell(first, axis)} |")
        lines.append(f"| total | {first.judge.total:.1f} |")
        return "\n".join(lines)

    lines.append(f"| Ось | v{first.n} | v{final.n} |")
    lines.append("| --- | --- | --- |")
    for axis in axes:
        lines.append(f"| {_escape_cell(axis)} | {cell(first, axis)} | {cell(final, axis)} |")
    lines.append(f"| total | {first.judge.total:.1f} | {final.judge.total:.1f} |")
    return "\n".join(lines)


def _unverified(state: RunState) -> list[Assumption]:
    return [a for a in state.assumptions if a.status == "unverified"]


def _partial(state: RunState) -> list[Assumption]:
    return [a for a in state.assumptions if a.status == "partial"]


def _assumptions_summary(state: RunState) -> str:
    """Одна строка над таблицей: сколько допущений реально закрыто.

    Прогон, не закрывший ни одного допущения, должен быть виден сразу, а не
    вычисляться пользователем глазами по колонке «статус».
    """
    total = len(state.assumptions)
    if not total:
        return "Реестр допущений пуст."
    closed = sum(1 for a in state.assumptions if a.status in CLOSED_STATUSES)
    open_count = len(_unverified(state))
    partial_count = len(_partial(state))
    if not closed:
        return ("**Ни одно допущение не проверено: прогон не закрыл фактами "
                f"ни одного из {total}.**")
    tail = f"осталось непроверенных: {open_count}"
    if partial_count:
        # частично подтверждённое не закрыто и не «не тронуто» — своя корзина,
        # иначе отчёт молча зачтёт его в одну из крайностей
        tail += f", подтверждено частично: {partial_count}"
    return f"Закрыто {closed} из {total} допущений; {tail}."


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
            f"{_escape_cell(_render_facts_cell(a.facts, FACTS_SECTION))} |"
        )
    return "\n".join(lines)


def _render_facts_section(state: RunState) -> str:
    """Полные факты со ссылками, вынесенные из таблицы.

    Каждый факт остаётся проверяемым — таблица выше только отсылает сюда.
    """
    lines = [f"## {FACTS_SECTION}", ""]
    with_facts = [a for a in state.assumptions if a.facts]
    if not with_facts:
        lines.append("Фактов не собрано.")
        return "\n".join(lines)
    for a in with_facts:
        lines.append(f"**{a.id} — {a.status}:** {a.text}")
        lines.append("")
        for fact in a.facts:
            dated = f" ({fact.date})" if fact.date else ""
            lines.append(f"- {_render_fact(fact)}{dated}")
        lines.append("")
    return "\n".join(lines).rstrip()


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
    partial = _partial(state)
    untestable = [a for a in state.assumptions if a.status == "untestable"]
    if not unverified and not partial and not untestable:
        lines.append(
            "Все допущения проверены по источникам — экспериментов в реальном мире не требуется."
        )
        return "\n".join(lines)

    blocks = [
        (unverified, "Эти допущения остались непроверенными — прогон не закрыл "
                     "их фактами:"),
        (partial, "Эти допущения подтверждены лишь частично — данные нашлись по "
                  "другому сегменту, региону или периоду:"),
        (untestable, "Эти допущения можно проверить только реальным экспериментом:"),
    ]
    for items, title in blocks:
        if not items:
            continue
        if len(lines) > 2:
            lines.append("")
        lines.append(title)
        lines.append("")
        lines.extend(f"- [{a.id}] {a.text}" for a in items)
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
        _render_facts_section(state),
        _render_evolution(state),
        _render_next_steps(state),
    ]
    return "\n\n".join(sections) + "\n"
