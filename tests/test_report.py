"""Тесты сборщика отчёта: build_report собирает markdown детерминированно из RunState."""
import pytest

from kaidzen.report import build_report
from kaidzen.state import (ApiUsage, Assumption, ChangelogEntry, Fact,
                            JudgeResult, RunState, Version)


def _judge(scores: dict, delta: float = 0.0) -> JudgeResult:
    return JudgeResult(scores=scores, total=sum(scores.values()),
                        delta_vs_previous=delta, critique=[], verdict="continue")


def _full_state() -> RunState:
    """Полноценный завершённый прогон с двумя оценёнными версиями и допущениями."""
    assumptions = [
        Assumption(
            id="A1", text="рынок существует", criticality="high",
            status="confirmed",
            facts=[Fact(claim="рынок растёт на 10% в год",
                        source_url="https://example.com/report",
                        source_title="Market Report 2026")],
        ),
        Assumption(
            id="A2", text="конкурентов нет", criticality="medium",
            status="untestable", facts=[],
        ),
        Assumption(
            id="A3", text="цена приемлема | со скидкой", criticality="low",
            status="unverified",
            facts=[Fact(claim="цена конкурентна",
                        source_url="https://example.com/price")],
        ),
    ]
    versions = [
        Version(n=1, idea_text="идея v1",
                changelog=[ChangelogEntry(change="уточнили аудиторию",
                                           reason="анализ показал разброс")],
                judge=_judge({"clarity": 5.0, "novelty": 3.0})),
        Version(n=2, idea_text="идея v2",
                changelog=[ChangelogEntry(change="добавили механику",
                                           reason="фидбек судьи"),
                           ChangelogEntry(change="убрали лишнее",
                                          reason="упрощение"),
                           ChangelogEntry(change="третье изменение",
                                          reason="не должно попасть в отчёт")],
                judge=_judge({"clarity": 8.0, "novelty": 6.0}, delta=6.0)),
    ]
    return RunState(
        run_id="run-42", candidate_id="gen000-generic", config={},
        original_idea="сырая идея", assumptions=assumptions, versions=versions,
        iteration=2, stop_reason="plateau",
        api_usage=ApiUsage(input_tokens=1000, output_tokens=500, web_searches=7),
    )


def test_header_and_summary_present():
    state = _full_state()
    report = build_report(state, summary_text="Идея готова к пилоту.")
    assert "run-42" in report
    assert "gen000-generic" in report
    assert "plateau" in report
    assert "Идея готова к пилоту." in report


def test_final_idea_section_present():
    state = _full_state()
    report = build_report(state)
    assert "Финальная версия идеи" in report
    assert "идея v2" in report


def test_rubric_table_shows_first_and_final_scored_versions():
    state = _full_state()
    report = build_report(state)
    assert "Оценки по рубрике" in report
    assert "clarity" in report and "novelty" in report
    # первая версия: clarity=5.0, финальная: clarity=8.0
    assert "5.0" in report and "8.0" in report
    assert "total" in report.lower()


def test_assumptions_table_present():
    state = _full_state()
    report = build_report(state)
    assert "Допущения" in report
    assert "A1" in report and "рынок существует" in report
    assert "high" in report and "confirmed" in report


def test_fact_link_renders_with_title():
    state = _full_state()
    report = build_report(state)
    assert "[Market Report 2026](https://example.com/report): рынок растёт на 10% в год" in report


def test_fact_without_title_falls_back_to_url_as_link_text():
    state = _full_state()
    report = build_report(state)
    assert "[https://example.com/price](https://example.com/price): цена конкурентна" in report


def test_assumption_without_facts_renders_dash():
    state = _full_state()
    report = build_report(state)
    lines = [l for l in report.splitlines() if l.startswith("| A2")]
    assert len(lines) == 1
    assert "| - |" in lines[0] or lines[0].rstrip().endswith("| - |")


def test_untestable_assumptions_appear_in_next_steps():
    state = _full_state()
    report = build_report(state)
    next_steps = report.split("Next steps")[1]
    assert "конкурентов нет" in next_steps


def test_confirmed_assumptions_do_not_appear_in_next_steps():
    state = _full_state()
    report = build_report(state)
    next_steps = report.split("Next steps")[1]
    assert "рынок существует" not in next_steps


def test_all_checked_wording_when_nothing_untestable():
    state = _full_state()
    # заменяем untestable на confirmed
    checked = state.model_copy(update={
        "assumptions": [a.model_copy(update={"status": "confirmed"}) if a.id == "A2" else a
                        for a in state.assumptions]
    })
    report = build_report(checked)
    next_steps = report.split("Next steps")[1]
    assert "провер" in next_steps.lower()  # "все допущения проверены..."


def _all_unverified_state() -> RunState:
    """Прогон, в котором не закрыто ни одно допущение (баг с чужими id)."""
    state = _full_state()
    return state.model_copy(update={
        "assumptions": [a.model_copy(update={"status": "unverified", "facts": []})
                        for a in state.assumptions]
    })


def test_next_steps_does_not_claim_everything_verified_when_all_unverified():
    report = build_report(_all_unverified_state())
    next_steps = report.split("Next steps")[1]
    assert "Все допущения проверены" not in next_steps
    assert "непровер" in next_steps.lower()


def test_next_steps_lists_unverified_assumptions():
    report = build_report(_all_unverified_state())
    next_steps = report.split("Next steps")[1]
    for text in ("рынок существует", "конкурентов нет"):
        assert text in next_steps


def test_next_steps_lists_both_unverified_and_untestable():
    state = _full_state()   # A2 untestable, A3 unverified
    next_steps = build_report(state).split("Next steps")[1]
    assert "конкурентов нет" in next_steps          # untestable
    assert "цена приемлема" in next_steps           # unverified


def test_next_steps_says_all_verified_only_when_nothing_open():
    state = _full_state()
    closed = state.model_copy(update={
        "assumptions": [a.model_copy(update={"status": "confirmed"})
                        for a in state.assumptions]
    })
    next_steps = build_report(closed).split("Next steps")[1]
    assert "Все допущения проверены" in next_steps


def test_assumptions_section_makes_all_unverified_run_obvious():
    assumptions = build_report(_all_unverified_state()).split("## Допущения")[1]
    header = assumptions.split("| id |")[0]
    assert "Ни одно допущение не проверено" in header


def test_assumptions_section_shows_verified_counter():
    assumptions = build_report(_full_state()).split("## Допущения")[1]
    header = assumptions.split("| id |")[0]
    # закрыты A1 и A2, открыто A3
    assert "2" in header and "3" in header


def test_empty_assumption_registry_does_not_claim_success():
    state = RunState(run_id="r1", candidate_id="c", config={},
                     original_idea="идея")
    next_steps = build_report(state).split("Next steps")[1]
    assert "Все допущения проверены" not in next_steps


def test_rolled_back_version_marked_and_excluded_from_rubric():
    state = _full_state()
    rolled = state.model_copy(update={
        "versions": state.versions + [
            Version(n=3, idea_text="откаченная идея", rolled_back=True,
                    judge=_judge({"clarity": 1.0, "novelty": 1.0}))
        ]
    })
    report = build_report(rolled)
    evolution = report.split("Эволюция версий")[1].split("Next steps")[0]
    assert "откачен" in evolution.lower() or "rolled" in evolution.lower()
    # оценка откаченной версии (1.0/1.0) не должна попасть в таблицу рубрики
    rubric = report.split("Оценки по рубрике")[1].split("Допущения")[0]
    assert "1.0" not in rubric


def test_evolution_limits_to_first_two_changelog_entries():
    state = _full_state()
    report = build_report(state)
    evolution = report.split("Эволюция версий")[1].split("Next steps")[0]
    assert "добавили механику" in evolution
    assert "убрали лишнее" in evolution
    assert "третье изменение" not in evolution


def test_zero_scored_versions_does_not_crash():
    state = RunState(run_id="r1", candidate_id="c", config={},
                      original_idea="идея без версий")
    report = build_report(state)
    assert "Оценки по рубрике" in report
    assert "run-report-should-not-crash" not in report  # просто убеждаемся что дошли сюда


def test_pipe_character_in_cell_is_escaped_and_does_not_break_table():
    state = _full_state()
    report = build_report(state)
    # строка с "|" внутри допущения A3 не должна порождать лишний столбец
    for line in report.splitlines():
        if line.startswith("| A3"):
            assert "\\|" in line  # исходный "|" экранирован
            # разделителей столбцов (неэкранированных "|") ровно 6 для 5 полей
            unescaped_pipes = line.replace("\\|", "").count("|")
            assert unescaped_pipes == 6


def test_api_usage_numbers_present():
    state = _full_state()
    report = build_report(state)
    assert "1000" in report
    assert "500" in report
    assert "7" in report


def test_missing_summary_text_defaults_to_empty_and_no_crash():
    state = _full_state()
    report = build_report(state)  # без summary_text
    assert isinstance(report, str) and len(report) > 0


# --- дефекты, найденные на smoke-прогоне -----------------------------------

def _state_with_statuses(*statuses):
    """Реестр из допущений с заданными статусами, по одному на статус."""
    return RunState(
        run_id="r", candidate_id="c", config={}, original_idea="raw",
        assumptions=[
            Assumption(id=f"A{i}", text=f"допущение {i}", criticality="high",
                       status=status)
            for i, status in enumerate(statuses, start=1)
        ])


def test_partial_does_not_count_as_closed():
    """`partial` — не закрытое допущение: оркестратор его таковым не считает."""
    md = build_report(_state_with_statuses("confirmed", "partial", "unverified"),
                      summary_text="s")
    assert "Закрыто 1 из 3" in md


def test_partial_counted_among_open():
    md = build_report(_state_with_statuses("confirmed", "partial"), summary_text="s")
    assert "Закрыто 1 из 2" in md
    assert "подтверждено частично: 1" in md
    # частичное подтверждение не даёт права заявить, что всё проверено
    assert "Все допущения проверены" not in md
    assert "подтверждены лишь частично" in md


def test_single_scored_version_renders_one_score_column():
    """Одна оценённая версия — одна колонка, а не 'v1 | v1'."""
    s = RunState(
        run_id="r", candidate_id="c", config={}, original_idea="raw",
        versions=[Version(n=1, idea_text="v1", judge=JudgeResult(
            scores={"clarity": 6.0}, total=6.0, delta_vs_previous=0.0,
            critique=[], verdict="continue"))])
    md = build_report(s, summary_text="s")
    rubric = md.split("## Оценки по рубрике")[1].split("##")[0]
    assert "| Ось | v1 |" in rubric
    assert "v1 | v1" not in rubric


def test_many_facts_are_summarised_in_table_cell():
    """Ячейка с фактами не должна раздувать таблицу до нечитаемости."""
    facts = [Fact(claim=f"факт {i}", source_url=f"https://src/{i}",
                  source_title=f"Источник {i}") for i in range(6)]
    s = RunState(
        run_id="r", candidate_id="c", config={}, original_idea="raw",
        assumptions=[Assumption(id="A1", text="t", criticality="high",
                                status="refuted", facts=facts)])
    md = build_report(s, summary_text="s")
    table_row = [ln for ln in md.splitlines() if ln.startswith("| A1 ")][0]
    assert len(table_row) < 400, f"строка таблицы раздута: {len(table_row)} символов"
    assert "6" in table_row  # число фактов названо
    # полные факты со ссылками никуда не делись — они ниже таблицы
    assert "https://src/5" in md
